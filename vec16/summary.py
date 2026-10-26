import statistics
from pathlib import Path

from vec16.common import headline_bits, model_dir, read_json, repo_name, run_dir


def agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


SPLITS = {"all": "mean", "ood": "mean_ood", "ind": "mean_ind"}


def pick(res, kind, dim, split="all"):
    if res is None:
        return None
    block = res.get(SPLITS[split]) or ({} if split != "all" else res.get("mean")) or {}
    return block.get(kind, {}).get(str(dim))


def vsplit(cfg):
    return cfg.get("verdict_split") or "all"


def scaling(cfg, mkey, head, bits):
    from vec16.common import scale_variants

    pts = []
    s0 = cfg["run"]["seeds"][0]
    names = scale_variants(cfg, mkey) + [head["variant"]]
    for v in names:
        r = load_result(cfg, mkey, f"{v}__s{s0}")
        if not r:
            continue
        n = r.get("meta", {}).get("train_queries")
        full = r["full_dim"]
        pts.append({"variant": v, "queries": n, "float": pick(r, "float", full, "ood") or pick(r, "float", full), "binary": pick(r, "binary", bits, "ood") or pick(r, "binary", bits)})
    pts = [p for p in pts if p["queries"]]
    return sorted({p["queries"]: p for p in pts}.values(), key=lambda p: p["queries"])


def teacher_results(cfg):
    out = {}
    for k in ("teacher_dense", "teacher_colbert"):
        p = cfg["paths"]["results"] / f"{k}.json"
        if p.exists():
            out[k] = read_json(p)
    return out


def binarized_refs(cfg, bres, bits, split):
    rows = []
    for k, r in bres.items():
        if not r.get("mrl"):
            continue
        b = pick(r, "binary", bits, split)
        if b is not None:
            rows.append({"key": k, "label": r["label"], "binary": b, "float": pick(r, "float", r["dim"], split), "dim": r["dim"]})
    return sorted(rows, key=lambda x: -x["binary"])


def findings(cfg, s):
    out = []
    ab = s["ablations"]
    full = ab.get("full method")
    names = {"no_mntp": "masked next token training", "no_qat": "the binary aware loss", "no_listwise": "teacher distillation", "causal": "bidirectional attention"}
    if full:
        for k, r in ab.items():
            if k == "full method" or r["float"] is None or full["float"] is None:
                continue
            for kind, label in (("float", "float32"), ("binary", f"binary {s['bits']} bits")):
                if r[kind] is not None and full[kind] is not None and r[kind] >= full[kind]:
                    out.append(f"Removing {names.get(k, k)} did not hurt {label}: {r[kind] * 100:.1f} without it against {full[kind] * 100:.1f} with it.")
    t = s["teachers"]
    if len(t) > 1:
        best = max(t, key=lambda v: t[v]["binary"] or 0)
        for v, r in t.items():
            if v != best and r["binary"] is not None and t[best]["binary"] is not None:
                out.append(f"The `{v}` teacher setup reached {r['binary'] * 100:.1f} at {s['bits']} bits against {t[best]['binary'] * 100:.1f} for `{best}`.")
    main = s["main"]
    b64, bb = pick(main, "binary", 64), pick(main, "binary", s["bits"])
    if b64 is not None and bb is not None and bb > 0:
        out.append(f"Going from {s['bits']} to 64 bits keeps {b64 / bb * 100:.0f}% of the {s['bits']} bit score, so 8 bytes per document is not free.")
    if not s["good"]:
        out += [f"Verdict failed: {r}." for r in s["reasons"]]
    return out



def headline(cfg, mkey):
    p = model_dir(cfg, mkey) / "headline.json"
    return read_json(p) if p.exists() else None


def select_headline(cfg, mkey):
    seed = cfg["run"]["seeds"][0]
    metric = "dev_" + cfg["select_metric"]
    best = None
    for v in cfg["variants"]:
        p = run_dir(cfg, mkey, v, seed) / "done.json"
        if not p.exists():
            continue
        d = read_json(p)
        score = d.get(metric, d.get("dev_float"))
        if best is None or score > best[1]:
            best = (v, score, d["spec"]["teacher"])
    if best is None:
        return None
    return {"variant": best[0], "teacher": best[2], "dev_score": best[1], "metric": metric}


def best_seed(cfg, mkey, vname):
    best = None
    for s in cfg["run"]["seeds"]:
        p = run_dir(cfg, mkey, vname, s) / "done.json"
        if p.exists():
            d = read_json(p)
            sc = d.get("dev_" + cfg["select_metric"], d.get("dev_float", 0.0))
            if best is None or sc > best[1]:
                best = (s, sc)
    return best[0] if best else None


def load_result(cfg, mkey, name):
    p = cfg["paths"]["results"] / mkey / f"{name}.json"
    return read_json(p) if p.exists() else None


def baseline_results(cfg):
    out = {}
    for b in cfg["run"]["baselines"]:
        p = cfg["paths"]["results"] / f"baseline_{b}.json"
        if p.exists():
            out[b] = read_json(p)
    return out


def verdict(cfg, tuned_float, tuned_bits, base_float, ref_float):
    v = cfg["run"]["verdict"]
    reasons = []
    ok = True
    if tuned_float is None or tuned_bits is None:
        return False, ["missing results"]
    gain = tuned_float - (base_float or 0.0)
    ratio = tuned_bits / tuned_float if tuned_float else 0.0
    if gain < v["min_gain"]:
        ok = False
        reasons.append(f"float gain over starting model {gain * 100:.1f} below {v['min_gain'] * 100:.1f}")
    if ratio < v["min_binary_ratio"]:
        ok = False
        reasons.append(f"binary keeps {ratio * 100:.1f}% of float, needs {v['min_binary_ratio'] * 100:.0f}%")
    if v.get("reference_ratio") and ref_float is not None and tuned_float < ref_float * v["reference_ratio"]:
        ok = False
        reasons.append(f"float {tuned_float * 100:.1f} below reference {ref_float * 100:.1f}")
    if ok:
        reasons.append(f"gain {gain * 100:.1f}, binary keeps {ratio * 100:.1f}% of float")
    return ok, reasons


def model_summary(cfg, mkey):
    run = cfg["run"]
    bits = headline_bits(cfg)
    head = headline(cfg, mkey)
    base = load_result(cfg, mkey, "base")
    if head is None:
        return None
    hv = head["variant"]
    seeds = run["seeds"]
    head_res = {s: load_result(cfg, mkey, f"{hv}__s{s}") for s in seeds}
    head_res = {s: r for s, r in head_res.items() if r}
    if not head_res:
        return None
    full = next(iter(head_res.values()))["full_dim"]

    def tuned_for(split):
        rs = list(head_res.values())
        return {
            "float": agg([pick(r, "float", full, split) for r in rs]),
            "float128": agg([pick(r, "float", 128, split) for r in rs]),
            "binary": agg([pick(r, "binary", bits, split) for r in rs]),
            "asym": agg([pick(r, "asym", bits, split) for r in rs]),
            "rescore": agg([pick(r, "rescore", bits, split) for r in rs]),
            "rescore_full": agg([pick(r, "rescore_full", bits, split) for r in rs]),
        }

    by_split = {sp: tuned_for(sp) for sp in SPLITS}
    vs = vsplit(cfg)
    if not by_split[vs]["float"]:
        vs = "all"
    tuned = by_split[vs]
    bs = best_seed(cfg, mkey, hv)
    main = head_res.get(bs) or next(iter(head_res.values()))
    teachers = {}
    for v in cfg["variants"]:
        r = load_result(cfg, mkey, f"{v}__s{seeds[0]}")
        if r:
            d = read_json(run_dir(cfg, mkey, v, seeds[0]) / "done.json")
            teachers[v] = {"float": pick(r, "float", full, vs), "binary": pick(r, "binary", bits, vs), "asym": pick(r, "asym", bits, vs), "dev": d.get("dev_mixed")}
    ablations = {}
    if mkey in (run.get("ablation_models") or []):
        ref = load_result(cfg, mkey, f"{hv}__s{seeds[0]}")
        if ref:
            ablations["full method"] = {"float": pick(ref, "float", full, vs), "binary": pick(ref, "binary", bits, vs)}
        for a in cfg["ablations"]:
            r = load_result(cfg, mkey, f"{a}__s{seeds[0]}")
            if r:
                ablations[a] = {"float": pick(r, "float", full, vs), "binary": pick(r, "binary", bits, vs)}
    bres = baseline_results(cfg)
    ref_key = run.get("reference")
    ref_float = None
    if ref_key and ref_key in bres:
        ref_float = pick(bres[ref_key], "float", bres[ref_key]["dim"], vs)
    base_float = pick(base, "float", base["full_dim"], vs) if base else None
    good, reasons = verdict(cfg, tuned["float"]["mean"] if tuned["float"] else None, tuned["binary"]["mean"] if tuned["binary"] else None, base_float, ref_float)
    reasons = [f"{r} ({'out of domain sets' if vs == 'ood' else 'all sets'})" for r in reasons]
    seconds = 0.0
    mntp = model_dir(cfg, mkey) / "mntp" / "done.json"
    if mntp.exists():
        seconds += read_json(mntp)["seconds"]
    for p in (model_dir(cfg, mkey)).glob("*__s*/done.json"):
        seconds += read_json(p)["seconds"]
    out = {
        "model": mkey,
        "id": cfg["models"][mkey]["id"],
        "repo": repo_name(cfg, mkey),
        "full_dim": full,
        "bits": bits,
        "headline": head,
        "best_seed": bs,
        "base": base,
        "base_float": base_float,
        "base_binary": pick(base, "binary", bits) if base else None,
        "main": main,
        "tuned": tuned,
        "teachers": teachers,
        "ablations": ablations,
        "reference": ref_key,
        "reference_float": ref_float,
        "good": good,
        "reasons": reasons,
        "gpu_hours": round(seconds / 3600, 2),
        "runs": sorted(p.parent.name for p in Path(model_dir(cfg, mkey)).glob("*__s*/done.json") if not p.parent.name.startswith("scale_")),
        "split": vs,
        "by_split": by_split,
        "base_by_split": {sp: (pick(base, "float", base["full_dim"], sp) if base else None) for sp in SPLITS},
        "scaling": scaling(cfg, mkey, head, bits),
        "teacher_scores": teacher_results(cfg),
        "binarized": binarized_refs(cfg, bres, bits, vs),
    }
    out["findings"] = findings(cfg, out)
    params = sum(f.stat().st_size for f in (run_dir(cfg, mkey, hv, bs) / "model").glob("*.safetensors")) / 4
    out["params"] = params
    return out
