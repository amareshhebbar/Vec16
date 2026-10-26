import statistics
from pathlib import Path

from vec16.common import headline_bits, model_dir, read_json, repo_name, run_dir


def agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def pick(res, kind, dim):
    if res is None:
        return None
    return res["mean"][kind].get(str(dim))


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
    tuned = {
        "float": agg([pick(r, "float", full) for r in head_res.values()]),
        "float128": agg([pick(r, "float", 128) for r in head_res.values()]),
        "binary": agg([pick(r, "binary", bits) for r in head_res.values()]),
        "asym": agg([pick(r, "asym", bits) for r in head_res.values()]),
    }
    bs = best_seed(cfg, mkey, hv)
    main = head_res.get(bs) or next(iter(head_res.values()))
    teachers = {}
    for v in cfg["variants"]:
        r = load_result(cfg, mkey, f"{v}__s{seeds[0]}")
        if r:
            d = read_json(run_dir(cfg, mkey, v, seeds[0]) / "done.json")
            teachers[v] = {"float": pick(r, "float", full), "binary": pick(r, "binary", bits), "asym": pick(r, "asym", bits), "dev": d.get("dev_mixed")}
    ablations = {}
    if mkey in (run.get("ablation_models") or []):
        ref = load_result(cfg, mkey, f"{hv}__s{seeds[0]}")
        if ref:
            ablations["full method"] = {"float": pick(ref, "float", full), "binary": pick(ref, "binary", bits)}
        for a in cfg["ablations"]:
            r = load_result(cfg, mkey, f"{a}__s{seeds[0]}")
            if r:
                ablations[a] = {"float": pick(r, "float", full), "binary": pick(r, "binary", bits)}
    bres = baseline_results(cfg)
    ref_key = run.get("reference")
    ref_float = None
    if ref_key and ref_key in bres:
        ref_float = pick(bres[ref_key], "float", bres[ref_key]["dim"])
    base_float = pick(base, "float", base["full_dim"]) if base else None
    good, reasons = verdict(cfg, tuned["float"]["mean"] if tuned["float"] else None, tuned["binary"]["mean"] if tuned["binary"] else None, base_float, ref_float)
    seconds = 0.0
    mntp = model_dir(cfg, mkey) / "mntp" / "done.json"
    if mntp.exists():
        seconds += read_json(mntp)["seconds"]
    for p in (model_dir(cfg, mkey)).glob("*__s*/done.json"):
        seconds += read_json(p)["seconds"]
    return {
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
        "runs": sorted(p.parent.name for p in Path(model_dir(cfg, mkey)).glob("*__s*/done.json")),
    }
