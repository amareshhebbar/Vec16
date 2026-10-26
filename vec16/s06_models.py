import subprocess
import sys
import traceback

from vec16 import charts, hub
from vec16.common import (
    REPO,
    hf_login,
    load_cfg,
    log,
    model_dir,
    model_status,
    read_json,
    repo_name,
    run_dir,
    scale_variants,
    write_json,
)
from vec16.summary import SPLITS, baseline_results, model_summary, select_headline


def sub(mod, *args):
    return subprocess.run([sys.executable, "-m", mod, *[str(x) for x in args]], cwd=str(REPO)).returncode


def train(cfg, m, v, s, teacher=None):
    rd = run_dir(cfg, m, v, s)
    if (rd / "done.json").exists():
        log(f"skip {m} {v} s{s}, already trained")
        return True
    model_status(cfg, m, state=f"training {v} seed {s}")
    extra = ["--teacher", teacher] if teacher else []
    rc = sub("vec16.train", m, v, s, *extra)
    batch = cfg["models"][m].get("batch_size", cfg["run"]["batch_size"])
    while rc == 3 and batch > 2:
        batch //= 2
        log(f"{m} {v} s{s} out of memory, retry with batch {batch}")
        (rd / "ckpt.pt").unlink(missing_ok=True)
        rc = sub("vec16.train", m, v, s, *extra, "--batch", batch)
    if rc != 0:
        log(f"FAILED {m} {v} s{s} code {rc}")
    return (rd / "done.json").exists()


def plan(cfg, m):
    run = cfg["run"]
    s0 = run["seeds"][0]
    if not (model_dir(cfg, m) / "mntp" / "done.json").exists():
        model_status(cfg, m, state="mntp")
        if sub("vec16.mntp", m) != 0:
            raise RuntimeError(f"mntp failed for {m}")
    for v in cfg["variants"]:
        train(cfg, m, v, s0)
    head = select_headline(cfg, m)
    if head is None:
        raise RuntimeError(f"no teacher variant finished for {m}")
    write_json(model_dir(cfg, m) / "headline.json", head)
    log(f"{m} headline {head['variant']} ({head['metric']} {head['dev_score']:.4f})")
    for s in run["seeds"][1:]:
        train(cfg, m, head["variant"], s, head["teacher"])
    if m in (run.get("ablation_models") or []):
        for a in cfg["ablations"]:
            train(cfg, m, a, s0, head["teacher"])
    default_n = cfg["models"][m].get("train_queries") or run.get("train_queries")
    for v in scale_variants(cfg, m):
        if default_n and int(v.split("_")[1]) == int(default_n):
            continue
        train(cfg, m, v, s0, head["teacher"])
    return head


def finished_runs(cfg, m):
    out = []
    for v in list(cfg["variants"]) + list(cfg["ablations"]) + scale_variants(cfg, m):
        for s in cfg["run"]["seeds"]:
            if (run_dir(cfg, m, v, s) / "done.json").exists():
                out.append(f"{v}:{s}")
    return out


def teacher_plot_args(s):
    ts = s.get("teacher_scores") or {}
    key = SPLITS[s["split"]]
    teacher = ("dense teacher", ts["teacher_dense"]) if "teacher_dense" in ts else None
    late = None
    if "teacher_colbert" in ts:
        c = ts["teacher_colbert"]
        late = ("ColBERT teacher", c.get("late_ood" if key == "mean_ood" else "late") or c.get("late"), c["bytes_per_doc"])
    return key, teacher, late


def draw(cfg, m, s):
    adir = cfg["paths"]["assets"] / m
    adir.mkdir(parents=True, exist_ok=True)
    tail = cfg["models"][m]["tail"].replace("_", " ")
    present = []
    bres = baseline_results(cfg)
    bl = [(r["label"], r, r.get("mrl")) for r in bres.values()]
    key, teacher, late = teacher_plot_args(s)
    where = "out of domain sets" if s["split"] == "ood" else "all sets"
    charts.curve(adir / "curve.png", [(f"Vec16 {tail}", s["main"])], bl, s["bits"], f"Quality against bytes per document, NanoBEIR {where}", key, teacher, late)
    present.append("curve.png")
    charts.mrl(adir / "mrl.png", f"Vec16 {tail}", s["main"], s["base"], key)
    present.append("mrl.png")
    if s["teachers"]:
        names = list(s["teachers"])
        charts.bars(adir / "teachers.png", "Which teacher works best", names, [("float32 full", [s["teachers"][n]["float"] or 0 for n in names]), (f"binary {s['bits']} bits", [s["teachers"][n]["binary"] or 0 for n in names])])
        present.append("teachers.png")
    if s["ablations"]:
        names = list(s["ablations"])
        charts.bars(adir / "ablation.png", "What each part adds", names, [("float32 full", [s["ablations"][n]["float"] or 0 for n in names]), (f"binary {s['bits']} bits", [s["ablations"][n]["binary"] or 0 for n in names])])
        present.append("ablation.png")
    if len(s.get("scaling") or []) >= 2:
        pts = s["scaling"]
        xs = [p["queries"] for p in pts]
        charts.scaling(adir / "data_scaling.png", f"Vec16 {tail}: more training queries", "training queries (log scale)", [("float32 full", xs, [p["float"] or 0 for p in pts]), (f"binary {s['bits']} bits", xs, [p["binary"] or 0 for p in pts])])
        present.append("data_scaling.png")
    log_p = run_dir(cfg, m, s["headline"]["variant"], s["best_seed"]) / "train_log.json"
    if log_p.exists():
        charts.loss_curve(adir / "loss.png", f"Vec16 {tail}", read_json(log_p))
        present.append("loss.png")
    return present


def export(cfg, m, s):
    if not cfg["run"].get("onnx"):
        return None
    out = cfg["paths"]["runs"] / m / "onnx"
    if (out / "export.json").exists():
        return read_json(out / "export.json")
    src = run_dir(cfg, m, s["headline"]["variant"], s["best_seed"]) / "model"
    size = sum(f.stat().st_size for f in src.glob("*.safetensors"))
    if size / 4 > float(cfg["onnx_max_params"]):
        log(f"skip onnx for {m}, {size / 4 / 1e9:.2f}B params over the limit")
        return None
    model_status(cfg, m, state="onnx export")
    if sub("vec16.export_onnx", src, out) != 0:
        log(f"onnx export failed for {m}")
        return None
    return read_json(out / "export.json")


def banner(cfg, m, s, private):
    t = s["tuned"]
    word = "GOOD" if s["good"] else "BAD"
    log("=" * 70)
    log(f"{word}  {m}  ({s['id']})")
    log(f"  starting LLM float32 nDCG@10   {(s['base_float'] or 0) * 100:.1f}")
    log(f"  Vec16 float32 {s['full_dim']:<5} nDCG@10   {t['float']['mean'] * 100:.1f}")
    log(f"  Vec16 binary {s['bits']} bits nDCG@10   {t['binary']['mean'] * 100:.1f}  ({s['bits'] // 8} bytes per document)")
    if t.get("rescore"):
        log(f"  binary, top {cfg['run'].get('rescore_top', 100)} rescored    {t['rescore']['mean'] * 100:.1f}")
    log(f"  scored on {'out of domain NanoBEIR sets' if s['split'] == 'ood' else 'all NanoBEIR sets'}")
    if s.get("reference_float") is not None:
        log(f"  reference {s['reference']} float32   {s['reference_float'] * 100:.1f}")
    log(f"  teacher used: {s['headline']['teacher']}   seeds: {t['float']['n']}   gpu hours: {s['gpu_hours']}")
    log(f"  {'; '.join(s['reasons'])}")
    log(f"  repo {cfg['project']['hf_user']}/{s['repo']} ({'private' if private else 'public'})")
    log("=" * 70)


def process(cfg, m, a):
    final = cfg["paths"]["results"] / m / "final.json"
    if final.exists():
        log(f"{m} already finished: {read_json(final)['verdict']}")
        return
    model_status(cfg, m, state="starting", repo=repo_name(cfg, m))
    plan(cfg, m)
    up = cfg["paths"]["results"] / m / "uploaded.json"
    up.parent.mkdir(parents=True, exist_ok=True)
    if a is not None and not up.exists():
        model_status(cfg, m, state="uploading trained weights (private)")
        from vec16.publish import upload_trained

        rid = upload_trained(cfg, m, a)
        write_json(up, {"repo": rid})
    model_status(cfg, m, state="testing")
    sub("vec16.evaluate", m, *finished_runs(cfg, m))
    s = model_summary(cfg, m)
    if s is None:
        raise RuntimeError(f"no evaluation results for {m}")
    present = draw(cfg, m, s)
    onnx = export(cfg, m, s)
    private = True
    if a is not None:
        from vec16.publish import upload_results

        _, private = upload_results(cfg, m, s, present, a)
    else:
        from vec16.publish import stage_main_extras

        stage_main_extras(cfg, m, s, present)
    banner(cfg, m, s, private)
    t = s["tuned"]
    rec = {
        "verdict": "GOOD" if s["good"] else "BAD",
        "reasons": s["reasons"],
        "repo": f"{cfg['project']['hf_user']}/{s['repo']}",
        "private": private,
        "teacher": s["headline"]["teacher"],
        "float": t["float"]["mean"],
        "binary": t["binary"]["mean"],
        "asym": t["asym"]["mean"] if t["asym"] else None,
        "rescore": t["rescore"]["mean"] if t.get("rescore") else None,
        "split": s["split"],
        "base_float": s["base_float"],
        "gpu_hours": s["gpu_hours"],
        "onnx": bool(onnx),
        "present": present,
    }
    write_json(final, rec)
    model_status(cfg, m, state="done", **rec)


def main():
    cfg = load_cfg()
    hf_login()
    a = hub.api()
    if a is None:
        log("publishing off, results stay local")
    fails = []
    models = cfg["run"]["models"]
    for i, m in enumerate(models, 1):
        log(f"[{i}/{len(models)}] model {m}")
        try:
            process(cfg, m, a)
        except Exception as ex:
            traceback.print_exc()
            fails.append({"model": m, "error": repr(ex)})
            model_status(cfg, m, state="failed", error=repr(ex))
    write_json(cfg["paths"]["results"] / "model_failures.json", fails)
    log(f"models finished, {len(fails)} failed")


if __name__ == "__main__":
    main()
