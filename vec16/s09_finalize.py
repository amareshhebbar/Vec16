import shutil

from vec16 import charts, hub
from vec16.common import hf_login, load_cfg, log, read_json, repo_name, update_status, write_json
from vec16.modelcard import SPLIT_KEY, SPLIT_NAME, bget, demo_section, pct
from vec16.s06_models import teacher_plot_args
from vec16.summary import baseline_results, model_summary


def leaderboard(cfg, sums, bres):
    bits = cfg["verdict_bits"]
    user = cfg["project"]["hf_user"]
    vs = sums[0]["split"]
    rows = [
        f"NanoBEIR nDCG@10, {SPLIT_NAME[vs]}.",
        "",
        f"| Model | Params | Dims | float32 | binary {bits} bits ({bits // 8} bytes) | binary, top {cfg['run'].get('rescore_top', 100)} rescored | Starting LLM | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(sums, key=lambda x: -(x["tuned"]["binary"]["mean"] if x["tuned"]["binary"] else 0)):
        t = s["tuned"]
        name = f"[Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}](https://huggingface.co/{user}/{s['repo']})"
        rows.append(f"| {name} | {s['params'] / 1e6:.0f}M | {s['full_dim']} | {pct(t['float']['mean'])} | **{pct(t['binary']['mean'])}** | {pct(t['rescore']['mean'] if t['rescore'] else None)} | {pct(s['base_float'])} | {'GOOD' if s['good'] else 'BAD'} |")
    for r in bres.values():
        d = r["dim"]
        b = bget(r, "binary", bits, vs)
        rows.append(f"| {r['label']} | | {d} | {pct(bget(r, 'float', d, vs))} | {pct(b) if b is not None else 'not Matryoshka'} | | | baseline |")
    return rows


def bench_lines(cfg):
    p = cfg["paths"]["results"] / "bench.json"
    if not p.exists():
        return []
    b = read_json(p)
    out = ["## Index size and speed", "", "![bench](assets/bench.png)", "", "| Index | Documents | Size | Queries per second |", "|---|---|---|---|"]
    for r in b["rows"]:
        size = r["bytes"] / 1024**3
        qps = f"{r['qps']:,.0f}" if r.get("qps") else "size only"
        out.append(f"| {r['label']} | {r['n']:,} | {size:,.2f} GB | {qps} |")
    out += ["", b["note"], ""]
    return out


def failures(cfg):
    out = []
    res = cfg["paths"]["results"]
    for f in ("model_failures.json", "baseline_failures.json"):
        p = res / f
        if p.exists():
            for r in read_json(p):
                out.append(f"`{f.replace('_failures.json', '')}` {r}")
    for p in cfg["paths"]["runs"].glob("*/*/oom.json"):
        out.append(f"out of memory once in `{p.parent.parent.name}/{p.parent.name}`, retried with a smaller batch")
    return out


def report(cfg, sums, bres, lb):
    fn = cfg["paths"]["results"] / "false_negatives.json"
    lines = ["# Vec16 run report", "", f"Profile `{cfg['profile']}` on `{cfg['platform']}`.", "", *lb, "", "![overall](assets/overall_curve.png)", ""]
    if len(sums) > 1:
        lines += ["![model size](assets/model_scaling.png)", ""]
    lines += bench_lines(cfg)
    lines += demo_section(cfg)
    if fn.exists():
        f = read_json(fn)
        lines += [f"Both teachers preferred an unlabeled candidate over the labeled positive for {f['rate'] * 100:.1f}% of training queries.", ""]
    lines += ["## Per model", ""]
    for s in sums:
        lines.append(f"* **{s['model']}** {'GOOD' if s['good'] else 'BAD'}: {'; '.join(s['reasons'])}. Teacher {s['headline']['teacher']}, {s['gpu_hours']} GPU hours.")
        for f in s.get("findings", []):
            lines.append(f"  * {f}")
    fails = failures(cfg)
    if fails:
        lines += ["", "## Failures", ""] + [f"* {f}" for f in fails]
    lines.append("")
    return "\n".join(lines)


def main():
    cfg = load_cfg()
    hf_login()
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    sums = []
    for m in cfg["run"]["models"]:
        if (res / m / "final.json").exists():
            s = model_summary(cfg, m)
            if s:
                sums.append(s)
    if not sums:
        log("no finished models")
        return
    bres = baseline_results(cfg)
    key, teacher, late = teacher_plot_args(sums[0])
    vs = sums[0]["split"]
    charts.curve(
        assets / "overall_curve.png",
        [(f"Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}", s["main"]) for s in sums],
        [(r["label"], r, r.get("mrl")) for r in bres.values()],
        cfg["verdict_bits"],
        f"Every Vec16 model against public embedders and teachers, {SPLIT_NAME[vs]}",
        SPLIT_KEY[vs],
        teacher,
        late,
    )
    names = [cfg["models"][s["model"]]["tail"].replace("_", " ") for s in sums]
    charts.bars(
        assets / "overall_bars.png",
        f"Vec16 models, NanoBEIR nDCG@10, {SPLIT_NAME[vs]}",
        names,
        [
            ("starting LLM", [s["base_float"] or 0 for s in sums]),
            ("float32 full", [s["tuned"]["float"]["mean"] for s in sums]),
            (f"binary {cfg['verdict_bits']} bits", [s["tuned"]["binary"]["mean"] for s in sums]),
        ],
    )
    if len(sums) > 1:
        ordered = sorted(sums, key=lambda x: x["params"])
        xs = [s["params"] for s in ordered]
        charts.scaling(
            assets / "model_scaling.png",
            "Bigger student, same data and teachers",
            "student parameters (log scale)",
            [("float32 full", xs, [s["tuned"]["float"]["mean"] for s in ordered]), (f"binary {cfg['verdict_bits']} bits", xs, [s["tuned"]["binary"]["mean"] for s in ordered])],
        )
    lb = leaderboard(cfg, sums, bres)
    (cfg["paths"]["work"] / "REPORT.md").write_text(report(cfg, sums, bres, lb))
    table = {
        s["model"]: {
            "repo": s["repo"],
            "verdict": "GOOD" if s["good"] else "BAD",
            "split": s["split"],
            "params": s["params"],
            "by_split": s["by_split"],
            "base_by_split": s["base_by_split"],
            "teacher": s["headline"]["teacher"],
            "gpu_hours": s["gpu_hours"],
            "findings": s["findings"],
        }
        for s in sums
    }
    write_json(res / "summary.json", table)
    a = hub.api()
    if a is not None:
        from vec16.publish import stage_main_extras

        user = cfg["project"]["hf_user"]
        for s in sums:
            m = s["model"]
            present = read_json(res / m / "final.json").get("present", [])
            root = stage_main_extras(cfg, m, s, present, lb)
            for f in ("overall_curve.png", "model_scaling.png", "bench.png"):
                if (assets / f).exists():
                    shutil.copy(assets / f, root / "assets" / f)
            rid = f"{user}/{repo_name(cfg, m)}"
            try:
                hub.retry(a.upload_folder, folder_path=str(root), repo_id=rid, revision="main", commit_message="Refresh card with every Vec16 model", allow_patterns=["README.md", "assets/*"])
                log(f"refreshed {rid}")
            except Exception as ex:
                log(f"refresh failed for {rid}: {ex}")
    update_status(cfg, "finalize", {"models": len(sums), "good": sum(1 for s in sums if s["good"])})
    print()
    print("\n".join(lb))
    print()
    log(f"report at {cfg['paths']['work'] / 'REPORT.md'}")


if __name__ == "__main__":
    main()
