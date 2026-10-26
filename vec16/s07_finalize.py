from vec16 import charts, hub
from vec16.common import hf_login, load_cfg, log, read_json, repo_name, update_status, write_json
from vec16.modelcard import pct
from vec16.summary import baseline_results, model_summary


def leaderboard(cfg, sums, bres):
    bits = cfg["verdict_bits"]
    user = cfg["project"]["hf_user"]
    rows = [
        f"| Model | Dims | float32 nDCG@10 | binary {bits} bits ({bits // 8} bytes) | Starting LLM | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(sums, key=lambda x: -(x["tuned"]["binary"]["mean"] if x["tuned"]["binary"] else 0)):
        t = s["tuned"]
        name = f"[Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}](https://huggingface.co/{user}/{s['repo']})"
        rows.append(f"| {name} | {s['full_dim']} | {pct(t['float']['mean'])} | **{pct(t['binary']['mean'])}** | {pct(s['base_float'])} | {'GOOD' if s['good'] else 'BAD'} |")
    for r in bres.values():
        d = r["dim"]
        rows.append(f"| {r['label']} | {d} | {pct(r['mean']['float'].get(str(d)))} | {pct(r['mean']['binary'].get(str(min(bits, d)))) if str(min(bits, d)) in r['mean']['binary'] else 'n/a'} | | baseline |")
    return rows


def report(cfg, sums, bres, lb):
    lines = [
        "# Vec16 run report",
        "",
        f"Profile `{cfg['profile']}` on `{cfg['platform']}`.",
        "",
        *lb,
        "",
        "![overall](assets/overall_curve.png)",
        "",
        "## Per model",
        "",
    ]
    for s in sums:
        lines.append(f"* **{s['model']}** {'GOOD' if s['good'] else 'BAD'}: {'; '.join(s['reasons'])}. Teacher {s['headline']['teacher']}, {s['gpu_hours']} GPU hours.")
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
    charts.curve(
        assets / "overall_curve.png",
        [(f"Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}", s["main"]) for s in sums],
        [(r["label"], r, r.get("mrl")) for r in bres.values()],
        cfg["verdict_bits"],
        "Every Vec16 model against public embedders",
    )
    names = [cfg["models"][s["model"]]["tail"].replace("_", " ") for s in sums]
    charts.bars(
        assets / "overall_bars.png",
        "Vec16 models, NanoBEIR nDCG@10",
        names,
        [
            ("starting LLM", [s["base_float"] or 0 for s in sums]),
            ("float32 full", [s["tuned"]["float"]["mean"] for s in sums]),
            (f"binary {cfg['verdict_bits']} bits", [s["tuned"]["binary"]["mean"] for s in sums]),
        ],
    )
    lb = leaderboard(cfg, sums, bres)
    (cfg["paths"]["work"] / "REPORT.md").write_text(report(cfg, sums, bres, lb))
    table = {
        s["model"]: {
            "repo": s["repo"],
            "verdict": "GOOD" if s["good"] else "BAD",
            "float": s["tuned"]["float"],
            "binary": s["tuned"]["binary"],
            "asym": s["tuned"]["asym"],
            "base_float": s["base_float"],
            "teacher": s["headline"]["teacher"],
            "gpu_hours": s["gpu_hours"],
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
            import shutil

            shutil.copy(assets / "overall_curve.png", root / "assets" / "overall_curve.png")
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
