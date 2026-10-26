import shutil

from vec16 import hub
from vec16.common import log, read_json, run_dir
from vec16.modelcard import branch_card, model_card
from vec16.summary import baseline_results, best_seed, headline, load_result


def branches(cfg, mkey):
    head = headline(cfg, mkey)
    hv = head["variant"]
    seeds = cfg["run"]["seeds"]
    main_seed = best_seed(cfg, mkey, hv)
    out = []
    for v in list(cfg["variants"]) + list(cfg["ablations"]):
        for s in seeds:
            rd = run_dir(cfg, mkey, v, s)
            if not (rd / "done.json").exists():
                continue
            if v == hv and s == main_seed:
                out.append(("main", v, s))
            elif s == seeds[0]:
                out.append((v, v, s))
            else:
                out.append((f"{v}_s{s}", v, s))
    return out


def upload_text(a, rid, branch, name, text, message):
    hub.retry(a.upload_file, path_or_fileobj=text.encode(), path_in_repo=name, repo_id=rid, revision=branch, commit_message=message)


def upload_trained(cfg, mkey, a):
    import json

    from vec16.common import repo_name

    user = cfg["project"]["hf_user"]
    repo = repo_name(cfg, mkey)
    rid = f"{user}/{repo}"
    for branch, v, s in branches(cfg, mkey):
        rd = run_dir(cfg, mkey, v, s)
        done = read_json(rd / "done.json")
        hub.push_folder(a, rid, rd / "model", private=True, branch=branch, message=f"Upload {v} seed {s}")
        upload_text(a, rid, branch, "training_run.json", json.dumps(done, indent=2), "Training run metadata")
        if branch == "main":
            card = f"# {repo}\n\nTraining finished, evaluation running. Results appear here when it ends.\n"
        else:
            card = branch_card(cfg, mkey, branch, s, repo, None, done["spec"])
        upload_text(a, rid, branch, "README.md", card, "Model card")
        log(f"uploaded {rid} branch {branch}")
    return rid


def stage_main_extras(cfg, mkey, s, present, leaderboard=None):
    from vec16.common import repo_name

    repo = repo_name(cfg, mkey)
    root = cfg["paths"]["publish"] / repo / "main"
    shutil.rmtree(root, ignore_errors=True)
    assets = cfg["paths"]["assets"] / mkey
    (root / "assets").mkdir(parents=True, exist_ok=True)
    for f in present:
        shutil.copy(assets / f, root / "assets" / f)
    rdir = root / "results"
    rdir.mkdir(exist_ok=True)
    for f in sorted((cfg["paths"]["results"] / mkey).glob("*.json")):
        shutil.copy(f, rdir / f.name)
    head = headline(cfg, mkey)
    tb = run_dir(cfg, mkey, head["variant"], s["best_seed"]) / "tb"
    if tb.exists():
        shutil.copytree(tb, root / "runs" / f"{head['variant']}__s{s['best_seed']}", dirs_exist_ok=True)
    onnx = cfg["paths"]["runs"] / mkey / "onnx"
    if (onnx / "model.onnx").exists():
        shutil.copytree(onnx, root / "onnx", dirs_exist_ok=True)
    (root / "README.md").write_text(model_card(cfg, mkey, s, baseline_results(cfg), present, leaderboard))
    return root


def upload_results(cfg, mkey, s, present, a, leaderboard=None):
    from vec16.common import repo_name

    user = cfg["project"]["hf_user"]
    repo = repo_name(cfg, mkey)
    rid = f"{user}/{repo}"
    root = stage_main_extras(cfg, mkey, s, present, leaderboard)
    hub.retry(a.upload_folder, folder_path=str(root), repo_id=rid, revision="main", commit_message="Results, charts and model card", allow_patterns=["README.md", "assets/*", "results/*", "runs/**", "onnx/*"])
    for branch, v, sd in branches(cfg, mkey):
        if branch == "main":
            continue
        res = load_result(cfg, mkey, f"{v}__s{sd}")
        spec = read_json(run_dir(cfg, mkey, v, sd) / "done.json")["spec"]
        upload_text(a, rid, branch, "README.md", branch_card(cfg, mkey, branch, sd, repo, res, spec), "Branch card with results")
    run = cfg["run"]
    private = bool(run["private"]) or (not s["good"] and bool(run["verdict"]["private_if_bad"]))
    hub.set_private(a, rid, private)
    if s["good"] or run.get("collect_all"):
        t = s["tuned"]
        note = (
            f"{'GOOD' if s['good'] else 'BAD'}. From {s['id']}. {s['bits']} bits ({s['bits'] // 8} bytes) nDCG@10 "
            f"{t['binary']['mean'] * 100:.1f}, float32 {t['float']['mean'] * 100:.1f}, starting LLM {(s['base_float'] or 0) * 100:.1f}"
        )
        hub.add_to_collection(cfg, a, rid, "model", note)
    return rid, private
