import argparse
import traceback

from vec16.common import (
    bin_dims,
    eval_dtype,
    hf_login,
    load_cfg,
    log,
    mrl_dims,
    read_json,
    release,
    run_dir,
    write_json,
)
from vec16.retrieval import evaluate_nano
from vec16.students import build_student, load_trained


def results_dir(cfg, mkey):
    d = cfg["paths"]["results"] / mkey
    d.mkdir(parents=True, exist_ok=True)
    return d


def score(cfg, model):
    full = model.get_sentence_embedding_dimension()
    return evaluate_nano(cfg, model, cfg["prompts"]["query"], cfg["prompts"]["document"], mrl_dims(cfg, full), bin_dims(cfg, full))


def eval_base(cfg, mkey):
    out = results_dir(cfg, mkey) / "base.json"
    if out.exists():
        return read_json(out)
    mcfg = cfg["models"][mkey]
    log(f"eval base {mcfg['id']}")
    model = build_student(cfg, mcfg["id"], False, 512, eval_dtype(mcfg))
    model.eval()
    r = score(cfg, model)
    r.update(kind="base", model=mkey, id=mcfg["id"])
    write_json(out, r)
    model = None
    release()
    return r


def eval_run(cfg, mkey, vname, seed):
    out = results_dir(cfg, mkey) / f"{vname}__s{seed}.json"
    if out.exists():
        return read_json(out)
    rd = run_dir(cfg, mkey, vname, seed)
    if not (rd / "done.json").exists():
        return None
    log(f"eval {mkey} {vname} s{seed}")
    model = load_trained(rd / "model", eval_dtype(cfg["models"][mkey]))
    model.max_seq_length = 512
    r = score(cfg, model)
    r.update(kind="run", model=mkey, variant=vname, seed=seed, meta=read_json(rd / "done.json"))
    write_json(out, r)
    model = None
    release()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    cfg = load_cfg()
    hf_login()
    fails = []
    try:
        eval_base(cfg, a.model)
    except Exception as ex:
        fails.append({"run": "base", "error": repr(ex)})
        traceback.print_exc()
    for r in a.runs:
        v, s = r.split(":")
        try:
            eval_run(cfg, a.model, v, int(s))
        except Exception as ex:
            fails.append({"run": r, "error": repr(ex)})
            traceback.print_exc()
            release()
    write_json(results_dir(cfg, a.model) / "eval_failures.json", fails)
    log(f"eval {a.model} finished, {len(fails)} failed")


if __name__ == "__main__":
    main()
