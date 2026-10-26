import traceback

from vec16.common import bin_dims, device, eval_dtype, hf_login, load_cfg, log, model_kwargs, mrl_dims, release, update_status, write_json
from vec16.retrieval import evaluate_nano, evaluate_nano_late


def baseline_dims(cfg, bcfg, full):
    if bcfg.get("mrl"):
        return mrl_dims(cfg, full), sorted(set(bin_dims(cfg, full)) | {full}, reverse=True)
    return [full], [full]


def teachers(cfg, res):
    from sentence_transformers import SentenceTransformer

    fails = []
    tc = cfg["teachers"]
    out = res / "teacher_dense.json"
    if not out.exists():
        tid = cfg["run"]["teacher_dense"]
        log(f"teacher {tid} on NanoBEIR")
        model = None
        try:
            model = SentenceTransformer(tid, device=device(), model_kwargs=model_kwargs(eval_dtype()), trust_remote_code=True)
            model.max_seq_length = 512
            full = model.get_sentence_embedding_dimension()
            r = evaluate_nano(cfg, model, tc.get("dense_query_prompt", ""), tc.get("dense_doc_prompt", ""), mrl_dims(cfg, full), sorted(set(bin_dims(cfg, full)) | {full}, reverse=True))
            r.update(key="teacher_dense", id=tid, label=f"teacher {tid.split('/')[-1]}", dim=full, mrl=True)
            write_json(out, r)
        except Exception as ex:
            fails.append({"baseline": "teacher_dense", "error": repr(ex)})
            traceback.print_exc()
        finally:
            model = None
            release()
    out = res / "teacher_colbert.json"
    if not out.exists():
        log(f"teacher {tc['colbert']} on NanoBEIR")
        model = None
        try:
            from vec16.s03_teacher_colbert import load_colbert

            model = load_colbert(cfg)
            r = evaluate_nano_late(cfg, model, cfg["run"]["eval_batch"])
            r.update(key="teacher_colbert", id=tc["colbert"], label=f"teacher {tc['colbert'].split('/')[-1]}")
            write_json(out, r)
            log(f"colbert teacher nDCG@10 {r['late']:.4f}, {r['bytes_per_doc']:.0f} bytes per document")
        except Exception as ex:
            fails.append({"baseline": "teacher_colbert", "error": repr(ex)})
            traceback.print_exc()
        finally:
            model = None
            release()
    return fails


def main():
    from sentence_transformers import SentenceTransformer

    cfg = load_cfg()
    hf_login()
    res = cfg["paths"]["results"]
    fails = []
    for key in cfg["run"]["baselines"]:
        out = res / f"baseline_{key}.json"
        if out.exists():
            log(f"baseline {key} already scored")
            continue
        bcfg = cfg["baselines"][key]
        log(f"baseline {key} {bcfg['id']}")
        model = None
        try:
            model = SentenceTransformer(bcfg["id"], device=device(), model_kwargs=model_kwargs(eval_dtype(bcfg)), trust_remote_code=bool(bcfg.get("trust_remote_code")))
            model.max_seq_length = min(512, model.max_seq_length or 512)
            full = model.get_sentence_embedding_dimension()
            fd, bd = baseline_dims(cfg, bcfg, full)
            r = evaluate_nano(cfg, model, bcfg.get("query_prompt", ""), bcfg.get("doc_prompt", ""), fd, bd)
            r.update(key=key, id=bcfg["id"], label=bcfg.get("label", key), dim=full, mrl=bool(bcfg.get("mrl")))
            write_json(out, r)
            log(f"baseline {key} float {r['mean']['float'][str(full)]:.4f} binary {r['mean']['binary'][str(full)]:.4f}")
        except Exception as ex:
            fails.append({"baseline": key, "error": repr(ex)})
            traceback.print_exc()
        finally:
            model = None
            release()
    fails += teachers(cfg, res)
    write_json(res / "baseline_failures.json", fails)
    update_status(cfg, "baselines", {"done": len(cfg["run"]["baselines"]) - len(fails), "failed": len(fails)})


if __name__ == "__main__":
    main()
