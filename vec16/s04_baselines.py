import traceback

from vec16.common import bin_dims, device, eval_dtype, hf_login, load_cfg, log, model_kwargs, mrl_dims, release, update_status, write_json
from vec16.retrieval import evaluate_nano


def baseline_dims(cfg, bcfg, full):
    if bcfg.get("mrl"):
        return mrl_dims(cfg, full), sorted(set(bin_dims(cfg, full)) | {full}, reverse=True)
    return [full], [full]


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
    write_json(res / "baseline_failures.json", fails)
    update_status(cfg, "baselines", {"done": len(cfg["run"]["baselines"]) - len(fails), "failed": len(fails)})


if __name__ == "__main__":
    main()
