import shutil

import numpy as np
import pandas as pd

from vec16 import hub
from vec16.common import dataset_name, hf_login, load_cfg, log, read_json, read_jsonl, update_status, write_json
from vec16.modelcard import dataset_card


def stage(cfg, root):
    data, tdir = cfg["paths"]["data"], cfg["paths"]["teacher"]
    queries = read_jsonl(data / "queries.jsonl")
    corpus = read_jsonl(data / "corpus.jsonl")
    cand = np.load(tdir / "cand.npy")
    dense = np.load(tdir / "dense.npy").astype(np.float32)
    colbert = np.load(tdir / "colbert.npy").astype(np.float32)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    counts = {}
    for split in ("train", "dev"):
        idx = [i for i, q in enumerate(queries) if q["split"] == split]
        df = pd.DataFrame(
            {
                "qid": [queries[i]["qid"] for i in idx],
                "query": [queries[i]["query"] for i in idx],
                "source": [queries[i]["src"] for i in idx],
                "positive_id": [queries[i]["pos"] for i in idx],
                "candidate_ids": [cand[i].tolist() for i in idx],
                "dense_scores": [np.round(dense[i], 5).tolist() for i in idx],
                "colbert_scores": [np.round(colbert[i], 5).tolist() for i in idx],
            }
        )
        df.to_parquet(root / f"{split}.parquet", index=False)
        counts[split] = len(idx)
    (root / "corpus").mkdir()
    pd.DataFrame(corpus, columns=["id", "text"]).to_parquet(root / "corpus" / "corpus.parquet", index=False)
    counts["corpus"] = len(corpus)
    info = {
        "data": read_json(data / "done.json"),
        "dense": read_json(tdir / "dense_done.json"),
        "colbert": read_json(tdir / "colbert_done.json"),
        "counts": counts,
    }
    (root / "README.md").write_text(dataset_card(cfg, info))
    return info


def main():
    cfg = load_cfg()
    hf_login()
    name = dataset_name(cfg)
    root = cfg["paths"]["publish"] / name
    info = stage(cfg, root)
    write_json(cfg["paths"]["results"] / "dataset.json", info)
    a = hub.api()
    if a is None:
        log("publish disabled, dataset staged at", root)
        return
    user = cfg["project"]["hf_user"]
    repo = f"{user}/{name}"
    hub.push_folder(a, repo, root, kind="dataset", private=bool(cfg["run"]["private"]), message="Upload distillation data")
    note = f"{info['counts']['train']} queries, top {info['dense']['k']} candidates scored by a dense and a late interaction teacher"
    slug = hub.add_to_collection(cfg, a, repo, "dataset", note)
    update_status(cfg, "dataset", {"repo": repo, "collection": slug})
    log(f"published {repo}, collection https://huggingface.co/collections/{slug}")


if __name__ == "__main__":
    main()
