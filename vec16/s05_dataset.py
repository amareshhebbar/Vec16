import shutil

import numpy as np
import pandas as pd

from vec16 import hub
from vec16.common import dataset_name, hf_login, load_cfg, log, read_json, read_jsonl, update_status, write_json
from vec16.modelcard import dataset_card


def snippet(t, n=300):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


def false_negatives(queries, corpus, cand, dense, colbert, top=25, dm=0.02, cm=0.01):
    idx = np.array([i for i, q in enumerate(queries) if q["split"] == "train"])
    if len(idx) == 0:
        return {"rate": 0.0, "examples": []}
    dd = dense[idx, 1:] - dense[idx, :1]
    cc = colbert[idx, 1:] - colbert[idx, :1]
    both = (dd > dm) & (cc > cm) & (cand[idx, 1:] != cand[idx, :1])
    hit = both.any(1)
    margin = np.where(both, dd / max(dm, 1e-6) + cc / max(cm, 1e-6), -np.inf)
    best_j = margin.argmax(1)
    best_m = margin.max(1)
    order = np.argsort(-best_m)
    texts = {c["id"]: c["text"] for c in corpus}
    examples, seen = [], set()
    for o in order:
        if not np.isfinite(best_m[o]) or len(examples) >= top:
            break
        i, j = int(idx[o]), int(best_j[o]) + 1
        q = queries[i]
        if q["query"] in seen:
            continue
        seen.add(q["query"])
        examples.append(
            {
                "query": q["query"],
                "source": q["src"],
                "labeled_positive": snippet(texts[int(cand[i][0])]),
                "teacher_pick": snippet(texts[int(cand[i][j])]),
                "dense": [round(float(dense[i][0]), 4), round(float(dense[i][j]), 4)],
                "colbert": [round(float(colbert[i][0]), 4), round(float(colbert[i][j]), 4)],
            }
        )
    by_src = {}
    for o, h in zip(idx, hit):
        src = queries[int(o)]["src"]
        a = by_src.setdefault(src, [0, 0])
        a[0] += int(h)
        a[1] += 1
    return {
        "rate": float(hit.mean()),
        "by_source": {k: v[0] / v[1] for k, v in by_src.items()},
        "rule": f"both teachers score another candidate above the labeled positive, dense by more than {dm}, ColBERT by more than {cm}",
        "examples": examples,
    }


def gallery_md(fn):
    lines = [
        "# Labeled negatives that are probably correct",
        "",
        f"For {fn['rate'] * 100:.1f}% of training queries, {fn['rule']}. Training never uses those candidates as negatives.",
        "",
        "| Source | Share of queries |",
        "|---|---|",
    ]
    for k, v in sorted(fn.get("by_source", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v * 100:.1f}% |")
    lines.append("")
    for i, e in enumerate(fn["examples"], 1):
        lines += [
            f"## {i}. {e['query']}",
            "",
            f"Labeled positive (dense {e['dense'][0]}, ColBERT {e['colbert'][0]}):",
            "",
            f"> {e['labeled_positive']}",
            "",
            f"Teachers prefer (dense {e['dense'][1]}, ColBERT {e['colbert'][1]}):",
            "",
            f"> {e['teacher_pick']}",
            "",
        ]
    return "\n".join(lines)


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
    fn = false_negatives(queries, corpus, cand, dense, colbert)
    write_json(cfg["paths"]["results"] / "false_negatives.json", fn)
    (root / "false_negatives.md").write_text(gallery_md(fn))
    info = {
        "data": read_json(data / "done.json"),
        "dense": read_json(tdir / "dense_done.json"),
        "colbert": read_json(tdir / "colbert_done.json"),
        "counts": counts,
        "false_negative_rate": fn["rate"],
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
