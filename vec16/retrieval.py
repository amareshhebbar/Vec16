import math

import torch
import torch.nn.functional as F

from vec16.common import log, read_json, write_json

NANO = {
    "arguana": "NanoArguAna",
    "climatefever": "NanoClimateFEVER",
    "dbpedia": "NanoDBPedia",
    "fever": "NanoFEVER",
    "fiqa2018": "NanoFiQA2018",
    "hotpotqa": "NanoHotpotQA",
    "msmarco": "NanoMSMARCO",
    "nfcorpus": "NanoNFCorpus",
    "nq": "NanoNQ",
    "quoraretrieval": "NanoQuoraRetrieval",
    "scidocs": "NanoSCIDOCS",
    "scifact": "NanoSciFact",
    "touche2020": "NanoTouche2020",
}


def _col(ds, *names):
    for n in names:
        if n in ds.column_names:
            return n
    return ds.column_names[0]


def load_nano(cfg, name):
    cache = cfg["paths"]["cache"] / "nanobeir" / f"{name}.json"
    if cache.exists():
        return read_json(cache)
    from datasets import load_dataset

    path = f"zeta-alpha-ai/{NANO[name]}"
    corpus = load_dataset(path, "corpus", split="train")
    queries = load_dataset(path, "queries", split="train")
    qrels = load_dataset(path, "qrels", split="train")
    cid, ctx = _col(corpus, "_id", "id"), _col(corpus, "text")
    qid, qtx = _col(queries, "_id", "id"), _col(queries, "text")
    rq, rc = _col(qrels, "query-id", "query_id"), _col(qrels, "corpus-id", "corpus_id")
    docs = {str(r[cid]): r[ctx] for r in corpus if r[ctx]}
    qs = {str(r[qid]): r[qtx] for r in queries if r[qtx]}
    rel = {}
    for r in qrels:
        q, c = str(r[rq]), str(r[rc])
        if q in qs and c in docs:
            rel.setdefault(q, []).append(c)
    qids = [q for q in qs if q in rel]
    dids = list(docs)
    index = {d: i for i, d in enumerate(dids)}
    out = {
        "queries": [qs[q] for q in qids],
        "docs": [docs[d] for d in dids],
        "qrels": [sorted({index[c] for c in rel[q]}) for q in qids],
    }
    write_json(cache, out)
    return out


def encode(model, texts, prompt, batch_size):
    kw = {"batch_size": batch_size, "convert_to_tensor": True, "show_progress_bar": False, "normalize_embeddings": False}
    if prompt:
        kw["prompt"] = prompt
    with torch.inference_mode():
        return model.encode(texts, **kw).float()


def ndcg_at_k(scores, qrels, k=10):
    k = min(k, scores.shape[1])
    top = torch.topk(scores, k, dim=1).indices.cpu().tolist()
    total = 0.0
    for row, rel in zip(top, qrels):
        rs = set(rel)
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(row) if d in rs)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rs), k)))
        total += dcg / idcg if idcg else 0.0
    return total / max(1, len(qrels))


def binarize(x):
    return torch.where(x > 0, 1.0, -1.0).to(x.dtype)


def chunked_scores(q, d, chunk=4096):
    out = []
    for i in range(0, q.shape[0], chunk):
        out.append(q[i : i + chunk] @ d.T)
    return torch.cat(out)


def score_grid(q, d, qrels, float_dims, binary_dims, k=10):
    res = {"float": {}, "binary": {}, "asym": {}}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q, d = q.to(dev), d.to(dev)
    for dim in float_dims:
        qq, dd = F.normalize(q[:, :dim], dim=-1), F.normalize(d[:, :dim], dim=-1)
        res["float"][str(dim)] = ndcg_at_k(chunked_scores(qq, dd), qrels, k)
    for dim in binary_dims:
        qb, db = binarize(q[:, :dim]), binarize(d[:, :dim])
        res["binary"][str(dim)] = ndcg_at_k(chunked_scores(qb, db), qrels, k)
        qf = F.normalize(q[:, :dim], dim=-1)
        res["asym"][str(dim)] = ndcg_at_k(chunked_scores(qf, db), qrels, k)
    return res


def evaluate_nano(cfg, model, qprompt, dprompt, float_dims, binary_dims):
    run = cfg["run"]
    per = {}
    for name in run["nanobeir"]:
        data = load_nano(cfg, name)
        q = encode(model, data["queries"], qprompt, run["eval_batch"])
        d = encode(model, data["docs"], dprompt, run["eval_batch"])
        per[name] = score_grid(q, d, data["qrels"], float_dims, binary_dims)
        del q, d
    agg = {}
    for kind in ("float", "binary", "asym"):
        agg[kind] = {}
        dims = float_dims if kind == "float" else binary_dims
        for dim in dims:
            vals = [per[n][kind][str(dim)] for n in per]
            agg[kind][str(dim)] = sum(vals) / len(vals)
    full = max(float_dims)
    log("nanobeir " + " ".join(f"{n}={per[n]['float'][str(full)]:.3f}" for n in per))
    return {"mean": agg, "detail": per, "full_dim": full}
