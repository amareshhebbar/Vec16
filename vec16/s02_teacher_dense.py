import numpy as np
import torch

from vec16.common import device, hf_login, infer_dtype, load_cfg, log, model_kwargs, read_jsonl, release, update_status, write_json


def load_teacher(cfg):
    from sentence_transformers import SentenceTransformer

    tid = cfg["run"]["teacher_dense"]
    dtype = infer_dtype()
    log(f"dense teacher {tid} dtype {dtype}")
    return SentenceTransformer(tid, device=device(), model_kwargs=model_kwargs(dtype), trust_remote_code=True)


def encode_chunks(model, texts, prompt, seq_len, batch, out_dir, tag, chunk=20000):
    model.max_seq_length = seq_len
    parts = []
    for start in range(0, len(texts), chunk):
        f = out_dir / f"{tag}_{start // chunk:05d}.npy"
        if f.exists():
            parts.append(np.load(f))
            continue
        kw = {"batch_size": batch, "convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": False}
        if prompt:
            kw["prompt"] = prompt
        with torch.inference_mode():
            emb = model.encode(texts[start : start + chunk], **kw).astype(np.float16)
        np.save(f, emb)
        parts.append(emb)
        log(f"{tag} {min(start + chunk, len(texts))}/{len(texts)}")
    return np.concatenate(parts)


def search(q, d, k, chunk=2048):
    dev = device()
    dt = torch.float16 if dev == "cuda" else torch.float32
    dt_ = torch.from_numpy(d).to(dev, dt)
    ids, scores = [], []
    for i in range(0, len(q), chunk):
        qq = torch.from_numpy(q[i : i + chunk]).to(dev, dt)
        s = (qq @ dt_.T).float()
        v, ix = torch.topk(s, min(k, s.shape[1]), dim=1)
        ids.append(ix.cpu())
        scores.append(v.cpu())
    return torch.cat(ids).numpy(), torch.cat(scores).numpy()


def assemble(queries, q, d, top_ids, top_scores, k):
    n = len(queries)
    cand = np.zeros((n, k), dtype=np.int32)
    score = np.zeros((n, k), dtype=np.float32)
    for i, r in enumerate(queries):
        pos = r["pos"]
        others = [(int(c), float(s)) for c, s in zip(top_ids[i], top_scores[i]) if int(c) != pos]
        ps = float(np.dot(q[i].astype(np.float32), d[pos].astype(np.float32)))
        row = [(pos, ps)] + others[: k - 1]
        while len(row) < k:
            row.append(row[-1])
        cand[i] = [c for c, _ in row]
        score[i] = [s for _, s in row]
    return cand, score


def main():
    cfg = load_cfg()
    hf_login()
    run, tc = cfg["run"], cfg["teachers"]
    data, tdir = cfg["paths"]["data"], cfg["paths"]["teacher"]
    done = tdir / "dense_done.json"
    if done.exists():
        log("dense teacher already done")
        return
    queries = read_jsonl(data / "queries.jsonl")
    corpus = read_jsonl(data / "corpus.jsonl")
    if not queries:
        raise SystemExit("run s01_data first")
    parts = tdir / "dense_parts"
    parts.mkdir(exist_ok=True)
    model = load_teacher(cfg)
    d = encode_chunks(model, [c["text"] for c in corpus], tc.get("dense_doc_prompt"), run["doc_len"], run["teacher_batch"], parts, "doc")
    q = encode_chunks(model, [r["query"] for r in queries], tc.get("dense_query_prompt"), run["query_len"], run["teacher_batch"], parts, "query")
    model = None
    release()
    k = run["teacher_k"]
    top_ids, top_scores = search(q, d, k + 1)
    cand, score = assemble(queries, q, d, top_ids, top_scores, k)
    np.save(tdir / "cand.npy", cand)
    np.save(tdir / "dense.npy", score.astype(np.float16))
    hit = float(np.mean([int(top_ids[i][0]) == r["pos"] for i, r in enumerate(queries)]))
    info = {"teacher": run["teacher_dense"], "k": k, "queries": len(queries), "corpus": len(corpus), "dim": int(d.shape[1]), "top1_is_positive": hit}
    write_json(done, info)
    update_status(cfg, "teacher_dense", info)
    log("dense teacher done", info)


if __name__ == "__main__":
    main()
