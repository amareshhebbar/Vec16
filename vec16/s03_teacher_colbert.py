import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from vec16.common import device, hf_login, infer_dtype, load_cfg, log, model_kwargs, read_jsonl, release, update_status, write_json


def load_colbert(cfg):
    from pylate import models

    tid = cfg["teachers"]["colbert"]
    dtype = infer_dtype()
    log(f"colbert teacher {tid} dtype {dtype}")
    return models.ColBERT(model_name_or_path=tid, device=device(), model_kwargs=model_kwargs(dtype))


def encode_tokens(model, texts, is_query, batch):
    with torch.inference_mode():
        out = model.encode(texts, batch_size=batch, is_query=is_query, convert_to_numpy=False, convert_to_tensor=False, show_progress_bar=False)
    return [t.to("cpu", torch.float16) for t in out]


def padded(tensors, dev):
    lens = torch.tensor([t.shape[0] for t in tensors])
    x = pad_sequence(tensors, batch_first=True).to(dev, torch.float32)
    mask = torch.arange(x.shape[1])[None, :] < lens[:, None]
    return x, mask.to(dev)


def maxsim(qx, qm, dx, dm):
    sim = torch.einsum("bqh,bkdh->bkqd", qx, dx)
    sim = sim.masked_fill(~dm[:, :, None, :], -1e4)
    best = sim.max(dim=-1).values
    best = best * qm[:, None, :]
    return best.sum(-1) / qm.sum(-1, keepdim=True).clamp(min=1)


def main():
    cfg = load_cfg()
    hf_login()
    run = cfg["run"]
    data, tdir = cfg["paths"]["data"], cfg["paths"]["teacher"]
    done = tdir / "colbert_done.json"
    if done.exists():
        log("colbert teacher already done")
        return
    if not (tdir / "dense_done.json").exists():
        raise SystemExit("run s02_teacher_dense first")
    queries = read_jsonl(data / "queries.jsonl")
    corpus = read_jsonl(data / "corpus.jsonl")
    cand = np.load(tdir / "cand.npy")
    model = load_colbert(cfg)
    bs = run["teacher_batch"]
    dev = device()
    parts = tdir / "colbert_parts"
    parts.mkdir(exist_ok=True)
    scores = np.zeros(cand.shape, dtype=np.float32)
    chunk = run["colbert_chunk"]
    inner = 32
    texts = [c["text"] for c in corpus]
    for start in range(0, len(queries), chunk):
        part = parts / f"part_{start // chunk:05d}.npy"
        stop = min(start + chunk, len(queries))
        if part.exists():
            scores[start:stop] = np.load(part)
            continue
        need = sorted({int(c) for c in cand[start:stop].reshape(-1)})
        enc = encode_tokens(model, [texts[c] for c in need], False, bs)
        docs = dict(zip(need, enc))
        qt = encode_tokens(model, [r["query"] for r in queries[start:stop]], True, bs)
        for j in range(0, len(qt), inner):
            rows = range(start + j, min(start + j + inner, stop))
            qx, qm = padded(qt[j : j + inner], dev)
            flat = [docs[int(c)] for r in rows for c in cand[r]]
            dx, dm = padded(flat, dev)
            b, k = len(rows), cand.shape[1]
            dx = dx.view(b, k, dx.shape[1], dx.shape[2])
            dm = dm.view(b, k, dm.shape[1])
            scores[start + j : start + j + b] = maxsim(qx, qm.float(), dx, dm).cpu().numpy()
        np.save(part, scores[start:stop])
        log(f"colbert queries {stop}/{len(queries)}, {len(need)} passages encoded for this chunk")
    model = None
    release()
    np.save(tdir / "colbert.npy", scores.astype(np.float16))
    dense = np.load(tdir / "dense.npy").astype(np.float32)
    agree = float(np.mean(np.argmax(scores, 1) == np.argmax(dense, 1)))
    pos_top = float(np.mean(np.argmax(scores, 1) == 0))
    info = {"teacher": cfg["teachers"]["colbert"], "k": int(cand.shape[1]), "top1_agree_with_dense": agree, "top1_is_positive": pos_top}
    write_json(done, info)
    update_status(cfg, "teacher_colbert", info)
    log("colbert teacher done", info)


if __name__ == "__main__":
    main()
