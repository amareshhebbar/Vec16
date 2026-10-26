import argparse
import math
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from vec16.common import (
    bin_dims,
    device,
    gpu_info,
    headline_bits,
    hf_login,
    linear_schedule,
    load_cfg,
    log,
    model_dir,
    mrl_dims,
    read_jsonl,
    run_dir,
    seed_all,
    train_precision,
    train_queries,
    variant_spec,
    write_json,
)
from vec16.retrieval import score_grid
from vec16.students import build_student, finalize_dir


def ste(x):
    return x + (torch.sign(x) - x).detach()


def teacher_probs(spec, tc, dense, colbert):
    pd = torch.softmax(dense / tc["temp_dense"], -1)
    pc = torch.softmax(colbert / tc["temp_colbert"], -1)
    if spec["teacher"] == "dense":
        return pd
    if spec["teacher"] == "colbert":
        return pc
    return 0.5 * (pd + pc)


def nce_mask(doc_ids, probs, k, margin):
    b = doc_ids.shape[0]
    flat = doc_ids.reshape(-1)
    pos = doc_ids[:, 0]
    mask = flat[None, :] == pos[:, None]
    own = torch.arange(b)[:, None] * k
    mask[torch.arange(b), own[:, 0]] = False
    for i in range(b):
        hard = probs[i, 1:] >= probs[i, 0] - margin
        cols = own[i, 0] + 1 + torch.nonzero(hard).flatten()
        mask[i, cols] = True
    return mask


def pair_losses(q, d, b, k, target, mask, spec, tr):
    out = {}
    if spec["contrastive"]:
        logits = (q @ d.T) / tr["temp_nce"]
        logits = logits.masked_fill(mask, -1e4)
        labels = torch.arange(b, device=q.device) * k
        out["nce"] = F.cross_entropy(logits.float(), labels)
    if spec["listwise"]:
        s = torch.einsum("bh,bkh->bk", q, d.view(b, k, -1)) / tr["temp_list"]
        out["list"] = F.kl_div(F.log_softmax(s.float(), -1), target, reduction="batchmean")
    return out


def total_loss(qe, de, b, k, target, mask, spec, tr, dims, bdims, qat_w):
    parts, total = {}, 0.0
    for dim in dims:
        qq, dd = F.normalize(qe[:, :dim], dim=-1), F.normalize(de[:, :dim], dim=-1)
        for name, v in pair_losses(qq, dd, b, k, target, mask, spec, tr).items():
            w = tr["w_nce"] if name == "nce" else tr["w_list"]
            total = total + w * v / len(dims)
            parts[f"{name}_{dim}"] = float(v.detach())
    if spec["qat"] and qat_w > 0:
        for dim in bdims:
            sc = 1.0 / math.sqrt(dim)
            qq, dd = ste(qe[:, :dim]) * sc, ste(de[:, :dim]) * sc
            for name, v in pair_losses(qq, dd, b, k, target, mask, spec, tr).items():
                w = tr["w_nce"] if name == "nce" else tr["w_list"]
                total = total + qat_w * w * v / len(bdims)
                parts[f"bin_{name}_{dim}"] = float(v.detach())
    return total, parts


def embed(st, texts, dev):
    feats = st.tokenize(texts)
    feats = {k: v.to(dev) for k, v in feats.items()}
    return st(feats)["sentence_embedding"]


class Data:
    def __init__(self, cfg, limit=None):
        data, tdir = cfg["paths"]["data"], cfg["paths"]["teacher"]
        self.queries = read_jsonl(data / "queries.jsonl")
        self.corpus = [c["text"] for c in read_jsonl(data / "corpus.jsonl")]
        self.cand = np.load(tdir / "cand.npy")
        self.dense = np.load(tdir / "dense.npy").astype(np.float32)
        self.colbert = np.load(tdir / "colbert.npy").astype(np.float32)
        self.train = [i for i, q in enumerate(self.queries) if q["split"] == "train"]
        if limit:
            self.train = self.train[:limit]
        self.dev = [i for i, q in enumerate(self.queries) if q["split"] == "dev"]


def dev_eval(cfg, st, data, bits):
    run = cfg["run"]
    idx = data.dev[: run["dev_max"]]
    if not idx:
        return {}
    pool = sorted({int(c) for i in idx for c in data.cand[i]})
    where = {c: j for j, c in enumerate(pool)}
    qrels = [[where[int(data.cand[i][0])]] for i in idx]
    st.eval()
    with torch.inference_mode():
        q = st.encode([data.queries[i]["query"] for i in idx], prompt=cfg["prompts"]["query"], batch_size=run["eval_batch"], convert_to_tensor=True, show_progress_bar=False).float()
        d = st.encode([data.corpus[c] for c in pool], prompt=cfg["prompts"]["document"], batch_size=run["eval_batch"], convert_to_tensor=True, show_progress_bar=False).float()
    st.train()
    full = q.shape[1]
    g = score_grid(q, d, qrels, [full, min(128, full)], [bits] if bits <= full else [])
    out = {"dev_float": g["float"][str(full)], "dev_float128": g["float"][str(min(128, full))]}
    if g["binary"]:
        out["dev_binary"] = g["binary"][str(bits)]
        out["dev_mixed"] = 0.5 * (out["dev_float"] + out["dev_binary"])
    else:
        out["dev_mixed"] = out["dev_float"]
    return out


def run_one(cfg, mkey, vname, seed, teacher, batch_override=None):
    run, tr, tc = cfg["run"], cfg["train"], cfg["teachers"]
    mcfg = cfg["models"][mkey]
    spec = variant_spec(cfg, vname, teacher)
    out = run_dir(cfg, mkey, vname, seed)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    prec = train_precision(mcfg)
    dev = device()
    source = model_dir(cfg, mkey) / "mntp" if spec["mntp"] else mcfg["id"]
    log(f"train {mkey} {vname} seed {seed} teacher {spec['teacher']} source {source} precision {prec}")
    st = build_student(cfg, source, spec["bidir"], run["doc_len"])
    if tr.get("grad_checkpointing"):
        st[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    st.to(dev)
    st.train()
    data = Data(cfg, train_queries(cfg, mkey, spec))
    full = st.get_sentence_embedding_dimension()
    dims, bdims = mrl_dims(cfg, full), bin_dims(cfg, full)
    bits = headline_bits(cfg)
    b = batch_override or mcfg.get("batch_size", run["batch_size"])
    k = min(run["train_k"], data.cand.shape[1])
    epochs = run["epochs"]
    per_epoch = max(1, len(data.train) // b)
    steps = per_epoch * epochs
    lr = mcfg.get("lr", tr["lr"])
    opt = torch.optim.AdamW([p for p in st.parameters() if p.requires_grad], lr=lr, weight_decay=tr["weight_decay"])
    warm = max(1, int(steps * tr["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_schedule(warm, steps))
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    scaler = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(data.train) for _ in range(epochs)])
    qp, dp = cfg["prompts"]["query"], cfg["prompts"]["document"]
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(out / "tb"))
    hist, start = [], 0
    ck = out / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        st.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start, hist = state["step"], state["hist"]
        log(f"resumed at step {start}")
    t0 = time.time()
    for step in range(start, steps):
        idx = order[step * b : (step + 1) * b]
        sel = np.stack([np.concatenate([[0], np.sort(rng.choice(np.arange(1, data.cand.shape[1]), k - 1, replace=False))]) for _ in idx])
        doc_ids = np.stack([data.cand[i][s] for i, s in zip(idx, sel)])
        td = torch.tensor(np.stack([data.dense[i][s] for i, s in zip(idx, sel)]))
        tcb = torch.tensor(np.stack([data.colbert[i][s] for i, s in zip(idx, sel)]))
        probs = teacher_probs(spec, tc, td, tcb)
        mask = nce_mask(torch.tensor(doc_ids), probs, k, tr["false_negative_margin"]).to(dev)
        target = probs.to(dev)
        qtexts = [qp + data.queries[i]["query"] for i in idx]
        dtexts = [dp + data.corpus[int(c)] for c in doc_ids.reshape(-1)]
        qat_w = tr["w_qat"] * min(1.0, (step + 1) / max(1, tr["qat_ramp"] * steps))
        with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=amp or torch.float32, enabled=amp is not None):
            qe = embed(st, qtexts, dev).float()
            de = embed(st, dtexts, dev).float()
        loss, parts = total_loss(qe, de, len(idx), k, target, mask, spec, tr, dims, bdims, qat_w)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), tr["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 10 == 0 or step == steps - 1:
            rec = {"step": step + 1, "loss": float(loss.detach()), "lr": sched.get_last_lr()[0], "grad_norm": float(gn), "qat_w": qat_w}
            rec.update({kk: v for kk, v in parts.items() if kk.endswith(f"_{full}") or kk.endswith(f"_{bits}")})
            hist.append(rec)
            for kk, v in rec.items():
                if kk != "step":
                    tb.add_scalar(f"train/{kk}", v, step + 1)
        if step % 50 == 0:
            log(f"{mkey} {vname} s{seed} step {step + 1}/{steps} loss {loss.item():.4f}")
        if run["eval_steps"] and (step + 1) % run["eval_steps"] == 0 and step + 1 < steps:
            m = dev_eval(cfg, st, data, bits)
            hist.append({"step": step + 1, **m})
            for kk, v in m.items():
                tb.add_scalar(f"dev/{kk}", v, step + 1)
            log(f"dev {m}")
        if run.get("save_steps") and (step + 1) % run["save_steps"] == 0 and step + 1 < steps:
            torch.save({"model": st.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step + 1, "hist": hist}, str(ck) + ".tmp")
            shutil.move(str(ck) + ".tmp", ck)
    final = dev_eval(cfg, st, data, bits)
    hist.append({"step": steps, **final})
    for kk, v in final.items():
        tb.add_scalar(f"dev/{kk}", v, steps)
    tb.close()
    st[0].auto_model.gradient_checkpointing_disable()
    shutil.rmtree(out / "model", ignore_errors=True)
    st.save(str(out / "model"))
    finalize_dir(cfg, mkey, out / "model", spec["bidir"])
    write_json(out / "train_log.json", hist)
    write_json(
        out / "done.json",
        {
            "model": mkey,
            "base_id": mcfg["id"],
            "variant": vname,
            "seed": seed,
            "spec": spec,
            "precision": prec,
            "seconds": round(time.time() - t0, 1),
            "gpu": gpu_info().get("name"),
            "steps": steps,
            "batch_size": b,
            "k": k,
            "lr": lr,
            "epochs": epochs,
            "doc_len": run["doc_len"],
            "dims": dims,
            "binary_dims": bdims,
            "train_queries": len(data.train),
            **final,
        },
    )
    ck.unlink(missing_ok=True)
    log(f"saved {out / 'model'} dev {final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("variant")
    ap.add_argument("seed", type=int)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    hf_login()
    try:
        run_one(cfg, a.model, a.variant, a.seed, a.teacher, a.batch)
    except torch.cuda.OutOfMemoryError:
        write_json(run_dir(cfg, a.model, a.variant, a.seed) / "oom.json", {"batch": a.batch})
        log("out of memory")
        sys.exit(3)


if __name__ == "__main__":
    main()
