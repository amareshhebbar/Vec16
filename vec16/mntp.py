import argparse
import math
import time

import torch
import torch.nn.functional as F

from vec16.bidirectional import to_bidirectional
from vec16.common import device, gpu_info, hf_login, linear_schedule, load_cfg, log, model_dir, read_jsonl, seed_all, train_precision, write_json
from vec16.students import fix_tokenizer


def mask_token_id(tok):
    if tok.mask_token_id is not None:
        return tok.mask_token_id
    for t in ("<mask>", "_"):
        ids = tok.convert_tokens_to_ids([t])
        if ids and ids[0] is not None and ids[0] != tok.unk_token_id:
            return ids[0]
    ids = tok("_", add_special_tokens=False)["input_ids"]
    return ids[0] if ids else tok.eos_token_id


def pack(tok, texts, seq, budget):
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    buf, chunks = [], []
    need = max(1, budget // seq)
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq:
            chunks.append(buf[:seq])
            buf = buf[seq:]
            if len(chunks) >= need:
                return torch.tensor(chunks)
    if not chunks and buf:
        chunks.append((buf + [eos] * seq)[:seq])
    return torch.tensor(chunks)


def mntp_loss(body, head, ids, mask_id, prob, softcap=None):
    labels = ids.clone()
    sel = torch.rand(ids.shape, device=ids.device) < prob
    sel[:, 0] = False
    if not sel.any():
        sel[:, 1] = True
    inp = ids.masked_fill(sel, mask_id)
    att = torch.ones_like(ids)
    hidden = body(input_ids=inp, attention_mask=att).last_hidden_state
    b, p = sel.nonzero(as_tuple=True)
    h = hidden[b, p - 1]
    logits = head(h).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return F.cross_entropy(logits, labels[b, p])


def run(cfg, mkey):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_cfg, tr = cfg["run"], cfg["train"]
    mcfg = cfg["models"][mkey]
    out = model_dir(cfg, mkey) / "mntp"
    if (out / "done.json").exists():
        log(f"mntp {mkey} already done")
        return
    seed_all(0)
    prec = train_precision(mcfg)
    log(f"mntp {mkey} precision {prec}")
    tok = fix_tokenizer(AutoTokenizer.from_pretrained(mcfg["id"]))
    lm = AutoModelForCausalLM.from_pretrained(mcfg["id"], dtype=torch.float32, attn_implementation="sdpa")
    body = getattr(lm, lm.base_model_prefix)
    head = lm.get_output_embeddings()
    to_bidirectional(body)
    body.config.use_cache = False
    if tr.get("grad_checkpointing"):
        body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = device()
    lm.to(dev)
    lm.train()
    texts = [r["text"] for r in read_jsonl(cfg["paths"]["data"] / "mntp.jsonl")]
    seq = run_cfg["mntp_seq"]
    chunks = pack(tok, texts, seq, run_cfg["mntp_tokens"])
    bs = mcfg.get("mntp_batch", run_cfg["mntp_batch"])
    steps = max(1, math.ceil(len(chunks) / bs))
    mid = mask_token_id(tok)
    softcap = getattr(lm.config, "final_logit_softcapping", None)
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=tr["mntp_lr"], weight_decay=tr["weight_decay"])
    warm = max(1, int(steps * tr["warmup_ratio"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_schedule(warm, steps))
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    scaler = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
    perm = torch.randperm(len(chunks))
    t0, hist = time.time(), []
    for step in range(steps):
        ids = chunks[perm[step * bs : (step + 1) * bs]].to(dev)
        with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=amp or torch.float32, enabled=amp is not None):
            loss = mntp_loss(body, head, ids, mid, tr["mntp_mask"], softcap)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lm.parameters(), tr["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 20 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss.item()), "lr": sched.get_last_lr()[0]})
            log(f"mntp {mkey} step {step + 1}/{steps} loss {loss.item():.4f}")
    out.mkdir(parents=True, exist_ok=True)
    body.gradient_checkpointing_disable()
    body.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    write_json(out / "log.json", hist)
    write_json(
        out / "done.json",
        {
            "model": mkey,
            "base_id": mcfg["id"],
            "steps": steps,
            "tokens": int(chunks.numel()),
            "seq": seq,
            "batch": bs,
            "mask_prob": tr["mntp_mask"],
            "mask_token_id": int(mid),
            "final_loss": hist[-1]["loss"] if hist else None,
            "seconds": round(time.time() - t0, 1),
            "precision": prec,
            "gpu": gpu_info().get("name"),
        },
    )
    log(f"mntp {mkey} saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()
    cfg = load_cfg()
    hf_login()
    run(cfg, a.model)


if __name__ == "__main__":
    main()
