import math
import random

from vec16.common import hf_login, load_cfg, log, read_json, update_status, write_json, write_jsonl


def open_source(src, streaming=True):
    from datasets import load_dataset

    if src.get("files"):
        return load_dataset("json", data_files=src["files"], split="train", streaming=streaming)
    kw = {"split": src.get("split", "train"), "streaming": streaming}
    if src.get("config"):
        try:
            return load_dataset(src["id"], src["config"], **kw)
        except Exception as ex:
            log(f"{src['id']} config {src['config']} failed ({type(ex).__name__}), trying default config")
    return load_dataset(src["id"], **kw)


def columns(row):
    skip = {"id", "_id", "qid", "url", "source", "title", "dataset"}
    keys = [k for k, v in row.items() if isinstance(v, str) and k.lower() not in skip]
    if len(keys) < 2:
        return None
    return keys[0], keys[1], keys[2] if len(keys) > 2 else None


def take_pairs(src, need, seed):
    ds = open_source(src)
    try:
        ds = ds.shuffle(seed=seed, buffer_size=20000)
    except Exception:
        pass
    out, cols = [], None
    for row in ds:
        if cols is None:
            cols = columns(row)
            if cols is None:
                raise ValueError(f"{src['key']} has no text columns: {list(row)}")
            log(f"{src['key']} columns query={cols[0]} positive={cols[1]} negative={cols[2]}")
        q, p = (row.get(cols[0]) or "").strip(), (row.get(cols[1]) or "").strip()
        n = (row.get(cols[2]) or "").strip() if cols[2] else ""
        if len(q) < 3 or len(p) < 10:
            continue
        out.append((q, p, n))
        if len(out) >= need:
            break
    return out


def build_pairs(cfg):
    run = cfg["run"]
    total = run["queries"] + run["dev_queries"]
    shares = sum(s["share"] for s in cfg["sources"])
    rng = random.Random(0)
    seen_q, docs, rows = set(), {}, []

    def doc_id(text):
        if text not in docs:
            docs[text] = len(docs)
        return docs[text]

    for i, src in enumerate(cfg["sources"]):
        need = math.ceil(total * src["share"] / shares * 1.25)
        try:
            pairs = take_pairs(src, need, seed=i)
        except Exception as ex:
            log(f"skip source {src['key']}: {type(ex).__name__}: {ex}")
            continue
        kept = 0
        for q, p, n in pairs:
            if q in seen_q:
                continue
            seen_q.add(q)
            rows.append({"query": q, "pos": doc_id(p), "neg": doc_id(n) if n and n != p else None, "src": src["key"]})
            kept += 1
        log(f"source {src['key']}: {kept} queries")
    if len(rows) < total:
        log(f"only {len(rows)} queries available, wanted {total}")
    rng.shuffle(rows)
    rows = rows[:total]
    used = sorted({r["pos"] for r in rows} | {r["neg"] for r in rows if r["neg"] is not None})
    remap = {old: new for new, old in enumerate(used)}
    texts = {v: k for k, v in docs.items()}
    corpus = [{"id": remap[o], "text": texts[o]} for o in used]
    dev_n = min(run["dev_queries"], max(1, len(rows) // 10))
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "qid": i,
                "query": r["query"],
                "pos": remap[r["pos"]],
                "neg": remap[r["neg"]] if r["neg"] is not None else None,
                "src": r["src"],
                "split": "dev" if i < dev_n else "train",
            }
        )
    return out, corpus


def build_mntp(cfg):
    run = cfg["run"]
    need = run["mntp_docs"]
    for src in cfg["mntp_sources"]:
        try:
            ds = open_source(src)
            field = src.get("field", "text")
            out = []
            for row in ds:
                t = (row.get(field) or "").strip()
                if len(t) < 200:
                    continue
                out.append({"text": t[:6000]})
                if len(out) >= need:
                    break
            if out:
                log(f"mntp text from {src.get('id', src.get('files'))}: {len(out)} docs")
                return out
        except Exception as ex:
            log(f"mntp source {src.get('id')} failed: {type(ex).__name__}: {ex}")
    raise SystemExit("no MNTP text source worked")


def main():
    cfg = load_cfg()
    hf_login()
    data = cfg["paths"]["data"]
    done = data / "done.json"
    if done.exists():
        log("data already built", read_json(done))
        return
    queries, corpus = build_pairs(cfg)
    write_jsonl(data / "queries.jsonl", queries)
    write_jsonl(data / "corpus.jsonl", corpus)
    mntp = build_mntp(cfg)
    write_jsonl(data / "mntp.jsonl", mntp)
    by_src = {}
    for q in queries:
        by_src[q["src"]] = by_src.get(q["src"], 0) + 1
    info = {
        "queries": len(queries),
        "train": sum(1 for q in queries if q["split"] == "train"),
        "dev": sum(1 for q in queries if q["split"] == "dev"),
        "corpus": len(corpus),
        "mntp_docs": len(mntp),
        "by_source": by_src,
    }
    write_json(done, info)
    update_status(cfg, "data", info)
    log("data done", info)


if __name__ == "__main__":
    main()
