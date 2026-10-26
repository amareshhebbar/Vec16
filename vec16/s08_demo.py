import json
import re
import shutil

import numpy as np

from vec16 import hub
from vec16.common import REPO, eval_dtype, hf_login, load_cfg, log, read_json, repo_name, run_dir, update_status, write_json
from vec16.students import load_trained
from vec16.summary import best_seed, headline

BLOCK = 100000


def pick_model(cfg):
    run = cfg["run"]
    order = [run.get("demo_model")] + [m for m in run["models"] if m != run.get("demo_model")]
    for m in order:
        if not m or m not in run["models"]:
            continue
        final = cfg["paths"]["results"] / m / "final.json"
        onnx = cfg["paths"]["runs"] / m / "onnx" / "model_quantized.onnx"
        if final.exists() and onnx.exists():
            return m
    return None


def lead(text, lo, hi):
    words = []
    for para in re.split(r"\n+", text):
        w = para.split()
        if not w:
            continue
        words += w
        if len(words) >= lo:
            break
    if len(words) < lo:
        return None
    return " ".join(words[:hi])


def articles(cfg):
    from vec16.s01_data import open_source

    w = cfg["wikipedia"]
    src = {"id": w["id"], "config": w.get("config"), "split": w.get("split", "train"), "files": w.get("files")}
    for row in open_source(src):
        t = lead(row.get("text") or "", w["min_words"], w["max_words"])
        if t:
            yield (row.get("title") or "").strip(), t


def build(cfg, model, root, limit):
    w = cfg["wikipedia"]
    bits = cfg["verdict_bits"]
    shard = w["shard_size"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    blocks = root / "blocks"
    blocks.mkdir(exist_ok=True)
    state_p = root / "state.json"
    state = read_json(state_p) if state_p.exists() else {"done": 0}
    buf, n = [], 0

    def flush(items, start):
        texts = [cfg["prompts"]["document"] + t for _, t in items]
        emb = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=False)
        codes = np.packbits(emb[:, :bits] > 0, axis=1)
        codes.tofile(blocks / f"{start:09d}.bin")
        for s in range(0, len(items), shard):
            sid = (start + s) // shard
            rows = [[title, t[: w["snippet_chars"]]] for title, t in items[s : s + shard]]
            (root / "shards" / f"{sid:05d}.json").write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
        write_json(state_p, {"done": start + len(items)})
        log(f"wikipedia index {start + len(items):,} passages")

    for title, t in articles(cfg):
        if limit and n >= limit:
            break
        n += 1
        if n <= state["done"]:
            continue
        buf.append((title, t))
        if len(buf) == BLOCK:
            flush(buf, n - len(buf))
            buf = []
    if buf:
        flush(buf, n - len(buf))
    with open(root / "codes.bin", "wb") as f:
        for b in sorted(blocks.glob("*.bin")):
            f.write(b.read_bytes())
    total = (root / "codes.bin").stat().st_size // (bits // 8)
    shutil.rmtree(blocks)
    return total


def page(cfg, model_repo, index_repo, meta):
    html = (REPO / "serving" / "browser" / "wikipedia.html").read_text()
    return html.replace("__MODEL_REPO__", model_repo).replace("__INDEX_REPO__", index_repo).replace("__COUNT__", f"{meta['count']:,}").replace("__MB__", f"{meta['bytes'] / 1024**2:.0f}")


def space_readme(cfg, meta, model_repo, index_repo):
    return "\n".join(
        [
            "---",
            "title: Vec16 Wikipedia Search",
            "colorFrom: blue",
            "colorTo: indigo",
            "sdk: static",
            "pinned: true",
            "license: mit",
            "---",
            "",
            f"Search {meta['count']:,} English Wikipedia passages inside your browser. Each passage is {meta['bits']} bits, the whole index is {meta['bytes'] / 1024**2:.0f} MB, and the search runs as Hamming distance in this tab with no server.",
            "",
            f"Model: [{model_repo}](https://huggingface.co/{model_repo}). Index: [{index_repo}](https://huggingface.co/datasets/{index_repo}). Code: https://github.com/{cfg['project']['github']}",
            "",
        ]
    )


def index_card(cfg, meta, model_repo):
    return "\n".join(
        [
            "---",
            "license: cc-by-sa-3.0",
            "language:",
            "- en",
            "tags:",
            "- vec16",
            "- binary-embeddings",
            "---",
            "",
            "# Vec16 Wikipedia index",
            "",
            f"{meta['count']:,} lead passages of English Wikipedia articles from `{cfg['wikipedia']['id']}` `{cfg['wikipedia'].get('config')}`, embedded with [{model_repo}](https://huggingface.co/{model_repo}) and stored as {meta['bits']} bit codes, {meta['bits'] // 8} bytes each.",
            "",
            "| File | Content |",
            "|---|---|",
            f"| `codes.bin` | {meta['count']:,} × {meta['bits'] // 8} bytes, row i is passage i, bits packed most significant first like `numpy.packbits` |",
            f"| `shards/NNNNN.json` | `[title, snippet]` for passages `NNNNN × {meta['shard_size']}` onward |",
            "| `meta.json` | counts and settings |",
            "",
            "```python",
            "import numpy as np",
            "from huggingface_hub import hf_hub_download",
            "",
            f'codes = np.fromfile(hf_hub_download("{cfg["project"]["hf_user"]}/{meta["index_repo"]}", "codes.bin", repo_type="dataset"), dtype=np.uint8).reshape(-1, {meta["bits"] // 8})',
            "```",
            "",
            "Text is from Wikipedia under CC BY SA 3.0.",
            "",
        ]
    )


def main():
    cfg = load_cfg()
    hf_login()
    run = cfg["run"]
    if not run.get("demo"):
        log("demo off")
        return
    m = pick_model(cfg)
    if m is None:
        log("no finished model with an ONNX build, skipping demo")
        return
    user = cfg["project"]["hf_user"]
    suffix = run["repo_suffix"]
    index_name, space_name = f"Vec16_Wikipedia{suffix}", f"Vec16_Search{suffix}"
    root = cfg["paths"]["publish"] / index_name
    done = cfg["paths"]["results"] / "demo.json"
    if done.exists():
        log("demo already built", read_json(done))
        return
    hv = headline(cfg, m)["variant"]
    model = load_trained(run_dir(cfg, m, hv, best_seed(cfg, m, hv)) / "model", eval_dtype(cfg["models"][m]))
    model.max_seq_length = 256
    log(f"building Wikipedia index with {m}")
    count = build(cfg, model, root, run.get("demo_passages"))
    model_repo = f"{user}/{repo_name(cfg, m)}"
    meta = {"count": int(count), "bits": cfg["verdict_bits"], "bytes": int(count * cfg["verdict_bits"] // 8), "shard_size": cfg["wikipedia"]["shard_size"], "model": model_repo, "index_repo": index_name, "query_prompt": cfg["prompts"]["query"], "document_prompt": cfg["prompts"]["document"]}
    write_json(root / "meta.json", meta)
    (root / "README.md").write_text(index_card(cfg, meta, model_repo))
    space = cfg["paths"]["publish"] / space_name
    shutil.rmtree(space, ignore_errors=True)
    space.mkdir(parents=True)
    (space / "index.html").write_text(page(cfg, model_repo, f"{user}/{index_name}", meta))
    (space / "README.md").write_text(space_readme(cfg, meta, model_repo, f"{user}/{index_name}"))
    (root / "state.json").unlink(missing_ok=True)
    info = dict(meta, space=f"{user}/{space_name}")
    a = hub.api()
    if a is not None:
        private = bool(run["private"])
        hub.push_folder(a, f"{user}/{index_name}", root, kind="dataset", private=private, message="Wikipedia binary index")
        hub.retry(a.create_repo, f"{user}/{space_name}", repo_type="space", space_sdk="static", private=private, exist_ok=True)
        hub.retry(a.upload_folder, folder_path=str(space), repo_id=f"{user}/{space_name}", repo_type="space", commit_message="Vec16 Wikipedia search")
        hub.add_to_collection(cfg, a, f"{user}/{index_name}", "dataset", f"{count:,} Wikipedia passages, {meta['bytes'] / 1024**2:.0f} MB of {meta['bits']} bit codes")
        hub.add_to_collection(cfg, a, f"{user}/{space_name}", "space", "Search Wikipedia in your browser, no server")
        log(f"demo live at https://huggingface.co/spaces/{user}/{space_name}")
    else:
        log(f"demo staged at {space}")
    write_json(done, info)
    update_status(cfg, "demo", info)


if __name__ == "__main__":
    main()
