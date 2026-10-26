def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}"


def ms(d):
    if not d:
        return "n/a"
    if d["n"] > 1:
        return f"{d['mean'] * 100:.1f} ± {d['std'] * 100:.1f}"
    return f"{d['mean'] * 100:.1f}"


def nbytes(dim, kind):
    b = dim * 4 if kind == "float" else dim // 8
    return f"{b:,}"


def header(cfg, mkey, s):
    mcfg = cfg["models"][mkey]
    user = cfg["project"]["hf_user"]
    from vec16.common import dataset_name

    return [
        "---",
        f"license: {mcfg['license']}",
        f"base_model: {mcfg['id']}",
        "library_name: sentence-transformers",
        "pipeline_tag: sentence-similarity",
        "language:",
        "- en",
        "tags:",
        "- sentence-transformers",
        "- embeddings",
        "- retrieval",
        "- matryoshka",
        "- binary-quantization",
        "- bidirectional",
        "- distillation",
        "- vec16",
        "datasets:",
        f"- {user}/{dataset_name(cfg)}",
        "---",
        "",
    ]


def usage(cfg, s):
    repo = f"{cfg['project']['hf_user']}/{s['repo']}"
    bits = s["bits"]
    gh = cfg["project"]["github"]
    return [
        "## Usage",
        "",
        "### Sentence Transformers",
        "",
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "from sentence_transformers.quantization import quantize_embeddings",
        "",
        f'model = SentenceTransformer("{repo}", trust_remote_code=True, truncate_dim={bits})',
        "",
        'queries = model.encode_query(["how do vaccines train the immune system"])',
        'docs = model.encode_document(["Vaccines show the immune system a harmless piece of a pathogen so it can react faster later."])',
        "print(model.similarity(queries, docs))",
        "",
        'packed = quantize_embeddings(docs, precision="ubinary")',
        "print(packed.shape, packed.dtype)",
        "```",
        "",
        f"`truncate_dim={bits}` with `ubinary` gives {bits // 8} bytes per document. Drop `truncate_dim` for the full {s['full_dim']} dimensions, or pick any of {', '.join(str(d) for d in sorted(int(x) for x in s['main']['mean']['float']))}.",
        "",
        "`trust_remote_code=True` loads `bidirectional.py` from this repo. It is a few lines that switch the attention mask from causal to full, nothing else.",
        "",
        "### Binary index with FAISS",
        "",
        "```python",
        "import faiss",
        "",
        f"index = faiss.IndexBinaryFlat({bits})",
        'index.add(quantize_embeddings(model.encode_document(corpus), precision="ubinary"))',
        'scores, ids = index.search(quantize_embeddings(model.encode_query(queries), precision="ubinary"), 10)',
        "```",
        "",
        "Qdrant and Milvus binary vectors work the same way.",
        "",
        "### Transformers",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModel, AutoTokenizer",
        "",
        f'tok = AutoTokenizer.from_pretrained("{repo}")',
        f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)',
        f'batch = tok(["{cfg["prompts"]["query"]}how do vaccines work"], padding=True, return_tensors="pt")',
        "hidden = model(**batch).last_hidden_state",
        'mask = batch["attention_mask"].unsqueeze(-1)',
        "emb = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)",
        "```",
        "",
        f'Queries start with `"{cfg["prompts"]["query"]}"` and documents with `"{cfg["prompts"]["document"]}"`. Sentence Transformers adds them for you.',
        "",
        "### vLLM",
        "",
        "```bash",
        f"vllm serve {repo} --runner pooling",
        "```",
        "",
        "```python",
        "from openai import OpenAI",
        "",
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")',
        f'out = client.embeddings.create(model="{repo}", input=["{cfg["prompts"]["query"]}how do vaccines work"], dimensions={bits})',
        "```",
        "",
        "`config.json` sets `is_causal: false`, which vLLM reads to run every layer with bidirectional attention. No plugin is needed. vLLM does not add the prompts, so send them in the text.",
        "",
        "### ONNX and the browser",
        "",
        "```python",
        "import numpy as np",
        "import onnxruntime as ort",
        "from huggingface_hub import hf_hub_download",
        "from transformers import AutoTokenizer",
        "",
        f'tok = AutoTokenizer.from_pretrained("{repo}")',
        f'sess = ort.InferenceSession(hf_hub_download("{repo}", "onnx/model.onnx"))',
        f'b = tok(["{cfg["prompts"]["query"]}how do vaccines work"], padding=True, return_tensors="np")',
        'emb = sess.run(["sentence_embedding"], {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]})[0]',
        f"bits = np.packbits(emb[:, :{bits}] > 0, axis=1)",
        "```",
        "",
        f"`onnx/model_quantized.onnx` is the int8 build for the browser. A static page that runs it with onnxruntime web is in [serving/browser](https://github.com/{gh}/tree/main/serving/browser).",
        "",
    ]


def build_section(cfg, s):
    tc = cfg["teachers"]
    head = s["headline"]
    meta = s["main"].get("meta", {})
    return [
        "## How it was built",
        "",
        "1. **Causal to bidirectional.** The attention mask of the starting LLM is replaced with a full mask, so every token sees the whole text.",
        f"2. **Masked next token training.** {cfg['run']['mntp_tokens']:,} tokens of web text, {int(cfg['train']['mntp_mask'] * 100)}% of tokens masked, each masked token predicted from the hidden state just before it. This teaches the model to use its new right side context.",
        f"3. **Two teachers.** `{cfg['run']['teacher_dense']}` ranks the top {meta.get('k', cfg['run']['train_k'])} of {cfg['run']['teacher_k']} candidates for every training query, and `{tc['colbert']}` scores the same candidates token by token.",
        "4. **Listwise distillation plus contrastive loss.** The student matches the teacher's softmax over each candidate list and also learns to pick the positive among every document in the batch. Candidates the teacher ranks above the positive are not used as negatives.",
        "5. **Matryoshka.** Both losses run at every vector size, so the first dimensions carry the most meaning.",
        "6. **Binary quantization aware training.** The same losses also run on sign bits with a straight through estimator, so the model learns to rank well when each number is one bit.",
        "",
        f"Three students were trained per model: dense teacher only, ColBERT teacher only, and both teachers fused. The best on a held out dev set (`{head['variant']}`) became `main` and was repeated over {len(cfg['run']['seeds'])} seeds. Every variant is kept as a branch of this repo.",
        "",
    ]


SPLIT_KEY = {"all": "mean", "ood": "mean_ood", "ind": "mean_ind"}
SPLIT_NAME = {"all": "all sets", "ood": "out of domain", "ind": "in domain"}


def bget(res, kind, dim, split):
    if not res:
        return None
    b = res.get(SPLIT_KEY[split]) or {}
    return b.get(kind, {}).get(str(dim))


def results(cfg, s, bres):
    full, bits = s["full_dim"], s["bits"]
    bys = s["by_split"]
    vs = s["split"]
    ood_names = s["main"].get("splits", {}).get("ood", [])
    ind_names = s["main"].get("splits", {}).get("in_domain", [])
    cols = [sp for sp in ("ood", "ind", "all") if bys[sp]["float"]]
    head = " | ".join(f"{SPLIT_NAME[c]} ({len(ood_names) if c == 'ood' else len(ind_names) if c == 'ind' else len(ood_names) + len(ind_names)})" for c in cols)
    sep = "|".join(["---"] * (len(cols) + 2))

    def row(label, nb, key):
        return f"| {label} | {nb} | " + " | ".join((("**" + ms(bys[c][key]) + "**") if key == "binary" else ms(bys[c][key])) for c in cols) + " |"

    base_cells = " | ".join(pct(s["base_by_split"][c]) for c in cols)
    rows = [
        "## Results",
        "",
        "NanoBEIR mean nDCG@10, ± is the standard deviation over seeds.",
        f"Out of domain: {', '.join(ood_names) or 'none'}. In domain: {', '.join(ind_names) or 'none'}. The train splits of the in domain sets are part of the training data, so the out of domain column is the honest number. The verdict uses {SPLIT_NAME[vs]}.",
        "",
        f"| Setting | Bytes per document in RAM | {head} |",
        f"|{sep}|",
        f"| Starting LLM, causal, mean pooled, float32 {full} | {nbytes(full, 'float')} | {base_cells} |",
        row(f"Vec16 float32 {full}", nbytes(full, "float"), "float"),
        row("Vec16 float32 128", nbytes(128, "float"), "float128"),
        row(f"Vec16 binary {bits} bits", nbytes(bits, "binary"), "binary"),
        row(f"Vec16 binary {bits} bits, top {cfg['run'].get('rescore_top', 100)} rescored with the float query", nbytes(bits, "binary"), "rescore"),
        row(f"Vec16 binary {bits} bits, top {cfg['run'].get('rescore_top', 100)} rescored with float32 vectors kept on disk", f"{nbytes(bits, 'binary')} (+{nbytes(full, 'float')} on disk)", "rescore_full"),
        row(f"Vec16 float query against binary {bits} bit documents", nbytes(bits, "binary"), "asym"),
        "",
        f"Verdict: **{'GOOD' if s['good'] else 'BAD'}**. {'; '.join(s['reasons'])}.",
        "",
    ]
    t = s["tuned"]
    vb = t["binary"]["mean"] if t["binary"] else None
    if s.get("binarized"):
        rows += [
            f"### Against simply binarizing a strong embedder at {bits} bits",
            "",
            f"Matryoshka embedders can be cut to {bits} dimensions and binarized without any training. This is the shortcut Vec16 has to beat. ({SPLIT_NAME[vs]})",
            "",
            f"| Model | Params | {bits} bits nDCG@10 | its own float32 nDCG@10 |",
            "|---|---|---|---|",
        ]
        for b in s["binarized"]:
            rows.append(f"| {b['label']} binarized | | {pct(b['binary'])} | {pct(b['float'])} |")
        rows.append(f"| **Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}** | {s['params'] / 1e6:.0f}M | **{pct(vb)}** | {pct(t['float']['mean'] if t['float'] else None)} |")
        rows.append("")
    ts = s.get("teacher_scores") or {}
    if ts:
        rows += [f"### Against the teachers ({SPLIT_NAME[vs]})", "", "| Teacher | Bytes per document | nDCG@10 | Vec16 at " + str(bits // 8) + " bytes keeps |", "|---|---|---|---|"]
        d = ts.get("teacher_dense")
        if d:
            df = bget(d, "float", d["dim"], vs) or bget(d, "float", d["dim"], "all")
            rows.append(f"| {d['id']} float32 {d['dim']} | {nbytes(d['dim'], 'float')} | {pct(df)} | {pct(vb / df) if vb and df else 'n/a'}% |")
            db = bget(d, "binary", bits, vs) or bget(d, "binary", bits, "all")
            if db is not None:
                rows.append(f"| {d['id']} binary {bits} bits | {nbytes(bits, 'binary')} | {pct(db)} | {pct(vb / db) if vb and db else 'n/a'}% |")
        c = ts.get("teacher_colbert")
        if c:
            cl = c.get("late_ood") if vs == "ood" and c.get("late_ood") is not None else c.get("late")
            rows.append(f"| {c['id']} late interaction | about {c['bytes_per_doc']:,.0f} | {pct(cl)} | {pct(vb / cl) if vb and cl else 'n/a'}% |")
        rows.append("")
    if bres:
        rows += [f"### Against public embedders ({SPLIT_NAME[vs]})", "", "| Model | Dims | float32 bytes | float32 nDCG@10 | binary bytes | binary nDCG@10 |", "|---|---|---|---|---|---|"]
        for k, r in bres.items():
            dd = r["dim"]
            rows.append(f"| {r['label']} | {dd} | {nbytes(dd, 'float')} | {pct(bget(r, 'float', dd, vs))} | {nbytes(dd, 'binary')} | {pct(bget(r, 'binary', dd, vs))} |")
        rows.append(f"| **Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}** | {full} | {nbytes(full, 'float')} | {pct(t['float']['mean'] if t['float'] else None)} | **{nbytes(bits, 'binary')}** ({bits} bits) | **{pct(vb)}** |")
        rows.append("")
    main = s["main"].get(SPLIT_KEY[vs]) or s["main"]["mean"]
    dims = sorted((int(d) for d in main["float"]), reverse=True)
    rows += [f"### Quality by vector size ({SPLIT_NAME[vs]}, main branch)", "", "| Dims | float32 bytes | float32 | binary bytes | binary | binary, rescored | float query, binary docs |", "|---|---|---|---|---|---|---|"]
    for d in dims:
        has = str(d) in main.get("binary", {})
        rows.append(f"| {d} | {nbytes(d, 'float')} | {pct(main['float'].get(str(d)))} | {nbytes(d, 'binary') if has else ''} | {pct(main['binary'].get(str(d))) if has else ''} | {pct(main.get('rescore', {}).get(str(d))) if has else ''} | {pct(main['asym'].get(str(d))) if has else ''} |")
    rows.append("")
    if s["teachers"]:
        rows += [f"### Which teacher ({SPLIT_NAME[vs]})", "", "| Branch | Teacher | float32 | binary " + str(bits) + " | dev |", "|---|---|---|---|---|"]
        names = {"dense": cfg["run"]["teacher_dense"], "colbert": cfg["teachers"]["colbert"], "fused": "both, averaged"}
        for v, r in s["teachers"].items():
            mark = " (main)" if v == s["headline"]["variant"] else ""
            rows.append(f"| `{v}`{mark} | {names.get(v, v)} | {pct(r['float'])} | {pct(r['binary'])} | {pct(r['dev'])} |")
        rows.append("")
    if s["ablations"]:
        desc = {"full method": "everything", "no_mntp": "skip masked next token training", "no_qat": "no binary aware loss", "no_listwise": "contrastive loss only, no teacher", "causal": "keep causal attention"}
        rows += [f"### Ablation ({SPLIT_NAME[vs]})", "", "| Branch | Change | float32 | binary " + str(bits) + " |", "|---|---|---|---|"]
        for v, r in s["ablations"].items():
            rows.append(f"| `{v}` | {desc.get(v, v)} | {pct(r['float'])} | {pct(r['binary'])} |")
        rows.append("")
    if len(s.get("scaling") or []) >= 2:
        rows += ["### More training data", "", "Same student, same teacher, seed 0, only the number of training queries changes (out of domain when available).", "", "| Training queries | float32 | binary " + str(bits) + " |", "|---|---|---|"]
        for p in s["scaling"]:
            rows.append(f"| {p['queries']:,} | {pct(p['float'])} | {pct(p['binary'])} |")
        rows.append("")
    return rows


def speed_section(cfg, s):
    import json

    onnx = cfg["paths"]["runs"] / s["model"] / "onnx" / "export.json"
    bench = cfg["paths"]["results"] / "bench.json"
    lines = []
    if onnx.exists() or bench.exists():
        lines += ["## Speed and size", ""]
    if onnx.exists():
        e = json.loads(onnx.read_text())
        lines += ["| Build | Parity with PyTorch (min cosine) | CPU documents per second |", "|---|---|---|"]
        for k, name in (("fp32", "ONNX float32"), ("int8", "ONNX int8")):
            if k in e:
                c = e[k].get("cpu", {})
                lines.append(f"| {name} | {e[k]['parity_cosine']:.4f} | {c.get('docs_per_second', 'n/a')} |")
        c = e.get("fp32", {}).get("cpu", {})
        lines += ["", f"Measured on `{e.get('cpu_name', 'cpu')}` with {c.get('threads', '?')} threads, batches of {c.get('batch', 32)}, {c.get('words_per_doc', 100)} word passages.", ""]
    if bench.exists():
        b = json.loads(bench.read_text())
        lines += ["| Index | Documents | Size | Queries per second | p50 latency |", "|---|---|---|---|---|"]
        for r in b["rows"]:
            size = r["bytes"] / 1024**3
            sz = f"{size:,.2f} GB" if size >= 1 else f"{r['bytes'] / 1024**2:,.0f} MB"
            qps = f"{r['qps']:,.0f}" if r.get("qps") else "too big to load"
            lat = f"{r['latency_ms']} ms" if r.get("latency_ms") is not None else ""
            lines.append(f"| {r['label']} | {r['n']:,} | {sz} | {qps} | {lat} |")
        lines += ["", f"FAISS {b['faiss']} flat exact search on {b['threads']} CPU threads. {b['note']}", ""]
    return lines


def findings_section(s):
    if not s.get("findings"):
        return []
    return ["## What did not work", ""] + [f"* {f}" for f in s["findings"]] + [""]


def demo_section(cfg):
    import json

    p = cfg["paths"]["results"] / "demo.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return [
        "## Try it",
        "",
        f"[Search {d['count']:,} Wikipedia passages in your browser](https://huggingface.co/spaces/{d['space']}). The whole index is {d['bytes'] / 1024**2:.0f} MB of {d['bits']} bit codes and the search runs in the tab.",
        "",
    ]


def config_section(cfg, s):
    meta = s["main"].get("meta", {})
    return [
        "## Training configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Start | `{s['id']}` |",
        f"| Training queries | {meta.get('train_queries', 'n/a'):,} |" if isinstance(meta.get("train_queries"), int) else "| Training queries | n/a |",
        f"| Candidates per query | {meta.get('k', 'n/a')} |",
        f"| Batch size | {meta.get('batch_size', 'n/a')} queries |",
        f"| Learning rate | {meta.get('lr', 'n/a')} |",
        f"| Epochs | {meta.get('epochs', 'n/a')} |",
        f"| Max length | {meta.get('doc_len', 'n/a')} tokens |",
        f"| Matryoshka dims | {', '.join(str(d) for d in meta.get('dims', []))} |",
        f"| Binary aware dims | {', '.join(str(d) for d in meta.get('binary_dims', []))} |",
        f"| Precision | {meta.get('precision', 'n/a')} |",
        f"| GPU | {meta.get('gpu', 'n/a')} |",
        f"| GPU hours, every branch | {s['gpu_hours']} |",
        "",
    ]


def model_card(cfg, mkey, s, bres, present, leaderboard=None):
    user = cfg["project"]["hf_user"]
    from vec16.common import dataset_name

    tail = cfg["models"][mkey]["tail"].replace("_", " ")
    t = s["tuned"]
    keep = (t["binary"]["mean"] / t["float"]["mean"] * 100) if t["binary"] and t["float"] and t["float"]["mean"] else None
    lines = header(cfg, mkey, s)
    lines += [
        f"# Vec16 {tail}",
        "",
        f"`{s['id']}` turned into an embedding model. {s['bits']} bits per document, {s['bits'] // 8} bytes, keep {keep:.0f}% of its float32 quality." if keep else f"`{s['id']}` turned into an embedding model.",
        "",
    ]
    if "curve.png" in present:
        lines += ["![Quality against bytes per document](assets/curve.png)", ""]
    lines += results(cfg, s, bres)
    for f, alt in (("mrl.png", "Quality by vector size"), ("teachers.png", "Teacher comparison"), ("ablation.png", "Ablation"), ("data_scaling.png", "More training data"), ("loss.png", "Training curve")):
        if f in present:
            lines += [f"![{alt}](assets/{f})", ""]
    if leaderboard:
        lines += ["### Every Vec16 model", ""] + leaderboard + [""]
    lines += findings_section(s)
    lines += demo_section(cfg)
    lines += usage(cfg, s)
    lines += speed_section(cfg, s)
    lines += build_section(cfg, s)
    lines += config_section(cfg, s)
    lines += [
        "## Prior work and what is new",
        "",
        "* LLM2Vec (BehnamGhader et al., 2024) showed decoder LLMs become strong embedders after enabling bidirectional attention and masked next token training. Vec16 uses the same first two steps on much smaller models.",
        "* Matryoshka Representation Learning (Kusupati et al., 2022) and binary embedding quantization (Shakir, Aarsen and Lee, 2024) shrink vectors after training or with light tuning.",
        "* ColBERTv2 and later work distill rankings from stronger teachers into retrievers.",
        "",
        "New here: all of these in one sub 2B decoder, binary aware training at every Matryoshka size, a direct comparison of a single vector teacher, a late interaction teacher and both fused, a comparison against simply binarizing a strong Matryoshka embedder, and one curve of quality against bytes stored for every student, both teachers and the public embedders.",
        "",
        "## Limitations",
        "",
        "* NanoBEIR has 50 queries per dataset, so small gaps are noise. The seed spread is reported for that reason.",
        "* Training used the train splits of MS MARCO, Natural Questions, HotpotQA and FEVER, so those four NanoBEIR sets are in domain. They are reported separately and the verdict uses the out of domain sets.",
        "* English only. Documents longer than 512 tokens are cut.",
        "* Binary vectors below 128 bits produce many tied scores.",
        "",
        "## Links",
        "",
        f"* Code: https://github.com/{cfg['project']['github']}",
        f"* Distillation data: https://huggingface.co/datasets/{user}/{dataset_name(cfg)}",
        f"* Branches: {', '.join('`' + v + '`' for v in sorted({r.split('__')[0] for r in s['runs']}))}",
        "",
    ]
    return "\n".join(lines)


def branch_card(cfg, mkey, vname, seed, main_repo, res, spec):
    full = res["full_dim"] if res else None
    bits = cfg["verdict_bits"]
    lines = [
        "---",
        f"license: {cfg['models'][mkey]['license']}",
        f"base_model: {cfg['models'][mkey]['id']}",
        "library_name: sentence-transformers",
        "pipeline_tag: sentence-similarity",
        "---",
        "",
        f"# {main_repo} branch `{vname}`",
        "",
        f"Variant `{vname}`, seed {seed}. Teacher: {spec['teacher']}. Bidirectional: {spec['bidir']}. Masked next token training: {spec['mntp']}. Binary aware loss: {spec['qat']}. Listwise distillation: {spec['listwise']}.",
        "",
    ]
    if res:
        lines += [f"NanoBEIR nDCG@10: float32 {full} = {pct(res['mean']['float'].get(str(full)))}, binary {bits} bits = {pct(res['mean']['binary'].get(str(bits)))}.", ""]
    lines += [
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "",
        f'model = SentenceTransformer("{cfg["project"]["hf_user"]}/{main_repo}", revision="{vname}", trust_remote_code=True)',
        "```",
        "",
        "See the `main` branch card for the full results and method.",
        "",
    ]
    return "\n".join(lines)


def dataset_card(cfg, info):
    run = cfg["run"]
    c = info["counts"]
    by = info["data"].get("by_source", {})
    lines = [
        "---",
        "license: other",
        "language:",
        "- en",
        "task_categories:",
        "- sentence-similarity",
        "- text-retrieval",
        "tags:",
        "- distillation",
        "- embeddings",
        "- vec16",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.parquet",
        "  - split: dev",
        "    path: dev.parquet",
        "- config_name: corpus",
        "  data_files:",
        "  - split: train",
        "    path: corpus/corpus.parquet",
        "---",
        "",
        "# Vec16 distillation data",
        "",
        f"{c['train']:,} training and {c['dev']:,} dev queries. Each query has its top {info['dense']['k']} candidate passages from a {c['corpus']:,} passage corpus, scored by two teachers.",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `query` | query text |",
        "| `source` | dataset the query came from |",
        "| `positive_id` | corpus id of the labeled answer, always the first candidate |",
        "| `candidate_ids` | corpus ids, positive first, then the dense teacher's top results |",
        f"| `dense_scores` | cosine from `{run['teacher_dense']}` |",
        f"| `colbert_scores` | late interaction MaxSim from `{cfg['teachers']['colbert']}`, divided by query length |",
        "",
        "The `corpus` config maps ids to passage text.",
        "",
        "## Sources",
        "",
        "| Source | Queries |",
        "|---|---|",
    ]
    for k, v in by.items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        f"Dense teacher top 1 equals the labeled positive for {info['dense']['top1_is_positive'] * 100:.1f}% of queries. The two teachers agree on the top candidate for {info['colbert']['top1_agree_with_dense'] * 100:.1f}% of queries.",
        "",
        f"For {info.get('false_negative_rate', 0) * 100:.1f}% of training queries both teachers score another candidate clearly above the labeled positive. Those are usually correct answers that were never labeled, so training does not push them away. Examples are in [false_negatives.md](false_negatives.md).",
        "",
        "Every source keeps its own license. Check them before commercial use.",
        "",
        f"Code: https://github.com/{cfg['project']['github']}",
        "",
    ]
    return "\n".join(lines)
