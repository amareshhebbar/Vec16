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


def results(cfg, s, bres):
    full, bits = s["full_dim"], s["bits"]
    t = s["tuned"]
    rows = [
        "## Results",
        "",
        f"NanoBEIR, {len(cfg['run']['nanobeir'])} datasets, mean nDCG@10. ± is the standard deviation over seeds.",
        "",
        "| Setting | Bytes per document | nDCG@10 |",
        "|---|---|---|",
        f"| Starting LLM, causal, mean pooled, float32 {full} | {nbytes(full, 'float')} | {pct(s['base_float'])} |",
        f"| Vec16 float32 {full} | {nbytes(full, 'float')} | {ms(t['float'])} |",
        f"| Vec16 float32 128 | {nbytes(128, 'float')} | {ms(t['float128'])} |",
        f"| Vec16 binary {bits} bits | {nbytes(bits, 'binary')} | **{ms(t['binary'])}** |",
        f"| Vec16 float query, binary {bits} bit documents | {nbytes(bits, 'binary')} | {ms(t['asym'])} |",
        "",
        f"Verdict: **{'GOOD' if s['good'] else 'BAD'}**. {'; '.join(s['reasons'])}.",
        "",
    ]
    if bres:
        rows += ["### Against public embedders", "", "| Model | Dims | float32 bytes | float32 nDCG@10 | binary bytes | binary nDCG@10 |", "|---|---|---|---|---|---|"]
        for k, r in bres.items():
            d = r["dim"]
            rows.append(f"| {r['label']} | {d} | {nbytes(d, 'float')} | {pct(r['mean']['float'].get(str(d)))} | {nbytes(d, 'binary')} | {pct(r['mean']['binary'].get(str(d)))} |")
        rows.append(f"| **Vec16 {cfg['models'][s['model']]['tail'].replace('_', ' ')}** | {full} | {nbytes(full, 'float')} | {pct(t['float']['mean'] if t['float'] else None)} | **{nbytes(bits, 'binary')}** ({bits} bits) | **{pct(t['binary']['mean'] if t['binary'] else None)}** |")
        rows.append("")
    main = s["main"]["mean"]
    dims = sorted((int(d) for d in main["float"]), reverse=True)
    rows += ["### Quality by vector size", "", "| Dims | float32 bytes | float32 | binary bytes | binary | float query, binary docs |", "|---|---|---|---|---|---|"]
    for d in dims:
        rows.append(f"| {d} | {nbytes(d, 'float')} | {pct(main['float'].get(str(d)))} | {nbytes(d, 'binary') if str(d) in main['binary'] else ''} | {pct(main['binary'].get(str(d))) if str(d) in main['binary'] else ''} | {pct(main['asym'].get(str(d))) if str(d) in main['asym'] else ''} |")
    rows.append("")
    if s["teachers"]:
        rows += ["### Which teacher", "", "| Branch | Teacher | float32 | binary " + str(bits) + " | dev |", "|---|---|---|---|---|"]
        names = {"dense": cfg["run"]["teacher_dense"], "colbert": cfg["teachers"]["colbert"], "fused": "both, averaged"}
        for v, r in s["teachers"].items():
            mark = " (main)" if v == s["headline"]["variant"] else ""
            rows.append(f"| `{v}`{mark} | {names.get(v, v)} | {pct(r['float'])} | {pct(r['binary'])} | {pct(r['dev'])} |")
        rows.append("")
    if s["ablations"]:
        desc = {"full method": "everything", "no_mntp": "skip masked next token training", "no_qat": "no binary aware loss", "no_listwise": "contrastive loss only, no teacher", "causal": "keep causal attention"}
        rows += ["### Ablation", "", "| Branch | Change | float32 | binary " + str(bits) + " |", "|---|---|---|---|"]
        for v, r in s["ablations"].items():
            rows.append(f"| `{v}` | {desc.get(v, v)} | {pct(r['float'])} | {pct(r['binary'])} |")
        rows.append("")
    return rows


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
    for f, alt in (("mrl.png", "Quality by vector size"), ("teachers.png", "Teacher comparison"), ("ablation.png", "Ablation"), ("loss.png", "Training curve")):
        if f in present:
            lines += [f"![{alt}](assets/{f})", ""]
    if leaderboard:
        lines += ["### Every Vec16 model", ""] + leaderboard + [""]
    lines += usage(cfg, s)
    lines += build_section(cfg, s)
    lines += config_section(cfg, s)
    lines += [
        "## Prior work and what is new",
        "",
        "* LLM2Vec (BehnamGhader et al., 2024) showed decoder LLMs become strong embedders after enabling bidirectional attention and masked next token training. Vec16 uses the same first two steps on much smaller models.",
        "* Matryoshka Representation Learning (Kusupati et al., 2022) and binary embedding quantization (Shakir, Aarsen and Lee, 2024) shrink vectors after training or with light tuning.",
        "* ColBERTv2 and later work distill rankings from stronger teachers into retrievers.",
        "",
        "New here: all of these in one sub 2B decoder, binary aware training at every Matryoshka size, a direct comparison of a single vector teacher, a late interaction teacher and both fused, and one curve of quality against bytes stored for every student and five larger public embedders.",
        "",
        "## Limitations",
        "",
        "* NanoBEIR has 50 queries per dataset, so small gaps are noise. The seed spread is reported for that reason.",
        "* Training used the train splits of MS MARCO, Natural Questions, HotpotQA and FEVER, so those four NanoBEIR sets are in domain.",
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
        "Every source keeps its own license. Check them before commercial use.",
        "",
        f"Code: https://github.com/{cfg['project']['github']}",
        "",
    ]
    return "\n".join(lines)
