# How Vec16 ships

Every Vec16 release is a normal Sentence Transformers repo plus one small Python file. This page explains what is inside a repo, why each piece is there, and how to run the model everywhere it runs.

## Where each release runs

| Runtime | Works | How |
|---|---|---|
| Sentence Transformers | Yes | `trust_remote_code=True` |
| HF Transformers | Yes | `AutoModel` with `trust_remote_code=True`, then mean pooling |
| vLLM | Yes, no plugin | `config.json` sets `is_causal: false`, which recent vLLM (checked on 0.32) reads natively |
| ONNX Runtime | Yes | `onnx/model.onnx` and the int8 `onnx/model_quantized.onnx`, pooling built in |
| Browser | Yes | onnxruntime web with the int8 file, `serving/browser/index.html` |
| FAISS, Qdrant, Milvus binary indexes | Yes | pack the sign bits with `quantize_embeddings(..., precision="ubinary")` |
| TEI, llama.cpp, Ollama | No | they assume causal attention for decoder architectures |

## What is inside a model repo

```
config.json                     architecture, is_causal false, auto_map, matryoshka_dimensions
bidirectional.py                the remote code, about 80 lines
model.safetensors               weights
tokenizer.json and friends      tokenizer, right padding, pad token set
modules.json                    Transformer, mean Pooling, Normalize
1_Pooling/config.json           mean pooling
config_sentence_transformers.json   query and document prompts
onnx/model.onnx                 float32 export, outputs sentence_embedding and token_embeddings
onnx/model_quantized.onnx       int8 export for CPU and the browser
onnx/export.json                parity of both exports against PyTorch
assets/                         charts used by the card
results/                        raw evaluation numbers behind every value in the card
runs/                           TensorBoard logs of the main run
training_run.json               exact settings of this run
README.md                       model card
```

### `bidirectional.py`

Decoder LLMs only let each token look at earlier tokens. An embedder should read the whole text, so Vec16 replaces the causal mask with a full mask that only hides padding.

The file defines `Vec16LlamaModel`, `Vec16Qwen2Model`, `Vec16Qwen3Model` and `Vec16Gemma3TextModel`. Each one is the stock Transformers class with two changes:

1. On load, `config.is_causal` and every attention module's `is_causal` flag are set to `False`.
2. In `forward`, the 2D padding mask is turned into a 4D additive mask of shape batch × 1 × length × length where every real token is visible. Transformers passes a ready 4D mask straight to the attention kernel, so SDPA and eager attention both run fully bidirectional.

No weights change shape and no new parameters are added. Loading the same weights without remote code gives a causal model, which is why `trust_remote_code=True` is required.

### `config.json`

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "auto_map": {"AutoModel": "bidirectional.Vec16Qwen3Model"},
  "is_causal": false,
  "matryoshka_dimensions": [1024, 768, 512, 256, 128, 64, 32]
}
```

* `auto_map` sends `AutoModel` with remote code to the class above.
* `architectures` names the class vLLM should build. Llama based students use `LlamaModel`, Qwen2.5 uses `Qwen2Model`, Gemma 3 uses `Gemma3TextModel` and Qwen3 uses `Qwen3ForCausalLM`, all registered as pooling models in vLLM.
* `is_causal: false` is read by vLLM's Llama, Qwen2, Qwen3 and Gemma3 layers, which then use encoder attention. Recent Transformers versions also read it.
* `matryoshka_dimensions` lets vLLM accept the `dimensions` field of the OpenAI embeddings API.

### Prompts

Queries start with `query: ` and documents with `passage: `. They are stored in `config_sentence_transformers.json`, so `encode_query` and `encode_document` add them. Every other runtime needs them in the text.

### Branches

`main` holds the best run. Every other run of the same student is a branch, so any result in the card can be loaded and checked:

```python
SentenceTransformer("AmareshHebbar/Vec16_Qwen3_0.6B", revision="colbert", trust_remote_code=True)
```

| Branch | What it is |
|---|---|
| `dense`, `colbert`, `fused` | one teacher setup each, seed 0 |
| `<variant>_s1`, `<variant>_s2` | extra seeds of the winning setup |
| `no_mntp` | no masked next token training |
| `no_qat` | no binary aware loss |
| `no_listwise` | contrastive loss only, no teacher |
| `causal` | causal attention kept, plain Transformers class |

## Running it

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer("AmareshHebbar/Vec16_Qwen3_0.6B", trust_remote_code=True, truncate_dim=128)
q = model.encode_query(["how do vaccines work"])
d = model.encode_document(["Vaccines show the immune system a harmless piece of a pathogen."])
print(model.similarity(q, d))
codes = quantize_embeddings(d, precision="ubinary")
```

`truncate_dim` picks the Matryoshka size. `ubinary` packs each dimension into one bit, so 128 dims become 16 `uint8` values.

### FAISS binary index

```bash
pip install faiss-cpu
python serving/python/search_binary.py "how does a vaccine work"
```

### vLLM

```bash
bash serving/vllm/serve.sh AmareshHebbar/Vec16_Qwen3_0.6B
python serving/vllm/client.py --model AmareshHebbar/Vec16_Qwen3_0.6B --dims 128
python serving/vllm/parity.py --model AmareshHebbar/Vec16_Qwen3_0.6B
```

`parity.py` embeds the same texts with Sentence Transformers and vLLM and fails if any cosine is below 0.99, so you can confirm the bidirectional path before serving traffic.

### ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

repo = "AmareshHebbar/Vec16_SmolLM2_135M"
tok = AutoTokenizer.from_pretrained(repo)
sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model_quantized.onnx"))
b = tok(["query: how do vaccines work"], padding=True, return_tensors="np")
emb = sess.run(["sentence_embedding"], {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]})[0]
codes = np.packbits(emb[:, :128] > 0, axis=1)
```

The graph already contains the bidirectional mask, mean pooling and normalization. `onnx/export.json` records the minimum cosine between ONNX and PyTorch on sample text, for both the float32 and the int8 file.

### Browser

Open `serving/browser/index.html` from any static host (or `python -m http.server` in that folder). It downloads the tokenizer and the int8 ONNX file from the public repo, embeds your documents in the tab, packs them into bits and ranks them by Hamming distance. Nothing is sent to a server.

## How a release is produced

1. `s06_models` finishes a student and uploads `main` and every branch as private.
2. The evaluation results, charts, ONNX files and the full model card are uploaded to `main`. Each branch gets a short card with its own numbers.
3. The verdict decides visibility: GOOD becomes public, BAD stays private.
4. The repo is added to the "Vec16 Embeddings" collection with a note carrying the headline numbers.
5. `s07_finalize` adds the cross model leaderboard to every card.

To rebuild one student from scratch, delete `<work>/runs/<key>` and `<work>/results/<key>` and run `VEC16_FROM=6 VEC16_MODELS=<key> bash run_all.sh`.
