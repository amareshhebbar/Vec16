# Vec16

Small decoder LLMs turned into embedding models that store each document in 16 bytes. One codebase runs the whole thing on a free Colab T4 or a RunPod A40 without changing a line.

### What we are building

Embedding models turn text into vectors so similar meanings land close together. Good embedders are big, and their vectors are big too: a 1024 dimension float32 vector is 4,096 bytes per document. A hundred million documents is 400 GB of RAM just for the index.

Vec16 takes small text generation models (135M to 1.7B), rewires them to read the whole sentence at once, distills them from two strong teachers, and trains them so the first 128 dimensions still work when every number is stored as a single bit. That is 128 bits, 16 bytes per document, 256 times smaller than float32 at 1024 dimensions.

### Why it matters

Small embedders are cheap to run but weak. Binary vectors are tiny but usually lose a lot of quality. This project stacks every known trick for making embedders small and cheap into one model family, measures exactly how much quality survives at each size, and ships every model in a form any developer can drop into Sentence Transformers, FAISS, vLLM, ONNX Runtime or a browser tab.

## How it works

| Step | What happens |
|---|---|
| 1. Causal to bidirectional | The attention mask of the LLM is switched from "see earlier tokens" to "see every token". `vec16/bidirectional.py` does this for Llama, Qwen2, Qwen3 and Gemma3 |
| 2. Masked next token training | Short adaptation on web text: random tokens are masked and predicted from the hidden state just before them, so the model learns to use right side context |
| 3. Dense teacher | `Qwen/Qwen3-Embedding-8B` finds the top 32 candidate passages for every training query and scores them |
| 4. Late interaction teacher | `lightonai/GTE-ModernColBERT-v1` rescores the same candidates token by token through PyLate |
| 5. Listwise distillation plus contrastive loss | The student copies the teacher's ranking over each candidate list and learns to pick the positive among every document in the batch. Candidates the teacher ranks above the labeled positive are never used as negatives |
| 6. Matryoshka | Both losses run at 1024, 768, 512, 256, 128, 64 and 32 dimensions, so vectors can be cut short |
| 7. Binary aware training | The same losses run again on sign bits with a straight through estimator, so ranking survives 1 bit per dimension |

Every student is trained three times: dense teacher only, ColBERT teacher only, and both teachers fused. The winner on a held out dev set becomes the release and is repeated over 3 seeds. Two students also get ablations (no masked training, no binary loss, no teacher, causal attention) so the model card shows which part actually mattered.

## Students

| Key | Starting model | Notes |
|---|---|---|
| `smollm2_135m` | `HuggingFaceTB/SmolLM2-135M` | Colab default |
| `smollm2_360m` | `HuggingFaceTB/SmolLM2-360M` | ablations |
| `gemma3_270m` | `google/gemma-3-270m` | gated, accept the license first |
| `qwen25_05b` | `Qwen/Qwen2.5-0.5B` | |
| `qwen3_06b` | `Qwen/Qwen3-0.6B-Base` | ablations |
| `qwen3_17b` | `Qwen/Qwen3-1.7B-Base` | largest, runs last, no ONNX build |

Public embedders scored on the same benchmark for the bytes against quality curve: all MiniLM L6 v2 (the reference), bge large en v1.5, e5 large v2, mxbai embed large v1, Qwen3 Embedding 0.6B and EmbeddingGemma 300M.

## Pipeline

| Stage | What it does |
|---|---|
| s00_gpu_check | GPU, precision, package versions, HuggingFace login, gated model access |
| s01_data | Query and positive pairs from MS MARCO, Natural Questions, HotpotQA, FEVER, GooAQ, SQuAD, TriviaQA and ELI5, a deduplicated passage corpus, and FineWeb Edu text for masked training |
| s02_teacher_dense | Qwen3 Embedding mines and scores the top candidates for every query |
| s03_teacher_colbert | GTE ModernColBERT rescores the same candidates with MaxSim |
| s04_baselines | Scores the public embedders on NanoBEIR at float32 and binary |
| s05_dataset | Uploads the distillation data and creates the collection |
| s06_models | The per model loop, below |
| s07_finalize | Cross model leaderboard, overall chart, refreshed model cards, `REPORT.md` |

### The per model loop

For each student, one after another:

1. Masked next token training (once per student).
2. Train with the dense, ColBERT and fused teachers, pick the best on dev, run the extra seeds, run the ablations.
3. Upload every run to HuggingFace as a private repo: `main` is the best run, every other run is a branch.
4. Test the starting LLM and every run on NanoBEIR at every vector size, as float32, binary, and float query against binary documents.
5. Print a GOOD or BAD verdict with the numbers, write the model card and charts, export ONNX, make the repo public if GOOD, add it to the collection.
6. Move to the next student.

Every step skips work that is already done, and training checkpoints every 1,000 steps, so a stopped pod or a dropped Colab session continues where it stopped.

## Verdict

On RunPod a model is GOOD when all three hold (see `verdict` in `vec16/config.yaml`):

* float32 NanoBEIR nDCG@10 is at least 5 points above the starting LLM
* 128 bit binary keeps at least 85% of the float32 score
* float32 is at least as good as all MiniLM L6 v2

GOOD models go public. BAD models stay private but are still added to the collection, so you can review them and flip them public by hand.

## Profiles

| Profile | Where | Students | Data | Seeds | Time |
|---|---|---|---|---|---|
| `small` | Colab, Kaggle, local | `smollm2_135m` | 3,000 queries, 3 NanoBEIR sets, small teacher | 1 | about 1 hour on a T4 |
| `full` | RunPod | all six | 100,000 queries, 13 NanoBEIR sets, 8B teacher | 3 | about 90 A40 hours |

The profile is picked from the platform. Override with `VEC16_PROFILE=small` or `VEC16_PROFILE=full`. Small runs publish with a `_smoke` suffix and stay private.

## Colab

Add your HuggingFace token in Colab secrets as `HF_TOKEN` (write access). Then open `notebooks/colab.ipynb`, or run:

```python
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

```bash
!git clone https://github.com/amareshhebbar/vec16.git
%cd vec16
!bash setup.sh
!bash run_all.sh
```

Output goes to `MyDrive/vec16_work/small`, so a disconnect loses nothing. Run `!bash run_all.sh` again to continue.

Try more students on the T4: `!VEC16_MODELS=smollm2_135m,smollm2_360m,qwen3_06b bash run_all.sh`

## RunPod

Use an A40 or A6000 (48 GB) with a network volume of at least 200 GB mounted at `/workspace`.

```bash
cd /workspace
git clone https://github.com/amareshhebbar/vec16.git
cd vec16
export HF_TOKEN=hf_xxx
bash setup.sh
mkdir -p /workspace/vec16_work
VEC16_STOP_POD=1 nohup bash run_all.sh > /workspace/vec16_work/run.log 2>&1 &
tail -f /workspace/vec16_work/run.log
```

Accept the licenses of `google/gemma-3-270m` and `google/embeddinggemma-300m` on huggingface.co first, or the GPU check stops the run and tells you which ones are missing.

## Checking on a run

```bash
python -m vec16.status
```

```
model          state                              verdict   float binary   base  repo
smollm2_135m   done                               GOOD       41.2   36.0   12.3  AmareshHebbar/Vec16_SmolLM2_135M
smollm2_360m   training fused seed 1
gemma3_270m    waiting
```

The log prints a block like this after every student:

```
GOOD  smollm2_135m  (HuggingFaceTB/SmolLM2-135M)
  starting LLM float32 nDCG@10   12.3
  Vec16 float32 576   nDCG@10   41.2
  Vec16 binary 128 bits nDCG@10   36.0  (16 bytes per document)
```

The numbers above only show the format. Training curves go to TensorBoard (`tensorboard --logdir <work>/runs`) and are uploaded to every model repo.

## Options

| Variable | Effect |
|---|---|
| `VEC16_PROFILE` | `small` or `full` |
| `VEC16_MODELS` | Comma separated student keys from `config.yaml` |
| `VEC16_BASELINES` | Comma separated baseline keys |
| `VEC16_FROM`, `VEC16_TO` | Run only stages in this range, for example `VEC16_FROM=6` |
| `VEC16_PUBLISH=0` | Skip every HuggingFace upload |
| `VEC16_STOP_POD=1` | Stop the RunPod pod when the run ends |
| `VEC16_WORK` | Custom output folder |
| `VEC16_CONFIG` | Use a different config file |

## What lands on HuggingFace

| Item | Name |
|---|---|
| Distillation data | `AmareshHebbar/Vec16_Distill` with `train`, `dev` and `corpus` |
| One repo per student | `AmareshHebbar/Vec16_<model>`, for example `Vec16_Qwen3_0.6B` |
| Branches in each repo | `dense`, `colbert`, `fused`, extra seeds, and ablations `no_mntp`, `no_qat`, `no_listwise`, `causal` |
| Inside `main` | weights, `bidirectional.py`, `onnx/model.onnx`, `onnx/model_quantized.onnx`, `assets/` charts, `results/` raw numbers, `runs/` TensorBoard logs |
| Collection | "Vec16 Embeddings" with the dataset and every student |

## Using a released model

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer("AmareshHebbar/Vec16_Qwen3_0.6B", trust_remote_code=True, truncate_dim=128)
docs = quantize_embeddings(model.encode_document(["Vaccines train the immune system."]), precision="ubinary")
print(docs.shape)
```

`SHIPPING.md` explains how every release is packaged and how to run it in Transformers, FAISS, vLLM, ONNX Runtime and the browser. Ready examples are in `serving/`.

## Budget

Rough A40 estimate for the `full` profile at about $0.44 per hour.

| Part | Hours |
|---|---|
| Teachers, baselines, masked training | about 8 |
| SmolLM2 135M, Gemma 3 270M | about 5 |
| SmolLM2 360M with ablations | about 15 |
| Qwen2.5 0.5B | about 9 |
| Qwen3 0.6B with ablations | about 20 |
| Qwen3 1.7B | about 35 |
| Evaluation and ONNX | about 4 |
| **Total** | **about 95 hours, about $42** |

Without Qwen3 1.7B (`VEC16_MODELS=smollm2_135m,smollm2_360m,gemma3_270m,qwen25_05b,qwen3_06b`) it is about 60 hours.

## Offline smoke test

Builds tiny random Llama, Qwen2, Qwen3 and Gemma3 models and runs every stage on CPU with no network:

```bash
bash tests/run_smoke.sh
```

It finishes with a check that every bidirectional model really sees right side context and that padding does not change the output.

## Layout

```
vec16/
  README.md
  SHIPPING.md
  requirements.txt
  setup.sh
  run_all.sh
  notebooks/colab.ipynb
  vec16/
    config.yaml
    common.py
    bidirectional.py
    students.py
    retrieval.py
    s00_gpu_check.py
    s01_data.py
    s02_teacher_dense.py
    s03_teacher_colbert.py
    s04_baselines.py
    s05_dataset.py
    s06_models.py
    s07_finalize.py
    mntp.py
    train.py
    evaluate.py
    export_onnx.py
    summary.py
    charts.py
    modelcard.py
    publish.py
    hub.py
    status.py
  serving/
    python/search_binary.py
    vllm/serve.sh
    vllm/client.py
    vllm/parity.py
    browser/index.html
  tests/
    fixtures.py
    check_bidirectional.py
    run_smoke.sh
```

## License

Code is MIT. Each released model follows the license of its starting model. The distillation data keeps the licenses of its sources.
