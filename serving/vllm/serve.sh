#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-AmareshHebbar/Vec16_Qwen3_0.6B}"
PORT="${PORT:-8000}"

pip install -q "vllm>=0.32"

vllm serve "$MODEL" \
  --runner pooling \
  --port "$PORT" \
  --max-model-len 512 \
  --dtype bfloat16
