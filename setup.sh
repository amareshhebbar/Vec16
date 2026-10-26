#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -q -U pip
python -m pip install -q -U -r requirements.txt

PLATFORM=$(python -m vec16.common)
echo "platform: $PLATFORM"

python - <<'EOF'
import os
from vec16.common import hf_login
print("hf login:", "ok" if hf_login() else "missing HF_TOKEN")
EOF

python -c "import torch, sentence_transformers, transformers, pylate, onnxruntime; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'sentence_transformers', sentence_transformers.__version__, 'transformers', transformers.__version__, 'pylate', pylate.__version__)"
echo "setup done"
