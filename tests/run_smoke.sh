#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:-/tmp/vec16_smoke}"
rm -rf "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export VEC16_ALLOW_CPU=1
export VEC16_PUBLISH=0
export VEC16_WORK="$ROOT/work"
export VEC16_CONFIG="$(python tests/fixtures.py "$ROOT" "$ROOT/work" | tail -n 1)"

bash run_all.sh
python tests/check_bidirectional.py "$ROOT/work/small/runs"
echo "smoke test passed"
