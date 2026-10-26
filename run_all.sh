#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

stop_pod() {
  if [ "${VEC16_STOP_POD:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap stop_pod EXIT

STAGES=(s00_gpu_check s01_data s02_teacher_dense s03_teacher_colbert s04_baselines s05_dataset s06_models s07_finalize)
FROM="${VEC16_FROM:-0}"
TO="${VEC16_TO:-7}"

for i in "${!STAGES[@]}"; do
  [ "$i" -lt "$FROM" ] && continue
  [ "$i" -gt "$TO" ] && continue
  st="${STAGES[$i]}"
  echo "=== stage $i $st"
  "$PY" -m "vec16.$st"
done

"$PY" -m vec16.status
echo "=== all stages finished"
