#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${AQA_PYTHON:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}
out=${AQA_OUTPUT:-/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01}
export PATH="$(dirname "$py"):$PATH" HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0
export OMP_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$py" -m experiments.ms_swift_aqa.run_tuners --output "$out" "$@"
