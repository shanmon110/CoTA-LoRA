#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
exec /root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python -m experiments.ms_swift_aqa.papers28.run run "$@"
