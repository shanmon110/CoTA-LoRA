#!/usr/bin/env bash
set -euo pipefail
package=$(cd "$(dirname "$0")" && pwd)
cd "$package/../.."
runtime=${SWIFT_ASR_HOME:-/root/autodl-tmp/ms-swift-librispeech}
export PATH="$runtime/venv/bin:$PATH" HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} PYTHONUNBUFFERED=1
case "${1:-}" in
    prepare) shift; exec "$runtime/venv/bin/python" -m experiments.ms_swift_librispeech.prepare "$@" ;;
    compare) shift; exec "$runtime/venv/bin/python" -m experiments.ms_swift_librispeech.compare "$@" ;;
    *) echo 'Usage: run.sh prepare --config CONFIG --output DATA | compare --config CONFIG --data DATA --output RUN' >&2; exit 2 ;;
esac
