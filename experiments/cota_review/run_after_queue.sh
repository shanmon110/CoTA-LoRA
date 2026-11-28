#!/usr/bin/env bash
# Runs the three-arm comparison AFTER the hear-side queue has finished.
#
#   nohup bash experiments/cota_review/run_after_queue.sh > /root/autodl-fs/CoTA-LoRA/cota-comparison.out 2>&1 &
#
# Why a separate script and not a seventh step inside the queue: the queue is running right now,
# and bash reads a running script by byte offset, so editing it in place could make it execute
# garbage.  This script is therefore separate, and it waits for the queue to exit.
#
# The comparison script itself carries the pre-declared reading in its header.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="$ROOT/cota-comparison.out"

export PYTHONPATH="$SRC:$ROOT/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for the hear-side queue to finish"
while pgrep -f 'hearside_pilot/run_queue\.sh' | grep -v "^$$\$" | grep -q .; do
    sleep 120
done
log "the queue is gone; the GPU should be free"

# The GPU can still be held by a run the queue started and did not wait for, so wait again.
while pgrep -f 'hearside_pilot/run_(letters|bands|probes|avqa_same_dist|avqa_surrogate)\.sh|cota_review/run_comparison\.sh' \
      | grep -v "^$$\$" | grep -q .; do
    sleep 60
done

log "starting the three-arm comparison"
bash experiments/cota_review/run_comparison.sh >> "$LOG" 2>&1
code=$?
log "the three-arm comparison finished with exit code $code"
echo "COTA-COMPARISON-WAITER EXIT $code" | tee -a "$LOG"
exit $code
