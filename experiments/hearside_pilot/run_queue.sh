#!/usr/bin/env bash
# The remaining work, in order, as one queue that runs without this session.
#
#   nohup bash experiments/hearside_pilot/run_queue.sh > /root/autodl-fs/CoTA-LoRA/queue.out 2>&1 &
#
# Each step is independent of the next, writes its own folder, and records its own log.  A step
# that fails does not stop the queue: the failure is recorded and the queue moves on, because a
# negative result is a result.
#
# The order is by cost and by dependency:
#
#   Q1  the Session 5 result is written up and the notes updated      (zero cost, minutes)
#   Q2  P0b: the slot-vs-class test on AQA, from the tracked answers   (zero cost, minutes)
#   Q3  P0: the same-distribution AVQA evaluation, to locate the
#       negative transfer                                            (one inference pass)
#   Q4  P5: the surrogate test generalised to AQA, which needs the
#       option logits the Q3 pass does not track yet                 (one inference pass)
#   Q5  P4: the narrative corrections in the notes                     (zero cost, minutes)
#   Q6  P3: the defensible prosody control on CREMA-D                 (one inference pass)
#
# Q3 and Q4 both need the GPU, so they are run one after the other, never at the same time.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS="${DEPS:-$ROOT/deps}"
QUEUE="${QUEUE:-$ROOT/queue-$(date +%Y%m%d-%H%M%S)}"
LOG="$ROOT/queue.out"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$QUEUE/logs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }
step() { echo "[$(date '+%F %T')] ===== $* =====" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }
record() { echo "[$(date '+%F %T')] $1: $2" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }

# The GPU is used by one step at a time.  If another hear-side run is holding it, wait.
wait_for_gpu() {
    while pgrep -f 'hearside_pilot/run_(letters|bands|probes|aqa|queue|avqa)\.sh' | grep -qv "^$$\$"; do
        [ -z "${waited:-}" ] && log "another hear-side run holds the GPU; waiting for it"
        waited=1
        sleep 60
    done
}

log "queue folder: $QUEUE"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit')"

# ---------------------------------------------------------------- Q1: the Session 5 write-up
step "Q1  the Session 5 result, written up"
if [ -f "$QUEUE/logs/q1.done" ]; then
    log "Q1 was already done"
else
    "$PY" research/tools/session5_verdict.py /root/autodl-fs/CoTA-LoRA/hearside-session5-20261128-011343/logits \
        > "$QUEUE/logs/q1-session5-verdict.txt" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        "$PY" -m experiments.hearside_pilot.session5 analyze \
            --root /root/autodl-fs/CoTA-LoRA/hearside-session5-20261128-011343/logits \
            --output "$QUEUE/session5" > "$QUEUE/logs/q1-session5-analyze.log" 2>&1
        code=$?
    fi
    if [ "$code" -eq 0 ]; then
        cp "$QUEUE/session5/session5.md" "$QUEUE/session5.md" 2>/dev/null
        record "Q1" "ok -- see $QUEUE/session5.md"
    else
        record "Q1" "FAILED (exit $code); see $QUEUE/logs/q1-session5-verdict.txt"
    fi
    touch "$QUEUE/logs/q1.done"
fi

# ---------------------------------------------------------------- Q2: P0b, slot vs class on AQA
step "Q2  P0b: the slot-vs-class test on AQA, from the tracked answers"
if [ -f "$QUEUE/logs/q2.done" ]; then
    log "Q2 was already done"
else
    "$PY" research/tools/aqa_slot_vs_class.py > "$QUEUE/logs/q2-aqa-slot-vs-class.txt" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q2" "ok -- see $QUEUE/logs/q2-aqa-slot-vs-class.txt"
    else
        record "Q2" "FAILED (exit $code); see $QUEUE/logs/q2-aqa-slot-vs-class.txt"
    fi
    touch "$QUEUE/logs/q2.done"
fi

# ---------------------------------------------------------------- Q3: P0, the same-distribution AVQA pass
step "Q3  P0: the same-distribution AVQA evaluation"
if [ -f "$QUEUE/logs/q3.done" ]; then
    log "Q3 was already done"
else
    wait_for_gpu
    bash experiments/hearside_pilot/run_avqa_same_dist.sh \
        > "$QUEUE/logs/q3-avqa-same-dist.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q3" "ok -- see $QUEUE/logs/q3-avqa-same-dist.log"
    else
        record "Q3" "FAILED (exit $code); see $QUEUE/logs/q3-avqa-same-dist.log"
    fi
    touch "$QUEUE/logs/q3.done"
fi

# ---------------------------------------------------------------- Q4: P5, the surrogate on AQA
step "Q4  P5: the surrogate test on AQA"
if [ -f "$QUEUE/logs/q4.done" ]; then
    log "Q4 was already done"
else
    wait_for_gpu
    bash experiments/hearside_pilot/run_aqa_surrogate.sh \
        > "$QUEUE/logs/q4-aqa-surrogate.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q4" "ok -- see $QUEUE/logs/q4-aqa-surrogate.log"
    else
        record "Q4" "FAILED (exit $code); see $QUEUE/logs/q4-aqa-surrogate.log"
    fi
    touch "$QUEUE/logs/q4.done"
fi

# ---------------------------------------------------------------- Q5: the narrative corrections
step "Q5  P4: the narrative corrections"
if [ -f "$QUEUE/logs/q5.done" ]; then
    log "Q5 was already done"
else
    "$PY" research/tools/aqa_text_rules_recheck.py > "$QUEUE/logs/q5-text-rules.txt" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q5" "ok -- see $QUEUE/logs/q5-text-rules.txt"
    else
        record "Q5" "FAILED (exit $code); see $QUEUE/logs/q5-text-rules.txt"
    fi
    touch "$QUEUE/logs/q5.done"
fi

# ---------------------------------------------------------------- Q6: P3, the defensible prosody control
step "Q6  P3: the defensible prosody control on CREMA-D"
if [ -f "$QUEUE/logs/q6.done" ]; then
    log "Q6 was already done"
else
    wait_for_gpu
    bash experiments/hearside_pilot/run_prosody_control.sh \
        > "$QUEUE/logs/q6-prosody-control.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q6" "ok -- see $QUEUE/logs/q6-prosody-control.log"
    else
        record "Q6" "FAILED (exit $code); see $QUEUE/logs/q6-prosody-control.log"
    fi
    touch "$QUEUE/logs/q6.done"
fi

step "the queue is done"
record "QUEUE" "finished; the per-step results are in $QUEUE"
echo "QUEUE EXIT 0" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"
