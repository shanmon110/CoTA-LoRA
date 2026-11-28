#!/usr/bin/env bash
# The remaining work, in order, as one queue that runs without this session.
#
#   nohup bash experiments/hearside_pilot/run_queue.sh > /root/autodl-fs/CoTA-LoRA/queue.out 2>&1 &
#
# Each step writes its own output under QUEUE and records its own log.  A step that fails does
# not stop the queue: the failure is recorded and the queue moves on, because a negative result
# is a result.
#
# The order is by dependency and by cost:
#
#   Q1  the Session 5 verdict, from the logits the run already produced   (zero cost)
#   Q2  P0b: the slot-vs-class test on AQA, from the tracked answers       (zero cost)
#   Q3  P4: the text scoring rules, every rule side by side               (zero cost)
#   Q4  P3: the defensible prosody control on CREMA-D                      (zero cost)
#   Q5  P0: the same-distribution AVQA evaluation, to locate the negative
#       transfer                                                          (one GPU pass)
#   Q6  P5: the surrogate test on AQA, which needs the option logits the
#       tracked runs do not have                                         (one GPU pass)
#
# Q5 and Q6 need the GPU, so they run one after the other, never at the same time.  The
# zero-cost steps run first, so the queue has something to report even if the GPU steps fail.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS="${DEPS:-$ROOT/deps}"
QUEUE="${QUEUE:-$ROOT/queue-$(date +%Y%m%d-%H%M%S)}"
LOG="$ROOT/queue.out"
SESSION5="${SESSION5:-$ROOT/hearside-session5-20261128-011343/logits}"
AQA_RUN="${AQA_RUN:-$ROOT/hearside-aqa-20261125-224035}"
CROSS="${CROSS:-$ROOT/hearside-cross-20261124-213539}"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$QUEUE/logs" "$QUEUE/results" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }
step() { echo "[$(date '+%F %T')] ===== $* =====" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }
record() { echo "[$(date '+%F %T')] $1: $2" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"; }

# The GPU is used by one step at a time.
wait_for_gpu() {
    while pgrep -f 'hearside_pilot/run_(letters|bands|probes|aqa|queue|avqa|avqa_surrogate)\.sh' > /dev/null; do
        [ -z "${waited:-}" ] && log "another hear-side run holds the GPU; waiting for it"
        waited=1
        sleep 60
    done
}

log "queue folder: $QUEUE"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit')"

# ---------------------------------------------------------------- Q1: the Session 5 verdict
step "Q1  the Session 5 verdict"
if [ -f "$QUEUE/logs/q1.done" ]; then
    log "Q1 was already done"
else
    code=0
    "$PY" research/tools/session5_verdict.py "$SESSION5" > "$QUEUE/results/q1-session5-verdict.txt" 2>&1 || code=$?
    if [ "$code" -eq 0 ]; then
        "$PY" -m experiments.hearside_pilot.session5 analyze --root "$SESSION5" --output "$QUEUE/results/session5" \
            > "$QUEUE/logs/q1-session5-analyze.log" 2>&1 || code=$?
    fi
    if [ "$code" -eq 0 ]; then
        record "Q1" "ok -- $QUEUE/results/q1-session5-verdict.txt and $QUEUE/results/session5/session5.md"
    else
        record "Q1" "FAILED (exit $code); see $QUEUE/logs/q1-session5-analyze.log"
    fi
    touch "$QUEUE/logs/q1.done"
fi

# ---------------------------------------------------------------- Q2: P0b on AQA
step "Q2  P0b: the slot-vs-class test on AQA"
if [ -f "$QUEUE/logs/q2.done" ]; then
    log "Q2 was already done"
else
    code=0
    "$PY" research/tools/aqa_slot_vs_class.py "$AQA_RUN/mmau" > "$QUEUE/results/q2-aqa-slot-vs-class.txt" 2>&1 || code=$?
    if [ "$code" -eq 0 ]; then
        record "Q2" "ok -- $QUEUE/results/q2-aqa-slot-vs-class.txt"
    else
        record "Q2" "FAILED (exit $code); see $QUEUE/logs/q2-*"
    fi
    touch "$QUEUE/logs/q2.done"
fi

# ---------------------------------------------------------------- Q3: P4, the text scoring rules
step "Q3  P4: the text scoring rules, every rule side by side"
if [ -f "$QUEUE/logs/q3.done" ]; then
    log "Q3 was already done"
else
    code=0
    "$PY" research/tools/text_rules2.py > "$QUEUE/results/q3-text-rules.txt" 2>&1 || code=$?
    if [ "$code" -eq 0 ]; then
        record "Q3" "ok -- $QUEUE/results/q3-text-rules.txt"
    else
        record "Q3" "FAILED (exit $code); see $QUEUE/results/q3-text-rules.txt"
    fi
    touch "$QUEUE/logs/q3.done"
fi

# ---------------------------------------------------------------- Q4: P3, the prosody control
step "Q4  P3: the defensible prosody control on CREMA-D"
if [ -f "$QUEUE/logs/q4.done" ]; then
    log "Q4 was already done"
else
    code=0
    "$PY" research/tools/prosody_control.py "$CROSS/cremad" > "$QUEUE/results/q4-prosody-control.txt" 2>&1 || code=$?
    if [ "$code" -eq 0 ]; then
        record "Q4" "ok -- $QUEUE/results/q4-prosody-control.txt"
    else
        record "Q4" "FAILED (exit $code); see $QUEUE/results/q4-prosody-control.txt"
    fi
    touch "$QUEUE/logs/q4.done"
fi

# ---------------------------------------------------------------- Q5: P0, the same-distribution AVQA pass
step "Q5  P0: the same-distribution AVQA evaluation"
if [ -f "$QUEUE/logs/q5.done" ]; then
    log "Q5 was already done"
else
    wait_for_gpu
    bash experiments/hearside_pilot/run_avqa_same_dist.sh > "$QUEUE/logs/q5-avqa-same-dist.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q5" "ok -- see $QUEUE/logs/q5-avqa-same-dist.log"
    else
        record "Q5" "FAILED (exit $code); see $QUEUE/logs/q5-avqa-same-dist.log"
    fi
    touch "$QUEUE/logs/q5.done"
fi

# ---------------------------------------------------------------- Q6: P5, the surrogate on AQA
step "Q6  P5: the surrogate test on AQA"
if [ -f "$QUEUE/logs/q6.done" ]; then
    log "Q6 was already done"
else
    wait_for_gpu
    bash experiments/hearside_pilot/run_aqa_surrogate.sh > "$QUEUE/logs/q6-aqa-surrogate.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        record "Q6" "ok -- see $QUEUE/logs/q6-aqa-surrogate.log"
    else
        record "Q6" "FAILED (exit $code); see $QUEUE/logs/q6-aqa-surrogate.log"
    fi
    touch "$QUEUE/logs/q6.done"
fi

step "the queue is done"
record "QUEUE" "finished; the results are under $QUEUE/results"
echo "QUEUE EXIT 0" | tee -a "$QUEUE/queue.log" | tee -a "$LOG"
