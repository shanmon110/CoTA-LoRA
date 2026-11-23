#!/usr/bin/env bash
# Hear-side pilot on IEMOCAP, start to finish:
#   unit tests -> manifests and configs -> frozen-base evaluation -> per arm: train, evaluate -> summary -> one archive
#
#   nohup bash experiments/hearside_pilot/run_pilot.sh > /root/autodl-fs/CoTA-LoRA/hearside-pilot.out 2>&1 &
#
# Paths default to this project's server; each can be overridden through the environment.
# Running again with RUN=<the same folder> continues where it stopped. Nothing outside RUN is written.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
IEMOCAP="${IEMOCAP:-/root/autodl-tmp/iemocap/swift-ser}"
BASE_CONFIG="${BASE_CONFIG:-$IEMOCAP/pair-01/config.json}"     # supplies the model path and the training recipe
DEPS="${DEPS:-$ROOT/deps}"                                      # praat-parselmouth, installed for the D2 diagnostic
RUN="${RUN:-$ROOT/hearside-pilot-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$SRC/experiments/hearside_pilot/pilot-iemocap-session4.json"
ARMS="${ARMS:-A-all B-audio D-all-shuffled C-post-audio}"       # the control C last: the criteria do not need it
CONDITIONS="orig orig_reworded text question_only asr-orig"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$RUN/logs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }

pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the small result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' -o -name '*.txt' \) \
          -size -40M ! -path './eval/audio/*' ! -name 'tokenizer*' ! -name 'vocab.json' ! -name 'added_tokens.json' \
          ! -name 'merges.txt' ! -name 'special_tokens_map.json' -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "PILOT EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

# ------------------------------------------------------------------ 0. what is being run, and on what
log "run folder: $RUN"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain 2>/dev/null | head -1 | sed 's/.*/(with local changes)/')"
for path in "$PY" "$BASE_CONFIG" "$IEMOCAP/data/train.jsonl" "$PROTOCOL"; do
    [ -e "$path" ] || stop "missing $path"
done
{
    "$PY" -c 'import sys, torch, transformers, peft, swift, parselmouth
print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
print("swift", getattr(swift, "__version__", "?"), swift.__file__)
print("parselmouth", parselmouth.__version__, "cuda devices", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null
    df -h "$RUN" | tail -1
} > "$RUN/logs/environment.txt" 2>&1 || stop "the environment check failed; see logs/environment.txt"
log "environment: $(tr '\n' ' ' < "$RUN/logs/environment.txt")"

# ------------------------------------------------------------------ 1. unit tests; nothing is trained unless they pass
if [ -f "$RUN/logs/tests-gating.ok" ]; then
    log "unit tests: passed in an earlier start of this run"
else
    log "unit tests (gating)"
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/ms_swift_librispeech/tests/test_update_positions.py \
        experiments/ms_swift_librispeech/tests/test_adapter_pair.py \
        experiments/ms_swift_librispeech/tests/test_conditional_gate.py \
        experiments/trr_diagnostics/tests experiments/hearside_pilot/tests > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was trained. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
    # The remaining project tests are recorded for review and do not decide whether the pilot runs.
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 timeout 900 \
        "$PY" -m pytest -q -rs -p no:cacheprovider experiments/ms_swift_aqa/tests experiments/ms_swift_librispeech/tests \
        experiments/ms_swift_iemocap/tests research/claude-review/checks > "$RUN/logs/tests-all.log" 2>&1
    log "all project tests (recorded only, exit $?): $(tail -1 "$RUN/logs/tests-all.log")"
fi

# ------------------------------------------------------------------ 2. manifests, arm configs, evaluation inputs
if [ ! -f "$RUN/data/splits.json" ]; then
    rm -rf "$RUN/data"
    "$PY" -m experiments.hearside_pilot.data --source "$IEMOCAP/data/train.jsonl" --protocol "$PROTOCOL" --output "$RUN/data" \
        > "$RUN/logs/data.log" 2>&1 || stop "building the training manifests failed; see logs/data.log"
fi
if [ ! -f "$RUN/configs/arms.json" ]; then
    rm -rf "$RUN/configs"
    "$PY" -m experiments.hearside_pilot.configs --base "$BASE_CONFIG" --data "$RUN/data" --output "$RUN/configs" \
        > "$RUN/logs/configs.log" 2>&1 || stop "building the arm configs failed; see logs/configs.log"
fi
if [ ! -f "$RUN/eval/prepare.json" ]; then
    rm -rf "$RUN/eval"
    "$PY" -m experiments.trr_diagnostics.d2 prepare --config "$PROTOCOL" --output "$RUN/eval" --workers 4 \
        > "$RUN/logs/prepare.log" 2>&1 || stop "preparing the Session 4 evaluation failed; see logs/prepare.log"
fi
log "manifests, configs and evaluation inputs are ready"

# ------------------------------------------------------------------ 3. evaluation and training
evaluate() {   # arm, model config, method, [checkpoint]
    local arm=$1 config=$2 method=$3 checkpoint=${4:-}
    local started=$SECONDS
    "$PY" -m experiments.trr_diagnostics.d2 run --prepared "$RUN/eval" --model-config "$config" --method "$method" \
        ${checkpoint:+--checkpoint "$checkpoint"} --arm "$arm" --only $CONDITIONS >> "$RUN/logs/eval-$arm.log" 2>&1
    local code=$?
    if [ -f "$RUN/eval/runs/$arm/orig/metrics.json" ]; then
        log "$arm evaluated in $(((SECONDS - started) / 60)) min (exit $code): $(grep -E '^(orig|orig_reworded|text|question_only|asr-orig): ' "$RUN/logs/eval-$arm.log" | tr '\n' ';')"
    else
        log "$arm: evaluation FAILED (exit $code); see logs/eval-$arm.log: $(tail -1 "$RUN/logs/eval-$arm.log")"
    fi
}

summarize() {
    "$PY" -m experiments.hearside_pilot.summarize --prepared "$RUN/eval" --output "$RUN/summary" > "$RUN/logs/summary.log" 2>&1 \
        || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
}

evaluate frozen-base "$BASE_CONFIG" frozen-base
[ -f "$RUN/eval/runs/frozen-base/orig/metrics.json" ] || stop "the frozen base could not be evaluated, so the adapters were not trained"

for arm in $ARMS; do
    out="$RUN/train/$arm"
    config="$RUN/configs/$arm.json"
    [ -f "$config" ] || { log "$arm: unknown arm, skipped"; continue; }
    if [ ! -f "$out/result.json" ]; then
        [ -d "$out" ] && mv "$out" "$out.incomplete-$(date +%H%M%S)"      # an interrupted training is kept, not resumed
        version=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["option_order"])' "$RUN/configs/arms.json" "$arm")
        log "$arm: training on train-$version.jsonl"
        "$PY" -m experiments.ms_swift_librispeech.train --config "$config" --method ta-lora \
            --train "$RUN/data/train-$version.jsonl" --dev "$RUN/data/dev-$version.jsonl" --output "$out" > "$RUN/logs/train-$arm.log" 2>&1
        code=$?
        if [ ! -f "$out/result.json" ]; then
            log "$arm: training FAILED (exit $code); see logs/train-$arm.log: $(grep -E 'Error|error|Killed' "$RUN/logs/train-$arm.log" | tail -1)"
            continue
        fi
    fi
    log "$arm trained: best step $(field "$out/result.json" best_step), Session 3 UAR $(field "$out/result.json" best_uar), $(field "$out/result.json" training_seconds) s of training, peak $(field "$out/result.json" peak_allocated_bytes) bytes"
    evaluate "$arm" "$config" ta-lora "$(field "$out/result.json" best_checkpoint)"
    summarize
done

# ------------------------------------------------------------------ 4. the comparison
summarize
[ -f "$RUN/summary/summary.md" ] && cat "$RUN/summary/summary.md" | tee -a "$RUN/pilot.log"
log "done"
