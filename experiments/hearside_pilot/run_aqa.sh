#!/usr/bin/env bash
# Second task for the layer-band question (declared 2026-11-25): train on AVQA, score on MMAU Test-mini.
# See research/second-task-and-grpo-20261125.md, part 1a. Twelve adapters are trained (four arms, three seeds).
#
#   nohup bash experiments/hearside_pilot/run_aqa.sh > /root/autodl-fs/CoTA-LoRA/hearside-aqa.out 2>&1 &
#
# The AVQA and MMAU folders are only read. Everything is written into RUN; running again with
# RUN=<the same folder> continues where it stopped.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
DEPS="${DEPS:-$ROOT/deps}"
AQA_DATA="${AQA_DATA:-/root/autodl-tmp/r1-aqa/data/pilot-01}"
RUN="${RUN:-$ROOT/hearside-aqa-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-$SRC/experiments/ms_swift_aqa/omni3b-pilot.json}"
ARMS="${ARMS:-AQ-all AQ-last4 AQ-first4 AQ-mid4 AQ-all-s43 AQ-last4-s43 AQ-first4-s43 AQ-mid4-s43 AQ-all-s44 AQ-last4-s44 AQ-first4-s44 AQ-mid4-s44}"
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
          -size -40M ! -name 'tokenizer*' ! -name 'vocab.json' ! -path '*/checkpoint-*' -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "AQA EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(next((d[k] for k in sys.argv[2:] if k in d), "n/a"))' "$@"; }

log "run folder: $RUN"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$AQA_DATA/train.jsonl" "$AQA_DATA/dev.jsonl" "$AQA_DATA/test.jsonl"; do
    [ -e "$path" ] || stop "missing $path"
done
while pgrep -f 'hearside_pilot/run_(bands|probes|letters).sh' > /dev/null; do
    [ -z "${waited:-}" ] && log "another hear-side run is using the GPU; waiting for it to finish"
    waited=1
    sleep 60
done

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/ms_swift_librispeech/tests/test_layer_band.py experiments/ms_swift_librispeech/tests/test_update_positions.py \
        experiments/hearside_pilot/tests/test_aqa.py > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

if [ ! -f "$RUN/configs/arms.json" ]; then
    rm -rf "$RUN/configs"
    "$PY" -m experiments.hearside_pilot.aqa configs --base "$BASE_CONFIG" --output "$RUN/configs" > "$RUN/logs/configs.log" 2>&1 \
        || stop "building the arm configs failed; see logs/configs.log: $(tail -1 "$RUN/logs/configs.log")"
fi
EVAL="$RUN/mmau"
if [ ! -f "$EVAL/aqa.json" ]; then
    rm -rf "$EVAL"
    "$PY" -m experiments.hearside_pilot.aqa build --test "$AQA_DATA/test.jsonl" --output "$EVAL" > "$RUN/logs/build.log" 2>&1 \
        || stop "building the MMAU manifests failed; see logs/build.log: $(tail -1 "$RUN/logs/build.log")"
fi
log "MMAU: $(tr -d '\n' < "$RUN/logs/build.log" | cut -c1-400)"

summary() {
    "$PY" -m experiments.hearside_pilot.aqa summarize --folder "$EVAL" --output "$RUN/summary" > "$RUN/logs/summary.log" 2>&1 \
        || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
}

started=$SECONDS
"$PY" -m experiments.hearside_pilot.aqa run --folder "$EVAL" --model-config "$BASE_CONFIG" --method frozen-base --arm frozen-base \
    > "$RUN/logs/run-frozen-base.log" 2>&1
log "frozen-base (exit $?, $(((SECONDS - started) / 60)) min)"
[ -f "$EVAL/runs/frozen-base/text/metrics.json" ] || stop "the frozen base could not be scored; see logs/run-frozen-base.log: $(tail -1 "$RUN/logs/run-frozen-base.log")"

for arm in $ARMS; do
    out="$RUN/train/$arm"
    if [ ! -f "$out/result.json" ]; then
        [ -d "$out" ] && mv "$out" "$out.incomplete-$(date +%H%M%S)"      # an interrupted training is kept, not resumed
        started=$SECONDS
        "$PY" -m experiments.ms_swift_librispeech.train --config "$RUN/configs/$arm.json" --method ta-lora \
            --train "$AQA_DATA/train.jsonl" --dev "$AQA_DATA/dev.jsonl" --output "$out" > "$RUN/logs/train-$arm.log" 2>&1
        code=$?
        if [ ! -f "$out/result.json" ]; then
            log "$arm: training FAILED (exit $code, $(((SECONDS - started) / 60)) min); see logs/train-$arm.log: $(grep -E 'Error|error|Killed' "$RUN/logs/train-$arm.log" | tail -1 | cut -c1-300)"
            continue
        fi
    fi
    log "$arm trained: best step $(field "$out/result.json" best_step), dev accuracy $(field "$out/result.json" best_accuracy best_metric best_uar)"
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.aqa run --folder "$EVAL" --model-config "$RUN/configs/$arm.json" --method ta-lora \
        --checkpoint "$(field "$out/result.json" best_checkpoint)" --arm "$arm" > "$RUN/logs/run-$arm.log" 2>&1
    log "$arm on MMAU (exit $?, $(((SECONDS - started) / 60)) min)"
    summary
done
summary
[ -f "$RUN/summary/aqa.md" ] && cat "$RUN/summary/aqa.md" | tee -a "$RUN/pilot.log"
log "done"
