#!/usr/bin/env bash
# P0: locate the AQA negative transfer.
#
# The AQA arms are trained on AVQA and scored on MMAU Test-mini, and every arm is below the
# frozen base there.  Two explanations are still open, and they need different chapters:
#
#   cross-corpus    the transfer from AVQA to MMAU is what costs the accuracy, so the layer-band
#                   question was never really tested on the second task;
#   cross-task      the AVQA task and the MMAU task are not comparable, so the second task cannot
#                   answer the layer-band question at all.
#
# The test: score the SAME arms on AVQA questions that were held out from both the training and
# the early-stopping split.  Same corpus, same task, same distribution, and nothing memorised.
# Any remaining deficit is the adapter's own, with no transfer in the way.
#
#   pre-declared   if the adapters are not below the frozen base on held-out AVQA, the negative
#                  transfer is cross-corpus: the layer-band question was not tested on the
#                  second task;
#                  if they are below the frozen base on held-out AVQA too, the negative transfer
#                  is cross-task: the second task is not comparable.
#
# Inference only.  Nothing is trained.  The manifest is built by
# experiments/hearside_pilot/avqa_holdout.py, which splits by source video, so no video is in
# both the evaluation fold and the reserve.
#
#   nohup bash experiments/hearside_pilot/run_avqa_same_dist.sh > /root/autodl-fs/CoTA-LoRA/avqa-same-dist.out 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
DEPS="${DEPS:-$ROOT/deps}"
AQA_RUN="${AQA_RUN:-$ROOT/hearside-aqa-20261125-224035}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$ROOT/hearside-avqa-same-dist-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-$SRC/experiments/ms_swift_aqa/omni3b-pilot.json}"
ARMS="${ARMS:-frozen-base AQ-all AQ-last4 AQ-first4 AQ-mid4 AQ-all-s43 AQ-last4-s43 AQ-first4-s43 AQ-mid4-s43 AQ-all-s44 AQ-last4-s44 AQ-first4-s44 AQ-mid4-s44}"
SIZE="${SIZE:-3000}"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$RUN/logs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }
pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' \) \
          -size -40M -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "AVQA-SAME-DIST EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT
stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(next((d[k] for k in sys.argv[2:] if k in d), "n/a"))' "$@"; }

log "run folder: $RUN"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit')"
for path in "$PY" "$BASE_CONFIG" "$AQA_RUN/train/AQ-all/result.json"; do
    [ -e "$path" ] || stop "missing $path"
done

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/hearside_pilot/tests/test_aqa_run.py > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

# The held-out AVQA manifest, and then the two MMAU-style manifests built from it.
if [ ! -f "$RUN/holdout/holdout.json" ]; then
    "$PY" -m experiments.hearside_pilot.avqa_holdout --output "$RUN/holdout" --size "$SIZE" \
        > "$RUN/logs/holdout.log" 2>&1 \
        || stop "building the held-out manifest failed; see logs/holdout.log: $(tail -1 "$RUN/logs/holdout.log")"
    log "held-out AVQA: $(tr -d '\n' < "$RUN/logs/holdout.log" | cut -c1-400)"
fi
if [ ! -f "$RUN/avqa/aqa.json" ]; then
    "$PY" -m experiments.hearside_pilot.aqa build --test "$RUN/holdout/holdout.jsonl" --output "$RUN/avqa" \
        > "$RUN/logs/build.log" 2>&1 \
        || stop "building the evaluation manifests failed; see logs/build.log: $(tail -1 "$RUN/logs/build.log")"
    log "manifests: $(tr -d '\n' < "$RUN/logs/build.log" | cut -c1-300)"
fi

for arm in $ARMS; do
    if [ "$arm" = frozen-base ]; then
        started=$SECONDS
        "$PY" -m experiments.hearside_pilot.aqa run --folder "$RUN/avqa" --model-config "$BASE_CONFIG" \
            --method frozen-base --arm frozen-base > "$RUN/logs/run-frozen-base.log" 2>&1
        code=$?
    else
        out="$AQA_RUN/train/$arm"
        [ -f "$out/result.json" ] || { log "$arm has no trained checkpoint, skipped"; continue; }
        started=$SECONDS
        "$PY" -m experiments.hearside_pilot.aqa run --folder "$RUN/avqa" \
            --model-config "$AQA_RUN/configs/$arm.json" --method ta-lora \
            --checkpoint "$(field "$out/result.json" best_checkpoint)" --arm "$arm" \
            > "$RUN/logs/run-$arm.log" 2>&1
        code=$?
    fi
    log "$arm on the held-out AVQA (exit $code, $(((SECONDS - started) / 60)) min)"
    [ "$code" -eq 0 ] || log "$arm could not be scored; see logs/run-$arm.log: $(tail -1 "$RUN/logs/run-$arm.log")"
done

"$PY" -m experiments.hearside_pilot.aqa summarize --folder "$RUN/avqa" --output "$RUN/summary" \
    > "$RUN/logs/summary.log" 2>&1 \
    || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"

# The same-distribution comparison against the MMAU numbers that are already tracked.
"$PY" research/tools/avqa_vs_mmau.py --avqa "$RUN/avqa" --mmau "$AQA_RUN/mmau" \
    > "$RUN/summary/avqa-vs-mmau.md" 2>&1 \
    || log "the comparison failed; see summary/avqa-vs-mmau.md: $(tail -1 "$RUN/summary/avqa-vs-mmau.md")"

[ -f "$RUN/summary/aqa.md" ] && cat "$RUN/summary/aqa.md" | tee -a "$RUN/pilot.log"
[ -f "$RUN/summary/avqa-vs-mmau.md" ] && cat "$RUN/summary/avqa-vs-mmau.md" | tee -a "$RUN/pilot.log"
log "done"
