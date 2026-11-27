#!/usr/bin/env bash
# Fourth run of the hear-side line (declared 2026-11-24): every adapter trained on IEMOCAP, and the frozen base,
# scored on CREMA-D. Nothing is trained. See research/cross-corpus-cremad-20261124.md.
#
#   CREMAD=/path/to/CREMA-D/AudioWAV nohup bash experiments/hearside_pilot/run_cross.sh > /root/autodl-fs/CoTA-LoRA/hearside-cross.out 2>&1 &
#
# The three earlier run folders are only read. Everything is written into RUN; running again with
# RUN=<the same folder> continues where it stopped.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
IEMOCAP="${IEMOCAP:-/root/autodl-tmp/iemocap/swift-ser}"
BASE_CONFIG="${BASE_CONFIG:-$IEMOCAP/pair-01/config.json}"
DEPS="${DEPS:-$ROOT/deps}"
PILOT="${PILOT:-$ROOT/hearside-pilot-20261123-160928}"
FOLLOWUP="${FOLLOWUP:-$ROOT/hearside-followup-20261123-212622}"
NEXT="${NEXT:-$ROOT/hearside-next-20261124-031913}"
CREMAD="${CREMAD:-/root/autodl-tmp/crema-d/AudioWAV}"
RUN="${RUN:-$ROOT/hearside-cross-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$SRC/experiments/hearside_pilot/pilot-iemocap-session4.json"
BATCH="${BATCH:-}"          # empty: 2 for the letter question and 4 for free text, as on Session 4
ARMS="${ARMS:-D-all-shuffled D-all-shuffled-s43 D-all-shuffled-s44 A-all A-all-s43 A-all-s44 A-last4 A-last4-s43 A-last4-s44 B-audio B-audio-s43 B-audio-s44 C-post-audio}"
MCQ_BATCH="${MCQ_BATCH:-}"       # empty: the batch size of the Session 4 protocol (2)
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
for old in "$PILOT" "$FOLLOWUP" "$NEXT"; do
    if [ -d "$old" ] && [ -d "$RUN" ] && [ "$(cd "$RUN" && pwd -P)" = "$(cd "$old" && pwd -P)" ]; then
        echo "RUN must be a new folder, not an earlier run: $RUN" >&2
        exit 1
    fi
done
mkdir -p "$RUN/logs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }

pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the small result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' -o -name '*.txt' \) \
          -size -40M -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "CROSS EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
home() {   # the run folder that trained an arm
    for root in "$NEXT" "$FOLLOWUP" "$PILOT"; do
        [ -f "$root/train/$1/result.json" ] && { echo "$root"; return 0; }
    done
    return 1
}

log "run folder: $RUN"
log "read only: $PILOT, $FOLLOWUP, $NEXT and $CREMAD"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null) $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(with local changes)/')"
[ -d "$CREMAD" ] || stop "CREMA-D audio folder not found: $CREMAD. Set CREMAD=/path/to/AudioWAV (the folder holding 1001_DFA_ANG_XX.wav and the other 7441 clips)"
clips=$(find "$CREMAD" -maxdepth 1 -name '*.wav' | wc -l)
[ "$clips" -ge 7000 ] || stop "$CREMAD holds $clips wav files; the full CREMA-D release has 7442. A partial copy would change the selected clips"
small=$(find "$CREMAD" -maxdepth 1 -name '*.wav' -size -2k | wc -l)
[ "$small" -eq 0 ] || stop "$small wav files in $CREMAD are under 2 KB; these are Git LFS pointers, not audio. Use experiments/hearside_pilot/get_cremad.sh"
log "CREMA-D: $clips wav files"
[ -f "$BASE_CONFIG" ] || stop "base config not found: $BASE_CONFIG"

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/ms_swift_librispeech/tests/test_update_positions.py \
        experiments/trr_diagnostics/tests experiments/hearside_pilot/tests > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

CROSS="$RUN/cremad"
if [ ! -f "$CROSS/crosscorpus.json" ]; then
    rm -rf "$CROSS"
    "$PY" -m experiments.hearside_pilot.crosscorpus build --audio "$CREMAD" --protocol "$PROTOCOL" --output "$CROSS" ${BATCH:+--batch-size "$BATCH"} \
        > "$RUN/logs/build.log" 2>&1 || stop "building the CREMA-D manifests failed; see logs/build.log: $(tail -1 "$RUN/logs/build.log")"
fi
[ -z "$BATCH" ] || [ "$(field "$CROSS/crosscorpus.json" mcq_generation)" = "{'batch_size': $BATCH}" ] \
    || stop "this run folder was built with another batch size; start a new folder instead of mixing them"
log "batch size: letter question $(field "$CROSS/crosscorpus.json" mcq_generation), free text $(field "$CROSS/crosscorpus.json" text_generation)"
log "manifests: $(tr -d '\n ' < "$RUN/logs/build.log" | cut -c1-300)"

summary() {
    "$PY" -m experiments.hearside_pilot.crosscorpus summarize --folder "$CROSS" --output "$RUN/summary" > "$RUN/logs/summary.log" 2>&1 \
        || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
}

started=$SECONDS
"$PY" -m experiments.hearside_pilot.crosscorpus run --folder "$CROSS" --model-config "$BASE_CONFIG" --method frozen-base \
    --arm frozen-base ${MCQ_BATCH:+--mcq-batch-size "$MCQ_BATCH"} > "$RUN/logs/run-frozen-base.log" 2>&1
log "frozen-base (exit $?, $(((SECONDS - started) / 60)) min)"
[ -f "$CROSS/runs/frozen-base/open/metrics.json" ] || stop "the frozen base could not be scored; see logs/run-frozen-base.log: $(tail -1 "$RUN/logs/run-frozen-base.log")"
for arm in $ARMS; do
    root=$(home "$arm") || { log "$arm has no trained checkpoint, skipped"; continue; }
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.crosscorpus run --folder "$CROSS" --model-config "$root/configs/$arm.json" --method ta-lora \
        --checkpoint "$(field "$root/train/$arm/result.json" best_checkpoint)" --arm "$arm" ${MCQ_BATCH:+--mcq-batch-size "$MCQ_BATCH"} > "$RUN/logs/run-$arm.log" 2>&1
    code=$?
    log "$arm (exit $code, $(((SECONDS - started) / 60)) min)"
    [ "$code" -eq 0 ] || stop "$arm could not be scored; see logs/run-$arm.log: $(tail -1 "$RUN/logs/run-$arm.log")"
    summary
done
summary
[ -f "$RUN/summary/crosscorpus.md" ] && cat "$RUN/summary/crosscorpus.md" | tee -a "$RUN/pilot.log"
log "done"
