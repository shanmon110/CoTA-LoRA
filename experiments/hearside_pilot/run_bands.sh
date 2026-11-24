#!/usr/bin/env bash
# Fifth run of the hear-side line (declared 2026-11-25): four trained layers at the start and in the middle of the
# stack, three seeds each, scored like every earlier arm on IEMOCAP Session 4 and on CREMA-D.
# See research/layer-bands-20261125.md.
#
#   nohup bash experiments/hearside_pilot/run_bands.sh > /root/autodl-fs/CoTA-LoRA/hearside-bands.out 2>&1 &
#
# The four earlier run folders are only read. Everything is written into RUN; running again with
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
CROSS="${CROSS:-$ROOT/hearside-cross-20261124-213539}"
RUN="${RUN:-$ROOT/hearside-bands-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$SRC/experiments/hearside_pilot/pilot-iemocap-session4.json"
ARMS="${ARMS:-A-first4 A-mid4 A-first4-s43 A-mid4-s43 A-first4-s44 A-mid4-s44}"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
for old in "$PILOT" "$FOLLOWUP" "$NEXT" "$CROSS"; do
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
          -size -40M ! -name 'tokenizer*' ! -name 'vocab.json' ! -path '*/checkpoint-*' -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "BANDS EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

log "run folder: $RUN"
log "read only: $PILOT, $FOLLOWUP, $NEXT, $CROSS"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$PROTOCOL" "$PILOT/data/train-fixed.jsonl" "$PILOT/data/dev-fixed.jsonl" "$PILOT/eval/manifests/orig.jsonl" \
            "$FOLLOWUP/eval/runs/A-last4/orig/predictions.jsonl" "$FOLLOWUP/train/A-last4/result.json" "$FOLLOWUP/configs/A-last4.json" \
            "$NEXT/formats/formats.json" "$CROSS/cremad/crosscorpus.json"; do
    [ -e "$path" ] || stop "missing $path"
done
first_audio=$("$PY" -c 'import json, sys; print(json.loads(open(sys.argv[1]).readline())["audios"][0])' "$CROSS/cremad/mcq.jsonl")
[ -f "$first_audio" ] || stop "the CREMA-D audio of the fourth run is no longer where it was: $first_audio"

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/ms_swift_librispeech/tests/test_layer_band.py \
        experiments/ms_swift_librispeech/tests/test_update_positions.py \
        experiments/ms_swift_librispeech/tests/test_adapter_pair.py \
        experiments/ms_swift_librispeech/tests/test_conditional_gate.py \
        experiments/trr_diagnostics/tests experiments/hearside_pilot/tests > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

# ------------------------------------------------------------------ configs; the training manifests are the pilot's own files
if [ ! -f "$RUN/configs/arms.json" ]; then
    rm -rf "$RUN/configs"
    "$PY" -m experiments.hearside_pilot.configs --base "$BASE_CONFIG" --data "$PILOT/data" --output "$RUN/configs" --plan layer-bands \
        > "$RUN/logs/configs.log" 2>&1 || stop "building the arm configs failed; see logs/configs.log"
fi

# ------------------------------------------------------------------ both corpora in one layout, with every earlier answer
S4="$RUN/session4"
if [ ! -f "$S4/crosscorpus.json" ]; then
    rm -rf "$S4"
    "$PY" -m experiments.hearside_pilot.bands assemble --evals "$PILOT/eval" "$FOLLOWUP/eval" "$NEXT/eval" --formats "$NEXT/formats" \
        --protocol "$PROTOCOL" --output "$S4" > "$RUN/logs/assemble.log" 2>&1 \
        || stop "gathering the Session 4 answers failed; see logs/assemble.log: $(tail -1 "$RUN/logs/assemble.log")"
fi
log "Session 4: $(tr -d '\n ' < "$RUN/logs/assemble.log" | cut -c1-330)"
CREMAD="$RUN/cremad"
if [ ! -f "$CREMAD/crosscorpus.json" ]; then
    rm -rf "$CREMAD" "$CREMAD.partial"
    cp -r "$CROSS/cremad" "$CREMAD.partial" && mv "$CREMAD.partial" "$CREMAD" || stop "copying the CREMA-D answers of the fourth run failed"
fi
log "CREMA-D: $(ls "$CREMAD/runs" | wc -l) earlier models, batch size $(field "$CREMAD/crosscorpus.json" mcq_generation)"

summary() {
    "$PY" -m experiments.hearside_pilot.bands summarize --session4 "$S4" --cremad "$CREMAD" --output "$RUN/summary" > "$RUN/logs/summary.log" 2>&1 \
        || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
}

# ------------------------------------------------------------------ does this route give an old arm its old answers?
if [ ! -f "$RUN/logs/recheck.ok" ]; then
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.crosscorpus run --folder "$S4" --model-config "$FOLLOWUP/configs/A-last4.json" --method ta-lora \
        --checkpoint "$(field "$FOLLOWUP/train/A-last4/result.json" best_checkpoint)" --arm recheck-A-last4 > "$RUN/logs/run-recheck.log" 2>&1
    [ -f "$S4/runs/recheck-A-last4/open/metrics.json" ] || stop "re-scoring A-last4 failed; see logs/run-recheck.log: $(tail -1 "$RUN/logs/run-recheck.log")"
    "$PY" -m experiments.hearside_pilot.bands same --folder "$S4" --arms A-last4 recheck-A-last4 > "$RUN/logs/recheck.log" 2>&1
    code=$?
    log "A-last4 scored again on Session 4 ($(((SECONDS - started) / 60)) min), rows answered differently from its earlier run: $(tail -1 "$RUN/logs/recheck.log")"
    [ "$code" -eq 0 ] || stop "more than 1% of the answers differ: the new route does not reproduce the earlier scoring, so old and new arms would not be comparable"
    touch "$RUN/logs/recheck.ok"
fi

# ------------------------------------------------------------------ train and score the six band arms
for arm in $ARMS; do
    out="$RUN/train/$arm"
    if [ ! -f "$out/result.json" ]; then
        [ -d "$out" ] && mv "$out" "$out.incomplete-$(date +%H%M%S)"      # an interrupted training is kept, not resumed
        "$PY" -m experiments.ms_swift_librispeech.train --config "$RUN/configs/$arm.json" --method ta-lora \
            --train "$PILOT/data/train-fixed.jsonl" --dev "$PILOT/data/dev-fixed.jsonl" --output "$out" > "$RUN/logs/train-$arm.log" 2>&1
        code=$?
        if [ ! -f "$out/result.json" ]; then
            log "$arm: training FAILED (exit $code); see logs/train-$arm.log: $(grep -E 'Error|error|Killed' "$RUN/logs/train-$arm.log" | tail -1)"
            continue
        fi
    fi
    log "$arm trained: best step $(field "$out/result.json" best_step), Session 3 UAR $(field "$out/result.json" best_uar), $(field "$out/result.json" training_seconds) s"
    for corpus in "$S4" "$CREMAD"; do
        started=$SECONDS
        "$PY" -m experiments.hearside_pilot.crosscorpus run --folder "$corpus" --model-config "$RUN/configs/$arm.json" --method ta-lora \
            --checkpoint "$(field "$out/result.json" best_checkpoint)" --arm "$arm" >> "$RUN/logs/run-$arm.log" 2>&1
        log "$arm on $(basename "$corpus") (exit $?, $(((SECONDS - started) / 60)) min)"
    done
    summary
done
summary
[ -f "$RUN/summary/bands.md" ] && cat "$RUN/summary/bands.md" | tee -a "$RUN/pilot.log"
log "done"
