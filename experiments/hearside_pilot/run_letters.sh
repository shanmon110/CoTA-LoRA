#!/usr/bin/env bash
# Letter logits of the frozen base and seven adapters (declared 2026-11-25); see research/letter-bias-20261125.md.
# Nothing is trained. Earlier run folders are only read; everything written is small and goes into RUN.
#
#   nohup bash experiments/hearside_pilot/run_letters.sh > /root/autodl-fs/CoTA-LoRA/hearside-letters.out 2>&1 &
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
BANDS="${BANDS:-$ROOT/hearside-bands-20261125-045245}"
RUN="${RUN:-$ROOT/hearside-letters-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS="${MODELS:-frozen-base A-last4 A-last4-s43 A-last4-s44 A-all A-first4 A-mid4 D-all-shuffled}"
BATCH="${BATCH:-4}"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$RUN/logs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }

pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' -o -name '*.npy' \) \
          -size -40M -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "LETTERS EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

log "run folder: $RUN; batch size $BATCH"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$PILOT/data/train-fixed.jsonl" "$PILOT/eval/manifests/orig.jsonl" "$CROSS/cremad/mcq.jsonl"; do
    [ -e "$path" ] || stop "missing $path"
done
while pgrep -f 'hearside_pilot/run_(bands|probes).sh' > /dev/null; do
    [ -z "${waited:-}" ] && log "another hear-side run is using the GPU; waiting for it to finish"
    waited=1
    sleep 60
done

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/hearside_pilot/tests/test_letters.py experiments/hearside_pilot/tests/test_conflict.py > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

analysis() {
    "$PY" -m experiments.hearside_pilot.letters analyze --root "$RUN/logits" --output "$RUN/summary" > "$RUN/logs/analysis.log" 2>&1 \
        || log "analysis failed; see logs/analysis.log: $(tail -1 "$RUN/logs/analysis.log")"
}

for model in $MODELS; do
    if [ "$model" = frozen-base ]; then
        options=(--model-config "$BASE_CONFIG" --method frozen-base)
        splits="train session4 cremad"
    else
        home=""
        for root in "$BANDS" "$NEXT" "$FOLLOWUP" "$PILOT"; do [ -f "$root/train/$model/result.json" ] && { home=$root; break; }; done
        [ -n "$home" ] || { log "$model has no trained checkpoint, skipped"; continue; }
        options=(--model-config "$home/configs/$model.json" --method ta-lora --checkpoint "$(field "$home/train/$model/result.json" best_checkpoint)")
        splits="session4 cremad"
    fi
    started=$SECONDS
    failed=""
    for split in $splits; do
        case $split in
            train) source="$PILOT/data/train-fixed.jsonl"; kind=training ;;
            session4) source="$PILOT/eval/manifests/orig.jsonl"; kind=evaluation ;;
            cremad) source="$CROSS/cremad/mcq.jsonl"; kind=evaluation ;;
        esac
        "$PY" -m experiments.hearside_pilot.letters extract --source "$source" --kind "$kind" "${options[@]}" \
            --output "$RUN/logits/$model/$split" --batch-size "$BATCH" >> "$RUN/logs/extract-$model.log" 2>&1
        [ -f "$RUN/logits/$model/$split/done.json" ] || { failed=$split; break; }
    done
    if [ -n "$failed" ]; then
        log "$model: FAILED on $failed; see logs/extract-$model.log: $(grep -E 'Error|error|Killed' "$RUN/logs/extract-$model.log" | tail -1)"
        [ "$model" = frozen-base ] && stop "without the frozen base there is nothing to analyse"
        continue
    fi
    log "$model: letter logits extracted ($(((SECONDS - started) / 60)) min)"
    [ "$model" = frozen-base ] || analysis
done
analysis
[ -f "$RUN/summary/letters.md" ] && cat "$RUN/summary/letters.md" | tee -a "$RUN/pilot.log"
log "done"
