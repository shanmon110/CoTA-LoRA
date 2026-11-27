#!/usr/bin/env bash
# Session 5 confirmation of the class-marginal correction (P1').
#
# Nothing is trained.  The frozen base and the seven adapters are scored on IEMOCAP Session 5
# under the four option orders, and the label-free class-marginal correction is then applied to
# the resulting logits.  Session 5 has never been read by the hear-side line; it is named as the
# confirmation split in experiments/hearside_pilot/README.md line 296.
#
# The pre-declared criteria are in research/notes-20261127-diagnosis-and-plan.md, P1'.
#
#   nohup bash experiments/hearside_pilot/run_session5.sh > /root/autodl-fs/CoTA-LoRA/hearside-session5.out 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
IEMOCAP="${IEMOCAP:-/root/autodl-tmp/iemocap/swift-ser}"
IEMOCAP_ARCHIVE="${IEMOCAP_ARCHIVE:-/root/autodl-tmp/iemocap/iemocapfullrelease-v1.zip}"
BASE_CONFIG="${BASE_CONFIG:-$IEMOCAP/pair-01/config.json}"
DEPS="${DEPS:-$ROOT/deps}"
PILOT="${PILOT:-$ROOT/hearside-pilot-20261123-160928}"
FOLLOWUP="${FOLLOWUP:-$ROOT/hearside-followup-20261123-212622}"
NEXT="${NEXT:-$ROOT/hearside-next-20261124-031913}"
BANDS="${BANDS:-$ROOT/hearside-bands-20261125-045245}"
RUN="${RUN:-$ROOT/hearside-session5-$(date +%Y%m%d-%H%M%S)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS="${MODELS:-frozen-base A-last4 A-last4-s43 A-last4-s44 A-all A-first4 A-mid4 D-all-shuffled}"
BATCH="${BATCH:-4}"
LIMIT="${LIMIT:-0}"            # 0 means every utterance; a smaller number is a smoke run
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
    echo "SESSION5 EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

log "run folder: $RUN; batch size $BATCH; limit $LIMIT"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$IEMOCAP/data/test.jsonl" "$IEMOCAP_ARCHIVE"; do
    [ -e "$path" ] || stop "missing $path"
done
while pgrep -f 'hearside_pilot/run_(letters|bands|probes).sh' > /dev/null; do
    [ -z "${waited:-}" ] && log "another hear-side run is using the GPU; waiting for it to finish"
    waited=1
    sleep 60
done

# The Session 5 manifest: the same four option orders and the same question as Session 4.
if [ ! -f "$RUN/manifest/orig.jsonl" ]; then
    "$PY" -m experiments.hearside_pilot.session5_manifest --output "$RUN/manifest" > "$RUN/logs/manifest.log" 2>&1 \
        || stop "building the Session 5 manifest failed; see logs/manifest.log: $(tail -1 "$RUN/logs/manifest.log")"
    log "manifest: $(tr -d '\n ' < "$RUN/logs/manifest.log" | cut -c1-300)"
fi

# The gates are the unit tests that cover the code this run uses.  They are run on the same
# interpreter and environment as the run itself.
if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/hearside_pilot/tests/test_letters.py experiments/hearside_pilot/tests/test_conflict.py \
        > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

# The confirmation itself: the label-free class-marginal correction on every arm, with the
# order gap before and after.  This is the measurement the pre-declared criteria are read off.
analysis() {
    "$PY" -m experiments.hearside_pilot.session5 analyze --root "$RUN/logits" --output "$RUN/summary" \
        > "$RUN/logs/analysis.log" 2>&1 \
        || log "analysis failed; see logs/analysis.log: $(tail -1 "$RUN/logs/analysis.log")"
}

for model in $MODELS; do
    if [ "$model" = frozen-base ]; then
        options=(--model-config "$BASE_CONFIG" --method frozen-base)
    else
        home=""
        for root in "$BANDS" "$NEXT" "$FOLLOWUP" "$PILOT"; do [ -f "$root/train/$model/result.json" ] && { home=$root; break; }; done
        [ -n "$home" ] || { log "$model has no trained checkpoint, skipped"; continue; }
        options=(--model-config "$home/configs/$model.json" --method ta-lora --checkpoint "$(field "$home/train/$model/result.json" best_checkpoint)")
    fi
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.letters extract --source "$RUN/manifest/orig.jsonl" --kind evaluation \
        "${options[@]}" --output "$RUN/logits/$model/session5" --batch-size "$BATCH" \
        >> "$RUN/logs/extract-$model.log" 2>&1
    if [ ! -f "$RUN/logits/$model/session5/done.json" ]; then
        log "$model: FAILED; see logs/extract-$model.log: $(grep -E 'Error|error|Killed' "$RUN/logs/extract-$model.log" | tail -1)"
        [ "$model" = frozen-base ] && stop "without the frozen base there is nothing to analyse"
        continue
    fi
    log "$model: letter logits extracted ($(((SECONDS - started) / 60)) min)"
    analysis
done
analysis
[ -f "$RUN/summary/session5.md" ] && cat "$RUN/summary/session5.md" | tee -a "$RUN/pilot.log"
log "done"
