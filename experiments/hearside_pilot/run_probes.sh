#!/usr/bin/env bash
# Linear probes of the emotion inside the frozen base and three adapters (declared 2026-11-25).
# See research/probes-20261125.md. Nothing is trained except the probes themselves.
#
#   nohup bash experiments/hearside_pilot/run_probes.sh > /root/autodl-fs/CoTA-LoRA/hearside-probes.out 2>&1 &
#
# Waits while run_bands.sh is running, so the two do not share the GPU. Earlier run folders are only read.
# The extracted states (about 7 GB) go to STATES on the local disk; small results go to RUN.
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
RUN="${RUN:-$ROOT/hearside-probes-$(date +%Y%m%d-%H%M%S)}"
STATES="${STATES:-/root/autodl-tmp/$(basename "$RUN")-states}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$SRC/experiments/hearside_pilot/pilot-iemocap-session4.json"
MODELS="${MODELS:-frozen-base A-all D-all-shuffled A-last4}"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$RUN/logs" "$STATES" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }

pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the small result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.md' -o -name '*.log' -o -name '*.txt' -o -name '*.npz' \) \
          -size -40M -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "PROBES EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

log "run folder: $RUN; states: $STATES"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$PROTOCOL" "$PILOT/data/train-fixed.jsonl" "$PILOT/data/dev-fixed.jsonl" "$PILOT/eval/manifests/orig.jsonl" \
            "$NEXT/formats/formats.json" "$CROSS/cremad/mcq.jsonl"; do
    [ -e "$path" ] || stop "missing $path"
done
free=$(df -Pk "$STATES" | tail -1 | awk '{print int($4 / 1048576)}')
[ "$free" -ge 10 ] || stop "only $free GB free under $STATES; the states need about 7 GB. Set STATES to a folder on a larger disk"

waited=0
while pgrep -f 'hearside_pilot/run_bands.sh' > /dev/null; do
    [ "$waited" -eq 0 ] && log "run_bands.sh is running; waiting for it to finish before using the GPU"
    waited=1
    sleep 60
done
[ "$waited" -eq 1 ] && log "run_bands.sh has finished"

if [ ! -f "$RUN/logs/tests-gating.ok" ]; then
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/hearside_pilot/tests/test_probes.py experiments/hearside_pilot/tests/test_bands.py > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

# ------------------------------------------------------------------ what to ask, and the answers already given
mkdir -p "$RUN/manifests"
for item in "train training $PILOT/data/train-fixed.jsonl" "dev training $PILOT/data/dev-fixed.jsonl" \
            "session4 evaluation $PILOT/eval/manifests/orig.jsonl" "cremad evaluation $CROSS/cremad/mcq.jsonl"; do
    set -- $item
    [ -f "$RUN/manifests/$1.jsonl" ] || "$PY" -m experiments.hearside_pilot.probes manifest --source "$3" --kind "$2" --output "$RUN/manifests/$1.jsonl" \
        >> "$RUN/logs/manifests.log" 2>&1 || stop "building the $1 manifest failed; see logs/manifests.log"
done
log "utterances: $(wc -l "$RUN"/manifests/*.jsonl | head -4 | awk '{printf "%s %s; ", $2, $1}' | sed "s|$RUN/manifests/||g")"
S4="$RUN/session4"
if [ ! -f "$S4/crosscorpus.json" ]; then
    rm -rf "$S4"
    "$PY" -m experiments.hearside_pilot.bands assemble --evals "$PILOT/eval" "$FOLLOWUP/eval" "$NEXT/eval" --formats "$NEXT/formats" \
        --protocol "$PROTOCOL" --output "$S4" > "$RUN/logs/assemble.log" 2>&1 || stop "gathering the Session 4 answers failed; see logs/assemble.log"
fi

summary() {
    "$PY" -m experiments.hearside_pilot.probes summarize --root "$STATES" --session4 "$S4" --cremad "$CROSS/cremad" --output "$RUN/summary" \
        > "$RUN/logs/summary.log" 2>&1 || log "summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
}

# ------------------------------------------------------------------ states and probes, one model at a time
for model in $MODELS; do
    if [ "$model" = frozen-base ]; then
        options=(--model-config "$BASE_CONFIG" --method frozen-base)
    else
        home=""
        for root in "$FOLLOWUP" "$PILOT"; do [ -f "$root/train/$model/result.json" ] && { home=$root; break; }; done
        [ -n "$home" ] || { log "$model has no trained checkpoint, skipped"; continue; }
        options=(--model-config "$home/configs/$model.json" --method ta-lora --checkpoint "$(field "$home/train/$model/result.json" best_checkpoint)")
    fi
    started=$SECONDS
    failed=""
    for split in train dev session4 cremad; do
        "$PY" -m experiments.hearside_pilot.probes extract --manifest "$RUN/manifests/$split.jsonl" "${options[@]}" \
            --output "$STATES/$model/$split" >> "$RUN/logs/extract-$model.log" 2>&1
        [ -f "$STATES/$model/$split/done.json" ] || { failed=$split; break; }
    done
    if [ -n "$failed" ]; then
        log "$model: extraction FAILED on $failed; see logs/extract-$model.log: $(grep -E 'Error|error|Killed' "$RUN/logs/extract-$model.log" | tail -1)"
        [ "$model" = frozen-base ] && stop "without the frozen base there is nothing to compare"
        continue
    fi
    log "$model: states extracted ($(((SECONDS - started) / 60)) min); answers differing from the earlier Session 4 scoring: $("$PY" -m experiments.hearside_pilot.probes agree --states "$STATES/$model/session4" --folder "$S4" --arm "$model" 2>&1 | tail -1)"
    if [ "$model" = A-last4 ]; then
        log "A-last4: largest difference from the frozen base in the states below its trained layers: $("$PY" -m experiments.hearside_pilot.probes unchanged --root "$STATES" --model A-last4 --last-layer 32 2>&1 | tail -1)"
    fi
    started=$SECONDS
    if [ ! -f "$STATES/$model/probes/probes.json" ]; then
        "$PY" -m experiments.hearside_pilot.probes fit --root "$STATES/$model" --output "$STATES/$model/probes" > "$RUN/logs/fit-$model.log" 2>&1 \
            || { log "$model: fitting the probes FAILED; see logs/fit-$model.log: $(tail -1 "$RUN/logs/fit-$model.log")"; continue; }
    fi
    mkdir -p "$RUN/probes/$model" && cp "$STATES/$model/probes/probes.json" "$STATES/$model/probes/correct.npz" "$RUN/probes/$model/"
    for split in session4 cremad; do mkdir -p "$RUN/rows/$model" && cp "$STATES/$model/$split/rows.jsonl" "$RUN/rows/$model/$split.json"; done
    log "$model: probes fitted ($(((SECONDS - started) / 60)) min); chosen layers $(tail -1 "$RUN/logs/fit-$model.log")"
    summary
done
summary
[ -f "$RUN/summary/probes.md" ] && cat "$RUN/summary/probes.md" | tee -a "$RUN/pilot.log"
log "done; the states in $STATES can be deleted once the results are checked"
