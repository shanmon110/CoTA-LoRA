#!/usr/bin/env bash
# P5: the three-parameter surrogate test, generalised to AQA.
#
# On IEMOCAP the surrogate is the repository's own method: one constant per option letter, added
# to the frozen base's option logits and fitted on the training split, which reconstructs 79% of
# the last-four-layers arm's training-order gain.  That shows that a 131,104-parameter adapter
# behaved, in the training order, like a three-degree-of-freedom output bias.
#
# P5 turns that into a general protocol: can the behaviour of ANY adapter be reconstructed by a
# few degrees of freedom?  AQA has no fixed class-to-letter correspondence, so the surrogate has
# to take a different form there: one constant per SLOT rather than per letter, and the score is
# reconstructed from the adapter's own logits under a per-slot rule fitted on half the questions.
#
# The pre-declared reading (research/notes-20261127-diagnosis-and-plan.md, P5):
#
#   if an affine rule on the arm's OWN option logits, fitted on one half of the questions and
#   scored on the other, reproduces most of the arm's gain over the frozen base, then the arm's
#   gain is a decision-rule artefact and the adapter learned a readout offset rather than
#   something about the audio;
#   if it does not, the arm's gain is a change of the representation.
#
# This needs the option LOGITS on AQA, which are not tracked, so this step extracts them.
# Inference only.  Nothing is trained.
#
#   nohup bash experiments/hearside_pilot/run_aqa_surrogate.sh > /root/autodl-fs/CoTA-LoRA/aqa-surrogate.out 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
DEPS="${DEPS:-$ROOT/deps}"
AQA_RUN="${AQA_RUN:-$ROOT/hearside-aqa-20261125-224035}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$ROOT/hearside-aqa-surrogate-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${BASE_CONFIG:-$SRC/experiments/ms_swift_aqa/omni3b-pilot.json}"
ARMS="${ARMS:-frozen-base AQ-all AQ-last4 AQ-first4 AQ-mid4}"
SIZE="${SIZE:-1000}"
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
    echo "AQA-SURROGATE EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT
stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(next((d[k] for k in sys.argv[2:] if k in d), "n/a"))' "$@"; }

log "run folder: $RUN"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit')"

# The manifests: the same questions as the AQA run, so the numbers are comparable with the
# tracked MMAU ones, plus the option logits.
if [ ! -f "$RUN/aqa/aqa.json" ]; then
    "$PY" -m experiments.hearside_pilot.aqa build --test /root/autodl-tmp/r1-aqa/data/pilot-01/test.jsonl \
        --output "$RUN/aqa" > "$RUN/logs/build.log" 2>&1 \
        || stop "building the manifests failed; see logs/build.log: $(tail -1 "$RUN/logs/build.log")"
    log "manifests: $(tr -d '\n' < "$RUN/logs/build.log" | cut -c1-300)"
fi

# The option logits, which the surrogate needs.  The extraction writes one .npy per arm.
if [ ! -f "$RUN/logits/frozen-base/mcq/done.json" ]; then
    "$PY" -m experiments.hearside_pilot.aqa_logits --folder "$RUN/aqa" --model-config "$BASE_CONFIG" \
        --method frozen-base --arm frozen-base --output "$RUN/logits" \
        > "$RUN/logs/logits-frozen-base.log" 2>&1 \
        || stop "extracting the frozen base's option logits failed; see logs/logits-frozen-base.log: $(tail -1 "$RUN/logs/logits-frozen-base.log")"
    log "frozen-base option logits extracted"
fi

for arm in $ARMS; do
    [ "$arm" = frozen-base ] && continue
    out="$AQA_RUN/train/$arm"
    [ -f "$out/result.json" ] || { log "$arm has no trained checkpoint, skipped"; continue; }
    [ -f "$RUN/logits/$arm/mcq/done.json" ] && { log "$arm already extracted"; continue; }
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.aqa_logits --folder "$RUN/aqa" \
        --model-config "$AQA_RUN/configs/$arm.json" --method ta-lora \
        --checkpoint "$(field "$out/result.json" best_checkpoint)" --arm "$arm" --output "$RUN/logits" \
        > "$RUN/logs/logits-$arm.log" 2>&1
    code=$?
    log "$arm option logits (exit $code, $(((SECONDS - started) / 60)) min)"
    [ "$code" -eq 0 ] || log "$arm could not be scored; see logs/logits-$arm.log: $(tail -1 "$RUN/logs/logits-$arm.log")"
done

"$PY" research/tools/aqa_surrogate.py --root "$RUN/logits" --output "$RUN/summary" \
    > "$RUN/logs/analysis.log" 2>&1 \
    || log "the analysis failed; see logs/analysis.log: $(tail -1 "$RUN/logs/analysis.log")"

[ -f "$RUN/summary/aqa-surrogate.md" ] && cat "$RUN/summary/aqa-surrogate.md" | tee -a "$RUN/pilot.log"
log "done"
