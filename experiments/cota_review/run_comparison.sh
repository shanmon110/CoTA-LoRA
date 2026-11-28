#!/usr/bin/env bash
# The one training run this project actually still needs: the three-arm comparison.
#
# WHY THIS RUN EXISTS
#
# The source-level review (research/notes-20261128-source-review.md) established that every
# arm this project has trained so far carries `gate_multiplier: 0.0`, so the transport term is
# a provable no-op: the gate has no optimizer state in any of the forty-four checkpoints, and
# the forward is bit-identical whether the transport contrast is computed or skipped.
#
# So the project has measured WHERE TO PUT A LOW-RANK UPDATE, and has never measured the
# transport mechanism.  Nor has it ever run a matched LoRA: `experiments/cota_review/` holds
# configs for eight comparison arms and no results for any of them.
#
# This run trains the three arms that would close that gap:
#
#   lora-r12        the MATCHED LoRA: rank 12, alpha 24, scale 2, the same 36 O projections.
#                   This is the baseline the handoff names explicitly.
#   lora-r8         the PARAMETER-MATCHED control: rank 8, alpha 16, 1,179,648 trainable
#                   parameters against the 1,179,936 of the thirty-six-layer arm this project
#                   already trained.  The difference is exactly the thirty-six gates of eight.
#                   It separates "more parameters" from "the transport term".
#   cota-qa-live    the transport arm, in the configuration that makes it NUMERICALLY EFFECTIVE:
#                   gate_multiplier 1, the FP32 transport path, and a 16x gate learning rate.
#                   The FP32 path is not optional: the source's own comment says BF16 autocast
#                   erases any multiplier within 2^-9 of one.
#
# Everything else is held fixed at the AQA protocol that produced the existing arms, so the
# only variables are the adapter.  The data, the model revision, the seeds and the evaluation
# are the ones the existing `AQ-*` arms used.
#
# THE PRE-DECLARED READING (written before the run)
#
# On the corrected baseline (the label-free class-marginal correction of the frozen base, the
# same correction used throughout the analysis):
#
#   if cota-qa-live beats lora-r12 by MORE THAN 2 pp, the transport term does something a
#       matched LoRA cannot, and the project has a method contribution to make;
#   if it does not, the transport term buys nothing over a matched LoRA, the transport line is
#       CLOSED, and the paper is the analysis paper it already is.
#
# "No gain" is a legal and useful result.  It is the expected one, and it is recorded as such.
#
# COST: three arms at roughly 0.15-0.22 GPU hours each for training, plus the evaluation.
#
#   nohup bash experiments/cota_review/run_comparison.sh > /root/autodl-fs/CoTA-LoRA/cota-comparison.out 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
DEPS="${DEPS:-$ROOT/deps}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$ROOT/cota-comparison-$(date +%Y%m%d-%H%M%S)}"
AQA_RUN="${AQA_RUN:-$ROOT/hearside-aqa-20261125-224035}"
# the working AQA arm configs are the template: same protocol, only the adapter differs
TEMPLATE="${TEMPLATE:-$AQA_RUN/configs/AQ-all.json}"
ARMS="${ARMS:-lora-r12 lora-r8 cota-qa-live}"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
mkdir -p "$RUN/logs" "$RUN/configs" || exit 1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pilot.log"; }
pack() {
    local code=$?
    log "finished with exit code $code after $((SECONDS / 60)) min; packing the result files"
    ( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' \) \
          -size -40M ! -name 'tokenizer*' ! -name 'vocab.json' ! -path '*/checkpoint-*' -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "COTA-COMPARISON EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT
stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }

log "run folder: $RUN"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit')"
log "the pre-declared reading is in the header of this script and in research/notes-20261128-source-review.md"
for path in "$PY" "$TEMPLATE" "$AQA_RUN/train/AQ-all/result.json"; do
    [ -e "$path" ] || stop "missing $path"
done

# The three configs: the template's protocol, with only the adapter replaced.
if [ ! -f "$RUN/configs/lora-r12.json" ]; then
    "$PY" - "$TEMPLATE" "$RUN/configs" "$ARMS" <<'PYEOF' > "$RUN/logs/configs.log" 2>&1 \
        || stop "building the arm configs failed; see logs/configs.log"
import copy, json, pathlib, sys

template = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
arms = sys.argv[3].split()

# The adapters.  `gate_multiplier` 1 with the FP32 transport path and a 16x gate learning
# rate is the configuration the source's own comment says is required: under BF16 autocast a
# multiplier within 2^-9 of one is rounded away.
#
# The two LoRA arms carry no transport fields at all: the trainer reads them only under
# `method == 'ta-lora'`, and leaving them out keeps the configs honest about what was trained.
ADAPTERS = {
    'lora-r12': dict(method='lora', adapter=dict(rank=12, alpha=24, last_n=36)),
    'lora-r8': dict(method='lora', adapter=dict(rank=8, alpha=16, last_n=36)),
    'cota-qa-live': dict(method='ta-lora', adapter=dict(rank=8, alpha=16, last_n=36,
                                                        gate_init=0.0, gate_multiplier=1.0,
                                                        independent_transport_up=True,
                                                        transport_gate='question_audio',
                                                        transport_precision='fp32',
                                                        transport_gate_lr_multiplier=16,
                                                        transport_queries='text',
                                                        transport_weighting='uniform')),
}

for arm in arms:
    if arm not in ADAPTERS:
        raise SystemExit(f'unknown arm {arm!r}; the three are {sorted(ADAPTERS)}')
    spec = ADAPTERS[arm]
    cfg = copy.deepcopy(template)
    cfg['adapter'] = spec['adapter']
    # the method is chosen on the command line, but the config's method list must allow it
    methods = set(cfg.get('methods', []))
    methods.add(spec['method'])
    cfg['methods'] = sorted(methods)
    out.joinpath(f'{arm}.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(f'wrote {out / (arm + ".json")}')
PYEOF
    log "configs: $(tr -d '\n' < "$RUN/logs/configs.log" | cut -c1-300)"
fi

# The manifests: the same ones the existing AQA arms used.
TRAIN="${TRAIN:-/root/autodl-tmp/r1-aqa/data/pilot-01/train.jsonl}"
DEV="${DEV:-/root/autodl-tmp/r1-aqa/data/pilot-01/dev.jsonl}"
for path in "$TRAIN" "$DEV"; do
    [ -e "$path" ] || stop "missing $path"
done

for arm in $ARMS; do
    out="$RUN/train/$arm"
    if [ ! -f "$out/result.json" ]; then
        [ -d "$out" ] && mv "$out" "$out.incomplete-$(date +%H%M%S)"   # an interrupted training is kept, not resumed
        started=$SECONDS
        method=$("$PY" -c 'import json, sys; a=json.load(open(sys.argv[1]))["adapter"]; print("ta-lora" if "transport_gate" in a else "lora")' "$RUN/configs/$arm.json")
        "$PY" -m experiments.ms_swift_librispeech.train --config "$RUN/configs/$arm.json" --method "$method" \
            --train "$TRAIN" --dev "$DEV" --output "$out" > "$RUN/logs/train-$arm.log" 2>&1
        code=$?
        if [ ! -f "$out/result.json" ]; then
            log "$arm: training FAILED (exit $code); see logs/train-$arm.log: $(grep -E 'Error|error|Killed' "$RUN/logs/train-$arm.log" | tail -1)"
            continue
        fi
        log "$arm trained ($method) in $(((SECONDS - started) / 60)) min; best step $(field "$out/result.json" best_step)"
    else
        log "$arm is already trained"
    fi
done

# What moved, and whether the gate was stepped at all.  This is the check that says whether
# the transport term was actually live in the run.
for arm in $ARMS; do
    [ -f "$RUN/train/$arm/result.json" ] || continue
    "$PY" research/tools/gate_optimizer_state.py "$RUN" > "$RUN/logs/gate-state-$arm.log" 2>&1
    log "$arm gate state: $(grep -E "gate w/ state" "$RUN/logs/gate-state-$arm.log" | tail -1)"
done

# ------------------------------------------------------------------ score the arms on the same MMAU test
# The existing `AQ-*` arms were scored on MMAU Test-mini.  The three new arms are scored on the
# SAME manifests with the same code, so the comparison is like for like -- but in a NEW folder.
# The historical run folder is never written to: its manifest digests and its summary stay
# exactly as they are.  The new folder copies the manifests and the manifest digest file, and
# the frozen base's answers are carried over by reference so that the summary can contrast
# against it.
if [ -f "$AQA_RUN/mmau/aqa.json" ]; then
    MM="$RUN/mmau"
    if [ ! -d "$MM" ]; then
        mkdir -p "$MM/runs"
        cp "$AQA_RUN/mmau/aqa.json" "$MM/aqa.json"
        for task in mcq text; do
            [ -e "$AQA_RUN/mmau/$task.jsonl" ] && cp "$AQA_RUN/mmau/$task.jsonl" "$MM/$task.jsonl"
        done
        # the frozen base's answers are copied in, so the summary's contrasts are complete
        for task in mcq text; do
            [ -d "$AQA_RUN/mmau/runs/frozen-base/$task" ] && cp -r "$AQA_RUN/mmau/runs/frozen-base" "$MM/runs/frozen-base"
        done
        log "the scoring folder is $MM; the historical $AQA_RUN/mmau is read only"
    fi
    for arm in $ARMS; do
        [ -f "$RUN/train/$arm/result.json" ] || continue
        method=$("$PY" -c 'import json, sys; a=json.load(open(sys.argv[1]))["adapter"]; print("ta-lora" if "transport_gate" in a else "lora")' "$RUN/configs/$arm.json")
        started=$SECONDS
        "$PY" -m experiments.hearside_pilot.aqa run --folder "$MM" \
            --model-config "$RUN/configs/$arm.json" --method "$method" \
            --checkpoint "$(field "$RUN/train/$arm/result.json" best_checkpoint)" --arm "$arm" \
            > "$RUN/logs/score-$arm.log" 2>&1
        code=$?
        log "$arm on MMAU (exit $code, $(((SECONDS - started) / 60)) min)"
        [ "$code" -eq 0 ] || log "$arm could not be scored on MMAU; see logs/score-$arm.log: $(tail -1 "$RUN/logs/score-$arm.log")"
    done
    "$PY" -m experiments.hearside_pilot.aqa summarize --folder "$MM" --output "$RUN/summary" \
        > "$RUN/logs/summary.log" 2>&1 \
        && log "summary written to $RUN/summary/aqa.md" \
        || log "the summary failed; see logs/summary.log: $(tail -1 "$RUN/logs/summary.log")"
else
    log "the MMAU folder is not at $AQA_RUN/mmau; the arms are trained but not scored"
fi

log "done"
