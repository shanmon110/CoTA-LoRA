#!/usr/bin/env bash
# Third run of the hear-side line (declared 2026-11-24):
#   1. two more seeds of the last-four-layer arm
#   2. every trained model answered in three formats it was not trained on (formats.py)
#   3. the headroom test for input-conditional adaptation (headroom.py)
#
#   nohup bash experiments/hearside_pilot/run_next.sh > /root/autodl-fs/CoTA-LoRA/hearside-next.out 2>&1 &
#
# The pilot and follow-up folders are only read. Everything is written into RUN; running again with
# RUN=<the same folder> continues where it stopped. STAGES="1 2 3" selects the parts.
set -uo pipefail

ROOT="${ROOT:-/root/autodl-fs/CoTA-LoRA}"
PY="${PY:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}"
IEMOCAP="${IEMOCAP:-/root/autodl-tmp/iemocap/swift-ser}"
BASE_CONFIG="${BASE_CONFIG:-$IEMOCAP/pair-01/config.json}"
DEPS="${DEPS:-$ROOT/deps}"
PILOT="${PILOT:-$ROOT/hearside-pilot-20261123-160928}"
FOLLOWUP="${FOLLOWUP:-$ROOT/hearside-followup-20261123-212622}"
RUN="${RUN:-$ROOT/hearside-next-$(date +%Y%m%d-%H%M%S)}"
STAGES="${STAGES:-1 2 3}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTOCOL="$SRC/experiments/hearside_pilot/pilot-iemocap-session4.json"
CONDITIONS="orig orig_reworded text question_only asr-orig"
FORMAT_ARMS="D-all-shuffled D-all-shuffled-s43 D-all-shuffled-s44 A-all A-all-s43 A-all-s44 B-audio B-audio-s43 B-audio-s44 A-last4 A-last4-s43 A-last4-s44 C-post-audio"
SPECIALISTS="gender-F gender-M random2-part0 random2-part1 duration-short duration-medium duration-long random3-part0 random3-part1 random3-part2 scripted-impro scripted-script"
ARCHIVE="${RUN%/}-for-claude.tar.gz"

export PYTHONPATH="$SRC:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 TOKENIZERS_PARALLELISM=false
cd "$SRC" || exit 1
for old in "$PILOT" "$FOLLOWUP"; do
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
          -size -40M ! -path './eval/audio/*' ! -name 'tokenizer*' ! -name 'vocab.json' ! -name 'added_tokens.json' \
          ! -name 'merges.txt' ! -name 'special_tokens_map.json' -print0 | tar czf "$ARCHIVE" --null -T - ) \
        && echo "ARCHIVE: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))" | tee -a "$RUN/pilot.log"
    echo "NEXT EXIT $code" | tee -a "$RUN/pilot.log"
}
trap pack EXIT

stop() { log "STOPPED: $*"; exit 1; }
field() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
home() {   # the run folder that trained an arm
    for root in "$RUN" "$FOLLOWUP" "$PILOT"; do
        [ -f "$root/train/$1/result.json" ] && { echo "$root"; return 0; }
    done
    return 1
}
train() {  # config, train manifest, dev manifest, output folder, log name; 0 when a result exists
    local config=$1 train=$2 dev=$3 out=$4 name=$5
    if [ ! -f "$out/result.json" ]; then
        [ -d "$out" ] && mv "$out" "$out.incomplete-$(date +%H%M%S)"      # an interrupted training is kept, not resumed
        "$PY" -m experiments.ms_swift_librispeech.train --config "$config" --method ta-lora \
            --train "$train" --dev "$dev" --output "$out" > "$RUN/logs/train-$name.log" 2>&1
        local code=$?
        if [ ! -f "$out/result.json" ]; then
            log "$name: training FAILED (exit $code); see logs/train-$name.log: $(grep -E 'Error|error|Killed' "$RUN/logs/train-$name.log" | tail -1)"
            return 1
        fi
    fi
    log "$name trained: best step $(field "$out/result.json" best_step), Session 3 UAR $(field "$out/result.json" best_uar), $(field "$out/result.json" training_seconds) s"
}

# ------------------------------------------------------------------ 0. what is being run, and on what
log "run folder: $RUN; stages: $STAGES"
log "read only: $PILOT and $FOLLOWUP"
log "source: $SRC at $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo 'unknown commit') $(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null | head -1 | sed 's/.*/(tracked files modified)/')"
for path in "$PY" "$BASE_CONFIG" "$IEMOCAP/data/train.jsonl" "$PROTOCOL" "$PILOT/eval/prepare.json" "$PILOT/configs/D-all-shuffled.json" "$FOLLOWUP/eval/prepare.json"; do
    [ -e "$path" ] || stop "missing $path"
done
for arm in A-all B-audio C-post-audio D-all-shuffled; do [ -f "$PILOT/train/$arm/result.json" ] || stop "the pilot folder has no trained $arm"; done
for arm in A-last4 A-all-s43 A-all-s44 B-audio-s43 B-audio-s44 D-all-shuffled-s43 D-all-shuffled-s44; do
    [ -f "$FOLLOWUP/train/$arm/result.json" ] || stop "the follow-up folder has no trained $arm"
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

if [ -f "$RUN/logs/tests-gating.ok" ]; then
    log "unit tests: passed in an earlier start of this run"
else
    env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider \
        experiments/ms_swift_librispeech/tests/test_update_positions.py \
        experiments/ms_swift_librispeech/tests/test_adapter_pair.py \
        experiments/ms_swift_librispeech/tests/test_conditional_gate.py \
        experiments/trr_diagnostics/tests experiments/hearside_pilot/tests > "$RUN/logs/tests-gating.log" 2>&1 \
        || stop "unit tests failed; nothing was run. See logs/tests-gating.log: $(tail -1 "$RUN/logs/tests-gating.log")"
    log "unit tests: $(tail -1 "$RUN/logs/tests-gating.log")"
    touch "$RUN/logs/tests-gating.ok"
fi

# ------------------------------------------------------------------ the same manifests and Session 4 inputs as the pilot
if [ ! -f "$RUN/data/splits.json" ]; then
    rm -rf "$RUN/data"
    "$PY" -m experiments.hearside_pilot.data --source "$IEMOCAP/data/train.jsonl" --protocol "$PROTOCOL" --output "$RUN/data" \
        > "$RUN/logs/data.log" 2>&1 || stop "building the training manifests failed; see logs/data.log"
fi
if [ ! -f "$RUN/configs/arms.json" ]; then
    rm -rf "$RUN/configs"
    "$PY" -m experiments.hearside_pilot.configs --base "$BASE_CONFIG" --data "$RUN/data" --output "$RUN/configs" --plan last4-seeds \
        > "$RUN/logs/configs.log" 2>&1 || stop "building the arm configs failed; see logs/configs.log"
fi
if [ ! -f "$RUN/eval/prepare.json" ]; then
    rm -rf "$RUN/eval"
    "$PY" -m experiments.trr_diagnostics.d2 prepare --config "$PROTOCOL" --output "$RUN/eval" --workers 4 \
        > "$RUN/logs/prepare.log" 2>&1 || stop "preparing the Session 4 evaluation failed; see logs/prepare.log"
fi
"$PY" -m experiments.hearside_pilot.followup check --pilot-run "$PILOT" --run "$RUN" > "$RUN/logs/inputs.log" 2>&1 \
    || stop "the inputs differ from the pilot's: $(tail -1 "$RUN/logs/inputs.log")"
log "inputs: $(tail -1 "$RUN/logs/inputs.log")"

followup_summary() {
    "$PY" -m experiments.hearside_pilot.followup summarize --pilot "$PILOT/eval" --prepared "$FOLLOWUP/eval" "$RUN/eval" \
        --output "$RUN/summary" > "$RUN/logs/summary-followup.log" 2>&1 \
        || log "follow-up summary failed; see logs/summary-followup.log: $(tail -1 "$RUN/logs/summary-followup.log")"
}

# ------------------------------------------------------------------ 1. two more seeds of the last-four-layer arm
case " $STAGES " in *" 1 "*)
    for arm in A-last4-s43 A-last4-s44; do
        log "$arm: training on train-fixed.jsonl"
        train "$RUN/configs/$arm.json" "$RUN/data/train-fixed.jsonl" "$RUN/data/dev-fixed.jsonl" "$RUN/train/$arm" "$arm" || continue
        "$PY" -m experiments.trr_diagnostics.d2 run --prepared "$RUN/eval" --model-config "$RUN/configs/$arm.json" --method ta-lora \
            --checkpoint "$(field "$RUN/train/$arm/result.json" best_checkpoint)" --arm "$arm" --only $CONDITIONS >> "$RUN/logs/eval-$arm.log" 2>&1
        log "$arm evaluated (exit $?): $(grep -E '^(orig|orig_reworded): ' "$RUN/logs/eval-$arm.log" | tr '\n' ';')"
    done
;; esac
followup_summary
[ -f "$RUN/summary/followup.md" ] && log "follow-up summary written"

# ------------------------------------------------------------------ 2. formats no arm was trained on
case " $STAGES " in *" 2 "*)
    if [ ! -f "$RUN/formats/formats.json" ]; then
        rm -rf "$RUN/formats"
        "$PY" -m experiments.hearside_pilot.formats build --manifest "$RUN/eval/manifests/orig.jsonl" --output "$RUN/formats" \
            > "$RUN/logs/formats-build.log" 2>&1 || stop "building the format manifests failed; see logs/formats-build.log"
    fi
    formats_summary() {
        "$PY" -m experiments.hearside_pilot.formats summarize --folder "$RUN/formats" --followup "$RUN/summary/followup.json" \
            --output "$RUN/summary" > "$RUN/logs/summary-formats.log" 2>&1 \
            || log "format summary failed; see logs/summary-formats.log: $(tail -1 "$RUN/logs/summary-formats.log")"
    }
    started=$SECONDS
    "$PY" -m experiments.hearside_pilot.formats run --folder "$RUN/formats" --model-config "$BASE_CONFIG" --method frozen-base \
        --arm frozen-base > "$RUN/logs/formats-frozen-base.log" 2>&1
    log "formats: frozen-base (exit $?, $(((SECONDS - started) / 60)) min)"
    [ -f "$RUN/formats/runs/frozen-base/open/metrics.json" ] || stop "the frozen base could not be scored on the new formats; see logs/formats-frozen-base.log"
    for arm in $FORMAT_ARMS; do
        root=$(home "$arm") || { log "formats: $arm has no trained checkpoint, skipped"; continue; }
        started=$SECONDS
        "$PY" -m experiments.hearside_pilot.formats run --folder "$RUN/formats" --model-config "$root/configs/$arm.json" --method ta-lora \
            --checkpoint "$(field "$root/train/$arm/result.json" best_checkpoint)" --arm "$arm" > "$RUN/logs/formats-$arm.log" 2>&1
        code=$?
        log "formats: $arm (exit $code, $(((SECONDS - started) / 60)) min)"
        [ "$code" -eq 0 ] || stop "formats: $arm could not be scored; see logs/formats-$arm.log: $(tail -1 "$RUN/logs/formats-$arm.log")"
        formats_summary
    done
    formats_summary
    [ -f "$RUN/summary/formats.md" ] && cat "$RUN/summary/formats.md" | tee -a "$RUN/pilot.log"
;; esac

# ------------------------------------------------------------------ 3. headroom for input-conditional adaptation
case " $STAGES " in *" 3 "*)
    if [ ! -f "$RUN/headroom/headroom.json" ]; then
        rm -rf "$RUN/headroom"
        "$PY" -m experiments.hearside_pilot.headroom build --data "$RUN/data" --test-manifest "$RUN/eval/manifests/orig.jsonl" \
            --base-config "$PILOT/configs/D-all-shuffled.json" --output "$RUN/headroom" > "$RUN/logs/headroom-build.log" 2>&1 \
            || stop "building the specialists' manifests failed; see logs/headroom-build.log: $(tail -1 "$RUN/logs/headroom-build.log")"
    fi
    headroom_summary() {
        "$PY" -m experiments.hearside_pilot.headroom summarize --folder "$RUN/headroom" --eval "$PILOT/eval" "$FOLLOWUP/eval" \
            --followup "$RUN/summary/followup.json" --output "$RUN/summary" > "$RUN/logs/summary-headroom.log" 2>&1 \
            || log "headroom summary failed; see logs/summary-headroom.log: $(tail -1 "$RUN/logs/summary-headroom.log")"
    }
    headroom_summary                                                      # step zero needs no training
    for key in $SPECIALISTS; do
        log "$key: training"
        train "$RUN/headroom/$key/config.json" "$RUN/headroom/$key/train.jsonl" "$RUN/headroom/$key/dev.jsonl" "$RUN/train/$key" "$key" || continue
        "$PY" -m experiments.hearside_pilot.headroom score --folder "$RUN/headroom" --specialist "$key" \
            --model-config "$RUN/headroom/$key/config.json" --checkpoint "$(field "$RUN/train/$key/result.json" best_checkpoint)" \
            > "$RUN/logs/headroom-eval-$key.log" 2>&1
        log "$key scored on its Session 4 utterances (exit $?)"
        headroom_summary
    done
    [ -f "$RUN/summary/headroom.md" ] && cat "$RUN/summary/headroom.md" | tee -a "$RUN/pilot.log"
;; esac
log "done"
