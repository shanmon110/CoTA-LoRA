#!/usr/bin/env bash
# LibriSpeech ASR with WEST's train/decode entry points. Run stages on a single GPU.
set -eCo pipefail
recipe=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$recipe/../../.." && pwd)
python_bin=${PYTHON:-python}
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

stage=train                         # data / train / decode / all
model_conf="$recipe/conf/qwen2.5-omni-7b.json"
decode_conf="$recipe/conf/generation_config.json"
data=                               # optional prepared index with plan/ and manifests/
corpus=                             # optional LibriSpeech root to index in the data stage
dir=/root/autodl-tmp/west-preflight/librispeech-7b-west
comparison_conf="$recipe/conf/comparison.json"
adapter_config=                     # optional single-method JSON
training_config=                    # optional training overrides
help_message='Usage: bash run.sh --stage data|train|decode|all [--dir EXP] [--data-dir INDEX] [--corpus LibriSpeech] [--model-conf JSON] [--decode-conf JSON] [--comparison-conf JSON] [--adapter-config JSON] [--training-config JSON]'

# Preserve the former JSON --config option. WEST parse_options reserves that name for sourced shell files.
args=()
for arg in "$@"; do
    case "$arg" in
        --config) args+=(--model-conf) ;;
        --data-dir) args+=(--data) ;;
        *) args+=("$arg") ;;
    esac
done
set -- "${args[@]}"
. "$root/tools/parse_options.sh"
[[ $# == 0 ]] || { echo "Unexpected positional argument: $1" >&2; exit 2; }
case "$stage" in data|train|decode|all) ;; *) echo "$help_message" >&2; exit 2 ;; esac
if [[ -n $corpus && ( -z $data || ( $stage != data && $stage != all ) ) ]]; then
    echo "--corpus requires a new --data-dir and the data or all stage" >&2
    exit 2
fi
protocol() { "$python_bin" -m examples.librispeech.asr.local.protocol "$@"; }
record() {
    printf '%q ' "$@" >> "$dir/commands.txt"
    printf '\n' >> "$dir/commands.txt"
    "$@"
}

if [[ $stage == data || $stage == all ]]; then
    inputs=(--config "$model_conf" --comparison "$comparison_conf")
    [[ -z $training_config ]] || inputs+=(--training "$training_config")
    [[ -z $data ]] || inputs+=(--data-dir "$data")
    [[ -z $corpus ]] || inputs+=(--corpus "$corpus")
    "$python_bin" -m examples.librispeech.asr.local.prepare_data "${inputs[@]}"
fi

if [[ $stage == train || $stage == all ]]; then
    selection=(--comparison "$comparison_conf")
    [[ -z $adapter_config ]] || selection+=(--adapter "$adapter_config")
    [[ -z $training_config ]] || selection+=(--training "$training_config")
    protocol init "$dir" "$model_conf" "${data:--}" "$decode_conf" "${selection[@]}"
    trap 'protocol fail "$dir"' ERR
    configured=$(protocol methods "$dir")
    read -r -a methods <<< "$configured"

    for method in "${methods[@]}"; do
        protocol start "$dir" "$method"
        arm="$dir/$method"
        # torchrun using the active environment's Python.
        record "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
            "$root/west/bin/train.py" \
            --model_config_or_dir "$dir/model_config.json" \
            --adapter_config "$dir/adapters/$method.json" --training_config "$dir/training_config.json" \
            --data_path "$dir/data/train.jsonl" --dev_data_path "$dir/data/dev.jsonl" \
            --output_dir "$arm/adapter" --smoke_log "$arm/train.jsonl" > "$arm/train.log" 2> "$arm/train.stderr"
        protocol finish "$dir" "$method"
    done
    protocol seal "$dir"
    trap - ERR
fi

if [[ $stage == decode || $stage == all ]]; then
    configured=$(protocol methods "$dir")
    read -r -a methods <<< "$configured"
    eval_batch=$(protocol eval-batch "$dir")
    for method in "${methods[@]}"; do
        for split in test-clean test-other; do
            checkpoint=$(protocol checkpoint "$dir" "$method" "$split")
            trap 'protocol fail "$dir"' ERR
            prefix="$dir/$method/$split"
            record "$python_bin" "$root/west/bin/decode.py" \
                --model_config_or_dir "$checkpoint" --generation_config "$dir/generation_config.json" \
                --data_path "$dir/data/$split.jsonl" --batch_size "$eval_batch" --device cuda:0 \
                --result_path "$prefix.hyp.jsonl" --reference_path "$prefix.ref.jsonl" \
                --smoke_log "$prefix.jsonl" > "$prefix.json" 2> "$prefix.stderr"
            record "$python_bin" "$root/tools/compute_wer.py" --char=0 --cs=1 --v=1 \
                "$prefix.ref.jsonl" "$prefix.hyp.jsonl" > "$prefix.wer"
            protocol score "$dir" "$method" "$split"
        done
    done
    protocol complete "$dir"
fi
