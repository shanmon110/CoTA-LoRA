# Copyright 2025 Binbin Zhang(binbzha@qq.com)

[ ! -s west ] && ln -s ../../../west
[ ! -s tools ] && ln -s ../../../tools
export PYTHONPATH="$(cd ../../.. && pwd):$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Change this to all your available gpus, such as "0,1,2,3"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train # data/train/decode
data=data
dir=exp/Qwen2-7B-Instruct-firered
steps=  # optional override of max_steps from the training config
training_config=  # optional override of the shared training profile
adapter_config=  # optional tagged adapter config; otherwise use the model's lora_config

# Available model_conf are in conf, such as:
# conf/qwen2-7b_firered.json
# conf/qwen3-1.7b-lora_firered.json
# conf/qwen2-7b-lora_paraformer.json
# conf/qwen2-1.5b-lora_whisper-large-v3-turbo.json
model_conf=conf/qwen2-7b_firered.json
decode_conf=conf/generation_config.json

. tools/parse_options.sh

if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    . ../../common/touch_asu_asr.sh || exit $?
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    if [[ -z "$steps" ]]; then
        steps=$(python - "$dir/trainer_state.json" "${training_config:-../../conf/training/touch_asu_asr.json}" <<'PYTHON'
import json
import sys
from pathlib import Path
state, config = map(Path, sys.argv[1:])
print(json.loads(state.read_text())["global_step"] if state.exists() else json.loads(config.read_text())["max_steps"])
PYTHON
        ) || exit $?
    fi
    mdir="$dir/checkpoint-$steps"
    [[ -d "$mdir" ]] || { echo "Missing checkpoint directory: $mdir" >&2; exit 1; }
    cp "$decode_conf" "$mdir/" || exit $?
    python west/bin/decode.py \
        --data_path "$data/test.jsonl" \
        --model_config_or_dir "$mdir" \
        --result_path "$mdir/result.jsonl" || exit $?
    python tools/compute_wer.py --char=1 --v=1 \
        "$data/test.jsonl" "$mdir/result.jsonl" > "$mdir/result.wer"
fi
