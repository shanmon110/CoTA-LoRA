# Shared training stage for the AISHELL TouchASU recipes. Source from the recipe directory.
# The caller supplies the configured model, data, output directory and number of GPUs.
if [[ -z $training_config ]]; then
    training_config=$(python - "$model_conf" "$adapter_config" <<'PYTHON'
import json
import sys
from pathlib import Path

model, override = sys.argv[1:]
if override:
    lora = json.loads(Path(override).read_text())
elif Path(model).is_dir():
    lora = json.loads((Path(model) / "config.json").read_text()).get("lora_config")
elif Path(model).is_file():
    lora = json.loads(Path(model).read_text()).get("lora_config")
else:
    lora = None
tagged = isinstance(lora, dict) and "method" in lora
print("../../conf/training/touch_asu_adapters.json" if tagged else "../../conf/training/touch_asu_asr.json")
PYTHON
    ) || exit $?
fi
extra=()
[[ -z $adapter_config ]] || extra+=(--adapter_config "$adapter_config")
[[ -z $steps ]] || extra+=(--max_steps "$steps")
torchrun --standalone --nnodes=1 --nproc_per_node="$num_gpus" west/bin/train.py \
    --model_config_or_dir "$model_conf" --training_config "$training_config" \
    --data_path "$data/train.jsonl" --output_dir "$dir" "${extra[@]}"
