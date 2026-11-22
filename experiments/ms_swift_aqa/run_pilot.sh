#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=${AQA_ROOT:-/root/autodl-tmp/r1-aqa}
py=${AQA_PYTHON:-/root/autodl-tmp/iemocap/qwen3-asr-capacity/venv/bin/python}
config=${AQA_CONFIG:-experiments/ms_swift_aqa/pilot.json}
data=${AQA_DATA:-$root/data/pilot-01}
out=${AQA_OUTPUT:-$root/runs/pilot-01}
export PATH="$(dirname "$py"):$PATH" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
if [[ ! -s "$data/audit.json" ]]; then
    "$py" -m experiments.ms_swift_aqa.freeze_data --train "$root/avqa/pilot/train.jsonl" --dev "$root/avqa/pilot/dev.jsonl" --test "$root/mmau/test.jsonl" --output "$data"
fi
"$py" - "$data" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); audit = json.loads((p / 'audit.json').read_text())
for name, record in audit['splits'].items():
    digest = hashlib.sha256((p / (name+'.jsonl')).read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise ValueError('Modified frozen manifest: ' + name)
PY
"$py" -m experiments.ms_swift_aqa.run --config "$config" --train "$data/train.jsonl" --dev "$data/dev.jsonl" --test "$data/test.jsonl" --output "$out"
"$py" -m experiments.ms_swift_aqa.summarize_pilot --run "$out" --data "$data" --mmau "$root/mmau"
