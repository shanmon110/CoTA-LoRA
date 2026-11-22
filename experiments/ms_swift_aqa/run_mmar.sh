#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${AQA_PYTHON:-/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python}
data=${MMAR_DATA:-/root/autodl-tmp/MMAR}
run=${AQA_SOURCE_RUN:-/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01}
out=${MMAR_OUTPUT:-$data/runs/omni3b-avqa-01}
export PATH="$(dirname "$py"):$PATH" OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export ENABLE_AUDIO_OUTPUT=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$out/official"
exec 9>"$run/gpu.lock"
flock 9
if [[ ! -f "$out/official/evaluation.py" ]]; then
    curl -fsSL https://raw.githubusercontent.com/ddlBoJack/MMAR/3bce090a967db576c8ad433b290a8da582d6d2a0/code/evaluation.py -o "$out/official/evaluation.py.tmp"
    mv "$out/official/evaluation.py.tmp" "$out/official/evaluation.py"
fi
"$py" -m experiments.ms_swift_aqa.evaluate_mmar --out "$out" prepare --dataset "$data" --source-run "$run"
methods=$("$py" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["methods"]))' "$out/protocol.json")
while IFS= read -r method; do
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$method" >> "$out/stages.tsv"
    if [[ ! -f "$out/$method/metrics.json" ]]; then
        "$py" -m experiments.ms_swift_aqa.evaluate_mmar --out "$out" infer --method "$method" >> "$out/$method.log" 2>&1
    fi
    "$py" -m experiments.ms_swift_aqa.summarize_mmar --out "$out" --method "$method"
done <<< "$methods"
"$py" -m experiments.ms_swift_aqa.summarize_mmar --out "$out"
printf '%s\tfinished\n' "$(date -u +%FT%TZ)" >> "$out/stages.tsv"
