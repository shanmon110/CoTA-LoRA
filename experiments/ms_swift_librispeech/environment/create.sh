#!/usr/bin/env bash
# Usage: bash environment/create.sh /path/to/new/venv /path/to/extracted/ms-swift
set -euo pipefail
venv=${1:?provide a new venv directory}
framework=${2:?provide the unpacked ms-swift 4.7.0.dev0 project}
here=$(cd "$(dirname "$0")" && pwd)
[[ -f "$framework/examples/train/multimodal/omni/ta_lora/plugin.py" ]]
[[ ! -e "$venv" ]]
"${PYTHON:-python3.12}" -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$py" -m pip install -c "$here/resolved-constraints.txt" "$framework" qwen-omni-utils decord librosa soundfile
"$py" -m pip check
