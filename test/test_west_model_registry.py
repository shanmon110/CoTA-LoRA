"""The ASR environment can load WEST without installing TTS inference tools."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(importlib.util.find_spec("wenet") is None, reason="requires WEST's WeNet dependency")
def test_full_registry_without_tts_tools():
    # Force a fresh package import, even in a test process that already used WEST.
    script = """
import sys
sys.modules["s3tokenizer"] = sys.modules["wespeaker"] = None
import west
from transformers import AutoConfig, AutoModel
from west.bin import train, decode
from west.dataset.extractor import Extractor

config = AutoConfig.from_pretrained("examples/aishell/asr/conf/qwen3-1.7b-lora_firered.json")
assert AutoModel._model_mapping[type(config)] is west.TouchASU
assert Extractor.get_class(config.model_type).model_type == "touch_asu"
for model in (west.TouchChat, west.TouchFlow, west.TouchTTS, west.FireredASRLLM,
              west.FunASR, west.KimiAudio, west.OSUMEChat):
    assert AutoModel._model_mapping[model.config_class] is model
for model in (west.TouchFlow, west.TouchTTS):
    assert Extractor.get_class(model.config_class.model_type)
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], check=True)
