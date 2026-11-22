"""Decode a selected Omni checkpoint through WEST and compare word counts."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from examples.omni_talora.data import load_manifest
from examples.omni_talora.evaluate import asr_metrics
from examples.omni_talora.train import main as train
if __package__:
    from .test_omni_ta_integration import write_tiny_checkpoint
else:
    from test_omni_ta_integration import write_tiny_checkpoint


@pytest.mark.parametrize("extra", [{"text": "other reference"}, {"audio": "other.wav"}])
def test_conflicting_west_manifest_aliases(tmp_path, extra):
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(dict(wav="clip.wav", txt="reference", **extra)))
    with pytest.raises(ValueError, match="conflicting"):
        load_manifest(manifest, "asr")


@pytest.mark.skipif(not os.getenv("OMNI_PROCESSOR_DIR"), reason="requires offline official processor")
def test_west_word_scoring_matches_selected_adapter_decode(tmp_path):
    write_tiny_checkpoint(tmp_path, Path(os.environ["OMNI_PROCESSOR_DIR"]))
    for i, size in enumerate([2400, 3200, 2800]):
        sf.write(tmp_path / f"u{i}.wav", np.sin(np.arange(size) * .04), 16000)
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text('{"wav":"u0.wav","txt":"hello"}\n')
    dest = tmp_path / "adapter"
    train(["--model-name-or-path", str(tmp_path), "--device", "cpu", "--task", "asr",
           "--manifest", str(train_manifest), "--output", str(dest), "--last-n", "1", "--rank", "2",
           "--max-steps", "1", "--accumulation-steps", "1"])
    train_manifest.unlink()
    (tmp_path / "u0.wav").unlink()

    manifest = tmp_path / "held-out.jsonl"
    rows = [dict(id="a", wav="u1.wav", txt="It’s fine."), dict(id="b", wav="u2.wav", txt="Again!")]
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    conf = tmp_path / "generation.json"
    conf.write_text('{"do_sample":false,"num_beams":1,"max_new_tokens":2}')
    refs, hyps = tmp_path / "ref.jsonl", tmp_path / "result.jsonl"
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root/"west/bin/decode.py"), "--model_config_or_dir", str(dest),
           "--data_path", str(manifest), "--result_path", str(hyps), "--reference_path", str(refs),
           "--batch_size", "2", "--generation_config", str(conf), "--device", "cpu"]
    result = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=""))
    metrics = json.loads(result.stdout)
    hypotheses = list(map(json.loads, hyps.read_text().splitlines()))
    assert [r["id"] for r in hypotheses] == ["a", "b"]
    assert [r["txt"] for r in map(json.loads, refs.read_text().splitlines())] == ["it's fine", "again"]
    assert metrics["utterances"] == metrics["max_new_tokens"] == 2
    expected = asr_metrics([r["txt"] for r in rows], [r["raw_txt"] for r in hypotheses])
    assert metrics["word_errors"] == expected["word_errors"]
    score = subprocess.check_output([sys.executable, str(root/"tools/compute_wer.py"), "--char=0",
                                     "--cs=1", "--v=0", str(refs), str(hyps)], text=True)
    pattern = r"N=(\d+) C=\d+ S=(\d+) D=(\d+) I=(\d+)"
    n, substitutions, deletions, insertions = map(int, re.search(pattern, score).groups())
    assert n == metrics["reference_words"] == 3
    assert substitutions + deletions + insertions == metrics["word_errors"]
