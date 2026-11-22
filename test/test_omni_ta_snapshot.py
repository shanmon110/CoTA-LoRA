"""A local adapter must detect a different base with the same tensor shapes."""
import sys

import pytest

from examples.omni_talora.runtime import local_snapshot_sha256


@pytest.mark.parametrize("changed", ["model.safetensors", "tokenizer.json"])
def test_local_snapshot_replacement_stops_evaluation(tmp_path, monkeypatch, changed):
    from examples.omni_talora import checkpoint, data, evaluate, runtime

    for name in ["config.json", "model.safetensors", "tokenizer.json"]:
        (tmp_path / name).write_bytes(b"base-A")
    config = dict(model_name_or_path=str(tmp_path), revision=None, task="asr",
                  base_sha256=local_snapshot_sha256(tmp_path))
    (tmp_path / "run_notes.txt").write_text("unrelated")
    assert local_snapshot_sha256(tmp_path) == config["base_sha256"]
    (tmp_path / changed).write_bytes(b"base-B")
    monkeypatch.setattr(checkpoint, "read_config", lambda _: config)
    monkeypatch.setattr(data, "load_manifest", lambda *_: [])
    monkeypatch.setattr(runtime, "load_base", lambda *_: pytest.fail("loaded changed base"))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", "dev", "--adapter", "run", "--device", "cpu"])
    with pytest.raises(ValueError, match="training snapshot"):
        evaluate.main()
