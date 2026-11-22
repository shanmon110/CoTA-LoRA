"""Manifest audits use the same Qwen chat rendering and expanded lengths as batching."""
import json

import pytest

from examples.omni_talora.budget import Budget
from examples.omni_talora.data import OmniCollator
from examples.omni_talora.preflight import audit
from test_omni_ta_data import processor, records


def manifests(records, tmp_path):
    paths = [tmp_path / (split + ".jsonl") for split in ("train", "dev")]
    for path, row in zip(paths, records):
        path.write_text(json.dumps(row) + "\n")
    return paths


def test_exact_stats_and_shared_limits(tmp_path, processor, records):
    train, dev = manifests(records, tmp_path)
    collate = OmniCollator(processor, "asr")
    out = tmp_path / "stats"
    report = audit(processor, "asr", train, dev, out)
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    for sample, batch in zip(samples, (collate(records[:1]), collate.prompts(records[1:]))):
        assert sample["sequence_length"] == batch["attention_mask"].sum().item()
        assert sample["frames"] == batch["feature_attention_mask"].sum().item()
    assert [s["duration_seconds"] for s in samples] == [0.2, 0.4]
    assert report["splits"]["train"]["over_limit"] == 0
    with pytest.raises(ValueError, match="max_batch_frames"):
        OmniCollator(processor, "asr", budget=Budget(max_batch_frames=59))(records)
    with pytest.raises(ValueError, match="x.*max_sequence_length"):
        OmniCollator(processor, "asr", budget=Budget(max_sequence_length=8)).prompts(records)
    with pytest.raises(ValueError, match="y.*max_audio_seconds"):
        OmniCollator(processor, "asr", budget=Budget(max_audio_seconds=.3))(records)


def test_new_manifests_audited_no_overwrite(tmp_path, processor, records):
    import hashlib
    train, dev = manifests(records, tmp_path)
    # A distinct training clip over the limit.
    from shutil import copyfile
    clip = tmp_path / "extra.wav"
    copyfile(records[1]["audio"], clip)
    with train.open("a") as stream:
        stream.write(json.dumps(dict(records[1], audio=str(clip), id="long")) + "\n")
    out = tmp_path / "filtered"
    limit = Budget(max_batch_frames=35)
    report = audit(processor, "asr", train, dev, out, budget=limit, prefilter=True)
    assert json.loads((out / "train.jsonl").read_text())["id"] == "x"
    assert report["splits"]["train"]["filtered_sha256"] == hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    assert report["splits"]["dev"]["over_limit"] == 1
    assert report["splits"]["dev"]["reason_counts"] == {"max_batch_frames": 1}
    assert json.loads((out / "samples.jsonl").read_text().splitlines()[1])["reasons"] == ["max_batch_frames"]
    with pytest.raises(FileExistsError):
        audit(processor, "asr", train, dev, out, budget=limit, prefilter=True)
    assert len(train.read_text().splitlines()) == 2


def test_split_contamination(tmp_path, processor, records):
    train, dev = manifests(records, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        audit(processor, "asr", train, train, tmp_path / "stats")
    train.write_text(json.dumps(dict(records[0], session=1, id="Ses01F_impro01_F001")) + "\n")
    dev.write_text(json.dumps(dict(records[1], session=4, id="Ses05F_impro01_F001")) + "\n")
    with pytest.raises(ValueError, match="session"):
        audit(processor, "ser", train, dev, tmp_path / "stats")


def test_reports_batch_ids_without_filtering_individually_eligible_rows(tmp_path, processor, records):
    import numpy as np
    import soundfile as sf
    train, dev = manifests(records, tmp_path)
    wav = tmp_path / "third.wav"
    sf.write(wav, np.zeros(3200), 16000)
    with train.open("a") as f:
        f.write(json.dumps(dict(records[0], id="third", audio=str(wav))) + "\n")
    out = tmp_path / "audit"
    report = audit(processor, "asr", train, dev, out, budget=Budget(max_batch_frames=32),
                   batch_size=2, prefilter=True)
    assert report["splits"]["train"]["eligible"] == 2
    assert report["splits"]["train"]["over_limit_manifest_order_batches"] == 1
    batch = json.loads((out / "batch_violations.jsonl").read_text().splitlines()[0])
    assert batch["ids"] == ["x", "third"] and batch["frames"] == 40
    assert batch["reason"] == "max_batch_frames"
