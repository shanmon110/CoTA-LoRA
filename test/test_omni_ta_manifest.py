"""Layout fixtures only. The converter indexes files; it does not decode audio."""
import json
from pathlib import Path

import pytest

from examples.omni_talora.prepare_manifest import main


def test_librispeech_indexes_one_subset_and_never_overwrites(tmp_path, capsys):
    subset = tmp_path / "LibriSpeech" / "dev-clean"
    chapter = subset / "19" / "198"
    chapter.mkdir(parents=True)
    flac = chapter / "19-198-0000.flac"
    flac.touch()
    (chapter / "19-198.trans.txt").write_text("19-198-0000 HELLO WORLD\n")
    dest = tmp_path / "splits" / "dev.jsonl"
    args = ["--task", "asr", "--root", str(subset), "--output", str(dest)]
    main(args)
    row = json.loads(dest.read_text())
    assert (dest.parent / row["audio"]).resolve() == flac.resolve()
    assert not Path(row["audio"]).is_absolute()
    assert (row["id"], row["text"], row["speaker"]) == ("19-198-0000", "HELLO WORLD", "19")
    assert json.loads(capsys.readouterr().out)["written"] == 1
    with pytest.raises((ValueError, SystemExit, FileExistsError)):
        main(args)
    assert json.loads(dest.read_text()) == row
    with pytest.raises((ValueError, SystemExit), match="subset"):
        main(["--task", "asr", "--root", str(subset.parent), "--output", str(tmp_path / "bad")])


def mock_session(root, session, emotions):
    base = root / f"Session{session}"
    dialog = f"Ses{session:02}F_impro01"
    labels = base / "dialog" / "EmoEvaluation"
    clips = base / "sentences" / "wav" / dialog
    labels.mkdir(parents=True)
    clips.mkdir(parents=True)
    lines = ["% annotation header\n"]
    for i, emotion in enumerate(emotions):
        utt = f"{dialog}_M{i:03}"
        lines.extend([f"[0.100 - 1.100]\t{utt}\t{emotion}\t[2.0, 2.5, 2.0]\n",
                      "C-F1:\tNeutral;\n"])
        (clips / f"{utt}.wav").touch()
    (labels / f"{dialog}.txt").write_text("".join(lines))
    return clips


def test_iemocap_maps_consensus_tags_and_uses_only_requested_sessions(tmp_path, capsys):
    mock_session(tmp_path, 1, ["ang", "hap", "exc", "sad", "neu", "fru", "xxx"])
    mock_session(tmp_path, 5, ["ang"])
    dest = tmp_path / "manifests" / "train.jsonl"
    main(["--task", "ser", "--root", str(tmp_path), "--sessions", "1", "--output", str(dest)])
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert [r["emotion"] for r in rows] == ["angry", "happy", "excited", "sad", "neutral"]
    assert all(r["session"] == 1 and r["speaker"] == "Ses01M" for r in rows)
    assert all((dest.parent / r["audio"]).is_file() for r in rows)
    report = json.loads(capsys.readouterr().out)
    assert report["written"] == 5 and report["skipped_emotions"] == {"fru": 1, "xxx": 1}


def test_missing_annotated_audio_fails_without_publishing_partial_split(tmp_path):
    clips = mock_session(tmp_path, 1, ["sad", "ang"])
    (clips / "Ses01F_impro01_M001.wav").unlink()
    dest = tmp_path / "bad.jsonl"
    with pytest.raises((ValueError, FileNotFoundError), match="M001.wav"):
        main(["--task", "ser", "--root", str(tmp_path), "--sessions", "1", "--output", str(dest)])
    assert not dest.exists()


@pytest.mark.parametrize("sessions", [[], ["--sessions", "0"], ["--sessions", "1,1"]])
def test_ser_requires_explicit_unique_sessions(tmp_path, sessions):
    dest = tmp_path / "bad.jsonl"
    with pytest.raises((ValueError, SystemExit), match="session"):
        main(["--task", "ser", "--root", str(tmp_path), "--output", str(dest), *sessions])
    assert not dest.exists()
