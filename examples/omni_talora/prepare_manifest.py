"""Index local LibriSpeech or IEMOCAP data. Never downloads or modifies audio."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile

IEMOCAP_EMOTIONS = {
    "ang": "angry", "hap": "happy", "exc": "excited", "sad": "sad", "neu": "neutral",
}


def _librispeech(subset):
    # Exactly one extracted subset, not a corpus root containing train and test.
    transcripts = sorted(subset.glob("*/*/*.trans.txt"))
    if not transcripts:
        raise ValueError(f"{subset}: expected one LibriSpeech subset with reader/chapter/*.trans.txt")
    for path in transcripts:
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                fields = line.strip().split(maxsplit=1)
                if len(fields) != 2 or not re.fullmatch(r"\d+-\d+-\d+", fields[0]):
                    raise ValueError(f"{path}:{n}: expected utterance-id and transcript")
                utt, text = fields
                speaker, chapter, _ = utt.split("-")
                if (speaker, chapter) != (path.parent.parent.name, path.parent.name):
                    raise ValueError(f"{path}:{n}: utterance {utt} does not belong to this chapter")
                yield {"id": utt, "audio": path.parent / f"{utt}.flac", "text": text, "speaker": speaker}


def _session_numbers(raw):
    if not raw:
        raise ValueError("--sessions is required for IEMOCAP; e.g. 1,2,3,4 (train) or 5 (held out)")
    fields = [x.strip() for x in raw.split(",")]
    if any(x not in {"1", "2", "3", "4", "5"} for x in fields) or len(set(fields)) != len(fields):
        raise ValueError("--sessions must be distinct session numbers from 1 to 5, separated by commas")
    return sorted(map(int, fields))


def _iemocap(root, sessions, skipped):
    for session in sessions:
        base = root / f"Session{session}"
        paths = sorted((base / "dialog" / "EmoEvaluation").glob("*.txt"))
        if not paths:
            raise ValueError(f"{base}: missing dialog/EmoEvaluation/*.txt")
        for path in paths:
            # Consensus rows begin with a time range; other lines are comments
            # or individual annotators' votes. Only ASCII IDs/labels are used.
            with path.open(encoding="latin-1") as f:
                for n, line in enumerate(f, 1):
                    line = line.lstrip()
                    if not line.startswith("["):
                        continue
                    _, delim, rest = line.partition("]")
                    fields = rest.split()
                    if not delim or len(fields) < 2:
                        raise ValueError(f"{path}:{n}: malformed consensus annotation")
                    utt, label = fields[:2]
                    if label not in IEMOCAP_EMOTIONS:
                        skipped[label] += 1
                        continue
                    turn = re.fullmatch(re.escape(path.stem) + r"_([FM])\d+", utt)
                    if not turn or not utt.startswith(f"Ses{session:02}"):
                        raise ValueError(f"{path}:{n}: invalid utterance id {utt!r} for this session/dialog")
                    wav = base / "sentences" / "wav" / path.stem / f"{utt}.wav"
                    # The first letter in Ses01F names the dialogue; the FINAL
                    # F/M utterance suffix identifies who actually speaks.
                    yield {"id": utt, "audio": wav, "emotion": IEMOCAP_EMOTIONS[label],
                           "session": session, "speaker": f"Ses{session:02}{turn[1]}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["asr", "ser"], required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="One LibriSpeech subset, or IEMOCAP root containing Session1...Session5")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; never overwritten")
    parser.add_argument("--sessions", help="IEMOCAP only: comma-separated session numbers (required)")
    a = parser.parse_args(argv)
    root = a.root.expanduser().resolve()
    # Do not resolve an output symlink: an existing entry, including a broken
    # symlink, must not be followed or overwritten.
    output = a.output.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    skipped, kept, seen = Counter(), Counter(), set()
    if a.task == "asr":
        if a.sessions is not None:
            raise ValueError("--sessions is only valid for IEMOCAP (--task ser)")
        rows = _librispeech(root)
    else:
        rows = _iemocap(root, _session_numbers(a.sessions), skipped)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".omni-manifest-") as staging:
        staged = Path(staging) / "manifest"
        with staged.open("x", encoding="utf-8") as tmp:
            for row in rows:
                if row["id"] in seen:
                    raise ValueError(f"duplicate utterance id: {row['id']}")
                seen.add(row["id"])
                if not row["audio"].is_file():
                    raise FileNotFoundError(f"missing annotated audio: {row['audio']}")
                row["audio"] = os.path.relpath(row["audio"], output.parent)
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
                if a.task == "ser":
                    kept[row["emotion"]] += 1
            if not seen:
                raise ValueError(f"{root}: no eligible utterances in the requested split")
        # Same-filesystem atomic, no-clobber publication; on failure only our
        # private staging file is removed. This is safe for late missing WAVs.
        os.link(staged, output)
    print(json.dumps({"task": a.task, "output": str(output), "written": len(seen),
                      "kept_emotions": dict(kept), "skipped_emotions": dict(skipped)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
