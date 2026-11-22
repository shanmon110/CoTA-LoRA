"""Prepare and audit the four LibriSpeech splits on CPU."""
import argparse
import json
from pathlib import Path

import soundfile as sf

from ..common import COMPARISON, MANIFEST_HASHES, ROOT, sha, training_strategy

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "conf/qwen2.5-omni-7b.json"
BATCH, ACCUM = 2, 4
COUNTS = {"train": 28539, "dev": 2703, "test-clean": 2620, "test-other": 2939}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare_splits(plan, test_dir, expected_hashes, batch_size=BATCH):
    paths = {key: (plan if key in MANIFEST_HASHES else test_dir) / (key + ".jsonl")
             for key in COUNTS}
    hashes = {key: sha(path) for key, path in paths.items()}
    if any(hashes[key] != value for key, value in expected_hashes.items()):
        raise ValueError("Training or development manifest differs from preflight")
    ids = {}
    speakers_seen, ids_seen = set(), set()
    # Include held-out audio lengths for capacity planning, without scoring test
    # hypotheses or using transcripts to determine training or selection settings.
    measures = read_rows(plan / "samples.jsonl")
    seconds = max(s["duration_seconds"] for s in measures)
    frames = max(s["frames"] for s in measures)
    length = max(s["sequence_length"] for s in measures)
    overheads = {s["sequence_length"] - s["audio_tokens"] for s in measures if s["split"] == "dev"}
    if len(overheads) != 1:
        raise ValueError("Expected a fixed ASR prompt")
    for key, path in paths.items():
        rows = read_rows(path)
        ids[key] = [row["id"] for row in rows]
        speakers = {row["speaker"] for row in rows}
        if (len(rows) != COUNTS[key] or len(set(ids[key])) != len(rows)
                or ids_seen.intersection(ids[key]) or speakers_seen.intersection(speakers)):
            raise ValueError(f"Unexpected split identity, overlap or count in {key}")
        ids_seen.update(ids[key])
        speakers_seen.update(speakers)
        if key.startswith("test-"):
            for row in rows:
                info = sf.info(path.parent / row["audio"])
                if info.samplerate != 16000:
                    raise ValueError("Unexpected LibriSpeech sample rate")
                n = (info.frames + 159) // 160
                seconds = max(seconds, info.duration)
                frames = max(frames, n)
                length = max(length, (n + 1) // 4 + next(iter(overheads)))
    limits = dict(max_audio_seconds=seconds, max_batch_frames=batch_size * frames, max_sequence_length=length)
    return paths, hashes, ids, limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--comparison", type=Path, default=COMPARISON)
    p.add_argument("--training", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--data-dir", type=Path)
    args = p.parse_args()
    conf = json.loads(args.config.read_text())
    strategy = training_strategy(conf, args.comparison, json.loads(args.comparison.read_text()), args.training)
    batch_size = max(strategy["per_device_train_batch_size"], strategy["per_device_eval_batch_size"])
    plan = ROOT / conf["plan"]
    test_dir = ROOT / conf["test_manifests"]
    base = ROOT / conf["model"]
    if args.corpus and not args.data_dir:
        p.error("--corpus requires --stage data and a new --data-dir")
    expected_hashes = MANIFEST_HASHES
    if args.data_dir:
        plan = args.data_dir.resolve() / "plan"
        test_dir = args.data_dir.resolve() / "manifests"
    if args.corpus:
        from examples.omni_talora import prepare_manifest, preflight

        args.data_dir.mkdir(parents=True, exist_ok=False)
        for split, subset in (("train", "train-clean-100"), ("dev", "dev-clean"),
                              ("test-clean", "test-clean"), ("test-other", "test-other")):
            prepare_manifest.main(["--task", "asr", "--root", str(args.corpus / subset),
                                   "--output", str(test_dir / (split + ".jsonl"))])
        preflight.main(["--task", "asr", "--train-manifest", str(test_dir / "train.jsonl"),
                        "--dev-manifest", str(test_dir / "dev.jsonl"), "--output-dir", str(plan),
                        "--batch-size", str(batch_size), "--model-name-or-path", str(base), "--prefilter"])
    if args.data_dir:
        summary = json.loads((plan / "summary.json").read_text())["splits"]
        expected_hashes = {name: summary[name]["filtered_sha256"] for name in ("train", "dev")}
    _, hashes, _, limits = prepare_splits(plan, test_dir, expected_hashes, batch_size)
    print(json.dumps(dict(counts=COUNTS, manifest_sha256=hashes, input_budget=limits), indent=2))


if __name__ == "__main__":
    main()
