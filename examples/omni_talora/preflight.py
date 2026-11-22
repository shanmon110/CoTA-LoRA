"""Audit paired train/dev manifests on CPU using the checkpoint's real tokenizer and template."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import soundfile as sf

from .budget import Budget, add_budget_args
from .data import OmniCollator, audio_lengths, load_manifest


def _validate_row(row, task, split):
    utt = row.get("id")
    if not isinstance(utt, str) or not utt.strip() or utt.strip() != utt:
        raise ValueError("every manifest row needs a nonempty utterance id without surrounding whitespace")
    if task == "ser":
        allowed = {"1", "2", "3"} if split == "train" else {"4"}
        declared = str(row.get("session"))
        tags = re.findall(r"(?:^|/)(?:Session|Ses)0*(\d+)(?=[FM/_.]|$)",
                          "/".join([utt, str(row.get("speaker", "")), row["audio"]]), re.I)
        if declared not in allowed or any(tag != declared for tag in tags):
            raise ValueError(f"{utt}: invalid session metadata/path for {split}: {declared}, tags {tags}")


def _measure(collator, row, split):
    extractor = collator.processor.feature_extractor
    duration, samples, frames = audio_lengths(sf.info(row["audio"]), extractor.hop_length)
    audio_tokens = ((frames + 1) // 2) // 2
    text = collator.training_text(row)[1] if split == "train" else collator.prompt_text(row)
    if text.count(collator.audio_token) != 1:
        raise ValueError(f"{row['id']}: expected one rendered audio placeholder")
    # HF replaces the placeholder by this many adjacent special tokens. Tokenize
    # the full rendered string so answer/header boundaries are treated identically.
    expanded = text.replace(collator.audio_token, collator.audio_token * audio_tokens)
    ids = collator.tokenizer(expanded, truncation=False, add_special_tokens=False)["input_ids"]
    if ids.count(collator.audio_id) != audio_tokens:
        raise ValueError(f"{row['id']}: expanded audio placeholder count mismatch")
    length = len(ids)
    reasons = []
    if samples > extractor.n_samples:
        reasons.append("processor_audio_limit")
    if not audio_tokens:
        reasons.append("insufficient_audio_frames")
    if length > collator.tokenizer.model_max_length:
        reasons.append("tokenizer_context_limit")
    return dict(split=split, id=row["id"], duration_seconds=duration, frames=frames,
                audio_tokens=audio_tokens, sequence_length=length, reasons=reasons)


def _distribution(values):
    quantiles = np.percentile(values, [50, 90, 95, 99])
    return {"min": min(values), **dict(zip(("p50", "p90", "p95", "p99"), map(float, quantiles))),
            "max": max(values)}


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_violations(measures, split, batch_size, budget, stage):
    count = 0
    with (stage / "batch_violations.jsonl").open("a", encoding="utf-8") as stream:
        for start in range(0, len(measures), batch_size):
            group = measures[start:start + batch_size]
            frames = sum(m["frames"] for m in group)
            if budget.reasons(frames=frames):
                record = dict(split=split, ids=[m["id"] for m in group], frames=frames,
                              reason="max_batch_frames")
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def audit(processor, task, train_manifest, dev_manifest, output_dir, *, budget=None,
          prefilter=False, batch_size=1, processor_source=None, revision=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    budget = budget or Budget()
    report = {"task": task, "processor": processor_source, "revision": revision,
              "budget": asdict(budget), "batch_size": batch_size, "prefilter": prefilter, "splits": {}}
    collator = OmniCollator(processor, task)
    seen_ids, seen_files = {}, {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)
        with (stage / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for split, path in (("train", Path(train_manifest)), ("dev", Path(dev_manifest))):
                before = _sha(path)
                rows = load_manifest(path, task, validate_row=lambda r: _validate_row(r, task, split))
                if _sha(path) != before:
                    raise ValueError(f"manifest changed while reading: {path}")
                measures, retained, counts = [], [], Counter()
                for row in rows:
                    utt = row["id"]
                    stat = Path(row["audio"]).stat()
                    file_id = (stat.st_dev, stat.st_ino)
                    if utt in seen_ids or file_id in seen_files:
                        previous = seen_ids.get(utt) or seen_files[file_id]
                        raise ValueError(f"duplicate utterance id or audio: {split}/{utt} and {previous}")
                    seen_ids[utt] = seen_files[file_id] = f"{split}/{utt}"
                    m = _measure(collator, row, split)
                    m["reasons"].extend(budget.reasons(seconds=m["duration_seconds"], frames=m["frames"],
                                                       tokens=m["sequence_length"]))
                    counts.update(m["reasons"])
                    measures.append(m)
                    if not m["reasons"]:
                        retained.append(row)
                    stream.write(json.dumps(m, ensure_ascii=False) + "\n")
                eligible_frames = sorted((m["frames"] for m in measures if not m["reasons"]), reverse=True)
                worst = sum(eligible_frames[:batch_size])
                summary = dict(input=str(path.resolve()), input_sha256=before, total=len(rows),
                               eligible=len(retained), over_limit=len(rows) - len(retained),
                               reason_counts=dict(counts), largest_eligible_batch_frames=worst,
                               eligible_batch_fits=not budget.reasons(frames=worst))
                summary["over_limit_manifest_order_batches"] = _batch_violations(
                    measures, split, batch_size, budget, stage)
                for key in ("duration_seconds", "frames", "sequence_length"):
                    summary[key] = _distribution([m[key] for m in measures])
                if prefilter:
                    dst = stage / f"{split}.jsonl"
                    with dst.open("w", encoding="utf-8") as filtered:
                        for row in retained:
                            filtered.write(json.dumps(row, ensure_ascii=False) + "\n")
                    summary.update(filtered_file=dst.name, filtered_sha256=_sha(dst))
                report["splits"][split] = summary
        (stage / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
        out.mkdir()  # Exclusive reservation; concurrent runs cannot publish to the same path.
        for p in sorted(stage.iterdir(), key=lambda p: p.name == "summary.json"):
            p.rename(out / p.name)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["asr", "ser"], required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--dev-manifest", required=True)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--revision")
    p.add_argument("--output-dir", required=True, help="New audit directory.")
    p.add_argument("--batch-size", type=int, default=1, help="Report worst valid-frame sum across eligible rows.")
    p.add_argument("--prefilter", action="store_true", help="Also publish new train.jsonl and dev.jsonl.")
    add_budget_args(p)
    a = p.parse_args(argv)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model_name_or_path, revision=a.revision, local_files_only=True)
    report = audit(processor, a.task, a.train_manifest, a.dev_manifest, a.output_dir,
                   budget=Budget.from_args(a), prefilter=a.prefilter, batch_size=a.batch_size,
                   processor_source=a.model_name_or_path, revision=a.revision)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
