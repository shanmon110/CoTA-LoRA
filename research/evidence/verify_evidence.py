#!/usr/bin/env python3
"""Verify the published evidence subset; optionally recheck local sources."""
import argparse
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sources", action="store_true", help="Also read original local source files")
args = parser.parse_args()
root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
errors = []


def check_json(value, path, at=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"messages", "choices", "options", "metadata", "input_ids", "decoded_target",
                       "first_supervised_letter", "sample_id", "utterance_id"} and item not in (None, [], {}, ""):
                errors.append(f"unexpected sample field: {path}:{at}.{key}")
            if re.search(r"(?:^|_)(?:api_key|access_token|password|secret|hub_token|swanlab_token)$", key) and item not in (None, "", False):
                errors.append(f"nonempty credential field: {path}:{at}.{key}; value suppressed")
            check_json(item, path, at + "." + key)
    elif isinstance(value, list):
        for item in value:
            check_json(item, path, at + "[]")
    elif isinstance(value, str):
        if re.search(r"Ses\d\d[FM]_(?:impro|script).*_[FM]\d\d\d|controlled-avqa-(?:dev|train)-\d{4}|music-\d{5}", value):
            errors.append(f"unexpected per-sample identifier: {path}:{at}")


seen = set()
for entry in manifest["files"]:
    rel = entry["path"]
    if rel in seen:
        errors.append(f"duplicate manifest path: {rel}")
    seen.add(rel)
    path = root / rel
    if not path.is_file():
        errors.append(f"missing exported file: {rel}")
        continue
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != entry["bytes"]:
        errors.append(f"export hash/size mismatch: {rel}")
    if path.suffix not in {".json", ".jsonl", ".md", ".log"}:
        errors.append(f"unexpected extension: {rel}")
    if path.name in {"predictions.jsonl", "hypotheses.jsonl", "pairs.json", "client-partitions.json", "args.json"}:
        errors.append(f"excluded artifact found: {rel}")
    text = raw.decode("utf-8")
    if re.search(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", text):
        errors.append(f"credential-shaped content: {rel}; value suppressed")
    if path.suffix == ".json":
        check_json(json.loads(text), rel)
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            check_json(json.loads(line), rel)
    if entry["kind"] == "byte_copy" and entry["sha256"] != entry["source_sha256"]:
        errors.append(f"byte-copy source/export hash differs: {rel}")
    if args.sources:
        source = Path(entry["source"])
        if not source.is_file():
            errors.append(f"source unavailable: {rel}")
        else:
            source_data = source.read_bytes()
            if hashlib.sha256(source_data).hexdigest() != entry["source_sha256"] or len(source_data) != entry["source_bytes"]:
                errors.append(f"source hash/size mismatch: {rel}")
            if entry["kind"] == "byte_copy" and raw != source_data:
                errors.append(f"source/export bytes differ: {rel}")

actual = {str(path.relative_to(root)) for part in ["raw", "derived"] for path in (root / part).rglob("*") if path.is_file()}
if actual != seen:
    errors.append(f"manifest/file set mismatch: unlisted={sorted(actual-seen)}, missing={sorted(seen-actual)}")
if manifest["failures"]:
    errors.append("export manifest records failures")
report = {"checked_files": len(seen), "checked_local_sources": args.sources,
          "bytes": sum(item["bytes"] for item in manifest["files"]), "errors": errors,
          "note": "File integrity and targeted content checks only; no training or model evaluation executed."}
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
