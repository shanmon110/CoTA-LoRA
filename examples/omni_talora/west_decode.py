"""Decode a WEST Omni adapter and emit word-scoring JSONL in manifest order."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .evaluate import _words, evaluate_checkpoint


def is_omni_adapter(directory):
    from safetensors import safe_open
    from .checkpoint import WEIGHTS, _FORMAT

    path = Path(directory) / WEIGHTS
    if not path.is_file():
        return False
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return (f.metadata() or {}).get("format") == _FORMAT


def write_scoring_files(rows, references, hypotheses, result_path, reference_path):
    outputs = [(result_path, hypotheses)]
    if reference_path is not None:
        outputs.append((reference_path, references))
    published = []
    with ExitStack() as stack:
        staged = []
        for path, texts in outputs:
            directory = stack.enter_context(TemporaryDirectory(dir=path.parent))
            temp = Path(directory) / "scoring.jsonl"
            with temp.open("x", encoding="utf-8") as out:
                for row, text in zip(rows, texts, strict=True):
                    item = {"wav": row["audio"], "txt": " ".join(_words(text)), "raw_txt": text}
                    if "id" in row:
                        item["id"] = row["id"]
                    print(json.dumps(item, ensure_ascii=False), file=out)
            staged.append((temp, path))
        try:
            for temp, path in staged:
                os.link(temp, path)
                published.append(path)
        except OSError:
            for path in published:
                path.unlink()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_config_or_dir", "--model-config-or-dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--reference_path", type=Path, help="Write matching WEST word-scoring references")
    parser.add_argument("--batch_size", "--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--generation_config", type=Path)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke_log")
    args = parser.parse_args(argv)
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        parser.error("Omni ASR evaluation expects one process per manifest")
    generation = {}
    if args.generation_config:
        generation = json.loads(args.generation_config.read_text())
        if (set(generation) - {"max_new_tokens", "num_beams", "do_sample"}
                or generation.get("num_beams", 1) != 1 or generation.get("do_sample", False) is not False):
            parser.error("Omni ASR supports greedy decoding with max_new_tokens, do_sample=false and num_beams=1")
    limit = generation.get("max_new_tokens", 128)
    if args.max_new_tokens is not None:
        if "max_new_tokens" in generation and args.max_new_tokens != limit:
            parser.error("max_new_tokens differs from the generation configuration")
        limit = args.max_new_tokens
    if args.batch_size < 1 or type(limit) is not int or limit < 1:
        parser.error("Batch size and generation limit must be positive integers")
    paths = [path for path in (args.result_path, args.reference_path) if path is not None]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Result and reference paths must be distinct")
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def publish(rows, refs, hyps):
        write_scoring_files(rows, refs, hyps, args.result_path, args.reference_path)

    score = evaluate_checkpoint(args.model_config_or_dir, args.data_path, device=args.device,
                                batch_size=args.batch_size, max_new_tokens=limit,
                                smoke_log=args.smoke_log, on_predictions=publish)
    print(json.dumps(score, indent=2, ensure_ascii=False))
