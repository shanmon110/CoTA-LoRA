"""Select native TouchASU checkpoints by full dev-clean WER and score the winner on test-clean."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from examples.librispeech.asr.common import ROOT, save, sha
from examples.librispeech.asr.local.prepare_data import COUNTS, read_rows
from examples.omni_talora.evaluate import asr_metrics
from examples.omni_talora.prepare_manifest import _librispeech

DEFAULT_GENERATION = ROOT / "examples/aishell/asr/conf/generation_config.json"
FEATURE_SEED = 42


def check(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(corpus, train, output):
    """Bind the active training manifest to the two official, speaker-disjoint evaluation splits."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_speakers, summary = set(), set(), {}
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        for split, key in (("train-clean-100", "train"), ("dev-clean", "dev"), ("test-clean", "test-clean")):
            rows = [dict(id=r["id"], speaker=r["speaker"], wav=str(r["audio"].resolve(strict=True)),
                         txt=r["text"]) for r in _librispeech(corpus / split)]
            ids = {r["id"] for r in rows}
            speakers = {r["speaker"] for r in rows}
            check(len(rows) == len(ids) == COUNTS[key], f"Invalid count or duplicate IDs in {split}")
            check(ids.isdisjoint(seen_ids) and speakers.isdisjoint(seen_speakers), f"Overlap in {split}")
            seen_ids.update(ids)
            seen_speakers.update(speakers)
            if key == "train":
                path = stage / "train.jsonl"
                shutil.copyfile(train, path)
                actual = read_rows(path)
                fields = ("id", "speaker", "wav", "txt")
                check(len(actual) == len(rows) and
                      {tuple(r[k] for k in fields) for r in actual} ==
                      {tuple(r[k] for k in fields) for r in rows},
                      "Active train.jsonl differs from the official train-clean-100 items")
            else:
                path = stage / f"{split}.jsonl"
                with path.open("x", encoding="utf-8") as out:
                    for row in rows:
                        print(json.dumps(row, ensure_ascii=False), file=out)
            summary[split] = dict(sha256=sha(path), utterances=len(rows), speakers=len(speakers))
        save(stage / "splits.json", summary)
        stage.rename(output)
    return summary


def model_sha(checkpoint):
    files = [checkpoint / "config.json"]
    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = checkpoint / name
        index = checkpoint / (name + ".index.json")
        if weights.is_file():
            files.append(weights)
            break
        if index.is_file():
            files.append(index)
            files.extend(checkpoint / f for f in sorted(set(json.loads(index.read_text())["weight_map"].values())))
            break
    else:
        raise FileNotFoundError(f"No native WEST checkpoint weights at {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        with path.open("rb") as f:
            digest.update(path.name.encode() + b"\0" + hashlib.file_digest(f, "sha256").digest())
    return digest.hexdigest()


def generated_hypothesis(ids, eos_ids, limit):
    # The WEST generation path supplies inputs_embeds only. Transformers returns
    # only continuation IDs (including EOS), even when the prompt is many tokens long.
    check(0 < len(ids) <= limit, "Unexpected generated width for inputs_embeds-only TouchASU")
    for i, token in enumerate(ids):
        if token in eos_ids:
            return ids[:i], False
    return ids, len(ids) == limit


def decode(checkpoint, dataset_path, rows, config, device):
    import torch
    from transformers import AutoModel, GenerationConfig
    from west.dataset.dataset import DataArguments, SpeechDataset
    from west.dataset.extractor import Extractor

    model, info = AutoModel.from_pretrained(checkpoint, torch_dtype="auto", output_loading_info=True)
    check(model.model_type == "touch_asu", f"Expected native TouchASU at {checkpoint}")
    for name in ("error_msgs", "missing_keys", "mismatched_keys", "unexpected_keys"):
        check(not info.get(name), f"Incomplete TouchASU checkpoint {checkpoint}, {name}={info.get(name)}")
    model.to(device).eval()
    tokenizer = model.init_tokenizer()
    generation = GenerationConfig(**config)
    generation.pad_token_id = tokenizer.pad_token_id
    eos_ids = config["eos_token_id"]
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    extractor = Extractor.get_class(model.model_type)(tokenizer, model.config, inference=True)
    dataset = SpeechDataset(extractor, DataArguments(data_path=str(dataset_path), batch_size=1))
    with torch.inference_mode():
        for index, row in enumerate(rows):
            # FireRed's original fbank uses dither. Each utterance gets the same
            # CPU noise across checkpoints regardless of the adapter initialization.
            torch.random.default_generator.manual_seed(FEATURE_SEED + index)
            item = extractor.extract({"wav": row["wav"], "txt": ""})
            check(item is not None, f"Extractor skipped {row['id']}")
            batch = {k: v.to(device) for k, v in dataset._batch([item]).items() if k != "labels"}
            # Qwen3 defaults to sampling. In Transformers 4.52 even an explicit
            # GenerationConfig(do_sample=False) would otherwise inherit that default.
            generated = model.generate(**batch, generation_config=generation, use_model_defaults=False)
            check(generated.ndim == 2 and generated.size(0) == 1, f"Unexpected generated shape for {row['id']}")
            ids = generated[0].tolist()
            tokens, capped = generated_hypothesis(ids, eos_ids, generation.max_new_tokens)
            yield dict(id=row["id"], txt=tokenizer.decode(tokens, skip_special_tokens=True),
                       generated_tokens=len(ids), capped_without_eos=capped)


def score_rows(rows, hypotheses, limit):
    check(len(rows) == len(hypotheses) and
          all(row["id"] == hyp["id"] for row, hyp in zip(rows, hypotheses)),
          "Prediction IDs must cover the entire manifest in order")
    for h in hypotheses:
        check(isinstance(h["txt"], str) and type(h["capped_without_eos"]) is bool and
              0 < h["generated_tokens"] <= limit and
              (not h["capped_without_eos"] or h["generated_tokens"] == limit),
              f"Invalid generated continuation for {h['id']}")
    score = asr_metrics([r["txt"] for r in rows], [h["txt"] for h in hypotheses])
    count = sum(h["capped_without_eos"] for h in hypotheses)
    return dict(**score, capped_without_eos=count, capped_without_eos_rate=count / len(rows))


def evaluate(args, step, split, dest, expected_model_sha=None):
    from transformers import GenerationConfig

    path = args.data / f"{split}.jsonl"
    info = json.loads((args.data / "splits.json").read_text())[split]
    check(sha(path) == info["sha256"], f"Changed {split} manifest")
    rows = read_rows(path)
    check(len({r["id"] for r in rows}) == len(rows) == info["utterances"], f"Invalid IDs/count in {split}")
    config = json.loads(args.generation.read_text())
    limit = config.get("max_new_tokens")
    eos = config.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    check(GenerationConfig(**config).get_generation_mode().value == "greedy_search" and
          config.get("num_return_sequences", 1) == 1 and
          type(limit) is int and limit > 0 and eos and all(type(i) is int and i >= 0 for i in eos),
          "Expected greedy decoding, one beam, explicit EOS and a positive common token limit")
    checkpoint = args.checkpoints.resolve() / f"checkpoint-{step}"
    weights = model_sha(checkpoint)
    check(expected_model_sha is None or expected_model_sha == weights, "Selected checkpoint weights changed")
    identity = dict(step=step, split=split, checkpoint_sha256=weights, generation_sha256=sha(args.generation),
                    manifest_sha256=info["sha256"], max_new_tokens=limit, feature_seed=FEATURE_SEED,
                    generation_mode="greedy_search")
    if dest.exists():
        hypotheses = dest / "hypotheses.jsonl"
        score = dict(**identity, **score_rows(rows, read_rows(hypotheses), limit), hypotheses_sha256=sha(hypotheses))
        check(score == json.loads((dest / "metrics.json").read_text()), f"Changed evaluation in {dest}")
        return score
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "score"
        stage.mkdir()
        path_hyp = stage / "hypotheses.jsonl"
        with path_hyp.open("x", encoding="utf-8") as out:
            for hyp in decode(checkpoint, path, rows, config, args.device):
                print(json.dumps(hyp, ensure_ascii=False), file=out)
        metrics = dict(**identity, **score_rows(rows, read_rows(path_hyp), limit), hypotheses_sha256=sha(path_hyp))
        save(stage / "metrics.json", metrics)
        stage.rename(dest)
    return metrics


def run(args):
    steps = sorted(args.steps)
    check(len(set(steps)) == len(steps) and min(steps) > 0, "Expected distinct positive checkpoint steps")
    selection_path = args.output / "dev-selection.json"
    if args.stage == "dev":
        scores = [evaluate(args, step, "dev-clean", args.output / f"dev-clean/step-{step}") for step in steps]
    else:
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        scores = [json.loads((args.output / f"dev-clean/step-{step}/metrics.json").read_text()) for step in steps]
    generation_sha = sha(args.generation)
    dev_sha = sha(args.data / "dev-clean.jsonl")
    check(all(s["step"] == step and s["split"] == "dev-clean" and s["generation_sha256"] == generation_sha
              and s.get("generation_mode") == "greedy_search" and s["manifest_sha256"] == dev_sha
              and s["reference_words"] == scores[0]["reference_words"]
              and s["utterances"] == scores[0]["utterances"] for step, s in zip(steps, scores)),
          "Development scores must use the same complete split and generation settings")
    winner = min(scores, key=lambda s: (s["word_errors"], s["step"]))
    step = winner["step"]
    selection = dict(selected_step=step, candidates=steps, score=winner, generation_sha256=generation_sha,
                     manifest_sha256=dev_sha, splits_sha256=sha(args.data / "splits.json"),
                     selected_checkpoint=str(args.checkpoints.resolve() / f"checkpoint-{step}"))
    if selection_path.exists():
        check(selection == json.loads(selection_path.read_text()), "Changed development selection")
    else:
        save(selection_path, selection)
    if args.stage == "dev":
        return selection
    return evaluate(args, step, "test-clean", args.output / "test-clean", winner["checkpoint_sha256"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--corpus", type=Path, required=True)
    prep.add_argument("--train", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for command in ("dev", "test"):
        stage = sub.add_parser(command)
        stage.add_argument("--checkpoints", type=Path, required=True)
        stage.add_argument("--steps", type=int, nargs="+", required=True)
        stage.add_argument("--data", type=Path, required=True)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
        stage.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if args.stage == "prepare":
        result = prepare(args.corpus, args.train, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".eval.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
