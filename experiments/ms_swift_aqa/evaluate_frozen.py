"""Score the unfine-tuned Omni base with the same prompts and constrained decoder as the SFT arms."""
import argparse
import os
from pathlib import Path
import time

import torch

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest, model_options
from .evaluate import score
from .summarize_pilot import load_official, read_jsonl


def load_frozen(cfg):
    from peft import PeftModel
    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template

    os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
    opts = model_options(cfg, 'lora')  # Includes verification of the entire base and processor snapshot.
    args = InferArguments(**{k: opts[k] for k in ('model', 'model_type', 'use_hf', 'attn_impl', 'torch_dtype')},
                          max_length=cfg['training']['max_length'], adapters=[], load_args=False, load_data_args=False)
    model, template = prepare_model_template(args)
    if isinstance(model, PeftModel) or args.adapters:
        raise ValueError('The frozen control must load the base without adapters')
    return model.eval().requires_grad_(False), template


def evaluate(run):
    cfg = load(run/'base.json')
    plan = load(run/'plan.json')
    data = Path(plan['data'])
    hashes = load(run/'manifests.json')
    dest = run/'frozen-base'
    if any((dest/split/'metrics.json').exists() for split in ('dev', 'test')):
        raise FileExistsError(f'Frozen predictions already recorded in {dest}')
    model, template = load_frozen(cfg)
    for split in ('dev', 'test'):
        manifest = data/f'{split}.jsonl'
        if digest(manifest) != hashes[split]:
            raise ValueError(f'Frozen {split} manifest changed')
        predictions = dest/split/'predictions.jsonl'
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        metrics = score(model, template, manifest, cfg['generation'], predictions)
        torch.cuda.synchronize()
        metrics.update(eval_seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), manifest_sha256=hashes[split],
                       predictions_sha256=digest(predictions), revision=cfg['model_revision'], trainable_parameters=0)
        if split == 'test':
            test, source, _, official = load_official(data, Path(plan['mmau']))
            strict, official_correct = 0, 0
            for row, gold, pred in zip(test, source, read_jsonl(predictions), strict=True):
                chosen = gold['choices'][ord(pred['prediction']) - ord('A')]
                if (row['id'] != gold['id'] or pred['id'] != gold['id'] or row['choices'] != gold['choices']
                        or chosen != pred['answer_text'] or (chosen == gold['answer']) != pred['is_correct']):
                    raise ValueError(f"Frozen prediction disagrees with official item {gold['id']}")
                strict += int(chosen == gold['answer'])
                official_correct += int(official.string_match(gold['answer'], chosen, gold['choices']))
            if strict != metrics['correct']:
                raise ValueError('Frozen score mismatch')
            metrics['official_text_accuracy'] = official_correct / len(source)
        write(predictions.parent/'metrics.json', metrics)
        print(f"{split}: {metrics['correct']}/{metrics['utterances']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    evaluate(parser.parse_args().run)
