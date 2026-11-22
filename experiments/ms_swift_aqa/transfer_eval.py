"""Clean-reload a transfer candidate on development data or the fixed public MMAR protocol."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import torch

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest, load_best
from .evaluate import score
from .evaluate_frozen import load_frozen
from .evaluate_mmar import METADATA_SHA256, SCORER_SHA256, jsonl
from .summarize_mmar import score_arm


@torch.no_grad()
def disable_transport_for_eval(model):
    """Zero contrast readouts on a freshly loaded, frozen TA model."""
    if model.training or getattr(model, '_ta_lora_config', {}).get('tuner_type') != 'ta_lora':
        raise ValueError('Transport ablation requires a TA-LoRA model in evaluation mode')
    branches = [layer.self_attn.ta_adapter for layer in model.thinker.model.layers
                if hasattr(layer.self_attn, 'ta_adapter')]
    gates = [branch.gate for branch in branches]
    readouts = [branch.transport_up for branch in branches if branch.transport_up is not None]
    tensors = gates + readouts
    if not gates or any(p.requires_grad for p in tensors):
        raise ValueError('Transport ablation requires frozen TA parameters')
    for p in tensors:
        p.zero_()
    return dict(gate_tensors=len(gates), transport_up_tensors=len(readouts),
                tensors_zeroed=len(tensors), scalars_zeroed=sum(p.numel() for p in tensors))


def evaluate(config, method, checkpoint, manifest, folder, mmar_run=None, no_transport=False):
    if no_transport and method != 'ta-lora':
        raise ValueError('--no-transport requires --method ta-lora')
    cfg = load(config)
    metrics_file = folder / 'metrics.json'
    if metrics_file.exists():
        raise FileExistsError(metrics_file)
    folder.mkdir(exist_ok=True, parents=True)
    identity = {'base_model': str(Path(cfg['model']).resolve()), 'model_revision': cfg['model_revision'], 'config_sha256': digest(config)}
    if mmar_run:
        protocol = load(mmar_run / 'protocol.json')
        source = Path(protocol['dataset']) / 'MMAR-meta.jsonl'
        script = mmar_run / 'official/evaluation.py'
        if (cfg['model_revision'] != protocol['model_revision'] or cfg['generation']['batch_size'] != protocol['eval_batch_size']
                or digest(manifest) != protocol['manifest_sha256'] or digest(source) != METADATA_SHA256
                or digest(script) != SCORER_SHA256):
            raise ValueError('The official MMAR protocol has changed')
    if method == 'frozen-base':
        if checkpoint:
            raise ValueError('Frozen controls cannot load an adapter')
        model, template = load_frozen(cfg)
    else:
        if not checkpoint:
            raise ValueError('A selected checkpoint is required')
        identity.update(checkpoint=str(checkpoint.resolve()), weights_sha256=digest(checkpoint/'adapter_model.safetensors'),
                        adapter_config_sha256=digest(checkpoint/'adapter_config.json'))
        model, template = load_best(cfg, method, checkpoint)
        if no_transport:
            identity['transport_ablation'] = disable_transport_for_eval(model)
    print(json.dumps(dict(method=method, no_transport=no_transport, identity=identity), sort_keys=True), flush=True)
    predictions = folder / 'predictions.jsonl'
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metrics = score(model, template, manifest, cfg['generation'], predictions, allow_empty_options=bool(mmar_run))
    torch.cuda.synchronize()
    metrics.update(method=folder.name, tuner=method, identity=identity, no_transport=no_transport, manifest_sha256=digest(manifest),
                   predictions_sha256=digest(predictions), eval_seconds=time.perf_counter()-start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write(metrics_file, metrics)
    if mmar_run:
        spec = importlib.util.spec_from_file_location('official_mmar', script)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        result, _ = score_arm(folder.parent, folder.name, protocol, jsonl(source), script, evaluator.string_match)
        print('MMAR', result['correct'], '/', result['total'])
    else:
        print(metrics['correct'], '/', metrics['utterances'], metrics.get('macro_domain_accuracy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'checkpoint', 'manifest', 'folder', 'mmar-run'):
        parser.add_argument('--' + key, type=Path, required=key in ('config','manifest','folder'))
    parser.add_argument('--method', required=True, choices=('frozen-base', 'ta-lora', 'lora', 'dora'))
    parser.add_argument('--no-transport', action='store_true',
                        help='Zero only TA transport gate and transport_up after loading the checkpoint')
    evaluate(**vars(parser.parse_args()))
