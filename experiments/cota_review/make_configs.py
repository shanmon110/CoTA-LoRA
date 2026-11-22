"""Generate the pre-declared cota-review-01 experiment configs.

    python experiments/cota_review/make_configs.py            # rewrite configs/ and matrix.json
    python experiments/cota_review/make_configs.py --check    # fail if the committed files are stale

Each config is a complete input for ``python -m experiments.ms_swift_librispeech.train``. Every arm
differs from its reference arm in exactly the fields listed under ``changes`` in matrix.json. Paths
are the ones recorded by the legacy runs (research/evidence/raw/ta-conditional-01/configs); change
BASE below, not the generated files, when the server layout changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIDDEN, LAYERS = 2048, 36  # Qwen2.5-Omni-3B Thinker: o_proj is 2048 x 2048 in each of 36 layers

BASE = {
    'generation': {'batch_size': 2},
    'methods': ['lora', 'dora', 'ta-lora'],
    'model': '/root/autodl-tmp/west-preflight/model',
    'model_revision': 'f75b40e3da2003cdd6e1829b1f420ca70797c34e',
    'model_type': 'qwen2_5_omni',
    'objective': 'sft',
    'snapshot_manifest': 'experiments/ms_swift_librispeech/evidence/verified_snapshot.json',
    'swift_source': '/root/autodl-tmp/ms-swift-librispeech/src/ms-swift',
    'task': 'aqa',
    'selection_metric': 'pair_accuracy',
    'diagnostics': {'adapter_stats_every': 8},
}

# The legacy protocol, byte-for-byte the training block of raw/ta-conditional-01/configs/*-mixed.json.
LEGACY_TRAINING = {
    'data_seed': 42, 'dataloader_num_workers': 0, 'disable_tqdm': True, 'eval_steps': 32,
    'gradient_accumulation_steps': 4, 'gradient_checkpointing': False, 'learning_rate': 0.0002,
    'logging_steps': 1, 'lr_scheduler_type': 'constant', 'max_length': 2048, 'num_train_epochs': 1,
    'optim': 'adamw_torch', 'per_device_eval_batch_size': 2, 'per_device_train_batch_size': 4,
    'save_steps': 32, 'seed': 42, 'warmup_ratio': 0, 'weight_decay': 0.0,
}
# Protocol v2: three epochs with warmup and linear decay, evaluated three times per epoch. Patience is
# set above the number of evaluations so nothing stops early; the pre-declared primary checkpoint is
# the last one (update 504). See research/claude-review/PROTOCOL.md for the reasons.
V2_TRAINING = dict(LEGACY_TRAINING, eval_steps=56, save_steps=56, lr_scheduler_type='linear',
                   num_train_epochs=3, warmup_ratio=0.05)

O_ONLY = {'last_n': 36, 'target_modules': ['o_proj'], 'gate_init': 0.0, 'gate_multiplier': 1.0,
          'transport_queries': 'text', 'transport_weighting': 'uniform'}


def adapter(rank, **extra):
    return dict(O_ONLY, rank=rank, alpha=2 * rank, **extra)


COTA = dict(independent_transport_up=True)
ARMS = {
    # name: (method, adapter, reference arm, the single factor that differs from the reference)
    'lora-r12': ('lora', adapter(12, **COTA), None, 'matched-budget LoRA baseline'),
    'lora-r8': ('lora', adapter(8, **COTA), 'lora-r12', 'ordinary rank 12 -> 8'),
    'cota-static': ('ta-lora', adapter(8, **COTA, transport_gate='static'), 'lora-r12',
                    'rank 12 -> rank 8 + answer-query contrast readout C (same budget)'),
    'ctrl-ordinary': ('ta-lora', adapter(8, **COTA, transport_gate='static', transport_signal='ordinary_matched'),
                      'cota-static', 'readout input: audio/text contrast -> norm-matched ordinary attention output'),
    'ctrl-audio-mean': ('ta-lora', adapter(8, **COTA, transport_gate='static', transport_signal='audio_mean'),
                        'cota-static', 'readout input: E[V|audio]-E[V|text] -> E[V|audio]'),
    'cota-qa': ('ta-lora', adapter(8, **COTA, transport_gate='question_audio'), 'cota-static',
                'adds the conditional gate R exactly as in the legacy runs'),
    'cota-qa-live': ('ta-lora', adapter(8, **COTA, transport_gate='question_audio', transport_precision='fp32',
                                        transport_gate_lr_multiplier=16), 'cota-qa',
                     'gate made numerically effective: FP32 transport path and 16x gate learning rate'),
    'shared-r12': ('ta-lora', adapter(12, transport_gate='static', gate_init=1.0), 'lora-r12',
                   'adds answer-query transport through the ordinary readout B only (g fixed start 1, no C)'),
}


def parameters(method, cfg):
    rank = cfg['rank']
    per_layer = 2 * rank * HIDDEN
    if method == 'ta-lora':
        per_layer += rank
        if cfg.get('independent_transport_up'):
            per_layer += rank * HIDDEN
        if cfg.get('transport_gate') in ('question', 'question_audio'):
            per_layer += rank * rank
    return per_layer * LAYERS


def build(arm, training, seed, patience, learning_rate=None):
    method, cfg, _, _ = ARMS[arm]
    cfg = copy.deepcopy(cfg)  # native LoRA ignores the transport keys; they are kept as in the legacy files
    block = dict(training, seed=seed, data_seed=seed)
    if learning_rate is not None:
        block['learning_rate'] = learning_rate
    return dict(BASE, adapter=cfg, training=block, patience=patience)


def plan():
    files = {}
    for seed in (42, 43, 44, 45, 46):  # stage 0: legacy protocol, seed noise of the published comparison
        for arm in ('lora-r12', 'cota-qa'):
            files[f'stage0-legacy/{arm}-s{seed}.json'] = build(arm, LEGACY_TRAINING, seed, 3)
    for seed in (42, 43, 44):
        for rate in (5e-5, 1e-4, 2e-4, 4e-4):  # stage 1: same four learning rates for both headline arms
            for arm in ('lora-r12', 'cota-static'):
                files[f'stage1-lr/{arm}-lr{rate:g}-s{seed}.json'] = build(arm, V2_TRAINING, seed, 100, rate)
        for arm in ARMS:  # stage 2: regenerate with --stage2-lr once stage 1 has fixed it on dev
            files[f'stage2-arms/{arm}-s{seed}.json'] = build(arm, V2_TRAINING, seed, 100)
    return files


def render(config):
    return json.dumps(config, indent=2, sort_keys=True) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--stage2-lr', type=float, help='learning rate chosen on dev in stage 1')
    args = parser.parse_args()
    files = plan()
    if args.stage2_lr is not None:
        for name, config in files.items():
            if name.startswith('stage2-arms/'):
                config['training']['learning_rate'] = args.stage2_lr
    matrix = dict(
        protocol='cota-review-01',
        arms={name: dict(method=method, trainable_parameters=parameters(method, cfg), reference=reference,
                         changes=changes, adapter=cfg)
              for name, (method, cfg, reference, changes) in ARMS.items()},
        configs={name: hashlib.sha256(render(config).encode()).hexdigest() for name, config in sorted(files.items())})
    outputs = {HERE / 'configs' / name: render(config) for name, config in files.items()}
    outputs[HERE / 'matrix.json'] = render(matrix)
    if args.check:
        stale = [str(path.relative_to(HERE)) for path, text in outputs.items()
                 if not path.is_file() or path.read_text() != text]
        if stale:
            raise SystemExit('stale generated files: ' + ', '.join(stale))
        print(f'{len(outputs)} generated files are current')
        return
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(f'wrote {len(outputs)} files')


if __name__ == '__main__':
    main()
