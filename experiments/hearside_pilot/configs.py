"""One training config per arm. The arms differ in exactly one setting each.

A  all          update at every position, fixed option order   (ordinary low-rank adaptation)
B  audio        update only at audio-token positions
C  post_audio   update only at positions after the audio
D  all          as A, trained and selected on shuffled option orders

The follow-up plan adds one arm on the last four layers and two more training seeds for A, B and D.
"""
import argparse
import copy
import json
from pathlib import Path

ARMS = {'A-all': ('all', 'fixed'), 'B-audio': ('audio', 'fixed'),
        'C-post-audio': ('post_audio', 'fixed'), 'D-all-shuffled': ('all', 'shuffled')}
ADAPTER = dict(rank=8, alpha=16, last_n=36, gate_multiplier=0.0)   # all Thinker layers, O projection; no transport term
TRAINING = dict(eval_steps=88, save_steps=88)                      # three checks in one epoch of Sessions 1-2
PATIENCE = 3                                                       # never stops early: the best of the three is kept

# Follow-up (declared 2026-11-23, after the pilot): each arm changes one thing in a pilot arm.
FOLLOWUP = {'A-last4': dict(like='A-all', last_n=4),               # the depth of the earlier pair-01 adapter, in this pipeline
            **{f'{arm}-s{seed}': dict(like=arm, seed=seed)         # the seed sets the adapter initialisation and the data order
               for seed in (43, 44) for arm in ('A-all', 'B-audio', 'D-all-shuffled')}}
LAST4 = {name: dict(like='A-all', last_n=4, seed=seed) for name, seed in (('A-last4-s43', 43), ('A-last4-s44', 44))}
# Layer bands (declared 2026-11-25): four layers at the start and in the middle of the 36, as A-last4 otherwise.
BANDS = {f'{arm}{suffix}': dict(like='A-all', last_n=4, first_layer=first, seed=seed)
         for arm, first in (('A-first4', 0), ('A-mid4', 16)) for suffix, seed in (('', 42), ('-s43', 43), ('-s44', 44))}
PLANS = {'pilot': {name: dict(like=name) for name in ARMS}, 'followup': FOLLOWUP, 'last4-seeds': LAST4, 'layer-bands': BANDS}


def arm_config(base, positions, last_n=None, seed=None, first_layer=None):
    cfg = copy.deepcopy(base)
    cfg.pop('methods', None)
    cfg['adapter'] = dict(ADAPTER, update_positions=positions)
    cfg['training'] = dict(cfg['training'], **TRAINING)
    if last_n is not None:
        cfg['adapter']['last_n'] = last_n
    if first_layer is not None:
        cfg['adapter']['first_layer'] = first_layer          # otherwise the last_n layers are the last ones
    if seed is not None:
        cfg['training'].update(seed=seed, data_seed=seed)
    cfg['patience'] = PATIENCE
    if cfg.get('task') != 'ser' or cfg.get('objective', 'sft') != 'sft':
        raise ValueError('The pilot expects the supervised emotion recipe with the ordinary loss')
    return cfg


def build(base_config, data, output, plan='pilot'):
    base, data, output = json.loads(Path(base_config).read_text()), Path(data).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for name, change in PLANS[plan].items():
        positions, version = ARMS[change['like']]
        cfg = arm_config(base, positions, change.get('last_n'), change.get('seed'), change.get('first_layer'))
        path = output / f'{name}.json'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        arms[name] = dict(config=str(path.resolve()), update_positions=positions, option_order=version,
                          train=str(data / f'train-{version}.jsonl'), dev=str(data / f'dev-{version}.jsonl'),
                          like=change['like'], last_n=cfg['adapter']['last_n'], seed=cfg['training']['seed'],
                          **({'first_layer': change['first_layer']} if 'first_layer' in change else {}))
    (output / 'arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    return arms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True, help='An IEMOCAP training config with this server\'s model path')
    parser.add_argument('--data', type=Path, required=True, help='Output folder of experiments.hearside_pilot.data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', choices=sorted(PLANS), default='pilot')
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.data, args.output, args.plan), indent=2))
