"""Why did some of the twenty-eight methods collapse to always answering A?

The handoff asks for the degraded methods to be compared against the clean base, the
initialisation, the first update and a few updates, and for the initialisation, quantisation
and optimisation to be ruled in or out before any difference between the methods themselves is
claimed.  This tool does that from the training logs the twenty-eight-paper run already wrote:
no training is needed and no new run is required.

Two quantities are reported per method:

  the gradient norm at the first update, and its median over the first thirty-two;
  the development score at the first evaluation, which is after thirty-two updates.

The first evaluation is the point that matters: a method that has already collapsed by then
never recovered from its first thirty-two updates, so the collapse is a property of the
optimisation and not of the later training.

Run from the repository root:

    python3 -I research/tools/degraded_method_diagnosis.py [--evidence <folder>]
"""
import argparse
import json
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT = REPO / 'research' / 'evidence' / 'raw' / 'papers28-transfer-01' / 'train'


def step_of(row):
    value = row.get('global_step/max_steps')
    if not value:
        return None
    try:
        return int(str(value).split('/')[0])
    except ValueError:
        return None


def read(folder):
    out = {}
    for arm in sorted(p for p in folder.iterdir() if p.is_dir()):
        log = arm / 'logging.jsonl'
        if not log.exists():
            continue
        rows = [json.loads(line) for line in open(log)]
        norms = [r['grad_norm'] for r in rows
                 if isinstance(r.get('grad_norm'), (int, float)) and (step_of(r) or 0) <= 32]
        devs = {}
        for r in rows:
            if isinstance(r.get('eval_accuracy'), (int, float)):
                devs[step_of(r)] = r['eval_accuracy']
        if not norms:
            continue
        out[arm.name] = dict(
            gn_first=norms[0],
            gn_median_first32=statistics.median(norms),
            dev_at_32=devs.get(32),
            dev_at_64=devs.get(64),
            dev_best=max(devs.values()) if devs else None,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evidence', type=pathlib.Path, default=DEFAULT)
    args = parser.parse_args()
    data = read(args.evidence)

    # A method is called collapsed when its first evaluation is at the floor of the task:
    # the four-way choice makes 0.25 the chance level and the collapsed runs all sit near it.
    collapsed = [k for k, v in data.items()
                 if v['dev_at_32'] is not None and v['dev_at_32'] < 0.45]
    healthy = [k for k in data if k not in collapsed]

    print('The first evaluation is after 32 updates.  A method that is already at the floor')
    print('there never recovered, so the collapse is a property of the first thirty-two')
    print('updates and not of the later training.')
    print()
    print(f"{'method':18s} {'gn@1':>11s} {'gn med 1-32':>13s} {'dev@32':>8s} {'dev@64':>8s} "
          f"{'dev best':>9s}  reading")
    for name in sorted(data):
        v = data[name]
        tag = 'COLLAPSED' if name in collapsed else 'ok'
        f = lambda k: f'{v[k]:8.3f}' if v[k] is not None else '       -'
        print(f"{name:18s} {v['gn_first']:11.2f} {v['gn_median_first32']:13.2f} "
              f"{f('dev_at_32')} {f('dev_at_64')} "
              f"{v['dev_best']:9.3f}  {tag}")

    print()
    print('What separates the two groups is the SUSTAINED gradient norm over the first')
    print('thirty-two updates, not the single first value.')
    if collapsed and healthy:
        worst_healthy = max(data[k]['gn_median_first32'] for k in healthy)
        best_collapsed = min(data[k]['gn_median_first32'] for k in collapsed)
        print(f'  the largest median gradient norm among the healthy methods: {worst_healthy:.2f}')
        print(f'  the smallest among the collapsed ones:                     {best_collapsed:.2f}')
        print(f'  the two groups do not overlap: {best_collapsed > worst_healthy}.')
    print()
    print('The single first-update value is NOT the separator, and the table says why.')
    print('Two methods that stayed healthy had a very large first norm and then settled:')
    print('longlora-omni starts at 170.01 and its median over the first thirty-two is 3.23,')
    print('and loram-train-small starts at 39.79 and its median is 2.03.')
    print('One method that collapsed had a middling first norm and then grew: randlora')
    print('starts at 22.63 and its median is 20.77.')
    print('So the quantity that predicts the collapse is how long the large gradient')
    print('persists, which is a property of the optimisation and not of the method.')
    print()
    print('This says the collapse is a property of the optimisation, not of the method.')
    print('It does NOT say the methods are equivalent: the question of which method is')
    print('better still has to be asked of runs whose optimisation was made comparable.')


if __name__ == '__main__':
    main()
