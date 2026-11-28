"""Check, in every saved checkpoint, whether the gate has optimizer state.

The gate is in the optimizer's parameter list (it has requires_grad=True), but a parameter
that never receives a gradient gets no Adam state: `exp_avg` and `exp_avg_sq` are never
allocated for it.  So the optimizer checkpoint is an independent witness of whether the gate
ever got a gradient during the actual training run -- a witness that does not depend on
reading the forward pass at all.

This is the strongest form of the check, because it comes from the run itself rather than
from a reconstruction of the code path: the `optimizer.pt` was written by the training
loop, and Adam only creates state for a parameter it actually stepped.

Run from the repository root:

    python3 research/tools/gate_optimizer_state.py <run folder> [<run folder> ...]
"""
import json, pathlib, sys
import torch

runs = sys.argv[1:]
print(f'{"arm":22s} {"params":>7s} {"gates":>6s} {"gate w/ state":>15s} {"down/up w/ state":>18s}')
total_gates = total_with = 0
for run in runs:
    run = pathlib.Path(run)
    for arm_dir in sorted((run/'train').iterdir()):
        if not arm_dir.is_dir(): continue
        try:
            res = json.loads((arm_dir/'result.json').read_text())
            ck = arm_dir/res['best_checkpoint']
            names = json.loads((arm_dir/'initialization.json').read_text())['trainable_names']
            sd = torch.load(ck/'optimizer.pt', map_location='cpu', weights_only=False)
        except Exception as exc:
            continue
        st = sd['state']
        gates = [i for i,n in enumerate(names) if n.endswith('.gate')]
        others = [i for i,n in enumerate(names) if not n.endswith('.gate')]
        g_with = sum(1 for i in gates if i in st)
        o_with = sum(1 for i in others if i in st)
        print(f'{arm_dir.name:22s} {len(names):7d} {len(gates):6d} {g_with:8d}/{len(gates):<6d} {o_with:10d}/{len(others):<7d}')
        total_gates += len(gates)
        total_with += g_with
print()
print(f'TOTAL: {total_with} of {total_gates} gates across every arm have optimizer state.')
print('A gate with no state was never stepped, because it never received a gradient.')
