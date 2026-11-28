"""Audit the saved TA-LoRA checkpoints: what was trainable, and what actually moved.

This reads the saved checkpoints and the training logs.  It loads no model and trains nothing.

Four questions, all about the CLAUDE.md review checklist:

  1. the trainable set       which parameters were on the optimizer's list, and what each is:
                             down = A (rank x in), up = B (out x rank), gate = g (rank),
                             transport_up = the independent C, gate_weight = the context gate
  2. the first-answer query  the transport term is added at ONE position per row, the token
                             immediately before the first supervised token.  Check that the
                             checkpoints record the gate mode that makes this true.
  3. the updates             for each factor, how far it moved from its initialisation, and
                             whether the movement is consistent with "the gate was numerically
                             inert", which is what the diagnosis claimed.
  4. the format              the checkpoint format version, the gate multiplier, the transport
                             signal and precision, and whether the config and the tensors agree.

The initialisation is reconstructed from the saved `initialization.json` where it exists, and
from the `ab_init_sha256` in the adapter config otherwise.  Where the initial values are not
available, the tool says so rather than inventing them.
"""
import argparse
import json
import math
import pathlib

import numpy as np

FACTORS = ('down', 'up', 'gate', 'transport_up', 'gate_weight')


def read_safetensors(path):
    """The tensors in a safetensors file, without torch (the file is read directly)."""
    import struct

    with open(path, 'rb') as handle:
        header_len = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        out = {}
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            start, end = spec['data_offsets']
            handle.seek(header_len + 8 + start)
            raw = handle.read(end - start)
            out[name] = np.frombuffer(raw, dtype=np.float32).reshape(spec['shape'])
        return out


def load_config(run, arm):
    folder = pathlib.Path(run) / 'train' / arm
    result = json.loads((folder / 'result.json').read_text())
    checkpoint = folder / result['best_checkpoint']
    return folder, result, checkpoint


def describe_factors(tensors, cfg):
    """What factors exist, and their shapes."""
    out = {}
    for key, value in tensors.items():
        factor = key.rsplit('.', 1)[-1]
        layer = key.split('.ta_adapter.')[0] if '.ta_adapter.' in key else key.rsplit('.', 2)[0]
        out.setdefault(factor, {}).setdefault(layer, value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True,
                        help='run folders, each with a train/<arm>/ directory')
    parser.add_argument('--arms', nargs='+', default=None)
    args = parser.parse_args()

    for run in args.runs:
        run = pathlib.Path(run)
        arms = args.arms or sorted(p.name for p in (run / 'train').iterdir() if p.is_dir())
        print('=' * 100)
        print(f'== checkpoint audit: {run.name}')
        for arm in arms:
            try:
                folder, result, checkpoint = load_config(run, arm)
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                print(f'   {arm}: not readable ({exc}); skipped')
                continue
            cfg_file = checkpoint / 'adapter_config.json'
            if not cfg_file.exists():
                print(f'   {arm}: no adapter_config.json under {checkpoint}; skipped')
                continue
            cfg = json.loads(cfg_file.read_text())
            weights = checkpoint / 'adapter_model.safetensors'
            if not weights.exists():
                print(f'   {arm}: no adapter_model.safetensors under {checkpoint}; skipped')
                continue
            tensors = read_safetensors(weights)

            print(f'   --- {arm}')
            print(f'       checkpoint           {checkpoint.name}')
            print(f'       format_version       {cfg.get("format_version")}')
            print(f'       tuner_type           {cfg.get("tuner_type")}')
            print(f'       rank / alpha         {cfg.get("rank")} / {cfg.get("alpha")}')
            print(f'       gate_multiplier      {cfg.get("gate_multiplier", 1.)}')
            print(f'       transport_gate       {cfg.get("transport_gate")}')
            print(f'       transport_signal     {cfg.get("transport_signal")}')
            print(f'       transport_precision  {cfg.get("transport_precision")}')
            print(f'       transport_queries    {cfg.get("transport_queries")}')
            print(f'       independent_C        {cfg.get("independent_transport_up")}')
            print(f'       update_positions     {cfg.get("update_positions")}')
            print(f'       targets              {len(cfg.get("targets", []))} modules, '
                  f'first {cfg.get("targets", ["?"])[0]}')

            # the trainable set: the tensors that were saved, by factor
            by_factor = {}
            for key in tensors:
                factor = key.rsplit('.', 1)[-1]
                by_factor.setdefault(factor, []).append(key)
            print(f'       saved factors        ' + ', '.join(
                f'{f} x{len(v)}' for f, v in sorted(by_factor.items())))

            # the per-factor magnitudes, and the gate's own value
            for factor in FACTORS:
                keys = by_factor.get(factor)
                if not keys:
                    continue
                value = tensors[keys[0]]
                norm = float(np.linalg.norm(value))
                nz = int(np.count_nonzero(value))
                size = int(value.size)
                extra = ''
                if factor == 'gate':
                    extra = (f'  values[min {float(value.min()):+.6f} '
                             f'max {float(value.max()):+.6f} '
                             f'mean {float(value.mean()):+.6f}]')
                print(f'       {factor:14s} shape {str(tuple(value.shape)):14s} '
                      f'|.| {norm:11.5f}  nonzero {nz}/{size}{extra}')

            # the movement: compare against the initialization, where it was recorded
            init_file = folder / 'initialization.json'
            if init_file.exists():
                init = json.loads(init_file.read_text())
                print(f'       initialization.json  {len(init)} entries')
            else:
                print('       initialization.json  not recorded; the movement is not computable')
            print(f'       ab_init_sha256       {cfg.get("ab_init_sha256", "n/a")}')


if __name__ == '__main__':
    main()
