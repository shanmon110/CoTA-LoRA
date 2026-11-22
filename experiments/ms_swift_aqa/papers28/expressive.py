"""Replay the theory paper's two constructive examples on CPU, checking upstream authors' bytes."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


REVISION = 'eaeb7d3f265853343b234172ed44cd7623c5212a'
URL = f'https://raw.githubusercontent.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA/{REVISION}/'
FILES = {
    'run_exp.py': '9d7db4f92b1f11744edd5109dc915ce5e7477185edca6bd8194e3753650aab6a',
    'load_model.py': '5bb6d03cb8a7e27ebf26c55679e7dfcef929719df685eaed9d9de366ca16dc3a',
    'helper.py': '788e3c40822977fa90974f14dfb83f0c83c75c8f0760f21414e7d6eaab530a04',
}
FNN = ('--width 8 --target_depth 1 --frozen_depth 2 --rank 3 --use_bias 1 --activation relu --std 0.25 '
       '--method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 '
       '--init_mode uniform_singular_values --exp fnn --wandb 0')
TFN = ('--exp tfn --width 8 --target_depth 1 --frozen_depth 1 --rank 4 --n_head 2 --batch_size 256 '
       '--seq_length 10 --method ours --seed 123 --std 0.25 --n_test 5000 --wandb 0')


def replay(output, cached_sources=None, extra_dependencies=None):
    output.mkdir(parents=True, exist_ok=False)
    official = output/'official'
    official.mkdir()
    for name,digest in FILES.items():
        if cached_sources:
            raw = (cached_sources/name).read_bytes()
        else:
            with urlopen(URL+name, timeout=30) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Unexpected author source bytes for {name}')
        (official/name).write_bytes(raw)
    source = (official/'run_exp.py').read_text()
    # Two bookkeeping typos in the upstream toy-transformer path. The construction and
    # MSE evaluation are identical. Preserve the original + minimal diff and full precision.
    patched = source.replace('if self.best_optim:', 'if self.better_optim:')
    old = "self.test_loss = loss.item()\n        \n        if self.wandb: wandb.log({'test_shift_loss': self.test_loss})"
    new = old.replace('self.test_loss', 'self.test_shift_loss')
    if patched == source or patched.count(old)!=1:
        raise ValueError('Unexpected upstream transformer diagnostic code')
    patched = patched.replace(old,new)
    (output/'transformer.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(keepends=True), patched.splitlines(keepends=True), fromfile='run_exp.py',tofile='run_tfn.py')))
    (output/'run_tfn.py').write_text(patched)
    paths = [str(official.resolve())]
    if extra_dependencies:
        paths.append(str(extra_dependencies.resolve()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', OMP_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join(paths+[os.environ.get('PYTHONPATH','')]))
    scores = {}
    for kind,script,args in [('fnn',official/'run_exp.py',FNN),('tfn',output/'run_tfn.py',TFN)]:
        cmd=[sys.executable,str(script),*args.split()]
        with (output/f'{kind}.log').open('w') as stream:
            subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        lines=(output/f'{kind}.log').read_text().splitlines()
        scores[kind] = {'command':cmd,'reported_MSE':{
            k:float(next(row.rsplit(' ',1)[1] for row in lines if row.startswith(prefix)))
            for k,prefix in [('normal','Test loss:'),('shifted','Test loss on shifted test set:')]}}
    (output/'result.json').write_text(json.dumps({'author_revision':REVISION,'source_hashes':FILES,'runs':scores},
                                                indent=2)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--cached-sources',type=Path)
    cli.add_argument('--extra-dependencies',type=Path)
    replay(**vars(cli.parse_args()))
