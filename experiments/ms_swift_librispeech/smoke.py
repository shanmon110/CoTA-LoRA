"""Exercise real Swift training interruption, resume, early selection and clean dev reload."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .compare import PACKAGE, load, write


def smoke(config, data, out):
    out.mkdir(parents=True, exist_ok=False)
    cfg = load(config)
    cfg['training'].update(max_steps=3, gradient_accumulation_steps=1, eval_steps=1, save_steps=1)
    cfg['generation']['max_new_tokens'] = 3
    cfg_path = out/'config.json'
    write(cfg_path,cfg)
    dev = out/'smoke-dev.jsonl'
    dev.write_text('\n'.join((data/'smoke-dev-clean.jsonl').read_text().splitlines()[:2])+'\n')
    rows={}
    for name in cfg['methods']:
        arm = out/name
        cmd = [sys.executable,'-m',PACKAGE+'.train','--config',cfg_path,'--method',name,'--train',
               data/'smoke-train.jsonl','--dev',dev,'--output',arm]
        for phase,extra in [('first',['--stop-after','1']),('resume',['--resume', arm/'checkpoint-1'])]:
            with (out/f'{name}-{phase}.log').open('x') as log:
                subprocess.run(cmd+extra,stdout=log,stderr=subprocess.STDOUT,check=True)
        res=load(arm/'result.json')
        cp=Path(res['best_checkpoint'])
        target=out/(name+'-reloaded-dev')
        with (out/f'{name}-reload.log').open('x') as log:
            subprocess.run([sys.executable,'-m',PACKAGE+'.evaluate','--method',name,'--config',cfg_path,
                            '--checkpoint',cp,'--manifest',dev,'--output',target],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        best_dev=arm/'dev'/('step-'+str(res['best_step']))
        if ((target/'predictions.jsonl').read_bytes() != (best_dev/'hypotheses.jsonl').read_bytes()
                or load(target/'metrics.json')['wer'] != res['best_wer'] or res['global_step'] < 2):
            raise ValueError(f'{name}: best-checkpoint clean replay mismatch')
        rows[name]=res
    if rows['lora']['init_sha256'] != rows['ta-lora']['init_sha256']:
        raise ValueError('Unequal initialization across smoke arms')
    write(out/'verified.json', rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    smoke(args.config.resolve(),args.data.resolve(),args.output.resolve())
