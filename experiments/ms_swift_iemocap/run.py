"""Serial Swift SER run with development-only model selection and clean held-out evaluation."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from experiments.ms_swift_librispeech.compare import write,load
from experiments.ms_swift_librispeech.train import digest,load_best
from .evaluate import score

def run(cfgfile,data,out,resume=False):
    cfg=load(cfgfile)
    splits=load(data/'splits.json')
    snapshot=out/'config.json'
    if resume:
        if (not (out/'failure.json').exists() or load(out/'status.json')['stage']!='train'
                or snapshot.read_bytes()!=cfgfile.read_bytes()):
            raise ValueError('Expected an interrupted training run with identical configuration')
    else:
        out.mkdir(parents=True,exist_ok=False)
        snapshot.write_bytes(cfgfile.read_bytes())
    results={}
    def status(stage,method=None):
        write(out/'status.json',dict(stage=stage,method=method))
    try:
        for split in cfg['sessions']:
            if digest(data/(split+'.jsonl'))!=splits[split]['sha256']:
                raise ValueError('Changed manifest: '+split)
        for name in cfg['methods']:
            if resume and (out/name/'result.json').exists():
                results[name]=load(out/name/'result.json')
                continue
            status('train',name)
            checkpoints=sorted((out/name).glob('checkpoint-[0-9]*'),key=lambda p:int(p.name.split('-')[-1]))
            extra=['--resume',checkpoints[-1]] if resume and checkpoints else []
            with (out/(name+'.log')).open('a' if resume else 'x') as log:
                if extra:
                    print('Resuming from',checkpoints[-1],flush=True,file=log)
                subprocess.run([sys.executable,'-m','experiments.ms_swift_librispeech.train','--config',snapshot,
                               '--method',name,'--train',data/'train.jsonl','--dev',data/'dev.jsonl','--output',out/name,*extra],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            results[name]=load(out/name/'result.json')
        first=next(iter(results.values()))
        for arm in results.values():
            if first['init_sha256']!=arm['init_sha256'] or first['input_trajectory']!=arm['input_trajectory']:
                raise ValueError('Unmatched arms')
        write(out/'dev-selection.json',results)
        for split in cfg.get('evaluation_splits', ('dev','test')):
            for name,res in results.items():
                status('reload-'+split,name)
                with (out/(name+'-'+split+'.log')).open('x') as log:
                    subprocess.run([sys.executable,'-m',__spec__.name,'--config',snapshot,'--data',data,'--output',out,
                                    '--score',split,'--method',name],stdout=log,stderr=subprocess.STDOUT,check=True)
                if split=='dev':
                    observed=out/name/'reload-dev'/'predictions.jsonl'
                    original=out/name/'dev'/f"step-{res['best_step']}"/'hypotheses.jsonl'
                    if digest(observed)!=digest(original):
                        raise ValueError('Clean development replay mismatch: '+name)
        if 'test' in cfg.get('evaluation_splits', ('dev','test')):
            write(out/'comparison.json',{n:load(out/n/'reload-test'/'metrics.json') for n in results})
        status('complete')
        if resume:
            (out/'failure.json').rename(out/'prior-training-failure.json')
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),stage=load(out/'status.json') if (out/'status.json').exists() else None))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('config','data','output'):
        parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--method',choices=['lora','ta-lora','dora'])
    parser.add_argument('--score',choices=['dev','test'])
    parser.add_argument('--resume-run',action='store_true')
    a=parser.parse_args()
    if a.score:
        c=load(a.config)
        sel=load(a.output/'dev-selection.json')[a.method]
        m,t=load_best(c,a.method,Path(sel['best_checkpoint']))
        dest=a.output/a.method/('reload-'+a.score)
        result=score(m,t,a.data/(a.score+'.jsonl'),c['generation'],dest/'predictions.jsonl')
        if a.score=='test':
            from .diagnostic import same_text
            result['same_text_different_emotion']=same_text(c['archive'],c['sessions']['test'],dest/'predictions.jsonl')
        write(dest/'metrics.json',result)
    else:
        run(a.config.resolve(),a.data.resolve(),a.output.resolve(),a.resume_run)
