"""Construct per-projection rank-r adapters from a dev-selected, same-data full teacher (paper 1)."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import torch
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.ms_swift_aqa.papers28.expressive import REVISION as AUTHOR_REVISION
from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity
from experiments.ms_swift_librispeech.train import digest


def read(path):
    return json.loads(Path(path).read_text())


def factors(frozen, teacher, rank, alpha):
    """Depth 1 of our_construction: B=(teacher-frozen)V_r/scale, A=V_r.T in PEFT notation."""
    if frozen.shape != teacher.shape or frozen.ndim != 2 or frozen.shape[0] != frozen.shape[1]:
        raise ValueError('Known-target depth-one square construction requires equal square matrices')
    if not 1 <= rank <= frozen.shape[0] or alpha <= 0:
        raise ValueError('Invalid rank or alpha')
    delta = teacher.float() - frozen.float()
    # Full LAPACK SVD, then retain r right singular vectors, as in the authors' implementation.
    _, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    a = vh[:rank].contiguous().clone()
    b = (delta @ a.T / (alpha / rank)).contiguous()
    energy = singular.double().square()
    total = energy.sum().item()
    retained = energy[:rank].sum().item()
    relative = math.sqrt(energy[rank:].sum().item()/total) if total else 0.
    return a,b,relative,total,retained


def tensor_names(base_keys, teacher_keys, depth):
    ordered = [f'thinker.model.layers.{layer}.self_attn.{name}.weight'
               for layer in range(depth) for name in ('q_proj','o_proj')]
    pattern = re.compile(r'thinker\.model\.layers\.\d+\.self_attn\.(q_proj|o_proj)\.weight')
    for keys in (base_keys,teacher_keys):
        if {k for k in keys if pattern.fullmatch(k)} != set(ordered):
            raise ValueError('Unexpected language-layer coverage of Q/O in checkpoint index')
    return ordered


def write_adapter(path, adapters, rank, alpha, base_model, revision, task_type='CAUSAL_LM'):
    path.mkdir()
    config = LoraConfig(base_model_name_or_path=base_model, revision=revision, r=rank, lora_alpha=alpha,
                        inference_mode=True, lora_dropout=0., bias='none', task_type=task_type,
                        target_modules='^(?:'+'|'.join(re.escape(n) for n in sorted(adapters))+')$')
    config.save_pretrained(path)
    weights = {f'base_model.model.{name}.lora_{side}.weight': tensor
               for name,(a,b) in adapters.items() for side,tensor in [('A',a),('B',b)]}
    save_file(weights, path/'adapter_model.safetensors', metadata={'format':'pt'})


def verified_inputs(protocol):
    full = Path(protocol['archive'])/'full'
    selection_path = full/'selection.json'
    selected = read(selection_path)
    if selected['method'] != 'full' or selected['step'] <= 0:
        raise ValueError('Expected a dev-selected full teacher')
    teacher = full/f"checkpoint-{selected['step']}"
    if teacher.resolve() != Path(selected['checkpoint']).resolve():
        raise ValueError('Selected checkpoint outside the declared archive')
    for key,value in checkpoint_identity(teacher,'full').items():
        if selected[key] != value:
            raise ValueError(f'Teacher checkpoint mismatch: {key}')
    trained = read(full/'protocol.json')
    model_cfg = read(protocol['base_config'])
    if (trained['method'] != 'full' or trained['config']['model_revision'] != protocol['model_revision']
            or model_cfg['model_revision'] != protocol['model_revision']):
        raise ValueError('Teacher protocol or model revision mismatch')
    data = Path(protocol['avqa_mmau'])
    for split in ('train','dev'):
        sha = digest(data/f'{split}.jsonl')
        if sha != trained[f'{split}_sha256'] or sha != protocol['hashes'][split]:
            raise ValueError(f'Teacher training protocol mismatch: {split}')
    state = read(teacher/'trainer_state.json')
    if (state['best_global_step'] != selected['step'] or state['best_metric'] != selected['dev_accuracy']
            or Path(state['best_model_checkpoint']).resolve() != teacher.resolve()):
        raise ValueError('Full teacher differs from trainer dev selection')
    snapshot = read(model_cfg['snapshot_manifest'])
    base = Path(model_cfg['model'])
    if (snapshot['revision'] != protocol['model_revision'] or Path(snapshot['path']).resolve() != base.resolve()
            or base.resolve() != Path(trained['config']['model']).resolve()):
        raise ValueError('Base snapshot identity mismatch')
    index = 'model.safetensors.index.json'
    for name in sorted({'config.json',index,*read(base/index)['weight_map'].values()}):
        if digest(base/name) != snapshot['files'][name]['sha256']:
            raise ValueError(f'Incorrect pinned base file: {name}')
    return base,teacher,selected,selection_path,model_cfg


def published(output, expected):
    """Check both identity and provenance, and finish an interrupted two-file publication."""
    checkpoint = output/'checkpoint-constructed'
    report = output/'construction.json'
    pending = output/'.construction.pending.json'
    if not report.is_file() and not checkpoint.exists():
        return False
    metadata = report if report.is_file() else pending
    if not metadata.is_file():
        raise ValueError(f'Checkpoint has no construction metadata: {checkpoint}')
    done = read(metadata)
    if (any(done.get(key) != value for key,value in expected.items())
            or done['checkpoint_identity'] != checkpoint_identity(checkpoint,'lora')):
        raise ValueError('Existing construction has different provenance or weights')
    if metadata == pending:
        pending.replace(report)
    return True


def construct(protocol_path, output, threads):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'constructor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _construct(protocol_path,output,threads)


def _construct(protocol_path, output, threads):
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    protocol = read(protocol_path)
    checkpoint = output/'checkpoint-constructed'
    report = output/'construction.json'
    start = time.monotonic()
    print(f'pid={os.getpid()} cpu_threads={threads} verifying teacher, train/dev and base',flush=True)
    base,teacher,selected,selection_path,model_cfg = verified_inputs(protocol)
    rank,alpha = model_cfg['adapter']['rank'],model_cfg['adapter']['alpha']
    if rank != 8:
        raise ValueError('Expected fixed rank-8 experiment')
    expected = dict(teacher_selection=selected,protocol_sha256=digest(protocol_path),
                    adapter_rank=rank,adapter_alpha=alpha)
    if published(output,expected):
        print(f'Already verified {report}',flush=True)
        return
    indices = [read(p/'model.safetensors.index.json')['weight_map'] for p in (base,teacher)]
    depth = read(base/'config.json')['thinker_config']['text_config']['num_hidden_layers']
    if depth != 36:
        raise ValueError('Unexpected base decoder layer count')
    layers,adapters = [],{}
    with ExitStack() as stack:
        files = {}
        for key in tensor_names(*indices,depth):
            mats = []
            for root,index in zip((base,teacher),indices):
                path = root/index[key]
                if path not in files:
                    files[path] = stack.enter_context(safe_open(str(path),framework='pt',device='cpu'))
                mats.append(files[path].get_tensor(key))
            a,b,rel,total,top = factors(*mats,rank,alpha)
            adapters[key[:-len('.weight')]] = a,b
            layers.append(dict(projection=key,shape=list(mats[0].shape),relative_frobenius_error=rel,
                               discrepancy_squared_norm=total,retained_squared_norm=top))
            print(f'{len(layers):02d}/72 {key} relative error={rel:.8f}',flush=True)
    with tempfile.TemporaryDirectory(dir=output) as temp:
        staged = Path(temp)/checkpoint.name
        write_adapter(staged,adapters,rank,alpha,str(base),protocol['model_revision'])
        identity = checkpoint_identity(staged,'lora')
        params = sum(a.numel()+b.numel() for a,b in adapters.values())
        record = dict(constructor='Expressive Power our_construction, per-projection depth=1',
                      scope='Independent square Q/O matrices in the real decoder. Same-data known teacher control.',
                      author_revision=AUTHOR_REVISION, adapter_rank=rank,adapter_alpha=alpha,
                      target_modules=['q_proj','o_proj'],num_projections=len(adapters),trainable_parameters=params,
                      optimizer_steps=0,teacher_step=selected['step'],device='cpu',threads=threads,
                      checkpoint=str(checkpoint.resolve()),checkpoint_identity=identity,
                      teacher_selection=selected,teacher_selection_path=str(selection_path.resolve()),
                      teacher_selection_sha256=digest(selection_path),model_revision=protocol['model_revision'],
                      protocol_sha256=digest(protocol_path),teacher_protocol_sha256=digest(selection_path.parent/'protocol.json'),
                      train_sha256=protocol['hashes']['train'],dev_sha256=protocol['hashes']['dev'],
                      layers=layers,construction_seconds=time.monotonic()-start)
        staged_report = Path(temp)/report.name
        staged_report.write_text(json.dumps(record,indent=2)+'\n')
        pending = output/'.construction.pending.json'
        staged_report.replace(pending)
        staged.rename(checkpoint)
        pending.replace(report)
    print(f'Published {report} and {checkpoint}',flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'))
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--threads',type=int,default=4)
    args = cli.parse_args()
    construct(args.protocol,args.output,args.threads)
