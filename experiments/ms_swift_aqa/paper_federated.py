"""Serial simulation of Chen et al. RoLoRA, NeurIPS 2025, Algorithm 1."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from transformers import TrainerCallback
from experiments.ms_swift_librispeech.adapter_audit import initialize_pair, audit_adapter


def partition(rows, clients, seed):
    if clients < 2 or len(rows)%clients:
        raise ValueError('Expected equal-size source-isolated clients')
    grouped = defaultdict(list)
    for i,row in enumerate(rows):
        grouped[row['source_id']].append(i)
    def key(pair):
        return -len(pair[1]), hashlib.sha256(f'{seed}-{pair[0]}'.encode()).hexdigest()
    shards = [[] for _ in range(clients)]
    for _, indices in sorted(grouped.items(), key=key):
        min(shards,key=len).extend(indices)
    if len({len(group) for group in shards}) != 1:
        raise ValueError('Could not balance source-isolated client data')
    return shards


def _source_key(row):
    audio_paths = tuple(str(Path(audio['path'] if isinstance(audio,dict) else audio).resolve())
                        for audio in row['audios'])
    messages = tuple((item['role'],item['content']) for item in row['messages'])
    return audio_paths, messages


def align_sources(rows, dataset):
    """Match the post-Swift dataset order to the unique source IDs before batching."""
    by_content = {_source_key(row):row for row in rows}
    if len(by_content) != len(rows):
        raise ValueError('Ambiguous audio/message source identity in manifest')
    result = []
    for i, row in enumerate(dataset):
        match = by_content.pop(_source_key(row), None)
        if match is None:
            raise ValueError(f'Unexpected or repeated Swift dataset source at {i}')
        result.append(match)
    if by_content:
        raise ValueError('Missing source examples after Swift preprocessing')
    return result


class ClientBatches:
    def __init__(self, shards, batch, examples_per_visit, seed):
        self.shards, self.batch, self.visit, self.seed = shards,batch,examples_per_visit,seed
        self.epoch = 0
        if examples_per_visit % batch or any(len(s)%examples_per_visit for s in shards):
            raise ValueError('Incomplete local batch or client visit')

    def set_epoch(self,epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(map(len,self.shards))//self.batch

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed+self.epoch)
        order = [[s[i] for i in torch.randperm(len(s),generator=g)] for s in self.shards]
        for start in range(0,len(order[0]),self.visit):
            for s in order:
                for i in range(start,start+self.visit,self.batch):
                    yield s[i:i+self.batch]


class FederatedRounds(TrainerCallback):
    def __init__(self, clients, steps, resume=None):
        self.clients, self.steps = clients, steps
        self.resume = Path(resume) if resume else None
        self.server = {}
        self.sums = {}

    def _select_factors(self, model, step):
        round_index = step // (self.steps * self.clients)
        active = 'lora_B' if round_index%2 == 0 else 'lora_A'
        params = {k:p for k,p in model.named_parameters() if '.lora_A.' in k or '.lora_B.' in k}
        self.factors = {k:p for k,p in params.items() if active in k}
        for k,p in params.items():
            p.requires_grad_(k in self.factors)

    def on_train_begin(self,args,state,control,model,**kwargs):
        step = state.global_step
        if not self.resume or step % (self.clients*self.steps) == 0:
            return
        # A preemption can land inside a client's local optimization or after one
        # client has reported. Its local weights and AdamW state are in the HF
        # checkpoint; the original broadcast and completed-client sum are also needed.
        payload = torch.load(self.resume/'federated-round.pt',map_location='cpu',weights_only=True)
        if (payload['step'],payload['clients'],payload['local_steps']) != (step,self.clients,self.steps):
            raise ValueError('Federated partial round does not match optimizer step or client protocol')
        self._select_factors(model,step)
        for name in ('server','sums'):
            values = payload[name]
            if values.keys() != self.factors.keys() or any(values[k].shape!=p.shape or values[k].dtype!=p.dtype
                                                          for k,p in self.factors.items()):
                raise ValueError(f'Incorrect checkpoint {name} factors for federated resume')
            setattr(self,name,{k:t.to(self.factors[k].device) for k,t in values.items()})

    def on_save(self,args,state,control,**kwargs):
        if state.global_step % (self.clients*self.steps) == 0:
            return
        file = Path(args.output_dir)/f'checkpoint-{state.global_step}'/'federated-round.pt'
        tmp = file.with_suffix('.tmp')
        state_dict = {'step':state.global_step,'clients':self.clients,'local_steps':self.steps,
                      'server':{k:t.cpu() for k,t in self.server.items()},'sums':{k:t.cpu() for k,t in self.sums.items()}}
        torch.save(state_dict,tmp)
        tmp.replace(file)

    def on_step_begin(self,args,state,control,model,optimizer,**kwargs):
        if state.global_step % self.steps:
            return
        slot = state.global_step//self.steps % self.clients
        self._select_factors(model,state.global_step)
        if slot == 0:
            self.server = {k:p.detach().clone() for k,p in self.factors.items()}
            self.sums = {k:torch.zeros_like(p) for k,p in self.factors.items()}
        if set(self.server) != set(self.factors):
            raise ValueError('Resume RoLoRA at a completed aggregation round')
        with torch.no_grad():
            for k,p in self.factors.items():
                p.copy_(self.server[k])
        # Each client starts from the broadcast model with a fresh local AdamW.
        optimizer.state.clear()

    def on_step_end(self,args,state,control,model,**kwargs):
        if state.global_step%self.steps:
            return
        last = state.global_step%(self.clients*self.steps) == 0
        with torch.no_grad():
            for k,p in self.factors.items():
                self.sums[k].add_(p)
                if last:
                    p.copy_(self.sums[k]/self.clients)
        if last:
            self.server.clear()
            self.sums.clear()
            for n,p in model.named_parameters():
                if '.lora_A.' in n or '.lora_B.' in n:
                    p.requires_grad_(True)


def options(method, config, base_options):
    return base_options


def audit(model, method, config):
    ad = config['adapter']
    result = audit_adapter(model,'lora',ad['last_n'],ad['target_modules'])
    result['param_sha256'] = result['ab_sha256']
    return result


def initialize(model, method, config):
    a = config['adapter']
    initialize_pair(model,'lora',config['training']['seed'],a['last_n'],a['target_modules'])
    return audit(model,method,config)


def prepare_trainer(trainer, method, resume=None):
    from accelerate.data_loader import SkipBatchSampler
    from swift.dataloader import DataLoaderShard
    from experiments.ms_swift_aqa.run import pin
    from experiments.ms_swift_librispeech.compare import load
    out = Path(trainer.args.output_dir)
    settings = load(out/'protocol.json')['config']['method_options'][method]
    clients, steps = settings['clients'], settings['local_steps']
    rows = [json.loads(line) for line in trainer.paper_train_manifest.open()]
    if len(rows) != len(trainer.train_dataset) or trainer.args.eval_steps%(clients*steps):
        raise ValueError('RoLoRA requires complete source coverage and selection after aggregation')
    rows = align_sources(rows, trainer.train_dataset.dataset)
    shards = partition(rows,clients,trainer.args.data_seed)
    ids = [[rows[i]['id'] for i in group] for group in shards]
    pin(out/'client-partitions.json',dict(clients=ids,local_steps=steps,seed=trainer.args.data_seed))
    def loader(self,skip_batches=0):
        batch = self.args.per_device_train_batch_size
        sampler = ClientBatches(shards,batch,steps*batch*self.args.gradient_accumulation_steps,self.args.data_seed)
        if skip_batches:
            sampler = SkipBatchSampler(sampler,skip_batches=skip_batches)
        return DataLoaderShard(self.train_dataset,device=self.accelerator.device,batch_sampler=sampler,
                               collate_fn=self.data_collator,num_workers=self.args.dataloader_num_workers)
    trainer.get_train_dataloader = MethodType(loader,trainer)
    trainer.add_callback(FederatedRounds(clients,steps,resume=resume))
