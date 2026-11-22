from pathlib import Path
import os,json,sys,shutil
import torch
from swift.arguments import SftArguments
from swift.pipelines.train.sft import SwiftSft
from swift.trainers import TrainerFactory
from transformers import Qwen2_5OmniConfig,Qwen2_5OmniForConditionalGeneration
from experiments.ms_swift_aqa.tuner_suite import implementation
work=Path(__file__).parent
model_path=work/'model'
source=Path('/root/autodl-tmp/west-preflight/model')
if not (model_path/'model.safetensors').is_file():
    model_path.mkdir(exist_ok=True)
    cfg=Qwen2_5OmniConfig.from_pretrained(source)
    cfg.enable_audio_output=False
    c=cfg.thinker_config
    t=c.text_config
    t.hidden_size=32;t.intermediate_size=64;t.num_hidden_layers=3;t.num_attention_heads=4;t.num_key_value_heads=2;t.rope_scaling['mrope_section']=[1,1,2]
    a=c.audio_config
    a.d_model=32;a.output_dim=32;a.encoder_layers=1;a.encoder_attention_heads=4;a.encoder_ffn_dim=64
    v=c.vision_config
    v.depth=1;v.hidden_size=32;v.intermediate_size=64;v.num_heads=4;v.out_hidden_size=32;v.fullatt_block_indexes=[0]
    m=Qwen2_5OmniForConditionalGeneration(cfg).to(torch.bfloat16)
    m.save_pretrained(model_path)
    del m
    for p in source.iterdir():
        if p.name=='spk_dict.pt' or p.suffix in ('.txt','.json') and p.name not in ('config.json','model.safetensors.index.json','generation_config.json'):
            q=model_path/p.name
            if not q.exists(): q.symlink_to(p)
manifest=Path('/root/autodl-tmp/r1-aqa/data/pilot-01/train.jsonl')
rows=manifest.read_text().splitlines()
method=sys.argv[1]
recipe=json.load(open(f'experiments/ms_swift_aqa/papers28/configs/{method}.json'))
settings=recipe['settings']
if method == 'eva': settings.update(samples=4,batch_size=2,rho=1.5,tau=.01,max_forwards=20)
if method == 'lora-ga': settings.update(calibration_batch_size=1,calibration_batches=2)
if method == 'loram-train-small': settings.update(first_intact=1,last_intact=1)
if method == 'rolora-federated': settings.update(clients=2,local_steps=1)
if method == 'randlora': settings.update(r=4,randlora_alpha=16)
cfg={'adapter':{'rank':2,'alpha':4,'last_n':3,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
     'training':{'seed':42,'data_seed':42},'method_options':{method:settings}}
impl=implementation(method)
opts=impl.options(method,cfg,dict(tuner_type='lora',tuner_backend='peft',target_regex=r'^thinker\.model\.layers\.\d+\.(mlp|self_attn)\.\w+_proj$',
                                 lora_rank=2,lora_alpha=4,lora_dropout=0,lora_dtype='float32'))
class TestSwiftSft(SwiftSft):
    def train(self,tr):
        import hashlib
        from experiments.ms_swift_aqa.paper_federated import align_sources,partition
        from experiments.ms_swift_aqa.evaluate import check_encoded_prompt
        src=[json.loads(r) for r in rows]
        actual=align_sources(src,tr.train_dataset.dataset)
        groups=partition(actual,4,42)
        seen=set()
        digest=hashlib.sha256()
        src_groups=[set(actual[i]['source_id'] for i in g) for g in groups]
        assert not any(src_groups[i]&src_groups[j] for i in range(4) for j in range(i))
        for i,row in enumerate(actual):
            x=tr.train_dataset[i]
            target=self.template.encode({'messages':row['messages'],'audios':row['audios']})
            h=hashlib.sha256()
            for key in ('input_ids','labels','feature_attention_mask','input_features'):
                value=torch.as_tensor(x[key]).contiguous()
                assert torch.equal(value,torch.as_tensor(target[key])),(row['id'],key)
                h.update(value.view(torch.uint8).numpy())
            first=torch.as_tensor(x['labels']).ne(-100).nonzero()[0].item()
            check_encoded_prompt(self.template,torch.as_tensor(x['input_ids'])[:first],row['messages'][0]['content'],x['feature_attention_mask'])
            target_letter=row['messages'][-1]['content']
            assert self.template.tokenizer.encode(target_letter,add_special_tokens=False)==[x['input_ids'][first]]
            assert h.hexdigest() not in seen
            seen.add(h.hexdigest());digest.update((row['id']+h.hexdigest()).encode())
            if i and i%256==0: print('verified',i,flush=True)
        assert len(seen)==len(src)==2048
        summary={'rows':len(src),'unique_audio_message_keys':len(actual),'unique_encoded_examples':len(seen),
                 'ordered_id_and_feature_sha256':digest.hexdigest(),'swift_reorders':sum(x['id']!=y['id'] for x,y in zip(actual,src)),
                 'client_examples':[len(g) for g in groups],'client_sources':[len(g) for g in src_groups],
                 'pairwise_source_overlap':0, 'processor_revision':'f75b40e3da2003cdd6e1829b1f420ca70797c34e'}
        (work/'client-audit-2048.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('FED-2048',summary,flush=True)
opts.update(model_type='qwen2_5_omni',model=str(model_path),dataset=[str(manifest)],use_hf=True,check_model=False,
            max_length=2048,split_dataset_ratio=0.,num_train_epochs=1,max_steps=2,per_device_train_batch_size=2,
            gradient_accumulation_steps=1,learning_rate=2e-4,eval_steps=2,use_cpu=True,bf16=False,fp16=False,
            torch_dtype='bfloat16',gradient_checkpointing=False,vit_gradient_checkpointing=False,attn_impl='eager',
            load_args=False,add_version=False,output_dir=str(work/method),report_to=[],save_steps=2,dataloader_num_workers=0,
            disable_tqdm=True,lazy_tokenize=True,strict=True,model_kwargs={'torch_dtype':'bfloat16'})
TestSwiftSft(SftArguments(**opts)).main()
