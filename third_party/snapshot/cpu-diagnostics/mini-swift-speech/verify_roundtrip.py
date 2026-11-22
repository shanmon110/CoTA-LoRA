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
rows=Path('/root/autodl-tmp/r1-aqa/data/pilot-01/train.jsonl').read_text().splitlines()[:8]
manifest=work/'first-eight-AVQA.jsonl'
manifest.write_text('\n'.join(rows)+'\n')
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
        out=Path(tr.args.output_dir)
        tr.paper_train_manifest=manifest
        (out/'protocol.json').write_text(json.dumps({'config':cfg}))
        pre = (impl.initialize_training(tr,method,cfg) if hasattr(impl,'initialize_training')
               else getattr(impl,'initialize',impl.audit)(tr.model,method,cfg))
        if method == 'rolora-federated':
            print('FED-SOURCE',json.loads(rows[0]),tr.train_dataset.dataset[0], flush=True)
        if hasattr(impl,'prepare_trainer'): impl.prepare_trainer(tr,method)
        result=tr.train()
        post=impl.audit(tr.model,method,cfg)
        assert pre['param_sha256'] != post['param_sha256']
        from contextlib import nullcontext
        from swift.arguments import InferArguments
        from swift.pipelines.utils import prepare_model_template
        def logits(m,t):
            e=json.loads(rows[0])
            with torch.inference_mode(), t.generate_context():
                t.set_mode('transformers')
                item=t.encode({'messages':[e['messages'][0]],'audios':e['audios']})
                inputs={k:v for k,v in t.data_collator([item]).items() if k in ('input_ids','attention_mask','input_features','feature_attention_mask')}
                assert inputs['feature_attention_mask'].any()
                return m(**inputs,use_cache=False).logits[:,-1].cpu()
        tr.model.eval()
        ctx=impl.evaluation_model(tr.model,method) if hasattr(impl,'evaluation_model') else nullcontext(tr.model)
        with ctx as live:
            expected=logits(live,tr.template)
        cp=out/'checkpoint-2'
        opt=dict(model=str(model_path),model_type='qwen2_5_omni',use_hf=True,load_args=False,torch_dtype='bfloat16',tuner_backend='peft',attn_impl='eager')
        if hasattr(impl,'load_best'):
            rebuilt,t=impl.load_best({**cfg,'training':{'max_length':2048}},method,cp,opt)
        else:
            rebuilt,t=prepare_model_template(InferArguments(**opt,adapters=[str(cp)],tuner_type='lora',load_data_args=False))
        actual=logits(rebuilt.eval(),t)
        assert torch.equal(actual,expected), (actual.float()-expected.float()).abs().max()
        print('EXACT-RELOAD',method,expected.shape, flush=True)
        print('MINI-SWIFT-CHECK',method,result.global_step,pre['trainable_parameters'],pre.get('calibration_forwards'),flush=True)
opts.update(model_type='qwen2_5_omni',model=str(model_path),dataset=[str(manifest)],use_hf=True,check_model=False,
            max_length=2048,split_dataset_ratio=0.,num_train_epochs=1,max_steps=2,per_device_train_batch_size=2,
            gradient_accumulation_steps=1,learning_rate=2e-4,eval_steps=2,use_cpu=True,bf16=False,fp16=False,
            torch_dtype='bfloat16',gradient_checkpointing=False,vit_gradient_checkpointing=False,attn_impl='eager',
            load_args=False,add_version=False,output_dir=str(work/('verify-'+method)),report_to=[],save_steps=2,dataloader_num_workers=0,
            disable_tqdm=True,lazy_tokenize=True,strict=True,model_kwargs={'torch_dtype':'bfloat16'})
TestSwiftSft(SftArguments(**opts)).main()
