"""Exercise the paired native tuners on the longest and a typical real training utterance."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import soundfile
import torch
from transformers import set_seed
from swift.arguments import SftArguments
from swift.pipelines.train.sft import SwiftSft
from swift.tuner_plugin import PeftTuner, tuners_map

from .adapter_audit import initialize_pair
from .train import model_options


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run(opts):
    cfg = json.loads(opts.config.read_text())
    seed = cfg['training']['seed']
    method = opts.method.replace('-', '_')
    opts.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (opts.data/'train.jsonl').read_text().splitlines()]
    if cfg.get('task') == 'ser':
        ordered = sorted(rows, key=lambda row: soundfile.info(row['audios'][0]).duration)
        typical, longest = ordered[len(ordered) // 2], ordered[-1]
    else:
        meta = json.loads((opts.data/'splits.json').read_text())['train-clean-100']
        typical, longest = rows[0], next(r for r in rows if r['id'] == meta['longest_id'])
        if abs(soundfile.info(typical['audios'][0]).duration - meta['median_seconds']) > 1:
            raise ValueError('Typical probe utterance deviates from the median')
    durations = [soundfile.info(r['audios'][0]).duration for r in (typical,longest)]
    args = SftArguments(**model_options(cfg, opts.method), dataset=[str(opts.data/'train.jsonl')],
                        output_dir=str(opts.output/'swift'), add_version=False, seed=seed, data_seed=seed,
                        gradient_checkpointing=False, vit_gradient_checkpointing=False, device_map='cuda:0',
                        per_device_train_batch_size=2, split_dataset_ratio=0,
                        report_to=[], max_length=cfg['training']['max_length'], packing=False, padding_free=False)
    tuner = tuners_map[method] if method=='ta_lora' else PeftTuner
    pipe = SwiftSft(args)
    model = pipe.prepare_model(args, pipe.model, template=pipe.template)
    stats = initialize_pair(model, method, seed, cfg['adapter']['last_n'], cfg['adapter'].get('target_modules', ('o_proj',)))
    set_seed(seed)
    template = pipe.template
    encoded = [template.encode({'messages':r['messages'], 'audios':r['audios']}) for r in (typical,longest)]
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k,v in template.data_collator(encoded).items()}
    ids, labels, valid = batch['input_ids'], batch['labels'], batch['attention_mask'].bool()
    print(dict(method=method, utterances=[typical['id'], longest['id']], durations_seconds=durations,
               expanded_tokens=valid.sum(-1).tolist(), trainable_parameters=stats['trainable_parameters']), flush=True)
    audio_id = model.config.thinker_config.audio_token_id
    assert labels.masked_select(~valid | ids.eq(audio_id)).eq(-100).all()
    for i, row in enumerate((typical,longest)):
        keep = labels[i].ne(-100)
        assert torch.equal(labels[i,keep], ids[i,keep])
        assert template.tokenizer.decode(labels[i,keep], skip_special_tokens=True).strip()==row['messages'][1]['content']
    template.register_post_encode_hook([model])
    def probe(m, inp):
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            m.eval()
            logits = m(**dict(inp, use_cache=False)).logits
            idx = torch.arange(labels.size(1),device=labels.device).expand_as(labels).masked_fill(~labels.ne(-100),0).max(-1).values
            return digest(logits[torch.arange(len(idx),device=idx.device),idx])
    start_logits = probe(model,batch)
    muted = probe(model,dict(batch,input_features=torch.zeros_like(batch['input_features'])))
    assert muted != start_logits, 'audio features had no effect'
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['learning_rate'],weight_decay=0.)
    grad_steps=[]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(2):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**dict(batch, use_cache=False)).loss
        loss.backward()
        norms={n: p.grad.norm().item() for n,p in model.named_parameters() if p.requires_grad}
        assert torch.isfinite(loss) and all(torch.isfinite(torch.tensor(v)) for v in norms.values())
        if step: assert all(v>0 for v in norms.values()),norms
        opt.step()
        grad_steps.append(dict(loss=loss.item(),gradient_norms=norms))
    torch.cuda.synchronize()
    seconds=time.monotonic()-start
    allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
    learned = probe(model,batch)
    assert learned != start_logits
    checkpoint=opts.output/'adapter'
    tuner.save_pretrained(model,str(checkpoint))
    template.remove_post_encode_hook()
    del opt, model, pipe, template, loss
    gc.collect()
    torch.cuda.empty_cache()
    pipe=SwiftSft(args)
    restored=tuner.from_pretrained(pipe.model,str(checkpoint),is_trainable=False)
    pipe.template.register_post_encode_hook([restored])
    assert probe(restored,batch)==learned, 'restored logits differ'
    pipe.template.remove_post_encode_hook()
    result=dict(**stats, method=method, initial_logits_sha256=start_logits,restored_logits_sha256=learned,
                muted_audio_logits_sha256=muted, trainable_gradients=grad_steps,durations_seconds=durations,
                utterances=[typical['id'],longest['id']],expanded_tokens=valid.sum(-1).tolist(),
                supervised_tokens=labels.ne(-100).sum(-1).tolist(), gpu=torch.cuda.get_device_name(),
                allocated_bytes=allocated,reserved_bytes=reserved,training_seconds=seconds,
                training_batch_size=2,torch=torch.__version__)
    (opts.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','data'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=['lora','ta-lora'],required=True)
    run(p.parse_args())
