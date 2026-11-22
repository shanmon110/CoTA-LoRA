"""Closed-set SER: choose the most likely of four one-token answers at the generation boundary."""
import json
from pathlib import Path
import torch


def metrics(pairs):
    matrix=[[0]*4 for _ in range(4)]
    choices='ABCD'
    for ref,pred in pairs:
        matrix[choices.index(ref)][choices.index(pred)]+=1
    if not all(map(sum,matrix)):
        raise ValueError('All four reference classes are required for UAR')
    recalls={k:matrix[i][i]/sum(matrix[i]) for i,k in enumerate(choices)}
    return dict(uar=sum(recalls.values())/4,accuracy=sum(matrix[i][i] for i in range(4))/sum(map(sum,matrix)),
                recalls=recalls,confusion_matrix=matrix,utterances=sum(map(sum,matrix)))


def score(model,template,manifest,generation,predictions):
    rows=[json.loads(s) for s in Path(manifest).read_text().splitlines()]
    choices='ABCD'
    ids=[template.tokenizer.encode(c,add_special_tokens=False) for c in choices]
    if any(len(x)!=1 for x in ids) or len(set(x[0] for x in ids))!=4:
        raise ValueError('Expected four distinct single-token answers')
    tokens=[x[0] for x in ids]
    pairs=[]
    predictions=Path(predictions)
    predictions.parent.mkdir(parents=True,exist_ok=True)
    tmp=predictions.with_suffix('.tmp')
    with torch.inference_mode(), template.generate_context(), tmp.open('w') as f:
        template.set_mode('transformers')
        size=generation['batch_size']
        for start in range(0,len(rows),size):
            items=rows[start:start+size]
            enc=[template.encode(dict(messages=[r['messages'][0]],audios=list(r['audios']))) for r in items]
            batch={k:v.to(model.device) for k,v in template.data_collator(enc).items()
                   if k in ('input_ids','attention_mask','input_features','feature_attention_mask')}
            if not batch['attention_mask'][:,-1].bool().all():
                raise ValueError('Expected left padded prompts')
            seq=template.generate(model,**batch,max_new_tokens=1,do_sample=False,num_beams=1,
                                  use_model_defaults=False,prefix_allowed_tokens_fn=lambda *_:tokens)
            if not torch.equal(seq[:,:-1],batch['input_ids']):
                raise ValueError('Generated prefix mismatch')
            for r,t in zip(items,seq[:,-1].tolist(),strict=True):
                pred=choices[tokens.index(t)]
                ref=r['messages'][1]['content']
                pairs.append((ref,pred))
                print(json.dumps(dict(id=r['id'],reference=ref,prediction=pred)),file=f)
    result=metrics(pairs)
    tmp.replace(predictions)
    return result
