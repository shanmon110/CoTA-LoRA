"""LongLoRA's shifted short attention, adapted to Omni GQA and multimodal RoPE.

The authors' LongLoRA implementation uses one quarter of the padded context
per group and shifts half the heads by half a group. Generation uses native
full attention. These short-context AQA tasks do not extend the base context.
"""
import hashlib
from types import MethodType

import torch
from torch.nn.functional import dropout, softmax
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_multimodal_rotary_pos_emb, repeat_kv


def shifted_heads(q, k, v, attention_mask, groups, dropout_p):
    length, heads = q.shape[-2], q.shape[1]
    if heads % 2 or q.shape != k.shape or k.shape != v.shape:
        raise ValueError('Expected equal expanded GQA shapes and an even head count')
    width = max(1, (length+groups-1)//groups)
    channels = []
    for first_head, offset in ((0,0),(heads//2,width//2)):
        head = slice(first_head,first_head+heads//2)
        positions = sorted({0,length,*range(offset,length,width)})
        parts = []
        for start, stop in zip(positions[:-1],positions[1:]):
            local = slice(start,stop)
            logits = (q[:,head,local] @ k[:,head,local].transpose(-2,-1)) * q.shape[-1]**-0.5
            valid = torch.ones((stop-start,stop-start),device=q.device,dtype=torch.bool).tril()
            if attention_mask is not None:
                valid = valid & (attention_mask[:,:,local,local] == 0)
            probs = softmax(logits.float().masked_fill(~valid,-torch.inf), dim=-1).nan_to_num().to(q.dtype)
            parts.append(dropout(probs, dropout_p, training=dropout_p>0) @ v[:,head,local])
        channels.append(torch.cat(parts,dim=2))
    return torch.cat(channels,dim=1)


def attach(model, groups):
    decoder = model.get_base_model().thinker.model
    if not hasattr(decoder, '_longlora_training_cache_hook'):
        def training_cache(module, args, kwargs):
            if module.training:
                past = kwargs.get('past_key_values')
                if past is not None and past.get_seq_length():
                    raise ValueError('LongLoRA SFT cannot consume a nonempty autoregressive cache')
                kwargs.update(use_cache=False, past_key_values=None)
            return args, kwargs
        decoder._longlora_training_cache_hook = decoder.register_forward_pre_hook(training_cache, with_kwargs=True)
    for layer in decoder.layers:
        attn = layer.self_attn
        if hasattr(attn, '_longlora_full'):
            continue
        attn._longlora_full = attn.forward
        def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None):
            if not self.training:
                return self._longlora_full(hidden_states,attention_mask,position_ids,past_key_value,
                                           output_attentions,use_cache,cache_position,position_embeddings)
            if past_key_value is not None or output_attentions:
                raise ValueError('LongLoRA training expects no cache or attention output')
            batch, length, _ = hidden_states.shape
            def project(fn):
                return fn(hidden_states).view(batch,length,-1,self.head_dim).transpose(1,2)
            q,k = apply_multimodal_rotary_pos_emb(project(self.q_proj),project(self.k_proj),
                                                  *position_embeddings,self.rope_scaling['mrope_section'])
            v = project(self.v_proj)
            result = shifted_heads(q,repeat_kv(k,self.num_key_value_groups),repeat_kv(v,self.num_key_value_groups),
                                   attention_mask,groups,self.attention_dropout)
            return self.o_proj(result.transpose(1,2).reshape(batch,length,-1)), None, None
        attn.forward = MethodType(forward,attn)


def options(method, config, base_options):
    opts = config['method_options'][method]
    if opts != {'groups':4}:
        raise ValueError('Expected the four-group LongLoRA protocol')
    n = config['adapter']['last_n']
    norms = [f'thinker.model.layers.{i}.{s}' for i in range(n) for s in ('input_layernorm','post_attention_layernorm')]
    return dict(base_options, modules_to_save=['thinker.model.embed_tokens','thinker.model.norm',*norms])


def audit(model, method, config):
    layers = model.get_base_model().thinker.model.layers
    ad = config['adapter']
    if len(layers) != ad['last_n']:
        raise ValueError('LongLoRA trains on all Thinker language layers')
    expected = set(options(method,config,{})['modules_to_save'])
    for i in range(len(layers)):
        for p in ad['target_modules']:
            parent = 'self_attn' if p in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'
            expected.update(f'thinker.model.layers.{i}.{parent}.{p}.{name}.default' for name in ('lora_A','lora_B'))
    checksum = hashlib.sha256()
    params = {}
    for name, p in sorted(model.get_base_model().named_parameters()):
        if not p.requires_grad:
            continue
        key = name.removesuffix('.weight').removesuffix('.modules_to_save.default')
        params[key] = p
        checksum.update(name.encode()+b'\0'+p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if set(params) != expected:
        raise ValueError(f'LongLoRA trainable targets differ: {set(params)^expected}')
    return dict(param_sha256=checksum.hexdigest(), trainable_parameters=sum(p.numel() for p in params.values()),
                embedding_and_norm_modules=options(method,config,{})['modules_to_save'])


def initialize(model, method, config):
    from experiments.ms_swift_librispeech.adapter_audit import initialize_pair
    ad = config['adapter']
    extras = [p for n,p in model.named_parameters() if p.requires_grad and '.modules_to_save.default.' in n]
    # The common initializer audits exclusively the low-rank parameter set.
    # Keep the embedding and norm weights at their pretrained values.
    try:
        for param in extras:
            param.requires_grad_(False)
        initial = initialize_pair(model,'lora',config['training']['seed'],ad['last_n'],ad['target_modules'])
    finally:
        for param in extras:
            param.requires_grad_(True)
    return {**audit(model,method,config),'ab_init_sha256':initial['ab_init_sha256']}


def prepare_trainer(trainer, method, resume=None):
    attach(trainer.model,4)
