"""Training-set gradient calibration for the native BF16/FP32 LoRA-GA recipe.

The ArB2r, stable scaling and weight subtraction below adapt the authors' PEFT
fork (Apache-2.0), LoraLayer.lora_ga_init at:
https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py
Copyright 2023-present the HuggingFace Inc. team. See LICENSE (Apache-2.0).
"""
import math
import time

import torch


@torch.no_grad()
def apply_ar_b2r(layer, average_gradient, gamma):
    weight = layer.get_base_layer().weight
    grad = average_gradient.to(weight.device, dtype=torch.float32)
    r = layer.r['default']
    if min(grad.shape) < 2*r or not torch.isfinite(grad).all() or not grad.any():
        raise ValueError('LoRA-GA needs a finite nonzero gradient and at least 2r directions')
    u, _, v = torch.svd_lowrank(grad, q=min(4*r, min(grad.shape)), niter=4)
    magnitude = grad.shape[0]**0.25 / math.sqrt(gamma)
    b = (u[:, r:2*r] * magnitude).contiguous()
    a = (v.T[:r, :] * magnitude).contiguous()
    layer.lora_A.default.weight.copy_(a)
    layer.lora_B.default.weight.copy_(b)
    offset = (b @ a) * layer.scaling['default']
    weight.copy_(weight.float() - offset)


def calibrate(trainer, settings, seed, layers):
    batch_count, batch_size = settings['calibration_batches'], settings['calibration_batch_size']
    count = batch_count * batch_size
    if count > len(trainer.train_dataset):
        raise ValueError('Insufficient training data for requested LoRA-GA calibration')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    indices = torch.randperm(len(trainer.train_dataset), generator=rng)[:count].tolist()
    model = trainer.model
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    gradients = {}
    hooks = []
    was_training = model.training
    def save_gradient(name):
        def add(parameter):
            value = parameter.grad.detach().to(device='cpu', dtype=torch.float32)
            if name in gradients:
                gradients[name].add_(value)
            else:
                gradients[name] = value.clone() if value.data_ptr() == parameter.grad.data_ptr() else value
            parameter.grad = None
        return add

    install_template_hook = not trainer.template._handles
    try:
        model.train()
        if install_template_hook:
            trainer.template.register_post_encode_hook([model])
        # Gradients of the unadapted language projections using the actual SFT loss.
        # Offload each gradient on production so the full dense gradient never resides on the GPU.
        with model.disable_adapter():
            for name, layer in layers.items():
                weight = layer.get_base_layer().weight
                if weight.requires_grad:
                    raise ValueError('Expected frozen base before LoRA-GA calibration')
                weight.requires_grad_(True)
                hooks.append(weight.register_post_accumulate_grad_hook(save_gradient(name)))
            for i in range(0, count, batch_size):
                rows = [trainer.train_dataset[idx] for idx in indices[i:i+batch_size]]
                inputs = trainer._prepare_inputs(trainer.data_collator(rows))
                if 'labels' not in inputs or not inputs['labels'].ne(-100).any():
                    raise ValueError('LoRA-GA calibration requires the training target mask')
                with trainer.compute_loss_context_manager(), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                    loss = trainer.compute_loss(model, inputs)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite LoRA-GA calibration loss')
                loss.backward()
    finally:
        if install_template_hook:
            trainer.template.remove_post_encode_hook()
        for hook in hooks:
            hook.remove()
        for layer in layers.values():
            weight = layer.get_base_layer().weight
            weight.requires_grad_(False)
            weight.grad = None
        model.train(was_training)
    if gradients.keys() != layers.keys():
        raise ValueError('LoRA-GA calibration did not reach each target projection')
    for name, layer in layers.items():
        apply_ar_b2r(layer, gradients.pop(name).div_(batch_count), settings['stable_gamma'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return dict(calibration_indices=indices, calibration_batches=batch_count,
                calibration_seconds=time.perf_counter()-start,
                calibration_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                calibration_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
