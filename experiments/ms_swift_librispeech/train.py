"""Native ms-swift SFT with corpus-level generated dev WER checkpoint selection."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import TrainerCallback, set_seed

from experiments.ms_swift_librispeech.adapter_audit import MLP_PROJECTIONS, audit_adapter, initialize_pair, ordered_projections

HERE = Path(__file__).resolve().parent


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _validate_gate_lr_multiplier(multiplier):
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('transport_gate_lr_multiplier must be a finite positive number')


def scale_transport_gate_lr(optimizer, model, multiplier):
    """Split the named trainable context weights before the native scheduler is attached."""
    _validate_gate_lr_multiplier(multiplier)
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('Transport gate learning rates require native AdamW')
    gate_ids = {id(param) for name, param in model.named_parameters()
                if param.requires_grad and name.rsplit('.', 2)[-2:] == ['ta_adapter', 'gate_weight']}
    if not gate_ids:
        raise ValueError('Expected a trainable conditional gate')
    groups = []
    found = set()
    for group in optimizer.param_groups:
        positions = [i for i, p in enumerate(group['params']) if id(p) in gate_ids]
        if not positions:
            groups.append(group)
            continue
        if 'initial_lr' in group:
            raise ValueError('Split the transport gate group before creating the scheduler')
        scaled = group['lr'] * multiplier
        if not math.isfinite(scaled) or scaled <= 0:
            raise ValueError('Scaled transport gate LR must be finite and positive')
        indices = set(positions)
        keep = [i for i in range(len(group['params'])) if i not in indices]
        for selected, lr in ((keep, group['lr']), (positions, scaled)):
            if not selected:
                continue
            params = [group['params'][i] for i in selected]
            subgroup = dict(group, params=params, lr=lr)
            if 'param_names' in group:
                subgroup['param_names'] = [group['param_names'][i] for i in selected]
            groups.append(subgroup)
        found.update(id(group['params'][i]) for i in positions)
    if found != gate_ids:
        raise ValueError('Native AdamW did not include every trainable conditional gate')
    optimizer.param_groups = groups
    return optimizer


def select(history, patience, metric="wer"):
    checks = [(e['step'], e['eval_'+metric]) for e in history if 'eval_'+metric in e]
    maximize = metric in ('accuracy', 'uar', 'pair_accuracy')
    if not checks or patience < 1 or any(not math.isfinite(w) or w < 0 or (maximize and w > 1) for _, w in checks):
        raise ValueError('Expected positive patience and valid development scores')
    if any(b[0] <= a[0] for a, b in zip(checks, checks[1:])):
        raise ValueError('Development scores must have unique increasing optimizer steps')
    best = (max if maximize else min)(range(len(checks)), key=lambda i: checks[i][1])
    step, wer = checks[best]
    return step, wer, len(checks) - 1 - best >= patience


class Schedule(TrainerCallback):
    def __init__(self, patience, stop_after=None, metric="wer"):
        self.metric = metric
        self.patience = patience
        self.stop_after = stop_after

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = control.should_save = True
        elif state.global_step == self.stop_after:
            control.should_training_stop = control.should_save = True

    def on_evaluate(self, args, state, control, **kwargs):
        if select(state.log_history, self.patience, self.metric)[2]:
            control.should_training_stop = True


class InputTrajectory(TrainerCallback):
    """Fingerprint the first eight post-collation microbatches (before Trainer mutates inputs)."""
    def __init__(self, resume=None, record_examples=False):
        self.batches = json.loads((resume / 'input-trajectory.json').read_text()) if resume else []
        self.record_examples = record_examples
        self.examples = (json.loads((resume / 'sample-trajectory.json').read_text())
                         if record_examples and resume else [])

    def record(self, inputs):
        if self.record_examples and len(self.examples) < 32:
            for i in range(min(len(inputs['input_ids']), 32 - len(self.examples))):
                valid = inputs['attention_mask'][i].bool()
                frames = inputs['feature_attention_mask'][i].bool()
                h = hashlib.sha256()
                for tensor in (inputs['input_ids'][i][valid], inputs['labels'][i][valid],
                               inputs['input_features'][i, :, frames]):
                    value = tensor.detach().cpu().contiguous()
                    h.update(str((list(value.shape), value.dtype)).encode() + b'\0')
                    h.update(value.view(torch.uint8).numpy().tobytes())
                self.examples.append(h.hexdigest())
        if len(self.batches) >= 8:
            return
        h = hashlib.sha256()
        for key in ('input_ids', 'labels', 'feature_attention_mask'):
            tensor = inputs[key].detach().cpu().contiguous()
            h.update(str((key, list(tensor.shape), tensor.dtype)).encode() + b'\0')
            h.update(tensor.numpy().tobytes())
        self.batches.append(h.hexdigest())

    def on_save(self, args, state, control, **kwargs):
        save_json(Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'input-trajectory.json', self.batches)
        if self.record_examples:
            save_json(Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'sample-trajectory.json', self.examples)


class Timing(TrainerCallback):
    """Training wall time includes checkpoint I/O and excludes generated validation."""
    def __init__(self, device, resume=None):
        self.device = torch.device(device)
        self.previous = (json.loads((resume / 'efficiency.json').read_text()) if resume else {})
        self.dev_seconds = 0.
        self.allocated = self.reserved = 0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def reset(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self):
        if self.device.type == 'cuda':
            self.allocated = max(self.allocated, torch.cuda.max_memory_allocated(self.device))
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved(self.device))

    def on_train_begin(self, *args, **kwargs):
        self.sync()
        self.reset()
        self.start = time.perf_counter()

    def stats(self):
        self.sync()
        self.peak()
        return dict(training_seconds=self.previous.get('training_seconds', 0) +
                    time.perf_counter() - self.start - self.dev_seconds,
                    dev_seconds=self.previous.get('dev_seconds', 0) + self.dev_seconds,
                    peak_allocated_bytes=max(self.allocated, self.previous.get('peak_allocated_bytes', 0)),
                    peak_reserved_bytes=max(self.reserved, self.previous.get('peak_reserved_bytes', 0)))

    def on_save(self, args, state, control, **kwargs):
        save_json(Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'efficiency.json', self.stats())


def model_options(config, method):
    if method not in ('ta-lora', 'lora', 'dora', 'rslora'):
        from experiments.ms_swift_aqa.tuner_suite import implementation
        if config.get('task') != 'aqa':
            raise ValueError('Additional native tuners require an AQA experiment config')
        return implementation(method).options(method, config, model_options(config, 'lora'))
    model_dir = Path(config['model'])
    pinned = json.loads(Path(config.get('snapshot_manifest', HERE / 'evidence/verified_snapshot.json')).read_text())
    if config['model_revision'] != pinned['revision']:
        raise ValueError('Requested model revision differs from the verified experiment snapshot')
    for name, expected in pinned['files'].items():
        path = model_dir / name
        checksum = expected['sha256'] if isinstance(expected, dict) else expected
        if digest(path) != checksum:
            raise ValueError(f'Changed model or processor input: {path}')
    layers = json.loads((model_dir / 'config.json').read_text())['thinker_config']['text_config']['num_hidden_layers']
    last_n = config['adapter']['last_n']
    if method not in ('ta-lora', 'lora', 'dora', 'rslora') or not 1 <= last_n <= layers:
        raise ValueError('Invalid method or target layer count')
    indices = '|'.join(map(str, range(layers - last_n, layers)))
    names = ordered_projections(config['adapter'].get('target_modules', ('o_proj',)), require_o=method == 'ta-lora')
    model_type = config.get('model_type', 'qwen2_5_omni')
    if method == 'ta-lora' and model_type not in ('qwen2_5_omni', 'qwen3_asr'):
        raise ValueError('The TA plugin requires Omni attention projections')
    modules = '|'.join((r'mlp\.' if p in MLP_PROJECTIONS else r'self_attn\.') + p for p in names)
    return dict(model=str(model_dir.resolve()), model_type=model_type, use_hf=True,
                attn_impl='eager', torch_dtype='bfloat16', tuner_backend='peft',
                tuner_type='ta_lora' if method == 'ta-lora' else 'lora', use_dora=method == 'dora', load_args=False,
                **({'use_rslora': True} if method == 'rslora' else {}),
                external_plugins=[str(HERE / 'plugin.py')] if method == 'ta-lora' else [],
                target_regex=rf'^thinker\.model\.layers\.({indices})\.({modules})$',
                lora_alpha=config['adapter']['alpha'], lora_rank=config['adapter']['rank'],
                # Let PEFT upcast all DoRA parameters, including its magnitude vector.
                lora_dropout=0., lora_dtype=None if method == 'dora' else 'float32')


def load_best(config, method, checkpoint):
    """Instantiate a clean base and load only the selected method's saved adapter."""
    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template
    os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
    opts = model_options(config, method)
    if method not in ('ta-lora', 'lora', 'dora', 'rslora', 'full'):
        from experiments.ms_swift_aqa.tuner_suite import implementation
        impl = implementation(method)
        if hasattr(impl, 'load_best'):
            return impl.load_best(config, method, Path(checkpoint), opts)
        if hasattr(impl, 'inference_path'):
            checkpoint = impl.inference_path(Path(checkpoint), method)
    if method == 'full':
        from experiments.ms_swift_aqa.full_suite import load_best as load_full
        return load_full(config, checkpoint, opts)
    # The selected checkpoint encodes the trained dimensions and the inference method.
    opts = {key: opts[key] for key in ('model', 'model_type', 'use_hf', 'attn_impl', 'torch_dtype',
                                        'tuner_backend', 'tuner_type', 'load_args', 'external_plugins')
            if key in opts}
    if method == 'bone':
        # Both values dispatch to Swift.from_pretrained, which reads BONE from disk. The
        # 'bone' inference branch additionally casts the trained FP32 adapter to BF16.
        opts['tuner_type'] = 'lora'
    args = InferArguments(**opts, adapters=[str(Path(checkpoint).resolve())],
                          max_length=config['training']['max_length'], load_data_args=False)
    model, template = prepare_model_template(args)
    return model.eval(), template


def inspect_supervision(dataset, template, processor):
    """Check a real encoded example at the exact lazy-dataset/collator boundary."""
    raw = dataset.dataset[0]
    batch = template.data_collator([dataset[0]])
    ids, labels = batch['input_ids'][0], batch['labels'][0]
    audio_id = template.tokenizer.convert_tokens_to_ids(processor.audio_token)
    audio = ids.eq(audio_id)
    active = labels.ne(-100)
    text = template.tokenizer.decode(labels[active].tolist(), skip_special_tokens=True).strip()
    expected = raw['messages'][-1]['content'].strip()
    if (raw['messages'][-1]['role'] != 'assistant' or text.split() != expected.split() or not audio.any()
            or not batch['feature_attention_mask'].any() or not torch.isfinite(batch['input_features']).all()
            or not torch.equal(labels[active], ids[active]) or (active & audio).any()):
        raise ValueError(f'Unexpected Swift audio input/ASR loss mask: target={expected!r}, labels={text!r}')
    return dict(audio_placeholders=audio.sum().item(), feature_shape=list(batch['input_features'].shape),
                sequence_tokens=len(ids), supervised_tokens=active.sum().item(), decoded_target=text)


def fit(config, method, train, dev, output, resume=None, stop_after=None):
    from swift.arguments import SftArguments
    from swift.pipelines.train.sft import SwiftSft
    from swift.trainers import Seq2SeqTrainer, TrainerFactory
    task_metric = {"ser": "uar", "aqa": "accuracy"}.get(config.get("task"), "wer")
    metric = config.get('selection_metric', task_metric)
    if metric != task_metric and not (task_metric == 'accuracy' and metric == 'pair_accuracy'):
        raise ValueError(f'Unsupported selection metric {metric!r} for this task')
    impl = None
    if method not in ('ta-lora', 'lora', 'dora', 'rslora'):
        from experiments.ms_swift_aqa.tuner_suite import implementation
        impl = implementation(method)
    if task_metric == "accuracy":
        from experiments.ms_swift_aqa.evaluate import score
    elif task_metric == "uar":
        from experiments.ms_swift_iemocap.evaluate import score
    else:
        from experiments.ms_swift_librispeech.evaluate import score

    os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
    training = config['training']
    gate_lr_multiplier = None
    if 'transport_gate_lr_multiplier' in config['adapter']:
        gate_lr_multiplier = config['adapter']['transport_gate_lr_multiplier']
        _validate_gate_lr_multiplier(gate_lr_multiplier)
        if (method != 'ta-lora' or config['adapter'].get('transport_gate') not in ('question', 'question_audio')
                or training.get('optim') != 'adamw_torch' or training.get('optimizer') not in (None, 'default')):
            raise ValueError('Transport gate learning rates require conditional TA and the default adamw_torch optimizer')
    if method == 'reft' and training.get('gradient_checkpointing'):
        raise ValueError('ReFT requires gradient_checkpointing=False to retain its forward interventions')
    if int(os.getenv('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('This protocol requires a single visible CUDA device')
    if training['eval_steps'] != training['save_steps'] or training['eval_steps'] < 1 or config['patience'] < 1:
        raise ValueError('Use a common positive eval/save interval and positive patience')
    if train.samefile(dev):
        raise ValueError('Training and development manifests must differ')
    identity = dict(config=config, method=method, train_sha256=digest(train), dev_sha256=digest(dev))
    protocol = output / 'protocol.json'
    if resume:
        if (json.loads(protocol.read_text()) != identity or resume.parent != output
                or (output / 'result.json').exists()):
            raise ValueError('Resume requires the original unfinished run and identical protocol')
        from experiments.ms_swift_aqa.tuner_suite import checkpoint_files
        checkpoint_files(resume, method)
        files = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                 'input-trajectory.json', 'efficiency.json']
        if impl:
            files.append('sample-trajectory.json')
        if method == 'adalora':
            files.append('adalora-allocator.pt')
        for file in files:
            if not (resume / file).is_file():
                raise ValueError(f'Incomplete checkpoint: {resume / file}')
        checkpoints = [int(p.name.split('-')[-1]) for p in output.glob('checkpoint-[0-9]*') if p.is_dir()]
        if int(resume.name.split('-')[-1]) != max(checkpoints):
            raise ValueError('Resume only the latest checkpoint to avoid overwriting subsequent results')
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f'Use a fresh output directory: {output}')
        save_json(protocol, identity)
    kwargs = dict(**model_options(config, method), **training)
    kwargs.update(dataset=[str(train)], val_dataset=[str(dev)], split_dataset_ratio=0., strict=True,
                  add_version=False, output_dir=str(output), lazy_tokenize=True, check_model=False,
                  group_by_length=False, truncation_strategy='delete',
                  packing=False, padding_free=False, bf16=True, fp16=False, gradient_checkpointing=training.get('gradient_checkpointing', False),
                  vit_gradient_checkpointing=training.get('vit_gradient_checkpointing', False),
                  metric_for_best_model=metric, greater_is_better=metric in ('uar', 'accuracy', 'pair_accuracy'),
                  # The downstream scorer re-instantiates each best checkpoint from a clean base.
                  load_best_model_at_end=False, save_strategy='steps', eval_strategy='steps',
                  save_total_limit=2, report_to=[], save_only_model=False,
                  resume_from_checkpoint=str(resume) if resume else None, ignore_data_skip=False)
    args = SftArguments(**kwargs)

    if gate_lr_multiplier is not None:
        from swift.optimizers import OptimizerCallback

        class TransportGateOptimizer(OptimizerCallback):
            def create_optimizer(self, model=None):
                is_new = self.trainer.optimizer is None
                optimizer = super().create_optimizer(model=model)
                if is_new:
                    opt_model = self.trainer.model if model is None else model
                    scale_transport_gate_lr(optimizer, opt_model, gate_lr_multiplier)
                return optimizer

    class AsrTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if method not in ('lokra', 'igu-lora', 'lora-mixer') or not model.training:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            from experiments.ms_swift_aqa.paper_late import training_penalty
            if method == 'lora-mixer':
                from experiments.ms_swift_aqa.paper_late_mixer import set_batch_mask
                set_batch_mask(model, inputs['attention_mask'])
            labels = inputs['labels']
            weight = ((labels != -100).sum() / num_items_in_batch if num_items_in_batch is not None
                      and (self.model_accepts_loss_kwargs or self.compute_loss_func) else 1.)
            loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
            loss = loss + weight * training_penalty(model, method, config['method_options'][method])
            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs, *args, **kwargs):
            self.trajectory.record(inputs)
            loss = super().training_step(model, inputs, *args, **kwargs)
            if impl and not torch.isfinite(loss).all().item():
                raise FloatingPointError(f'Nonfinite {method} training loss at step {self.state.global_step}')
            return loss

        def evaluate(self, *args, **kw):
            self.clock.sync()
            self.clock.peak()
            self.clock.reset()
            start = time.perf_counter()
            step = self.state.global_step
            dest = output / 'dev' / f'step-{step}'
            dest.mkdir(parents=True, exist_ok=True)
            was_training = self.model.training
            try:
                predictions = dest / 'hypotheses.jsonl'
                model = self.model.eval()
                context = (impl.evaluation_model(model, method) if impl and hasattr(impl, 'evaluation_model')
                           else nullcontext(model))
                with context as eval_model:
                    metrics = score(eval_model, self.template, dev, config['generation'], predictions)
                if metric == 'pair_accuracy':
                    from experiments.ms_swift_aqa.pair_eval import measure_pairs
                    pairs = measure_pairs(dev, predictions)['by_kind']
                    metrics['pair_accuracy'] = pairs['change']['both_correct_rate']
                    metrics['by_pair_kind'] = pairs
                self.clock.sync()
                seconds = time.perf_counter() - start
                metrics.update(seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                save_json(dest / 'metrics.json', metrics)
            finally:
                self.model.train(was_training)
                self.clock.dev_seconds += time.perf_counter() - start
                self.clock.reset()
            result = {'eval_'+metric: metrics[metric]}
            self.log(result.copy())
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, result)
            return result

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if method == 'ta-lora' or (impl and method != 'full'):
                # SwiftSft.prepare_model already restored these tuners in trainable mode.
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    class AsrSft(SwiftSft):
        def train(self, trainer):
            trainer.paper_train_manifest = train
            native_method = method if method in ('dora', 'rslora') else args.tuner_type
            n = config['adapter']['last_n']
            projections = config['adapter'].get('target_modules', ('o_proj',))
            objective = config.get('objective', 'sft')
            if objective == 'closed_set_4way' and config.get('task') == 'ser':
                from experiments.ms_swift_iemocap.closed_set_loss import make_closed_set_loss
                trainer.compute_loss_func = make_closed_set_loss(trainer)
            elif objective != 'sft':
                raise ValueError(f'Unsupported training objective {objective!r}')
            if impl:
                inspect = getattr(impl, 'initialize', impl.audit) if not resume else impl.audit
                if not resume and hasattr(impl, 'initialize_training'):
                    initial = impl.initialize_training(trainer, method, config)
                else:
                    initial = inspect(trainer.model, method, config)
                if resume:
                    initial = json.loads((output / 'initialization.json').read_text())
                else:
                    initial['initial_param_sha256'] = initial['param_sha256']
            elif resume:
                initial = json.loads((output / 'initialization.json').read_text())
                audit_adapter(trainer.model, native_method, n, projections)
            else:
                initial = initialize_pair(trainer.model, native_method, args.seed, n, projections)
                if method == 'ta-lora':
                    adapter = config['adapter']
                    gate_init = adapter.get('gate_init', 0.)
                    if not math.isfinite(gate_init):
                        raise ValueError('gate_init must be finite')
                    with torch.no_grad():
                        for layer in trainer.model.thinker.model.layers[-n:]:
                            layer.self_attn.ta_adapter.gate.fill_(gate_init)
                    initial['gate_init'] = gate_init
                    if 'gate_multiplier' in adapter:
                        from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier
                        set_gate_multiplier(trainer.model, adapter['gate_multiplier'])
                        initial['gate_multiplier'] = adapter['gate_multiplier']
                    if 'transport_queries' in adapter:
                        from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_queries
                        set_transport_queries(trainer.model, adapter['transport_queries'])
                        initial['transport_queries'] = adapter['transport_queries']
                    if 'transport_weighting' in adapter:
                        from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_weighting
                        set_transport_weighting(trainer.model, adapter['transport_weighting'])
                        initial['transport_weighting'] = adapter['transport_weighting']
                    if adapter.get('independent_transport_up', False):
                        from experiments.ms_swift_librispeech.swift_asr_ta_runtime import enable_transport_up
                        enable_transport_up(trainer.model)
                    if 'transport_gate' in adapter:
                        from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_gate
                        if config.get('task') != 'aqa':
                            raise ValueError('Answer-query transport is only supported for single-turn AQA')
                        set_transport_gate(trainer.model, adapter['transport_gate'])
                    for key in ('transport_signal', 'transport_precision'):
                        if key in adapter:
                            from experiments.ms_swift_librispeech import swift_asr_ta_runtime as runtime
                            getattr(runtime, 'set_' + key)(trainer.model, adapter[key])
                            initial[key] = adapter[key]
                    if adapter.get('independent_transport_up', False) or 'transport_gate' in adapter:
                        # Recount after C and/or the conditional gate were added to the trainable set.
                        initial.update(audit_adapter(trainer.model, native_method, n, projections))
            if gate_lr_multiplier is not None:
                trainer.optimizer_callback = TransportGateOptimizer(trainer.args, trainer)
                initial['transport_gate_lr_multiplier'] = gate_lr_multiplier
            if not resume:
                initial['supervision'] = inspect_supervision(trainer.train_dataset, self.template, self.processor)
                if task_metric == 'accuracy':
                    from experiments.ms_swift_aqa.evaluate import inspect_training_prompt
                    initial['supervision'].update(inspect_training_prompt(trainer.train_dataset, self.template))
                save_json(output / 'initialization.json', initial)
            # Fresh PEFT and TA attachment may consume different numbers of random draws.
            # On resume HF restores all training RNG states after initialization.
            set_seed(args.seed)
            clock = Timing(trainer.args.device, resume)
            trainer.clock = clock
            trainer.add_callback(clock)
            trainer.trajectory = InputTrajectory(resume, record_examples=bool(impl))
            trainer.add_callback(trainer.trajectory)
            trainer.add_callback(Schedule(config['patience'], stop_after, metric))
            stats_every = config.get('diagnostics', {}).get('adapter_stats_every')
            if stats_every is not None:
                from experiments.ms_swift_librispeech.adapter_stats import AdapterStats
                trainer.add_callback(AdapterStats(output / 'adapter-stats.jsonl', stats_every))
            if impl and hasattr(impl, 'prepare_trainer'):
                impl.prepare_trainer(trainer, method, resume=resume)
            res = super().train(trainer)
            if stop_after == trainer.state.global_step < trainer.state.max_steps:
                return res
            best_step, wer, stopped = select(trainer.state.log_history, config['patience'], metric)
            best = output / f'checkpoint-{best_step}'
            from experiments.ms_swift_aqa.tuner_suite import checkpoint_files
            checkpoint_files(best, method)
            if Path(trainer.state.best_model_checkpoint).resolve() != best or trainer.state.best_metric != wer:
                raise ValueError('Swift best checkpoint and independent development selection disagree')
            final = impl.audit(trainer.model, method, config) if impl else audit_adapter(trainer.model, native_method, n, projections)
            before = initial['initial_param_sha256'] if impl else initial['ab_init_sha256']
            after = final['param_sha256'] if impl else final['ab_sha256']
            if before == after:
                raise ValueError('The optimizer did not change the intended trainable weights')
            if 'transport_up_sha256' in initial and initial['transport_up_sha256'] == final['transport_up_sha256']:
                raise ValueError('The independent transport output projection did not change')
            if 'transport_gate_sha256' in initial and initial['transport_gate_sha256'] == final['transport_gate_sha256']:
                raise ValueError('The conditional transport gate did not change')
            if method == 'dora' and final['magnitude_sha256'] == initial['magnitude_sha256']:
                raise ValueError('DoRA magnitudes did not change during training')
            if method == 'lora_llm':
                changes = {k: v != initial['component_sha256'][k] for k, v in final['component_sha256'].items()}
                if changes != dict(language_lora=True, audio_tower=True, visual=False):
                    raise ValueError(f'Unexpected weight updates on audio-only training: {changes}')
            summary = dict(method=method, best_checkpoint=str(best), **{"best_"+metric:wer}, best_step=best_step,
                           global_step=trainer.state.global_step, early_stopped=stopped,
                           trainable_parameters=initial['trainable_parameters'],
                           input_trajectory=trainer.trajectory.batches,
                           init_sha256=before, **({'last_param_sha256': after,
                                                    'sample_trajectory': trainer.trajectory.examples} if impl
                                                  else {'last_ab_sha256': after}),
                           gpu=torch.cuda.get_device_name(), **clock.stats())
            if 'shared_ab_sha256' in initial:
                summary['shared_ab_sha256'] = initial['shared_ab_sha256']
            if method == 'lora_llm':
                summary['changed_components'] = changes
            if method == 'adalora':
                # The selected/last steps can have different budgets under adaptive allocation.
                allocation = torch.load(best / 'adalora-allocator.pt', map_location='cpu', weights_only=True)
                ranks = {k: int(sum(mask)) for k, mask in (allocation['rank_pattern'] or {}).items()}
                settings = config['method_options'][method]
                if (set(ranks) != set(final['allocated_rank_per_module']) or allocation['global_step'] != best_step
                        or not 0 < sum(ranks.values()) < settings['adalora_init_r'] * len(ranks)):
                    raise ValueError('Best AdaLoRA checkpoint does not contain the learned rank allocation')
                save_json(output / 'selected-ranks.json', ranks)
                summary.update(best_active_rank_sum=sum(ranks.values()), last_active_rank_sum=final['allocated_rank'],
                               allocated_projections=len(ranks))
            save_json(output / 'result.json', summary)
            return res

    # ms-swift resolves trainer classes by module path via its public registry.
    globals()['AsrTrainer'] = AsrTrainer
    previous = TrainerFactory.TRAINER_MAPPING['causal_lm']
    TrainerFactory.TRAINER_MAPPING['causal_lm'] = f'{__name__}.AsrTrainer'
    try:
        return AsrSft(args).main()
    finally:
        TrainerFactory.TRAINER_MAPPING['causal_lm'] = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method', required=True)
    for name in ('train', 'dev', 'output', 'resume'):
        parser.add_argument('--' + name, type=Path, required=name != 'resume')
    parser.add_argument('--stop-after', type=int, help='Interrupt after a saved update to exercise full training resume')
    cli = parser.parse_args()
    fit(json.loads(cli.config.read_text()), cli.method, cli.train.resolve(), cli.dev.resolve(),
        cli.output.resolve(), cli.resume.resolve() if cli.resume else None, cli.stop_after)
