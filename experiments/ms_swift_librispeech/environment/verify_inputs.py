"""Audit native processor/label alignment and ASR token budgets without scoring test audio."""
import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf
import torch
from swift.model import get_model_processor
from swift.template import get_template


def check(config, data):
    cfg = json.loads(config.read_text())
    model, processor = get_model_processor(cfg['model'], model_type='qwen2_5_omni', attn_impl='eager',
                                          download_model=False, device_map={'': 'cpu'}, torch_dtype=torch.bfloat16)
    assert not model.config.enable_audio_output and not hasattr(model, 'talker')
    template = get_template(processor, template_type='qwen2_5_omni')
    template.set_mode('train')
    audio_id = model.config.thinker_config.audio_token_index
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    eos = tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
    audits = {}
    prefix_length = None
    for split in ('train', 'dev-clean', 'test-clean'):
        with (data / (split + '.jsonl')).open() as stream:
            rows = [json.loads(line) for line in stream]
        frames = []
        for row in rows:
            wave = sf.info(row['audios'][0])
            assert wave.samplerate == feature_extractor.sampling_rate and wave.channels == 1
            assert 0 < wave.frames <= feature_extractor.n_samples
            frames.append((wave.frames + feature_extractor.hop_length - 1) // feature_extractor.hop_length)
        audio_tokens = [template._get_feat_extract_output_lengths(n) for n in frames]
        samples = []
        if split != 'test-clean':
            # Check native encode + collate on the shortest, median and longest true waveforms.
            ordered = sorted(range(len(rows)), key=lambda i: frames[i])
            for i in [ordered[0], ordered[len(ordered)//2], ordered[-1]]:
                row = rows[i]
                assert row['messages'][0]['content'] == '<audio>' + cfg['prompt']
                batch = template.data_collator([template.encode(row)])
                labels, ids = batch['labels'][0], batch['input_ids'][0]
                expected = tokenizer.encode(row['messages'][1]['content'], add_special_tokens=False) + eos
                count = len(expected)
                assert labels[-count:].tolist() == ids[-count:].tolist() == expected
                assert labels[:-count].eq(-100).all() and labels[ids == audio_id].eq(-100).all()
                assert int(batch['feature_attention_mask'].sum()) == frames[i]
                assert int((ids == audio_id).sum()) == audio_tokens[i]
                with template.generate_context():
                    template.set_mode('transformers')
                    prompt = template.encode(dict(messages=[dict(row['messages'][0])], audios=row['audios']))
                assert prompt['input_ids'] == ids[:-count].tolist()
                template.set_mode('train')
                fixed = len(ids) - count - audio_tokens[i]
                if prefix_length is None:
                    prefix_length = fixed
                assert fixed == prefix_length
                samples.append(dict(id=row['id'], feature_frames=frames[i], audio_tokens=audio_tokens[i],
                                    sequence_tokens=len(ids), supervised_tokens=count))
        count_by_sample = []
        if split == 'test-clean':
            # During final scoring references are withheld from the model.
            count_by_sample = [0] * len(rows)
        else:
            for i in range(0, len(rows), 512):
                answers = [r['messages'][1]['content'] for r in rows[i:i+512]]
                ids = tokenizer(answers, add_special_tokens=False)['input_ids']
                count_by_sample.extend(len(seq) + len(eos) for seq in ids)
        lengths = [prefix_length + a + n for a, n in zip(audio_tokens, count_by_sample, strict=True)]
        limit = cfg['training']['max_length']
        assert max(lengths) <= limit
        audits[split] = dict(count=len(rows), min_frames=min(frames), max_frames=max(frames),
                             min_audio_tokens=min(audio_tokens), max_audio_tokens=max(audio_tokens),
                             max_sequence_tokens=max(lengths), median_sequence_tokens=statistics.median(lengths),
                             over_limit=sum(n > limit for n in lengths), native_encoded_examples=samples,
                             lengths_include_reference=(split != 'test-clean'))
    return dict(model_type=type(model).__name__, processor_type=type(processor).__name__,
                template_type=type(template).__name__, system=template.template_meta.default_system,
                talker_active=hasattr(model, 'talker'), template_prefix_without_audio_tokens=prefix_length,
                supervised_suffix=tokenizer.decode(eos), splits=audits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = check(args.config, args.data)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
