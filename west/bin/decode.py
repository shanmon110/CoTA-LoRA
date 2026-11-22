# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, HfArgumentParser


@dataclass
class DecodeArguments:
    result_path: str = field(default=None, metadata={"help": "Path to result"})
    model_config_or_dir: str = field(default='')


def main():
    # Adapter-only Omni checkpoints bind an external base and processor. Detect
    # their archive format before importing other WEST models and speech toolkits.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--model_config_or_dir", "--model-config-or-dir")
    args, _ = probe.parse_known_args()
    if args.model_config_or_dir and os.path.isfile(
            os.path.join(args.model_config_or_dir, "adapter.safetensors")):
        from examples.omni_talora import west_decode
        if west_decode.is_omni_adapter(args.model_config_or_dir):
            return west_decode.main()
    from west.dataset.dataset import DataArguments, SpeechDataset
    from west.dataset.extractor import Extractor

    parser = HfArgumentParser((DataArguments, DecodeArguments))
    data_args, decode_args = parser.parse_args_into_dataclasses()
    if os.path.isfile(decode_args.model_config_or_dir):
        config = AutoConfig.from_pretrained(decode_args.model_config_or_dir)
        model = AutoModel.from_config(config)
    else:
        model = AutoModel.from_pretrained(decode_args.model_config_or_dir)
    tokenizer = model.init_tokenizer()
    extractor = Extractor.get_class(model.model_type)(tokenizer,
                                                      model.config,
                                                      inference=True)
    test_dataset = SpeechDataset(extractor, data_args)
    data_loader = DataLoader(test_dataset, collate_fn=lambda x: x[0])
    if torch.cuda.is_available():
        model = model.cuda()
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()
    fid = open(decode_args.result_path, 'w', encoding='utf8')
    with torch.no_grad():
        for i, item in enumerate(tqdm(data_loader)):
            generated_ids = model.generate(**item)
            text = tokenizer.batch_decode(generated_ids,
                                          skip_special_tokens=True)
            print(text)
            for t in text:
                t = t.replace('\n', ' ')
                item = {'txt': t}
                fid.write(json.dumps(item, ensure_ascii=False) + '\n')
            sys.stdout.flush()
    fid.close()


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
