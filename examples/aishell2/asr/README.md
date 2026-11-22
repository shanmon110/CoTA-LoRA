## Tutorial

First, prepare the train data `data/train.jsonl`, the data is like:

```
{"key": "IC0001W0001", "wav": "AISHELL-2/iOS/data/wav/C0001/IC0001W0001.wav", "txt": "厨房用具"}
{"key": "IC0001W0002", "wav": "AISHELL-2/iOS/data/wav/C0001/IC0001W0002.wav", "txt": "电压力锅"}
```
where `wav` is the wav path, `txt` is the transcript.

To train the model, just run

``` shell
bash run.sh --stage train
```

To decode, just prepare the test data `data/test.jsonl` the same as train. then just run

``` shell
bash run.sh --stage decode
```

## Choosing a fine-tuning method

From this recipe directory, use the same model and training entry point with a [shared adapter preset](../../conf/adapters/README.md). The following commands train TA-LoRA on the Qwen3/FireRed model and decode its checkpoint.

```bash
bash run.sh --stage train --model-conf conf/qwen3-1.7b_firered.json \
  --adapter-config ../../conf/adapters/ta-lora.json --dir exp/qwen3-firered-ta
bash run.sh --stage decode --dir exp/qwen3-firered-ta
```

Set `--training-config FILE` to select a different training profile. The original model JSON controls ordinary PEFT LoRA via its untagged `lora_config`. Tagged configurations use the shared [TouchASU adapter training profile](../../conf/training/touch_asu_adapters.json). Each method needs its own output directory.

## Results

| LLM        | Speech Encoder | LoRA | test CER | Details                               |
|------------|----------------|------|----------|---------------------------------------|
| Qwen3-1.7B | firered        | No   | 5.41     | 4 A800 GPUs, pack 18000, 10000 steps  |
