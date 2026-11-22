# ms-swift 运行环境

本次运行使用 Python 3.12.3、ms-swift 4.7.0.dev0、PyTorch 2.6.0 CUDA 12.4、Transformers 4.52.3、PEFT 0.15.2 和 BF16。框架代码及自带 TA-LoRA 示例来自交付包 `ms-swift-1997fa6-TA-LoRA-gate-ablation.zip`。解包文件内容与运行副本逐一匹配，源码包校验值和依赖版本见 [environment.json](../evidence/environment.json)。

当前环境位于 `/root/autodl-tmp/ms-swift-librispeech/venv`，复用了先前环境的 CUDA PyTorch 和部分依赖。训练和测试均使用这套实际环境。继承的 openai-whisper 20231117 要求旧版 Triton，和现有 Triton 3.2.0 存在声明冲突；本实验的语音加载和特征提取使用 ms-swift、librosa 和 Transformers。独立复建脚本按本实验的运行依赖安装，不需要 openai-whisper。

在提供同一 ms-swift 源码副本、可联网获取依赖的 Linux CUDA 机器上，可按照已记录的运行依赖版本创建新的独立环境。脚本使用 Python 3.12，可用 `PYTHON` 指定解释器路径。

```bash
bash environment/create.sh /path/to/new-venv /path/to/unpacked/ms-swift
source /path/to/new-venv/bin/activate
```

[constraints.txt](constraints.txt) 列出主要兼容版本。[resolved-constraints.txt](resolved-constraints.txt) 列出运行环境中 ms-swift、Omni 音视频扩展和 PyTorch 所用的依赖版本，供脚本复建。当前服务器采用的是上面说明的复用方式；新环境命令尚未经另一台独立机器验证。

在实验目录内，可重验当前快照和输入。模型下载时保留的官方元数据列出每个文件的字节数及 LFS/Git 内容标识。输入审查读取真实音频做特征提取，用 ms-swift 原生 encode/collate 核验训练标签和音频展开，并按所有音频的长度、模板和 tokenizer 统计序列上限。测试清单只统计输入长度。

```bash
python environment/verify_snapshot.py /root/autodl-tmp/west-preflight/model \
  /root/autodl-tmp/west-preflight/model-official.json
CUDA_VISIBLE_DEVICES='' ENABLE_AUDIO_OUTPUT=0 HF_HUB_OFFLINE=1 python environment/verify_inputs.py \
  config.json /root/autodl-tmp/ms-swift-librispeech/data /tmp/swift_asr_inputs.json
```
