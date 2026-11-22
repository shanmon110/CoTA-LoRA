"""LoRAGen upstream decoder-only autoencoder, Transformer and GaussianDiffusion1D.

Source: https://github.com/tsinghua-fib-lab/LoRAGen/tree/911a738e63439732e94897f1aefd0d92123c4d9a
License: Apache-2.0, included alongside these files. autoencoder.py is upstream
stage1/modules/lora_modules.py. position.py is upstream stage2/train/Transformer/util.py.
transformer.py corrects the utils/util import spelling. diffusion.py omits unused trainer/dependency imports
and Trainer1D; the GaussianDiffusion1D implementation is unchanged.
"""
