# TA-LoRA on the official Omni Thinker

Target a single labelled speech-understanding task per run. Start with English ASR
and four-class SER; share the implementation but never training data, optimizer
state, or adapters. The recipe is independent of WEST's TouchASU registry and GRPO.

Adapt the Thinker language decoder's last four attention output projections, not
the speech encoder, projector, Talker, embeddings, or LM head. Use shared down/up
atoms for the native attention output and an audio-minus-nonaudio conditional
value contrast, with one static per-rank gate per layer. Only valid non-audio
queries with both key types visible can receive the contrast. Normalize each group
separately on the native post-RoPE, causal, pre-dropout scores. No use of future
tokens, no equality assumption between the mel axis and the token axis.

Initial up/gate zero; gate zero recovers the rank-matched ordinary O-LoRA function.
No claim of static merging for nonzero gates, lower deployment memory, or better
WER/UAR without empirical runs. Pin Transformers eager attention at 4.52.3.
Support one audio per example, un-packed SFT and greedy DynamicCache decoding.
Save trainable parameters only. Compare methods at identical prompts, update
counts, seeds, task splits, layers and rank; report the extra gates explicitly.
