# WEST implementation / verification plan

1. Implement and test native eager Thinker attention injection with shared atoms;
   compare to hand-computed conditional means, ordinary LoRA, padding and KV cache.
2. Turn one local audio file and one task label into target-only CE supervision;
   validate audio placeholder lengths, prompt boundaries and four SER labels.
3. Add per-task SFT, adapter-only checkpoint I/O, greedy ASR WER and forced-choice
   SER UAR. Ensure the loader understands upstream full-checkpoint key names.
4. Exercise tiny native Thinker training, gradients, save/reload, cached decoding
   and real tokenizer/processor fixtures; ship the same offline tests with the code.
5. On the target Linux machine run short-clip single-GPU smoke fits before matched
   ASR/SER experiments. Full benchmark scores and hardware measurements are not
   supplied by unit tests.
