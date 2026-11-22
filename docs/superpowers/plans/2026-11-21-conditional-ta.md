# Question-conditioned TA-LoRA experiment

**Goal**  
Test whether a gate conditioned on the question and audible evidence improves both-correct accuracy for pairs of recordings that require different answers, with Qwen2.5-Omni-3B on one GPU.

**Data and budget**  
Retain the original AVQA train split for SFT and add auditable acoustic pairs derived only from its audio. Use source-disjoint edited recordings from the original AVQA development split for checkpoint selection. Build invariance pairs to detect spurious answer flips. Match methods for input order and exposure. The trained matrix budget is approximately 1.77 million parameters. Compare rank-12 O-LoRA, static rank-8 TA, question-only rank-8 TA and question-plus-audio rank-8 TA.

**Steps**

- [x] Build deterministic source-isolated pairs and verify label semantics, equal lengths and complete same-order choices.
- [x] Add versioned answer-query-gate configuration. Compare zero-initialized residual with the common base, verify a nonzero startup gradient on transport-up and clean reloading of the learned gate. Preserve legacy modes.
- [x] Select the earliest checkpoint with highest paired change accuracy. Keep pair-level correctness, invariance, and individual counts available in every run.
- [x] Run clean-reload controls, including an evaluation-only transport-off arm. Use the independent sound/music/speech development set to check transfer and the frozen base to establish the zero-training reference.
- [x] Inspect failures by task and study source- and task-disjoint transfer. Use this evidence for a bounded second experiment before final MMAR scoring. Results on edited audio assess the labeled acoustic transformations.
- [x] Score the declared candidate and matched controls with the official MMAR script. Document which seed, selection path and inference constraints produced each result.
