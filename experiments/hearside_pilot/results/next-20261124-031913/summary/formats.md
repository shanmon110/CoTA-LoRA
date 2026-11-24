# Answers in formats no arm was trained on

Session 4, 1026 utterances. Score: mean per-group recall in % (yes/no: the two answers; word and open: the four emotions); an unparseable answer counts as wrong. In brackets: share of parseable answers.

| model (mean over its seeds) | yesno | word | open |
|---|---:|---:|---:|
| frozen-base (1) | 69.7 [100%] | 57.2 [100%] | 44.6 [83%] |
| C-post-audio (1) | 74.7 [100%] | 61.1 [100%] | 45.3 [84%] |
| A-last4 (3) | 69.8 [100%] | 53.6 [100%] | 44.1 [87%] |
| A-all (3) | 74.3 [100%] | 73.1 [100%] | 55.2 [79%] |
| B-audio (3) | 73.9 [100%] | 72.0 [100%] | 49.0 [73%] |
| D-all-shuffled (3) | 78.5 [100%] | 70.9 [100%] | 55.5 [82%] |

| difference pp [95% CI] | yesno | word | open |
|---|---|---|---|
| C-post-audio - frozen-base | +5.0 [+3.3, +6.7] | +3.9 [+1.8, +6.2] | +0.7 [-1.3, +2.5] |
| A-last4 - frozen-base | +0.1 [-0.2, +0.5] | -3.6 [-5.6, -1.6] | -0.6 [-2.8, +1.2] |
| A-all - frozen-base | +4.6 [+2.2, +7.3] | +15.8 [+12.7, +19.7] | +10.6 [+7.7, +13.7] |
| B-audio - frozen-base | +4.2 [+1.9, +6.7] | +14.8 [+11.3, +19.1] | +4.3 [+0.5, +8.0] |
| D-all-shuffled - frozen-base | +8.8 [+6.7, +11.2] | +13.7 [+9.6, +18.2] | +10.9 [+7.3, +14.4] |
| B-audio - A-all | -0.4 [-2.3, +1.8] | -1.0 [-2.5, +0.3] | -6.3 [-10.4, -2.3] |
| D-all-shuffled - A-all | +4.2 [+2.4, +6.3] | -2.2 [-4.9, -0.0] | +0.3 [-2.4, +2.4] |
| B-audio - D-all-shuffled | -4.6 [-6.7, -2.9] | +1.2 [-1.6, +3.8] | -6.5 [-12.3, -1.6] |
| A-last4 - A-all | -4.5 [-7.2, -1.9] | -19.4 [-24.2, -15.1] | -11.2 [-15.4, -7.6] |

Gain of D-all-shuffled over the frozen base on the trained question: +17.4 pp. Reading declared before scoring:

- yesno: **transfers**
- word: **transfers**
- open: **transfers**
