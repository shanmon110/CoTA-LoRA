# Adapters trained on IEMOCAP, scored on CREMA-D

1092 clips of 91 actors; nothing was trained on this corpus. Mean per-class recall in %; the order gap is in pp. In brackets: share of parseable free-text answers.

| model (mean over its seeds) | training order | other orders | all orders | order gap | yesno | word | open |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen-base (1) | 46.2 | 48.1 | 47.7 | -1.9 | 63.2 [100%] | 51.3 [100%] | 36.8 [91%] |
| C-post-audio (1) | 59.7 | 60.7 | 60.5 | -1.0 | 69.6 [100%] | 56.1 [100%] | 36.4 [96%] |
| A-last4 (3) | 52.7 | 44.7 | 46.7 | +7.9 | 63.3 [100%] | 49.2 [100%] | 36.4 [95%] |
| A-all (3) | 60.5 | 62.7 | 62.1 | -2.2 | 67.5 [100%] | 64.3 [100%] | 46.0 [90%] |
| B-audio (3) | 62.2 | 62.8 | 62.7 | -0.6 | 65.6 [100%] | 62.6 [100%] | 43.9 [85%] |
| D-all-shuffled (3) | 64.0 | 64.3 | 64.2 | -0.3 | 70.9 [100%] | 66.4 [100%] | 47.1 [90%] |

| difference pp [95% CI over actors and seeds] | training order | other orders | all orders | order gap | yesno | word | open |
|---|---|---|---|---|---|---|---|
| C-post-audio - frozen-base | +13.5 [+11.1, +15.8] | +12.6 [+10.9, +14.2] | +12.8 [+11.1, +14.4] | +0.9 [-1.0, +2.7] | +6.3 [+4.8, +7.9] | +4.8 [+3.5, +6.0] | -0.5 [-1.6, +0.6] |
| A-last4 - frozen-base | +6.4 [+3.7, +9.8] | -3.4 [-5.0, -1.9] | -0.9 [-1.9, +0.0] | +9.8 [+6.0, +14.3] | +0.0 [-0.5, +0.6] | -2.1 [-3.3, -1.0] | -0.5 [-1.6, +0.7] |
| A-all - frozen-base | +14.2 [+11.2, +17.1] | +14.5 [+11.8, +17.0] | +14.4 [+11.7, +16.9] | -0.3 [-2.3, +1.7] | +4.3 [+1.3, +6.8] | +13.0 [+11.2, +14.7] | +9.2 [+7.1, +11.3] |
| B-audio - frozen-base | +15.9 [+12.0, +19.5] | +14.7 [+11.5, +17.5] | +15.0 [+11.7, +17.9] | +1.3 [-0.5, +3.0] | +2.3 [+0.5, +4.3] | +11.2 [+9.2, +13.3] | +7.1 [+5.1, +9.2] |
| D-all-shuffled - frozen-base | +17.8 [+15.1, +20.4] | +16.2 [+13.8, +18.5] | +16.6 [+14.3, +18.8] | +1.6 [-0.3, +3.5] | +7.7 [+5.9, +9.4] | +15.1 [+13.0, +17.2] | +10.3 [+8.5, +12.4] |
| A-last4 - A-all | -7.8 [-11.2, -4.6] | -17.9 [-20.8, -14.8] | -15.4 [-17.7, -12.8] | +10.1 [+6.3, +14.9] | -4.2 [-6.7, -1.2] | -15.1 [-17.2, -13.0] | -9.6 [-11.8, -7.6] |
| D-all-shuffled - A-all | +3.6 [+2.2, +5.2] | +1.6 [-0.5, +3.8] | +2.1 [+0.4, +4.0] | +1.9 [-0.2, +4.2] | +3.4 [+1.8, +5.3] | +2.1 [+0.9, +3.3] | +1.1 [-0.5, +3.0] |
| B-audio - A-all | +1.7 [-0.1, +3.4] | +0.2 [-1.2, +1.4] | +0.5 [-0.8, +1.7] | +1.6 [+0.0, +3.1] | -2.0 [-4.5, +0.6] | -1.8 [-2.9, -0.5] | -2.1 [-3.7, -0.6] |

Readings declared before scoring:

- full_depth_gain_on_unseen_corpus: **transfers**
- last_layers_bind_to_option_order_on_unseen_corpus: **replicated**
- last_layers_gain_is_format_specific_on_unseen_corpus: **replicated**
- formats_where_full_depth_gains: **['yesno', 'word', 'open']**
