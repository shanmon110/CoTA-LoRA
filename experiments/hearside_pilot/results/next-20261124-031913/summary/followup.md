# Hear-side follow-up: depth and seeds

Session 4, 1026 utterances, 4104 rows over four option orders. UAR in %.
The order gap is UAR in the training order minus UAR in the other three orders.

| model | training order | other orders | all orders | order gap pp [95% CI] | reworded | transcript only | WER % |
|---|---:|---:|---:|---|---:|---:|---:|
| frozen-base | 52.1 | 57.4 | 56.1 | -5.3 [-6.75, -3.91] | 59.0 | 48.3 | 9.0 |
| C-post-audio | 72.7 | 66.1 | 67.8 | +6.7 [+4.22, +9.51] | 66.2 | 48.6 | 8.8 |
| A-last4 | 65.9 | 46.4 | 51.3 | +19.4 [+15.26, +22.79] | 55.2 | 46.8 | 12.5 |
| A-last4-s43 | 64.0 | 48.6 | 52.5 | +15.4 [+11.45, +18.54] | 56.4 | 47.3 | 12.6 |
| A-last4-s44 | 62.2 | 49.6 | 52.8 | +12.6 [+9.13, +15.50] | 56.8 | 47.2 | 12.5 |
| A-all | 76.2 | 72.7 | 73.6 | +3.5 [+1.52, +6.06] | 74.5 | 49.8 | 7.9 |
| A-all-s43 | 76.5 | 73.0 | 73.9 | +3.4 [+1.75, +5.48] | 74.2 | 49.8 | 7.2 |
| A-all-s44 | 70.2 | 66.9 | 67.7 | +3.3 [+2.15, +4.27] | 70.0 | 50.2 | 7.8 |
| B-audio | 75.4 | 74.4 | 74.6 | +1.1 [+0.06, +2.20] | 75.6 | 48.3 | 8.8 |
| B-audio-s43 | 76.2 | 75.4 | 75.6 | +0.8 [-0.77, +2.44] | 74.4 | 48.3 | 12.3 |
| B-audio-s44 | 69.0 | 69.2 | 69.1 | -0.2 [-1.79, +1.20] | 72.5 | 48.3 | 11.6 |
| D-all-shuffled | 77.4 | 75.6 | 76.0 | +1.9 [+0.78, +3.01] | 76.4 | 50.0 | 7.3 |
| D-all-shuffled-s43 | 75.8 | 75.6 | 75.6 | +0.2 [-0.99, +1.29] | 76.2 | 49.8 | 11.2 |
| D-all-shuffled-s44 | 68.3 | 69.0 | 68.8 | -0.6 [-2.67, +0.79] | 71.3 | 49.5 | 8.0 |

## Training seeds

| arm | seeds | measure | per seed | mean | SD |
|---|---|---|---|---:|---:|
| A-last4 | 42, 43, 44 | uar_all_orders | 51.3 / 52.5 / 52.8 | 52.2 | 0.79 |
| A-last4 | 42, 43, 44 | uar_other_orders | 46.4 / 48.6 / 49.6 | 48.2 | 1.64 |
| A-last4 | 42, 43, 44 | order_gap | +19.4 / +15.4 / +12.6 | +15.8 | 3.45 |
| A-last4 | 42, 43, 44 | uar_reworded | 55.2 / 56.4 / 56.8 | 56.1 | 0.81 |
| A-last4 | 42, 43, 44 | wer | 12.5 / 12.6 / 12.5 | 12.5 | 0.03 |
| A-all | 42, 43, 44 | uar_all_orders | 73.6 / 73.9 / 67.7 | 71.7 | 3.46 |
| A-all | 42, 43, 44 | uar_other_orders | 72.7 / 73.0 / 66.9 | 70.9 | 3.43 |
| A-all | 42, 43, 44 | order_gap | +3.5 / +3.4 / +3.3 | +3.4 | 0.12 |
| A-all | 42, 43, 44 | uar_reworded | 74.5 / 74.2 / 70.0 | 72.9 | 2.50 |
| A-all | 42, 43, 44 | wer | 7.9 / 7.2 / 7.8 | 7.7 | 0.38 |
| B-audio | 42, 43, 44 | uar_all_orders | 74.6 / 75.6 / 69.1 | 73.1 | 3.47 |
| B-audio | 42, 43, 44 | uar_other_orders | 74.4 / 75.4 / 69.2 | 73.0 | 3.31 |
| B-audio | 42, 43, 44 | order_gap | +1.1 / +0.8 / -0.2 | +0.5 | 0.68 |
| B-audio | 42, 43, 44 | uar_reworded | 75.6 / 74.4 / 72.5 | 74.1 | 1.56 |
| B-audio | 42, 43, 44 | wer | 8.8 / 12.3 / 11.6 | 10.9 | 1.85 |
| D-all-shuffled | 42, 43, 44 | uar_all_orders | 76.0 / 75.6 / 68.8 | 73.5 | 4.06 |
| D-all-shuffled | 42, 43, 44 | uar_other_orders | 75.6 / 75.6 / 69.0 | 73.4 | 3.82 |
| D-all-shuffled | 42, 43, 44 | order_gap | +1.9 / +0.2 / -0.6 | +0.5 | 1.28 |
| D-all-shuffled | 42, 43, 44 | uar_reworded | 76.4 / 76.2 / 71.3 | 74.6 | 2.85 |
| D-all-shuffled | 42, 43, 44 | wer | 7.3 / 11.2 / 8.0 | 8.8 | 2.10 |

| difference | seeds | measure | per seed pp | mean pp | 95% CI (seeds and dialogues) |
|---|---|---|---|---:|---|
| B-audio - A-all | 42, 43, 44 | uar_all_orders | +1.06 / +1.68 / +1.41 | +1.38 | [+0.32, +2.60] |
| B-audio - A-all | 42, 43, 44 | uar_training_order | -0.78 / -0.30 / -1.22 | -0.76 | [-2.35, +0.67] |
| B-audio - A-all | 42, 43, 44 | uar_other_orders | +1.68 / +2.34 / +2.28 | +2.10 | [+0.86, +3.45] |
| B-audio - A-all | 42, 43, 44 | order_gap | -2.46 / -2.63 / -3.50 | -2.86 | [-4.58, -1.36] |
| B-audio - A-all | 42, 43, 44 | uar_reworded | +1.10 / +0.19 / +2.45 | +1.25 | [-0.81, +3.89] |
| D-all-shuffled - A-all | 42, 43, 44 | uar_all_orders | +2.46 / +1.74 / +1.07 | +1.76 | [+0.12, +3.46] |
| D-all-shuffled - A-all | 42, 43, 44 | uar_training_order | +1.22 / -0.70 / -1.87 | -0.45 | [-2.60, +1.51] |
| D-all-shuffled - A-all | 42, 43, 44 | uar_other_orders | +2.88 / +2.55 / +2.04 | +2.49 | [+0.77, +4.41] |
| D-all-shuffled - A-all | 42, 43, 44 | order_gap | -1.66 / -3.25 / -3.92 | -2.94 | [-5.16, -1.05] |
| D-all-shuffled - A-all | 42, 43, 44 | uar_reworded | +1.90 / +2.00 / +1.33 | +1.75 | [+0.26, +3.27] |
| B-audio - D-all-shuffled | 42, 43, 44 | uar_all_orders | -1.40 / -0.06 / +0.34 | -0.37 | [-2.11, +1.35] |
| B-audio - D-all-shuffled | 42, 43, 44 | uar_training_order | -2.00 / +0.40 / +0.66 | -0.31 | [-2.60, +2.05] |
| B-audio - D-all-shuffled | 42, 43, 44 | uar_other_orders | -1.20 / -0.21 / +0.24 | -0.39 | [-2.12, +1.30] |
| B-audio - D-all-shuffled | 42, 43, 44 | order_gap | -0.79 / +0.61 / +0.42 | +0.08 | [-1.48, +1.91] |
| B-audio - D-all-shuffled | 42, 43, 44 | uar_reworded | -0.80 / -1.81 / +1.12 | -0.50 | [-2.40, +1.94] |
| A-last4 - A-all | 42, 43, 44 | uar_all_orders | -22.28 / -21.41 / -14.97 | -19.55 | [-23.82, -15.02] |
| A-last4 - A-all | 42, 43, 44 | uar_training_order | -10.34 / -12.46 / -8.00 | -10.27 | [-14.78, -6.66] |
| A-last4 - A-all | 42, 43, 44 | uar_other_orders | -26.25 / -24.40 / -17.30 | -22.65 | [-27.53, -17.23] |
| A-last4 - A-all | 42, 43, 44 | order_gap | +15.91 / +11.94 / +9.30 | +12.38 | [+7.45, +17.21] |
| A-last4 - A-all | 42, 43, 44 | uar_reworded | -19.25 / -17.78 / -13.25 | -16.76 | [-20.24, -13.00] |

## Last four layers against all 36 (one seed each)

| measure | A-last4 | A-all | difference pp | 95% CI (dialogues) |
|---|---:|---:|---:|---|
| uar_all_orders | 51.3 | 73.6 | -22.28 | [-25.13, -19.57] |
| uar_training_order | 65.9 | 76.2 | -10.34 | [-14.88, -6.70] |
| uar_other_orders | 46.4 | 72.7 | -26.25 | [-29.29, -23.06] |
| order_gap | 19.4 | 3.5 | +15.91 | [+10.80, +20.21] |
| uar_reworded | 55.2 | 74.5 | -19.25 | [-21.53, -16.73] |
| A-last4 minus frozen base, other orders | | | -11.00 | [-13.13, -8.55] |

Claims declared before the follow-up:

- audio_positions_match_all_positions: **supported**
- audio_positions_less_bound_to_option_order: **supported**
- shuffled_options_improve_all_positions: **supported**
- last_layers_bind_to_option_order: **supported**
- last_layers_bind_to_option_order_across_seeds: **supported**
