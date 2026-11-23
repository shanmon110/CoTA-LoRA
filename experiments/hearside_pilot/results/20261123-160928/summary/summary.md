# Hear-side pilot: Session 4

1026 utterances, 4104 rows over the option orders, 30 dialogues. UAR in %, scored on classes.
One training seed per arm: the intervals cover evaluation noise only.

| arm | recordings, training order | recordings, other orders | reworded question, all orders | transcript only | question only | transcription WER % |
|---|---:|---:|---:|---:|---:|---:|
| frozen-base | 52.1 | 57.4 | 59.0 | 48.3 | 25.0 | 9.0 |
| A-all | 76.2 | 72.7 | 74.5 | 49.8 | 25.0 | 7.9 |
| B-audio | 75.4 | 74.4 | 75.6 | 48.3 | 25.0 | 8.8 |
| C-post-audio | 72.7 | 66.1 | 66.2 | 48.6 | 25.0 | 8.8 |
| D-all-shuffled | 77.4 | 75.6 | 76.4 | 50.0 | 25.0 | 7.3 |

B-audio adds nothing to an input without audio, so its transcript-only and question-only scores equal the frozen base by construction.

Share of each answer letter on the recordings (the gold letters are balanced over the four orders):

| arm | A | B | C | D |
|---|---:|---:|---:|---:|
| frozen-base | 29.6 | 24.1 | 25.6 | 20.7 |
| A-all | 20.1 | 24.5 | 26.1 | 29.3 |
| B-audio | 24.1 | 25.0 | 25.8 | 25.1 |
| C-post-audio | 16.8 | 24.6 | 28.4 | 30.2 |
| D-all-shuffled | 26.4 | 25.1 | 25.2 | 23.3 |

| contrast | input | option orders | dUAR pp | 95% CI | answers changed % |
|---|---|---|---:|---|---:|
| B-audio - A-all | orig | training_order | -0.78 | [-2.89, +0.99] | 10.0 |
| B-audio - A-all | orig | other_orders | +1.68 | [+0.55, +2.87] | 7.4 |
| B-audio - A-all | orig | all_orders | +1.06 | [+0.16, +2.08] | 8.1 |
| B-audio - A-all | orig_reworded | training_order | +1.31 | [-0.45, +3.73] | 8.5 |
| B-audio - A-all | orig_reworded | other_orders | +1.03 | [-0.58, +3.20] | 8.1 |
| B-audio - A-all | orig_reworded | all_orders | +1.10 | [-0.37, +3.16] | 8.2 |
| B-audio - A-all | text | all_orders | -1.56 | [-3.25, +0.27] | 12.2 |
| B-audio - A-all | question_only | all_orders | +0.00 | [+0.00, +0.00] | 50.0 |
| C-post-audio - A-all | orig | training_order | -3.45 | [-5.19, -1.76] | 8.1 |
| C-post-audio - A-all | orig | other_orders | -6.58 | [-8.62, -4.73] | 11.5 |
| C-post-audio - A-all | orig | all_orders | -5.80 | [-7.63, -4.13] | 10.6 |
| C-post-audio - A-all | orig_reworded | training_order | -5.91 | [-7.83, -3.59] | 14.4 |
| C-post-audio - A-all | orig_reworded | other_orders | -9.05 | [-11.02, -6.83] | 16.2 |
| C-post-audio - A-all | orig_reworded | all_orders | -8.27 | [-10.05, -6.22] | 15.8 |
| C-post-audio - A-all | text | all_orders | -1.27 | [-2.09, -0.25] | 5.8 |
| C-post-audio - A-all | question_only | all_orders | +0.00 | [+0.00, +0.00] | 25.0 |
| D-all-shuffled - A-all | orig | training_order | +1.22 | [-0.10, +2.56] | 6.6 |
| D-all-shuffled - A-all | orig | other_orders | +2.88 | [+0.93, +5.07] | 15.6 |
| D-all-shuffled - A-all | orig | all_orders | +2.46 | [+0.83, +4.25] | 13.4 |
| D-all-shuffled - A-all | orig_reworded | training_order | +0.01 | [-1.26, +1.28] | 6.1 |
| D-all-shuffled - A-all | orig_reworded | other_orders | +2.53 | [+0.94, +4.50] | 9.7 |
| D-all-shuffled - A-all | orig_reworded | all_orders | +1.90 | [+0.67, +3.40] | 8.8 |
| D-all-shuffled - A-all | text | all_orders | +0.20 | [-0.54, +0.92] | 8.5 |
| D-all-shuffled - A-all | question_only | all_orders | +0.00 | [+0.00, +0.00] | 75.0 |
| B-audio - D-all-shuffled | orig | training_order | -2.00 | [-4.00, -0.22] | 8.7 |
| B-audio - D-all-shuffled | orig | other_orders | -1.20 | [-2.97, +0.28] | 13.4 |
| B-audio - D-all-shuffled | orig | all_orders | -1.40 | [-3.08, +0.04] | 12.2 |
| B-audio - D-all-shuffled | orig_reworded | training_order | +1.30 | [-0.62, +3.60] | 8.0 |
| B-audio - D-all-shuffled | orig_reworded | other_orders | -1.50 | [-2.60, -0.22] | 6.5 |
| B-audio - D-all-shuffled | orig_reworded | all_orders | -0.80 | [-1.92, +0.51] | 6.9 |
| B-audio - D-all-shuffled | text | all_orders | -1.75 | [-3.20, -0.21] | 10.7 |
| B-audio - D-all-shuffled | question_only | all_orders | +0.00 | [+0.00, +0.00] | 25.0 |
| B-audio - C-post-audio | orig | training_order | +2.68 | [-0.28, +5.49] | 16.6 |
| B-audio - C-post-audio | orig | other_orders | +8.26 | [+6.57, +10.21] | 15.4 |
| B-audio - C-post-audio | orig | all_orders | +6.86 | [+5.19, +8.73] | 15.7 |
| B-audio - C-post-audio | orig_reworded | training_order | +7.22 | [+4.51, +10.12] | 18.5 |
| B-audio - C-post-audio | orig_reworded | other_orders | +10.08 | [+7.54, +13.29] | 19.2 |
| B-audio - C-post-audio | orig_reworded | all_orders | +9.37 | [+7.00, +12.27] | 19.0 |
| B-audio - C-post-audio | text | all_orders | -0.29 | [-2.27, +1.66] | 14.8 |
| B-audio - C-post-audio | question_only | all_orders | +0.00 | [+0.00, +0.00] | 75.0 |
| A-all - frozen-base | orig | training_order | +24.09 | [+20.71, +28.32] | 31.2 |
| A-all - frozen-base | orig | other_orders | +15.25 | [+12.39, +18.30] | 25.5 |
| A-all - frozen-base | orig | all_orders | +17.46 | [+14.70, +20.63] | 26.9 |
| A-all - frozen-base | orig_reworded | training_order | +17.30 | [+14.05, +20.63] | 25.0 |
| A-all - frozen-base | orig_reworded | other_orders | +14.79 | [+12.50, +17.11] | 26.1 |
| A-all - frozen-base | orig_reworded | all_orders | +15.42 | [+13.05, +17.75] | 25.8 |
| A-all - frozen-base | text | all_orders | +1.56 | [-0.27, +3.25] | 12.2 |
| A-all - frozen-base | question_only | all_orders | +0.00 | [+0.00, +0.00] | 50.0 |
| B-audio - frozen-base | orig | training_order | +23.31 | [+20.05, +27.12] | 28.0 |
| B-audio - frozen-base | orig | other_orders | +16.93 | [+13.91, +20.29] | 25.6 |
| B-audio - frozen-base | orig | all_orders | +18.53 | [+15.52, +21.86] | 26.2 |
| B-audio - frozen-base | orig_reworded | training_order | +18.62 | [+14.84, +23.03] | 26.9 |
| B-audio - frozen-base | orig_reworded | other_orders | +15.82 | [+13.10, +19.22] | 26.7 |
| B-audio - frozen-base | orig_reworded | all_orders | +16.52 | [+13.63, +20.07] | 26.7 |
| B-audio - frozen-base | text | all_orders | +0.00 | [+0.00, +0.00] | 0.0 |
| B-audio - frozen-base | question_only | all_orders | +0.00 | [+0.00, +0.00] | 0.0 |
| C-post-audio - frozen-base | orig | training_order | +20.63 | [+16.81, +25.38] | 33.8 |
| C-post-audio - frozen-base | orig | other_orders | +8.67 | [+6.40, +11.12] | 23.9 |
| C-post-audio - frozen-base | orig | all_orders | +11.66 | [+9.27, +14.46] | 26.4 |
| C-post-audio - frozen-base | orig_reworded | training_order | +11.39 | [+8.27, +15.03] | 19.3 |
| C-post-audio - frozen-base | orig_reworded | other_orders | +5.74 | [+3.81, +7.96] | 16.7 |
| C-post-audio - frozen-base | orig_reworded | all_orders | +7.15 | [+5.24, +9.41] | 17.3 |
| C-post-audio - frozen-base | text | all_orders | +0.29 | [-1.66, +2.27] | 14.8 |
| C-post-audio - frozen-base | question_only | all_orders | +0.00 | [+0.00, +0.00] | 75.0 |
| D-all-shuffled - frozen-base | orig | training_order | +25.31 | [+21.95, +29.32] | 31.9 |
| D-all-shuffled - frozen-base | orig | other_orders | +18.13 | [+15.12, +21.93] | 25.3 |
| D-all-shuffled - frozen-base | orig | all_orders | +19.93 | [+16.91, +23.70] | 26.9 |
| D-all-shuffled - frozen-base | orig_reworded | training_order | +17.32 | [+13.88, +20.73] | 26.1 |
| D-all-shuffled - frozen-base | orig_reworded | other_orders | +17.32 | [+14.84, +20.34] | 25.0 |
| D-all-shuffled - frozen-base | orig_reworded | all_orders | +17.32 | [+14.77, +20.33] | 25.2 |
| D-all-shuffled - frozen-base | text | all_orders | +1.75 | [+0.21, +3.20] | 10.7 |
| D-all-shuffled - frozen-base | question_only | all_orders | +0.00 | [+0.00, +0.00] | 25.0 |

| transcription contrast | dWER pp | 95% CI |
|---|---:|---|
| B-audio - A-all | +0.83 | [+0.15, +1.64] |
| C-post-audio - A-all | +0.86 | [+0.15, +1.62] |
| D-all-shuffled - A-all | -0.62 | [-1.85, +0.38] |
| B-audio - D-all-shuffled | +1.45 | [+0.44, +2.71] |
| B-audio - C-post-audio | -0.03 | [-0.65, +0.55] |
| A-all - frozen-base | -1.03 | [-2.02, -0.17] |
| B-audio - frozen-base | -0.21 | [-0.71, +0.31] |
| C-post-audio - frozen-base | -0.18 | [-0.55, +0.14] |
| D-all-shuffled - frozen-base | -1.65 | [-2.87, -0.62] |

Checks declared before training:

- hear_side_learns_the_task: **yes**
- hear_side_not_below_base_in_other_orders: **yes**
- hear_side_beats_plain_in_other_orders: **no**
- hear_side_beats_shuffling_somewhere: **no**

Outcome under the declared rule: **stop**
