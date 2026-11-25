# Linear probes of the emotion beside the answers the model gave

Mean per-class recall in %. Probes: trained on Sessions 1-2, regularisation and layer chosen on Session 3. Control: the same probe trained on permuted labels, mean of five permutations (25 expected).

## session4 (1026 utterances)

| model | answer, all orders | answer, training order | probe at audio positions (layer) | probe at answer position (layer) | controls | best probe minus answer [95% CI] |
|---|---:|---:|---:|---:|---|---|
| A-all | 73.6 | 75.9 | 77.3 (27) | 78.3 (27) | 26.0, 26.5 | +4.7 [+2.7, +7.1] |
| A-last4 | 51.3 | 65.4 | 76.5 (27) | 74.6 (32) | 25.6, 25.2 | +25.3 [+22.2, +29.4] |
| D-all-shuffled | 76.0 | 77.8 | 77.4 (27) | 77.6 (27) | 25.7, 25.6 | +1.6 [-0.3, +3.7] |
| frozen-base | 56.1 | 52.0 | 76.5 (27) | 74.6 (32) | 25.6, 25.2 | +20.4 [+17.1, +24.8] |

Readings declared before extraction:

- perception_use_gap: **present**
- information_reaches_the_answer_position: **unclear**
- what_full_depth_training_adds: **no more than a linear readout of the base**
- A-all: its answers minus the best probe of the frozen base (audio site): -3.0 [-6.8, +0.4] pp
- A-last4: its answers minus the best probe of the frozen base (audio site): -25.3 [-29.4, -22.2] pp
- D-all-shuffled: its answers minus the best probe of the frozen base (audio site): -0.5 [-4.0, +2.8] pp

## cremad (1092 utterances)

| model | answer, all orders | answer, training order | probe at audio positions (layer) | probe at answer position (layer) | controls | best probe minus answer [95% CI] |
|---|---:|---:|---:|---:|---|---|
| A-all | 63.2 | 60.5 | 37.7 (27) | 60.3 (27) | 25.4, 27.6 | -2.9 [-5.1, -0.8] |
| A-last4 | 46.6 | 55.1 | 33.4 (27) | 54.9 (32) | 24.2, 25.6 | +8.3 [+6.2, +10.3] |
| D-all-shuffled | 63.7 | 64.3 | 39.1 (27) | 61.6 (27) | 25.5, 26.9 | -2.0 [-4.1, +0.0] |
| frozen-base | 47.7 | 46.6 | 33.4 (27) | 54.9 (32) | 24.2, 25.6 | +7.2 [+4.9, +9.5] |

Readings declared before extraction:

- perception_use_gap: **unclear**
- information_reaches_the_answer_position: **yes**
- what_full_depth_training_adds: **between**
- A-all: its answers minus the best probe of the frozen base (answer site): +8.4 [+6.1, +10.6] pp
- A-last4: its answers minus the best probe of the frozen base (answer site): -8.3 [-10.3, -6.2] pp
- D-all-shuffled: its answers minus the best probe of the frozen base (answer site): +8.8 [+6.5, +11.1] pp

