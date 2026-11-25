# Keeping the emotion or keeping the letter when the option order changes

Agreement: share of the six pairs of option orders in which a model gave the same letter, or chose the same emotion. Training letter: the letter the true emotion had in the training order, counted under the three other orders; among errors, 33.3% is expected if errors were spread evenly. Percent, with 95% intervals over clusters and seeds.

## IEMOCAP Session 4 (1026 utterances)

| model (seeds) | same letter | same emotion | training letter among errors | training letter, all answers |
|---|---|---|---|---|
| frozen-base (1) | 5.0 [4.3, 5.7] | 86.2 [84.5, 88.1] | 30.5 [28.5, 32.2] | 11.5 [9.0, 14.2] |
| A-first4 (3) | 3.3 [2.4, 4.3] | 89.8 [87.1, 92.3] | 31.6 [29.6, 33.3] | 9.4 [7.7, 11.4] |
| A-mid4 (3) | 3.8 [3.1, 4.7] | 87.8 [85.2, 90.2] | 31.0 [29.0, 33.2] | 9.8 [7.8, 12.1] |
| A-last4 (3) | 12.8 [10.6, 15.4] | 65.3 [59.6, 70.5] | 27.5 [21.9, 32.9] | 12.6 [8.8, 17.0] |
| A-all (3) | 4.1 [3.2, 5.1] | 88.4 [86.2, 90.6] | 32.9 [31.2, 34.7] | 9.4 [7.3, 11.8] |
| B-audio (3) | 2.9 [2.4, 3.3] | 91.0 [89.5, 92.6] | 32.6 [30.4, 34.6] | 8.5 [6.8, 10.5] |
| D-all-shuffled (3) | 2.3 [1.9, 2.8] | 93.5 [92.3, 94.8] | 31.3 [29.6, 32.6] | 7.7 [6.1, 9.5] |

A-last4 minus A-all, training letter among errors: -5.5 pp [-10.4, -0.8]. Reading: **not shown**

## CREMA-D (1092 utterances)

| model (seeds) | same letter | same emotion | training letter among errors | training letter, all answers |
|---|---|---|---|---|
| frozen-base (1) | 6.2 [5.5, 7.0] | 82.6 [80.8, 84.4] | 32.9 [32.0, 33.8] | 17.1 [16.2, 17.9] |
| A-first4 (3) | 6.6 [4.5, 9.5] | 82.3 [76.3, 86.7] | 31.3 [30.0, 32.5] | 11.7 [10.3, 13.0] |
| A-mid4 (3) | 3.5 [2.3, 5.3] | 88.9 [84.3, 92.2] | 32.2 [30.1, 33.9] | 13.3 [11.7, 14.9] |
| A-last4 (3) | 10.9 [8.9, 13.3] | 70.1 [65.3, 74.2] | 35.7 [34.6, 36.9] | 19.7 [18.9, 20.6] |
| A-all (3) | 3.7 [3.3, 4.2] | 88.3 [86.9, 89.5] | 32.9 [31.7, 34.0] | 12.3 [11.2, 13.4] |
| B-audio (3) | 3.4 [2.7, 4.0] | 88.8 [87.0, 90.4] | 32.3 [31.1, 33.6] | 12.0 [10.8, 13.3] |
| D-all-shuffled (3) | 2.7 [2.2, 3.4] | 92.5 [91.0, 93.9] | 32.5 [31.2, 33.8] | 11.6 [10.4, 12.8] |

A-last4 minus A-all, training letter among errors: +2.9 pp [+1.1, +4.5]. Reading: **not shown**

