# Headroom for input-conditional adaptation

Shared adapter: D-all-shuffled. Session 4, UAR in % over the four option orders.

## Step zero: the shared adapter within each group

| grouping | group | utterances | classes present | shared UAR per seed |
|---|---|---:|---:|---|
| gender | F | 523 | 4 | 75.5 / 74.1 / 68.7 |
| gender | M | 503 | 4 | 75.5 / 76.2 / 68.2 |
| duration | short | 354 | 4 | 69.6 / 69.0 / 64.8 |
| duration | medium | 358 | 4 | 75.1 / 74.0 / 67.0 |
| duration | long | 314 | 4 | 79.6 / 80.2 / 72.3 |
| scripted | impro | 529 | 4 | 79.9 / 79.4 / 71.7 |
| scripted | script | 497 | 4 | 69.3 / 69.5 / 63.5 |

## Group specialists against the shared adapter (seed 42)

| grouping | specialists | shared | random specialists | specialists − shared pp [95% CI] | specialists − random pp [95% CI] | headroom |
|---|---:|---:|---:|---|---|---|
| gender | 72.8 | 76.0 | 69.4 | -3.20 [-6.06, -0.81] | +3.40 [+0.50, +6.95] | no |
| duration | 70.2 | 76.0 | 69.1 | -5.85 [-8.72, -3.59] | +1.03 [-0.83, +3.18] | no |
| scripted | 74.3 | 76.0 | 69.4 | -1.72 [-4.31, +0.32] | +4.88 [+2.52, +7.26] | no |

Seed standard deviation of the shared adapter: 4.06 pp; a gain must exceed 2 times it and 2 pp.

Outcome under the declared rule: **closed**
