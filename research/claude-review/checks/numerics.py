"""NumPy-only checks behind research/claude-review/DIAGNOSIS.md.

Nothing here loads a model. Each function either re-derives a number from the
evidence files already published in research/evidence, or verifies an
arithmetic/algebraic claim about experiments/ms_swift_librispeech/swift_asr_ta_runtime.py
on a NumPy mirror of the relevant lines.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'research' / 'evidence'
RANK = 8  # adapter.rank of every conditional arm in raw/ta-conditional-01/configs


# --------------------------------------------------------------------------- BF16

def to_bf16(values):
    """Round float32 to bfloat16 (round-to-nearest-even), returned as float32."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits & 0xFFFF0000).astype(np.uint32).view(np.float32)


def gate_product_survival(deviation, samples=200_000, seed=0):
    """Fraction of BF16 coefficients changed by `(z * m).to(bf16)` for m = 1 +/- deviation.

    Mirrors swift_asr_ta_runtime.py line 72, where `extra` is the BF16 output of an
    autocast F.linear and `multiplier` is FP32.
    """
    rng = np.random.default_rng(seed)
    z = to_bf16(rng.standard_normal(samples).astype(np.float32))
    sign = rng.choice(np.array([-1., 1.], dtype=np.float32), samples)
    gated = to_bf16(z * (np.float32(1) + sign * np.float32(deviation)))
    return float(np.mean(gated != z))


def bf16_gate_summary():
    probe = json.loads((EVIDENCE / 'derived/ta-conditional-01/gate-probe.json').read_text())
    fast = json.loads((EVIDENCE / 'derived/ta-conditional-01/gate-probe-fastgate.json').read_text())
    out = dict(never_changes_below=2.0 ** -9, always_changes_above=2.0 ** -8, arms={})
    rng = np.random.default_rng(1)
    for name, data in (('question_audio lr=2e-4', probe), ('fastgate lr=3.2e-3', fast)):
        stats = [v['abs_gate_deviation_from_one'] for v in data['domain_metrics'].values()]
        n = sum(s['n'] for s in stats)
        mean = sum(s['mean'] * s['n'] for s in stats) / n
        median = float(np.median([s['median'] for s in stats]))
        # The published quantiles are consistent with an exponential law of the same mean
        # (its median is mean*ln2); use it only to turn quantiles into a survival estimate.
        draws = rng.exponential(mean, 200_000).astype(np.float32)
        z = to_bf16(rng.standard_normal(draws.size).astype(np.float32))
        sign = rng.choice(np.array([-1., 1.], dtype=np.float32), draws.size)
        changed = float(np.mean(to_bf16(z * (1 + sign * draws)) != z))
        out['arms'][name] = dict(
            mean_abs_deviation=mean, median_abs_deviation=median, exponential_median=mean * math.log(2),
            share_of_domains_with_median_below_2_pow_minus_9=float(np.mean([s['median'] < 2.0 ** -9 for s in stats])),
            estimated_fraction_of_coefficients_changed_in_bf16=changed)
    return out


# ------------------------------------------------------------- Adam random walk of R

def adam_displacement(steps, lr, drift, noise=1.0, params=20_000, seed=0, betas=(0.9, 0.999), eps=1e-8):
    """RMS displacement of zero-initialised parameters under Adam for g_t = drift + noise*xi_t."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(params)
    m = np.zeros(params)
    v = np.zeros(params)
    for t in range(1, steps + 1):
        g = drift + noise * rng.standard_normal(params)
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        theta -= lr * (m / (1 - betas[0] ** t)) / (np.sqrt(v / (1 - betas[1] ** t)) + eps)
    return float(np.sqrt(np.mean(theta ** 2)))


def gate_random_walk_summary():
    steps = 160  # both probes read checkpoint-160
    audit = json.loads((EVIDENCE / 'raw/ta-conditional-01/fastgate-optimizer-audit.json').read_text())
    rates = sorted({g['initial_lr'] for g in audit['groups'] if g['tensors']})
    assert rates == [0.0002, 0.0032], rates
    files = {0.0002: 'gate-probe.json', 0.0032: 'gate-probe-fastgate.json'}
    out = dict(steps=steps, rank=RANK, arms={})
    per_layer = {}
    for lr, name in files.items():
        data = json.loads((EVIDENCE / 'derived/ta-conditional-01' / name).read_text())
        frob = np.array([layer['frobenius'] for layer in data['gate_weights']])
        rms = frob / RANK  # R is rank x rank, so RMS element = ||R||_F / rank
        per_layer[lr] = rms
        out['arms'][name] = dict(
            lr=lr, mean_rms_element=float(rms.mean()), min_rms_element=float(rms.min()),
            max_rms_element=float(rms.max()),
            noise_only_prediction_lr_sqrt_T=lr * math.sqrt(steps),
            constant_sign_prediction_lr_T=lr * steps,
            observed_over_noise_only=float(rms.mean() / (lr * math.sqrt(steps))),
            observed_over_constant_sign=float(rms.mean() / (lr * steps)))
    out['fastgate_over_base_per_layer'] = dict(
        mean=float(np.mean(per_layer[0.0032] / per_layer[0.0002])), lr_ratio=16.0)
    # Reference: what Adam does to a zero-init weight for a given gradient signal-to-noise ratio.
    out['adam_reference_rms_over_lr_sqrt_T'] = {
        f'snr={snr}': adam_displacement(steps, 1.0, snr) / math.sqrt(steps) for snr in (0.0, 0.1, 0.25, 0.5, 1.0, 4.0)}
    return out


# ---------------------------------------------------------- gate / readout redundancy

def transport(params, z, feature, scale=2.0):
    """NumPy mirror of _SharedAtoms.answer_update's transport term for one row."""
    b, c, g, r = params
    m = 2.0 / (1.0 + np.exp(-(r @ feature)))
    coeff = m * z
    return scale * (b @ (g * coeff) + c @ coeff), m


def redundancy_check(seed=0, rank=4, width=12):
    """m_k dL/dm_k == (C^T dL/dC)_kk + g_k dL/dg_k, exactly, for any loss of the residual."""
    rng = np.random.default_rng(seed)
    b, c = rng.standard_normal((2, width, rank))
    g = rng.standard_normal(rank)
    r = 0.3 * rng.standard_normal((rank, rank))
    z = rng.standard_normal(rank)
    feature = rng.standard_normal(rank)
    target = rng.standard_normal(width)

    def loss_with(multiplier=None, c_=c, g_=g, scale=2.0):
        m = 2.0 / (1.0 + np.exp(-(r @ feature))) if multiplier is None else multiplier
        out = scale * (b @ (g_ * m * z) + c_ @ (m * z))
        return 0.5 * np.sum((out - target) ** 2)

    _, m0 = transport((b, c, g, r), z, feature)
    h = 1e-6

    def grad(fn, x):
        flat = x.astype(float).ravel()
        out = np.zeros_like(flat)
        for i in range(flat.size):
            hi, lo = flat.copy(), flat.copy()
            hi[i] += h
            lo[i] -= h
            out[i] = (fn(hi.reshape(x.shape)) - fn(lo.reshape(x.shape))) / (2 * h)
        return out.reshape(x.shape)

    d_m = grad(lambda v: loss_with(multiplier=v), m0)
    d_c = grad(lambda v: loss_with(c_=v), c)
    d_g = grad(lambda v: loss_with(g_=v), g)
    lhs = m0 * d_m
    rhs = np.einsum('ik,ik->k', c, d_c) + g * d_g
    # At the repository's initialisation (C = 0, g = 0) the right-hand side is identically zero.
    d_m_init = grad(lambda v: 0.5 * np.sum((2.0 * (b @ (0 * v * z) + np.zeros_like(c) @ (v * z)) - target) ** 2), m0)
    return dict(max_abs_identity_error=float(np.max(np.abs(lhs - rhs))), scale=float(np.max(np.abs(lhs))),
                gate_gradient_at_zero_init=float(np.max(np.abs(d_m_init))))


# ------------------------------------------------------------ selection and test noise

def mcnemar_exact(only_a, only_b):
    """Two-sided exact binomial test on discordant pairs."""
    n, k = only_a + only_b, min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def selection_summary():
    base = EVIDENCE / 'raw/ta-conditional-01/train'
    arms = {}
    for folder in sorted(base.iterdir()):
        rows = [json.loads(line) for line in (folder / 'logging.jsonl').read_text().splitlines()]
        curve = [(int(r['global_step/max_steps'].split('/')[0]), r['eval_pair_accuracy'])
                 for r in rows if 'eval_pair_accuracy' in r]
        steps = dict(curve)
        best = max(curve, key=lambda item: (item[1], -item[0]))
        arms[folder.name] = dict(
            curve_pairs_of_96=[round(v * 96) for _, v in curve], best_step=best[0],
            last_regular_eval_is_best=best[0] == 160,
            still_rising_at_selection=steps[160] > steps[128],
            change_160_to_168_pp=round(100 * (steps[168] - steps[160]), 2))
    swings = [abs(a['change_160_to_168_pp']) for a in arms.values()]
    p = 0.7
    return dict(arms=arms, median_abs_change_in_8_updates_pp=float(np.median(swings)),
                max_abs_change_in_8_updates_pp=float(max(swings)),
                binomial_se_pp_at_70pct_of_96=100 * math.sqrt(p * (1 - p) / 96),
                same_seed_gate_variants_pairs_of_96=dict(static=66, question=70, question_audio=71, fastgate=68))


def discordance_summary():
    ser = json.loads((EVIDENCE / 'raw/ta-conditional-transfer-01/ser-verified.json').read_text())['contrasts']
    mmau = json.loads((EVIDENCE / 'raw/ta-conditional-transfer-01/mmau_scores.json').read_text())['pairs']
    out = {}
    for label, source, key in (
            ('MMAU CoTA vs LoRA', mmau, 'question_audio-mixed vs lora-mixed'),
            ('MMAU CoTA vs CoTA-off', mmau, 'question_audio-mixed vs question_audio-mixed-off'),
            ('MMAU LoRA vs frozen', mmau, 'lora-mixed vs frozen-base'),
            ('IEMOCAP-acc CoTA vs LoRA', ser, 'question_audio-mixed vs lora-mixed'),
            ('IEMOCAP-acc CoTA vs CoTA-off', ser, 'question_audio-mixed vs question_audio-mixed-off'),
            ('IEMOCAP-acc LoRA vs frozen', ser, 'lora-mixed vs frozen-base')):
        row = source[key].get('overall', source[key])
        a, b = row['candidate_only_correct'], row['reference_only_correct']
        out[label] = dict(candidate_only=a, reference_only=b, exact_mcnemar_p=mcnemar_exact(a, b))
    return out


def run_all():
    return dict(bf16_gate=bf16_gate_summary(), gate_random_walk=gate_random_walk_summary(),
                gate_readout_redundancy=redundancy_check(), selection=selection_summary(),
                discordance=discordance_summary())


if __name__ == '__main__':
    result = run_all()
    target = Path(__file__).with_name('results.json')
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
