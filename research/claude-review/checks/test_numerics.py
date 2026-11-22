"""Run with: python -m pytest -q research/claude-review/checks  (NumPy only, no model)."""
import math

import numpy as np

import numerics


def test_bf16_rounding_matches_the_format():
    # 1 + 2^-7 is the next BF16 value after 1; halfway cases round to even.
    values = np.array([1.0, 1 + 2.0 ** -7, 1 + 2.0 ** -8, 1 + 3 * 2.0 ** -8, 0.1], dtype=np.float32)
    rounded = numerics.to_bf16(values)
    assert rounded[0] == 1 and rounded[1] == np.float32(1 + 2.0 ** -7)
    assert rounded[2] == 1 and rounded[3] == np.float32(1 + 2.0 ** -6)
    assert abs(rounded[4] - 0.1) <= 0.1 * 2.0 ** -8


def test_small_gate_deviations_are_erased_by_the_bf16_cast():
    assert numerics.gate_product_survival(0.0019) == 0.0      # below 2^-9: never visible
    assert numerics.gate_product_survival(0.0040) == 1.0      # above 2^-8: always visible
    assert 0.0 < numerics.gate_product_survival(0.0030) < 1.0


def test_published_gate_probe_sits_below_the_bf16_threshold():
    arms = numerics.bf16_gate_summary()['arms']
    base, fast = arms['question_audio lr=2e-4'], arms['fastgate lr=3.2e-3']
    assert base['share_of_domains_with_median_below_2_pow_minus_9'] == 1.0
    assert base['estimated_fraction_of_coefficients_changed_in_bf16'] < 0.3
    assert fast['estimated_fraction_of_coefficients_changed_in_bf16'] > 0.8


def test_gate_weights_match_a_signal_free_adam_random_walk():
    summary = numerics.gate_random_walk_summary()
    reference = summary['adam_reference_rms_over_lr_sqrt_T']
    for arm in summary['arms'].values():
        assert 0.8 < arm['observed_over_noise_only'] < 1.1
        assert arm['observed_over_constant_sign'] < 0.1
        assert arm['observed_over_noise_only'] < reference['snr=0.1']
    assert math.isclose(summary['fastgate_over_base_per_layer']['mean'], 16.0, rel_tol=0.1)


def test_gate_gradient_is_the_radial_gradient_of_the_readouts():
    for seed in range(5):
        check = numerics.redundancy_check(seed)
        assert check['max_abs_identity_error'] < 1e-6 * max(1.0, check['scale'])
        assert check['gate_gradient_at_zero_init'] == 0.0


def test_selection_metric_is_still_rising_and_noisier_than_arm_gaps():
    summary = numerics.selection_summary()
    assert all(arm['still_rising_at_selection'] for arm in summary['arms'].values())
    assert sum(arm['last_regular_eval_is_best'] for arm in summary['arms'].values()) == 8
    variants = summary['same_seed_gate_variants_pairs_of_96'].values()
    assert (max(variants) - min(variants)) / 96 * 100 < summary['median_abs_change_in_8_updates_pp']


def test_cota_versus_lora_discordance():
    rows = numerics.discordance_summary()
    assert rows['MMAU CoTA vs LoRA']['exact_mcnemar_p'] > 0.5
    assert rows['MMAU CoTA vs CoTA-off']['exact_mcnemar_p'] > 0.5
    assert rows['MMAU LoRA vs frozen']['exact_mcnemar_p'] < 0.01
