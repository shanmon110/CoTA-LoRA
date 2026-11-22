import numpy as np
import pytest
from scipy.signal import hilbert

from experiments.trr_diagnostics import twin

RATE = 16000
PARAMS = dict(twin.DEFAULTS)


class SineVocoder:
    """A one-harmonic stand-in for WORLD: F0 from instantaneous frequency, amplitude per frame."""
    name = 'sine'

    def __init__(self, period_ms=5.0):
        self.hop = int(RATE * period_ms / 1000)

    def _frames(self, values):
        usable = values[:values.size // self.hop * self.hop]
        return usable.reshape(-1, self.hop)

    def f0(self, samples, rate):
        phase = np.unwrap(np.angle(hilbert(samples)))
        return np.median(self._frames(np.diff(phase, append=phase[-1]) * rate / (2 * np.pi)), axis=1)

    def analyze(self, samples, rate):
        return self.f0(samples, rate), np.sqrt(2 * np.mean(np.square(self._frames(samples)), axis=1))

    def synthesize(self, f0, amplitude, rate):
        track = np.repeat(f0, self.hop)
        return np.repeat(amplitude, self.hop) * np.sin(2 * np.pi * np.cumsum(track) / rate)


def test_flatten_f0_moves_only_voiced_frames_to_their_median():
    f0 = np.array([0., 100., 200., 0., 400.])
    flat = twin.flatten_f0(f0)
    assert np.allclose(flat, [0., 200., 200., 0., 200.]) and flat[0] == 0 and flat[3] == 0
    assert np.allclose(twin.flatten_f0(f0, strength=0.0), f0)
    half = twin.flatten_f0(f0, strength=0.5)
    assert np.allclose(half[[1, 2, 4]], [200 / 2 ** 0.5, 200., 200 * 2 ** 0.5])
    assert np.array_equal(twin.flatten_f0(np.zeros(4)), np.zeros(4))


def _speech_like(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    envelope = 0.02 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) ** 2
    signal = rng.standard_normal(n) * envelope
    signal[int(1.2 * RATE):int(1.9 * RATE)] *= 1e-4  # a pause
    return signal


def test_energy_stats_use_the_given_mask():
    signal = _speech_like()
    mask = twin.speech_mask(twin.slow_power(signal, RATE, 250), 30)
    flat = signal * twin.energy_gain(signal, RATE, window_ms=250, floor_db=30, max_gain_db=12, edge_ms=50)
    vocoder = SineVocoder()
    before = twin.contour_stats(signal, RATE, vocoder, PARAMS, mask)['energy_sd_db']
    after = twin.contour_stats(flat, RATE, vocoder, PARAMS, mask)['energy_sd_db']
    assert after < 0.35 * before
    assert twin.contour_stats(signal, RATE, vocoder, PARAMS)['energy_sd_db'] == pytest.approx(before)


def test_energy_gain_flattens_speech_and_leaves_pauses_alone():
    signal = _speech_like()
    kwargs = dict(window_ms=250, floor_db=30, max_gain_db=12, edge_ms=50)
    gain = twin.energy_gain(signal, RATE, **kwargs)
    assert gain.shape == signal.shape
    assert np.all(gain <= 10 ** (12 / 20) + 1e-9) and np.all(gain >= 10 ** (-12 / 20) - 1e-9)
    pause = slice(int(1.45 * RATE), int(1.65 * RATE))
    assert np.allclose(gain[pause], 1.0, atol=0.02)

    def spread(x):
        power = twin.slow_power(x, RATE, 250)
        active = twin.speech_mask(twin.slow_power(signal, RATE, 250), 30)
        return (10 * np.log10(power[active])).std()

    assert spread(signal * gain) < 0.35 * spread(signal)
    assert np.allclose(twin.energy_gain(signal, RATE, strength=0.0, **kwargs), 1.0)
    assert np.allclose(twin.energy_gain(np.zeros(RATE), RATE, **kwargs), 1.0)


def test_level_matching_and_the_shared_peak_guard():
    reference = 0.1 * np.sin(np.linspace(0, 200, 4000))
    out = twin.match_level(np.ones(5000) * 0.5, reference)
    assert out.size == 4000 and twin.rms(out) == pytest.approx(twin.rms(reference))
    short = twin.match_level(np.ones(10), reference)
    assert short.size == 4000 and np.all(short[10:] == 0)
    assert twin.common_peak_scale([np.array([0.5, -0.2]), np.array([0.1])]) == 1.0
    assert twin.common_peak_scale([np.array([0.5]), np.array([-1.98])]) == pytest.approx(0.5)


class PeakyVocoder(SineVocoder):
    """Resynthesis that adds a click, as a parametric vocoder can on a near-full-scale recording."""

    def synthesize(self, f0, amplitude, rate):
        out = super().synthesize(f0, amplitude, rate)
        if np.ptp(f0) > 1e-6:      # only the carrier whose pitch still moves gets the click
            out[100] = 10 * np.max(np.abs(out))
        return out


def test_twins_of_one_utterance_share_one_level_even_when_one_would_clip():
    source = twin.synthetic_utterance(RATE, harmonics=1, amplitude=(0.3, 0.9))
    twins, scale = twin.make_twins(source, RATE, PeakyVocoder(), PARAMS)
    assert scale < 1.0
    levels = {name: twin.rms(wave_) for name, wave_ in twins.items()}
    assert max(levels.values()) / min(levels.values()) == pytest.approx(1.0, abs=1e-9)
    assert levels['sham'] == pytest.approx(scale * twin.rms(source))
    assert max(np.max(np.abs(wave_)) for wave_ in twins.values()) == pytest.approx(0.99)


def test_wav_round_trip_and_clipping_count(tmp_path):
    samples = np.array([0.0, 0.5, -0.5, 0.25])
    assert twin.write_wav(tmp_path / 'a.wav', samples, RATE) == 0
    back, rate = twin.read_wav(tmp_path / 'a.wav')
    assert rate == RATE and np.allclose(back, samples, atol=1 / 32768)
    assert twin.write_wav(tmp_path / 'b.wav', np.array([1.5, -1.5, 0.0]), RATE) == 2


def test_twins_edit_only_the_contour_they_name():
    result = twin.self_check(SineVocoder(), PARAMS, harmonics=1)
    assert result['passed'], result['checks']


def test_zero_strength_twins_equal_the_sham():
    source = twin.synthetic_utterance(RATE, harmonics=1)
    twins, _ = twin.make_twins(source, RATE, SineVocoder(), dict(PARAMS, strength=0.0))
    for name in ('flat_f0', 'flat_energy', 'flat_both'):
        assert np.allclose(twins[name], twins['sham'], atol=1e-9), name


def test_only_requested_twins_are_made_and_unknown_names_fail():
    source = twin.synthetic_utterance(RATE, harmonics=1)
    twins, scale = twin.make_twins(source, RATE, SineVocoder(), PARAMS, ['flat_both'])
    assert list(twins) == ['flat_both'] and scale == 1.0
    with pytest.raises(ValueError):
        twin.make_twins(source, RATE, SineVocoder(), PARAMS, ['flat_rhythm'])


def test_backends():
    assert set(twin.BACKENDS) == {'praat', 'world'} and twin.DEFAULTS['backend'] == 'praat'
    with pytest.raises(ValueError):
        twin.make_backend(dict(PARAMS, backend='griffin-lim'))
