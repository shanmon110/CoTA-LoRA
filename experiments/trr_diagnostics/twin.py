"""Prosody-flattened twins of an utterance.

sham         pitch analysis -> resynthesis with the contour untouched (resynthesis artefacts only)
flat_f0      same path, voiced F0 pulled to its utterance median
flat_energy  sham output with its slow energy contour pulled to its median
flat_both    flat_f0 output with its slow energy contour pulled to its median

The default backend is Praat's overlap-add (PSOLA) resynthesis, which moves pitch periods of
the recorded waveform instead of rebuilding the signal from vocoder parameters. The first GPU
smoke run showed why: WORLD copy-synthesis of these recordings tripled the base model's word
error rate before any contour was edited. WORLD stays available as backend "world".

Every output is trimmed or padded to the input length and scaled to the input RMS. If any twin
of an utterance would clip, all twins of that utterance are scaled down together, so the twins
always share one level and differ only in the contour that was edited.

Scope: F0 and slow energy only. Rhythm, pausing and voice quality are not edited.
"""
import argparse
import json
import sys
import wave

import numpy as np
from scipy.signal import fftconvolve

AUDIO_CONDITIONS = ('sham', 'flat_f0', 'flat_energy', 'flat_both')
DEFAULTS = dict(backend='praat', frame_period_ms=10.0, f0_floor_hz=75.0, f0_ceil_hz=600.0,
                energy_window_ms=250.0, energy_floor_db=30.0, max_gain_db=12.0, gain_edge_ms=50.0,
                strength=1.0)


def read_wav(path):
    with wave.open(str(path), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f'{path}: expected mono 16-bit PCM')
        rate, frames = stream.getframerate(), stream.readframes(stream.getnframes())
    samples = np.frombuffer(frames, dtype='<i2').astype(np.float64) / 32768.0
    if not samples.size:
        raise ValueError(f'{path}: empty recording')
    return samples, rate


def write_wav(path, samples, rate):
    """Write mono 16-bit PCM and return the number of samples that had to be clipped."""
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 32768.0)
    clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(np.clip(scaled, -32768, 32767).astype('<i2').tobytes())
    return clipped


def flatten_f0(f0, strength=1.0):
    """Compress the voiced log-F0 contour toward its median; unvoiced frames stay at zero."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    out = f0.copy()
    if voiced.any():
        centre = np.log(np.median(f0[voiced]))
        out[voiced] = np.exp(centre + (1.0 - strength) * (np.log(f0[voiced]) - centre))
    return out


def _hann(rate, milliseconds):
    width = max(3, int(round(rate * milliseconds / 1000.0)) | 1)
    window = np.hanning(width + 2)[1:-1]
    return window / window.sum()


def _smooth(values, window):
    # Normalising by the smoothed support keeps the edges unbiased.
    weight = fftconvolve(np.ones_like(values), window, mode='same')
    return fftconvolve(values, window, mode='same') / weight


def slow_power(samples, rate, window_ms):
    # FFT convolution can leave tiny negative values where the signal is silent.
    return np.maximum(_smooth(np.square(np.asarray(samples, dtype=np.float64)), _hann(rate, window_ms)), 0.0)


def speech_mask(power, floor_db):
    """Samples whose slow power is within floor_db of loud speech (95th percentile)."""
    reference = np.percentile(power, 95)
    if reference <= 0:
        return np.zeros(power.shape, dtype=bool)
    return power > reference * 10.0 ** (-floor_db / 10.0)


def energy_gain(samples, rate, *, window_ms, floor_db, max_gain_db, edge_ms, strength=1.0):
    """Slowly varying gain that pulls the speech energy contour to its median level.

    Pauses and low-level noise (below floor_db) keep unit gain, so silence is not pumped up.
    """
    power = slow_power(samples, rate, window_ms)
    active = speech_mask(power, floor_db)
    gain_db = np.zeros(power.shape)
    if active.any():
        level = 10.0 * np.log10(power[active])
        gain_db[active] = np.clip(strength * (np.median(level) - level), -max_gain_db, max_gain_db)
        gain_db = _smooth(gain_db, _hann(rate, edge_ms))
    return 10.0 ** (gain_db / 20.0)


def fit_length(samples, length):
    samples = np.asarray(samples, dtype=np.float64)[:length]
    return np.pad(samples, (0, length - samples.size))


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples))))


def match_level(samples, reference):
    """Trim or pad to the reference length and scale to the reference RMS."""
    out = fit_length(samples, reference.size)
    level = rms(out)
    return out * (rms(reference) / level) if level > 0 else out


def common_peak_scale(waves, limit=0.99):
    """One gain for a set of waveforms that keeps the loudest of them below the clipping limit."""
    peak = max(float(np.max(np.abs(wave_))) for wave_ in waves)
    return min(1.0, limit / peak) if peak > 0 else 1.0


class Praat:
    """Praat pitch tier + overlap-add resynthesis (praat-parselmouth)."""
    name = 'praat'

    def __init__(self, frame_period_ms, f0_floor_hz, f0_ceil_hz):
        import parselmouth
        from parselmouth.praat import call
        self.pm, self.call = parselmouth, call
        self.version = f'{parselmouth.__version__} (Praat {parselmouth.PRAAT_VERSION})'
        self.step, self.floor, self.ceil = float(frame_period_ms) / 1000.0, float(f0_floor_hz), float(f0_ceil_hz)

    def _sound(self, samples, rate):
        return self.pm.Sound(np.ascontiguousarray(samples, dtype=np.float64), sampling_frequency=float(rate))

    def _manipulation(self, sound):
        return self.call(sound, 'To Manipulation', self.step, self.floor, self.ceil)

    def f0(self, samples, rate):
        pitch = self._sound(samples, rate).to_pitch(time_step=self.step, pitch_floor=self.floor, pitch_ceiling=self.ceil)
        return np.asarray(pitch.selected_array['frequency'], dtype=np.float64)

    def analyze(self, samples, rate):
        """F0 at the pitch-tier points (voiced only), plus what resynthesis needs."""
        sound = self._sound(samples, rate)
        tier = self.call(self._manipulation(sound), 'Extract pitch tier')
        count = int(self.call(tier, 'Get number of points'))
        times = [self.call(tier, 'Get time from index', i) for i in range(1, count + 1)]
        values = [self.call(tier, 'Get value at index', i) for i in range(1, count + 1)]
        return np.asarray(values, dtype=np.float64), (sound, times)

    def synthesize(self, f0, state, rate):
        sound, times = state
        manipulation = self._manipulation(sound)
        tier = self.call(manipulation, 'Extract pitch tier')
        if times:
            self.call(tier, 'Remove points between', sound.xmin - 1.0, sound.xmax + 1.0)
            for time, value in zip(times, f0, strict=True):
                self.call(tier, 'Add point', float(time), float(value))
            self.call([tier, manipulation], 'Replace pitch tier')
        return np.asarray(self.call(manipulation, 'Get resynthesis (overlap-add)').values[0], dtype=np.float64)


class World:
    """WORLD vocoder (pyworld): DIO + StoneMask F0, CheapTrick envelope, D4C aperiodicity."""
    name = 'world'

    def __init__(self, frame_period_ms, f0_floor_hz, f0_ceil_hz):
        import pyworld
        self.pw = pyworld
        self.version = getattr(pyworld, '__version__', 'unknown')
        self.period, self.floor, self.ceil = float(frame_period_ms), float(f0_floor_hz), float(f0_ceil_hz)

    def _f0(self, samples, rate):
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        raw, times = self.pw.dio(samples, rate, f0_floor=self.floor, f0_ceil=self.ceil, frame_period=self.period)
        return samples, self.pw.stonemask(samples, raw, times, rate), times

    def f0(self, samples, rate):
        return self._f0(samples, rate)[1]

    def analyze(self, samples, rate):
        samples, f0, times = self._f0(samples, rate)
        size = self.pw.get_cheaptrick_fft_size(rate, self.floor)
        envelope = self.pw.cheaptrick(samples, f0, times, rate, f0_floor=self.floor, fft_size=size)
        aperiodicity = self.pw.d4c(samples, f0, times, rate, fft_size=size)
        return f0, (envelope, aperiodicity)

    def synthesize(self, f0, state, rate):
        envelope, aperiodicity = state
        return self.pw.synthesize(np.ascontiguousarray(f0, dtype=np.float64), envelope, aperiodicity, rate, self.period)


BACKENDS = {'praat': Praat, 'world': World}


def make_backend(params):
    if params['backend'] not in BACKENDS:
        raise ValueError(f"Unknown resynthesis backend {params['backend']!r}")
    return BACKENDS[params['backend']](params['frame_period_ms'], params['f0_floor_hz'], params['f0_ceil_hz'])


def make_twins(samples, rate, backend, params, conditions=AUDIO_CONDITIONS):
    """Return {condition: waveform} for the requested twins and the one gain applied to all of them."""
    unknown = set(conditions) - set(AUDIO_CONDITIONS)
    if unknown:
        raise ValueError(f'Unknown twin conditions: {sorted(unknown)}')
    f0, state = backend.analyze(samples, rate)
    energy = dict(window_ms=params['energy_window_ms'], floor_db=params['energy_floor_db'],
                  max_gain_db=params['max_gain_db'], edge_ms=params['gain_edge_ms'], strength=params['strength'])
    resynth = {}

    def carrier(flat):
        if flat not in resynth:
            contour = flatten_f0(f0, params['strength']) if flat else f0
            resynth[flat] = fit_length(backend.synthesize(contour, state, rate), samples.size)
        return resynth[flat]

    out = {}
    for name in conditions:
        wave_ = carrier(name in ('flat_f0', 'flat_both'))
        if name in ('flat_energy', 'flat_both'):
            wave_ = wave_ * energy_gain(wave_, rate, **energy)
        out[name] = match_level(wave_, samples)
    scale = common_peak_scale(out.values())
    return {name: wave_ * scale for name, wave_ in out.items()}, scale


def voiced_frames(samples, rate, backend):
    return backend.f0(samples, rate) > 0


def contour_stats(samples, rate, backend, params, active=None, voiced_in=None):
    """How much F0 and slow-energy movement a waveform has (the quantities the twins edit).

    active: samples over which the energy contour is measured.
    voiced_in: frames over which the F0 contour is measured.
    Pass the original recording's speech mask and voiced frames when measuring a twin, so both
    are measured over the same stretch of speech. A twin's own masks grow once quiet passages
    have been raised: background noise and cross-talk become loud enough to be tracked as pitch.
    """
    f0 = backend.f0(samples, rate)
    voiced = f0 > 0
    if voiced_in is not None:
        n = min(voiced.size, voiced_in.size)
        f0, voiced = f0[:n], voiced[:n] & voiced_in[:n]
    semitones = 12.0 * np.log2(f0[voiced] / np.median(f0[voiced])) if voiced.any() else np.zeros(0)
    power = slow_power(samples, rate, params['energy_window_ms'])
    if active is None:
        active = speech_mask(power, params['energy_floor_db'])
    active = active & (power > 0)
    level = 10.0 * np.log10(power[active]) if active.any() else np.zeros(0)
    return dict(voiced_fraction=float(voiced.mean()), f0_median_hz=float(np.median(f0[voiced])) if voiced.any() else None,
                f0_sd_semitones=float(semitones.std()) if semitones.size else None,
                energy_sd_db=float(level.std()) if level.size else None,
                speech_fraction=float(active.mean()), rms=rms(samples))


def synthetic_utterance(rate=16000, seconds=2.0, harmonics=8, f0=(120.0, 220.0), amplitude=(0.05, 0.4)):
    """A voiced glide with rising pitch and loudness: both contours have known, large movement."""
    n = int(rate * seconds)
    track = np.linspace(f0[0], f0[1], n)
    phase = 2.0 * np.pi * np.cumsum(track) / rate
    tone = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
    return tone / np.max(np.abs(tone)) * np.linspace(amplitude[0], amplitude[1], n)


def self_check(backend, params, *, harmonics=8, rate=16000):
    """Run every twin on a synthetic glide and test that each edit did what it claims.

    This only shows that the backend is installed and wired correctly. Whether resynthesis
    damages real speech is decided by the sham-versus-original word error rate, not here.
    """
    source = synthetic_utterance(rate, harmonics=harmonics)
    twins, scale = make_twins(source, rate, backend, params)
    mask = speech_mask(slow_power(source, rate, params['energy_window_ms']), params['energy_floor_db'])
    voiced = voiced_frames(source, rate, backend)
    stats = {name: contour_stats(wave_, rate, backend, params, mask, voiced)
             for name, wave_ in {'orig': source, **twins}.items()}
    base = stats['orig']
    checks = {
        'source_has_pitch_movement': base['f0_sd_semitones'] > 2.0,
        'source_has_energy_movement': base['energy_sd_db'] > 3.0,
        'lengths_equal': all(wave_.size == source.size for wave_ in twins.values()),
        'rms_matched': all(abs(s['rms'] / (scale * base['rms']) - 1.0) < 0.02 for n, s in stats.items() if n != 'orig'),
        'sham_keeps_pitch': abs(stats['sham']['f0_sd_semitones'] / base['f0_sd_semitones'] - 1.0) < 0.2,
        'sham_keeps_energy': abs(stats['sham']['energy_sd_db'] / base['energy_sd_db'] - 1.0) < 0.2,
        'flat_f0_removes_pitch': stats['flat_f0']['f0_sd_semitones'] < 0.15 * base['f0_sd_semitones'],
        'flat_f0_keeps_energy': abs(stats['flat_f0']['energy_sd_db'] / base['energy_sd_db'] - 1.0) < 0.2,
        'flat_energy_removes_energy': stats['flat_energy']['energy_sd_db'] < 0.3 * base['energy_sd_db'],
        'flat_energy_keeps_pitch': abs(stats['flat_energy']['f0_sd_semitones'] / base['f0_sd_semitones'] - 1.0) < 0.2,
        'flat_both_removes_pitch': stats['flat_both']['f0_sd_semitones'] < 0.15 * base['f0_sd_semitones'],
        'flat_both_removes_energy': stats['flat_both']['energy_sd_db'] < 0.3 * base['energy_sd_db'],
    }
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()}, stats=stats,
                backend=backend.name, backend_version=getattr(backend, 'version', None), params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the resynthesis backend on a synthetic glide, or flatten one file.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--condition', choices=AUDIO_CONDITIONS, default='flat_both')
    parser.add_argument('--backend', choices=sorted(BACKENDS), default=DEFAULTS['backend'])
    args = parser.parse_args()
    settings = dict(DEFAULTS, backend=args.backend)
    vocoder = make_backend(settings)
    if args.self_check:
        result = self_check(vocoder, settings)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result['passed'] else 1)
    if not args.input or not args.output:
        parser.error('Give --self-check, or --input and --output')
    audio, sample_rate = read_wav(args.input)
    produced, _ = make_twins(audio, sample_rate, vocoder, settings, [args.condition])
    print(json.dumps(dict(clipped_samples=write_wav(args.output, produced[args.condition], sample_rate))))
