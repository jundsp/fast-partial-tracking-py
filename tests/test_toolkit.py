from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking import AnalysisConfig, Partials, rsnr, synthesize, track, transforms

from .conftest import harmonic_signal

FS = 16000.0
CONFIG = AnalysisConfig(window_length=1023, threshold_db=-60)


@pytest.fixture(scope="module")
def partials() -> Partials:
    rng = np.random.default_rng(0)
    x = harmonic_signal(FS, duration=1.0, f0=220.0, harmonics=5) + 0.003 * rng.standard_normal(int(FS))
    return track(x, FS, CONFIG)


def tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(int(FS * seconds)) / FS)


def peak_frequency(y: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    return float(np.argmax(spectrum) * FS / y.size)


def test_repr_and_stats(partials: Partials) -> None:
    assert f"{partials.num_tracks} partials" in repr(partials)
    stats = partials.stats()
    assert stats["partials"] == partials.num_tracks and stats["rows"] == len(partials)
    assert 0 < stats["min_frequency_hz"] < stats["max_frequency_hz"] < FS / 2


def test_tracks_cover_all_rows(partials: Partials) -> None:
    tracks = list(partials.tracks())
    assert len(tracks) == partials.num_tracks
    assert sum(len(t) for t in tracks) == len(partials)
    for t in tracks[:20]:
        assert np.all(np.diff(t.frame) == 1), "a partial is an unbroken run of frames"
        assert t.duration_s == pytest.approx(t.time_s[-1] - t.time_s[0])
    longest = max(tracks, key=len)
    assert partials.track(longest.id).frame.tolist() == longest.frame.tolist()
    assert "Track(id=" in repr(longest)
    with pytest.raises(KeyError):
        partials.track(10**6)


def test_harmonics_are_the_long_partials(partials: Partials) -> None:
    pruned = partials.prune(min_frames=20)
    assert pruned.num_tracks <= partials.num_tracks
    freqs = sorted(t.mean_frequency_hz for t in pruned.tracks())
    np.testing.assert_allclose(freqs[:5], [220, 440, 660, 880, 1100], atol=2.0)


def test_prune_relabels_and_keeps_order(partials: Partials) -> None:
    pruned = partials.prune(min_frames=5)
    assert pruned.track_id.max() == pruned.num_tracks - 1
    assert np.all(np.diff(pruned.frame) >= 0)
    assert np.all(pruned.track_lengths() >= 5)


def test_select(partials: Partials) -> None:
    part = partials.select(time_s=(0.2, 0.6), frequency_hz=(300, 1000), min_amplitude_db=-40)
    assert len(part) > 0
    assert part.time_s.min() >= 0.2 and part.time_s.max() <= 0.6
    assert part.frequency_hz.min() >= 300 and part.frequency_hz.max() <= 1000
    assert part.amplitude_db.min() >= -40
    only = partials.select(track_ids=[0])
    assert only.num_tracks == 1


def test_filter_tracks(partials: Partials) -> None:
    low = partials.filter_tracks(lambda t: t.mean_frequency_hz < 500)
    assert all(t.mean_frequency_hz < 500 for t in low.tracks())


def test_dataframes(partials: Partials) -> None:
    pl = pytest.importorskip("polars")
    df = partials.to_polars()
    assert df.height == len(partials)
    per_track = df.group_by("track_id").agg(pl.len())
    assert per_track.height == partials.num_tracks
    assert partials.to_pandas().shape[0] == len(partials)


def test_measured_synthesis_reproduces_a_stationary_tone() -> None:
    x = tone(440.0)
    y = synthesize(track(x, FS, CONFIG), phase="measured")
    inner = slice(int(0.1 * FS), int(0.9 * FS))
    assert rsnr(x[inner], y[inner]) > 60.0


def test_integrated_synthesis_follows_frequency_without_drift() -> None:
    """Integrated phase keeps a constant offset (from the onset frames) but never drifts."""
    from scipy.signal import hilbert

    x = tone(440.0)
    y = synthesize(track(x, FS, CONFIG), phase="integrated")
    inner = slice(int(0.1 * FS), int(0.9 * FS))
    phase_error = np.unwrap(np.angle(hilbert(y))) - np.unwrap(np.angle(hilbert(x)))
    assert np.ptp(phase_error[inner]) < 0.01
    assert abs(np.median(phase_error[inner])) < 0.2
    assert np.sqrt(np.mean(y[inner] ** 2)) == pytest.approx(np.sqrt(np.mean(x[inner] ** 2)), rel=0.01)


def test_pitch_shift() -> None:
    p = track(tone(440.0), FS, CONFIG)
    up = transforms.pitch_shift(p, semitones=12)
    assert up.metadata["synthesis_phase"] == "integrated"
    assert peak_frequency(synthesize(up)) == pytest.approx(880.0, abs=2.0)
    np.testing.assert_allclose(up.alpha[:, 1].imag, 2 * p.alpha[:, 1].imag)


def test_pitch_shift_drops_partials_above_nyquist() -> None:
    p = track(tone(5000.0), FS, CONFIG)
    assert len(transforms.pitch_shift(p, ratio=2.0)) == 0


def test_time_stretch_keeps_pitch() -> None:
    x = tone(440.0)
    p = track(x, FS, CONFIG)
    slow = transforms.time_stretch(p, 1.5)
    y = synthesize(slow)
    assert slow.metadata["time_stretch"] == pytest.approx(1.5, rel=0.01)
    assert y.size == pytest.approx(1.5 * x.size, rel=0.01)
    assert peak_frequency(y) == pytest.approx(440.0, abs=2.0)
    inner = y[int(0.2 * y.size) : int(0.8 * y.size)]
    assert np.sqrt(np.mean(inner**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)


def test_gain() -> None:
    p = track(tone(440.0), FS, CONFIG)
    quieter = transforms.gain(p, -6.0)
    np.testing.assert_allclose(quieter.amplitude_db, p.amplitude_db - 6.0)
    assert "synthesis_phase" not in quieter.metadata


@pytest.mark.parametrize("bad", [dict(ratio=0.0), dict(ratio=-1.0)])
def test_pitch_shift_validation(bad: dict) -> None:
    with pytest.raises(ValueError):
        transforms.pitch_shift(track(tone(440.0, 0.2), FS, CONFIG), **bad)
