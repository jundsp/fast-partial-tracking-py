from __future__ import annotations

import numpy as np
import pytest

import fast_partial_tracking as fpt
from fast_partial_tracking.selection import by_rule, harmonics, summarize


@pytest.fixture(scope="module")
def harmonic_partials() -> fpt.Partials:
    x, fs = fpt.data.synthetic("harmonic", duration=0.5, noise_db=-50)
    return fpt.track(x, fs, fpt.AnalysisConfig(window_length=511, threshold_db=-70))


def test_summary_matches_tracks(harmonic_partials: fpt.Partials) -> None:
    summary = summarize(harmonic_partials)
    assert len(summary) == harmonic_partials.num_tracks
    for track in list(harmonic_partials.tracks())[:20]:
        assert summary.frames[track.id] == len(track)
        assert summary.start_s[track.id] == pytest.approx(track.start_s)
        assert summary.end_s[track.id] == pytest.approx(track.end_s)
        assert summary.mean_frequency_hz[track.id] == pytest.approx(track.mean_frequency_hz)
        assert summary.peak_db[track.id] == pytest.approx(track.amplitude_db.max())


def test_harmonics_finds_the_eight_harmonics(harmonic_partials: fpt.Partials) -> None:
    summary = summarize(harmonic_partials)
    long = by_rule(harmonic_partials, min_frames=20, min_level_db=-35, summary=summary)
    ids = np.intersect1d(harmonics(harmonic_partials, 220.0, tolerance_cents=20), long)
    numbers = np.rint(summary.mean_frequency_hz[ids] / 220.0).astype(int)
    assert set(numbers) == set(range(1, 9))
    assert set(np.rint(summary.mean_frequency_hz[harmonics(harmonic_partials, 220.0, 20, max_harmonic=3)] / 220.0)) <= {1, 2, 3}


def test_harmonics_rejects_bad_input(harmonic_partials: fpt.Partials) -> None:
    with pytest.raises(ValueError):
        harmonics(harmonic_partials, 0.0)
    with pytest.raises(ValueError):
        harmonics(harmonic_partials, 220.0, tolerance_cents=0)


def test_by_rule_combines_conditions(harmonic_partials: fpt.Partials) -> None:
    summary = summarize(harmonic_partials)
    ids = by_rule(harmonic_partials, frequency_hz=(400, 700), min_level_db=-30, summary=summary)
    assert ids.size >= 1
    assert np.all((summary.mean_frequency_hz[ids] >= 400) & (summary.mean_frequency_hz[ids] <= 700))
    assert np.all(summary.peak_db[ids] >= -30)
    late = by_rule(harmonic_partials, time_s=(0.45, 0.5), summary=summary)
    assert np.all(summary.end_s[late] >= 0.45)
    assert by_rule(harmonic_partials).size == np.count_nonzero(summary.frames)
