from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking import AnalysisConfig, PartialTracker, Partials, TrackingConfig, track
from fast_partial_tracking.config import matlab_round, nextpow2
from fast_partial_tracking.tracker import reference_level_db

from .conftest import MatlabCase, harmonic_signal


@pytest.mark.parametrize("block", [1, 37, 256, 4096, 100000])
def test_streaming_equals_batch(demo_case: MatlabCase, block: int) -> None:
    batch = track(demo_case.x, demo_case.fs, demo_case.analysis, demo_case.tracking)
    tracker = PartialTracker(
        demo_case.fs, demo_case.analysis, demo_case.tracking, threshold_db=batch.metadata["threshold_db"]
    )
    frames = []
    for start in range(0, demo_case.x.size, block):
        frames += tracker.push(demo_case.x[start : start + block])
    frames += tracker.flush()
    streamed = Partials.from_frames(frames, tracker.grid, tracker.num_tracks)

    assert [f.index for f in frames] == list(range(batch.num_frames))
    np.testing.assert_array_equal(streamed.frame, batch.frame)
    np.testing.assert_array_equal(streamed.track_id, batch.track_id)
    np.testing.assert_array_equal(streamed.omega, batch.omega)
    np.testing.assert_array_equal(streamed.log_amp, batch.log_amp)


def test_frames_are_emitted_with_one_frame_latency() -> None:
    fs = 16000.0
    tracker = PartialTracker(fs, AnalysisConfig(window_length=511))
    x = harmonic_signal(fs)
    hop = tracker.grid.hop
    emitted = tracker.push(x[: 10 * hop])
    assert emitted, "frames must be emitted before the stream ends"
    analysed = (tracker.grid.pad_front + 10 * hop - tracker.grid.window_length) // hop + 1
    assert emitted[-1].index == analysed - 2


def test_push_after_flush_fails() -> None:
    tracker = PartialTracker(16000.0, AnalysisConfig(window_length=255))
    tracker.flush()
    with pytest.raises(RuntimeError):
        tracker.push(np.zeros(10))


def test_silence_produces_no_partials() -> None:
    partials = track(np.zeros(8000), 16000.0, AnalysisConfig(window_length=511))
    assert len(partials) == 0
    assert partials.num_frames > 0


def test_tracks_harmonics() -> None:
    fs = 16000.0
    partials = track(harmonic_signal(fs, f0=220.0, harmonics=4), fs, AnalysisConfig(window_length=1023))
    longest = np.bincount(partials.track_id).argsort()[::-1][:4]
    freqs = sorted(np.median(partials.frequency_hz[partials.track_id == t]) for t in longest)
    np.testing.assert_allclose(freqs, [220, 440, 660, 880], atol=1.0)


def test_multichannel_uses_first_channel() -> None:
    fs = 16000.0
    x = harmonic_signal(fs)
    stereo = np.stack([x, np.zeros_like(x)], axis=1)
    a = track(stereo, fs, AnalysisConfig(window_length=511))
    b = track(x, fs, AnalysisConfig(window_length=511))
    np.testing.assert_array_equal(a.omega, b.omega)


def test_reference_level_of_silence() -> None:
    tracker = PartialTracker(16000.0)
    assert reference_level_db(np.zeros(100), tracker.grid, tracker.estimator.window) == -np.inf


def test_config_validation_and_sanitizing() -> None:
    with pytest.raises(ValueError):
        TrackingConfig(delta=1.5)
    with pytest.raises(ValueError):
        AnalysisConfig(order=0)
    config = AnalysisConfig(window_length=2.5, hop_factor=3, oversample=3)
    assert config.effective_window_length == 4
    assert config.effective_oversample == 4
    assert config.effective_hop_factor == 8
    assert matlab_round(2.5) == 3 and matlab_round(-2.5) == -3
    assert nextpow2(1) == 0 and nextpow2(1025) == 11
