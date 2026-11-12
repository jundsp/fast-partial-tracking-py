from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fast_partial_tracking import AnalysisConfig, rsnr, synthesize, track
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.analysis.windows import get_window

from .conftest import CASES, MatlabCase, load_mat


@pytest.mark.parametrize("window_length", [511, 1024, 2047])
@pytest.mark.parametrize("window", ["hann", "blackman_harris"])
def test_centered_frames_sit_on_hop_multiples(window_length: int, window: str) -> None:
    config = AnalysisConfig(window_length=window_length, window=window)
    grid = FrameGrid.from_config(16000.0, config).with_signal_length(12345)
    k = np.arange(grid.num_frames)
    np.testing.assert_array_equal(grid.center(k) - grid.pad_front, k * grid.hop)
    np.testing.assert_allclose(grid.frame_times(), k * grid.hop / 16000.0)
    np.testing.assert_array_equal(grid.frame_index(grid.frame_times()), k)
    assert grid.center(grid.num_frames - 1) - grid.pad_front >= 12345 - 1
    assert grid.pad_back >= 0
    if window_length % 2 or window == "blackman_harris":
        assert grid.origin == int(np.argmax(get_window(window, window_length)[0]))


def test_defaults_resolve_from_window() -> None:
    config = AnalysisConfig(window_length=2047)
    assert (config.resolved_hop_size, config.resolved_n_fft, config.resolved_ddm_bins) == (255, 4096, 4)
    hann = AnalysisConfig(window_length=1023, window="hann")
    assert hann.resolved_hop_size == 255
    assert AnalysisConfig.from_duration(44100.0, window_ms=46.0).window_length == 2029


def test_matlab_preset_reproduces_matlab_settings(matlab_case: MatlabCase) -> None:
    config = matlab_case.analysis
    assert config.geometry == "matlab"
    assert config.window_length == matlab_case["N"]
    assert config.resolved_hop_size == matlab_case["H"]
    assert config.resolved_n_fft == matlab_case["Ndft"]


def test_centered_geometry_is_at_least_as_good_as_matlab() -> None:
    """Quality gate for the clean default geometry against the MATLAB frame placement."""
    diffs = []
    for name in CASES:
        case = MatlabCase(name, load_mat(name))
        scores = []
        for config in [case.analysis, replace(case.analysis, geometry="centered")]:
            partials = track(case.x, case.fs, config, case.tracking)
            scores.append(rsnr(case.x, synthesize(partials)))
        diffs.append(scores[1] - scores[0])
    assert np.mean(diffs) >= -0.1
    assert min(diffs) >= -1.0


def test_center_amplitude_guard_keeps_cubic_models_stable() -> None:
    """Q = 3 with MATLAB's loose guard lets exploding estimates through; the default does not."""
    from fast_partial_tracking import data

    x, fs = data.synthetic("chord", duration=1.0, noise_db=-50)
    scores = {}
    for guard in ("center", "matlab"):
        config = AnalysisConfig(window_length=1023, poly_order=3, amplitude_guard=guard)
        scores[guard] = rsnr(x, synthesize(track(x, fs, config)))
    assert scores["center"] > 15.0
    assert scores["center"] > scores["matlab"] + 5.0
