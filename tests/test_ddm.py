from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking.config import AnalysisConfig
from fast_partial_tracking.ddm import DdmEstimator
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.model import angular_frequency, log_amplitude
from fast_partial_tracking.tracker import reference_level_db

from .conftest import MatlabCase


def _setup(case: MatlabCase) -> tuple[FrameGrid, DdmEstimator, np.ndarray, float]:
    grid = FrameGrid.from_config(case.fs, case.analysis).with_signal_length(case.x.size)
    estimator = DdmEstimator(grid, case.analysis)
    padded = np.concatenate([np.zeros(grid.pad_front), case.x, np.zeros(grid.pad_back)])
    threshold = reference_level_db(case.x, grid, estimator.window) + case.analysis.threshold_db
    return grid, estimator, padded, threshold


def test_windows_match_matlab(matlab_case: MatlabCase) -> None:
    _, estimator, _, _ = _setup(matlab_case)
    np.testing.assert_allclose(estimator.window, matlab_case["win"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(estimator.window_derivative, matlab_case["winD"], rtol=0, atol=1e-15)


def test_threshold_matches_matlab(matlab_case: MatlabCase) -> None:
    _, _, _, threshold = _setup(matlab_case)
    assert threshold == pytest.approx(float(matlab_case["threshold"]), abs=1e-9)


def test_frames_match_matlab(matlab_case: MatlabCase) -> None:
    grid, estimator, padded, threshold = _setup(matlab_case)
    for ref in matlab_case.ddm_frames():
        k = int(ref["frame"]) - 1
        start = k * grid.hop
        est = estimator(padded[start : start + grid.window_length], threshold)

        ref_s = ref["S"][: est.spectrum.size]
        np.testing.assert_allclose(est.spectrum, ref_s, rtol=0, atol=1e-9 * np.abs(ref_s).max())
        np.testing.assert_array_equal(est.peaks.bin + 1, np.atleast_1d(ref["peak_bin"]))
        np.testing.assert_array_equal(est.peaks.left + 1, np.atleast_1d(ref["left_bin"]))
        np.testing.assert_array_equal(est.peaks.right + 1, np.atleast_1d(ref["right_bin"]))

        assert len(est) == int(ref["num_peaks"])
        alpha_ref = np.asarray(ref["alpha"]).reshape(estimator.order + 1, -1).T
        scale = np.array([1.0] + [float(grid.window_length) ** i for i in range(1, estimator.order + 1)])
        np.testing.assert_allclose(est.alpha * scale, alpha_ref * scale, rtol=1e-7, atol=1e-7)


def test_recovers_stationary_sinusoid() -> None:
    fs = 16000.0
    analysis = AnalysisConfig(window_length=1023, n_fft=2048, poly_order=2)
    grid = FrameGrid.from_config(fs, analysis)
    estimator = DdmEstimator(grid, analysis)
    n = grid.offsets
    freq, amp, phi = 1234.5, 0.3, 0.7
    frame = amp * np.cos(2 * np.pi * freq / fs * n + phi)
    est = estimator(frame, threshold_db=0.0)

    assert len(est) == 1
    assert angular_frequency(est.alpha, 0.0)[0] * fs / (2 * np.pi) == pytest.approx(freq, abs=0.05)
    assert 2 * np.exp(log_amplitude(est.alpha, 0.0)[0]) == pytest.approx(amp, rel=1e-3)
    assert np.angle(np.exp(1j * est.alpha[0, 0].imag)) == pytest.approx(phi, abs=1e-3)


def test_silent_frame_has_no_peaks() -> None:
    analysis = AnalysisConfig(window_length=511)
    grid = FrameGrid.from_config(16000.0, analysis)
    est = DdmEstimator(grid, analysis)(np.zeros(511), threshold_db=-100.0)
    assert len(est) == 0
    assert est.alpha.shape == (0, 3)
