from __future__ import annotations

import numpy as np

from fast_partial_tracking.peaks import pick_peaks


def _reference(level: np.ndarray, threshold: float) -> tuple[list[int], list[int], list[int]]:
    """Direct transcription of the jun_pick_peaks.m loops (0-based)."""
    peaks, lefts, rights = [], [], []
    size = level.size
    for n in range(1, size - 1):
        if level[n] > threshold and level[n] > level[n - 1] and level[n] > level[n + 1]:
            left = n - 1
            while left > 0 and level[left] > level[left - 1]:
                left -= 1
            right = n + 1
            while right + 1 < size and level[right] > level[right + 1]:
                right += 1
            peaks.append(n)
            lefts.append(left)
            rights.append(right)
    return peaks, lefts, rights


def test_matches_loop_reference_on_random_spectra() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        spectrum = rng.standard_normal(rng.integers(3, 80)) + 1j * rng.standard_normal(1)
        spectrum[rng.random(spectrum.size) < 0.1] = 0.0
        with np.errstate(divide="ignore"):
            level = 20 * np.log10(np.abs(spectrum))
        threshold = rng.uniform(-20, 5)
        result = pick_peaks(spectrum, threshold)
        peaks, lefts, rights = _reference(level, threshold)
        np.testing.assert_array_equal(result.bin, peaks)
        np.testing.assert_array_equal(result.left, lefts)
        np.testing.assert_array_equal(result.right, rights)


def test_plateaus_and_threshold() -> None:
    spectrum = np.array([1.0, 3.0, 3.0, 1.0, 5.0, 2.0, 0.5, 4.0, 1.0])
    result = pick_peaks(spectrum, threshold_db=20 * np.log10(3.5))
    np.testing.assert_array_equal(result.bin, [4, 7])
    np.testing.assert_array_equal(result.left, [3, 6])
    np.testing.assert_array_equal(result.right, [6, 8])


def test_too_short() -> None:
    assert len(pick_peaks(np.ones(2), -100.0)) == 0
