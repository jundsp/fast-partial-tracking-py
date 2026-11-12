"""Spectral peak picking (``jun_pick_peaks.m``), vectorized."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Peaks:
    """Local maxima of a magnitude spectrum and the valleys that bound them.

    Attributes:
        bin: 0-based DFT bin of each peak, ascending.
        left: Bin of the valley to the left of each peak.
        right: Bin of the valley to the right of each peak.
    """

    bin: np.ndarray
    left: np.ndarray
    right: np.ndarray

    def __len__(self) -> int:
        return int(self.bin.size)


def pick_peaks(spectrum: np.ndarray, threshold_db: float) -> Peaks:
    """Finds strict local maxima above a threshold and their surrounding valleys.

    A bin n (excluding the first and last) is a peak if its level exceeds the threshold
    and both neighbours. Each peak's left valley is found by walking left while the level
    keeps strictly decreasing, and likewise to the right; this equals the
    while-loops of the MATLAB version but runs in O(K) with no Python loop.

    Args:
        spectrum: Complex or magnitude spectrum of the positive-frequency bins,
            shape (K,).
        threshold_db: Absolute level in dB (20*log10|X|) that a peak must exceed.

    Returns:
        The detected peaks.
    """
    with np.errstate(divide="ignore"):
        level = 20.0 * np.log10(np.abs(spectrum))
    size = level.size
    if size < 3:
        empty = np.zeros(0, dtype=np.intp)
        return Peaks(empty, empty, empty)

    inner = level[1:-1]
    is_peak = (inner > threshold_db) & (inner > level[:-2]) & (inner > level[2:])
    peak = np.flatnonzero(is_peak) + 1

    index = np.arange(size)
    # rising[i]: level[i] > level[i-1]. The left walk from i stops at the last
    # non-rising index at or before i (index 0 always stops the walk).
    rising = np.zeros(size, dtype=bool)
    rising[1:] = level[1:] > level[:-1]
    left_stop = np.maximum.accumulate(np.where(rising, 0, index))
    # falling[i]: level[i] > level[i+1]. The right walk from i stops at the first
    # non-falling index at or after i (the last index always stops the walk).
    falling = np.zeros(size, dtype=bool)
    falling[:-1] = level[:-1] > level[1:]
    right_stop = np.minimum.accumulate(np.where(falling, size - 1, index)[::-1])[::-1]

    return Peaks(bin=peak, left=left_stop[peak - 1], right=right_stop[peak + 1])
