"""Choosing whole partials: per-partial summaries, rule-based and harmonic selection.

These functions return partial labels (``track_id`` values), which can be passed to
``Partials.select(track_ids=...)`` or combined as sets.

Example:
    >>> ids = fpt.selection.harmonics(partials, f0_hz=220, tolerance_cents=40)
    >>> voice = partials.select(track_ids=ids)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fast_partial_tracking.partials import Partials


@dataclass(frozen=True)
class TrackSummary:
    """One value per partial label 0..num_tracks-1 (labels without points have length 0).

    Attributes:
        id: Partial labels.
        frames: Number of frames each partial spans.
        start_s: Time of the first point.
        end_s: Time of the last point.
        mean_frequency_hz: Amplitude-weighted mean frequency.
        peak_db: Highest level (dB re a full-scale sinusoid).
    """

    id: np.ndarray
    frames: np.ndarray
    start_s: np.ndarray
    end_s: np.ndarray
    mean_frequency_hz: np.ndarray
    peak_db: np.ndarray

    def __len__(self) -> int:
        return int(self.id.size)


def summarize(partials: Partials) -> TrackSummary:
    """Per-partial summary numbers, computed without a Python loop over partials."""
    n = partials.num_tracks
    ids = partials.track_id
    frames = np.bincount(ids, minlength=n)
    time = partials.time_s
    start = np.full(n, np.inf)
    end = np.full(n, -np.inf)
    peak = np.full(n, -np.inf)
    np.minimum.at(start, ids, time)
    np.maximum.at(end, ids, time)
    np.maximum.at(peak, ids, partials.amplitude_db)
    weight = partials.amplitude
    total = np.bincount(ids, weights=weight, minlength=n)
    weighted = np.bincount(ids, weights=weight * partials.frequency_hz, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_frequency = np.where(total > 0, weighted / total, np.nan)
    empty = frames == 0
    start[empty] = end[empty] = peak[empty] = np.nan
    return TrackSummary(np.arange(n), frames, start, end, mean_frequency, peak)


def by_rule(
    partials: Partials,
    time_s: tuple[float, float] | None = None,
    frequency_hz: tuple[float, float] | None = None,
    min_level_db: float | None = None,
    min_frames: int | None = None,
    summary: TrackSummary | None = None,
) -> np.ndarray:
    """Labels of the partials that satisfy every given rule.

    Args:
        partials: The result to choose from.
        time_s: Keep partials that overlap this (start, end) time range.
        frequency_hz: Keep partials whose mean frequency lies in this (low, high) range.
        min_level_db: Keep partials whose peak level is at least this.
        min_frames: Keep partials spanning at least this many frames.
        summary: A precomputed ``summarize(partials)``, to avoid recomputing it.

    Returns:
        Sorted partial labels.
    """
    s = summary or summarize(partials)
    keep = s.frames > 0
    if time_s is not None:
        keep &= (s.start_s <= time_s[1]) & (s.end_s >= time_s[0])
    if frequency_hz is not None:
        keep &= (s.mean_frequency_hz >= frequency_hz[0]) & (s.mean_frequency_hz <= frequency_hz[1])
    if min_level_db is not None:
        keep &= s.peak_db >= min_level_db
    if min_frames is not None:
        keep &= s.frames >= min_frames
    return s.id[keep]


def harmonics(
    partials: Partials,
    f0_hz: float,
    tolerance_cents: float = 50.0,
    max_harmonic: int | None = None,
    time_s: tuple[float, float] | None = None,
    summary: TrackSummary | None = None,
) -> np.ndarray:
    """Labels of the partials near integer multiples of a fundamental.

    A partial matches harmonic h when its mean frequency is within ``tolerance_cents`` of
    h·f0 (h = 1, 2, ...). This suits sounds with a steady pitch; for a varying pitch,
    select one time range at a time.

    Args:
        partials: The result to choose from.
        f0_hz: Fundamental frequency in Hz.
        tolerance_cents: Largest allowed distance from a harmonic, in cents.
        max_harmonic: Highest harmonic number to include (None for all).
        time_s: Only consider partials overlapping this time range.
        summary: A precomputed ``summarize(partials)``.

    Returns:
        Sorted partial labels.
    """
    if f0_hz <= 0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}")
    if tolerance_cents <= 0:
        raise ValueError(f"tolerance_cents must be positive, got {tolerance_cents}")
    s = summary or summarize(partials)
    candidates = by_rule(partials, time_s=time_s, summary=s)
    frequency = s.mean_frequency_hz[candidates]
    number = np.maximum(np.rint(frequency / f0_hz), 1)
    cents = 1200.0 * np.abs(np.log2(frequency / (number * f0_hz)))
    keep = cents <= tolerance_cents
    if max_harmonic is not None:
        keep &= number <= max_harmonic
    return candidates[keep]
