"""Building ``Partials`` from bare points (time, partial index, frequency, amplitude).

Formats from other tools (SDIF ``1TRC``, SPEAR text) store only these values, without the
sampling rate or analysis settings. ``from_points`` infers a frame grid from the point
times: the hop is the typical spacing of distinct times, and every time is placed on the
nearest frame of that grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.partials import Partials

DEFAULT_FS = 44100.0


def from_points(
    time_s: np.ndarray,
    index: np.ndarray,
    frequency_hz: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray | None = None,
    fs: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Partials:
    """Builds partials from points, inferring the frame grid from their times.

    Args:
        time_s: Time of each point in seconds.
        index: Partial index of each point (any integers). A partial whose points skip
            frames is split at each gap, since partials are continuous here.
        frequency_hz: Frequency of each point in Hz.
        amplitude: Linear amplitude of each point's real sinusoid.
        phase: Phase of each point in radians, or None if unknown. Without phases (or if
            they are all zero) the result is marked for phase-integrating synthesis.
        fs: Sampling rate in Hz (not stored by most formats; default 44.1 kHz).
        metadata: Extra metadata to store.

    Returns:
        The partials, relabelled 0..n-1 in order of their first point.
    """
    fs = float(fs or DEFAULT_FS)
    time_s = np.asarray(time_s, dtype=float)
    index = np.asarray(index).astype(np.int64)
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    measured = phase is not None and np.any(np.asarray(phase) != 0)
    phase = np.zeros_like(time_s) if phase is None else np.asarray(phase, dtype=float)
    if not (time_s.shape == index.shape == frequency_hz.shape == amplitude.shape == phase.shape):
        raise ValueError("all point columns must have the same length")
    keep = (amplitude > 0) & (frequency_hz > 0) & (frequency_hz < fs / 2)
    time_s, index, frequency_hz, amplitude, phase = (a[keep] for a in (time_s, index, frequency_hz, amplitude, phase))

    grid, frame = _grid(time_s, fs)
    # Sort by partial then frame, drop repeated (partial, frame) points, split at gaps.
    order = np.lexsort((frame, index))
    frame_sorted, index_sorted = frame[order], index[order]
    duplicate = np.zeros(order.size, dtype=bool)
    duplicate[1:] = (index_sorted[1:] == index_sorted[:-1]) & (frame_sorted[1:] == frame_sorted[:-1])
    order, frame_sorted, index_sorted = order[~duplicate], frame_sorted[~duplicate], index_sorted[~duplicate]
    starts = np.ones(order.size, dtype=bool)
    starts[1:] = (index_sorted[1:] != index_sorted[:-1]) | (frame_sorted[1:] != frame_sorted[:-1] + 1)
    piece = np.cumsum(starts) - 1
    # Number pieces in order of their first frame (then original index), like the tracker does.
    first_frame = frame_sorted[starts]
    rank = np.empty(first_frame.size, dtype=np.int64)
    rank[np.lexsort((index_sorted[starts], first_frame))] = np.arange(first_frame.size)
    track_id = rank[piece]

    by_frame = np.lexsort((track_id, frame_sorted))
    rows = order[by_frame]
    return Partials(
        grid=grid,
        frame=frame_sorted[by_frame],
        track_id=track_id[by_frame],
        omega=2.0 * np.pi * frequency_hz[rows] / fs,
        log_amplitude=np.log(amplitude[rows] / 2.0),
        phase=phase[rows],
        num_tracks=int(first_frame.size),
        metadata={**({} if measured else {"synthesis_phase": "integrated"}), **(metadata or {})},
    )


def _grid(time_s: np.ndarray, fs: float) -> tuple[FrameGrid, np.ndarray]:
    """A centered frame grid that fits the given times, and each time's frame index."""
    unique = np.unique(np.round(time_s * fs).astype(np.int64))
    steps = np.diff(unique)
    hop = int(np.median(steps[steps > 0])) if np.any(steps > 0) else 512
    hop = max(hop, 1)
    first = int(unique[0]) if unique.size else 0
    offset = first - int(np.rint(first / hop)) * hop  # where frame 0 sits, within +-hop/2
    window_length = 4 * hop + 1
    origin = window_length // 2
    frame = np.rint((time_s * fs - offset) / hop).astype(np.int64)
    if frame.size and frame.min() < 0:
        raise ValueError("point times must not be negative")
    last = int(frame.max()) if frame.size else 0
    grid = FrameGrid(
        fs=fs, window_length=window_length, hop=hop, n_dft=int(2 ** np.ceil(np.log2(2 * window_length))),
        origin=origin, pad_front=origin - offset,
    ).with_signal_length(max(last * hop + offset + 1, 1))
    return grid, frame
