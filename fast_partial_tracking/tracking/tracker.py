"""Frame-by-frame partial tracking (Section 3 of the paper).

Each analysis frame goes through:

1. Short-term parameter estimation with the DDM (Section 3.2, ``ddm.py``).
2. Costs of linking every peak of frame k-1 to every peak of frame k (Sections 3.3-3.4,
   ``cost.py``), solved as a linear assignment problem (Section 3.5, ``assignment.py``).
3. Labeling of useful assignments as continuations or births (Section 3.6,
   ``labeling.py``).

Assignment and labeling only need frames k-1 and k, which is what makes the method
real-time capable (Section 3.7). ``PartialTracker`` exposes that: it accepts audio in
blocks of any size and returns frames as soon as they are final.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

from fast_partial_tracking.tracking.assignment import useful_assignments
from fast_partial_tracking.config import AnalysisConfig, TrackingConfig
from fast_partial_tracking.tracking.cost import assignment_costs, cost_variances, midpoint_parameters
from fast_partial_tracking.analysis.ddm import DdmEstimator, FrameEstimate
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.tracking.labeling import label_assignments
from fast_partial_tracking.analysis.model import frame_parameters
from fast_partial_tracking.partials import Frame, Partials

if TYPE_CHECKING:
    from fast_partial_tracking.io.base import PartialsWriter

log = logging.getLogger(__name__)

# Frames analysed per thread-pool batch; bounds the estimates held in memory at once.
_CHUNK_FRAMES = 64


@dataclass(frozen=True)
class _PendingFrame:
    """Rows of the newest frame, which stays open until the next frame is linked.

    Attributes:
        alpha: Model coefficients of each row, shape (rows, Q + 1).
        track_id: Label of each row.
        peak: Index of each linked row's peak within the frame's estimate; partials born
            in this frame are appended later and have no entry here.
    """

    alpha: np.ndarray
    track_id: np.ndarray
    peak: np.ndarray

    @classmethod
    def empty(cls, order: int) -> _PendingFrame:
        return cls(np.zeros((0, order + 1), complex), np.zeros(0, np.int64), np.zeros(0, np.intp))

    def with_births(self, alpha: np.ndarray, track_id: np.ndarray) -> _PendingFrame:
        """Appends partials that start in this frame (found when linking the next one)."""
        return _PendingFrame(
            np.concatenate([self.alpha, alpha]), np.concatenate([self.track_id, track_id]), self.peak
        )


class PartialTracker:
    """Streaming partial tracker.

    Frame k-1 is only final after frame k has been analyzed, because a partial born in
    frame k-1 is discovered when it is first linked to frame k. ``push`` therefore returns
    the frames that just became final. Memory use is bounded by one frame of audio plus
    two frames of estimates, regardless of the input length.

    Example:
        >>> tracker = PartialTracker(fs=44100)
        >>> frames = [f for block in blocks for f in tracker.push(block)]
        >>> frames += tracker.flush()

    Args:
        fs: Sampling rate in Hz.
        analysis: Short-term analysis settings.
        tracking: Assignment cost settings.
        reference_db: Level (dB of DFT magnitude) that ``analysis.threshold_db`` is
            relative to. Needed when ``analysis.threshold_reference == "signal"``, because
            a stream cannot look ahead to its loudest frame (``track`` computes it from the
            whole input); otherwise derived from the configuration.
        keep_alpha: Whether emitted frames carry the full model coefficients.
        workers: Threads used to analyse frames in parallel when a pushed block holds
            several frames. Linking stays sequential, so results do not depend on it.
    """

    def __init__(
        self,
        fs: float,
        analysis: AnalysisConfig | None = None,
        tracking: TrackingConfig | None = None,
        reference_db: float | None = None,
        keep_alpha: bool = True,
        workers: int = 1,
    ) -> None:
        self.analysis = analysis or AnalysisConfig()
        self.tracking = tracking or TrackingConfig()
        self.grid = FrameGrid.from_config(fs, self.analysis)
        self.estimator = DdmEstimator(self.grid, self.analysis)
        self.var_f, self.var_a = cost_variances(self.tracking)
        self.reference_db = self._resolve_reference(reference_db)
        self.threshold_db = self.reference_db + self.analysis.threshold_db
        self.keep_alpha = keep_alpha
        self.n_prev, self.n_curr = self.grid.midpoint_offsets[1], self.grid.midpoint_offsets[0]

        self._buffer = np.zeros(self.grid.pad_front)
        self._num_samples = 0
        self._next_frame = 0
        self._finished = False
        self.num_tracks = 0
        self._prev: FrameEstimate | None = None
        self._pending = _PendingFrame.empty(self.estimator.order)
        self._executor = ThreadPoolExecutor(workers) if workers > 1 else None

    def _resolve_reference(self, reference_db: float | None) -> float:
        """Reference level for the peak threshold, from the argument or the configuration."""
        if reference_db is not None:
            return float(reference_db)
        reference = self.analysis.threshold_reference
        if isinstance(reference, (int, float)):
            return float(reference)
        if reference == "signal":
            log.warning("threshold_reference='signal' needs the whole signal; using full scale for this stream")
        return full_scale_level_db(self.estimator.window)

    def push(self, samples: np.ndarray) -> list[Frame]:
        """Feeds a block of audio and returns the frames that became final.

        Args:
            samples: Mono audio block of any length.

        Returns:
            Finalized frames in increasing order (possibly empty).
        """
        if self._finished:
            raise RuntimeError("tracker already flushed; create a new tracker")
        samples = np.asarray(samples, dtype=float).ravel()
        self._num_samples += samples.size
        self._buffer = np.concatenate([self._buffer, samples])
        return self._process_available()

    def process(self, blocks: Iterable[np.ndarray]) -> Iterator[Frame]:
        """Tracks a whole stream of audio blocks, yielding frames as they become final.

        Example:
            >>> for frame in PartialTracker(fs).process(sf.blocks("in.wav", 4096)):
            ...     writer.write_frame(frame)

        Args:
            blocks: Mono audio blocks (any sizes), e.g. from ``soundfile.blocks``.

        Yields:
            Every frame 0..M-1 in order; the tracker is flushed at the end.
        """
        for block in blocks:
            yield from self.push(block)
        yield from self.flush()

    def flush(self) -> list[Frame]:
        """Ends the stream: zero-pads the tail and returns the remaining frames.

        After flushing, ``grid`` holds the signal length and frame count.

        Returns:
            The remaining frames, ending with the last frame M-1.
        """
        if self._finished:
            return []
        self.grid = self.grid.with_signal_length(self._num_samples)
        self._buffer = np.concatenate([self._buffer, np.zeros(self.grid.pad_back)])
        # The padded tail can hold more than M frames when n_dft > N; MATLAB stops at M.
        frames = self._process_available(limit=self.grid.num_frames)
        frames.append(self._emit(self.grid.num_frames - 1, self._pending))
        self._finished = True
        if self._executor is not None:
            self._executor.shutdown()
        return frames

    def _process_available(self, limit: int | None = None) -> list[Frame]:
        """Analyses every complete frame in the buffer (up to frame ``limit``)."""
        n, hop = self.grid.window_length, self.grid.hop
        count = max(0, (self._buffer.size - n) // hop + 1)
        if limit is not None:
            count = min(count, limit - self._next_frame)
        frames: list[Frame] = []
        for first in range(0, count, _CHUNK_FRAMES):
            starts = range(first * hop, min(first + _CHUNK_FRAMES, count) * hop, hop)
            for estimate in self._estimate([self._buffer[s : s + n] for s in starts]):
                emitted = self._consume(estimate)
                if emitted is not None:
                    frames.append(emitted)
        self._buffer = self._buffer[count * hop :].copy()
        return frames

    def _estimate(self, frames: list[np.ndarray]) -> list[FrameEstimate]:
        """DDM estimates of several frames, in parallel when a thread pool is configured."""
        if self._executor is None or len(frames) < 2:
            return [self.estimator(f, self.threshold_db) for f in frames]
        return list(self._executor.map(lambda f: self.estimator(f, self.threshold_db), frames))

    def _consume(self, estimate: FrameEstimate) -> Frame | None:
        """Links the next frame's estimate and returns the frame before it, now final."""
        k = self._next_frame
        self._next_frame += 1
        prev, self._prev = self._prev, estimate
        if prev is None:
            return None
        return self._advance(k, prev, estimate)

    def _advance(self, k: int, prev: FrameEstimate, estimate: FrameEstimate) -> Frame:
        """Labels the links between frames k-1 and k (Section 3.6) and rolls the state."""
        rows, cols = self._link(prev, estimate)
        labels = label_assignments(self._pending.peak, self._pending.track_id, rows, cols, self.num_tracks)
        # Births are discovered now but start in frame k-1, so they are appended to it.
        finished = self._emit(k - 1, self._pending.with_births(prev.alpha[labels.birth_prev], labels.birth_track))
        self.num_tracks += labels.birth_track.size
        self._pending = _PendingFrame(estimate.alpha[labels.curr_peaks], labels.curr_tracks, labels.curr_peaks)
        return finished

    def _link(self, prev: FrameEstimate, curr: FrameEstimate) -> tuple[np.ndarray, np.ndarray]:
        """Useful assignments between two consecutive frames (eqs. 8-14)."""
        if len(prev) == 0 or len(curr) == 0:
            empty = np.zeros(0, dtype=np.intp)
            return empty, empty
        fs = self.grid.fs
        log_amp_prev, freq_prev = midpoint_parameters(prev.alpha, self.n_prev, fs)
        log_amp_curr, freq_curr = midpoint_parameters(curr.alpha, self.n_curr, fs)
        cost, useful = assignment_costs(
            log_amp_prev, freq_prev, log_amp_curr, freq_curr, self.var_f, self.var_a, self.tracking.delta
        )
        return useful_assignments(cost, useful)

    def _emit(self, index: int, pending: _PendingFrame) -> Frame:
        """Converts an open frame's rows into a final ``Frame`` (eqs. 5-7 at n = 0)."""
        omega, log_amplitude, phase = frame_parameters(pending.alpha)
        return Frame(
            index=index,
            center=int(self.grid.center(index)),
            time_s=float(self.grid.time_seconds(index)),
            fs=self.grid.fs,
            omega=omega,
            log_amplitude=log_amplitude,
            phase=phase,
            track_id=pending.track_id.astype(np.int64),
            alpha=pending.alpha.copy() if self.keep_alpha else None,
        )


def full_scale_level_db(window: np.ndarray) -> float:
    """Peak DFT level (dB) of a sinusoid with amplitude 1 analysed with ``window``."""
    return float(20.0 * np.log10(np.sum(window) / 2.0))


def reference_level_at(x: np.ndarray, loudest: int, grid: FrameGrid, window: np.ndarray) -> float:
    """Spectral peak level (dB) of the frame whose origin is input sample ``loudest``.

    Args:
        x: Input samples (only those within one window of ``loudest`` are read).
        loudest: Index of the loudest sample.
        grid: Frame geometry.
        window: Analysis window.
    """
    start = loudest - grid.origin
    segment = np.zeros(grid.window_length)
    lo, hi = max(start, 0), min(start + grid.window_length, x.size)
    segment[lo - start : hi - start] = x[lo:hi]
    return float(20.0 * np.log10(np.max(np.abs(np.fft.fft(window * segment, grid.n_dft)))))


def reference_level_db(x: np.ndarray, grid: FrameGrid, window: np.ndarray) -> float:
    """Spectral peak level of the frame centered on the loudest sample (MATLAB ``maxS``).

    Used as the reference for the relative peak-picking threshold. Rather than computing
    the whole STFT, only the frame around the largest absolute sample is transformed.

    Args:
        x: Mono input signal.
        grid: Frame geometry.
        window: Analysis window.

    Returns:
        Maximum DFT magnitude of that frame in dB (-inf for an all-zero signal).
    """
    if x.size == 0 or not np.any(x):
        return -np.inf
    return reference_level_at(x, int(np.argmax(np.abs(x))), grid, window)


def track(
    x: np.ndarray,
    fs: float,
    analysis: AnalysisConfig | None = None,
    tracking: TrackingConfig | None = None,
    writer: PartialsWriter | None = None,
    keep_alpha: bool = True,
    workers: int | None = None,
) -> Partials:
    """Tracks the partials of a whole signal.

    With ``threshold_reference="signal"`` (the default) the peak threshold is relative to
    the loudest frame of the input, as in ``jun_track_partials.m``. Use
    ``PartialTracker`` directly for live input.

    Args:
        x: Input audio. For multi-channel input of shape (L, C) the first channel is
            used, like MATLAB.
        fs: Sampling rate in Hz.
        analysis: Short-term analysis settings.
        tracking: Assignment cost settings.
        writer: Optional writer that receives every frame as soon as it is final.
        keep_alpha: Whether to keep the full model coefficients.
        workers: Threads for frame analysis (default: all CPU cores). Results are
            identical for any value.

    Returns:
        All tracked partials.
    """
    analysis = analysis or AnalysisConfig()
    tracking = tracking or TrackingConfig()
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    grid = FrameGrid.from_config(fs, analysis)
    window = DdmEstimator(grid, analysis).window
    reference = reference_level_db(x, grid, window) if analysis.threshold_reference == "signal" else None
    workers = workers or os.cpu_count() or 1
    tracker = PartialTracker(fs, analysis, tracking, reference_db=reference, keep_alpha=keep_alpha, workers=workers)

    frames = _collect(tracker.push(x), writer) + _collect(tracker.flush(), writer)
    metadata = build_metadata(tracker)
    if writer is not None:
        writer.set_summary(tracker.grid, tracker.num_tracks, metadata)
    log.debug(f"tracked {tracker.num_tracks} partials over {tracker.grid.num_frames} frames")
    return Partials.from_frames(frames, tracker.grid, tracker.num_tracks, metadata)


def build_metadata(tracker: PartialTracker) -> dict[str, Any]:
    """Settings and provenance recorded with a tracking result."""
    try:
        package_version = version("fast-partial-tracking")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "analysis": tracker.analysis.to_dict(),
        "tracking": tracker.tracking.to_dict(),
        "reference_db": tracker.reference_db,
        "threshold_db": tracker.threshold_db,
        "package_version": package_version,
    }


def _collect(frames: Iterable[Frame], writer: PartialsWriter | None) -> list[Frame]:
    frames = list(frames)
    if writer is not None:
        for frame in frames:
            writer.write_frame(frame)
    return frames
