"""Matplotlib helpers for partials (optional dependency: ``pip install fast-partial-tracking[plot]``).

Example:
    >>> import fast_partial_tracking as fpt
    >>> ax = fpt.plot.partials(partials, audio=x)        # partials over the spectrogram
    >>> fig = fpt.plot.overview(partials, x)             # waveform, resynthesis, partials
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fast_partial_tracking.partials import Partials

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ColorBy = Literal["amplitude", "track", "none"]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting needs matplotlib: pip install 'fast-partial-tracking[plot]'") from exc
    return plt


def spectrogram(
    audio: np.ndarray,
    fs: float,
    ax: Axes | None = None,
    n_fft: int | None = None,
    hop: int | None = None,
    floor_db: float = 90.0,
    cmap: str = "gray_r",
) -> Axes:
    """Draws a log-magnitude spectrogram (edges zero-padded so it spans the whole signal).

    Args:
        audio: Mono signal.
        fs: Sampling rate in Hz.
        ax: Axes to draw on (new figure if None).
        n_fft: Window and DFT length (default: about 23 ms, a power of two).
        hop: Hop size (default n_fft // 8).
        floor_db: Dynamic range below the maximum that is shown.
        cmap: Colormap.

    Returns:
        The axes.
    """
    from scipy.signal import stft

    ax = ax or _pyplot().subplots(figsize=(8, 4))[1]
    n_fft = n_fft or int(2 ** np.round(np.log2(0.023 * fs)))
    hop = hop or n_fft // 8
    freqs, times, spectrum = stft(audio, fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    level = 20 * np.log10(np.abs(spectrum) + 1e-12)
    ax.pcolormesh(times, freqs, level, shading="auto", cmap=cmap, vmin=level.max() - floor_db, vmax=level.max(),
                  rasterized=True)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    return ax


def partials(
    partials: Partials,
    ax: Axes | None = None,
    audio: np.ndarray | None = None,
    color_by: ColorBy = "amplitude",
    dynamic_range_db: float = 80.0,
    max_frequency_hz: float | None = None,
    cmap: str = "plasma",
    linewidth: float = 1.0,
    colorbar: bool = True,
) -> Axes:
    """Draws every partial as a line in the time-frequency plane.

    Args:
        partials: Tracking result.
        ax: Axes to draw on (new figure if None).
        audio: If given, its spectrogram is drawn underneath.
        color_by: ``"amplitude"`` (dB), ``"track"`` (one color per partial) or ``"none"``.
        dynamic_range_db: Amplitude range of the color scale below the loudest point.
        max_frequency_hz: Upper frequency limit (default Nyquist).
        cmap: Colormap for the lines.
        linewidth: Line width.
        colorbar: Add a colorbar when coloring by amplitude.

    Returns:
        The axes.
    """
    from matplotlib.collections import LineCollection

    plt = _pyplot()
    fs = partials.grid.fs
    if ax is None:
        ax = plt.subplots(figsize=(8, 4))[1]
    if audio is not None:
        spectrogram(audio, fs, ax=ax)
    segments, values = _segments(partials, color_by)
    collection = LineCollection(segments, linewidths=linewidth, cmap=cmap)
    if color_by == "none":
        collection.set_color("tab:red")
    else:
        collection.set_array(values)
        if color_by == "amplitude" and values.size:
            collection.set_clim(values.max() - dynamic_range_db, values.max())
    ax.add_collection(collection)
    duration = (partials.grid.signal_length or 0) / fs
    ax.set_xlim(0, duration)
    ax.set_ylim(0, max_frequency_hz or fs / 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    if colorbar and color_by == "amplitude" and values.size:
        plt.colorbar(collection, ax=ax, label="amplitude (dB)")
    return ax


_draw_partials = partials


def _segments(partials: Partials, color_by: ColorBy) -> tuple[np.ndarray, np.ndarray]:
    """Line segments between consecutive points of each partial, and their color values."""
    order = np.lexsort((partials.frame, partials.track_id))
    track_id = partials.track_id[order]
    points = np.column_stack([partials.time_s[order], partials.frequency_hz[order]])
    link = track_id[1:] == track_id[:-1]
    segments = np.stack([points[:-1][link], points[1:][link]], axis=1)
    if color_by == "track":
        values = track_id[:-1][link].astype(float)
    else:
        level = partials.amplitude_db[order]
        values = 0.5 * (level[:-1] + level[1:])[link]
    return segments, values


def overview(
    partials: Partials,
    audio: np.ndarray,
    resynthesis: np.ndarray | None = None,
    max_frequency_hz: float | None = None,
    out_path: str | Path | None = None,
) -> Figure:
    """Waveforms (input and resynthesis) above the partials drawn over the spectrogram.

    Args:
        partials: Tracking result.
        audio: Input signal.
        resynthesis: Resynthesized signal (computed if None).
        max_frequency_hz: Upper frequency limit of the partials plot.
        out_path: If given, the figure is saved there.

    Returns:
        The figure.
    """
    from fast_partial_tracking.metrics import rsnr
    from fast_partial_tracking.synthesis import synthesize

    plt = _pyplot()
    y = synthesize(partials) if resynthesis is None else resynthesis
    fs = partials.grid.fs
    t = np.arange(audio.size) / fs
    fig, (ax_wave, ax_tf) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[1, 3], sharex=True)
    ax_wave.plot(t, audio, color="k", lw=0.6, label="input")
    ax_wave.plot(t, y, color="tab:red", lw=0.6, alpha=0.8, label="resynthesis")
    ax_wave.legend(loc="upper right", fontsize=8)
    ax_wave.set_title(f"R-SNR {rsnr(audio, y):.2f} dB, {partials.num_tracks} partials")
    _draw_partials(partials, ax=ax_tf, audio=audio, max_frequency_hz=max_frequency_hz)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig
