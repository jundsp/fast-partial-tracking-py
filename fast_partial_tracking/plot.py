"""Publication-style plots of partials (optional dependency: ``pip install fast-partial-tracking[plot]``).

The default look: each partial drawn as a smooth line in its own saturated color over a
light, low-contrast spectrogram, with serif (Computer Modern) typography.

Example:
    >>> import fast_partial_tracking as fpt
    >>> fig = fpt.plot.figure(partials, audio=x, out_path="partials.pdf")
    >>> with fpt.plot.style():           # same look for your own layouts
    ...     ax = fpt.plot.partials(partials, audio=x)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fast_partial_tracking.partials import Partials

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ColorBy = Literal["track", "amplitude", "none"]

# Saturated, mutually distinct colors; consecutive partial labels get different colors.
PALETTE = (
    "#0a3cff", "#ff1a1a", "#00d000", "#ff8c00", "#e6008a", "#008b8b",
    "#b8a000", "#ff8fc7", "#8b1a8b", "#00a6e6", "#e0e000", "#a0522d",
    "#5a00ff", "#00c78c", "#c80046", "#1e6e1e",
)

STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10", "STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting needs matplotlib: pip install 'fast-partial-tracking[plot]'") from exc
    return plt


@contextmanager
def style() -> Iterator[None]:
    """Applies the package's figure style (serif fonts, clean axes) inside a ``with`` block.

    Fonts are resolved when a figure is drawn, so create *and* save or show figures
    inside the block. ``use_style()`` applies it for the rest of the session instead.
    """
    plt = _pyplot()
    with plt.rc_context(STYLE):
        yield


def use_style() -> None:
    """Applies the package's figure style globally (e.g. at the top of a notebook)."""
    _pyplot().rcParams.update(STYLE)


def spectrogram(
    audio: np.ndarray,
    fs: float,
    ax: Axes | None = None,
    n_fft: int | None = None,
    hop: int | None = None,
    dynamic_range_db: float = 80.0,
    light: bool = True,
    max_frequency_hz: float | None = None,
) -> Axes:
    """Draws a smooth log-magnitude spectrogram.

    Args:
        audio: Mono signal.
        fs: Sampling rate in Hz.
        ax: Axes to draw on (new figure if None).
        n_fft: Window and DFT length (default: about 46 ms, a power of two).
        hop: Hop size (default n_fft // 16, for a smooth image).
        dynamic_range_db: Levels shown below the maximum.
        light: Light, low-contrast grays (a background for partials); False gives a
            full-contrast spectrogram with loud parts dark.
        max_frequency_hz: Only compute and draw up to this frequency.

    Returns:
        The axes.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.signal import stft

    ax = ax or _pyplot().subplots(figsize=(6.4, 4.4))[1]
    n_fft = n_fft or int(2 ** np.round(np.log2(0.046 * fs)))
    hop = hop or max(1, n_fft // 16)
    freqs, times, spectrum = stft(audio, fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    keep = freqs <= (max_frequency_hz or fs / 2) * 1.02
    level = 20 * np.log10(np.abs(spectrum[keep]) + 1e-12)
    top = level.max()
    if light:
        cmap = LinearSegmentedColormap.from_list("fpt_light", ["#7a7a7a", "#bdbdbd", "#ffffff"])
    else:
        cmap = "gray_r"
    ax.pcolormesh(times, freqs[keep], level, shading="gouraud", cmap=cmap, vmin=top - dynamic_range_db, vmax=top,
                  rasterized=True)
    _label_axes(ax)
    return ax


def partials(
    partials: Partials,
    ax: Axes | None = None,
    audio: np.ndarray | None = None,
    color_by: ColorBy = "track",
    max_frequency_hz: float | None = None,
    min_frames: int = 1,
    dynamic_range_db: float = 80.0,
    linewidth: float = 1.6,
    cmap: str = "plasma",
    colorbar: bool = True,
) -> Axes:
    """Draws every partial as a line in the time-frequency plane.

    Args:
        partials: Tracking result.
        ax: Axes to draw on (new figure if None).
        audio: If given, a light spectrogram is drawn underneath.
        color_by: ``"track"`` (one saturated color per partial), ``"amplitude"`` (level in
            dB on ``cmap``) or ``"none"``.
        max_frequency_hz: Upper frequency limit (default Nyquist).
        min_frames: Hide partials shorter than this (declutters noisy analyses).
        dynamic_range_db: Amplitude range of the color scale (``color_by="amplitude"``).
        linewidth: Line width in points.
        cmap: Colormap for ``color_by="amplitude"``.
        colorbar: Add a colorbar when coloring by amplitude.

    Returns:
        The axes.
    """
    from matplotlib.collections import LineCollection

    plt = _pyplot()
    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    if ax is None:
        ax = plt.subplots(figsize=(6.4, 4.4))[1]
    if audio is not None:
        spectrogram(audio, fs, ax=ax, max_frequency_hz=max_frequency_hz)
    shown = partials.prune(min_frames) if min_frames > 1 else partials
    segments, values, ids = _segments(shown)
    collection = LineCollection(segments, linewidths=linewidth, capstyle="round", joinstyle="round")
    if color_by == "track":
        collection.set_color([PALETTE[i % len(PALETTE)] for i in ids])
    elif color_by == "amplitude":
        collection.set_array(values)
        collection.set_cmap(cmap)
        if values.size:
            collection.set_clim(values.max() - dynamic_range_db, values.max())
    else:
        collection.set_color("#d62728")
    ax.add_collection(collection)
    ax.set_xlim(0, (partials.grid.signal_length or 0) / fs)
    ax.set_ylim(0, max_frequency_hz)
    _label_axes(ax)
    if colorbar and color_by == "amplitude" and values.size:
        plt.colorbar(collection, ax=ax, label="Amplitude (dB)")
    return ax


_draw_partials = partials


def _segments(partials: Partials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Line segments between consecutive points of each partial, with levels and labels."""
    order = np.lexsort((partials.frame, partials.track_id))
    track_id = partials.track_id[order]
    points = np.column_stack([partials.time_s[order], partials.frequency_hz[order]])
    link = track_id[1:] == track_id[:-1]
    segments = np.stack([points[:-1][link], points[1:][link]], axis=1)
    level = partials.amplitude_db[order]
    return segments, 0.5 * (level[:-1] + level[1:])[link], track_id[:-1][link]


def _label_axes(ax: Axes) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


def figure(
    partials: Partials,
    audio: np.ndarray | None = None,
    max_frequency_hz: float | None = 5000.0,
    min_frames: int = 3,
    color_by: ColorBy = "track",
    size: tuple[float, float] = (6.4, 4.4),
    out_path: str | Path | None = None,
) -> Figure:
    """A single-panel, publication-style figure of the partials over the spectrogram.

    Args:
        partials: Tracking result.
        audio: Input signal for the background spectrogram (optional).
        max_frequency_hz: Upper frequency limit.
        min_frames: Hide partials shorter than this.
        color_by: See ``partials``.
        size: Figure size in inches.
        out_path: If given, the figure is saved there (format from the extension).

    Returns:
        The figure.
    """
    plt = _pyplot()
    with style():
        fig, ax = plt.subplots(figsize=size)
        _draw_partials(partials, ax=ax, audio=audio, color_by=color_by, max_frequency_hz=max_frequency_hz,
                       min_frames=min_frames)
        fig.tight_layout()
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
    return fig


def overview(
    partials: Partials,
    audio: np.ndarray,
    resynthesis: np.ndarray | None = None,
    max_frequency_hz: float | None = 5000.0,
    min_frames: int = 3,
    out_path: str | Path | None = None,
) -> Figure:
    """Input and resynthesis waveforms above the partials drawn over the spectrogram.

    Args:
        partials: Tracking result.
        audio: Input signal.
        resynthesis: Resynthesized signal (computed if None).
        max_frequency_hz: Upper frequency limit of the partials plot.
        min_frames: Hide partials shorter than this in the partials panel.
        out_path: If given, the figure is saved there.

    Returns:
        The figure.
    """
    from fast_partial_tracking.metrics import rsnr
    from fast_partial_tracking.synthesis import synthesize

    plt = _pyplot()
    y = synthesize(partials) if resynthesis is None else resynthesis
    t = np.arange(audio.size) / partials.grid.fs
    with style():
        fig, (ax_wave, ax_tf) = plt.subplots(2, 1, figsize=(7.2, 6.4), height_ratios=[1, 3], sharex=True)
        ax_wave.plot(t, audio, color="0.15", lw=0.6, label="Input")
        ax_wave.plot(t, y, color="#ff1a1a", lw=0.6, alpha=0.75, label="Resynthesis")
        ax_wave.legend(loc="upper right", fontsize=9, frameon=False, ncols=2)
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title(f"R-SNR {rsnr(audio, y):.1f} dB, {partials.num_tracks} partials", fontsize=12)
        _draw_partials(partials, ax=ax_tf, audio=audio, max_frequency_hz=max_frequency_hz, min_frames=min_frames)
        fig.tight_layout()
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
    return fig
