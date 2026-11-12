"""Interactive Plotly figures used by the web app (also usable in notebooks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fast_partial_tracking.partials import Partials
from fast_partial_tracking.plot import PALETTE

_MAX_COLUMNS = 700
_AMPLITUDE_BANDS = 6


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("interactive figures need plotly: pip install 'fast-partial-tracking[ui]'") from exc
    return go


def spectrogram_trace(audio: np.ndarray, fs: float, max_frequency_hz: float, floor_db: float = 90.0) -> Any:
    """Heatmap trace of the log-magnitude spectrogram, decimated for fast rendering.

    Args:
        audio: Mono signal.
        fs: Sampling rate in Hz.
        max_frequency_hz: Highest frequency shown.
        floor_db: Dynamic range below the maximum.
    """
    from scipy.signal import spectrogram

    n_fft = int(2 ** np.ceil(np.log2(0.04 * fs)))
    freqs, times, magnitude = spectrogram(audio, fs, nperseg=n_fft, noverlap=n_fft * 3 // 4, mode="magnitude")
    keep = freqs <= max_frequency_hz
    step = max(1, times.size // _MAX_COLUMNS)
    level = 20 * np.log10(magnitude[keep][:, ::step] + 1e-12)
    return _go().Heatmap(
        x=times[::step], y=freqs[keep], z=level, zmin=level.max() - floor_db, zmax=level.max(),
        colorscale=[[0.0, "#7a7a7a"], [0.5, "#bdbdbd"], [1.0, "#ffffff"]], showscale=False, hoverinfo="skip",
    )


def partial_traces(
    partials: Partials, color_by: str = "track", dynamic_range_db: float = 80.0, colorscale: str = "Plasma"
) -> list[Any]:
    """Line traces of all partials, one WebGL trace per color.

    Partials are drawn as polylines separated by gaps, so each color group needs a single
    trace. ``color_by="track"`` cycles the saturated palette of ``fpt.plot`` so neighbouring
    partials differ; ``"amplitude"`` groups partials into level bands. Hovering shows the
    partial label, frequency and level.
    """
    if len(partials) == 0:
        return []
    order = np.lexsort((partials.frame, partials.track_id))
    if color_by == "track":
        group = np.arange(max(partials.num_tracks, 1)) % len(PALETTE)
        return [_trace(partials, order, group, g, PALETTE[g], None) for g in range(len(PALETTE))
                if np.any(group[partials.track_id] == g)]
    from plotly.colors import sample_colorscale

    level = partials.amplitude_db
    top = level.max()
    track_level = np.full(max(partials.num_tracks, 1), -np.inf)
    np.maximum.at(track_level, partials.track_id, level)
    edges = np.linspace(top - dynamic_range_db, top, _AMPLITUDE_BANDS + 1)
    group = np.clip(np.digitize(track_level, edges) - 1, 0, _AMPLITUDE_BANDS - 1)
    colors = sample_colorscale(colorscale, np.linspace(0.1, 0.95, _AMPLITUDE_BANDS))
    return [_trace(partials, order, group, b, colors[b], f"{edges[b]:.0f} to {edges[b + 1]:.0f} dB")
            for b in range(_AMPLITUDE_BANDS) if np.any(group[partials.track_id] == b)]


def _trace(partials: Partials, order: np.ndarray, group: np.ndarray, value: int, color: str, name: str | None) -> Any:
    """One polyline trace for the partials whose group equals ``value``."""
    rows = order[group[partials.track_id[order]] == value]
    x, y, custom = _polylines(partials, rows)
    return _go().Scattergl(
        x=x, y=y, customdata=custom, mode="lines", line={"color": color, "width": 2},
        name=name or "", showlegend=name is not None,
        hovertemplate="partial %{customdata[0]}<br>%{x:.3f} s<br>%{y:.1f} Hz<br>%{customdata[1]:.1f} dB<extra></extra>",
    )


def _polylines(partials: Partials, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of the given rows (sorted by partial, then frame) with NaN breaks between partials."""
    breaks = np.flatnonzero(np.diff(partials.track_id[rows]) != 0) + 1
    x = np.insert(partials.time_s[rows], breaks, np.nan)
    y = np.insert(partials.frequency_hz[rows], breaks, np.nan)
    custom = np.column_stack([partials.track_id[rows], partials.amplitude_db[rows]]).astype(float)
    custom = np.insert(custom, breaks, np.nan, axis=0)
    return x, y, custom


def partials_figure(
    partials: Partials,
    audio: np.ndarray | None = None,
    max_frequency_hz: float | None = None,
    height: int = 520,
    color_by: str = "track",
) -> Any:
    """Partials over a light spectrogram, zoomable, with hover details."""
    go = _go()
    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    fig = go.Figure()
    if audio is not None:
        fig.add_trace(spectrogram_trace(audio, fs, max_frequency_hz))
    for trace in partial_traces(partials, color_by=color_by):
        fig.add_trace(trace)
    fig.update_layout(
        height=height, margin={"l": 70, "r": 20, "t": 30, "b": 50},
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        yaxis={"range": [0, max_frequency_hz], "tickformat": ",d"},
        xaxis={"range": [0, (partials.grid.signal_length or 0) / fs]},
        legend={"title": "partial level", "orientation": "h", "y": 1.08},
        plot_bgcolor="white", font={"family": "Latin Modern Roman, CMU Serif, Georgia, serif", "size": 14},
    )
    return fig


def waveform_figure(audio: np.ndarray, resynthesis: np.ndarray, fs: float, height: int = 220) -> Any:
    """Input and resynthesis waveforms (decimated), plus the residual."""
    go = _go()
    fig = go.Figure(_waveform_traces(audio, resynthesis, fs))
    fig.update_layout(height=height, margin={"l": 70, "r": 20, "t": 10, "b": 40}, xaxis_title="Time (s)",
                      legend={"orientation": "h", "y": 1.15})
    return fig


def _waveform_traces(audio: np.ndarray, resynthesis: np.ndarray, fs: float) -> list[Any]:
    go = _go()
    step = max(1, audio.size // 6000)
    t = np.arange(0, audio.size, step) / fs
    return [
        go.Scattergl(x=t, y=signal[::step], name=name, line={"color": color, "width": 1})
        for name, signal, color in [("input", audio, "black"), ("resynthesis", resynthesis, "#ff1a1a"),
                                    ("residual", audio - resynthesis, "#7f7f7f")]
    ]


def overview_figure(
    partials: Partials,
    audio: np.ndarray,
    resynthesis: np.ndarray,
    max_frequency_hz: float | None = None,
    color_by: str = "track",
    height: int = 760,
) -> Any:
    """Waveforms above the partials, in one figure with a shared time axis.

    Sharing the axis keeps both panels exactly aligned, and zooming or panning one moves
    the other.
    """
    from plotly.subplots import make_subplots

    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.25, 0.75], vertical_spacing=0.03)
    for trace in _waveform_traces(audio, resynthesis, fs):
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(spectrogram_trace(audio, fs, max_frequency_hz), row=2, col=1)
    for trace in partial_traces(partials, color_by=color_by):
        fig.add_trace(trace, row=2, col=1)
    duration = (partials.grid.signal_length or 0) / fs
    fig.update_xaxes(range=[0, duration])
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[0, max_frequency_hz], tickformat=",d", row=2, col=1)
    fig.update_layout(
        height=height, margin={"l": 70, "r": 20, "t": 30, "b": 50}, plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.04}, font={"family": "Latin Modern Roman, CMU Serif, Georgia, serif", "size": 14},
    )
    return fig


def length_histogram(partials: Partials, height: int = 260) -> Any:
    """Histogram of partial lengths in frames."""
    go = _go()
    lengths = partials.track_lengths()
    fig = go.Figure(go.Histogram(x=lengths[lengths > 0], marker_color="#1f77b4"))
    fig.update_layout(height=height, margin={"l": 60, "r": 20, "t": 10, "b": 40},
                      xaxis_title="partial length (frames)", yaxis_title="partials", bargap=0.05)
    return fig
