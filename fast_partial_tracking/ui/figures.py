"""Interactive Plotly figures used by the web app (also usable in notebooks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fast_partial_tracking.partials import Partials

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
        colorscale="Greys", showscale=False, hoverinfo="skip",
    )


def partial_traces(partials: Partials, dynamic_range_db: float = 80.0, colorscale: str = "Plasma") -> list[Any]:
    """Line traces of all partials, grouped into amplitude bands (one color per band).

    Each partial is drawn as one polyline; partials are separated by gaps so a band needs a
    single WebGL trace. Hovering shows the partial label, frequency and level.
    """
    from plotly.colors import sample_colorscale

    go = _go()
    if len(partials) == 0:
        return []
    level = partials.amplitude_db
    top = level.max()
    track_level = np.full(max(partials.num_tracks, 1), -np.inf)
    np.maximum.at(track_level, partials.track_id, level)
    edges = np.linspace(top - dynamic_range_db, top, _AMPLITUDE_BANDS + 1)
    band = np.clip(np.digitize(track_level, edges) - 1, 0, _AMPLITUDE_BANDS - 1)
    colors = sample_colorscale(colorscale, np.linspace(0.1, 0.95, _AMPLITUDE_BANDS))
    order = np.lexsort((partials.frame, partials.track_id))
    traces = []
    for b in range(_AMPLITUDE_BANDS):
        rows = order[band[partials.track_id[order]] == b]
        if rows.size == 0:
            continue
        x, y, custom = _polylines(partials, rows)
        traces.append(
            go.Scattergl(
                x=x, y=y, customdata=custom, mode="lines", line={"color": colors[b], "width": 1.5},
                name=f"{edges[b]:.0f} to {edges[b + 1]:.0f} dB",
                hovertemplate="partial %{customdata[0]}<br>%{x:.3f} s<br>%{y:.1f} Hz<br>%{customdata[1]:.1f} dB<extra></extra>",
            )
        )
    return traces


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
) -> Any:
    """Partials over the spectrogram, zoomable, with hover details."""
    go = _go()
    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    fig = go.Figure()
    if audio is not None:
        fig.add_trace(spectrogram_trace(audio, fs, max_frequency_hz))
    for trace in partial_traces(partials):
        fig.add_trace(trace)
    fig.update_layout(
        height=height, margin={"l": 60, "r": 20, "t": 30, "b": 50},
        xaxis_title="time (s)", yaxis_title="frequency (Hz)",
        yaxis_range=[0, max_frequency_hz], legend={"title": "partial level", "orientation": "h", "y": 1.08},
        plot_bgcolor="white",
    )
    return fig


def waveform_figure(audio: np.ndarray, resynthesis: np.ndarray, fs: float, height: int = 220) -> Any:
    """Input and resynthesis waveforms (decimated), plus the residual."""
    go = _go()
    step = max(1, audio.size // 6000)
    t = np.arange(0, audio.size, step) / fs
    fig = go.Figure()
    for name, signal, color in [("input", audio, "black"), ("resynthesis", resynthesis, "#d62728"),
                                ("residual", audio - resynthesis, "#7f7f7f")]:
        fig.add_trace(go.Scattergl(x=t, y=signal[::step], name=name, line={"color": color, "width": 1}))
    fig.update_layout(height=height, margin={"l": 60, "r": 20, "t": 10, "b": 40}, xaxis_title="time (s)",
                      legend={"orientation": "h", "y": 1.15})
    return fig


def length_histogram(partials: Partials, height: int = 260) -> Any:
    """Histogram of partial lengths in frames."""
    go = _go()
    lengths = partials.track_lengths()
    fig = go.Figure(go.Histogram(x=lengths[lengths > 0], marker_color="#1f77b4"))
    fig.update_layout(height=height, margin={"l": 60, "r": 20, "t": 10, "b": 40},
                      xaxis_title="partial length (frames)", yaxis_title="partials", bargap=0.05)
    return fig
