"""Interactive Plotly figures used by the web app (also usable in notebooks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fast_partial_tracking.partials import Partials
from fast_partial_tracking.plot import PALETTE

_MAX_COLUMNS = 700
_AMPLITUDE_BANDS = 6
_UNSELECTED = "rgba(40, 40, 40, 0.35)"


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("interactive figures need plotly: pip install plotly") from exc
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
    # Pad half a window on each side so frames are centred from t = 0 to the end, like the partials.
    padded = np.pad(np.asarray(audio, dtype=float), n_fft // 2)
    freqs, times, magnitude = spectrogram(padded, fs, nperseg=n_fft, noverlap=n_fft * 3 // 4, mode="magnitude")
    times = times - (n_fft // 2) / fs
    keep = freqs <= max_frequency_hz
    step = max(1, times.size // _MAX_COLUMNS)
    level = 20 * np.log10(magnitude[keep][:, ::step] + 1e-12)
    return _go().Heatmap(
        x=times[::step], y=freqs[keep], z=level, zmin=level.max() - floor_db, zmax=level.max(),
        colorscale=[[0.0, "#7a7a7a"], [0.5, "#bdbdbd"], [1.0, "#ffffff"]], showscale=False, hoverinfo="skip",
    )


def partial_traces(
    partials: Partials,
    color_by: str = "track",
    dynamic_range_db: float = 80.0,
    colorscale: str = "Plasma",
    selected: np.ndarray | None = None,
) -> list[Any]:
    """Line traces of all partials, one WebGL trace per color.

    Partials are drawn as polylines separated by gaps, so each color group needs a single
    trace. ``color_by="track"`` cycles the saturated palette of ``fpt.plot`` so neighbouring
    partials differ; ``"amplitude"`` groups partials into level bands. Hovering shows the
    partial label, frequency and level.

    Each point also carries an invisible marker, so box and lasso selection pick up the
    lines; ``track_ids_from_points`` turns such a selection back into partial labels.

    Args:
        partials: The partials to draw.
        color_by: ``"track"`` or ``"amplitude"``.
        dynamic_range_db: Level range spanned by the amplitude bands.
        colorscale: Plotly colorscale of the amplitude bands.
        selected: Labels of selected partials. When given and not empty, the other
            partials are drawn thin and grey behind the selected ones.
    """
    if len(partials) == 0:
        return []
    order = np.lexsort((partials.frame, partials.track_id))
    n = max(partials.num_tracks, 1)
    is_selected = np.ones(n, dtype=bool)
    traces = []
    if selected is not None and len(selected):
        is_selected = np.zeros(n, dtype=bool)
        is_selected[np.asarray(selected, dtype=np.int64)] = True
        background = np.where(is_selected, -1, 0)
        if not is_selected.all():
            traces.append(_trace(partials, order, background, 0, _UNSELECTED, "not selected", width=1.0))
    if color_by == "track":
        group = np.where(is_selected, np.arange(n) % len(PALETTE), -1)
        return traces + [_trace(partials, order, group, g, PALETTE[g], None) for g in range(len(PALETTE))
                         if np.any(group[partials.track_id] == g)]
    from plotly.colors import sample_colorscale

    level = partials.amplitude_db
    top = level.max()
    track_level = np.full(n, -np.inf)
    np.maximum.at(track_level, partials.track_id, level)
    edges = np.linspace(top - dynamic_range_db, top, _AMPLITUDE_BANDS + 1)
    group = np.clip(np.digitize(track_level, edges) - 1, 0, _AMPLITUDE_BANDS - 1)
    group = np.where(is_selected, group, -1)
    colors = sample_colorscale(colorscale, np.linspace(0.1, 0.95, _AMPLITUDE_BANDS))
    return traces + [_trace(partials, order, group, b, colors[b], f"{edges[b]:.0f} to {edges[b + 1]:.0f} dB")
                     for b in range(_AMPLITUDE_BANDS) if np.any(group[partials.track_id] == b)]


def track_ids_from_points(figure: Any, points: list[dict[str, Any]]) -> np.ndarray:
    """Partial labels of selected points, as reported by a Plotly selection event.

    Args:
        figure: The figure the selection was made on.
        points: Selected points, each with ``curve_number`` and ``point_index`` (the
            ``selection.points`` list of ``st.plotly_chart(on_select=...)``).

    Returns:
        Sorted unique labels; points on non-partial traces are ignored.
    """
    ids = set()
    for point in points:
        trace = figure.data[point["curve_number"]]
        meta = trace.meta or {}
        index = point.get("point_index", point.get("point_number"))
        if not meta.get("partials") or index is None:
            continue
        value = trace.customdata[index][0]
        if np.isfinite(value):
            ids.add(int(value))
    return np.array(sorted(ids), dtype=np.int64)


def _trace(
    partials: Partials, order: np.ndarray, group: np.ndarray, value: int, color: str, name: str | None,
    width: float = 2.0,
) -> Any:
    """One polyline trace for the partials whose group equals ``value``."""
    rows = order[group[partials.track_id[order]] == value]
    x, y, custom = _polylines(partials, rows)
    return _go().Scattergl(
        x=x, y=y, customdata=custom, mode="lines+markers", line={"color": color, "width": width},
        marker={"size": 6, "opacity": 0, "color": color}, meta={"partials": True},
        selected={"marker": {"opacity": 0}}, unselected={"marker": {"opacity": 0}},
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
    selected: np.ndarray | None = None,
) -> Any:
    """Partials over a light spectrogram, zoomable, with hover details (see ``partial_traces``)."""
    go = _go()
    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    fig = go.Figure()
    if audio is not None:
        fig.add_trace(spectrogram_trace(audio, fs, max_frequency_hz))
    for trace in partial_traces(partials, color_by=color_by, selected=selected):
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
    selected: np.ndarray | None = None,
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
    for trace in partial_traces(partials, color_by=color_by, selected=selected):
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


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(frequency_hz: float, a4_hz: float = 440.0) -> str:
    """Nearest equal-tempered note with its deviation in cents, e.g. ``"A4 +3"``."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        return ""
    midi = 69.0 + 12.0 * np.log2(frequency_hz / a4_hz)
    nearest = int(np.rint(midi))
    cents = int(np.rint(100.0 * (midi - nearest)))
    return f"{_NOTE_NAMES[nearest % 12]}{nearest // 12 - 1} {cents:+d}"


def track_figure(track: Any, height: int = 360) -> Any:
    """Frequency and level of one partial over time, in two panels with a shared time axis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(_go().Scatter(x=track.time_s, y=track.frequency_hz, mode="lines+markers", name="frequency",
                                line={"color": PALETTE[track.id % len(PALETTE)], "width": 2}, marker={"size": 4},
                                hovertemplate="%{x:.3f} s<br>%{y:.2f} Hz<extra></extra>"), row=1, col=1)
    fig.add_trace(_go().Scatter(x=track.time_s, y=track.amplitude_db, mode="lines+markers", name="level",
                                line={"color": "#444444", "width": 2}, marker={"size": 4},
                                hovertemplate="%{x:.3f} s<br>%{y:.1f} dB<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Level (dB)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=height, margin={"l": 70, "r": 20, "t": 10, "b": 40}, showlegend=False,
                      plot_bgcolor="white")
    return fig
