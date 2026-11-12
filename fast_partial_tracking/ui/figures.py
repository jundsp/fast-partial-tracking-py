"""Interactive Plotly figures used by the web app (also usable in notebooks)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fast_partial_tracking.partials import Partials
from fast_partial_tracking.plot import PALETTE

_MAX_COLUMNS = 700
_AMPLITUDE_BANDS = 6
_UNSELECTED_OPACITY = 0.6
_SELECTED_EXTRA_WIDTH = 1.0
# (line width, opacity) for quiet, medium and loud partials; loud is the default weight.
_WEIGHTS = ((0.8, 0.4), (1.4, 0.7), (2.2, 1.0))
LOG_MIN_HZ = 20.0
# Dragging draws a zoom box (which cannot leave the frame); uirevision keeps the user's zoom
# and chosen tool when the figure is redrawn, e.g. after a selection.
_INTERACTION = {"dragmode": "zoom", "uirevision": "partials"}


def _go() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("interactive figures need plotly: pip install plotly") from exc
    return go


def spectrogram_trace(
    audio: np.ndarray, fs: float, max_frequency_hz: float, floor_db: float = 90.0, min_frequency_hz: float = 0.0
) -> Any:
    """Heatmap trace of the log-magnitude spectrogram, decimated for fast rendering.

    Args:
        audio: Mono signal.
        fs: Sampling rate in Hz.
        max_frequency_hz: Highest frequency shown.
        floor_db: Dynamic range below the maximum.
        min_frequency_hz: Lowest frequency shown (use > 0 on a log axis).
    """
    from scipy.signal import spectrogram

    n_fft = int(2 ** np.ceil(np.log2(0.04 * fs)))
    # Pad half a window on each side so frames are centred from t = 0 to the end, like the partials.
    padded = np.pad(np.asarray(audio, dtype=float), n_fft // 2)
    freqs, times, magnitude = spectrogram(padded, fs, nperseg=n_fft, noverlap=n_fft * 3 // 4, mode="magnitude")
    times = times - (n_fft // 2) / fs
    keep = (freqs <= max_frequency_hz) & (freqs >= min_frequency_hz)
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
    weight_by_level: bool = False,
) -> list[Any]:
    """Line traces of all partials, one WebGL trace per color (and line weight).

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
        selected: Labels of selected partials. When given and not empty, the selected
            partials are drawn thicker and the others thin and faded behind them.
        weight_by_level: Draw loud partials thick and opaque and quiet ones thin and faint.
            Plotly sets one width per trace, so partials are sorted into three weights by
            their peak level (within 20 dB of the loudest, within 40 dB, quieter).
    """
    if len(partials) == 0:
        return []
    order = np.lexsort((partials.frame, partials.track_id))
    n = max(partials.num_tracks, 1)
    track_level = np.full(n, -np.inf)
    np.maximum.at(track_level, partials.track_id, partials.amplitude_db)
    top = float(track_level.max())
    weight = np.full(n, len(_WEIGHTS) - 1)
    if weight_by_level:
        weight = np.digitize(track_level, [top - 40.0, top - 20.0])
    is_selected = np.ones(n, dtype=bool)
    emphasis = 0.0
    if selected is not None and len(selected):
        is_selected = np.zeros(n, dtype=bool)
        is_selected[np.asarray(selected, dtype=np.int64)] = True
        emphasis = _SELECTED_EXTRA_WIDTH
    if color_by == "track":
        color = np.arange(n) % len(PALETTE)
        palette = list(PALETTE)
        names: list[str | None] = [None] * len(palette)
    else:
        from plotly.colors import sample_colorscale

        edges = np.linspace(top - dynamic_range_db, top, _AMPLITUDE_BANDS + 1)
        color = np.clip(np.digitize(track_level, edges) - 1, 0, _AMPLITUDE_BANDS - 1)
        palette = sample_colorscale(colorscale, np.linspace(0.1, 0.95, _AMPLITUDE_BANDS))
        names = [f"{edges[b]:.0f} to {edges[b + 1]:.0f} dB" for b in range(_AMPLITUDE_BANDS)]
    traces = []
    # Unselected partials keep their colors but are drawn thin and faded, behind the selection.
    if not is_selected.all():
        background = np.where(is_selected, -1, color)
        present = set(np.unique(background[partials.track_id]).tolist())
        first = True
        for c, hue in enumerate(palette):
            if c in present:
                traces.append(_trace(partials, order, background, c, hue, "not selected", width=1.3,
                                     opacity=_UNSELECTED_OPACITY, showlegend=first))
                first = False
    group = np.where(is_selected, color * len(_WEIGHTS) + weight, -1)
    present = set(np.unique(group[partials.track_id]).tolist())
    for c, (hue, name) in enumerate(zip(palette, names)):
        shown = False
        for w in reversed(range(len(_WEIGHTS))):
            if c * len(_WEIGHTS) + w not in present:
                continue
            width, opacity = _WEIGHTS[w]
            traces.append(_trace(partials, order, group, c * len(_WEIGHTS) + w, hue, name, width=width + emphasis,
                                 opacity=opacity, showlegend=name is not None and not shown))
            shown = True
    return traces


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
    width: float = 2.0, opacity: float = 1.0, showlegend: bool | None = None,
) -> Any:
    """One polyline trace for the partials whose group equals ``value``."""
    rows = order[group[partials.track_id[order]] == value]
    x, y, custom = _polylines(partials, rows)
    return _go().Scattergl(
        x=x, y=y, customdata=custom, mode="lines+markers", line={"color": color, "width": width},
        marker={"size": 6, "opacity": 0, "color": color}, meta={"partials": True}, opacity=opacity,
        selected={"marker": {"opacity": 0}}, unselected={"marker": {"opacity": 0}},
        name=name or "", legendgroup=name or None, showlegend=name is not None if showlegend is None else showlegend,
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


def frequency_axis(scale: str, low_hz: float, high_hz: float, nyquist_hz: float | None = None) -> dict[str, Any]:
    """Plotly y-axis settings for a frequency range on a ``"linear"``, ``"log"`` or ``"notes"`` scale.

    ``"notes"`` is a log axis labelled with note names: every C when the range spans more
    than three octaves, otherwise every natural note. With ``nyquist_hz``, panning and
    zooming are limited to 0 Hz (or ``LOG_MIN_HZ`` on log axes) up to that frequency.
    """
    if scale == "linear":
        bounds = {} if nyquist_hz is None else {"minallowed": 0.0, "maxallowed": nyquist_hz}
        return {"type": "linear", "range": [low_hz, high_hz], "tickformat": ",d", "title_text": "Frequency (Hz)",
                **bounds}
    if scale not in ("log", "notes"):
        raise ValueError(f"scale must be 'linear', 'log' or 'notes', got {scale!r}")
    low_hz = max(low_hz, LOG_MIN_HZ)
    axis = {"type": "log", "range": [np.log10(low_hz), np.log10(high_hz)], "title_text": "Frequency (Hz)"}
    if nyquist_hz is not None:
        axis.update(minallowed=np.log10(LOG_MIN_HZ), maxallowed=np.log10(nyquist_hz))
    if scale == "log":
        return {**axis, "tickformat": ",d"}
    lo, hi = (int(np.ceil(69 + 12 * np.log2(low_hz / 440.0))), int(np.floor(69 + 12 * np.log2(high_hz / 440.0))))
    steps = {0} if hi - lo > 36 else {0, 2, 4, 5, 7, 9, 11}
    midi = [m for m in range(lo, hi + 1) if m % 12 in steps]
    return {**axis, "title_text": "Note", "tickvals": [440.0 * 2 ** ((m - 69) / 12) for m in midi],
            "ticktext": [f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in midi]}


def partials_figure(
    partials: Partials,
    audio: np.ndarray | None = None,
    max_frequency_hz: float | None = None,
    height: int = 520,
    color_by: str = "track",
    selected: np.ndarray | None = None,
    frequency_scale: str = "linear",
    min_frequency_hz: float = 0.0,
    weight_by_level: bool = False,
    time_range_s: tuple[float, float] | None = None,
) -> Any:
    """Partials over a light spectrogram, zoomable, with hover details (see ``partial_traces``).

    ``min_frequency_hz``/``max_frequency_hz`` and ``time_range_s`` set the initial view;
    panning and zooming can reach 0 Hz to Nyquist over the whole signal.
    """
    go = _go()
    fs = partials.grid.fs
    max_frequency_hz = max_frequency_hz or fs / 2
    axis = frequency_axis(frequency_scale, min_frequency_hz, max_frequency_hz, nyquist_hz=fs / 2)
    fig = go.Figure()
    if audio is not None:
        # The whole band is drawn so that panning beyond the initial range still shows the spectrogram.
        fig.add_trace(spectrogram_trace(audio, fs, fs / 2, min_frequency_hz=_allowed_low(axis)))
    for trace in partial_traces(partials, color_by=color_by, selected=selected, weight_by_level=weight_by_level):
        fig.add_trace(trace)
    fig.update_yaxes(**axis)
    fig.update_layout(
        height=height, margin={"l": 70, "r": 20, "t": 30, "b": 50}, **_INTERACTION,
        xaxis_title="Time (s)",
        xaxis=_time_axis((partials.grid.signal_length or 0) / fs, time_range_s),
        legend={"title": "partial level", "orientation": "h", "y": 1.08},
        plot_bgcolor="white", font={"family": "Latin Modern Roman, CMU Serif, Georgia, serif", "size": 14},
    )
    return fig


def _allowed_low(axis: dict[str, Any]) -> float:
    """Lowest frequency an axis from ``frequency_axis`` can be panned or zoomed to."""
    if axis["type"] == "log":
        return float(10 ** axis.get("minallowed", axis["range"][0]))
    return float(axis.get("minallowed", axis["range"][0]))


def _time_axis(duration_s: float, view_s: tuple[float, float] | None = None) -> dict[str, Any]:
    """Time axis showing ``view_s`` (default: everything), limited to the signal's duration."""
    view = list(view_s) if view_s is not None else [0, duration_s]
    return {"range": view, "minallowed": 0, "maxallowed": duration_s}


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
    frequency_scale: str = "linear",
    min_frequency_hz: float = 0.0,
    weight_by_level: bool = False,
    time_range_s: tuple[float, float] | None = None,
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
    axis = frequency_axis(frequency_scale, min_frequency_hz, max_frequency_hz, nyquist_hz=fs / 2)
    fig.add_trace(spectrogram_trace(audio, fs, fs / 2, min_frequency_hz=_allowed_low(axis)), row=2, col=1)
    for trace in partial_traces(partials, color_by=color_by, selected=selected, weight_by_level=weight_by_level):
        fig.add_trace(trace, row=2, col=1)
    duration = (partials.grid.signal_length or 0) / fs
    fig.update_xaxes(**_time_axis(duration, time_range_s))
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", fixedrange=True, row=1, col=1)
    fig.update_yaxes(**axis, row=2, col=1)
    fig.update_layout(
        height=height, margin={"l": 70, "r": 20, "t": 30, "b": 50}, plot_bgcolor="white", **_INTERACTION,
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
