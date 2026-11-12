from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from fast_partial_tracking import AnalysisConfig, data, track  # noqa: E402
from fast_partial_tracking.ui import figures  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "fast_partial_tracking" / "ui" / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders_results(app: AppTest) -> None:
    labels = [m.label for m in app.metric]
    assert labels == ["R-SNR", "Partials", "Frames", "Analysis time"]
    assert float(app.metric[0].value.split()[0]) > 15.0
    assert len(app.tabs) == 5


def test_changing_settings_reruns(app: AppTest) -> None:
    before = app.metric[1].value
    threshold = next(s for s in app.sidebar.slider if s.label.startswith("Peak threshold"))
    threshold.set_value(-30).run()
    assert not app.exception
    assert int(app.metric[1].value) < int(before)


def test_synthetic_source_and_matlab_preset(app: AppTest) -> None:
    app.sidebar.selectbox[0].select("Synthetic: vibrato").run()
    app.sidebar.radio[0].set_value("MATLAB (jun_demo)").run()
    assert not app.exception
    assert int(app.metric[1].value) > 0


def test_figures() -> None:
    x, fs = data.synthetic("chord", duration=0.5)
    partials = track(x, fs, AnalysisConfig(window_length=511))
    fig = figures.partials_figure(partials, x, max_frequency_hz=4000)
    assert fig.data[0].type == "heatmap" and len(fig.data) > 1
    assert figures.length_histogram(partials).data
    assert len(figures.waveform_figure(x, x, fs).data) == 3
    combined = figures.overview_figure(partials, x, x, max_frequency_hz=4000)
    assert combined.layout.xaxis.matches == "x2" or combined.layout.xaxis2.matches == "x"
    assert {trace.xaxis for trace in combined.data} == {"x", "x2"}


def test_harmonic_selection_highlights_partials() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.selectbox[0].select("Synthetic: harmonic").run()
    next(n for n in at.number_input if n.label == "Fundamental (Hz)").set_value(220.0)
    [b for b in at.button if b.label == "Select"][1].click().run()  # rule form first, then harmonics
    assert not at.exception, at.exception
    heading = next(h.value for h in at.subheader if h.value.startswith("Selection:"))
    assert int(heading.split()[1]) >= 8
    assert any(sb.label == "Inspect partial" for sb in at.selectbox)

    # Listen: selected partials over a time range at half speed, looped.
    play = next(r for r in at.radio if r.label == "Play")
    assert play.options == ["Selected partials", "All partials"]
    at.slider(key="play_range").set_value((0.2, 0.6)).run()
    next(s for s in at.select_slider if s.label == "Speed").set_value(0.5).run()
    assert not at.exception, at.exception

    # Transform and Export can be limited to the selection.
    at.toggle(key="transform_scope").set_value(True).run()
    at.toggle(key="export_scope").set_value(True).run()
    assert not at.exception, at.exception


def test_selection_traces_and_point_mapping() -> None:
    x, fs = data.synthetic("chord", duration=0.5)
    partials = track(x, fs, AnalysisConfig(window_length=511))
    picked = np.array([0, 3])
    fig = figures.partials_figure(partials, x, max_frequency_hz=4000, selected=picked)
    ids = lambda traces: {int(v) for t in traces for v in t.customdata[:, 0] if np.isfinite(v)}  # noqa: E731
    background = [t for t in fig.data if t.name == "not selected"]
    foreground = [t for t in fig.data if t.meta and t.name != "not selected"]
    # Every partial is still drawn: the selection in front, the rest faded behind it.
    assert ids(foreground) == {0, 3}
    assert ids(background) | ids(foreground) == set(range(partials.num_tracks))
    assert max(t.opacity for t in background) < min(t.opacity for t in foreground)
    assert sum(t.showlegend for t in background) == 1
    assert min(t.line.width for t in foreground) > max(t.line.width for t in background)
    curve = next(i for i, t in enumerate(fig.data) if t.meta and t.name != "not selected")
    index = int(np.flatnonzero(np.isfinite(fig.data[curve].customdata[:, 0]))[0])
    points = [{"curve_number": curve, "point_index": index}, {"curve_number": 0, "point_index": 0}]
    assert figures.track_ids_from_points(fig, points).tolist() == [int(fig.data[curve].customdata[index, 0])]


def test_note_names_and_track_figure() -> None:
    assert figures.note_name(440.0) == "A4 +0"
    assert figures.note_name(261.63) == "C4 +0"
    assert figures.note_name(1010.2) == "B5 +39"
    assert figures.note_name(0.0) == ""
    x, fs = data.synthetic("vibrato", duration=0.5)
    partials = track(x, fs, AnalysisConfig(window_length=511))
    fig = figures.track_figure(partials.track(0))
    assert len(fig.data) == 2


def test_frequency_axes_and_line_weights() -> None:
    x, fs = data.synthetic("chord", duration=0.5)
    partials = track(x, fs, AnalysisConfig(window_length=511))
    assert figures.frequency_axis("linear", 0, 4000)["range"] == [0, 4000]
    log = figures.frequency_axis("log", 0, 4000)
    assert log["type"] == "log" and log["range"][0] == pytest.approx(np.log10(figures.LOG_MIN_HZ))
    wide = figures.frequency_axis("notes", 30, 8000)
    assert all(t.startswith("C") for t in wide["ticktext"]) and "C4" in wide["ticktext"]
    narrow = figures.frequency_axis("notes", 200, 800)
    assert {"A4", "E5", "B3"} <= set(narrow["ticktext"])
    assert narrow["tickvals"][narrow["ticktext"].index("A4")] == pytest.approx(440.0)
    with pytest.raises(ValueError):
        figures.frequency_axis("mel", 0, 4000)

    fig = figures.overview_figure(partials, x, x, max_frequency_hz=4000, frequency_scale="notes", min_frequency_hz=100)
    assert fig.layout.yaxis2.type == "log"
    heatmap = next(t for t in fig.data if t.type == "heatmap")
    # The spectrogram covers the whole band, so panning beyond the initial range still shows it.
    assert min(heatmap.y) >= figures.LOG_MIN_HZ and max(heatmap.y) == pytest.approx(fs / 2, rel=0.01)
    # Panning and zooming stay within the signal and 0 Hz (20 Hz on log axes) to Nyquist.
    assert fig.layout.yaxis2.minallowed == pytest.approx(np.log10(figures.LOG_MIN_HZ))
    assert fig.layout.yaxis2.maxallowed == pytest.approx(np.log10(fs / 2))
    assert fig.layout.xaxis2.minallowed == 0 and fig.layout.xaxis2.maxallowed == pytest.approx(0.5, abs=0.01)
    assert fig.layout.yaxis.fixedrange  # waveform panel does not pan vertically
    linear = figures.partials_figure(partials, x, max_frequency_hz=4000)
    assert (linear.layout.yaxis.minallowed, linear.layout.yaxis.maxallowed) == (0, fs / 2)

    flat = figures.partial_traces(partials)
    weighted = figures.partial_traces(partials, weight_by_level=True)
    assert len({t.line.width for t in flat}) == 1
    assert len({t.line.width for t in weighted}) > 1
    ids = lambda traces: sorted(int(v) for t in traces for v in t.customdata[:, 0] if np.isfinite(v))  # noqa: E731
    assert ids(flat) == ids(weighted)


def test_max_partials_per_frame_slider() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    before = int(at.metric[1].value)
    next(s for s in at.sidebar.slider if s.label == "Max partials per frame").set_value(10).run()
    assert not at.exception, at.exception
    assert int(at.metric[1].value) < before


def test_view_and_selection_controls() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    x, fs = data.demo_sound()
    highest = track(x, fs).frequency_hz.max()
    band = next(s for s in at.slider if s.label == "Frequency range (Hz)")
    assert band.value[0] == 0 and highest <= band.value[1] < highest + 500
    assert next(r for r in at.radio if r.label == "Drag on the plot to").options == ["Zoom", "Box", "Lasso", "Pan"]
    clear = next(b for b in at.button if b.label == "Clear selection")
    assert clear.disabled
    next(b for b in at.button if b.label == "Select all").click().run()
    total = int(at.metric[1].value)
    assert any(m.value == f"**{total} of {total} partials selected**" for m in at.markdown)
    next(b for b in at.button if b.label == "Clear selection").click().run()
    assert any(m.value == "**No partials selected**" for m in at.markdown)
    next(s for s in at.slider if s.label == "Time range (s)").set_value((0.1, 0.3)).run()
    assert not at.exception, at.exception


def test_time_view_range() -> None:
    x, fs = data.synthetic("chord", duration=0.5)
    partials = track(x, fs, AnalysisConfig(window_length=511))
    fig = figures.overview_figure(partials, x, x, time_range_s=(0.1, 0.3))
    assert list(fig.layout.xaxis2.range) == [0.1, 0.3]
    assert fig.layout.xaxis2.maxallowed == pytest.approx(0.5, abs=0.01)
