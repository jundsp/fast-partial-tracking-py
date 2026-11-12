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
    background = [t for t in fig.data if t.name == "not selected"]
    assert len(background) == 1
    coloured_ids = {int(v) for t in fig.data if t.meta and t.name != "not selected"
                    for v in t.customdata[:, 0] if np.isfinite(v)}
    assert coloured_ids == {0, 3}
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
