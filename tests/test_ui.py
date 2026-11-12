from __future__ import annotations

from pathlib import Path

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
