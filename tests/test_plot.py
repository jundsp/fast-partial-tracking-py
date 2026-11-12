from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from fast_partial_tracking import AnalysisConfig, Partials, plot, track  # noqa: E402

from .conftest import harmonic_signal  # noqa: E402

FS = 16000.0


@pytest.fixture(scope="module")
def signal_and_partials() -> tuple[np.ndarray, Partials]:
    x = harmonic_signal(FS, duration=0.5)
    return x, track(x, FS, AnalysisConfig(window_length=511))


@pytest.mark.parametrize("color_by", ["amplitude", "track", "none"])
def test_partials_draws_one_segment_per_consecutive_pair(signal_and_partials, color_by: str) -> None:
    _, partials = signal_and_partials
    ax = plot.partials(partials, color_by=color_by, colorbar=False)
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines) == 1
    assert len(lines[0].get_segments()) == len(partials) - partials.num_tracks
    plt.close("all")


def test_overview_and_method(tmp_path: Path, signal_and_partials) -> None:
    x, partials = signal_and_partials
    out = tmp_path / "overview.png"
    fig = plot.overview(partials, x, out_path=out)
    assert out.stat().st_size > 0 and len(fig.axes) >= 2
    ax = partials.plot(audio=x, max_frequency_hz=4000)
    assert ax.get_ylim()[1] == 4000
    plt.close("all")
