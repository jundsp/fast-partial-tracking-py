from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fast_partial_tracking.cli import main
from fast_partial_tracking.io import read, read_sdif_tracks

from .conftest import MatlabCase


@pytest.fixture()
def demo_wav(tmp_path: Path, demo_case: MatlabCase) -> Path:
    path = tmp_path / "demo.wav"
    sf.write(path, demo_case.x, int(demo_case.fs), subtype="DOUBLE")
    return path


def test_analyze_matches_matlab_demo(tmp_path: Path, demo_wav: Path, demo_case: MatlabCase) -> None:
    out = tmp_path / "demo.parquet"
    main(["analyze", str(demo_wav), "-o", str(out), "--preset", "matlab", "--threshold-db", "-50"])
    partials = read(out)
    assert partials.num_tracks == demo_case["num_tracks"]
    np.testing.assert_array_equal(partials.track_id + 1, demo_case.partials[:, 5])


def test_streaming_analyze_convert_synth_info(
    tmp_path: Path, demo_wav: Path, caplog: pytest.LogCaptureFixture
) -> None:
    arrows = tmp_path / "demo.arrows"
    main(["analyze", str(demo_wav), "-o", str(arrows), "--stream", "--block-size", "1000"])
    sdif = tmp_path / "demo.sdif"
    main(["convert", str(arrows), "-o", str(sdif)])
    assert read_sdif_tracks(sdif)
    wav = tmp_path / "resynth.wav"
    main(["synth", str(arrows), "-o", str(wav)])
    y, fs = sf.read(wav)
    assert fs == 44100 and y.size == sf.info(demo_wav).frames
    with caplog.at_level(logging.INFO, logger="fast_partial_tracking.cli"):
        main(["info", str(arrows)])
    assert "partials=" in caplog.text
