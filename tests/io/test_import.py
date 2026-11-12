from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fast_partial_tracking as fpt
from fast_partial_tracking.io import from_points, read, read_sdif


@pytest.fixture(scope="module")
def voice() -> tuple[np.ndarray, int, fpt.Partials]:
    x, fs = fpt.data.demo_sound()
    return x, fs, fpt.track(x, fs)


def test_sdif_round_trip_keeps_points_and_sound(tmp_path: Path, voice: tuple[np.ndarray, int, fpt.Partials]) -> None:
    x, fs, partials = voice
    path = tmp_path / "voice.sdif"
    partials.save(path)
    loaded = read(path, fs=fs)
    assert loaded.grid.hop == partials.grid.hop
    # SDIF skips frames before t = 0; the importer drops points at or below 0 Hz.
    original = partials.select(time_s=(0.0, np.inf), frequency_hz=(1e-9, np.inf))
    assert loaded.num_tracks == original.num_tracks
    np.testing.assert_allclose(np.sort(loaded.frequency_hz), np.sort(original.frequency_hz), rtol=1e-12)
    np.testing.assert_allclose(np.unique(loaded.time_s), np.unique(original.time_s), atol=1e-12)
    y = fpt.synthesize(loaded)[: x.size]
    assert fpt.rsnr(x[: y.size], y) == pytest.approx(fpt.rsnr(x, fpt.synthesize(partials)), abs=0.01)


def test_sdif_without_fs_uses_default_and_reports_phase(tmp_path: Path, voice: tuple[np.ndarray, int, fpt.Partials]) -> None:
    _, _, partials = voice
    path = tmp_path / "voice.sdif"
    partials.save(path)
    loaded = read_sdif(path)
    assert loaded.grid.fs == 44100.0
    assert "synthesis_phase" not in loaded.metadata  # measured phases are kept
    assert loaded.metadata["source_format"] == "sdif"


def test_from_points_infers_grid_with_offset_and_splits_gaps() -> None:
    fs, hop = 48000.0, 480
    t0 = 0.0037  # first frame not at 0
    frames = np.arange(10)
    time = np.concatenate([t0 + frames * hop / fs, t0 + frames * hop / fs])
    index = np.concatenate([np.full(10, 7), np.full(10, 3)])
    frequency = np.concatenate([np.full(10, 440.0), np.full(10, 1000.0)])
    amplitude = np.full(20, 0.1)
    # partial 3 skips frame 5: it becomes two partials
    keep = ~((index == 3) & (np.isclose(time, t0 + 5 * hop / fs)))
    p = from_points(time[keep], index[keep], frequency[keep], amplitude[keep], fs=fs)
    assert p.grid.hop == hop
    np.testing.assert_allclose(np.unique(p.time_s), t0 + frames * hop / fs, atol=0.5 / fs)
    assert p.num_tracks == 3
    np.testing.assert_allclose(p.amplitude, 0.1)
    assert p.metadata["synthesis_phase"] == "integrated"
    y = fpt.synthesize(p)
    assert y.size >= int((t0 + 9 * hop / fs) * fs) and np.max(np.abs(y)) > 0.05


def test_from_points_validates_input() -> None:
    with pytest.raises(ValueError):
        from_points(np.zeros(3), np.zeros(2), np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        from_points(np.array([-1.0, 0.0]), np.zeros(2), np.full(2, 100.0), np.ones(2))
