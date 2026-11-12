from __future__ import annotations

import io as stdio
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from fast_partial_tracking import PartialTracker, Partials, synthesize, track
from fast_partial_tracking.io import (
    ArrowStreamWriter,
    iter_arrow_stream,
    open_writer,
    read,
    read_matlab_bin,
    read_sdif_tracks,
    write,
)

from ..conftest import FIXTURES, MatlabCase, load_mat


@pytest.fixture(scope="module")
def demo_partials(demo_case: MatlabCase) -> Partials:
    return track(demo_case.x, demo_case.fs, demo_case.analysis, demo_case.tracking)


def assert_same(a: Partials, b: Partials, exact: bool = False) -> None:
    assert a.grid == b.grid
    assert a.num_tracks == b.num_tracks
    np.testing.assert_array_equal(a.frame, b.frame)
    np.testing.assert_array_equal(a.track_id, b.track_id)
    tol = 0 if exact else 1e-12
    np.testing.assert_allclose(a.omega, b.omega, rtol=tol, atol=0)
    np.testing.assert_allclose(a.log_amplitude, b.log_amplitude, rtol=0, atol=tol)
    np.testing.assert_array_equal(a.phase, b.phase)


@pytest.mark.parametrize("suffix", [".parquet", ".arrows"])
def test_round_trip(tmp_path: Path, demo_partials: Partials, suffix: str) -> None:
    path = tmp_path / f"out{suffix}"
    demo_partials.save(path)
    loaded = Partials.load(path)
    assert_same(demo_partials, loaded)
    np.testing.assert_array_equal(loaded.alpha, demo_partials.alpha)
    assert loaded.metadata["analysis"] == demo_partials.metadata["analysis"]
    np.testing.assert_allclose(synthesize(loaded), synthesize(demo_partials), atol=1e-12)


def test_parquet_without_alpha_and_row_groups(tmp_path: Path, demo_partials: Partials) -> None:
    path = tmp_path / "out.parquet"
    write(demo_partials, path, include_alpha=False, frames_per_group=16)
    meta = pq.read_metadata(path)
    assert meta.num_row_groups == -(-demo_partials.num_frames // 16)
    assert "alpha_re" not in pq.read_schema(path).names
    assert read(path).alpha is None


def test_streaming_writer_from_tracker(tmp_path: Path, demo_case: MatlabCase, demo_partials: Partials) -> None:
    path = tmp_path / "live.parquet"
    tracker = PartialTracker(
        demo_case.fs, demo_case.analysis, demo_case.tracking, reference_db=demo_partials.metadata["reference_db"]
    )
    with open_writer(path, tracker.grid, order=tracker.estimator.order, frames_per_group=8) as writer:
        for start in range(0, demo_case.x.size, 1000):
            for frame in tracker.push(demo_case.x[start : start + 1000]):
                writer.write_frame(frame)
        for frame in tracker.flush():
            writer.write_frame(frame)
        writer.set_summary(tracker.grid, tracker.num_tracks)
    assert_same(read(path), demo_partials)


def test_arrow_stream_is_readable_frame_by_frame(demo_partials: Partials) -> None:
    sink = stdio.BytesIO()
    with ArrowStreamWriter(sink, demo_partials.grid, order=2) as writer:
        writer.write_partials(demo_partials)
    frames = list(iter_arrow_stream(stdio.BytesIO(sink.getvalue())))
    assert [f.index for f in frames] == list(range(demo_partials.num_frames))
    for frame in frames[:: max(1, len(frames) // 10)]:
        expected = demo_partials.get_frame(frame.index)
        np.testing.assert_array_equal(frame.track_id, expected.track_id)
        np.testing.assert_allclose(frame.omega, expected.omega, rtol=1e-12)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_sdif(tmp_path: Path, demo_partials: Partials, precision: str) -> None:
    path = tmp_path / "out.sdif"
    write(demo_partials, path, precision=precision)
    raw = path.read_bytes()
    assert raw[:4] == b"SDIF" and raw[16:20] == b"1TRC"
    tracks = read_sdif_tracks(path)
    frames = [f for f in demo_partials.frames() if f.time_s >= 0]
    assert len(tracks) == len(frames)
    rtol = 1e-15 if precision == "float64" else 1e-6
    for (time, matrix), frame in zip(tracks, frames):
        assert time == frame.time_s
        assert matrix.shape == (len(frame), 4)
        np.testing.assert_array_equal(matrix[:, 0], frame.track_id)
        np.testing.assert_allclose(matrix[:, 1], frame.frequency_hz, rtol=rtol)
        np.testing.assert_allclose(matrix[:, 2], frame.amplitude, rtol=rtol)
    assert len(raw) % 8 == 0


def test_sdif_with_pysdif3(tmp_path: Path, demo_partials: Partials) -> None:
    """Cross-checks with IRCAM's SDIF library (pysdif3 1.0.0 + numpy<2; 1.0.1 is broken)."""
    pysdif = pytest.importorskip("pysdif")
    path = tmp_path / "out.sdif"
    write(demo_partials, path)
    try:
        sdif = pysdif.SdifFile(str(path))
    except TypeError as exc:
        pytest.skip(f"installed pysdif3 cannot open files: {exc}")
    frames = [f for f in demo_partials.frames() if f.time_s >= 0]
    count = 0
    for sdif_frame, frame in zip(sdif, frames):
        matrices = [m.get_data() for m in sdif_frame]
        assert sdif_frame.time == frame.time_s
        np.testing.assert_array_equal(matrices[0][:, 0], frame.track_id)
        np.testing.assert_allclose(matrices[0][:, 1], frame.frequency_hz, rtol=1e-15)
        count += 1
    assert count == len(frames)


def test_matlab_bin_reader(demo_case: MatlabCase) -> None:
    partials = read_matlab_bin(FIXTURES / "demo_partials.bin")
    ref = demo_case.partials
    assert partials.num_tracks == demo_case["num_tracks"]
    np.testing.assert_array_equal(partials.frame + 1, ref[:, 0])
    np.testing.assert_array_equal(partials.track_id + 1, ref[:, 5])
    np.testing.assert_array_equal(partials.omega, ref[:, 2])
    np.testing.assert_allclose(synthesize(partials), demo_case["y"], rtol=0, atol=1e-14)


def test_matlab_bin_writer_is_byte_identical(tmp_path: Path) -> None:
    source = FIXTURES / "demo_partials.bin"
    path = tmp_path / "out.bin"
    write(read_matlab_bin(source), path)
    assert path.read_bytes() == source.read_bytes()


def test_unsupported_extension(tmp_path: Path, demo_partials: Partials) -> None:
    with pytest.raises(ValueError):
        demo_partials.save(tmp_path / "out.csv")
    with pytest.raises(ValueError):
        read(tmp_path / "out.csv")


@pytest.mark.parametrize("suffix", [".parquet", ".arrows"])
def test_reads_version_1_files(suffix: str) -> None:
    """Files written by package 0.1 (MATLAB geometry, old setting names) still load."""
    partials = read(FIXTURES.parent / "v1" / f"vibrato_v1{suffix}")
    case = MatlabCase("vibrato", load_mat("vibrato"))
    # 0.1 used the legacy amplitude check, so compare with the same analysis today.
    expected = track(case.x, case.fs, replace(case.analysis, amplitude_guard="legacy"), case.tracking)
    assert partials.grid.geometry == "matlab"
    np.testing.assert_array_equal(partials.track_id, expected.track_id)
    np.testing.assert_allclose(partials.omega, expected.omega, atol=1e-10)
    np.testing.assert_allclose(synthesize(partials), synthesize(expected), atol=1e-9)


def test_files_record_format_version(tmp_path: Path, demo_partials: Partials) -> None:
    path = tmp_path / "out.parquet"
    demo_partials.save(path)
    header = json.loads(pq.read_metadata(path).metadata[b"fast_partial_tracking.summary"])
    assert header["format_version"] == 2
    assert header["grid"]["geometry"] == demo_partials.grid.geometry
