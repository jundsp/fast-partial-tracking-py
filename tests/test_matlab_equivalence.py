"""End-to-end equivalence with the MATLAB implementation on every reference case."""

from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking import rsnr, synthesize, track
from fast_partial_tracking.framing import FrameGrid

from .conftest import MatlabCase


@pytest.fixture(scope="module")
def results(matlab_case: MatlabCase):
    partials = track(matlab_case.x, matlab_case.fs, matlab_case.analysis, matlab_case.tracking)
    return matlab_case, partials, synthesize(partials)


def test_frame_grid(matlab_case: MatlabCase) -> None:
    grid = FrameGrid.from_config(matlab_case.fs, matlab_case.analysis).with_signal_length(matlab_case.x.size)
    assert grid.window_length == matlab_case["N"]
    assert grid.hop == matlab_case["H"]
    assert grid.n_dft == matlab_case["Ndft"]
    assert grid.num_frames == matlab_case["M"]
    assert grid.origin == matlab_case["n_center"]
    assert (grid.pad_front, grid.pad_back) == tuple(matlab_case["padding"])
    np.testing.assert_array_equal(grid.center(np.arange(grid.num_frames)) + 1, matlab_case["time"])


def test_partial_structure_is_identical(results) -> None:
    case, partials, _ = results
    ref = case.partials
    assert partials.num_tracks == case["num_tracks"]
    assert len(partials) == ref.shape[0]
    np.testing.assert_array_equal(partials.frame + 1, ref[:, 0])
    np.testing.assert_array_equal(partials.track_id + 1, ref[:, 5])


def test_partial_parameters_match(results) -> None:
    case, partials, _ = results
    ref = case.partials
    np.testing.assert_allclose(partials.omega, ref[:, 2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(partials.log_amp, ref[:, 3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(partials.phase, ref[:, 4], rtol=0, atol=1e-9)


def test_synthesis_matches(results) -> None:
    case, _, y = results
    np.testing.assert_allclose(y, case["y"], rtol=0, atol=1e-9)
    assert rsnr(case.x, y) == pytest.approx(float(case["rsnr"]), abs=1e-6)
