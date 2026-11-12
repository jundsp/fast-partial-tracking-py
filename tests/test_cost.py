from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking.assignment import useful_assignments
from fast_partial_tracking.config import TrackingConfig
from fast_partial_tracking.cost import assignment_costs, cost_variances, midpoint_parameters
from fast_partial_tracking.ddm import DdmEstimator
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.tracker import reference_level_db

from .conftest import MatlabCase


def test_variances_match_matlab(matlab_case: MatlabCase) -> None:
    var_f, var_a = cost_variances(matlab_case.tracking)
    assert var_f == pytest.approx(float(matlab_case["varF"]), rel=1e-14)
    assert var_a == pytest.approx(float(matlab_case["varA"]), rel=1e-14)


def test_cost_matrices_match_matlab(matlab_case: MatlabCase) -> None:
    case = matlab_case
    grid = FrameGrid.from_config(case.fs, case.analysis).with_signal_length(case.x.size)
    estimator = DdmEstimator(grid, case.analysis)
    padded = np.concatenate([np.zeros(grid.pad_front), case.x, np.zeros(grid.pad_back)])
    threshold = reference_level_db(case.x, grid, estimator.window) - abs(case.analysis.peak_threshold_db)
    var_f, var_a = cost_variances(case.tracking)
    n_curr, n_prev = grid.midpoint_offsets

    def estimate(k: int):
        return estimator(padded[k * grid.hop : k * grid.hop + grid.window_length], threshold)

    for ddm_ref, cost_ref in zip(case.ddm_frames(), case.cost_frames()):
        k = int(ddm_ref["frame"]) - 1
        prev, curr = estimate(k - 1), estimate(k)
        if len(prev) == 0 or len(curr) == 0:
            continue
        a_prev, f_prev = midpoint_parameters(prev.alpha, n_prev, grid.fs)
        a_curr, f_curr = midpoint_parameters(curr.alpha, n_curr, grid.fs)
        np.testing.assert_allclose(f_prev, np.atleast_1d(cost_ref["mF1"]), rtol=0, atol=1e-6)
        np.testing.assert_allclose(a_curr, np.atleast_1d(cost_ref["mA2"]), rtol=0, atol=1e-8)

        cost, useful = assignment_costs(a_prev, f_prev, a_curr, f_curr, var_f, var_a, case.tracking.delta)
        shape = (len(prev), len(curr))
        np.testing.assert_allclose(cost, np.reshape(cost_ref["C"], shape), rtol=0, atol=1e-9)
        np.testing.assert_array_equal(useful, np.reshape(cost_ref["type"], shape) == 2)

        rows, cols = useful_assignments(cost, useful)
        np.testing.assert_array_equal(rows + 1, np.atleast_1d(cost_ref["useful_rows"]))
        np.testing.assert_array_equal(cols + 1, np.atleast_1d(cost_ref["useful_cols"]))


def test_cost_crossover_and_ties() -> None:
    tracking = TrackingConfig(delta=0.2, zeta_f=50.0, zeta_a=15.0)
    var_f, var_a = cost_variances(tracking)
    zeros = np.zeros(1)
    near, _ = assignment_costs(zeros, zeros, zeros, np.array([1.0]), var_f, var_a, tracking.delta)
    _, useful_near = assignment_costs(zeros, zeros, zeros, np.array([1.0]), var_f, var_a, tracking.delta)
    _, useful_far = assignment_costs(zeros, zeros, zeros, np.array([500.0]), var_f, var_a, tracking.delta)
    assert useful_near.all() and not useful_far.any()
    assert near[0, 0] < 0.01

    # A = B exactly when A = 1/(2 - delta): such ties count as spurious, like MATLAB.
    a_tie = 1.0 / (2.0 - tracking.delta)
    df = np.sqrt(-var_f * np.log(1.0 - a_tie))
    cost, useful = assignment_costs(zeros, zeros, zeros, np.array([df]), var_f, var_a, tracking.delta)
    assert cost[0, 0] == pytest.approx(a_tie)


@pytest.mark.parametrize("delta", [0.1, 0.2, 0.5, 0.9])
def test_crossover_is_at_zeta(delta: float) -> None:
    """Eqs. (12)-(13): useful and spurious costs are equal exactly at |Δ| = ζ."""
    tracking = TrackingConfig(delta=delta, zeta_f=37.0, zeta_a=12.0)
    var_f, var_a = cost_variances(tracking)
    zeros = np.zeros(1)
    for offset, expect_useful in [(-1e-6, True), (1e-6, False)]:
        _, useful = assignment_costs(zeros, zeros, zeros, np.array([37.0 + offset]), var_f, var_a, delta)
        assert bool(useful[0, 0]) is expect_useful
    a_zeta = tracking.zeta_a_log
    for offset, expect_useful in [(-1e-9, True), (1e-9, False)]:
        _, useful = assignment_costs(zeros, zeros, np.array([a_zeta + offset]), zeros, var_f, var_a, delta)
        assert bool(useful[0, 0]) is expect_useful
