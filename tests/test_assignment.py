from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking.assignment import solve_assignment, useful_assignments

from .conftest import mat_struct, load_mat

REF = load_mat("assignment")


@pytest.mark.parametrize("index", range(len(REF["cases"])))
def test_matches_munkres(index: int) -> None:
    case = mat_struct(REF["cases"][index])
    cost = np.atleast_2d(case["C"])
    rows, cols = solve_assignment(cost)
    assignment = np.zeros(cost.shape[0], dtype=int)
    assignment[rows] = cols + 1
    np.testing.assert_array_equal(assignment, np.atleast_1d(case["assign"]))
    assert cost[rows, cols].sum() == pytest.approx(float(case["cost"]))


def test_empty() -> None:
    rows, cols = solve_assignment(np.zeros((0, 4)))
    assert rows.size == cols.size == 0


def test_spurious_pairs_are_dropped() -> None:
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    useful = np.array([[True, False], [False, False]])
    rows, cols = useful_assignments(cost, useful)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(cols, [0])
