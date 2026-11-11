"""Optimal peak-to-peak assignment (Sections 2 and 3.5).

The linear assignment problem (eq. 3) is solved with SciPy's
``linear_sum_assignment`` (a shortest augmenting path variant of the Hungarian
algorithm), which handles rectangular cost matrices directly. It replaces Yi Cao's
``munkres.m`` used by the MATLAB implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment of rows to columns (eqs. 3a-3c).

    When the matrix is rectangular, every row (or every column, whichever is fewer) is
    assigned.

    Args:
        cost: Cost matrix of shape (R_{k-1}, R_k).

    Returns:
        (rows, cols): assigned index pairs, sorted by row.
    """
    if cost.size == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    return linear_sum_assignment(cost)


def useful_assignments(cost: np.ndarray, useful: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solves the assignment and keeps only the useful pairs (Section 3.5).

    Following eq. (14), an assigned pair is useful when its cost came from A_ij, i.e.
    A_ij < B_ij. Spurious pairs are assigned but do not link peaks.

    Args:
        cost: Cost matrix C, shape (R_{k-1}, R_k).
        useful: Mask that is True where A_ij < B_ij.

    Returns:
        (rows, cols) of the useful assignments, sorted by row.
    """
    rows, cols = solve_assignment(cost)
    keep = useful[rows, cols]
    return rows[keep], cols[keep]
