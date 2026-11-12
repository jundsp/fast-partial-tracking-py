from __future__ import annotations

import numpy as np

from fast_partial_tracking.tracking.labeling import label_assignments, match_sets


def test_match_sets_orders_like_matlab() -> None:
    match = match_sets(np.array([7, 3, 9, 1]), np.array([1, 5, 7]))
    np.testing.assert_array_equal(match.a, [0, 3])
    np.testing.assert_array_equal(match.b, [2, 0])
    np.testing.assert_array_equal(match.unmatched_a, [1, 2])
    np.testing.assert_array_equal(match.unmatched_b, [1])


def test_match_sets_empty() -> None:
    match = match_sets(np.zeros(0, int), np.array([4, 2]))
    assert match.a.size == 0
    np.testing.assert_array_equal(match.unmatched_b, [0, 1])
    match = match_sets(np.array([4, 2]), np.zeros(0, int))
    np.testing.assert_array_equal(match.unmatched_a, [0, 1])


def test_continuations_and_births() -> None:
    labels = label_assignments(
        active_peaks=np.array([2, 0]),
        active_tracks=np.array([10, 11]),
        rows=np.array([0, 1, 2]),
        cols=np.array([3, 0, 1]),
        next_track=12,
    )
    np.testing.assert_array_equal(labels.continue_prev, [2, 0])
    np.testing.assert_array_equal(labels.continue_track, [10, 11])
    np.testing.assert_array_equal(labels.birth_prev, [1])
    np.testing.assert_array_equal(labels.birth_track, [12])
    np.testing.assert_array_equal(labels.curr_peaks, [1, 3, 0])
    np.testing.assert_array_equal(labels.curr_tracks, [10, 11, 12])
