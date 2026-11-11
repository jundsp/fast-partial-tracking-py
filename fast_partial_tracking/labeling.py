"""Partial labeling (Section 3.6).

A useful assignment that continues an existing trajectory keeps that trajectory's
label; one that does not starts a new trajectory with a new label (Fig. 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SetMatch:
    """Result of matching the elements of two sets of unique values.

    Attributes:
        a: Indices into the first set of matched elements, in first-set order.
        b: Indices into the second set of the same elements.
        unmatched_a: Indices of first-set elements absent from the second set, ascending.
        unmatched_b: Indices of second-set elements absent from the first set, ascending.
    """

    a: np.ndarray
    b: np.ndarray
    unmatched_a: np.ndarray
    unmatched_b: np.ndarray


def match_sets(first: np.ndarray, second: np.ndarray) -> SetMatch:
    """Matches equal values between two arrays of unique values (``jun_match_sets.m``).

    Args:
        first: Unique values, shape (n_a,).
        second: Unique values, shape (n_b,).

    Returns:
        The matched and unmatched index sets.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    order = np.argsort(second, kind="stable")
    if second.size:
        pos = np.minimum(np.searchsorted(second[order], first), second.size - 1)
        found = second[order][pos] == first
    else:
        pos = np.zeros(first.size, dtype=np.intp)
        found = np.zeros(first.size, dtype=bool)
    a = np.flatnonzero(found)
    b = order[pos[found]]
    matched_b = np.zeros(second.size, dtype=bool)
    matched_b[b] = True
    return SetMatch(a=a, b=b, unmatched_a=np.flatnonzero(~found), unmatched_b=np.flatnonzero(~matched_b))


@dataclass(frozen=True)
class Labeling:
    """Track labels after linking frame k-1 to frame k.

    Attributes:
        continue_prev: Peak index in frame k-1 of each continuing partial.
        continue_curr: Peak index in frame k of each continuing partial.
        continue_track: Track label of each continuing partial.
        birth_prev: Peak index in frame k-1 where each new partial starts.
        birth_curr: Peak index in frame k of each new partial.
        birth_track: New track label of each new partial.
    """

    continue_prev: np.ndarray
    continue_curr: np.ndarray
    continue_track: np.ndarray
    birth_prev: np.ndarray
    birth_curr: np.ndarray
    birth_track: np.ndarray

    @property
    def curr_peaks(self) -> np.ndarray:
        """Peak indices of frame k in output row order (continuations, then births)."""
        return np.concatenate([self.continue_curr, self.birth_curr])

    @property
    def curr_tracks(self) -> np.ndarray:
        """Track labels of frame k in output row order."""
        return np.concatenate([self.continue_track, self.birth_track])


def label_assignments(
    active_peaks: np.ndarray,
    active_tracks: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    next_track: int,
) -> Labeling:
    """Labels the useful assignments between frames k-1 and k (Section 3.6).

    Args:
        active_peaks: Peak indices in frame k-1 that are already part of a partial (the
            frame-k ends of the previous step's assignments), in row order.
        active_tracks: Track label of each of those peaks.
        rows: Frame k-1 peak index of each useful assignment, ascending.
        cols: Frame k peak index of each useful assignment.
        next_track: First unused track label.

    Returns:
        The continuing and newly born partials.
    """
    match = match_sets(active_peaks, rows)
    births = match.unmatched_b
    return Labeling(
        continue_prev=rows[match.b],
        continue_curr=cols[match.b],
        continue_track=np.asarray(active_tracks)[match.a],
        birth_prev=rows[births],
        birth_curr=cols[births],
        birth_track=next_track + np.arange(births.size),
    )
