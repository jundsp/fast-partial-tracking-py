"""Linking peaks into partials (Sections 3.3-3.7 of the paper).

Costs of useful and spurious assignments (eqs. 8-14), the linear assignment problem
(eq. 3, Section 3.5), partial labeling (Section 3.6) and the streaming tracker that
combines them frame by frame (Section 3.7).
"""

from __future__ import annotations

from fast_partial_tracking.tracking.assignment import solve_assignment, useful_assignments
from fast_partial_tracking.tracking.cost import assignment_costs, cost_variances, midpoint_parameters
from fast_partial_tracking.tracking.labeling import Labeling, SetMatch, label_assignments, match_sets
from fast_partial_tracking.tracking.tracker import (
    PartialTracker,
    full_scale_level_db,
    reference_level_at,
    reference_level_db,
    track,
)

__all__ = [
    "Labeling",
    "PartialTracker",
    "SetMatch",
    "assignment_costs",
    "cost_variances",
    "full_scale_level_db",
    "label_assignments",
    "match_sets",
    "midpoint_parameters",
    "reference_level_at",
    "reference_level_db",
    "solve_assignment",
    "track",
    "useful_assignments",
]
