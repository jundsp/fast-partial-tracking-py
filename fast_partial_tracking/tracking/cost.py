"""Costs of useful and spurious peak-to-peak assignments (Sections 3.3-3.4).

Peak i of frame k-1 and peak j of frame k are compared at the midpoint between the two
frames by extending each peak's short-term model forward/backward by H/2 (eqs. 10-11,
Fig. 2). A useful assignment (continuing a partial) costs A_ij (eq. 8); a spurious one
costs B_ij (eq. 9); the assignment problem uses C_ij = min(A_ij, B_ij) (eq. 14).
"""

from __future__ import annotations

import numpy as np

from fast_partial_tracking.config import TrackingConfig
from fast_partial_tracking.analysis.model import angular_frequency, log_amplitude


def cost_variances(tracking: TrackingConfig) -> tuple[float, float]:
    """Returns the denominators (2σ_f², 2σ_a²) of eq. (8) from ζ_f, ζ_a and δ (eqs. 12-13).

    The useful cost A (eq. 8) equals the spurious cost B = 1 - (1-δ)A (eq. 9) when
    A = 1/(2-δ). Requiring this crossover at |Δf| = ζ_f (and |Δa| = ζ_a) gives
    2σ² = ζ² / ln((2-δ)/(1-δ)).

    Args:
        tracking: Tracking settings.

    Returns:
        (2σ_f² in Hz², 2σ_a² in natural-log-amplitude units²).
    """
    ratio = np.log((2.0 - tracking.delta) / (1.0 - tracking.delta))
    return tracking.zeta_f_hz**2 / ratio, tracking.zeta_a_log**2 / ratio


def midpoint_parameters(
    alpha: np.ndarray, n: float, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log-amplitude and frequency of each peak's model extended to time n.

    Args:
        alpha: Coefficients, shape (R, Q + 1).
        n: Model time relative to the frame origin (e.g. +H/2 or -H/2).
        fs: Sampling rate in Hz.

    Returns:
        (log_amplitude, frequency_hz), each of shape (R,).
    """
    return log_amplitude(alpha, n), fs / (2.0 * np.pi) * angular_frequency(alpha, n)


def assignment_costs(
    log_amp_prev: np.ndarray,
    freq_prev: np.ndarray,
    log_amp_curr: np.ndarray,
    freq_curr: np.ndarray,
    var_f: float,
    var_a: float,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the cost matrix C and the useful-assignment mask (eqs. 8-11, 14).

    Args:
        log_amp_prev: a_i^{k-1}(+H/2) of the R_{k-1} previous peaks.
        freq_prev: f_i^{k-1}(+H/2) in Hz.
        log_amp_curr: a_j^k(-H/2) of the R_k current peaks.
        freq_curr: f_j^k(-H/2) in Hz.
        var_f: Frequency denominator 2σ_f² (see ``cost_variances``).
        var_a: Log-amplitude denominator 2σ_a².
        delta: δ of eq. (9).

    Returns:
        (C, useful): the (R_{k-1}, R_k) cost matrix and a boolean mask that is True where
        A_ij < B_ij. Ties count as spurious, matching MATLAB's ``min``.
    """
    delta_a = log_amp_prev[:, None] - log_amp_curr[None, :]  # eq. (10)
    delta_f = freq_prev[:, None] - freq_curr[None, :]  # eq. (11)
    useful_cost = 1.0 - np.exp(-(delta_f**2) / var_f - delta_a**2 / var_a)  # eq. (8)
    spurious_cost = 1.0 - (1.0 - delta) * useful_cost  # eq. (9)
    useful = useful_cost < spurious_cost
    cost = np.where(useful, useful_cost, spurious_cost)  # eq. (14)
    # Degenerate estimates (inf/nan parameters) must not reach the solver: treat as spurious.
    invalid = ~np.isfinite(cost)
    if np.any(invalid):
        cost[invalid] = 1.0
        useful &= ~invalid
    return cost, useful
