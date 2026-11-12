"""Short-term generalized sinusoidal model (Section 3.2, eqs. 4-7).

Within frame k, peak j is modeled as ``exp(sum_i alpha_ij n**i)`` (eq. 4) with complex
coefficients alpha_ij, i = 0..Q. Coefficient arrays here have shape (R, Q + 1): one row
per peak, column i holding alpha_i.
"""

from __future__ import annotations

import numpy as np


def polynomial_basis(n: np.ndarray | float, order: int) -> np.ndarray:
    """Monomials n**i for i = 0..Q (MATLAB ``p``).

    Args:
        n: Model time(s) in samples relative to the frame origin.
        order: Q.

    Returns:
        Array of shape (..., Q + 1).
    """
    return np.power(np.asarray(n, dtype=float)[..., None], np.arange(order + 1))


def polynomial_basis_derivative(n: np.ndarray | float, order: int) -> np.ndarray:
    """Derivatives i * n**(i-1) for i = 1..Q (MATLAB ``pD``).

    Args:
        n: Model time(s) in samples relative to the frame origin.
        order: Q.

    Returns:
        Array of shape (..., Q).
    """
    i = np.arange(1, order + 1)
    return i * np.power(np.asarray(n, dtype=float)[..., None], i - 1)


def log_amplitude(alpha: np.ndarray, n: float) -> np.ndarray:
    """Instantaneous log-amplitude a(n) = Re{sum_i alpha_i n**i} (eq. 5).

    Args:
        alpha: Coefficients, shape (R, Q + 1).
        n: Model time in samples.

    Returns:
        Log-amplitude of each peak, shape (R,).
    """
    return (alpha @ polynomial_basis(n, alpha.shape[-1] - 1)).real


def phase(alpha: np.ndarray, n: float) -> np.ndarray:
    """Instantaneous phase phi(n) = Im{sum_i alpha_i n**i} (eq. 6), not wrapped.

    Args:
        alpha: Coefficients, shape (R, Q + 1).
        n: Model time in samples.

    Returns:
        Phase of each peak in radians, shape (R,).
    """
    return (alpha @ polynomial_basis(n, alpha.shape[-1] - 1)).imag


def angular_frequency(alpha: np.ndarray, n: float) -> np.ndarray:
    """Instantaneous frequency Im{sum_i i alpha_i n**(i-1)} in radians/sample (eq. 7).

    Multiply by fs / (2 pi) for Hz.

    Args:
        alpha: Coefficients, shape (R, Q + 1).
        n: Model time in samples.

    Returns:
        Angular frequency of each peak, shape (R,).
    """
    return (alpha[:, 1:] @ polynomial_basis_derivative(n, alpha.shape[-1] - 1)).imag


def frame_parameters(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frequency, log-amplitude and phase at the frame origin n = 0 (``jun_framph.m``).

    At n = 0 eqs. (5)-(7) reduce to omega = Im{alpha_1}, a = Re{alpha_0} and
    phi = Im{alpha_0}.

    Args:
        alpha: Coefficients, shape (R, Q + 1).

    Returns:
        (omega, log_amplitude, phase), each of shape (R,).
    """
    return alpha[:, 1].imag.copy(), alpha[:, 0].real.copy(), alpha[:, 0].imag.copy()
