"""Quality measures."""

from __future__ import annotations

import numpy as np


def rsnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Reconstruction signal-to-noise ratio in dB (eq. 18).

    R-SNR = 10 log10( sum s(n)^2 / sum (s(n) - s_hat(n))^2 )

    Args:
        reference: Original signal s(n).
        estimate: Reconstruction s_hat(n), same shape as ``reference``.

    Returns:
        R-SNR in dB (inf for a perfect reconstruction).
    """
    reference = np.asarray(reference, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    if reference.shape != estimate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {estimate.shape}")
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(np.sum(reference**2) / np.sum((reference - estimate) ** 2)))
