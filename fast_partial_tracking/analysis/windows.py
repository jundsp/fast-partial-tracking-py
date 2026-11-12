"""Analysis windows and their time derivatives.

The Distribution Derivative Method needs both the window w(n) and its derivative
w'(n) (Betser 2009), so each window is defined analytically rather than taken from
``scipy.signal``. Sample positions match ``jun_hannwindow.m`` / ``jun_bhwindow.m``.
"""

from __future__ import annotations

import numpy as np

_BLACKMAN_HARRIS = np.array([0.35875, 0.48829, 0.14128, 0.01168])


def hann(length: int, derivative: bool = False) -> np.ndarray:
    """Symmetric Hann window (or its time derivative), zero at both ends.

    Args:
        length: Window length N in samples.
        derivative: If True, returns dw/dn instead of w(n).

    Returns:
        Array of shape (length,).
    """
    step = 2.0 * np.pi / (length - 1)
    if length % 2 == 1:
        n = np.arange(length) - (length - 1) / 2
        return -0.5 * step * np.sin(step * n) if derivative else 0.5 + 0.5 * np.cos(step * n)
    n = np.arange(length, dtype=float)
    return 0.5 * step * np.sin(step * n) if derivative else 0.5 - 0.5 * np.cos(step * n)


def blackman_harris(length: int, derivative: bool = False) -> np.ndarray:
    """4-term Blackman-Harris window (or its time derivative), centered on n = 0.

    Odd lengths are symmetric about the middle sample; even lengths span
    n = -N/2 .. N/2 - 1.

    Args:
        length: Window length N in samples.
        derivative: If True, returns dw/dn instead of w(n).

    Returns:
        Array of shape (length,).
    """
    half = (length - 1) // 2
    n = np.arange(length) - (half if length % 2 else half + 1)
    a = _BLACKMAN_HARRIS
    s1, s2, s3 = np.array([2.0, 4.0, 6.0]) * np.pi / length
    # Terms are summed in the same order as jun_bhwindow.m so results are bit-identical.
    if derivative:
        return -a[1] * s1 * np.sin(s1 * n) - a[2] * s2 * np.sin(s2 * n) - a[3] * s3 * np.sin(s3 * n)
    return a[0] + a[1] * np.cos(s1 * n) + a[2] * np.cos(s2 * n) + a[3] * np.cos(s3 * n)


def get_window(name: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns a window and its derivative by name.

    Args:
        name: ``"hann"`` or ``"blackman_harris"``.
        length: Window length N.

    Returns:
        (w, dw): the window and its time derivative, each of shape (length,).
    """
    windows = {"hann": hann, "blackman_harris": blackman_harris}
    if name not in windows:
        raise ValueError(f"unknown window {name!r}; expected one of {sorted(windows)}")
    func = windows[name]
    return func(length), func(length, derivative=True)
