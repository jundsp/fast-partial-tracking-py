"""Sound transformations that operate on tracked partials.

Each function returns a new ``Partials``. Transforms that change frequencies or frame
times mark the result for phase-integrating synthesis (``synthesize`` picks it up
automatically), because the measured phases no longer fit the new trajectories.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fast_partial_tracking.partials import Partials

_INTEGRATED = {"synthesis_phase": "integrated"}


def pitch_shift(partials: Partials, semitones: float = 0.0, ratio: float | None = None) -> Partials:
    """Scales every partial's frequency, keeping timing and amplitudes.

    Partials pushed to or above Nyquist are removed.

    Args:
        partials: Input partials.
        semitones: Shift in equal-tempered semitones (ignored if ``ratio`` is given).
        ratio: Frequency ratio, e.g. 2.0 for an octave up.

    Returns:
        The shifted partials.
    """
    ratio = float(2.0 ** (semitones / 12.0)) if ratio is None else float(ratio)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    alpha = None
    if partials.alpha is not None:
        # Frequency and its derivatives live in the imaginary parts of alpha_1..Q.
        alpha = partials.alpha.copy()
        alpha[:, 1:] = alpha[:, 1:].real + 1j * alpha[:, 1:].imag * ratio
    shifted = replace(
        partials, omega=partials.omega * ratio, alpha=alpha, metadata={**partials.metadata, **_INTEGRATED}
    )
    return shifted.subset(shifted.omega < np.pi)


def time_stretch(partials: Partials, factor: float) -> Partials:
    """Changes the duration without changing pitch, by rescaling the frame spacing.

    The hop size is rounded to whole samples, so the effective factor can differ slightly
    from the requested one (it is stored as ``metadata["time_stretch"]``).

    Args:
        partials: Input partials.
        factor: Duration ratio (> 1 slows down, < 1 speeds up).

    Returns:
        The stretched partials.
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    grid = partials.grid
    hop = max(1, int(round(grid.hop * factor)))
    actual = hop / grid.hop
    signal_length = int(round(grid.signal_length * actual))
    last_center = (grid.num_frames - 1) * hop + grid.origin
    pad_back = max(0, last_center + 1 - grid.pad_front - signal_length)
    stretched_grid = replace(grid, hop=hop, signal_length=signal_length, pad_back=pad_back)

    alpha = None
    if partials.alpha is not None:
        # a(n) and the frequency w(n) are slowed by `actual`: a'(n) = a(n/s), w'(n) = w(n/s).
        powers = np.arange(partials.alpha.shape[1])
        alpha = partials.alpha.real / actual**powers + 1j * partials.alpha.imag / actual ** np.maximum(powers - 1, 0)
    metadata = {**partials.metadata, **_INTEGRATED, "time_stretch": actual}
    return replace(partials, grid=stretched_grid, alpha=alpha, metadata=metadata)


def gain(partials: Partials, db: float) -> Partials:
    """Changes the level of every partial by ``db`` decibels.

    Args:
        partials: Input partials.
        db: Gain in dB (negative attenuates).

    Returns:
        The rescaled partials.
    """
    offset = db / 20.0 * np.log(10.0)
    alpha = None
    if partials.alpha is not None:
        alpha = partials.alpha.copy()
        alpha[:, 0] += offset
    return replace(partials, log_amplitude=partials.log_amplitude + offset, alpha=alpha, metadata=dict(partials.metadata))
