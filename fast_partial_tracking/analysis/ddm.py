"""Short-term parameter estimation with the Distribution Derivative Method.

Port of ``jun_ddm.m``. For each spectral peak the DDM (M. Betser, "Sinusoidal polynomial
parameter estimation using the distribution derivative", IEEE Trans. Signal Process.,
2009) estimates the complex coefficients alpha_1..alpha_Q of the model in eq. (4) by
solving a small linear system built from DFTs of the signal weighted by the window,
its derivative and the polynomial derivatives. alpha_0 (log-amplitude and phase at n = 0)
is then obtained by matching the peak bin's DFT value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from fast_partial_tracking.compat.matlab import matlab_round
from fast_partial_tracking.config import AnalysisConfig
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.analysis.model import polynomial_basis
from fast_partial_tracking.analysis.peaks import Peaks, pick_peaks
from fast_partial_tracking.analysis.windows import get_window

log = logging.getLogger(__name__)

try:  # optional, multi-threaded evaluation of the dominant complex exponential
    import numexpr as _numexpr
except ImportError:  # pragma: no cover - exercised when the [fast] extra is absent
    _numexpr = None


def complex_exp(z: np.ndarray) -> np.ndarray:
    """exp(z) for a complex array, using numexpr's threaded kernel when installed."""
    if _numexpr is not None and z.size > 4096:
        return _numexpr.evaluate("exp(z)")
    return np.exp(z)

# Guards against numerically exploding estimates; same limit as jun_ddm.m.
_EDGE_LIMIT = 1e100


@dataclass(frozen=True)
class FrameEstimate:
    """DDM estimates for one analysis frame.

    Attributes:
        alpha: Complex coefficients of the accepted peaks, shape (R, Q + 1), ordered by
            ascending frequency bin.
        peak_bin: DFT bin of each accepted peak, shape (R,).
        spectrum: Windowed DFT of the frame, bins 0..n_dft/2, shape (n_dft // 2 + 1,).
        peaks: All peaks found before the validity checks.
    """

    alpha: np.ndarray
    peak_bin: np.ndarray
    spectrum: np.ndarray
    peaks: Peaks

    def __len__(self) -> int:
        return int(self.alpha.shape[0])


@dataclass(frozen=True)
class _DdmSpectra:
    """DFTs that form the DDM linear system of one frame (all phase-centred on n = 0).

    Attributes:
        spectrum: Plain DFT of w(n) x(n) (first-sample phase reference), shape (K,).
        weighted: Column i holds DFT{w(n) x(n) (i+1) n**i}, i = 0..Q-1, shape (K, Q) with
            K = n_dft // 2 + 1.
        derivative: DFT{w'(n) x(n)} - j omega DFT{w(n) x(n)}, shape (K,).
    """

    spectrum: np.ndarray
    weighted: np.ndarray
    derivative: np.ndarray


class DdmEstimator:
    """Estimates the short-term model (eq. 4) for every spectral peak of a frame.

    All frame-independent quantities (windows, polynomial bases, DFT phase centering)
    are precomputed once, so calling the estimator per frame only costs Q + 1 FFTs plus
    one batched least-squares solve.

    Args:
        grid: Frame geometry.
        analysis: Analysis settings (window, order Q, DDM bins).
    """

    def __init__(self, grid: FrameGrid, analysis: AnalysisConfig) -> None:
        self.grid = grid
        self.order = analysis.poly_order
        # Number of bins on each side of a peak (peak included) entering the DDM system.
        self.num_bins = analysis.resolved_ddm_bins
        self.window, self.window_derivative = get_window(analysis.window, grid.window_length)

        n_dft = grid.n_dft
        self.n = grid.offsets
        self.basis = polynomial_basis(self.n, self.order)  # (N, Q+1), MATLAB p
        i = np.arange(1, self.order + 1)
        self.basis_derivative = i * np.power(self.n[:, None], i - 1)  # (N, Q), MATLAB pD
        self.omega = 2.0 * np.pi * np.arange(n_dft // 2 + 1) / n_dft
        # Shifts the DFT phase reference from the first frame sample to the origin n = 0.
        self.centering = np.exp(1j * self.omega * grid.origin)
        mid = int(matlab_round(grid.window_length / 2)) - 1
        self._check_rows = self.basis[[0, mid, grid.window_length - 1]]
        self.amplitude_guard = analysis.amplitude_guard

    def __call__(self, frame: np.ndarray, threshold_db: float) -> FrameEstimate:
        """Estimates the model parameters of every peak in one frame.

        Args:
            frame: N samples of the (padded) signal.
            threshold_db: Absolute peak-picking threshold in dB.

        Returns:
            The accepted estimates and the frame's spectrum.
        """
        spectra = self._ddm_spectra(frame)
        spectrum = spectra.spectrum
        peaks = pick_peaks(spectrum[: self.grid.n_dft // 2], threshold_db)
        if len(peaks) == 0:
            return FrameEstimate(np.zeros((0, self.order + 1), complex), peaks.bin, spectrum, peaks)

        alpha = np.zeros((len(peaks), self.order + 1), dtype=complex)
        alpha[:, 1:], solved = self._solve_slopes(spectra, peaks)
        valid = solved & self._is_stable(alpha)
        alpha[valid, 0] = self._estimate_alpha0(alpha[valid], spectra.weighted[peaks.bin[valid], 0], peaks.bin[valid])
        valid &= self._within_amplitude_bound(alpha, frame)
        return FrameEstimate(alpha[valid], peaks.bin[valid], spectrum, peaks)

    def _ddm_spectra(self, frame: np.ndarray) -> _DdmSpectra:
        """DFTs of the frame against the window, its derivative and the polynomial terms.

        All Q + 1 transforms are computed by one real FFT call over the stacked signals.
        Only bins 0..n_dft/2 are kept: peaks are searched below Nyquist.

        Args:
            frame: Raw frame samples x(n).
        """
        windowed = self.window * frame
        signals = np.column_stack([windowed[:, None] * self.basis_derivative, self.window_derivative * frame])
        raw = np.fft.rfft(signals, self.grid.n_dft, axis=0)
        centred = raw * self.centering[:, None]
        weighted = centred[:, : self.order]
        # Window-derivative DFT plus the DFT-kernel derivative term: together they form
        # -(right-hand side) of the DDM equations.
        derivative = centred[:, self.order] + weighted[:, 0] * (-1j * self.omega)
        return _DdmSpectra(raw[:, 0], weighted, derivative)

    def _solve_slopes(self, spectra: _DdmSpectra, peaks: Peaks) -> tuple[np.ndarray, np.ndarray]:
        """Solves the DDM systems A alpha_{1..Q} = b of all peaks in one batch.

        Returns:
            (alpha_1..Q per peak with shape (P, Q), mask of systems that could be solved).
        """
        rows, mask, num_rows = self._bin_rows(peaks)
        a = spectra.weighted[rows] * mask[..., None]  # (P, rows, Q)
        b = -spectra.derivative[rows] * mask  # (P, rows)
        return _least_squares(a, b, solvable=num_rows >= self.order)

    def _bin_rows(self, peaks: Peaks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """DFT bins entering each peak's DDM system.

        For peak j the rows are the peak bin followed by the bins
        max(left, peak - (R-1)) .. min(right, peak + (R-1), n_dft/2 - 1) in descending
        order (the peak bin appears twice, as in MATLAB). Shorter systems are padded with
        masked zero rows, which leaves their least-squares solutions unchanged.

        Returns:
            (rows (P, max_rows) bin indices, mask of used rows, number of used rows per peak).
        """
        span = self.num_bins - 1
        lo = np.maximum(peaks.left, peaks.bin - span)
        hi = np.minimum(np.minimum(peaks.right, peaks.bin + span), self.grid.n_dft // 2 - 1)
        width = hi - lo + 1
        # At least Q rows so every R factor is square; extra rows are zero padding.
        k = np.arange(max(1 + int(width.max()), self.order) - 1)
        used = k[None, :] < width[:, None]
        rows = np.concatenate([peaks.bin[:, None], np.where(used, hi[:, None] - k[None, :], 0)], axis=1)
        mask = np.concatenate([np.ones((len(peaks), 1), bool), used], axis=1)
        return rows, mask, width + 1

    def _is_stable(self, alpha: np.ndarray) -> np.ndarray:
        """Rejects non-finite slopes and models that explode at the frame edges."""
        with np.errstate(all="ignore"):
            edges = np.abs(alpha[:, 1:] @ self.basis[[0, -1], 1:].T)
        return np.all(np.isfinite(alpha[:, 1:]), axis=1) & np.all(edges <= _EDGE_LIMIT, axis=1)

    def _estimate_alpha0(self, alpha: np.ndarray, peak_value: np.ndarray, peak_bin: np.ndarray) -> np.ndarray:
        """Estimates alpha_0 from the peak bin (jun_ddm.m's single-bin approximation).

        With alpha_1..Q known, the model's DFT at the peak frequency omega_b is
        sum_n w(n) exp(sum_{i>=1} alpha_i n**i - j omega_b n); alpha_0 is the log-ratio of
        the measured peak value to it. The DFT kernel is folded into the exponent so only
        one complex exponential is evaluated per sample and peak.
        """
        with np.errstate(all="ignore"):
            exponent = self.basis[:, 1:] @ alpha[:, 1:].T  # (N, P)
            exponent -= 1j * self.n[:, None] * self.omega[peak_bin][None, :]
            model_value = self.window @ complex_exp(exponent)
            return np.log(peak_value) - np.log(model_value)

    def _within_amplitude_bound(self, alpha: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Rejects estimates whose amplitude is implausibly large.

        ``"center"``: the real sinusoid's amplitude at n = 0, 2 exp(Re alpha_0), must not
        exceed twice the largest sample, i.e. Re alpha_0 <= log max|x|. Only the centre is
        checked because the window vanishes at the edges, where the polynomial is
        unconstrained.

        ``"legacy"``: the original jun_ddm.m check, which compares the log-amplitude at the
        edges and centre with twice the largest *linear* sample value (a loose bound kept
        for reproducing results of the MATLAB code before the fix).
        """
        with np.errstate(all="ignore"):
            if self.amplitude_guard == "center":
                return alpha[:, 0].real <= np.log(np.max(np.abs(frame)))
            checks = alpha.real @ self._check_rows.T
        return np.all(checks <= 2.0 * np.max(np.abs(frame)), axis=1)


def _least_squares(a: np.ndarray, b: np.ndarray, solvable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched complex least squares via QR (Householder, like MATLAB's backslash).

    Args:
        a: System matrices, shape (P, rows, Q).
        b: Right-hand sides, shape (P, rows).
        solvable: Systems with enough rows to be determined.

    Returns:
        (solutions with NaN for unsolvable systems, mask of solved systems).
    """
    q_mat, r_mat = np.linalg.qr(a)
    rhs = np.einsum("prq,pr->pq", q_mat.conj(), b)
    solvable = solvable & np.all(np.abs(np.diagonal(r_mat, axis1=1, axis2=2)) > 0, axis=1)
    x = np.full(rhs.shape, np.nan + 0j)
    if np.any(solvable):
        x[solvable] = _back_substitute(r_mat[solvable], rhs[solvable])
    return x, solvable


def _back_substitute(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solves the batched upper-triangular systems r @ x = y.

    Args:
        r: Shape (P, Q, Q), upper triangular.
        y: Shape (P, Q).

    Returns:
        x with shape (P, Q).
    """
    q = r.shape[-1]
    x = np.zeros_like(y)
    for i in range(q - 1, -1, -1):
        x[:, i] = (y[:, i] - np.einsum("pj,pj->p", r[:, i, i + 1 :], x[:, i + 1 :])) / r[:, i, i]
    return x
