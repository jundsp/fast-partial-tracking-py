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

from fast_partial_tracking.config import AnalysisConfig, matlab_round
from fast_partial_tracking.framing import FrameGrid
from fast_partial_tracking.model import polynomial_basis
from fast_partial_tracking.peaks import Peaks, pick_peaks
from fast_partial_tracking.windows import get_window

log = logging.getLogger(__name__)

# Guards against numerically exploding estimates; same limit as jun_ddm.m.
_EDGE_LIMIT = 1e100


@dataclass(frozen=True)
class FrameEstimate:
    """DDM estimates for one analysis frame.

    Attributes:
        alpha: Complex coefficients of the accepted peaks, shape (R, Q + 1), ordered by
            ascending frequency bin.
        peak_bin: DFT bin of each accepted peak, shape (R,).
        spectrum: Windowed DFT of the frame, shape (n_dft,).
        peaks: All peaks found before the validity checks.
    """

    alpha: np.ndarray
    peak_bin: np.ndarray
    spectrum: np.ndarray
    peaks: Peaks

    def __len__(self) -> int:
        return int(self.alpha.shape[0])


class DdmEstimator:
    """Estimates the short-term model (eq. 4) for every spectral peak of a frame.

    All frame-independent quantities (windows, polynomial bases, DFT phase centering)
    are precomputed once, so calling the estimator per frame only costs Q + 1 FFTs plus
    one batched least-squares solve.

    Args:
        grid: Frame geometry.
        analysis: Analysis settings (window, oversampling, order Q).
    """

    def __init__(self, grid: FrameGrid, analysis: AnalysisConfig) -> None:
        self.grid = grid
        self.order = analysis.effective_order
        # Number of bins on each side of a peak that may enter the DDM system.
        self.num_bins = self.order * analysis.effective_oversample
        self.window, self.window_derivative = get_window(analysis.window_name, grid.window_length)

        n_dft = grid.n_dft
        self.n = grid.offsets
        self.basis = polynomial_basis(self.n, self.order)  # (N, Q+1), MATLAB p
        i = np.arange(1, self.order + 1)
        self.basis_derivative = i * np.power(self.n[:, None], i - 1)  # (N, Q), MATLAB pD
        self.omega = 2.0 * np.pi * np.arange(n_dft) / n_dft
        # Shifts the DFT phase reference from the first frame sample to the origin n = 0.
        self.centering = np.exp(1j * self.omega * grid.origin)
        mid = int(matlab_round(grid.window_length / 2)) - 1
        self._check_rows = self.basis[[0, mid, grid.window_length - 1]]

    def __call__(self, frame: np.ndarray, threshold_db: float) -> FrameEstimate:
        """Estimates the model parameters of every peak in one frame.

        Args:
            frame: N samples of the (padded) signal.
            threshold_db: Absolute peak-picking threshold in dB.

        Returns:
            The accepted estimates and the frame's spectrum.
        """
        n_dft = self.grid.n_dft
        q = self.order
        windowed = self.window * frame
        spectrum = np.fft.fft(windowed, n_dft)
        peaks = pick_peaks(spectrum[: n_dft // 2], threshold_db)
        if len(peaks) == 0:
            return FrameEstimate(np.zeros((0, q + 1), complex), peaks.bin, spectrum, peaks)

        # Columns of the DDM system: DFTs of the windowed signal multiplied by the
        # polynomial derivatives i*n**(i-1), i = 1..Q (the first is the plain DFT).
        weighted = np.empty((n_dft, q), dtype=complex)
        weighted[:, 0] = spectrum * self.centering
        if q > 1:
            weighted[:, 1:] = np.fft.fft(windowed[:, None] * self.basis_derivative[:, 1:], n_dft, axis=0)
            weighted[:, 1:] *= self.centering[:, None]
        # DFT of the signal against the window derivative, plus the DFT-kernel derivative
        # term; together they form -(right-hand side) of the DDM equations.
        derivative = np.fft.fft(self.window_derivative * frame, n_dft) * self.centering
        derivative += weighted[:, 0] * (-1j * self.omega)

        alpha_tail, rows_valid = self._solve(weighted, derivative, peaks)
        alpha = np.zeros((len(peaks), q + 1), dtype=complex)
        alpha[:, 1:] = alpha_tail

        valid = rows_valid & np.all(np.isfinite(alpha_tail), axis=1)
        with np.errstate(all="ignore"):
            edges = np.abs(alpha_tail @ self.basis[[0, -1], 1:].T)
        valid &= np.all(edges <= _EDGE_LIMIT, axis=1)
        if np.any(~valid):
            log.debug(f"skipping {np.count_nonzero(~valid)} peak(s) with non-finite DDM estimates")

        alpha[valid, 0] = self._alpha0(alpha[valid], weighted[peaks.bin[valid], 0], peaks.bin[valid])
        with np.errstate(all="ignore"):
            checks = alpha.real @ self._check_rows.T
        # Same test as jun_ddm.m: log-amplitude at the frame edges and centre compared
        # against twice the frame's peak sample value.
        valid &= np.all(checks <= 2.0 * np.max(np.abs(frame)), axis=1)

        return FrameEstimate(alpha[valid], peaks.bin[valid], spectrum, peaks)

    def _solve(self, weighted: np.ndarray, derivative: np.ndarray, peaks: Peaks) -> tuple[np.ndarray, np.ndarray]:
        """Solves the per-peak DDM least-squares systems in one batch.

        For peak j the rows are the peak bin followed by the bins
        max(left, peak - (R-1)) .. min(right, peak + (R-1), n_dft/2 - 1) in descending
        order (the peak bin appears twice, as in MATLAB). Systems with fewer rows are
        padded with zero rows, which leaves their least-squares solutions unchanged.

        Returns:
            (alpha_1..Q per peak with shape (P, Q), mask of systems that could be solved).
        """
        q = self.order
        span = self.num_bins - 1
        lo = np.maximum(peaks.left, peaks.bin - span)
        hi = np.minimum(np.minimum(peaks.right, peaks.bin + span), self.grid.n_dft // 2 - 1)
        width = hi - lo + 1
        # At least Q rows so every R factor is square; extra rows are zero padding.
        max_rows = max(1 + int(width.max()), q)
        k = np.arange(max_rows - 1)
        side_rows = hi[:, None] - k[None, :]
        used = k[None, :] < width[:, None]
        rows = np.concatenate([peaks.bin[:, None], np.where(used, side_rows, 0)], axis=1)
        mask = np.concatenate([np.ones((len(peaks), 1), bool), used], axis=1)

        a = weighted[rows] * mask[..., None]  # (P, rows, Q)
        b = -derivative[rows] * mask  # (P, rows)
        q_mat, r_mat = np.linalg.qr(a)
        rhs = np.einsum("prq,pr->pq", q_mat.conj(), b)
        solvable = (width + 1 >= q) & np.all(np.abs(np.diagonal(r_mat, axis1=1, axis2=2)) > 0, axis=1)
        x = np.full((len(peaks), q), np.nan + 0j)
        if np.any(solvable):
            x[solvable] = _back_substitute(r_mat[solvable], rhs[solvable])
        return x, solvable

    def _alpha0(self, alpha: np.ndarray, peak_value: np.ndarray, peak_bin: np.ndarray) -> np.ndarray:
        """Estimates alpha_0 from the peak bin (jun_ddm.m's single-bin approximation).

        With alpha_1..Q known, the DFT of w(n) exp(sum_{i>=1} alpha_i n**i) at the peak
        bin is computed directly; alpha_0 is the log-ratio of the measured peak value to
        that model DFT value.
        """
        with np.errstate(all="ignore"):
            gamma = self.window[:, None] * np.exp(self.basis[:, 1:] @ alpha[:, 1:].T)  # (N, P)
            kernel = np.exp(-1j * self.n[:, None] * self.omega[peak_bin][None, :])
            model_value = np.sum(kernel * gamma, axis=0)
            return np.log(peak_value) - np.log(model_value)


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
