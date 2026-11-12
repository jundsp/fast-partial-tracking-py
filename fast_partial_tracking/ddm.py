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


@dataclass(frozen=True)
class _DdmSpectra:
    """DFTs that form the DDM linear system of one frame (all phase-centred on n = 0).

    Attributes:
        weighted: Column i holds DFT{w(n) x(n) (i+1) n**i}, i = 0..Q-1, shape (n_dft, Q).
        derivative: DFT{w'(n) x(n)} - j omega DFT{w(n) x(n)}, shape (n_dft,).
    """

    weighted: np.ndarray
    derivative: np.ndarray


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
        windowed = self.window * frame
        spectrum = np.fft.fft(windowed, self.grid.n_dft)
        peaks = pick_peaks(spectrum[: self.grid.n_dft // 2], threshold_db)
        if len(peaks) == 0:
            return FrameEstimate(np.zeros((0, self.order + 1), complex), peaks.bin, spectrum, peaks)

        spectra = self._ddm_spectra(frame, windowed, spectrum)
        alpha = np.zeros((len(peaks), self.order + 1), dtype=complex)
        alpha[:, 1:], solved = self._solve_slopes(spectra, peaks)
        valid = solved & self._is_stable(alpha)
        alpha[valid, 0] = self._estimate_alpha0(alpha[valid], spectra.weighted[peaks.bin[valid], 0], peaks.bin[valid])
        valid &= self._within_amplitude_bound(alpha, frame)
        return FrameEstimate(alpha[valid], peaks.bin[valid], spectrum, peaks)

    def _ddm_spectra(self, frame: np.ndarray, windowed: np.ndarray, spectrum: np.ndarray) -> _DdmSpectra:
        """DFTs of the frame against the window, its derivative and the polynomial terms.

        Args:
            frame: Raw frame samples x(n).
            windowed: w(n) x(n).
            spectrum: DFT of ``windowed``.
        """
        weighted = np.empty((self.grid.n_dft, self.order), dtype=complex)
        weighted[:, 0] = spectrum * self.centering
        if self.order > 1:
            weighted[:, 1:] = np.fft.fft(windowed[:, None] * self.basis_derivative[:, 1:], self.grid.n_dft, axis=0)
            weighted[:, 1:] *= self.centering[:, None]
        # Window-derivative DFT plus the DFT-kernel derivative term: together they form
        # -(right-hand side) of the DDM equations.
        derivative = np.fft.fft(self.window_derivative * frame, self.grid.n_dft) * self.centering
        derivative += weighted[:, 0] * (-1j * self.omega)
        return _DdmSpectra(weighted, derivative)

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

        With alpha_1..Q known, the DFT of w(n) exp(sum_{i>=1} alpha_i n**i) at the peak
        bin is computed directly; alpha_0 is the log-ratio of the measured peak value to
        that model DFT value.
        """
        with np.errstate(all="ignore"):
            gamma = self.window[:, None] * np.exp(self.basis[:, 1:] @ alpha[:, 1:].T)  # (N, P)
            kernel = np.exp(-1j * self.n[:, None] * self.omega[peak_bin][None, :])
            model_value = np.sum(kernel * gamma, axis=0)
            return np.log(peak_value) - np.log(model_value)

    def _within_amplitude_bound(self, alpha: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Same final test as jun_ddm.m.

        The log-amplitude at the frame edges and centre is compared with twice the
        largest sample value (a loose sanity bound that only rejects exploding models).
        """
        with np.errstate(all="ignore"):
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
