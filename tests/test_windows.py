from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking.windows import blackman_harris, get_window, hann

from .conftest import load_mat

REF = load_mat("windows")


@pytest.mark.parametrize("length", [int(n) for n in REF["lengths"]])
@pytest.mark.parametrize(("name", "func"), [("hann", hann), ("bh", blackman_harris)])
def test_windows_match_matlab(name: str, func, length: int) -> None:
    ref = REF[f"{name}_{length}"]
    np.testing.assert_allclose(func(length), ref[:, 0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(func(length, derivative=True), ref[:, 1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", ["hann", "blackman_harris"])
def test_derivative_matches_finite_difference(name: str) -> None:
    w, dw = get_window(name, 1023)
    np.testing.assert_allclose(np.gradient(w), dw, atol=1e-5)


def test_unknown_window() -> None:
    with pytest.raises(ValueError):
        get_window("kaiser", 16)
