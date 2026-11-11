from __future__ import annotations

import numpy as np
import pytest

from fast_partial_tracking import Synthesizer, rsnr, synthesize, track
from fast_partial_tracking.synthesis import cubic_phase_segments

from .conftest import MatlabCase


def test_cubic_phase_reproduces_linear_chirp() -> None:
    hop = 256
    w1, w2 = 0.1, 0.12
    t = np.arange(hop + 1)
    true_phase = 0.3 + w1 * t + (w2 - w1) / (2 * hop) * t**2
    segment = cubic_phase_segments(
        np.array([[w1, w2]]), np.array([[0.5, 0.5]]), np.array([[true_phase[0], true_phase[-1]]]), hop
    )
    np.testing.assert_allclose(segment[0], np.cos(true_phase[:hop]), atol=1e-9)


def test_phase_unwrapping_handles_wrapped_endpoint() -> None:
    hop = 128
    w = 0.3
    wrapped_end = np.angle(np.exp(1j * (1.0 + w * hop)))
    segment = cubic_phase_segments(np.array([[w, w]]), np.array([[0.5, 0.5]]), np.array([[1.0, wrapped_end]]), hop)
    np.testing.assert_allclose(segment[0], np.cos(1.0 + w * np.arange(hop)), atol=1e-9)


def test_streaming_synthesis_equals_batch(demo_case: MatlabCase) -> None:
    partials = track(demo_case.x, demo_case.fs, demo_case.analysis, demo_case.tracking)
    synth = Synthesizer(partials.grid)
    blocks = [synth.push(frame) for frame in partials.frames()]
    blocks.append(synth.finish(partials.grid.padded_length))
    padded = np.concatenate(blocks)
    assert padded.size == partials.grid.padded_length
    np.testing.assert_array_equal(padded, synthesize(partials, trim=False))
    assert rsnr(demo_case.x, synthesize(partials)) == pytest.approx(float(demo_case["rsnr"]), abs=1e-6)


def test_frames_must_be_consecutive(demo_case: MatlabCase) -> None:
    partials = track(demo_case.x, demo_case.fs, demo_case.analysis, demo_case.tracking)
    synth = Synthesizer(partials.grid)
    with pytest.raises(ValueError):
        synth.push(partials.get_frame(3))


def test_rsnr() -> None:
    x = np.array([1.0, -1.0, 2.0])
    assert rsnr(x, x) == np.inf
    assert rsnr(x, 0.9 * x) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        rsnr(x, x[:2])
