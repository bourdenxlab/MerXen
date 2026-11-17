"""Tests of the image evidence module (channel roles, representations, checks)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.test_manual_registration._scripts import load_script

from merxen.manual_registration import image_evidence as ie

derive_script = load_script("derive_test_landmarks")


def _blob(shape: int, centre_xy: tuple[float, float], sigma: float) -> np.ndarray:
    rows, columns = np.mgrid[:shape, :shape]
    return np.asarray(
        np.exp(
            -((columns - centre_xy[0]) ** 2 + (rows - centre_xy[1]) ** 2)
            / (2 * sigma**2)
        ),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Helpers moved from the test-landmark script
# ---------------------------------------------------------------------------


def test_script_aliases_are_the_package_functions() -> None:
    assert derive_script._bandpass is ie.bandpass_log
    assert derive_script._parabolic_peak is ie.parabolic_peak
    assert derive_script._match_window is ie.match_window


def test_a_peak_on_the_search_border_is_not_confident() -> None:
    search = 20
    window = _blob(60, (30.0, 30.0), 12.0)

    inside = _blob(100, (50.0 + 10.0, 50.0), 12.0)
    peak, second, x, y, on_border = ie.match_window(inside, window, exclusion_px=8)
    assert not on_border and peak > 0.9 and second < peak
    assert x - search == pytest.approx(10.0, abs=0.1)
    assert y - search == pytest.approx(0.0, abs=0.1)

    # 35 px to the right is beyond the +-20 px search: the best correlation within
    # the search is a high one on its border, which says nothing about the match.
    beyond = _blob(100, (50.0 + 35.0, 50.0), 12.0)
    peak, _, x, _, on_border = ie.match_window(beyond, window, exclusion_px=8)
    assert on_border and peak > 0.3
    assert x == 2 * search


@pytest.mark.parametrize(("shift_x", "shift_y"), [(10.4, -3.3), (-7.25, 5.6)])
def test_match_window_recovers_a_fractional_shift(
    shift_x: float, shift_y: float
) -> None:
    search = 20
    window = _blob(60, (30.0, 30.0), 12.0)
    region = _blob(100, (50.0 + shift_x, 50.0 + shift_y), 12.0)

    peak, _, x, y, on_border = ie.match_window(region, window, exclusion_px=8)

    assert not on_border and peak > 0.9
    assert x - search == pytest.approx(shift_x, abs=0.05)
    assert y - search == pytest.approx(shift_y, abs=0.05)


def test_parabolic_peak_is_the_vertex_of_a_sampled_parabola() -> None:
    samples = 5.0 - (np.arange(8) - 3.3) ** 2

    assert ie.parabolic_peak(samples, 3) == pytest.approx(3.3)
    assert ie.parabolic_peak(samples[::-1], 4) == pytest.approx(7 - 3.3)
    assert ie.parabolic_peak(samples, 0) == 0.0
    assert ie.parabolic_peak(samples, 7) == 7.0
    assert ie.parabolic_peak(np.ones(5), 2) == 2.0
