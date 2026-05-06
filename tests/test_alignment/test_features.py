"""Tests for alignment feature extraction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from merxen.alignment.features import _robust_centroid_xy


class _FakePoints:
    def __init__(self: _FakePoints, x: list[float], y: list[float]) -> None:
        self.x = pd.Series(x, dtype=float)
        self.y = pd.Series(y, dtype=float)


class _FakeGeometry:
    def __init__(self: _FakeGeometry) -> None:
        self.centroid = _FakePoints([-9_879_796.0, 2.0], [-12_683_331.0, 3.0])
        self.bounds = pd.DataFrame(
            {
                "minx": [337.0, 1.0],
                "miny": [4857.0, 2.0],
                "maxx": [338.0, 3.0],
                "maxy": [4858.0, 4.0],
            }
        )

    def representative_point(self: _FakeGeometry) -> _FakePoints:
        return _FakePoints([337.5, 2.5], [4857.5, 3.5])


def test_robust_centroid_xy_falls_back_for_pathological_centroids() -> None:
    """Invalid warped polygons can report centroids far outside their bounds."""
    x, y = _robust_centroid_xy(SimpleNamespace(geometry=_FakeGeometry()))

    np.testing.assert_allclose(x, [337.5, 2.0])
    np.testing.assert_allclose(y, [4857.5, 3.0])
