"""Tests for transcript table helpers."""

from __future__ import annotations

import pandas as pd

from merxen.io.transcript_io import assignment_mask, assignment_mask_from_points


def test_assignment_mask_treats_nullable_zero_as_assigned() -> None:
    """Nullable numeric assignment uses null, not zero, as unassigned."""
    series = pd.Series([0, 1, pd.NA], dtype="UInt32")

    mask = assignment_mask(series)

    assert mask.tolist() == [True, True, False]


def test_assignment_mask_keeps_legacy_numeric_zero_unassigned() -> None:
    """Dense numeric assignment columns still use zero as the unassigned code."""
    series = pd.Series([0, 1, 2], dtype="uint32")

    mask = assignment_mask(series)

    assert mask.tolist() == [False, True, True]


def test_assignment_mask_from_points_prefers_background_column() -> None:
    """ProSeg foreground status should come from ``background`` when present."""
    points = pd.DataFrame(
        {
            "assignment": pd.Series([0, pd.NA, 2], dtype="UInt32"),
            "background": [False, True, False],
        }
    )

    mask = assignment_mask_from_points(points, assign_col="assignment")

    assert mask.tolist() == [True, False, True]
