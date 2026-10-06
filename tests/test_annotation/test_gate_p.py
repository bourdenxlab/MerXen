"""Tests for gate-P scoring (M13; plan §14 new panel family, NP4 first)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merxen.annotation import resolvability as res

from .test_resolvability import BROAD, bin_cells, settings


def _decisions_at_10() -> pd.DataFrame:
    """Frozen decisions: the provisional regime emits broad X at 10 counts (0.70)."""
    return res.decide(
        bin_cells(np.full(400, 0.95), np.ones(400, dtype=bool), depth=10),
        [BROAD],
        [10],
        settings(),
    )


# --------------------------------------------------------------------------
# The frozen confident-call mask (shared by gate_p_tested_sets and NP3-NP7)


def test_frozen_confident_mask_needs_a_call_an_emitted_bin_and_the_threshold() -> None:
    lookup = res.emission_lookup(_decisions_at_10(), "provisional")
    assert lookup[("broad", "X", 10)][:2] == (res.STATUS_EMITTED, 0.70)
    lookup[("broad", "W", 10)] = (res.STATUS_NOT_RESOLVABLE, 0.70, False)
    lookup[("broad", "V", 10)] = (res.STATUS_EMITTED, None, False)
    frame = pd.DataFrame(
        {
            "level": "broad",
            "parent": ["X", "X", "X", "X", None, "Y", "W", "V", "X"],
            "depth": [10, 10, 10, 10, 10, 10, 10, 10, 30],
            "bp": [0.95, 0.70 - 5e-10, 0.69, np.nan, 0.99, 0.99, 0.99, 0.99, 0.99],
        }
    )
    mask = res.frozen_confident_mask(frame, lookup)
    assert mask.dtype == bool
    # At the threshold within 1e-9 counts; below it, a NaN bp, no call, a
    # class without a bin, a withheld bin, a bin without a threshold and a
    # depth the decisions never saw do not.
    assert mask.tolist() == [True, True] + [False] * 7
    assert res.frozen_confident_mask(frame.iloc[:0], lookup).shape == (0,)
