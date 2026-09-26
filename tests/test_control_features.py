"""Tests for the platform control-feature registry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merxen.analysis import clustering_squidpy
from merxen.control_features import (
    CONTROL_TOKENS,
    control_token_mask,
    has_control_token,
)


def test_control_tokens_are_shared_with_clustering() -> None:
    """Clustering keeps importing the registry's token tuple."""
    assert clustering_squidpy.CONTROL_TOKENS is CONTROL_TOKENS


def test_has_control_token_is_case_insensitive_substring() -> None:
    """Token matching keeps the legacy case-insensitive substring rule."""
    assert has_control_token("Blank-12")
    assert has_control_token("NegControlProbe_00002")
    assert has_control_token("control_probe_counts")
    assert not has_control_token("GFAP")
    assert not has_control_token("HLA-DMB")


def test_control_token_mask_matches_scalar_rule_and_handles_missing() -> None:
    """The vectorised mask equals the scalar rule; missing names never match."""
    values = pd.Series(
        ["GeneA", "BLANK_0006", None, "UnassignedCodeword_0003", "GeneA", np.nan]
    )

    mask = control_token_mask(values)

    assert mask.tolist() == [False, True, False, True, False, False]
    assert control_token_mask([]).tolist() == []
