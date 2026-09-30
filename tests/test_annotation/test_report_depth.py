"""Tests for the cortical-depth item's grouping (``report_depth``; H12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merxen.annotation.report_depth import DEEP_NP_CT_6B, ORDER, group_labels


def test_group_labels_merge_near_projecting_and_ct_6b_confident_cells() -> None:
    table = pd.DataFrame(
        {
            "ct_supercluster_name": [
                "Upper-layer intratelencephalic",
                "Deep-layer near-projecting",
                "Deep-layer corticothalamic and 6b",
                "Deep-layer intratelencephalic",
                "Deep-layer near-projecting",
            ],
            "ct_supercluster_status": ["confident"] * 4 + ["low_confidence"],
        }
    )
    groups = group_labels(table)
    np.testing.assert_array_equal(
        groups,
        [ORDER[0], DEEP_NP_CT_6B, DEEP_NP_CT_6B, ORDER[1], ""],
    )
    assert ORDER == (
        "Upper-layer intratelencephalic",
        "Deep-layer intratelencephalic",
        DEEP_NP_CT_6B,
    )
