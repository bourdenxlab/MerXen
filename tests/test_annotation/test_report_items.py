"""Unit tests of the report item helpers (``report_items``; plan §9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import report_items as ri
from merxen.annotation.report_inputs import ReportInputs, ReportSources, SampleData
from merxen.annotation.schema import safe_token
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, load_vocab


def _sample(labels: pd.DataFrame, summary: dict | None = None) -> SampleData:
    return SampleData(
        sample_id="S_MERSCOPE",
        platform="MERSCOPE",
        labels=labels,
        summary=summary or {},
        manifest={},
        labels_path=Path("labels.parquet"),
    )


def test_validated_share_rows_count_confident_labels_only() -> None:
    labels = pd.DataFrame(
        {
            "in_table": [True] * 5,
            "ct_broad_name": ["Neurons", "Neurons", "Neurons", "Astrocytes", "Neurons"],
            "ct_broad_status": ["confident"] * 4 + ["low_confidence"],
            "ct_broad_validated": [True, False, True, True, True],
        }
    )
    rows = {
        row["label"]: row
        for row in ri.validated_share_rows(_sample(labels), labels, "human")
    }
    assert rows["Neurons"]["n_confident"] == 3
    assert rows["Neurons"]["validated_share"] == pytest.approx(2 / 3)
    assert rows["Astrocytes"]["validated_share"] == 1.0


def test_whb_and_seaad_broad_names_follow_the_vocabularies() -> None:
    table = pd.DataFrame(
        {
            "mmc_whb_supercluster_name": [
                "Astrocyte",
                "Splatter",
                None,
                "Committed oligodendrocyte precursor",
            ],
            "mmc_seaad_subclass_name": [
                "Astrocyte",
                "VLMC & Perivascular",
                "VLMC & Perivascular",
                None,
            ],
            "mmc_seaad_supertype_name": ["Astro_1", "VLMC_1", "Pericyte_1", None],
        }
    )
    assert list(ri.whb_broad_names(table)) == [
        "Astrocytes",
        "Mixed/Unknown",
        "Mixed/Unknown",
        "Oligodendrocyte precursors",
    ]
    assert list(ri.seaad_broad_names(table)) == [
        "Astrocytes",
        "Fibroblasts",
        "Vascular cells",
        "Mixed/Unknown",
    ]
    assert list(
        ri.lineage_of(np.array(["Oligodendrocyte precursors", "Neurons", "x"]))
    ) == [
        "Oligodendrocyte lineage",
        "Neurons",
        "Mixed/Unknown",
    ]


def test_names_array_finds_a_column_case_insensitively() -> None:
    table = pd.DataFrame({"mmc_seaad_Subclass_name": ["a", None]})
    assert list(ri.names_array(table, "mmc_seaad_subclass_name")) == ["a", ""]
    assert list(ri.names_array(table, "absent")) == ["", ""]


def test_soft_matrix_appends_the_mouse_residual() -> None:
    classes = load_vocab("wmb_class").names
    frame = pd.DataFrame(
        0.0, index=range(2), columns=[f"soft_class_{safe_token(c)}" for c in classes]
    )
    frame.iloc[0, 0] = 0.7
    frame.iloc[1, 1] = 1.0
    matrix = ri.soft_matrix(frame, "mouse")
    assert matrix.shape == (2, len(classes) + 1)
    np.testing.assert_allclose(matrix[:, -1], [0.3, 0.0])
    human = pd.DataFrame(
        {f"soft_broad_{safe_token(c)}": [1.0 / 8] for c in HUMAN_BROAD_CLASSES}
        | {"soft_broad_unallocated": [1.0 / 8]}
    )
    np.testing.assert_allclose(ri.soft_matrix(human, "human").sum(axis=1), 1.0)
    assert ri.soft_matrix(pd.DataFrame({"x": [1]}), "human") is None


def test_depth_matched_ratio_weights_bins_by_real_cells() -> None:
    real = np.array([0.4, 0.4, 0.4, 0.8])
    real_bins = np.array([10, 10, 10, 60])
    sim = np.array([0.8, 0.8, 0.8, 0.8])
    sim_bins = np.array([10, 10, 60, 60])
    ratio, n = ri.depth_matched_ratio(real, real_bins, sim, sim_bins)
    assert ratio == pytest.approx((3 * 0.5 + 1 * 1.0) / 4)
    assert n == 4
    none, zero = ri.depth_matched_ratio(real, real_bins, sim, np.array([250] * 4))
    assert np.isnan(none) and zero == 0


def test_emission_metrics_use_the_resolve_regime_and_skip_off_levels() -> None:
    emission = pd.DataFrame(
        {
            "reference_id": ["whb"] * 6,
            "level": ["broad"] * 4 + ["cluster"] * 2,
            "class": ["Astro"] * 6,
            "depth": [10, 30, 10, 30, 10, 30],
            "regime": [
                "validated",
                "validated",
                "provisional",
                "provisional",
                "validated",
                "validated",
            ],
            "status": [
                "emitted",
                "not_resolvable",
                "emitted",
                "emitted",
                "emitted",
                "emitted",
            ],
            "t_star": [0.8, 0.73, 0.95, 0.95, 0.99, 0.99],
            "default_threshold": [0.73] * 6,
            "emitted_reweighted_S_MERSCOPE": [True, True, False, False, False, False],
        }
    )
    summary = {
        "resolution": {"levels": {"broad": {"emission": {"regime": "validated"}}}}
    }
    sample = _sample(pd.DataFrame({"in_table": [True]}), summary)
    inputs = ReportInputs(
        sources=ReportSources(
            species="human", pair_id="S", segmentation="seg", resolve_dir=Path(".")
        ),
        summary={},
        samples={"S_MERSCOPE": sample},
    )
    produced = ri.emission_metrics(emission, inputs)
    # Only broad: the cluster level has no RESOLVE regime (off in v1).
    assert {record.level for record in produced} == {"broad"} and len(produced) == 3
    records = {record.name: record for record in produced}
    assert set(records) == {
        "prep_bins_emitted",
        "reweighted_bins_emitted",
        "thresholds_raised_by_local_rule",
    }
    assert (
        records["prep_bins_emitted"].value == 1 and records["prep_bins_emitted"].n == 2
    )
    assert records["reweighted_bins_emitted"].value == 2
    assert records["thresholds_raised_by_local_rule"].value == 1
    assert records["prep_bins_emitted"].kind == "whb:validated"
    assert all(record.criterion == "H18" for record in records.values())
