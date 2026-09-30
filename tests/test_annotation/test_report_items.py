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


def test_heldout_item_shows_the_h4_headline_label_set(tmp_path: Path) -> None:
    """Item 8 shows resolve_criteria's non-circular H4 headline set first."""
    rows = []
    for label_set in ("heldout_argmax", ri.HELDOUT_HEADLINE_SET, "m4_resolve_heldout"):
        for platform in ("MERSCOPE", "XENIUM"):
            rows.append(
                {
                    "pair": "S",
                    "platform": platform,
                    "broad_class": "Neurons",
                    "label_set": label_set,
                    "fold": 10.0 if label_set == ri.HELDOUT_HEADLINE_SET else 2.0,
                    "auroc": 0.9,
                    "n_assigned": 100,
                }
            )
    inputs = ReportInputs(
        sources=ReportSources(
            species="human", pair_id="S", segmentation="seg", resolve_dir=tmp_path
        ),
        summary={},
        samples={"S_MERSCOPE": _sample(pd.DataFrame({"in_table": [True]}))},
        heldout=pd.DataFrame(rows),
    )
    item = ri.item_heldout(
        inputs, ri.ItemWriter(tmp_path / "out", make_figures=False), ri.ReportOptions()
    )
    assert item.status == "ok"
    folds = [record for record in item.metrics if record.name == "heldout_fold"]
    assert {record.kind for record in folds} == {ri.HELDOUT_HEADLINE_SET}
    assert {record.value for record in folds} == {10.0} and len(folds) == 2
    # Every label set stays in the item's table.
    table = pd.read_csv(
        tmp_path / "out" / "tables" / "item08_heldout_genes__heldout.csv"
    )
    assert set(table["label_set"]) == {
        "heldout_argmax",
        ri.HELDOUT_HEADLINE_SET,
        "m4_resolve_heldout",
    }


COP = "Committed oligodendrocyte precursor"


def _composition_table(n_cop: int, n_opc: int, n_neuron: int) -> pd.DataFrame:
    """A human label table: COP, OPC and CGE neuron cells with soft columns.

    Every COP cell has bp 0.8 on COP and a 0.2 runner-up on OPC; every OPC
    cell 0.9 on OPC; every neuron 1.0 on the CGE supercluster. Three cells
    carry ``flag_implausible``.
    """
    n = n_cop + n_opc + n_neuron
    name = np.r_[
        np.repeat(COP, n_cop),
        np.repeat("Oligodendrocyte precursor", n_opc),
        np.repeat("CGE interneuron", n_neuron),
    ]
    bp = np.r_[np.full(n_cop, 0.8), np.full(n_opc, 0.9), np.full(n_neuron, 1.0)]
    runner = np.r_[
        np.repeat("Oligodendrocyte precursor", n_cop),
        np.repeat(None, n_opc + n_neuron),
    ]
    runner_bp = np.r_[np.full(n_cop, 0.2), np.zeros(n_opc + n_neuron)]
    frame = pd.DataFrame(
        {
            "cell_id": [f"c{index}" for index in range(n)],
            "in_table": np.ones(n, dtype=bool),
            "total_counts": np.full(n, 50.0),
            "mmc_whb_supercluster_name": name,
            "mmc_whb_supercluster_bp": bp,
            "mmc_whb_supercluster_runner_up_1_name": runner,
            "mmc_whb_supercluster_runner_up_1_bp": runner_bp,
            "ct_broad_name": np.where(
                name == "CGE interneuron", "Neurons", "Oligodendrocyte precursors"
            ),
            "ct_broad_status": np.where(
                np.arange(n) < n_cop - 2, "parent_unresolved", "confident"
            ),
            "ct_supercluster_name": name,
            "ct_supercluster_status": np.where(
                name == "CGE interneuron", "confident", "not_resolvable"
            ),
            "ct_final_level": np.where(np.arange(n) < n_cop - 2, "none", "broad"),
            "flag_implausible": np.arange(n) < 3,
        }
    )
    soft = {f"soft_broad_{safe_token(c)}": np.zeros(n) for c in HUMAN_BROAD_CLASSES}
    soft["soft_broad_oligodendrocyte_precursors"] = np.r_[
        np.ones(n_cop), np.full(n_opc, 0.9), np.zeros(n_neuron)
    ]
    soft["soft_broad_neurons"] = np.r_[np.zeros(n_cop + n_opc), np.ones(n_neuron)]
    soft["soft_broad_unallocated"] = np.r_[
        np.zeros(n_cop), np.full(n_opc, 0.1), np.zeros(n_neuron)
    ]
    return frame.assign(**soft)


def test_composition_reports_the_cop_part_of_opc_and_the_sink_prone_nodes(
    tmp_path: Path,
) -> None:
    """M7 review: soft OPC is mostly COP mass; the report must say so."""
    labels = _composition_table(n_cop=30, n_opc=10, n_neuron=60)
    broad_only = SampleData(
        sample_id="S_MERSCOPE",
        platform="MERSCOPE",
        labels=labels,
        summary={"resolution": {"gate": {"level": "broad_only"}}},
        manifest={},
        labels_path=Path("labels.parquet"),
    )
    full = SampleData(
        sample_id="S_XENIUM",
        platform="XENIUM",
        labels=labels.copy(),
        summary={"resolution": {"gate": {"level": "full"}}},
        manifest={},
        labels_path=Path("labels.parquet"),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    pd.DataFrame(
        {
            "level": ["CCN_SUPC"] * 4 + ["CCN_CLUS"],
            "node_name": [COP, "Oligodendrocyte precursor", "CGE interneuron"]
            + ["Upper-layer intratelencephalic", "x"],
            "n_cells": [1, 99, 100, 400, 7],
            "gene_id": ["g"] * 5,
        }
    ).to_parquet(bundle / "profiles.parquet")
    assert ri.reference_node_shares(bundle) == pytest.approx(
        {
            COP: 1 / 600,
            "Oligodendrocyte precursor": 99 / 600,
            "CGE interneuron": 100 / 600,
            "Upper-layer intratelencephalic": 400 / 600,
        }
    )
    inputs = ReportInputs(
        sources=ReportSources(
            species="human", pair_id="S", segmentation="seg", resolve_dir=tmp_path
        ),
        summary={},
        samples={"S_MERSCOPE": broad_only, "S_XENIUM": full},
        bundles={"whb_frontal_supc_clus": bundle},
    )
    item = ri.item_composition(
        inputs,
        ri.ItemWriter(tmp_path / "out", make_figures=False),
        ri.ReportOptions(n_bootstrap=10),
    )
    records = {
        (record.name, record.sample_id): record
        for record in item.metrics
        if record.criterion in ("H2", "H5")
    }
    # COP mass 30 x 0.8 = 24 of soft OPC mass 30 x 1.0 + 10 x 0.9 = 39.
    soft = records[("cop_derived_soft_opc_share", "S_MERSCOPE")]
    assert soft.value == pytest.approx(24 / 39) and soft.kind == "soft"
    argmax = records[("cop_derived_argmax_opc_share", "S_MERSCOPE")]
    assert argmax.value == pytest.approx(30 / 40) and argmax.kind == "argmax"
    # H2: three implausible cells of 100 (not the complement).
    assert records[("flag_implausible_share", "S_MERSCOPE")].value == pytest.approx(
        0.03
    )
    composition = pd.read_csv(
        tmp_path / "out" / "tables" / "item02_composition__composition.csv"
    )
    opc = composition[
        (composition["category"] == "Oligodendrocyte precursors")
        & (composition["kind"] == "soft")
        & (composition["level"] == "broad")
        & (composition["sample_id"] == "S_MERSCOPE")
    ]
    assert opc["cop_derived_share"].iloc[0] == pytest.approx(24 / 100)
    assert opc["share"].iloc[0] == pytest.approx(39 / 100)
    others = composition[composition["category"] != "Oligodendrocyte precursors"]
    assert others["cop_derived_share"].isna().all()
    # Leaf-level rows of the broad-only sample are withheld for comparison.
    leaf = composition[composition["level"] == "supercluster"]
    status = leaf.groupby("sample_id")["comparison_status"].unique()
    assert list(status["S_MERSCOPE"]) == ["withheld_for_comparison"]
    assert list(status["S_XENIUM"]) == ["comparable"]
    assert any("withheld for comparison" in note for note in item.notes)
    sinks = pd.read_csv(
        tmp_path / "out" / "tables" / "item02_composition__sinks_implausible.csv"
    )
    merscope = sinks[sinks["sample_id"] == "S_MERSCOPE"].set_index("node")
    assert {COP, "CGE interneuron"} <= set(merscope.index)
    assert merscope.loc[COP, "share_table"] == pytest.approx(0.30)
    assert merscope.loc[COP, "soft_mass_share"] == pytest.approx(0.24)
    assert merscope.loc[COP, "confident_share"] == 0.0
    assert merscope.loc["CGE interneuron", "confident_share"] == pytest.approx(0.6)
    assert "named_sink_prone" in merscope.loc[COP, "reasons"]
    # COP holds 75% of the argmax OPC-broad cells against 1% of the reference
    # OPC-broad cells: over the 3x ratio; the OPC node itself is under it.
    assert merscope.loc[COP, "argmax_over_reference_in_broad"] == pytest.approx(75.0)
    assert "argmax_over_reference_in_broad" in merscope.loc[COP, "reasons"]
    assert "Oligodendrocyte precursor" not in merscope.index
