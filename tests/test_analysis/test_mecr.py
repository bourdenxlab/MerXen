"""Tests for paper-standard mutually exclusive co-expression rate analysis."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.analysis.mecr import (
    compute_mecr_pair_metrics,
    discover_reference_markers,
    load_whb_panel_reference,
    plot_class_pair_mecr_heatmaps,
    plot_platform_mecr_comparison,
    plot_reference_mecr_histogram,
    run_mecr,
    select_barnyard_pairs,
    summarize_mecr_pair_metrics,
)
from merxen.config import MecrConfig, MecrReferenceConfig, MecrSampleConfig


def test_compute_mecr_pair_metrics_uses_binary_intersection_over_union() -> None:
    """Each cross-class pair should use detected-both divided by detected-either."""
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [3, 2, 0],
                    [1, 0, 0],
                    [0, 4, 0],
                    [0, 0, 0],
                ]
            )
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC"]),
    )
    markers = pd.DataFrame(
        {
            "gene": ["GeneA", "GeneB", "GeneC"],
            "broad_class": ["Astrocytes", "Neurons", "Astrocytes"],
        }
    )

    result = compute_mecr_pair_metrics(adata, markers)

    assert len(result) == 2
    gene_a_pair = result[
        (result["gene_1"] == "GeneA") & (result["gene_2"] == "GeneB")
    ].iloc[0]
    assert gene_a_pair["cells_gene_1"] == 2
    assert gene_a_pair["cells_gene_2"] == 2
    assert gene_a_pair["cells_both"] == 1
    assert gene_a_pair["cells_either"] == 3
    assert gene_a_pair["mecr"] == pytest.approx(1 / 3)
    assert not ((result["gene_1"] == "GeneA") & (result["gene_2"] == "GeneC")).any()


def test_compute_mecr_pair_metrics_records_zero_union_as_nan() -> None:
    """Pairs with neither marker detected should remain auditable but unscored."""
    adata = ad.AnnData(
        X=sparse.csr_matrix((3, 2), dtype=np.int64),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    markers = pd.DataFrame(
        {
            "gene": ["GeneA", "GeneB"],
            "broad_class": ["Astrocytes", "Neurons"],
        }
    )

    result = compute_mecr_pair_metrics(adata, markers)
    summary = summarize_mecr_pair_metrics(
        result,
        sample_id="sample",
        platform="XENIUM",
        segmentation="reseg",
        n_cells=3,
        n_panel_genes=2,
        n_reference_markers=2,
    )

    assert len(result) == 1
    assert result.iloc[0]["cells_either"] == 0
    assert np.isnan(result.iloc[0]["mecr"])
    assert summary["mecr"] is None
    assert summary["n_zero_union_pairs"] == 1


def test_run_mecr_writes_platform_and_combined_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage entry point should produce its complete reporting contract."""
    markers_path = tmp_path / "markers.csv"
    pd.DataFrame(
        {
            "gene": ["GeneA", "GeneB"],
            "broad_class": ["Astrocytes", "Neurons"],
        }
    ).to_csv(markers_path, index=False)
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1, 1], [1, 0], [0, 1]]),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    monkeypatch.setattr(
        "merxen.analysis.mecr.load_spatial_count_table",
        lambda _sample: adata.copy(),
    )
    config = MecrConfig(
        pair_id="PAIR1",
        segmentation="reseg",
        output_dir=tmp_path / "out",
        samples=[
            MecrSampleConfig(
                sample_id="PAIR1_XENIUM",
                platform="XENIUM",
                zarr_path=tmp_path / "unused.zarr",
            )
        ],
        reference_markers_path=markers_path,
    )

    outputs = run_mecr(config)

    assert all(path.exists() for path in outputs.values())
    summary = pd.read_csv(outputs["combined_summary_csv"]).iloc[0]
    assert summary["mecr"] == pytest.approx(1 / 3)
    assert summary["n_scored_pairs"] == 1
    pairs = pd.read_csv(outputs["combined_pairs_csv"])
    assert pairs.loc[0, "cells_either"] == 3
    assert outputs["distribution_plot"].with_suffix(".pdf").exists()


def test_select_barnyard_pairs_combines_selection_strategies() -> None:
    """Barnyard selection should cover canonical, high-rate, and detected pairs."""
    pair_metrics = pd.DataFrame(
        {
            "gene_1": ["SLC17A7", "GeneA", "GeneC"],
            "class_1": ["Neurons", "Astrocytes", "Fibroblasts"],
            "gene_2": ["GFAP", "GeneB", "GeneD"],
            "class_2": ["Astrocytes", "Microglia", "Vascular cells"],
            "mecr": [0.1, 0.9, 0.2],
            "cells_either": [50, 2, 1_000],
        }
    )

    selected = select_barnyard_pairs(pair_metrics, top_n=3)

    assert set(zip(selected["gene_1"], selected["gene_2"], strict=True)) == {
        ("SLC17A7", "GFAP"),
        ("GeneA", "GeneB"),
        ("GeneC", "GeneD"),
    }
    reasons = selected.set_index("gene_1")["selection_reason"]
    assert "canonical" in reasons["SLC17A7"]
    assert "highest_mean_mecr" in reasons["GeneA"]
    assert "most_detected" in reasons["GeneC"]


def test_mecr_summary_plots_write_png_and_pdf(tmp_path: Path) -> None:
    """Reference, shared-platform, and class-pair plots should be reproducible."""
    rows = []
    for platform, rates in (("MERSCOPE", [0.1, 0.3]), ("XENIUM", [0.2, 0.25])):
        for index, rate in enumerate(rates):
            rows.append(
                {
                    "sample_id": f"sample_{platform}",
                    "platform": platform,
                    "gene_1": f"Gene{index}A",
                    "class_1": "Astrocytes",
                    "gene_2": f"Gene{index}B",
                    "class_2": "Neurons" if index == 0 else "Microglia",
                    "cells_either": 100 + index,
                    "mecr": rate,
                }
            )
    pair_metrics = pd.DataFrame(rows)
    output_paths = [
        plot_reference_mecr_histogram(
            pair_metrics,
            tmp_path / "reference.png",
        ),
        plot_platform_mecr_comparison(
            pair_metrics,
            tmp_path / "platform.png",
            title="Platform comparison",
        ),
        plot_class_pair_mecr_heatmaps(
            pair_metrics,
            tmp_path / "classes.png",
            title="Class-pair comparison",
        ),
    ]

    for path in output_paths:
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_discover_reference_markers_applies_strict_detection_rules() -> None:
    """Markers must pass 25%/1%, and genes qualifying twice must be removed."""
    labels = np.array(["A", "A", "B", "B", *(["C"] * 100)])
    matrix = np.zeros((len(labels), 4), dtype=np.float32)
    matrix[0, 0] = 2.0  # A marker: 50% in A, 0% elsewhere.
    matrix[2, 1] = 2.0  # B marker: 50% in B, 0% elsewhere.
    matrix[4:34, 2] = 2.0  # C marker: 30% in C, 0% elsewhere.
    matrix[0, 3] = 1.0
    matrix[2, 3] = 1.0  # Qualifies for A and B (1/102 in each rest).
    reference = ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=pd.DataFrame(
            {"broad_class": pd.Categorical(labels)},
            index=[f"cell_{index}" for index in range(len(labels))],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "Ambiguous"]),
    )

    statistics, markers = discover_reference_markers(
        reference,
        class_key="broad_class",
        target_classes=["A", "B", "C"],
        min_target_fraction=0.25,
        max_other_fraction=0.01,
    )

    assert set(markers["gene"]) == {"GeneA", "GeneB", "GeneC"}
    assert "Ambiguous" not in set(markers["gene"])
    ambiguous = statistics[statistics["gene"] == "Ambiguous"]
    assert ambiguous["qualifying_class_count"].max() == 2
    assert not ambiguous["is_unique_marker"].any()


def test_discover_reference_markers_uses_strict_threshold_boundaries() -> None:
    """Exactly 25% target detection must not pass the paper's >25% rule."""
    labels = np.array([*(["A"] * 4), *(["B"] * 100)])
    matrix = np.zeros((len(labels), 1), dtype=np.float32)
    matrix[0, 0] = 1.0
    reference = ad.AnnData(
        X=sparse.csr_matrix(matrix),
        obs=pd.DataFrame(
            {"broad_class": pd.Categorical(labels)},
            index=[f"cell_{index}" for index in range(len(labels))],
        ),
        var=pd.DataFrame(index=["BoundaryGene"]),
    )

    _, markers = discover_reference_markers(
        reference,
        class_key="broad_class",
        target_classes=["A", "B"],
        min_target_fraction=0.25,
        max_other_fraction=0.01,
    )

    assert markers.empty


def test_load_whb_panel_reference_joins_taxonomy_and_aggregates_symbols(
    tmp_path: Path,
) -> None:
    """WHB cells should be classed through metadata and genes exposed by symbol."""
    var = pd.DataFrame(
        {"gene_symbol": ["GeneA", "GeneB", "GeneB"]},
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    neurons_path = tmp_path / "neurons.h5ad"
    nonneurons_path = tmp_path / "nonneurons.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix([[2, 0, 0]]),
        obs=pd.DataFrame(index=["neuron_cell"]),
        var=var,
    ).write_h5ad(neurons_path)
    ad.AnnData(
        X=sparse.csr_matrix([[0, 1, 2]]),
        obs=pd.DataFrame(index=["astro_cell"]),
        var=var,
    ).write_h5ad(nonneurons_path)

    cell_metadata_path = tmp_path / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["neuron_cell", "astro_cell"],
            "cluster_alias": ["1", "2"],
        }
    ).to_csv(cell_metadata_path, index=False)
    taxonomy_path = tmp_path / "taxonomy.csv"
    pd.DataFrame(
        {
            "label": ["neuron_label", "astro_label"],
            "name": ["Upper-layer intratelencephalic", "Astrocyte"],
            "cluster_annotation_term_set_label": ["SUPC", "SUPC"],
        }
    ).to_csv(taxonomy_path, index=False)
    membership_path = tmp_path / "membership.csv"
    pd.DataFrame(
        {
            "cluster_annotation_term_label": ["neuron_label", "astro_label"],
            "cluster_annotation_term_set_label": ["SUPC", "SUPC"],
            "cluster_alias": ["1", "2"],
        }
    ).to_csv(membership_path, index=False)
    config = MecrReferenceConfig(
        output_dir=tmp_path / "out",
        samples=[
            MecrSampleConfig(
                sample_id="sample",
                platform="XENIUM",
                zarr_path=tmp_path / "unused.zarr",
            )
        ],
        neurons_h5ad_path=neurons_path,
        nonneurons_h5ad_path=nonneurons_path,
        cell_metadata_path=cell_metadata_path,
        taxonomy_metadata_path=taxonomy_path,
        cluster_membership_path=membership_path,
        taxonomy_level="SUPC",
        target_broad_classes=["Neurons", "Astrocytes"],
        reference_chunk_rows=1,
    )

    result = load_whb_panel_reference(
        config,
        panel_genes=["GeneA", "GeneB", "MissingGene"],
    )

    assert result.shape == (2, 2)
    assert list(result.var_names) == ["GeneA", "GeneB"]
    assert set(result.obs["broad_class"].astype(str)) == {
        "Neurons",
        "Astrocytes",
    }
    dense = result.X.toarray()
    astro_row = np.flatnonzero(
        result.obs["broad_class"].astype(str).to_numpy() == "Astrocytes"
    )[0]
    assert np.count_nonzero(dense[astro_row]) == 1
