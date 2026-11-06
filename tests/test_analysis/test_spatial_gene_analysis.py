"""Tests for per-gene spatial autocorrelation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import merxen.analysis.spatial_gene_analysis as sga_mod
from merxen.analysis.spatial_gene_analysis import ranked_spatial_autocorr_genes
from merxen.config import SpatialGeneAnalysisConfig


def test_ranked_spatial_autocorr_genes_returns_metric_extremes() -> None:
    """Ranking output should expose top and bottom genes for both metrics."""
    metrics = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "moran_i": [0.4, -0.1, 0.9, 0.2],
            "moran_i_pval_norm": [0.1, 0.2, 0.01, 0.5],
            "moran_i_pval_fdr_bh": [0.2, 0.3, 0.02, 0.6],
            "geary_c": [0.7, 1.3, 0.2, 0.9],
            "geary_c_pval_norm": [0.1, 0.4, 0.01, 0.5],
            "geary_c_pval_fdr_bh": [0.2, 0.5, 0.02, 0.6],
        }
    )

    ranked = ranked_spatial_autocorr_genes(metrics, top_n=2)

    observed = {
        key: list(group.gene)
        for key, group in ranked.groupby(["metric", "direction"], sort=False)
    }
    assert observed[("moran_i", "top")] == ["C", "A"]
    assert observed[("moran_i", "bottom")] == ["B", "D"]
    assert observed[("geary_c", "top")] == ["B", "D"]
    assert observed[("geary_c", "bottom")] == ["C", "A"]


@pytest.mark.integration
def test_native_atera_row_runs_without_transcript_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ATERA native row: per-gene autocorrelation only, top-N plots only.

    Transcript spatial patterns are off for ATERA rows (AT-6), so the
    transcript pass must never start, and per-gene expression vectors are
    pulled only for the ranked (top-N) genes, never for the whole panel.
    """
    from test_io.atera_native_fixture import make_native_reference

    # More cells and transcripts than the default bundle, so the per-gene
    # statistics and their histograms have something to work with.
    _, latest = make_native_reference(
        tmp_path / "ref", n_cells=20, tiles={"0,0": 1500, "1,0": 1500}
    )

    def refuse(**_: object) -> None:
        raise AssertionError("transcript analysis must stay off for ATERA rows")

    monkeypatch.setattr(sga_mod, "run_transcript_pattern_analysis", refuse)
    vectors: list[str] = []
    real_vector = sga_mod._gene_expression_vector

    def counting_vector(adata: object, gene: str) -> object:
        vectors.append(gene)
        return real_vector(adata, gene)  # type: ignore[arg-type]

    monkeypatch.setattr(sga_mod, "_gene_expression_vector", counting_vector)
    top_n = 1
    config = SpatialGeneAnalysisConfig.model_validate(
        {
            "pair_id": "ATERA_SYNTH",
            "output_dir": tmp_path / "sga_out",
            "samples": [
                {
                    "sample_id": "ATERA_SYNTH",
                    "platform": "ATERA",
                    "zarr_path": latest,
                    "segmentation": "original_seg",
                    "table_key": "table_original",
                    "shape_key": "atera_cell_boundaries",
                    "nuclei_shape_key": "atera_nucleus",
                }
            ],
            "min_cells": 1,
            "top_n": top_n,
            "figure_dpi": 72,
            "transcript_analysis_enabled": False,
        }
    )

    results = sga_mod.run_spatial_gene_analysis(config)["ATERA_SYNTH"]

    metrics = pd.read_csv(results["metrics_csv"])
    manifest = json.loads(Path(results["manifest"]).read_text())
    assert manifest["platform"] == "ATERA"
    assert manifest["n_genes"] == len(metrics) > 0
    assert manifest["parameters"]["transcript_analysis_enabled"] is False
    assert "transcript_summary_csv" not in results
    assert 0 < len(vectors) <= 4 * top_n


def test_prepared_table_records_no_panel_size_expectation() -> None:
    """Spatial gene analysis checks no panel size, for any platform."""
    import anndata as ad
    import numpy as np

    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.poisson(2.0, size=(20, 6)).astype(np.float32),
        var=pd.DataFrame(index=["A", "B", "C", "D", "E", "NegControlProbe_1"]),
    )

    prepared = sga_mod.prepare_spatial_autocorr_adata(adata, min_cells=1)

    summary = prepared.uns["merxen_clustering_squidpy"]["control_feature_filter"]
    assert summary["n_control_features_removed"] == 1
    assert "expected_panel_gene_count" not in summary
    assert "has_expected_panel_gene_count" not in summary
