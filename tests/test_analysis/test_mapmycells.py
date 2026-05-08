"""Tests for local MapMyCells annotation wrappers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merxen.analysis.mapmycells import (
    build_mapmycells_command,
    prepare_mapmycells_query,
    run_mapmycells,
)
from merxen.config import MapMyCellsConfig, MapMyCellsSampleConfig


def test_prepare_mapmycells_query_uses_counts_layer(tmp_path: Path) -> None:
    """MapMyCells query H5AD should place raw counts in X."""
    input_h5ad = tmp_path / "clustered.h5ad"
    adata = ad.AnnData(
        X=np.log1p(np.array([[2, 0], [0, 5]], dtype=np.float32)),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame({"ensembl_id": ["ENSG1", "ENSG2"]}, index=["GeneA", "GeneB"]),
    )
    counts = np.array([[2, 0], [0, 5]], dtype=np.int64)
    adata.layers["counts"] = counts
    adata.write_h5ad(input_h5ad)

    output_h5ad = prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query.h5ad",
        query_layer="counts",
        gene_id_column="ensembl_id",
    )

    out = ad.read_h5ad(output_h5ad)
    np.testing.assert_array_equal(out.X, counts)
    assert list(out.var_names) == ["ENSG1", "ENSG2"]


def test_build_mapmycells_command_includes_bootstrap_factor(tmp_path: Path) -> None:
    """The local mapper command should expose spatial-friendly bootstrap tuning."""
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[
            MapMyCellsSampleConfig(
                sample_id="PAIR1_MERSCOPE",
                platform="MERSCOPE",
                anndata_path=tmp_path / "clustered.h5ad",
            )
        ],
        marker_lookup_path=tmp_path / "markers.json",
        precomputed_stats_path=tmp_path / "stats.h5",
        drop_level="CCN20230722_SUPT",
        bootstrap_factor=0.9,
        n_processors=12,
    )

    command = build_mapmycells_command(
        cfg,
        query_h5ad=tmp_path / "query.h5ad",
        extended_json=tmp_path / "extended.json",
        csv_path=tmp_path / "result.csv",
        log_path=tmp_path / "mapper.log",
    )

    assert command[command.index("--type_assignment.bootstrap_factor") + 1] == "0.9"
    assert command[command.index("--type_assignment.n_processors") + 1] == "12"
    assert command[command.index("--drop_level") + 1] == "CCN20230722_SUPT"


def test_run_mapmycells_writes_annotated_h5ad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage should prepare inputs, call the mapper, and attach CSV labels."""
    input_h5ad = tmp_path / "PAIR1_XENIUM_clustered.h5ad"
    adata = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.layers["counts"] = np.array([[3, 0], [0, 4]], dtype=np.int64)
    adata.write_h5ad(input_h5ad)

    marker_lookup = tmp_path / "markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "stats.h5"
    precomputed_stats.write_bytes(b"stats")

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        csv_path = Path(command[command.index("--csv_result_path") + 1])
        extended_path = Path(command[command.index("--extended_result_path") + 1])
        log_path = Path(command[command.index("--log_path") + 1])
        csv_path.write_text(
            "# metadata = extended.json\n"
            "cell_id,class_label,class_name,class_bootstrapping_probability\n"
            "cell1,CLAS_1,Neuron,0.93\n"
            "cell2,CLAS_2,Astrocyte,0.88\n"
        )
        extended_path.write_text(json.dumps({"results": []}) + "\n")
        log_path.write_text("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("merxen.analysis.mapmycells.subprocess.run", fake_run)

    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[
            MapMyCellsSampleConfig(
                sample_id="PAIR1_XENIUM",
                platform="XENIUM",
                anndata_path=input_h5ad,
            )
        ],
        marker_lookup_path=marker_lookup,
        precomputed_stats_path=precomputed_stats,
        bootstrap_factor=0.9,
        n_processors=2,
    )

    results = run_mapmycells(cfg)

    annotated = ad.read_h5ad(results["PAIR1_XENIUM"]["annotated_h5ad"])
    assert list(annotated.obs["mapmycells_class_name"]) == ["Neuron", "Astrocyte"]
    np.testing.assert_allclose(
        annotated.obs["mapmycells_class_bootstrapping_probability"].to_numpy(float),
        [0.93, 0.88],
    )
    assert (cfg.output_dir / "PAIR1_mapmycells_manifest.json").exists()
