"""Tests for local MapMyCells annotation wrappers."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import TextIO

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merxen.analysis.mapmycells import (
    _run_command,
    build_mapmycells_command,
    choose_mapmycells_assignment_column,
    prepare_mapmycells_query,
    run_mapmycells,
)
from merxen.analysis.mapmycells_gpu_compat import (
    HostMemoryCollator,
    apply_mapmycells_gpu_compat_patch,
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
    assert out.var_names.name is None


def test_prepare_mapmycells_query_handles_missing_gene_ids(tmp_path: Path) -> None:
    """Missing gene IDs should fall back to existing symbols and remain writable."""
    input_h5ad = tmp_path / "clustered_missing_ids.h5ad"
    adata = ad.AnnData(
        X=np.log1p(np.array([[2, 0, 1], [0, 5, 2]], dtype=np.float32)),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=pd.DataFrame(
            {"ensembl_id": ["ENSG1", "", "ENSG3"]},
            index=["GeneA", "MissingIdGene", "GeneC"],
        ),
    )
    counts = np.array([[2, 0, 1], [0, 5, 2]], dtype=np.int64)
    adata.layers["counts"] = counts
    adata.write_h5ad(input_h5ad)

    output_h5ad = prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query_missing_ids.h5ad",
        query_layer="counts",
        gene_id_column="ensembl_id",
    )

    out = ad.read_h5ad(output_h5ad)
    np.testing.assert_array_equal(out.X, counts)
    assert list(out.var_names) == ["ENSG1", "MissingIdGene", "ENSG3"]
    assert out.var_names.name is None


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

    assert command[:3] == [
        sys.executable,
        "-m",
        "merxen.analysis.mapmycells_entrypoint",
    ]
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
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    adata.obsm["spatial"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    adata.layers["counts"] = np.array([[3, 0], [0, 4]], dtype=np.int64)
    adata.write_h5ad(input_h5ad)

    marker_lookup = tmp_path / "markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "stats.h5"
    precomputed_stats.write_bytes(b"stats")

    def fake_run(
        command: list[str],
        check: bool,
        stdout: TextIO,
        stderr: TextIO,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert text is True
        stdout.write("mapper stdout\n")
        stderr.write("mapper stderr\n")
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

    stdout_log = results["PAIR1_XENIUM"]["stdout_log"]
    stderr_log = results["PAIR1_XENIUM"]["stderr_log"]
    umap_plot = results["PAIR1_XENIUM"]["umap_plot"]
    spatial_plot = results["PAIR1_XENIUM"]["spatial_plot"]
    annotated = ad.read_h5ad(results["PAIR1_XENIUM"]["annotated_h5ad"])
    assert list(annotated.obs["mapmycells_class_name"]) == ["Neuron", "Astrocyte"]
    np.testing.assert_allclose(
        annotated.obs["mapmycells_class_bootstrapping_probability"].to_numpy(float),
        [0.93, 0.88],
    )
    mapmycells_uns = annotated.uns["merxen_mapmycells"]
    assert list(mapmycells_uns["assignment_columns"]) == [
        "mapmycells_cell_id",
        "mapmycells_class_label",
        "mapmycells_class_name",
        "mapmycells_class_bootstrapping_probability",
    ]
    assert mapmycells_uns["plot_assignment_column"] == "mapmycells_class_name"
    assert mapmycells_uns["extended_json_text"] == json.dumps({"results": []}) + "\n"
    assert "mapper stdout" in mapmycells_uns["stdout_log_text"]
    assert "mapper stderr" in mapmycells_uns["stderr_log_text"]
    assert "mapper stdout" in stdout_log.read_text()
    assert "mapper stderr" in stderr_log.read_text()
    assert umap_plot.exists()
    assert spatial_plot.exists()
    assert (cfg.output_dir / "PAIR1_mapmycells_manifest.json").exists()


def test_choose_mapmycells_assignment_column_prefers_plottable_specificity() -> None:
    """Plot labels should stay readable when fine taxonomy levels are too granular."""
    adata = ad.AnnData(
        X=np.ones((4, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "mapmycells_supercluster_name": ["A", "A", "B", "B"],
                "mapmycells_cluster_name": ["C1", "C2", "C3", "C4"],
                "mapmycells_subcluster_name": ["S1", "S2", "S3", "S4"],
            },
            index=["cell1", "cell2", "cell3", "cell4"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )

    chosen = choose_mapmycells_assignment_column(adata, max_categories=2)

    assert chosen == "mapmycells_supercluster_name"


def test_run_command_writes_logs_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess stdout/stderr should be persisted even when the mapper fails."""

    def fake_run(
        command: list[str],
        check: bool,
        stdout: TextIO,
        stderr: TextIO,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert text is True
        stdout.write("started mapper\n")
        stderr.write("ModuleNotFoundError: No module named 'cell_type_mapper'\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("merxen.analysis.mapmycells.subprocess.run", fake_run)
    stdout_path = tmp_path / "mapper.stdout.log"
    stderr_path = tmp_path / "mapper.stderr.log"

    with pytest.raises(RuntimeError) as exc_info:
        _run_command(
            ["python", "-m", "cell_type_mapper.cli.from_specified_markers"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    message = str(exc_info.value)
    assert "MapMyCells failed with exit code 1" in message
    assert "stderr tail" in message
    assert "ModuleNotFoundError" in message
    assert "started mapper" in stdout_path.read_text()
    assert "cell_type_mapper" in stderr_path.read_text()


def test_mapmycells_gpu_patch_keeps_collator_data_on_host() -> None:
    """The patched GPU loader should leave batches as host arrays."""
    from cell_type_mapper.gpu_utils.anndata_iterator import anndata_iterator

    applied = apply_mapmycells_gpu_compat_patch()
    assert applied or anndata_iterator.Collator is HostMemoryCollator

    collator = anndata_iterator.Collator(
        all_query_identifiers=["gene_a", "gene_b", "gene_c"],
        normalization="raw",
        all_query_markers=["gene_c", "gene_a"],
        device="cuda:0",
    )
    assert isinstance(collator, HostMemoryCollator)
    collator = pickle.loads(pickle.dumps(collator))

    matrix, r0, r1 = collator(
        [
            (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), 10, 11),
            (np.array([[4.0, 5.0, 6.0]], dtype=np.float32), 11, 12),
        ]
    )

    assert r0 == 10
    assert r1 == 12
    assert matrix.normalization == "log2CPM"
    assert matrix.gene_identifiers == ["gene_c", "gene_a"]
    assert isinstance(matrix.data, np.ndarray)
