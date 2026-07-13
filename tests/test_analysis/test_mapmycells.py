"""Tests for local MapMyCells annotation wrappers."""

from __future__ import annotations

import io
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
    RegionReferenceArtifacts,
    _ensure_url_file,
    _ensure_wmb_expression_inputs,
    _resolve_full_reference_artifacts,
    _run_command,
    _write_region_cell_metadata,
    build_mapmycells_command,
    choose_mapmycells_assignment_column,
    prepare_mapmycells_query,
    prepare_region_mapmycells_reference,
    read_mapmycells_extended_qc,
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


def test_mapmycells_config_validates_reference_modes(tmp_path: Path) -> None:
    """Reference mode decides which input paths are required."""
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        marker_lookup_path=tmp_path / "markers.json",
        precomputed_stats_path=tmp_path / "stats.h5",
    )
    assert cfg.reference_mode == "both"
    assert cfg.region_name == "frontal_a44_a45_a46_a32_acc"
    assert cfg.region_labels == [
        "Human A44-A45",
        "Human A46",
        "Human A32",
        "Human ACC",
    ]

    region_only = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="region",
        region_labels="Human A46, Human A32",
    )
    assert region_only.marker_lookup_path is None
    assert region_only.region_labels == ["Human A46", "Human A32"]

    with pytest.raises(ValueError, match="marker_lookup_path"):
        MapMyCellsConfig(
            pair_id="PAIR1",
            output_dir=tmp_path / "mapmycells_out",
            samples=[],
            reference_mode="whole_brain",
            auto_download_references=False,
        )

    with pytest.raises(ValueError, match="region_labels"):
        MapMyCellsConfig(
            pair_id="PAIR1",
            output_dir=tmp_path / "mapmycells_out",
            samples=[],
            reference_mode="region",
            region_labels=[],
        )

    plots_only = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="both",
        region_labels=[],
        plots_only=True,
    )
    assert plots_only.plots_only is True


def test_mapmycells_config_enables_human_to_wmb_defaults(tmp_path: Path) -> None:
    """Human-to-WMB mapping should select the Allen cross-species drop level."""
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="whole_brain",
        reference_atlas="wmb",
        query_species="human",
    )

    assert cfg.drop_level == "CCN20230722_SUPT"
    assert cfg.auto_download_references is True

    with pytest.raises(ValueError, match="mouse region_of_interest_acronym"):
        MapMyCellsConfig(
            pair_id="PAIR1",
            output_dir=tmp_path / "mapmycells_out",
            samples=[],
            reference_mode="region",
            reference_atlas="wmb",
            query_species="human",
        )


def test_build_mapmycells_command_includes_gene_mapping_db(tmp_path: Path) -> None:
    """Cross-species mapping should pass the mmc_gene_mapper database."""
    gene_db = tmp_path / "gene_mapper.db"
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="whole_brain",
        reference_atlas="wmb",
        gene_mapping_db_path=gene_db,
        marker_lookup_path=tmp_path / "markers.json",
        precomputed_stats_path=tmp_path / "stats.h5",
    )

    command = build_mapmycells_command(
        cfg,
        query_h5ad=tmp_path / "query.h5ad",
        extended_json=tmp_path / "extended.json",
        csv_path=tmp_path / "result.csv",
        log_path=tmp_path / "mapper.log",
    )

    assert command[command.index("--gene_mapping.db_path") + 1] == str(gene_db)


def test_full_wmb_reference_downloads_manifest_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing full-WMB files should be downloaded from the Allen manifest."""
    manifest = {
        "file_listing": {
            "WMB-10X": {
                "mapmycells": {
                    "mouse_markers_230821": {"files": {"json": {"url": "markers"}}},
                    "precomputed_stats_ABC_revision_230821": {
                        "files": {"h5": {"url": "stats"}}
                    },
                }
            }
        }
    }
    monkeypatch.setattr(
        "merxen.analysis.mapmycells._load_abc_manifest", lambda: manifest
    )

    def fake_ensure(info: dict[str, object], cache_dir: Path) -> Path:
        suffix = ".json" if info["url"] == "markers" else ".h5"
        path = cache_dir / f"downloaded{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
        return path

    monkeypatch.setattr("merxen.analysis.mapmycells._ensure_manifest_file", fake_ensure)
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="whole_brain",
        reference_atlas="wmb",
        query_species="mouse",
        region_cache_dir=tmp_path / "cache",
    )

    marker_path, stats_path, metadata = _resolve_full_reference_artifacts(cfg)

    assert marker_path.name == "downloaded.json"
    assert stats_path.name == "downloaded.h5"
    assert metadata["reference_atlas"] == "wmb"
    assert metadata["downloaded"] is True


def test_wmb_expression_download_resolves_feature_matrix_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WMB expression downloads should resolve labels across rig directories."""
    matrix_label = "WMB-10Xv3-Isocortex-1"
    manifest = {
        "file_listing": {
            "WMB-10Xv2": {"expression_matrices": {}},
            "WMB-10Xv3": {
                "expression_matrices": {
                    matrix_label: {"raw": {"files": {"h5ad": {"url": "matrix"}}}}
                }
            },
            "WMB-10XMulti": {"expression_matrices": {}},
        }
    }
    monkeypatch.setattr(
        "merxen.analysis.mapmycells._load_abc_manifest", lambda: manifest
    )

    def fake_ensure(
        info: dict[str, object],
        cache_dir: Path,
        *,
        force_download: bool,
    ) -> Path:
        assert info["url"] == "matrix"
        assert force_download is False
        return cache_dir / "matrix.h5ad"

    monkeypatch.setattr("merxen.analysis.mapmycells._ensure_manifest_file", fake_ensure)

    paths = _ensure_wmb_expression_inputs(
        tmp_path / "cache",
        matrix_labels=[matrix_label],
    )

    assert paths == {f"{matrix_label}_raw": tmp_path / "cache/abc_atlas/matrix.h5ad"}


def test_reference_download_resumes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large Allen downloads should resume a retained partial file."""
    output_path = tmp_path / "reference.db"
    output_path.with_name("reference.db.tmp").write_bytes(b"abc")

    class PartialResponse(io.BytesIO):
        status = 206

        def __enter__(self: PartialResponse) -> PartialResponse:
            return self

        def __exit__(self: PartialResponse, *args: object) -> None:
            self.close()

    def fake_urlopen(request: object) -> PartialResponse:
        assert request.headers["Range"] == "bytes=3-"  # type: ignore[attr-defined]
        return PartialResponse(b"def")

    monkeypatch.setattr(
        "merxen.analysis.mapmycells.urllib.request.urlopen", fake_urlopen
    )

    result = _ensure_url_file(
        "https://example.invalid/reference.db",
        output_path,
        expected_size=6,
    )

    assert result.read_bytes() == b"abcdef"


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
            "cell_id,supercluster_label,supercluster_name,"
            "supercluster_bootstrapping_probability,cluster_label,cluster_name,"
            "cluster_bootstrapping_probability,class_label,class_name,"
            "class_bootstrapping_probability\n"
            "cell1,SUPC_1,Neuronal,0.96,CLUS_1,Excitatory,0.93,"
            "CLAS_1,Neuron,0.93\n"
            "cell2,SUPC_2,Glial,0.91,CLUS_2,Astro,0.88,"
            "CLAS_2,Astrocyte,0.88\n"
        )
        extended_path.write_text(
            json.dumps(
                {
                    "taxonomy_tree": {
                        "hierarchy_mapper": {
                            "SUPC": "supercluster",
                            "CLUS": "cluster",
                        },
                        "name_mapper": {
                            "SUPC": {
                                "SUPC_1": {"name": "Neuronal"},
                                "SUPC_2": {"name": "Glial"},
                            },
                            "CLUS": {
                                "CLUS_1": {"name": "Excitatory"},
                                "CLUS_2": {"name": "Astro"},
                            },
                        },
                    },
                    "results": [
                        {
                            "cell_id": "cell1",
                            "SUPC": {
                                "assignment": "SUPC_1",
                                "bootstrapping_probability": 0.96,
                                "aggregate_probability": 0.96,
                                "avg_correlation": 0.52,
                                "directly_assigned": True,
                                "runner_up_probability": [0.03],
                            },
                            "CLUS": {
                                "assignment": "CLUS_1",
                                "bootstrapping_probability": 0.93,
                                "aggregate_probability": 0.89,
                                "avg_correlation": 0.48,
                                "directly_assigned": True,
                                "runner_up_probability": [0.05],
                            },
                        },
                        {
                            "cell_id": "cell2",
                            "SUPC": {
                                "assignment": "SUPC_2",
                                "bootstrapping_probability": 0.91,
                                "aggregate_probability": 0.91,
                                "avg_correlation": 0.45,
                                "directly_assigned": True,
                                "runner_up_probability": [0.07],
                            },
                            "CLUS": {
                                "assignment": "CLUS_2",
                                "bootstrapping_probability": 0.88,
                                "aggregate_probability": 0.80,
                                "avg_correlation": 0.41,
                                "directly_assigned": True,
                                "runner_up_probability": [0.10],
                            },
                        },
                    ],
                }
            )
            + "\n"
        )
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
        reference_mode="whole_brain",
        marker_lookup_path=marker_lookup,
        precomputed_stats_path=precomputed_stats,
        bootstrap_factor=0.9,
        n_processors=2,
    )

    results = run_mapmycells(cfg)

    whole_brain_results = results["PAIR1_XENIUM"]["whole_brain"]
    stdout_log = whole_brain_results["stdout_log"]
    stderr_log = whole_brain_results["stderr_log"]
    umap_plot = whole_brain_results["umap_plot"]
    spatial_plot = whole_brain_results["spatial_plot"]
    umap_cluster_dir = whole_brain_results["umap_cluster_by_supercluster_dir"]
    quality_scatter_plot = whole_brain_results["quality_scatter_plot"]
    supercluster_qc_plot = whole_brain_results["supercluster_qc_plot"]
    cluster_qc_plot = whole_brain_results["cluster_qc_plot"]
    spatial_supercluster_grid_plot = whole_brain_results[
        "spatial_supercluster_grid_plot"
    ]
    annotated = ad.read_h5ad(whole_brain_results["annotated_h5ad"])
    assert list(annotated.obs["mapmycells_class_name"]) == ["Neuron", "Astrocyte"]
    assert list(annotated.obs["mapmycells_supercluster_name"]) == [
        "Neuronal",
        "Glial",
    ]
    np.testing.assert_allclose(
        annotated.obs["mapmycells_class_bootstrapping_probability"].to_numpy(float),
        [0.93, 0.88],
    )
    mapmycells_uns = annotated.uns["merxen_mapmycells"]
    assert list(mapmycells_uns["assignment_columns"]) == [
        "mapmycells_cell_id",
        "mapmycells_supercluster_label",
        "mapmycells_supercluster_name",
        "mapmycells_supercluster_bootstrapping_probability",
        "mapmycells_cluster_label",
        "mapmycells_cluster_name",
        "mapmycells_cluster_bootstrapping_probability",
        "mapmycells_class_label",
        "mapmycells_class_name",
        "mapmycells_class_bootstrapping_probability",
    ]
    assert mapmycells_uns["plot_assignment_column"] == "mapmycells_cluster_name"
    assert "quality_scatter" in mapmycells_uns["plot_paths"]
    assert "umap_cluster_by_supercluster" in mapmycells_uns["plot_paths"]
    assert "spatial_supercluster_grid" in mapmycells_uns["plot_paths"]
    assert "taxonomy_tree" in mapmycells_uns["extended_json_text"]
    assert "mapper stdout" in mapmycells_uns["stdout_log_text"]
    assert "mapper stderr" in mapmycells_uns["stderr_log_text"]
    assert "mapper stdout" in stdout_log.read_text()
    assert "mapper stderr" in stderr_log.read_text()
    assert umap_plot.exists()
    assert spatial_plot.exists()
    assert umap_cluster_dir.exists()
    assert umap_plot.with_suffix(".pdf").exists()
    assert spatial_plot.with_suffix(".pdf").exists()
    assert len(list(umap_cluster_dir.glob("*.png"))) == 2
    assert len(list(umap_cluster_dir.glob("*.pdf"))) == 2
    for plot in (
        quality_scatter_plot,
        supercluster_qc_plot,
        cluster_qc_plot,
        spatial_supercluster_grid_plot,
    ):
        assert plot.exists()
        assert plot.with_suffix(".pdf").exists()
    extended_qc = read_mapmycells_extended_qc(whole_brain_results["extended_json"])
    assert set(extended_qc["level_token"]) == {"supercluster", "cluster"}
    np.testing.assert_allclose(
        extended_qc.loc[
            extended_qc["level_token"].eq("cluster"), "runner_up_margin"
        ].to_numpy(float),
        [0.88, 0.78],
    )
    assert (cfg.output_dir / "PAIR1_mapmycells_manifest.json").exists()


def test_run_mapmycells_default_both_writes_region_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode should keep whole-brain outputs and add region outputs."""
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

    marker_lookup = tmp_path / "whole_markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "whole_stats.h5"
    precomputed_stats.write_bytes(b"stats")
    region_marker_lookup = tmp_path / "region_markers.json"
    region_marker_lookup.write_text("{}\n")
    region_stats = tmp_path / "region_stats.h5"
    region_stats.write_bytes(b"stats")
    region_manifest = tmp_path / "region_manifest.json"
    region_manifest.write_text("{}\n")

    def fake_region_reference(config: MapMyCellsConfig) -> RegionReferenceArtifacts:
        return RegionReferenceArtifacts(
            marker_lookup_path=region_marker_lookup,
            precomputed_stats_path=region_stats,
            manifest_path=region_manifest,
            manifest={
                "reference_type": "region",
                "config": {"region_name": "frontal_a44_a45_a46_a32_acc"},
            },
        )

    def fake_run(
        command: list[str],
        check: bool,
        stdout: TextIO,
        stderr: TextIO,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        csv_path = Path(command[command.index("--csv_result_path") + 1])
        extended_path = Path(command[command.index("--extended_result_path") + 1])
        log_path = Path(command[command.index("--log_path") + 1])
        csv_path.write_text(
            "cell_id,class_label,class_name,class_bootstrapping_probability\n"
            "cell1,CLAS_1,Neuron,0.93\n"
            "cell2,CLAS_2,Astrocyte,0.88\n"
        )
        extended_path.write_text(json.dumps({"results": []}) + "\n")
        log_path.write_text("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "merxen.analysis.mapmycells.prepare_region_mapmycells_reference",
        fake_region_reference,
    )
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
    )

    results = run_mapmycells(cfg)

    assert set(results["PAIR1_XENIUM"]) == {
        "whole_brain",
        "region_frontal_a44_a45_a46_a32_acc",
    }
    assert (
        results["PAIR1_XENIUM"]["whole_brain"]["csv"].parent
        == cfg.output_dir / "xenium"
    )
    assert (
        results["PAIR1_XENIUM"]["region_frontal_a44_a45_a46_a32_acc"]["csv"].parent
        == cfg.output_dir / "region_frontal_a44_a45_a46_a32_acc" / "xenium"
    )
    region_annotated = ad.read_h5ad(
        results["PAIR1_XENIUM"]["region_frontal_a44_a45_a46_a32_acc"]["annotated_h5ad"]
    )
    assert list(
        region_annotated.obs["mapmycells_region_frontal_a44_a45_a46_a32_acc_class_name"]
    ) == ["Neuron", "Astrocyte"]
    assert (
        region_annotated.uns["merxen_mapmycells_region_frontal_a44_a45_a46_a32_acc"][
            "column_prefix"
        ]
        == "mapmycells_region_frontal_a44_a45_a46_a32_acc_"
    )


def test_run_mapmycells_plots_only_reuses_existing_mapper_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plots-only mode should skip mapper execution and reuse CSV/JSON outputs."""
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

    sample_dir = tmp_path / "mapmycells_out" / "region_test" / "xenium"
    sample_dir.mkdir(parents=True)
    (sample_dir / "PAIR1_XENIUM_mapmycells.csv").write_text(
        "cell_id,supercluster_label,supercluster_name,"
        "supercluster_bootstrapping_probability,cluster_label,cluster_name,"
        "cluster_bootstrapping_probability\n"
        "cell1,SUPC_1,Neuronal,0.96,CLUS_1,Excitatory,0.93\n"
        "cell2,SUPC_2,Glial,0.91,CLUS_2,Astro,0.88\n"
    )
    (sample_dir / "PAIR1_XENIUM_mapmycells_extended.json").write_text(
        json.dumps(
            {
                "taxonomy_tree": {
                    "hierarchy_mapper": {
                        "SUPC": "supercluster",
                        "CLUS": "cluster",
                    }
                },
                "results": [
                    {
                        "cell_id": "cell1",
                        "SUPC": {
                            "assignment": "SUPC_1",
                            "bootstrapping_probability": 0.96,
                            "aggregate_probability": 0.96,
                            "avg_correlation": 0.52,
                            "directly_assigned": True,
                            "runner_up_probability": [0.03],
                        },
                        "CLUS": {
                            "assignment": "CLUS_1",
                            "bootstrapping_probability": 0.93,
                            "aggregate_probability": 0.89,
                            "avg_correlation": 0.48,
                            "directly_assigned": True,
                            "runner_up_probability": [0.05],
                        },
                    },
                    {
                        "cell_id": "cell2",
                        "SUPC": {
                            "assignment": "SUPC_2",
                            "bootstrapping_probability": 0.91,
                            "aggregate_probability": 0.91,
                            "avg_correlation": 0.45,
                            "directly_assigned": True,
                            "runner_up_probability": [0.07],
                        },
                        "CLUS": {
                            "assignment": "CLUS_2",
                            "bootstrapping_probability": 0.88,
                            "aggregate_probability": 0.80,
                            "avg_correlation": 0.41,
                            "directly_assigned": True,
                            "runner_up_probability": [0.10],
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    def fail_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("MapMyCells subprocess should not run in plots-only mode")

    def fail_region_reference(config: MapMyCellsConfig) -> RegionReferenceArtifacts:
        raise AssertionError("Region reference should not rebuild in plots-only mode")

    monkeypatch.setattr("merxen.analysis.mapmycells.subprocess.run", fail_run)
    monkeypatch.setattr(
        "merxen.analysis.mapmycells.prepare_region_mapmycells_reference",
        fail_region_reference,
    )

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
        reference_mode="region",
        region_name="test",
        region_labels=[],
        plots_only=True,
    )

    results = run_mapmycells(cfg)

    region_results = results["PAIR1_XENIUM"]["region_test"]
    assert not region_results["query_h5ad"].exists()
    assert region_results["quality_scatter_plot"].exists()
    assert region_results["umap_cluster_by_supercluster_dir"].exists()
    assert region_results["supercluster_qc_plot"].exists()
    assert region_results["cluster_qc_plot"].exists()
    assert region_results["spatial_supercluster_grid_plot"].exists()
    annotated = ad.read_h5ad(region_results["annotated_h5ad"])
    assert list(annotated.obs["mapmycells_region_test_cluster_name"]) == [
        "Excitatory",
        "Astro",
    ]
    assert annotated.uns["merxen_mapmycells_region_test"]["plot_paths"][
        "quality_scatter"
    ].endswith("_mapmycells_quality_scatter.png")


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


def test_region_cell_metadata_filters_multiple_rois_and_sparse_leaves(
    tmp_path: Path,
) -> None:
    """Strict ROI metadata should support multiple labels and leaf count filters."""
    cell_metadata_path = tmp_path / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "region_of_interest_label": [
                "Human A46",
                "Human A46",
                "Human A32",
                "Human A32",
                "Human A32",
                "Human MTG",
            ],
            "cluster_alias": [1, 1, 2, 3, 3, 4],
        }
    ).to_csv(cell_metadata_path, index=False)
    roi_map_path = tmp_path / "roi.csv"
    pd.DataFrame(
        {"region_of_interest_label": ["Human A46", "Human A32", "Human MTG"]}
    ).to_csv(roi_map_path, index=False)
    output_path = tmp_path / "region_cell_metadata.csv"

    summary = _write_region_cell_metadata(
        cell_metadata_path=cell_metadata_path,
        output_path=output_path,
        region_labels=["Human A46", "Human A32"],
        min_cells_per_leaf=2,
        roi_map_path=roi_map_path,
    )

    filtered = pd.read_csv(output_path)
    assert list(filtered["cell_label"]) == ["c1", "c2", "c4", "c5"]
    assert summary["n_cells_after_region_filter"] == 5
    assert summary["n_cells_after_min_leaf_filter"] == 4
    assert summary["dropped_leaf_aliases"] == {"2": 1}


def test_region_cell_metadata_supports_wmb_acronyms(tmp_path: Path) -> None:
    """Mouse ROI filtering should retain the required matrix shard labels."""
    cell_metadata_path = tmp_path / "wmb_cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["c1", "c2", "c3"],
            "region_of_interest_acronym": ["MOp", "MOp", "VIS"],
            "feature_matrix_label": [
                "WMB-10Xv3-Isocortex-1",
                "WMB-10Xv3-Isocortex-1",
                "WMB-10Xv2-Isocortex-2",
            ],
            "cluster_alias": [1, 1, 2],
        }
    ).to_csv(cell_metadata_path, index=False)
    roi_path = tmp_path / "wmb_roi.csv"
    pd.DataFrame({"region_of_interest_acronym": ["MOp", "VIS"]}).to_csv(
        roi_path, index=False
    )

    summary = _write_region_cell_metadata(
        cell_metadata_path=cell_metadata_path,
        output_path=tmp_path / "filtered.csv",
        region_labels=["MOp"],
        min_cells_per_leaf=2,
        roi_map_path=roi_path,
        region_column="region_of_interest_acronym",
    )

    assert summary["matched_region_labels"] == ["MOp"]
    assert summary["feature_matrix_labels"] == ["WMB-10Xv3-Isocortex-1"]


def test_prepare_region_reference_reuses_cache_and_force_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region reference preparation should reuse matching cached artifacts."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    cell_metadata_path = input_dir / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["c1", "c2"],
            "region_of_interest_label": ["Human A46", "Human A46"],
            "cluster_alias": [1, 1],
        }
    ).to_csv(cell_metadata_path, index=False)
    roi_map_path = input_dir / "roi.csv"
    pd.DataFrame({"region_of_interest_label": ["Human A46"]}).to_csv(
        roi_map_path,
        index=False,
    )
    cluster_annotation_path = input_dir / "cluster_annotation_term.csv"
    cluster_annotation_path.write_text("label\n")
    cluster_membership_path = input_dir / "cluster_membership.csv"
    cluster_membership_path.write_text("cluster_alias\n")
    neurons_path = input_dir / "neurons.h5ad"
    neurons_path.write_text("neurons\n")
    nonneurons_path = input_dir / "nonneurons.h5ad"
    nonneurons_path.write_text("nonneurons\n")

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._ensure_whb_reference_inputs",
        lambda cache_dir, force_download=False: {
            "cell_metadata": cell_metadata_path,
            "region_of_interest_structure_map": roi_map_path,
            "cluster_annotation_term": cluster_annotation_path,
            "cluster_to_cluster_annotation_membership": cluster_membership_path,
            "WHB-10Xv3-Neurons_raw": neurons_path,
            "WHB-10Xv3-Nonneurons_raw": nonneurons_path,
        },
    )
    calls = {"precompute": 0, "reference": 0, "query": 0}

    def fake_precompute(config: dict[str, object]) -> None:
        calls["precompute"] += 1
        Path(str(config["output_path"])).write_bytes(b"stats")

    def fake_reference(config: dict[str, object]) -> None:
        calls["reference"] += 1
        output_dir = Path(str(config["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reference_markers.h5").write_bytes(b"markers")

    def fake_query(config: dict[str, object]) -> None:
        calls["query"] += 1
        Path(str(config["output_path"])).write_text("{}\n")

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._run_precomputation_abc",
        fake_precompute,
    )
    monkeypatch.setattr(
        "merxen.analysis.mapmycells._run_reference_markers",
        fake_reference,
    )
    monkeypatch.setattr("merxen.analysis.mapmycells._run_query_markers", fake_query)

    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[],
        reference_mode="region",
        region_cache_dir=tmp_path / "cache",
        region_min_cells_per_leaf=2,
    )

    first = prepare_region_mapmycells_reference(cfg)
    second = prepare_region_mapmycells_reference(cfg)

    assert first.marker_lookup_path == second.marker_lookup_path
    assert calls == {"precompute": 1, "reference": 1, "query": 1}

    force_cfg = cfg.model_copy(update={"region_force_rebuild": True})
    prepare_region_mapmycells_reference(force_cfg)
    assert calls == {"precompute": 2, "reference": 2, "query": 2}


def test_prepare_wmb_region_reference_selects_required_matrix_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WMB region builds should download only shards represented in the ROI."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cell_metadata = inputs / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["c1", "c2"],
            "region_of_interest_acronym": ["MOp", "MOp"],
            "feature_matrix_label": [
                "WMB-10Xv3-Isocortex-1",
                "WMB-10Xv3-Isocortex-1",
            ],
            "cluster_alias": [1, 1],
        }
    ).to_csv(cell_metadata, index=False)
    roi_metadata = inputs / "roi.csv"
    pd.DataFrame({"region_of_interest_acronym": ["MOp"]}).to_csv(
        roi_metadata, index=False
    )
    cluster_annotation = inputs / "cluster_annotation.csv"
    cluster_annotation.write_text("label\n")
    cluster_membership = inputs / "cluster_membership.csv"
    cluster_membership.write_text("cluster_alias\n")
    expression = inputs / "isocortex.h5ad"
    expression.write_text("expression")

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._ensure_wmb_reference_metadata_inputs",
        lambda cache_dir, force_download=False: {
            "cell_metadata": cell_metadata,
            "region_of_interest_metadata": roi_metadata,
            "cluster_annotation_term": cluster_annotation,
            "cluster_to_cluster_annotation_membership": cluster_membership,
        },
    )

    def fake_expression_inputs(
        cache_dir: Path,
        *,
        matrix_labels: list[str],
        force_download: bool = False,
    ) -> dict[str, Path]:
        assert matrix_labels == ["WMB-10Xv3-Isocortex-1"]
        return {"WMB-10Xv3-Isocortex-1_raw": expression}

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._ensure_wmb_expression_inputs",
        fake_expression_inputs,
    )

    def fake_precompute(config: dict[str, object]) -> None:
        assert config["hierarchy"] == [
            "CCN20230722_CLAS",
            "CCN20230722_SUBC",
            "CCN20230722_SUPT",
            "CCN20230722_CLUS",
        ]
        assert config["h5ad_path_list"] == [str(expression)]
        Path(str(config["output_path"])).write_bytes(b"stats")

    def fake_reference(config: dict[str, object]) -> None:
        output_dir = Path(str(config["output_dir"]))
        (output_dir / "reference_markers.h5").write_bytes(b"markers")

    def fake_query(config: dict[str, object]) -> None:
        Path(str(config["output_path"])).write_text("{}\n")

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._run_precomputation_abc", fake_precompute
    )
    monkeypatch.setattr(
        "merxen.analysis.mapmycells._run_reference_markers", fake_reference
    )
    monkeypatch.setattr("merxen.analysis.mapmycells._run_query_markers", fake_query)
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "out",
        samples=[],
        reference_mode="region",
        reference_atlas="wmb",
        query_species="mouse",
        region_name="motor",
        region_labels=["MOp"],
        region_cache_dir=tmp_path / "cache",
        region_min_cells_per_leaf=2,
    )

    artifacts = prepare_region_mapmycells_reference(cfg)

    assert "wmb_region_motor" in str(artifacts.manifest_path)
    assert artifacts.manifest["config"]["reference_atlas"] == "wmb"


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
