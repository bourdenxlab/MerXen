"""Tests for the Scanpy/Squidpy clustering shim."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box

import merxen.analysis.clustering_squidpy as clustering_mod
from merxen.analysis.clustering_squidpy import (
    AtlasMarkerSet,
    _add_spatial_scale_bar,
    _clean_spatial_axis,
    _clustered_spatialdata_table_key,
    _make_neuron_split_marker_sets,
    _run_gpu_clustering,
    adata_from_spatialdata,
    build_clustered_spatialdata_table,
    collapse_atlas_label_to_broad_class,
    compute_clustering_squidpy,
    compute_group_gene_summary,
    finalize_clustering_squidpy,
    load_atlas_marker_sets,
    plot_annotation_score_heatmap,
    plot_group_gene_dotplot,
    plot_spatial_cluster_grid,
    plot_spatial_scatter,
    prepare_clustering_squidpy,
    remove_control_features,
    run_clustering_squidpy,
    run_scanpy_clustering,
    score_clusters_by_atlas_markers,
    write_clustered_spatialdata_table,
)
from merxen.config import ClusteringSquidpyConfig


def test_adata_from_spatialdata_adds_spatial_area_and_control_metrics() -> None:
    """SpatialData table extraction should add Squidpy-ready coordinates."""
    obs = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "control_probe_counts": [1, 0, 2, 0],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    var = pd.DataFrame(index=["GeneA", "Blank-1", "GeneB", "NegControlProbe-1"])
    adata = ad.AnnData(
        X=np.array(
            [
                [10, 1, 0, 2],
                [0, 0, 12, 1],
                [3, 4, 5, 0],
                [6, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        obs=obs,
        var=var,
    )
    adata.obsm["blank"] = pd.DataFrame(
        {"Blank-A": [5, 0, 1, 0]},
        index=adata.obs_names,
    )
    adata.uns["spatialdata_attrs"] = {"region": "MOSAIK_proseg"}

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "geometry": [
                box(0, 0, 1, 1),
                box(2, 0, 3, 1),
                box(0, 2, 1, 3),
                box(2, 2, 3, 3),
            ],
        },
        geometry="geometry",
    )
    aligned_gdf = gdf.copy()
    aligned_gdf["geometry"] = aligned_gdf.geometry.translate(xoff=10.0)
    fake_sdata = SimpleNamespace(
        tables={"table": adata},
        shapes={
            "MOSAIK_proseg": gdf,
            "MOSAIK_proseg_aligned_nonrigid": aligned_gdf,
        },
    )

    out = adata_from_spatialdata(fake_sdata, platform="MERSCOPE")

    assert out.uns["merxen_clustering_squidpy"]["shape_key"] == (
        "MOSAIK_proseg_aligned_nonrigid"
    )
    np.testing.assert_allclose(out.obsm["spatial"][0], [10.5, 0.5])
    np.testing.assert_allclose(out.obs["cell_area"].to_numpy(float), 1.0)
    np.testing.assert_allclose(
        out.obs["control_counts"].to_numpy(float),
        [9.0, 1.0, 7.0, 0.0],
    )
    assert out.obs["nucleus_ratio"].isna().all()


def test_adata_from_spatialdata_adds_xenium_nucleus_ratio_from_shapes() -> None:
    """Xenium nucleus shapes should fill nucleus_area when tables lack it."""
    obs = pd.DataFrame(
        {"cell_id": ["x1", "x2"]},
        index=["x1", "x2"],
    )
    adata = ad.AnnData(
        X=np.array([[10, 1], [2, 8]], dtype=np.int64),
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.uns["spatialdata_attrs"] = {"region": "xenium_cell_boundaries"}
    cell_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 2, 2), box(4, 0, 6, 2)],
        },
        geometry="geometry",
    )
    nucleus_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 1, 1), box(4, 0, 5, 1)],
        },
        geometry="geometry",
    )
    fake_sdata = SimpleNamespace(
        tables={"table": adata},
        shapes={
            "xenium_cell_boundaries": cell_gdf,
            "xenium_nucleus": nucleus_gdf,
        },
    )

    out = adata_from_spatialdata(fake_sdata, platform="XENIUM")

    np.testing.assert_allclose(out.obs["cell_area"].to_numpy(float), [4.0, 4.0])
    np.testing.assert_allclose(out.obs["nucleus_area"].to_numpy(float), [1.0, 1.0])
    np.testing.assert_allclose(out.obs["nucleus_ratio"].to_numpy(float), [0.25, 0.25])


def test_adata_from_spatialdata_adds_ensembl_ids_from_original_table() -> None:
    """Gene IDs from one SpatialData table should annotate clustering tables."""
    obs = pd.DataFrame({"cell_id": ["x1", "x2"]}, index=["x1", "x2"])
    adata = ad.AnnData(
        X=np.array([[10, 1], [2, 8]], dtype=np.int64),
        obs=obs,
        var=pd.DataFrame({"gene": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.uns["spatialdata_attrs"] = {"region": "xenium_cell_boundaries"}
    original = ad.AnnData(
        X=np.ones((1, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["cell0"]),
        var=pd.DataFrame(
            {
                "gene_ids": ["ENSG000001", "ENSG000002"],
                "feature_types": ["Gene Expression", "Gene Expression"],
            },
            index=["GeneA", "GeneB"],
        ),
    )
    cell_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    fake_sdata = SimpleNamespace(
        tables={"table": adata, "table_original": original},
        shapes={"xenium_cell_boundaries": cell_gdf},
    )

    out = adata_from_spatialdata(fake_sdata, platform="XENIUM")

    assert list(out.var["ensembl_id"]) == ["ENSG000001", "ENSG000002"]
    assert out.uns["merxen_clustering_squidpy"]["ensembl_id_mapping"] == {
        "n_features": 2,
        "n_mapped": 2,
        "column": "ensembl_id",
    }


def test_collect_gene_ids_falls_back_to_reference_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone MERSCOPE should recover Ensembl IDs without a Xenium table."""
    config = SimpleNamespace(samples=[])
    monkeypatch.setattr(
        clustering_mod,
        "_load_configured_marker_alias_lookup",
        lambda _config: {
            "GeneA": "ENSG000001",
            "ENSG000001": "GeneA",
            "GeneB": "ENSG000002",
        },
    )

    lookup = clustering_mod.collect_gene_id_lookup_for_samples(config)

    assert lookup == {
        "GeneA": "ENSG000001",
        "GeneB": "ENSG000002",
    }


def test_run_scanpy_clustering_adds_umap_and_leiden() -> None:
    """The gentle Scanpy workflow should produce expected clustering fields."""
    rng = np.random.default_rng(1)
    adata = ad.AnnData(
        X=rng.poisson(lam=4, size=(12, 6)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(6)]),
    )
    adata.obsm["spatial"] = rng.normal(size=(12, 2))

    out = run_scanpy_clustering(
        adata,
        min_counts=1,
        min_cells=1,
        normalize_exclude_highly_expressed=False,
        n_pcs=3,
        n_neighbors=3,
        umap_min_dist=0.2,
        umap_spread=1.5,
        random_seed=1,
        use_gpu=False,
    )

    assert "counts" in out.layers
    assert "X_umap" in out.obsm
    assert "leiden" in out.obs
    assert out.uns["merxen_clustering_params"]["umap_min_dist"] == 0.2
    assert out.uns["merxen_clustering_params"]["umap_spread"] == 1.5


def test_run_scanpy_clustering_can_start_from_counts_layer() -> None:
    """Branch reclustering should renormalize from raw counts, not parent log X."""
    rng = np.random.default_rng(11)
    counts = rng.poisson(lam=4, size=(12, 5)).astype(np.float32)
    adata = ad.AnnData(
        X=np.zeros_like(counts),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(5)]),
        layers={"counts": counts.copy()},
    )
    adata.obsm["spatial"] = rng.normal(size=(12, 2))

    out = run_scanpy_clustering(
        adata,
        min_counts=1,
        min_cells=1,
        n_pcs=2,
        n_neighbors=3,
        random_seed=11,
        use_gpu=False,
        key_added="leiden_branch",
        input_layer="counts",
    )

    assert "leiden_branch" in out.obs
    assert "leiden" not in out.obs
    assert float(out.layers["counts"].sum()) > 0.0
    assert out.uns["merxen_clustering_params_leiden_branch"]["input_layer"] == (
        "counts"
    )


def test_run_scanpy_clustering_preserves_ensembl_ids() -> None:
    """Filtering and clustering should retain gene IDs needed by MapMyCells."""
    rng = np.random.default_rng(2)
    adata = ad.AnnData(
        X=rng.poisson(lam=4, size=(12, 4)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB", "Blank-1", "GeneC"],
                "ensembl_id": ["ENSG000001", "ENSG000002", "", "ENSG000003"],
            },
            index=["GeneA", "GeneB", "Blank-1", "GeneC"],
        ),
    )
    adata.obsm["spatial"] = rng.normal(size=(12, 2))

    out = run_scanpy_clustering(
        adata,
        min_counts=1,
        min_cells=1,
        n_pcs=2,
        n_neighbors=3,
        random_seed=2,
        use_gpu=False,
    )

    assert list(out.var["ensembl_id"]) == [
        "ENSG000001",
        "ENSG000002",
        "ENSG000003",
    ]


def test_remove_control_features_drops_blank_negative_and_unassigned() -> None:
    """Control-like features should be excluded from clustering inputs."""
    adata = ad.AnnData(
        X=np.ones((4, 5), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(
            index=[
                "GeneA",
                "Blank-1",
                "NegControlProbe_00001",
                "UnassignedCodeword_0001",
                "GeneB",
            ]
        ),
    )

    filtered = remove_control_features(adata)

    assert list(filtered.var_names) == ["GeneA", "GeneB"]
    summary = filtered.uns["merxen_clustering_squidpy"]["control_feature_filter"]
    assert summary["n_features_before"] == 5
    assert summary["n_control_features_removed"] == 3
    assert summary["removed_control_features"] == [
        "Blank-1",
        "NegControlProbe_00001",
        "UnassignedCodeword_0001",
    ]


def test_load_atlas_marker_sets_joins_taxonomy_and_collapses_labels(
    tmp_path: Path,
) -> None:
    """MapMyCells marker keys should resolve through Allen taxonomy metadata."""
    marker_path = tmp_path / "markers.json"
    marker_path.write_text(
        """
{
  "CCN202210140_SUPC/CS1": ["ENSG1", "ENSG2"],
  "CCN202210140_SUPC/CS2": ["ENSG3"],
  "CCN202210140_SUPC/CS3": ["ENSG4"],
  "CCN202210140_SUBC/ignored": ["ENSG4"]
}
""".strip()
    )
    taxonomy_path = tmp_path / "cluster_annotation_term.csv"
    taxonomy_path.write_text(
        "\n".join(
            [
                "label,name,cluster_annotation_term_set_label",
                "CS1,Oligodendrocyte,CCN202210140_SUPC",
                "CS2,Upper-layer intratelencephalic,CCN202210140_SUPC",
                "CS3,MGE interneuron,CCN202210140_SUPC",
                "ignored,Ignored,CCN202210140_SUBC",
            ]
        )
        + "\n"
    )
    membership_path = tmp_path / "cluster_to_cluster_annotation_membership.csv"
    membership_path.write_text(
        "\n".join(
            [
                "cluster_annotation_term_label,cluster_annotation_term_set_label,"
                "cluster_alias,cluster_annotation_term_name",
                "CS2,CCN202210140_SUPC,1,Upper-layer intratelencephalic",
                "CS202210140_3820,CCN202210140_NEUR,1,VGLUT1",
                "CS3,CCN202210140_SUPC,2,MGE interneuron",
                "CS202210140_3810,CCN202210140_NEUR,2,GABA",
            ]
        )
        + "\n"
    )

    marker_sets = load_atlas_marker_sets(
        marker_path,
        taxonomy_path,
        cluster_membership_path=membership_path,
    )

    assert [marker_set.label_name for marker_set in marker_sets] == [
        "Oligodendrocyte",
        "Upper-layer intratelencephalic",
        "MGE interneuron",
    ]
    assert marker_sets[0].broad_class == "Oligodendrocytes"
    assert marker_sets[1].broad_class == "Neurons"
    assert marker_sets[1].neuron_split == "Excitatory"
    assert marker_sets[2].neuron_split == "Inhibitory"
    assert collapse_atlas_label_to_broad_class("Choroid plexus") == ("Choroid plexus")


def test_wmb_marker_sets_fill_missing_classes_from_subclass_markers(
    tmp_path: Path,
) -> None:
    """WMB classes omitted by the marker release should reuse child markers."""
    marker_path = tmp_path / "mouse_markers.json"
    marker_path.write_text(
        """
{
  "CCN20230722_CLAS/class_glut": ["ENSMUSG1"],
  "CCN20230722_SUBC/subclass_dopa": ["ENSMUSG2", "ENSMUSG3"],
  "CCN20230722_SUBC/subclass_oec": ["ENSMUSG4"]
}
""".strip()
    )
    taxonomy_path = tmp_path / "cluster_annotation_term.csv"
    pd.DataFrame(
        {
            "label": ["class_glut", "class_dopa", "class_oec"],
            "name": ["01 IT-ET Glut", "21 MB Dopa", "32 OEC"],
            "cluster_annotation_term_set_label": ["CCN20230722_CLAS"] * 3,
        }
    ).to_csv(taxonomy_path, index=False)
    membership_path = tmp_path / "cluster_membership.csv"
    pd.DataFrame(
        {
            "cluster_annotation_term_label": [
                "class_glut",
                "neur_glut",
                "class_dopa",
                "subclass_dopa",
                "neur_dopa",
                "class_oec",
                "subclass_oec",
                "neur_none",
            ],
            "cluster_annotation_term_set_label": [
                "CCN20230722_CLAS",
                "CCN20230722_NEUR",
                "CCN20230722_CLAS",
                "CCN20230722_SUBC",
                "CCN20230722_NEUR",
                "CCN20230722_CLAS",
                "CCN20230722_SUBC",
                "CCN20230722_NEUR",
            ],
            "cluster_alias": ["1", "1", "2", "2", "2", "3", "3", "3"],
            "cluster_annotation_term_name": [
                "01 IT-ET Glut",
                "Glut",
                "21 MB Dopa",
                "Dopaminergic subclass",
                "Dopa",
                "32 OEC",
                "OEC subclass",
                "none",
            ],
        }
    ).to_csv(membership_path, index=False)

    marker_sets = load_atlas_marker_sets(
        marker_path,
        taxonomy_path,
        marker_level="CCN20230722_CLAS",
        cluster_membership_path=membership_path,
    )

    by_label = {marker_set.label_id: marker_set for marker_set in marker_sets}
    assert by_label["class_dopa"].marker_ids == ("ENSMUSG2", "ENSMUSG3")
    assert by_label["class_oec"].marker_ids == ("ENSMUSG4",)
    assert by_label["class_glut"].neuron_split == "Excitatory"
    assert by_label["class_dopa"].neuron_split == "Other"
    assert by_label["class_oec"].broad_class == "OEC"


def test_score_clusters_by_atlas_markers_resolves_ensembl_then_symbol() -> None:
    """Synthetic marker expression should recover known broad labels."""
    adata = ad.AnnData(
        X=np.array(
            [
                [9, 8, 1, 1],
                [8, 7, 1, 1],
                [1, 1, 8, 9],
                [1, 1, 7, 8],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"leiden_broad": ["0", "0", "1", "1"]}),
        var=pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB", "GeneC", "GeneD"],
                "ensembl_id": ["ENSGA", "ENSGB", "ENSGC", "ENSGD"],
            },
            index=["GeneA", "GeneB", "GeneC", "GeneD"],
        ),
    )
    marker_sets = [
        AtlasMarkerSet(
            level="level",
            label_id="oligo",
            label_name="Oligodendrocyte",
            broad_class="Oligodendrocytes",
            marker_ids=("ENSGA", "ENSGB"),
        ),
        AtlasMarkerSet(
            level="level",
            label_id="astro",
            label_name="Astrocyte",
            broad_class="Astrocytes",
            marker_ids=("GeneC", "GeneD"),
        ),
    ]

    assignments, scores, markers = score_clusters_by_atlas_markers(
        adata,
        cluster_key="leiden_broad",
        marker_sets=marker_sets,
        min_marker_overlap=2,
    )

    label_by_cluster = dict(
        zip(assignments["cluster"], assignments["atlas_label"], strict=True)
    )
    assert label_by_cluster == {"0": "Oligodendrocyte", "1": "Astrocyte"}
    assert set(scores["atlas_label"]) == {"Oligodendrocyte", "Astrocyte"}
    assert set(markers["n_resolved_markers"]) == {2}


def test_score_clusters_by_atlas_markers_uses_marker_alias_lookup() -> None:
    """Reference gene metadata should bridge Ensembl markers to symbol panels."""
    adata = ad.AnnData(
        X=np.array(
            [
                [9, 8, 1, 1],
                [8, 7, 1, 1],
                [1, 1, 8, 9],
                [1, 1, 7, 8],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame({"leiden_broad": ["0", "0", "1", "1"]}),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneD"]),
    )
    marker_sets = [
        AtlasMarkerSet(
            level="level",
            label_id="oligo",
            label_name="Oligodendrocyte",
            broad_class="Oligodendrocytes",
            marker_ids=("ENSGA", "ENSGB"),
        ),
        AtlasMarkerSet(
            level="level",
            label_id="astro",
            label_name="Astrocyte",
            broad_class="Astrocytes",
            marker_ids=("ENSGC", "ENSGD"),
        ),
    ]

    assignments, _, markers = score_clusters_by_atlas_markers(
        adata,
        cluster_key="leiden_broad",
        marker_sets=marker_sets,
        marker_alias_lookup={
            "ENSGA": "GeneA",
            "ENSGB": "GeneB",
            "ENSGC": "GeneC",
            "ENSGD": "GeneD",
        },
        min_marker_overlap=2,
    )

    label_by_cluster = dict(
        zip(assignments["cluster"], assignments["atlas_label"], strict=True)
    )
    assert label_by_cluster == {"0": "Oligodendrocyte", "1": "Astrocyte"}
    assert list(markers["n_resolved_markers"]) == [2, 2]


def test_score_clusters_by_atlas_markers_unknown_for_low_overlap() -> None:
    """Panels with too little marker overlap should still get stable outputs."""
    adata = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"leiden_broad": ["0", "0", "1", "1"]}),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )

    assignments, scores, markers = score_clusters_by_atlas_markers(
        adata,
        cluster_key="leiden_broad",
        marker_sets=[
            AtlasMarkerSet(
                level="level",
                label_id="missing",
                label_name="Microglia",
                broad_class="Microglia",
                marker_ids=("MissingGene",),
            )
        ],
        min_marker_overlap=2,
        unknown_label="Mixed/Unknown",
    )

    assert list(assignments["atlas_label"]) == ["Mixed/Unknown", "Mixed/Unknown"]
    assert scores.empty
    assert list(scores.columns) == [
        "cluster",
        "label_id",
        "atlas_label",
        "broad_class",
        "score",
        "n_markers",
        "resolved_markers",
    ]
    assert list(markers["n_resolved_markers"]) == [0]


def test_neuron_split_marker_sets_group_exc_inh_and_other() -> None:
    """Neuron supercluster marker sets should collapse to Exc/Inh/Other groups."""
    marker_sets = [
        AtlasMarkerSet(
            level="level",
            label_id="exc",
            label_name="Upper-layer intratelencephalic",
            broad_class="Neurons",
            marker_ids=("ENSG1",),
        ),
        AtlasMarkerSet(
            level="level",
            label_id="inh",
            label_name="MGE interneuron",
            broad_class="Neurons",
            marker_ids=("ENSG2",),
        ),
        AtlasMarkerSet(
            level="level",
            label_id="other",
            label_name="Unclassified neuron",
            broad_class="Neurons",
            marker_ids=("ENSG3",),
        ),
    ]

    split_sets = _make_neuron_split_marker_sets(marker_sets)

    markers_by_split = {
        marker_set.label_name: marker_set.marker_ids for marker_set in split_sets
    }
    assert markers_by_split == {
        "Excitatory": ("ENSG1",),
        "Inhibitory": ("ENSG2",),
        "Other": ("ENSG3",),
    }


def test_plot_annotation_score_heatmap_writes_png_and_pdf(tmp_path: Path) -> None:
    """Annotation heatmaps should be emitted as regular plot artifacts."""
    score_table = pd.DataFrame(
        {
            "cluster": ["0", "1"],
            "atlas_label": ["Astrocyte", "Microglia"],
            "score": [1.2, 0.9],
        }
    )

    output_path = plot_annotation_score_heatmap(
        score_table,
        tmp_path / "scores.png",
        title="Synthetic scores",
    )

    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()


def test_clustering_squidpy_config_defaults_enable_hierarchical_mode() -> None:
    """Minimal stage configs should run broad annotation and subclustering."""
    cfg = ClusteringSquidpyConfig.model_validate(
        {
            "pair_id": "pair1",
            "output_dir": "/tmp/out",
            "samples": [
                {
                    "sample_id": "sample1",
                    "platform": "MERSCOPE",
                    "zarr_path": "/tmp/input.zarr",
                }
            ],
        }
    )

    assert cfg.hierarchical_enabled is True
    assert cfg.leiden_resolution == 0.5
    assert cfg.broad_round.leiden_resolution == 0.2
    assert cfg.subcluster_round.leiden_resolution == 0.5
    assert cfg.spatial_point_size == 0.5
    assert cfg.spatial_scatter_point_size == 2.0
    assert cfg.write_spatialdata_table is True
    assert cfg.broad_annotation.reference_atlas == "whb"
    assert cfg.broad_annotation.marker_level == "CCN202210140_SUPC"


def test_mouse_clustering_annotation_uses_wmb_defaults() -> None:
    """Selecting WMB should not retain human paths or taxonomy levels."""
    cfg = ClusteringSquidpyConfig.model_validate(
        {
            "pair_id": "pair1",
            "output_dir": "/tmp/out",
            "samples": [],
            "broad_annotation": {"reference_atlas": "wmb"},
        }
    )

    assert cfg.broad_annotation.marker_lookup_path is None
    assert cfg.broad_annotation.taxonomy_metadata_path is None
    assert cfg.broad_annotation.cluster_membership_path is None
    assert cfg.broad_annotation.marker_level == "CCN20230722_CLAS"


@pytest.mark.parametrize(
    ("atlas_label", "expected"),
    [
        ("01 IT-ET Glut", "Neurons"),
        ("06 CTX-CGE GABA", "Neurons"),
        ("15 HY Gnrh1 Glut", "Neurons"),
        ("21 MB Dopa", "Neurons"),
        ("22 MB-HB Sero", "Neurons"),
        ("25 Pineal Glut", "Neurons"),
        ("30 Astro-Epen", "Astrocytes/Ependymal"),
        ("31 OPC-Oligo", "Oligodendrocyte lineage"),
        ("32 OEC", "OEC"),
        ("33 Vascular", "Vascular cells"),
        ("34 Immune", "Microglia"),
    ],
)
def test_wmb_classes_collapse_to_merxen_broad_classes(
    atlas_label: str,
    expected: str,
) -> None:
    """WMB class names should feed the existing hierarchical branches."""
    assert collapse_atlas_label_to_broad_class(atlas_label) == expected


def test_extract_gene_ids_accepts_mouse_ensembl_ids() -> None:
    """Mouse Ensembl IDs must survive clustering metadata extraction."""
    var = pd.DataFrame(
        {
            "ensembl_id": ["ENSMUSG00000000001", "control", ""],
            "gene": ["Gnai3", "Blank-1", "Missing"],
        },
        index=["Gnai3", "Blank-1", "Missing"],
    )

    lookup = clustering_mod._extract_gene_id_lookup_from_var(var, var.index)

    assert lookup == {"Gnai3": "ENSMUSG00000000001"}


def test_wmb_cache_discovery_stays_within_mouse_taxonomy(tmp_path: Path) -> None:
    """WMB lookup discovery must not select nearby WHB cache artifacts."""
    whb_dir = tmp_path / "abc_whb" / "metadata" / "WHB-taxonomy" / "20240330"
    wmb_dir = tmp_path / "abc_atlas" / "metadata" / "WMB-taxonomy" / "20230630"
    whb_dir.mkdir(parents=True)
    wmb_dir.mkdir(parents=True)
    (whb_dir / "cluster_annotation_term.csv").write_text("human\n")
    wmb_taxonomy = wmb_dir / "cluster_annotation_term.csv"
    wmb_taxonomy.write_text("mouse\n")
    wmb_markers = tmp_path / "mouse_markers_230821.json"
    wmb_markers.write_text("{}\n")

    assert (
        clustering_mod._find_cached_marker_lookup(
            tmp_path,
            reference_atlas="wmb",
        )
        == wmb_markers
    )
    assert (
        clustering_mod._find_cached_taxonomy_metadata(
            tmp_path,
            reference_atlas="wmb",
        )
        == wmb_taxonomy
    )


def test_wmb_gene_csv_provides_marker_id_aliases(tmp_path: Path) -> None:
    """The compact Allen gene table should resolve mouse Ensembl markers."""
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(
        {
            "gene_identifier": ["ENSMUSG00000000001", "not-an-ensembl-id"],
            "gene_symbol": ["Gnai3", "Ignored"],
        }
    ).to_csv(gene_path, index=False)

    lookup = clustering_mod._reference_gene_symbol_lookup(gene_path)

    assert lookup["ENSMUSG00000000001"] == "Gnai3"
    assert lookup["GNAI3"] == "ENSMUSG00000000001"
    assert "IGNORED" not in lookup


def test_wmb_gene_csv_avoids_opening_large_expression_shards(tmp_path: Path) -> None:
    """Compact WMB gene metadata should take precedence over raw H5AD files."""
    gene_dir = tmp_path / "abc_atlas" / "metadata" / "WMB-10X" / "release"
    matrix_dir = (
        tmp_path / "abc_atlas" / "expression_matrices" / "WMB-10Xv3" / "release"
    )
    gene_dir.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    gene_path = gene_dir / "gene.csv"
    gene_path.write_text("gene_identifier,gene_symbol\n")
    (matrix_dir / "WMB-10Xv3-test-raw.h5ad").write_text("not opened")

    paths = clustering_mod._find_cached_reference_gene_metadata_paths(
        tmp_path,
        reference_atlas="wmb",
    )

    assert paths == [gene_path]


def test_clustered_spatialdata_table_key_uses_segmentation_defaults() -> None:
    """Clustered SpatialData table names should be stable for analysis branches."""
    assert (
        _clustered_spatialdata_table_key("table_MOSAIK_proseg", "reseg")
        == "table_MOSAIK_proseg_clustering_squidpy"
    )
    assert (
        _clustered_spatialdata_table_key("table_original", "original_seg")
        == "table_original_clustering_squidpy"
    )
    assert (
        _clustered_spatialdata_table_key("table_custom", None)
        == "table_custom_clustering_squidpy"
    )


def test_build_clustered_spatialdata_table_retargets_region() -> None:
    """Existing SpatialData attrs should be rebuilt for the clustering shape."""
    adata = ad.AnnData(
        X=np.array([[1, 0], [0, 2]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "cell": ["c1", "c2"],
                "region": pd.Categorical(["MOSAIK_proseg", "MOSAIK_proseg"]),
                "leiden": pd.Categorical(["0", "1"]),
                "broad_class": pd.Categorical(["Astrocytes", "Neurons"]),
            },
            index=pd.Index(["c1", "c2"], name="cell"),
        ),
        var=pd.DataFrame({"gene": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["X_umap"] = np.array([[0.0, 1.0], [1.0, 0.0]])
    adata.obsm["spatial"] = np.array([[10.0, 11.0], [20.0, 21.0]])
    adata.uns["spatialdata_attrs"] = {
        "region": "MOSAIK_proseg",
        "region_key": "region",
        "instance_key": "cell",
    }
    adata.uns["merxen_clustering_squidpy"] = {
        "table_key": "table_MOSAIK_proseg",
        "shape_key": "MOSAIK_proseg_aligned_nonrigid",
    }

    table = build_clustered_spatialdata_table(
        adata,
        output_table_key="table_MOSAIK_proseg_clustering_squidpy",
        output_region="MOSAIK_proseg_aligned_nonrigid",
        source_table_key="table_MOSAIK_proseg",
        source_region="MOSAIK_proseg",
    )

    assert table.uns["spatialdata_attrs"] == {
        "region": "MOSAIK_proseg_aligned_nonrigid",
        "region_key": "region",
        "instance_key": "cell",
    }
    assert table.obs["region"].astype(str).tolist() == [
        "MOSAIK_proseg_aligned_nonrigid",
        "MOSAIK_proseg_aligned_nonrigid",
    ]
    assert "counts" in table.layers
    assert "X_umap" in table.obsm
    assert "spatial" in table.obsm
    assert list(table.obs["broad_class"].astype(str)) == ["Astrocytes", "Neurons"]
    assert table.uns["merxen_clustering_squidpy"]["source_table_key"] == (
        "table_MOSAIK_proseg"
    )
    assert table.uns["merxen_clustering_squidpy"]["written_table_key"] == (
        "table_MOSAIK_proseg_clustering_squidpy"
    )
    assert table.uns["merxen_clustering_squidpy"]["written_region"] == (
        "MOSAIK_proseg_aligned_nonrigid"
    )
    assert table.uns["merxen_clustering_squidpy"]["spatialdata_region"] == (
        "MOSAIK_proseg_aligned_nonrigid"
    )


def test_write_clustered_spatialdata_table_persists_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Clustered AnnData should be parsed and written as a SpatialData table."""
    adata = ad.AnnData(
        X=np.array([[1, 0], [0, 2]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "cell": ["c1", "c2"],
                "region": pd.Categorical(["MOSAIK_proseg", "MOSAIK_proseg"]),
            },
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.uns["spatialdata_attrs"] = {
        "region": "MOSAIK_proseg",
        "region_key": "region",
        "instance_key": "cell",
    }
    adata.uns["merxen_clustering_squidpy"] = {
        "table_key": "table_MOSAIK_proseg",
        "shape_key": "MOSAIK_proseg_aligned_nonrigid",
    }
    fake_sdata = SimpleNamespace(tables={})
    calls: dict[str, object] = {}

    monkeypatch.setattr(sd, "read_zarr", lambda path: fake_sdata)

    def _fake_write(
        sdata_obj: object,
        key: str,
        element_type: str,
        value: ad.AnnData,
        *,
        overwrite: bool,
    ) -> bool:
        calls["sdata_obj"] = sdata_obj
        calls["key"] = key
        calls["element_type"] = element_type
        calls["value"] = value
        calls["overwrite"] = overwrite
        return True

    monkeypatch.setattr(
        "merxen.io.spatialdata_io.write_or_replace_element",
        _fake_write,
    )

    zarr_path, table_key = write_clustered_spatialdata_table(
        tmp_path / "latest_spatialdata.zarr",
        adata,
        segmentation="reseg",
    )

    assert zarr_path == tmp_path / "latest_spatialdata.zarr"
    assert table_key == "table_MOSAIK_proseg_clustering_squidpy"
    assert calls["sdata_obj"] is fake_sdata
    assert calls["key"] == "table_MOSAIK_proseg_clustering_squidpy"
    assert calls["element_type"] == "tables"
    assert calls["overwrite"] is True
    written = calls["value"]
    assert isinstance(written, ad.AnnData)
    assert written.uns["spatialdata_attrs"]["region"] == (
        "MOSAIK_proseg_aligned_nonrigid"
    )


def test_run_clustering_squidpy_skips_spatialdata_write_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The config flag should allow H5AD-only clustering output."""
    input_adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(3)]),
    )
    input_adata.obsm["spatial"] = np.arange(8, dtype=float).reshape(4, 2)
    input_adata.uns["spatialdata_attrs"] = {
        "region": "MOSAIK_proseg",
        "region_key": "region",
        "instance_key": "cell_id",
    }
    input_adata.uns["merxen_clustering_squidpy"] = {
        "table_key": "table_MOSAIK_proseg",
        "shape_key": "MOSAIK_proseg",
    }
    clustered = input_adata.copy()
    clustered.obs["leiden"] = pd.Categorical(["0", "0", "1", "1"])
    clustered.obsm["X_umap"] = np.arange(8, dtype=float).reshape(4, 2)

    cfg = ClusteringSquidpyConfig.model_validate(
        {
            "pair_id": "pair1",
            "output_dir": tmp_path / "out",
            "samples": [
                {
                    "sample_id": "pair1_MERSCOPE",
                    "platform": "MERSCOPE",
                    "zarr_path": tmp_path / "latest_spatialdata.zarr",
                    "segmentation": "reseg",
                    "table_key": "table_MOSAIK_proseg",
                    "shape_key": "MOSAIK_proseg",
                }
            ],
            "hierarchical_enabled": False,
            "write_spatialdata_table": False,
        }
    )

    monkeypatch.setattr(
        clustering_mod,
        "collect_gene_id_lookup_for_samples",
        lambda config: {},
    )
    monkeypatch.setattr(
        clustering_mod,
        "load_spatialdata_adata",
        lambda *args, **kwargs: input_adata.copy(),
    )
    monkeypatch.setattr(
        clustering_mod,
        "run_scanpy_clustering",
        lambda *args, **kwargs: clustered.copy(),
    )
    monkeypatch.setattr(
        clustering_mod,
        "plot_qc_histograms",
        lambda _adata, output_path, **kwargs: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "save_qc_metrics",
        lambda _adata, output_path: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "plot_umap",
        lambda _adata, output_path, **kwargs: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "plot_spatial_scatter",
        lambda _adata, output_path, **kwargs: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "plot_spatial_cluster_grid",
        lambda _adata, output_path, **kwargs: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "save_clustered_adata",
        lambda _adata, output_path: Path(output_path),
    )
    monkeypatch.setattr(
        clustering_mod,
        "write_clustered_spatialdata_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("SpatialData write should be skipped")
        ),
    )

    results = run_clustering_squidpy(cfg)

    sample_results = results["pair1_MERSCOPE"]
    assert "h5ad" in sample_results
    assert "spatialdata_table_key" not in sample_results
    assert "spatialdata_zarr" not in sample_results


def test_run_gpu_clustering_uses_chunked_pca_for_sparse_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse GPU PCA should use a dtype consistent with cuML IncrementalPCA."""
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((20, 6), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(6)]),
    )
    calls: dict[str, object] = {}

    def fake_to_gpu(data: ad.AnnData) -> None:
        calls["to_gpu"] = data
        calls["gpu_input_dtype"] = data.X.dtype

    fake_get = SimpleNamespace(
        anndata_to_GPU=fake_to_gpu,
        anndata_to_CPU=lambda data: calls.setdefault("to_cpu", data),
    )

    def fake_pca(data: ad.AnnData, **kwargs: object) -> None:
        calls["pca"] = kwargs
        data.obsm["X_pca"] = np.ones((data.n_obs, 3), dtype=np.float32)

    def fake_neighbors(data: ad.AnnData, **kwargs: object) -> None:
        calls["neighbors"] = kwargs

    fake_pp = SimpleNamespace(pca=fake_pca, neighbors=fake_neighbors)
    fake_tl = SimpleNamespace(
        umap=lambda data, **kwargs: calls.setdefault("umap", kwargs),
        leiden=lambda data, **kwargs: calls.setdefault("leiden", kwargs),
    )
    fake_rsc = SimpleNamespace(get=fake_get, pp=fake_pp, tl=fake_tl)
    monkeypatch.setitem(sys.modules, "rapids_singlecell", fake_rsc)

    gpu_used = _run_gpu_clustering(
        adata,
        max_pcs=3,
        n_pcs_for_neighbors=3,
        effective_neighbors=5,
        umap_min_dist=0.4,
        umap_spread=1.2,
        leiden_resolution=0.8,
        random_seed=7,
    )

    assert gpu_used is True
    assert calls["to_gpu"] is adata
    assert calls["to_cpu"] is adata
    assert calls["gpu_input_dtype"] == np.dtype(np.float64)
    assert calls["pca"] == {
        "n_comps": 3,
        "random_state": 7,
        "chunked": True,
        "chunk_size": adata.n_obs,
    }
    assert calls["neighbors"] == {
        "n_neighbors": 5,
        "n_pcs": 3,
        "use_rep": "X_pca",
        "random_state": 7,
    }


def test_plot_spatial_scatter_suppresses_squidpy_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Image-less spatial scatter should not emit Squidpy library warnings."""
    adata = ad.AnnData(
        X=np.ones((5, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(["0", "1", "0", "1", "2"])},
            index=[f"cell{i}" for i in range(5)],
        ),
        var=pd.DataFrame(index=["Gene0", "Gene1"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(adata.n_obs), np.arange(adata.n_obs)]
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        output_path = plot_spatial_scatter(
            adata,
            tmp_path / "spatial.png",
            point_size=0.2,
        )

    captured = capsys.readouterr()
    warning_text = "\n".join(str(item.message) for item in recorded)
    output_text = f"{captured.out}\n{captured.err}"
    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()
    assert "No data for colormapping provided via 'c'" not in warning_text
    assert "Please specify a valid `library_id`" not in output_text


def test_spatial_axis_cleanup_adds_scale_bar_without_coordinate_labels() -> None:
    """Spatial helper should hide raw coordinate axes and add a 200 um scale bar."""
    fig, ax = plt.subplots()
    ax.scatter([0, 100, 300], [0, 50, 100])
    ax.set_xlabel("spatial1")
    ax.set_ylabel("spatial2")

    _clean_spatial_axis(ax)
    _add_spatial_scale_bar(ax, length_um=200)

    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert not ax.get_xticks().size
    assert not ax.get_yticks().size
    assert any(text.get_text() == "200 um" for text in ax.texts)
    plt.close(fig)


def test_plot_spatial_cluster_grid_writes_png_and_pdf(tmp_path: Path) -> None:
    """Spatial cluster grid should highlight each Leiden cluster separately."""
    adata = ad.AnnData(
        X=np.ones((6, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(["0", "1", "0", "1", "2", "2"])},
            index=[f"cell{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["Gene0", "Gene1"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(adata.n_obs), np.arange(adata.n_obs)]
    )

    output_path = plot_spatial_cluster_grid(
        adata,
        tmp_path / "spatial_leiden_grid.png",
        point_size_highlight=0.4,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()


def test_group_gene_dotplot_writes_summary_plot(tmp_path: Path) -> None:
    """Branch dotplot helpers should summarize mean and fraction by group."""
    adata = ad.AnnData(
        X=np.array(
            [
                [3.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        ),
        obs=pd.DataFrame(
            {"leiden_subcluster": ["0", "0", "1", "1"]},
            index=[f"cell{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC"]),
    )

    mean_expression, fraction_expression = compute_group_gene_summary(
        adata,
        ["GeneA", "GeneB", "GeneC"],
        groupby="leiden_subcluster",
    )
    output_path = plot_group_gene_dotplot(
        mean_expression,
        fraction_expression,
        tmp_path / "gene_dotplot.png",
    )

    assert mean_expression.loc["0", "GeneA"] == pytest.approx(2.0)
    assert mean_expression.loc["1", "GeneC"] == pytest.approx(4.5)
    assert fraction_expression.loc["0", "GeneB"] == pytest.approx(0.5)
    assert fraction_expression.loc["1", "GeneC"] == pytest.approx(1.0)
    assert output_path.exists()
    assert output_path.with_suffix(".pdf").exists()


def test_prepare_compute_finalize_clustering_process_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """H5AD should cross the GPU boundary before SpatialData write-back."""
    cfg = ClusteringSquidpyConfig.model_validate(
        {
            "pair_id": "pair1",
            "output_dir": tmp_path / "final",
            "samples": [
                {
                    "sample_id": "pair1_MERSCOPE",
                    "platform": "MERSCOPE",
                    "zarr_path": tmp_path / "latest.zarr",
                    "segmentation": "reseg",
                }
            ],
            "hierarchical_enabled": False,
        }
    )
    prepared_adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(3)]),
    )
    prepared_adata.obsm["spatial"] = np.arange(8, dtype=float).reshape(4, 2)
    monkeypatch.setattr(
        clustering_mod,
        "collect_gene_id_lookup_for_samples",
        lambda _config: {},
    )
    monkeypatch.setattr(
        clustering_mod,
        "load_spatialdata_adata",
        lambda *args, **kwargs: prepared_adata.copy(),
    )

    prepared_dir = tmp_path / "prepared"
    manifest_path = prepare_clustering_squidpy(cfg, prepared_dir)
    assert manifest_path.exists()
    assert (prepared_dir / "merscope/pair1_MERSCOPE_prepared.h5ad").exists()

    def _fake_cluster(
        adata: ad.AnnData,
        config: ClusteringSquidpyConfig,
        *,
        sample: object,
        sample_dir: Path,
    ) -> tuple[ad.AnnData, dict[str, Path | str]]:
        clustered = adata.copy()
        clustered.obs["leiden"] = pd.Categorical(["0", "0", "1", "1"])
        h5ad_path = sample_dir / "pair1_MERSCOPE_clustered.h5ad"
        h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        clustered.write_h5ad(h5ad_path)
        return clustered, {"h5ad": h5ad_path}

    monkeypatch.setattr(clustering_mod, "_cluster_loaded_adata", _fake_cluster)
    computed_dir = tmp_path / "computed"
    compute_clustering_squidpy(cfg, prepared_dir, computed_dir)
    assert (computed_dir / "merscope/pair1_MERSCOPE_clustered.h5ad").exists()

    writes: list[tuple[Path, str | None]] = []

    def _fake_write(
        zarr_path: Path,
        clustered: ad.AnnData,
        *,
        segmentation: str | None,
    ) -> tuple[Path, str]:
        writes.append((Path(zarr_path), segmentation))
        return Path(zarr_path), "table_clustered"

    monkeypatch.setattr(
        clustering_mod,
        "write_clustered_spatialdata_table",
        _fake_write,
    )
    results = finalize_clustering_squidpy(cfg, computed_dir)

    assert (tmp_path / "final/merscope/pair1_MERSCOPE_clustered.h5ad").exists()
    assert writes == [(tmp_path / "latest.zarr", "reseg")]
    assert results["pair1_MERSCOPE"]["spatialdata_table_key"] == "table_clustered"
