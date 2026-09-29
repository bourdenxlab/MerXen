"""map_first tables in the downstream stages: MENDER and cortical depth.

The clustered table of a map_first run carries branch-level labels that
legacy never wrote (``Oligodendrocyte lineage`` in ``broad_class``,
``Neurons/unresolved`` and ``*:unresolved`` states, ``unresolved`` leaves)
and state names with ``/``. These tests push a synthetic map_first section
through FINALIZE's zarr writer and the MENDER prepare / compute / finalize /
import stages under both ``unassigned_state_policy`` values (plan §4.9), and
through cortical depth's cluster-annotation join and violin plots.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from merxen.annotation.vocab import (
    OLIGODENDROCYTE_LINEAGE,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
)
from merxen.clustering.map_first import run_map_first_hierarchy
from merxen.config import CorticalDepthTableConfig, MenderConfig
from merxen.table_keys import clustered_table_key

from .conftest import Section, make_config, make_section

SAMPLE = "P0001_MERSCOPE"


def _map_first(section: Section, tmp_path: Path) -> ad.AnnData:
    clustered, _ = run_map_first_hierarchy(
        section.adata,
        section.labels,
        make_config(tmp_path),
        tmp_path / "hierarchy",
        SAMPLE,
        provenance=section.provenance,
        plots=False,
        stability=False,
    )
    return clustered


def _finalized_zarr(tmp_path: Path, section: Section) -> tuple[Path, str, Path]:
    """Write a zarr with shapes and a source table, then FINALIZE into it."""
    import spatialdata as sd
    from spatialdata.models import ShapesModel, TableModel

    from merxen.analysis.clustering_squidpy import write_clustered_spatialdata_table

    clustered = _map_first(section, tmp_path)
    coords = section.adata.obsm["spatial"]
    shapes = gpd.GeoDataFrame(
        {"radius": np.full(len(coords), 5.0)},
        geometry=[Point(x, y) for x, y in coords],
        index=section.adata.obs["instance_id"].to_numpy(),
    )
    table = ad.AnnData(
        X=section.adata.X.copy(),
        obs=section.adata.obs[["instance_id", "region"]].copy(),
    )
    zarr_path = tmp_path / "latest.zarr"
    sd.SpatialData(
        shapes={"cell_boundaries": ShapesModel.parse(shapes)},
        tables={
            "table": TableModel.parse(
                table,
                region="cell_boundaries",
                region_key="region",
                instance_key="instance_id",
            )
        },
    ).write(zarr_path)
    h5ad_path = tmp_path / f"{SAMPLE}_clustered.h5ad"
    clustered.write_h5ad(h5ad_path)
    _, table_key = write_clustered_spatialdata_table(
        zarr_path,
        ad.read_h5ad(h5ad_path),
        segmentation="proseg_hybrid",
        table_key_suffix="mapfirst",
    )
    return zarr_path, table_key, h5ad_path


class _NamingMender:
    """Fake ``MENDER_single`` with MENDER's per-scale and feature naming."""

    last: _NamingMender | None = None

    def __init__(self, adata: ad.AnnData, ct_obs: str, random_seed: int) -> None:
        self.adata = adata
        self.ct_unique = np.array(adata.obs[ct_obs].cat.categories)
        self.ct_array = np.array(adata.obs[ct_obs])
        _NamingMender.last = self

    def set_MENDER_para(self, **kwargs: Any) -> None:  # noqa: N802
        self.params = kwargs

    def run_representation(self) -> None:
        coords = np.asarray(self.adata.obsm["spatial"])
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        columns, names = [], []
        for scale in range(int(self.params["n_scales"])):
            radius = float(self.params["nn_para"]) * (scale + 1)
            counts = np.column_stack(
                [
                    ((distances <= radius) & (self.ct_array[None, :] == state)).sum(1)
                    for state in self.ct_unique
                ]
            ).astype(float)
            self.adata.obsm[f"scale{scale}"] = counts
        for index in range(len(self.ct_unique)):
            for scale in range(int(self.params["n_scales"])):
                columns.append(self.adata.obsm[f"scale{scale}"][:, index])
                names.append(f"ct{index}scale{scale}")
        self.adata_MENDER = ad.AnnData(
            X=np.column_stack(columns),
            obs=self.adata.obs.copy(),
            var=pd.DataFrame(index=pd.Index(names)),
        )
        self.adata_MENDER.obsm["spatial"] = coords.copy()

    def run_clustering_normal(self, request: float | int, run_umap: bool) -> None:
        dominant = np.asarray(self.adata_MENDER.X).argmax(axis=1)
        self.adata_MENDER.obs["MENDER"] = pd.Categorical(
            [str(value % 4) for value in dominant]
        )


@pytest.mark.parametrize("policy", ["state", "exclude_from_features"])
def test_mender_runs_on_a_map_first_table_under_both_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    import spatialdata as sd

    from merxen.analysis.mender import (
        finalize_mender,
        import_mender_spatialdata,
        prepare_mender,
    )
    from merxen.mender_compute import run_mender_compute

    section = make_section("human")
    zarr_path, table_key, h5ad_path = _finalized_zarr(tmp_path, section)
    config = MenderConfig.model_validate(
        {
            "pair_id": "P0001",
            "sample_id": SAMPLE,
            "platform": "MERSCOPE",
            "segmentation": "proseg_hybrid",
            "source_h5ad": h5ad_path,
            "spatialdata_path": zarr_path,
            "source_spatialdata_table": table_key,
            "native_shape_key": "cell_boundaries",
            "output_dir": tmp_path / "mender_out",
            "radius_um": 150.0,
            "n_scales": 2,
            "unassigned_state_policy": policy,
        }
    )
    prepared = tmp_path / "prepared"
    input_manifest = json.loads(prepare_mender(config, prepared).read_text())
    assert input_manifest["unassigned_state_policy"] == policy
    assert input_manifest["annotation"]["gate_level"] == "full"
    excluded = input_manifest["excluded_feature_states"]
    if policy == "state":
        assert excluded == []
    else:
        assert excluded == [
            "Mixed/Unknown:unresolved",
            "Neurons/unresolved:unresolved",
            "Oligodendrocyte lineage/unresolved:unresolved",
        ]
    compute_config = tmp_path / "mender_config.json"
    compute_config.write_text(config.model_dump_json())
    monkeypatch.setitem(
        sys.modules, "MENDER", types.SimpleNamespace(MENDER_single=_NamingMender)
    )
    computed = run_mender_compute(compute_config, prepared, tmp_path / "computed")
    compute_manifest = json.loads(computed["manifest"].read_text())
    model = _NamingMender.last
    assert model is not None
    n_states = len(model.ct_unique)
    assert compute_manifest["n_context_features"] == (n_states - len(excluded)) * 2
    finalized = finalize_mender(
        config, prepared, tmp_path / "computed", tmp_path / "final"
    )
    record = ad.read_h5ad(finalized["annotated_h5ad"]).uns["merxen_mender"]
    assert "state_counts" not in record  # "/" in map_first states
    assert json.loads(str(record["state_counts_json"]))
    import_mender_spatialdata(
        config, tmp_path / "final", tmp_path / "import_manifest.json"
    )
    table = sd.read_zarr(zarr_path).tables[table_key]
    assert table.obs["mender_domain"].notna().all()
    assert table.n_obs == int(section.labels["in_table"].sum())
    assert table.uns["merxen_mender"]["unassigned_state_policy"] == policy
    # Unassigned cells are still nodes: they carry a domain.
    unassigned = (
        table.obs["hierarchical_cluster"].astype(str).str.startswith(UNASSIGNED_LABEL)
    )
    assert unassigned.any()
    assert table.obs.loc[unassigned, "mender_domain"].notna().all()


def _depth_cells(clustered: ad.AnnData) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ids = clustered.obs["instance_id"].astype(str).to_numpy()
    depth = rng.uniform(0.0, 1.0, size=len(ids))
    depth[:5] = np.nan  # outside the ribbon
    return pd.DataFrame(
        {"laplace_depth": depth, "equivolumetric_depth": depth},
        index=pd.Index(ids),
    )


def test_cortical_depth_reads_map_first_labels_and_gate(tmp_path: Path) -> None:
    from merxen.cortical_depth.pipeline import (
        _load_cluster_annotations,
        _plot_depth_distributions_by_cluster,
        cluster_annotation_summary,
    )
    from merxen.cortical_depth.plotting import (
        _depth_violin_frame,
        _ordered_cluster_labels,
    )

    section = make_section("human", gate_level="broad_only")
    clustered = _map_first(section, tmp_path)
    table_config = CorticalDepthTableConfig(
        segmentation="proseg_hybrid", table_key="table"
    )
    key = clustered_table_key("table", "proseg_hybrid")
    fake = types.SimpleNamespace(tables={key: clustered})
    annotations = _load_cluster_annotations(fake, table_config)
    assert annotations is not None
    broad = set(annotations["broad_class"].astype(str))
    assert {OLIGODENDROCYTE_LINEAGE, UNASSIGNED_LABEL} <= broad
    assert set(annotations["subcluster_label"].astype(str)) == {UNRESOLVED_LABEL}

    cells = _depth_cells(clustered).join(annotations)
    frame = _depth_violin_frame(
        cells, depth_column="laplace_depth", group_columns=["broad_class"]
    )
    assert OLIGODENDROCYTE_LINEAGE in set(frame["broad_class"])
    assert UNASSIGNED_LABEL not in set(frame["broad_class"])
    assert frame["laplace_depth"].notna().all()
    order = _ordered_cluster_labels(cells, "broad_class", frame)
    assert OLIGODENDROCYTE_LINEAGE in order
    assert UNASSIGNED_LABEL not in order
    paths = _plot_depth_distributions_by_cluster(
        cells,
        segmentation_dir=tmp_path / "depth",
        sample_stem="p0001_merscope_proseg_hybrid",
        segmentation="proseg_hybrid",
    )
    assert len(paths) == 4
    for path in paths.values():
        assert path.exists()
        assert path.with_suffix(".pdf").exists()

    summary = cluster_annotation_summary(fake, table_config)
    assert summary is not None
    assert summary["gate_level"] == "broad_only"
    assert summary["gate_warning"] is True
    assert summary["panel_trust"] == "validated"
    # Legacy tables (no annotation provenance) add nothing to the summary.
    legacy = clustered.copy()
    del legacy.uns["merxen_annotation_json"]
    assert (
        cluster_annotation_summary(
            types.SimpleNamespace(tables={key: legacy}), table_config
        )
        is None
    )
    assert (
        cluster_annotation_summary(types.SimpleNamespace(tables={}), table_config)
        is None
    )


def test_cortical_depth_subcluster_violins_group_mapped_leaves(tmp_path: Path) -> None:
    from merxen.cortical_depth.pipeline import _load_cluster_annotations
    from merxen.cortical_depth.plotting import _depth_violin_frame

    section = make_section("human")
    clustered = _map_first(section, tmp_path)
    table_config = CorticalDepthTableConfig(
        segmentation="proseg_hybrid", table_key="table"
    )
    key = clustered_table_key("table", "proseg_hybrid")
    annotations = _load_cluster_annotations(
        types.SimpleNamespace(tables={key: clustered}), table_config
    )
    assert annotations is not None
    cells = _depth_cells(clustered).join(annotations)
    frame = _depth_violin_frame(
        cells,
        depth_column="laplace_depth",
        group_columns=["broad_class", "subcluster_label"],
    )
    lineage = frame.loc[frame["broad_class"] == OLIGODENDROCYTE_LINEAGE]
    assert set(lineage["subcluster_label"]) == {UNRESOLVED_LABEL}
    oligo = frame.loc[frame["broad_class"] == "Oligodendrocytes"]
    assert set(oligo["subcluster_label"]) == {"Oligodendrocyte"}
