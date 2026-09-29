"""MENDER ``unassigned_state_policy`` on map_first tables (plan §4.9).

Under ``"state"`` (legacy default) every cell state is a neighbourhood
feature and nothing changes. Under ``"exclude_from_features"`` (map_first
default) unassigned cells (``Mixed/Unknown``, ``*/unresolved`` branches)
stay spatial nodes and get a domain, but add no state to any neighbourhood.
The manifests record the policy and, for map_first tables, the dataset gate
and panel trust.
"""

from __future__ import annotations

import ast
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
from scipy import sparse
from shapely.geometry import Point

import merxen.mender_compute as mender_compute
from merxen.analysis import mender
from merxen.analysis.mender import (
    EXCLUDE_FROM_FEATURES_POLICY,
    IN_FEATURES_COLUMN,
    excluded_feature_states,
    prepare_mender,
)
from merxen.annotation.provenance import (
    PROVENANCE_UNS_KEY,
    AnnotationProvenance,
    DatasetGateProvenance,
    PanelProvenance,
)
from merxen.config import MenderConfig
from merxen.mender_compute import (
    exclude_state_features,
    feature_excluded_states,
    run_mender_compute,
)

MAP_FIRST_STATES = [
    "Neurons/Excitatory:Upper-layer intratelencephalic",
    "Mixed/Unknown:unresolved",
    "Astrocytes:Astrocyte",
    "Neurons/unresolved:unresolved",
    "Astrocytes:unresolved",
]


def _config(tmp_path: Path, **overrides: Any) -> MenderConfig:
    values: dict[str, Any] = {
        "pair_id": "pair1",
        "sample_id": "pair1_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "source_h5ad": tmp_path / "clustered.h5ad",
        "spatialdata_path": tmp_path / "latest.zarr",
        "source_spatialdata_table": "table_MOSAIK_proseg_hybrid_clustering_squidpy",
        "native_shape_key": "MOSAIK_proseg_hybrid",
        "output_dir": tmp_path / "mender_out",
    }
    values.update(overrides)
    return MenderConfig.model_validate(values)


def _write_clustered(
    tmp_path: Path, config: MenderConfig, *, provenance: str | None = None
) -> list[str]:
    n_cells = 40
    cell_ids = [f"c{index}" for index in range(n_cells)]
    states = [
        MAP_FIRST_STATES[index % len(MAP_FIRST_STATES)] for index in range(n_cells)
    ]
    obs = pd.DataFrame(
        {"cell_id": cell_ids, "hierarchical_cluster": pd.Categorical(states)},
        index=pd.Index([f"row_{index}" for index in range(n_cells)]),
    )
    clustered = ad.AnnData(X=sparse.csr_matrix((n_cells, 2)), obs=obs)
    clustered.uns["spatialdata_attrs"] = {
        "region": config.native_shape_key,
        "region_key": "region",
        "instance_key": "cell_id",
    }
    if provenance is not None:
        clustered.uns[PROVENANCE_UNS_KEY] = provenance
    clustered.write_h5ad(config.source_h5ad)
    return cell_ids


def _fake_spatialdata(
    monkeypatch: pytest.MonkeyPatch, config: MenderConfig, cell_ids: list[str]
) -> None:
    table = ad.read_h5ad(config.source_h5ad)
    shapes = gpd.GeoDataFrame(
        {"cell_id": cell_ids},
        geometry=[Point(10.0 * (i % 8), 10.0 * (i // 8)) for i in range(len(cell_ids))],
    )
    fake = types.SimpleNamespace(
        tables={config.source_spatialdata_table: table},
        shapes={config.native_shape_key: shapes},
    )
    monkeypatch.setitem(
        sys.modules, "spatialdata", types.SimpleNamespace(read_zarr=lambda _p: fake)
    )


def test_state_policy_keeps_the_legacy_portable_table_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    assert config.unassigned_state_policy == "state"
    cell_ids = _write_clustered(tmp_path, config)
    _fake_spatialdata(monkeypatch, config, cell_ids)
    manifest_path = prepare_mender(config, tmp_path / "prepared")
    portable = pd.read_parquet(manifest_path.parent / "mender_input.parquet")
    assert list(portable.columns) == ["cell_id", "native_x", "native_y", "cell_state"]
    manifest = json.loads(manifest_path.read_text())
    for key in (
        "unassigned_state_policy",
        "excluded_feature_states",
        "n_cells_excluded_from_features",
        "annotation",
    ):
        assert key not in manifest
    assert "Mixed/Unknown:unresolved" in manifest["state_counts"]


def test_exclude_policy_marks_unassigned_states_out_of_the_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, unassigned_state_policy=EXCLUDE_FROM_FEATURES_POLICY)
    cell_ids = _write_clustered(tmp_path, config)
    _fake_spatialdata(monkeypatch, config, cell_ids)
    manifest_path = prepare_mender(config, tmp_path / "prepared")
    portable = pd.read_parquet(manifest_path.parent / "mender_input.parquet")
    assert portable[IN_FEATURES_COLUMN].dtype == bool
    out = set(portable.loc[~portable[IN_FEATURES_COLUMN], "cell_state"].astype(str))
    assert out == {"Mixed/Unknown:unresolved", "Neurons/unresolved:unresolved"}
    # Every cell stays a node of the portable table.
    assert len(portable) == len(cell_ids)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["unassigned_state_policy"] == EXCLUDE_FROM_FEATURES_POLICY
    assert manifest["excluded_feature_states"] == sorted(out)
    assert manifest["n_cells_excluded_from_features"] == 16
    assert "annotation" not in manifest


def test_map_first_tables_record_gate_and_panel_trust_under_both_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = AnnotationProvenance(
        species="human",
        gate=DatasetGateProvenance(level="broad_only", warning=True),
        panel=PanelProvenance(panel_trust="validated"),
    )
    for policy in ("state", EXCLUDE_FROM_FEATURES_POLICY):
        config = _config(tmp_path / policy, unassigned_state_policy=policy)
        config.source_h5ad.parent.mkdir(parents=True, exist_ok=True)
        cell_ids = _write_clustered(
            tmp_path, config, provenance=provenance.to_uns_json()
        )
        _fake_spatialdata(monkeypatch, config, cell_ids)
        manifest = json.loads(
            prepare_mender(config, tmp_path / policy / "prepared").read_text()
        )
        annotation = manifest["annotation"]
        assert annotation["mode"] == "map_first"
        assert annotation["gate_level"] == "broad_only"
        assert annotation["gate_warning"] is True
        assert annotation["panel_trust"] == "validated"
        assert manifest["unassigned_state_policy"] == policy
        expected = (
            []
            if policy == "state"
            else [
                "Mixed/Unknown:unresolved",
                "Neurons/unresolved:unresolved",
            ]
        )
        assert manifest["excluded_feature_states"] == expected


def test_exclude_policy_refuses_sections_without_an_assigned_state() -> None:
    config = MenderConfig.model_validate(
        {
            "pair_id": "p",
            "sample_id": "p_MERSCOPE",
            "platform": "MERSCOPE",
            "segmentation": "proseg_hybrid",
            "source_h5ad": "x.h5ad",
            "spatialdata_path": "x.zarr",
            "source_spatialdata_table": "t",
            "native_shape_key": "s",
            "output_dir": "o",
            "unassigned_state_policy": EXCLUDE_FROM_FEATURES_POLICY,
        }
    )
    states = pd.Series(pd.Categorical(["Mixed/Unknown:unresolved"] * 3))
    with pytest.raises(ValueError, match="leave no feature"):
        excluded_feature_states(states, config)
    legacy = config.model_copy(update={"unassigned_state_policy": "state"})
    assert excluded_feature_states(states, legacy) == ()


class _NamingMender:
    """A fake ``MENDER_single`` with MENDER's feature naming.

    ``run_representation`` counts, per cell and scale, the neighbours of each
    state within ``(scale + 1) * radius`` into features ``ct<i>scale<s>``,
    exactly as ``generate_ct_representation`` names them; clustering assigns
    each cell the domain of its dominant feature.
    """

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
        n_scales = int(self.params["n_scales"])
        per_scale = []
        for scale in range(n_scales):
            radius = float(self.params["nn_para"]) * (scale + 1)
            counts = np.column_stack(
                [
                    ((distances <= radius) & (self.ct_array[None, :] == state)).sum(1)
                    for state in self.ct_unique
                ]
            ).astype(float)
            per_scale.append(counts)
            self.adata.obsm[f"scale{scale}"] = counts
        columns, names = [], []
        for index in range(len(self.ct_unique)):
            for scale in range(n_scales):
                columns.append(per_scale[scale][:, index])
                names.append(f"ct{index}scale{scale}")
        self.adata_MENDER = ad.AnnData(
            X=np.column_stack(columns),
            obs=self.adata.obs.copy(),
            var=pd.DataFrame(index=pd.Index(names)),
        )
        self.adata_MENDER.obsm["spatial"] = coords.copy()

    def run_clustering_normal(self, request: float | int, run_umap: bool) -> None:
        dominant = np.asarray(self.adata_MENDER.X).argmax(axis=1)
        labels = [str(value % 3) for value in dominant]
        if len(set(labels)) < 2:
            labels[0] = "extra"
        self.adata_MENDER.obs["MENDER"] = pd.Categorical(labels)


def _portable(path: Path, *, policy: str) -> pd.DataFrame:
    rows = []
    for index in range(36):
        state = MAP_FIRST_STATES[index % len(MAP_FIRST_STATES)]
        rows.append(
            {
                "cell_id": f"cell_{index}",
                "native_x": float(10 * (index % 6)),
                "native_y": float(10 * (index // 6)),
                "cell_state": state,
            }
        )
    frame = pd.DataFrame(rows)
    frame["cell_state"] = pd.Categorical(frame["cell_state"])
    if policy == EXCLUDE_FROM_FEATURES_POLICY:
        frame[IN_FEATURES_COLUMN] = ~frame["cell_state"].astype(str).isin(
            ["Mixed/Unknown:unresolved", "Neurons/unresolved:unresolved"]
        )
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / "mender_input.parquet", index=False)
    return frame


def _compute_config(tmp_path: Path) -> Path:
    config = {
        "sample_id": "pair1_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "random_seed": 666,
        "nn_mode": "radius",
        "radius_um": 15.0,
        "n_scales": 3,
        "count_rep": "s",
        "include_self": True,
        "clustering_mode": "resolution",
        "leiden_resolution": 1.0,
        "target_k": None,
        "run_umap": False,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("policy", ["state", EXCLUDE_FROM_FEATURES_POLICY])
def test_compute_drops_only_the_excluded_states_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    monkeypatch.setitem(
        sys.modules, "MENDER", types.SimpleNamespace(MENDER_single=_NamingMender)
    )
    portable = _portable(tmp_path / "prepared", policy=policy)
    outputs = run_mender_compute(
        _compute_config(tmp_path), tmp_path / "prepared", tmp_path / "computed"
    )
    model = _NamingMender.last
    assert model is not None
    kept = list(model.adata_MENDER.var_names)
    states = list(model.ct_unique)
    manifest = json.loads(outputs["manifest"].read_text())
    domains = pd.read_parquet(outputs["domains"])
    # Excluded cells stay nodes: every cell gets a domain.
    assert set(domains["cell_id"]) == set(portable["cell_id"])
    if policy == "state":
        assert len(kept) == len(states) * 3
        assert "excluded_feature_states" not in manifest
    else:
        excluded = {"Mixed/Unknown:unresolved", "Neurons/unresolved:unresolved"}
        excluded_index = {i for i, state in enumerate(states) if state in excluded}
        assert len(kept) == (len(states) - 2) * 3
        assert not [
            name
            for name in kept
            if int(name[2 : name.index("scale")]) in excluded_index
        ]
        assert manifest["excluded_feature_states"] == sorted(excluded)
        assert manifest["n_excluded_features"] == 6
        assert manifest["n_cells_excluded_from_features"] == int(
            (~portable[IN_FEATURES_COLUMN]).sum()
        )
    assert manifest["n_context_features"] == len(kept)


def test_feature_exclusion_guards() -> None:
    frame = pd.DataFrame(
        {"cell_state": ["a", "a", "b"], IN_FEATURES_COLUMN: [True, False, True]}
    )
    with pytest.raises(ValueError, match="both in and out"):
        feature_excluded_states(frame)
    assert feature_excluded_states(frame.drop(columns=IN_FEATURES_COLUMN)) == []
    model = types.SimpleNamespace(
        ct_unique=np.array(["a", "b"]),
        adata_MENDER=ad.AnnData(
            X=np.ones((2, 2)), var=pd.DataFrame(index=pd.Index(["ct0scale0", "x"]))
        ),
    )
    with pytest.raises(RuntimeError, match="unexpected MENDER feature"):
        exclude_state_features(model, ["a"])
    only_a = types.SimpleNamespace(
        ct_unique=np.array(["a"]),
        adata_MENDER=ad.AnnData(
            X=np.ones((2, 1)), var=pd.DataFrame(index=pd.Index(["ct0scale0"]))
        ),
    )
    with pytest.raises(RuntimeError, match="no MENDER feature"):
        exclude_state_features(only_a, ["a"])
    assert exclude_state_features(only_a, []) == []


def test_compute_module_stays_python39_compatible() -> None:
    source = Path(mender_compute.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 9))
    assert mender_compute.IN_FEATURES_COLUMN == mender.IN_FEATURES_COLUMN
