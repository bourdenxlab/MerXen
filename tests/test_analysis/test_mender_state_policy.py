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
    SKIPPED_NO_ASSIGNED_STATE,
    cross_platform_comparability,
    excluded_feature_states,
    finalize_mender,
    import_mender_spatialdata,
    prepare_mender,
    skip_reasons,
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


MAPFIRST_TABLE = "table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst"


def _write_clustered(
    tmp_path: Path,
    config: MenderConfig,
    *,
    provenance: str | None = None,
    hierarchy: dict[str, Any] | None = None,
    state_values: list[str] | None = None,
) -> list[str]:
    n_cells = 40
    cell_ids = [f"c{index}" for index in range(n_cells)]
    values = state_values or MAP_FIRST_STATES
    states = [values[index % len(values)] for index in range(n_cells)]
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
    if hierarchy is not None:
        clustered.uns["merxen_hierarchical_clustering"] = hierarchy
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


def test_exclude_policy_skips_sections_without_an_assigned_state() -> None:
    """A refused panel or failed gate is a status, not an error (plan §3.1)."""
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
    states = pd.Series(
        pd.Categorical(["Mixed/Unknown:unresolved", "Neurons/unresolved:unresolved"])
    )
    excluded = excluded_feature_states(states, config)
    assert excluded == ("Mixed/Unknown:unresolved", "Neurons/unresolved:unresolved")
    (reason,) = skip_reasons(states, excluded, config)
    assert "leaves no neighbourhood feature" in reason
    assigned = pd.Series(pd.Categorical(["Mixed/Unknown:unresolved", "Astrocytes:x"]))
    assert (
        skip_reasons(assigned, excluded_feature_states(assigned, config), config) == []
    )
    legacy = config.model_copy(update={"unassigned_state_policy": "state"})
    assert excluded_feature_states(states, legacy) == ()
    assert skip_reasons(states, (), legacy) == []


def test_an_all_unassigned_table_runs_every_step_without_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PREPARE records the skip; COMPUTE, FINALIZE and IMPORT write outputs only."""
    config = _config(tmp_path, source_spatialdata_table=MAPFIRST_TABLE)
    cell_ids = _write_clustered(
        tmp_path,
        config,
        provenance=AnnotationProvenance(
            species="human",
            gate=DatasetGateProvenance(level="failed", warning=True),
            panel=PanelProvenance(panel_trust="refused"),
        ).to_uns_json(),
        hierarchy=MAP_FIRST_RECORD,
        state_values=["Mixed/Unknown:unresolved"],
    )
    _fake_spatialdata(monkeypatch, config, cell_ids)
    prepared = tmp_path / "prepared"
    manifest = json.loads(prepare_mender(config, prepared).read_text())
    assert manifest["status"] == SKIPPED_NO_ASSIGNED_STATE
    assert manifest["annotation"]["gate_level"] == "failed"
    assert manifest["cross_platform_comparable"] is False

    compute_config = tmp_path / "mender_config.json"
    compute_config.write_text(
        json.dumps(
            {
                "sample_id": config.sample_id,
                "platform": config.platform,
                "segmentation": config.segmentation,
            }
        )
    )
    # No MENDER module is importable here: a skipped run must not need one.
    monkeypatch.setitem(sys.modules, "MENDER", None)
    computed = tmp_path / "computed"
    outputs = run_mender_compute(compute_config, prepared, computed)
    compute_manifest = json.loads(outputs["manifest"].read_text())
    assert compute_manifest["status"] == mender_compute.SKIPPED_STATUS
    assert mender_compute.SKIPPED_STATUS == SKIPPED_NO_ASSIGNED_STATE

    finalized = finalize_mender(config, prepared, computed, tmp_path / "mender_out")
    output_manifest = json.loads(finalized["manifest"].read_text())
    assert output_manifest["status"] == SKIPPED_NO_ASSIGNED_STATE
    assert output_manifest["domain_counts"] == {}
    assert output_manifest["annotation"]["panel_trust"] == "refused"
    sample_dir = finalized["manifest"].parent
    assert not list(sample_dir.glob("*.h5ad"))
    assert (sample_dir / "input" / "input_manifest.json").is_file()

    def _no_read(_path: Any) -> None:
        raise AssertionError("a skipped run must not open the SpatialData store")

    monkeypatch.setitem(
        sys.modules, "spatialdata", types.SimpleNamespace(read_zarr=_no_read)
    )
    imported = json.loads(
        import_mender_spatialdata(
            config, sample_dir, tmp_path / "import" / "manifest.json"
        ).read_text()
    )
    assert imported["imported"] is False
    assert imported["status"] == SKIPPED_NO_ASSIGNED_STATE
    assert imported["status_reasons"] == manifest["status_reasons"]


def test_mender_refuses_a_table_of_the_other_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy and map_first tables share their cells: the key must match the mode."""
    legacy_h5ad_to_mapfirst = _config(
        tmp_path / "a", source_spatialdata_table=MAPFIRST_TABLE
    )
    legacy_h5ad_to_mapfirst.source_h5ad.parent.mkdir(parents=True)
    cell_ids = _write_clustered(tmp_path, legacy_h5ad_to_mapfirst)
    _fake_spatialdata(monkeypatch, legacy_h5ad_to_mapfirst, cell_ids)
    with pytest.raises(ValueError, match="no map_first record"):
        prepare_mender(legacy_h5ad_to_mapfirst, tmp_path / "a" / "prepared")

    mapfirst_h5ad_to_legacy = _config(tmp_path / "b")
    mapfirst_h5ad_to_legacy.source_h5ad.parent.mkdir(parents=True)
    cell_ids = _write_clustered(
        tmp_path, mapfirst_h5ad_to_legacy, hierarchy=MAP_FIRST_RECORD
    )
    _fake_spatialdata(monkeypatch, mapfirst_h5ad_to_legacy, cell_ids)
    with pytest.raises(ValueError, match="map_first table built for"):
        prepare_mender(mapfirst_h5ad_to_legacy, tmp_path / "b" / "prepared")

    # MENDER_IMPORT checks again before it opens the store.
    sample_dir = tmp_path / "b" / "finalized" / "merscope"
    sample_dir.mkdir(parents=True)
    annotated = ad.read_h5ad(mapfirst_h5ad_to_legacy.source_h5ad)
    annotated.obs["mender_domain"] = pd.Categorical(["0"] * annotated.n_obs)
    annotated.uns["merxen_mender"] = {"sample_id": "pair1_MERSCOPE"}
    annotated.write_h5ad(sample_dir / "pair1_MERSCOPE_mender_annotated.h5ad")
    with pytest.raises(ValueError, match="map_first table built for"):
        import_mender_spatialdata(
            mapfirst_h5ad_to_legacy, sample_dir, tmp_path / "b" / "import.json"
        )


def test_niches_are_comparable_across_platforms_only_on_the_mender_state() -> None:
    """§4.9: cross-platform MENDER uses ct_mender_state within the pair scope."""
    base = {
        "pair_id": "p",
        "sample_id": "p_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "source_h5ad": "x.h5ad",
        "spatialdata_path": "x.zarr",
        "source_spatialdata_table": "t",
        "native_shape_key": "s",
        "output_dir": "o",
    }
    full = {"cross_platform": {"statistics_level": "full", "flag": False}}
    broad = {"cross_platform": {"statistics_level": "broad_only", "flag": True}}
    hierarchical = MenderConfig.model_validate(base)
    mender_state = MenderConfig.model_validate(
        {**base, "cell_state_key": "ct_mender_state"}
    )

    assert cross_platform_comparability(hierarchical, full) == {
        "cross_platform_comparable": False,
        "cross_platform_comparable_reasons": ["state_key:hierarchical_cluster"],
    }
    assert cross_platform_comparability(mender_state, full) == {
        "cross_platform_comparable": True,
        "cross_platform_comparable_reasons": [],
    }
    assert cross_platform_comparability(mender_state, broad)[
        "cross_platform_comparable_reasons"
    ] == ["cross_platform:broad_only"]
    assert cross_platform_comparability(mender_state, None)[
        "cross_platform_comparable_reasons"
    ] == ["cross_platform:no_record"]


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


MAP_FIRST_RECORD = {
    "mode": "map_first",
    "table_key_suffix": "mapfirst",
    "mender_unassigned_state_policy": EXCLUDE_FROM_FEATURES_POLICY,
    "cross_platform_statistics_level": "broad_only",
    "cross_platform_json": json.dumps(
        {
            "statistics_level": "broad_only",
            "flag": True,
            "reasons": ["intersection_genes:80<100"],
            "panel_mode": "per_platform",
            "jsd_runs": ["whb_frontal_supc_clus_xpanel"],
            "jsd_purpose": "intersection_xpanel",
            "source": "resolve_summary",
        }
    ),
}


def test_a_map_first_table_brings_its_runs_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MENDER_PREPARE's legacy script names no policy: the table's applies (§4.9).

    The map_first run records ``mender_unassigned_state_policy`` in the
    clustered table; a config that sets a policy explicitly wins, and a legacy
    table (no record) keeps ``state``.
    """
    provenance = AnnotationProvenance(
        species="human",
        gate=DatasetGateProvenance(level="full", warning=False),
        panel=PanelProvenance(panel_trust="validated"),
    ).to_uns_json()
    config = _config(tmp_path, source_spatialdata_table=MAPFIRST_TABLE)
    assert "unassigned_state_policy" not in config.model_fields_set
    cell_ids = _write_clustered(
        tmp_path, config, provenance=provenance, hierarchy=MAP_FIRST_RECORD
    )
    _fake_spatialdata(monkeypatch, config, cell_ids)

    manifest_path = prepare_mender(config, tmp_path / "prepared")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["unassigned_state_policy"] == EXCLUDE_FROM_FEATURES_POLICY
    assert manifest["excluded_feature_states"] == [
        "Mixed/Unknown:unresolved",
        "Neurons/unresolved:unresolved",
    ]
    portable = pd.read_parquet(manifest_path.parent / "mender_input.parquet")
    assert IN_FEATURES_COLUMN in portable.columns
    # The pair's cross-platform scope reaches the MENDER manifest (§8.5), and
    # hierarchical_cluster niches are not comparable across platforms (§4.9).
    assert manifest["annotation"]["cross_platform"]["statistics_level"] == "broad_only"
    assert manifest["cross_platform_comparable"] is False
    assert manifest["cross_platform_comparable_reasons"] == [
        "state_key:hierarchical_cluster",
        "cross_platform:broad_only",
    ]

    explicit = _config(
        tmp_path / "explicit",
        unassigned_state_policy="state",
        source_spatialdata_table=MAPFIRST_TABLE,
    )
    explicit.source_h5ad.parent.mkdir(parents=True, exist_ok=True)
    cell_ids = _write_clustered(
        tmp_path, explicit, provenance=provenance, hierarchy=MAP_FIRST_RECORD
    )
    _fake_spatialdata(monkeypatch, explicit, cell_ids)
    manifest = json.loads(
        prepare_mender(explicit, tmp_path / "explicit" / "prepared").read_text()
    )
    assert manifest["unassigned_state_policy"] == "state"


def test_effective_config_keeps_legacy_tables_and_explicit_policies() -> None:
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
        }
    )
    legacy_uns = {"merxen_hierarchical_clustering": {"mode": "legacy"}}

    assert mender.effective_mender_config(config, legacy_uns) is config
    assert mender.effective_mender_config(config, {}) is config
    updated = mender.effective_mender_config(
        config, {"merxen_hierarchical_clustering": MAP_FIRST_RECORD}
    )
    assert updated.unassigned_state_policy == EXCLUDE_FROM_FEATURES_POLICY
    assert updated.sample_id == config.sample_id
    explicit = MenderConfig.model_validate(
        {**config.model_dump(exclude_unset=True), "unassigned_state_policy": "state"}
    )
    assert (
        mender.effective_mender_config(
            explicit, {"merxen_hierarchical_clustering": MAP_FIRST_RECORD}
        )
        is explicit
    )
