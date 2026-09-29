"""The map_first compute and FINALIZE steps of CLUSTERING_MAP_FIRST (plan §3.5, §4.8).

CLUSTERING_SQUIDPY_COMPUTE_CPU runs ``python -m merxen.clustering_squidpy_stages
compute --mode map_first --labels-dir … --table-key-suffix …
--mender-unassigned-state-policy …`` on the PREPARE outputs and RESOLVE's
label tables. FINALIZE's (legacy, pinned) script passes neither a mode nor a
suffix, so the clustered H5AD carries its own suffix; MENDER_PREPARE's
(legacy) script passes no policy, so the H5AD carries the run's policy too.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import pytest

from merxen.analysis.clustering_squidpy import (
    MapFirstComputeOptions,
    compute_clustering_squidpy,
    finalize_clustering_squidpy,
    finalize_table_key_suffix,
)
from merxen.annotation.pipeline import write_label_table
from merxen.annotation.provenance import annotation_manifest_filename
from merxen.annotation.schema import label_table_filename
from merxen.clustering import map_first as map_first_module
from merxen.clustering.cross_platform import resolve_summary_filename
from merxen.clustering.map_first import (
    CROSS_PLATFORM_FIELD,
    HIERARCHICAL_UNS_KEY,
    recorded_cross_platform,
    recorded_mender_policy,
    recorded_table_key_suffix,
)
from merxen.clustering_squidpy_stages import apply_mode_overrides
from merxen.clustering_squidpy_stages import main as stages_main
from merxen.config import ClusteringSquidpyConfig, ClusteringSquidpySampleConfig
from merxen.table_keys import clustered_table_key

from .conftest import CONTROL_FEATURES, Section, make_section

PAIR = "P0001"
PLATFORMS = ("MERSCOPE", "XENIUM")
CROSS_PLATFORM = {
    "panel_mode": "per_platform",
    "jsd_run": ["whb_frontal_supc_clus_xpanel"],
    "jsd_purpose": "intersection_xpanel",
    "statistics_level": "broad_only",
    "flag": True,
    "reasons": ["intersection_genes:80<100"],
}


def _legacy_config_json(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    """PREPARE's clustering_squidpy_config.json: no mode, labels or suffix."""
    return {
        "pair_id": PAIR,
        "output_dir": str(tmp_path / "clustering_squidpy_out"),
        "samples": [
            {
                "sample_id": f"{PAIR}_{platform}",
                "platform": platform,
                "zarr_path": str(tmp_path / f"{platform.lower()}.zarr"),
                "segmentation": "proseg_hybrid",
            }
            for platform in PLATFORMS
        ],
        "use_gpu": True,
        "n_pcs": 10,
        "n_neighbors": 15,
        "figure_dpi": 72,
        **extra,
    }


def _inputs(
    tmp_path: Path, *, control_features: bool = True
) -> tuple[Path, Path, Path, dict[str, Section]]:
    """Write PREPARE's outputs and RESOLVE's annotation_resolve_out.

    The prepared sections carry control features, as real sections do.
    """
    sections = {
        platform: make_section("human", seed=index, control_features=control_features)
        for index, platform in enumerate(PLATFORMS)
    }
    prepared = tmp_path / "clustering_prepare_out"
    labels = tmp_path / "annotation_resolve_out"
    manifest = {}
    for platform, section in sections.items():
        sample_id = f"{PAIR}_{platform}"
        relative = f"{platform.lower()}/{sample_id}_prepared.h5ad"
        (prepared / platform.lower()).mkdir(parents=True, exist_ok=True)
        section.adata.write_h5ad(prepared / relative)
        manifest[sample_id] = relative
        folder = labels / platform.lower()
        write_label_table(
            section.labels,
            folder / label_table_filename(sample_id),
            section.provenance.to_uns_json(),
        )
        (folder / annotation_manifest_filename(sample_id)).write_text(
            section.provenance.to_uns_json()
        )
    (prepared / "manifest.json").write_text(json.dumps({"samples": manifest}))
    (labels / resolve_summary_filename(PAIR)).write_text(
        json.dumps({"pair_id": PAIR, "pair": {"cross_platform": CROSS_PLATFORM}})
    )
    config = tmp_path / "clustering_squidpy_config.json"
    config.write_text(json.dumps(_legacy_config_json(tmp_path)))
    return config, prepared, labels, sections


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["clustering_squidpy_stages", *args])
    stages_main()


@pytest.fixture(scope="module")
def computed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the COMPUTE_CPU command line once on a two-platform pair.

    The hierarchy's figures and stability subsamples are skipped: no test
    here reads them (test_map_first.py covers both), and they dominate the
    run time.
    """
    tmp_path = tmp_path_factory.mktemp("map_first_compute")
    config, prepared, labels, sections = _inputs(tmp_path)
    output = tmp_path / "clustering_compute_out"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        map_first_module,
        "run_map_first_hierarchy",
        functools.partial(
            map_first_module.run_map_first_hierarchy, plots=False, stability=False
        ),
    )
    try:
        _run_cli(
            monkeypatch,
            "compute",
            "--config",
            str(config),
            "--input-dir",
            str(prepared),
            "--output-dir",
            str(output),
            "--mode",
            "map_first",
            "--labels-dir",
            str(labels),
            "--table-key-suffix",
            "mapfirst",
            "--mender-unassigned-state-policy",
            "exclude_from_features",
        )
    finally:
        monkeypatch.undo()
    return {"root": tmp_path, "output": output, "sections": sections}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_compute_cli_writes_the_map_first_table_under_the_legacy_name(
    computed: dict[str, Any], platform: str
) -> None:
    """FINALIZE, MENDER and cortical depth find the H5AD where legacy puts it."""
    sample_id = f"{PAIR}_{platform}"
    h5ad = computed["output"] / platform.lower() / f"{sample_id}_clustered.h5ad"
    clustered = ad.read_h5ad(h5ad)
    section: Section = computed["sections"][platform]

    record = clustered.uns[HIERARCHICAL_UNS_KEY]
    assert record["mode"] == "map_first"
    assert recorded_table_key_suffix(clustered.uns) == "mapfirst"
    assert recorded_mender_policy(clustered.uns) == "exclude_from_features"
    # Table cells only (plan §4.4), with the hierarchy columns.
    in_table = section.labels.loc[section.labels["in_table"], "cell_id"]
    assert set(clustered.obs_names) == set(in_table.astype(str))
    for column in ("hierarchical_cluster", "broad_class", "subcluster_status"):
        assert column in clustered.obs.columns
    # Controls are removed before the table cells are selected, as RESOLVE
    # removes them (plan §4.4): the low-count objects that controls alone
    # lift over min_counts stay out, and no control is a clustered feature.
    assert not set(CONTROL_FEATURES) & set(clustered.var_names)
    assert set(CONTROL_FEATURES) <= set(section.adata.var_names)
    control_filter = clustered.uns["merxen_clustering_squidpy"][
        "control_feature_filter"
    ]
    assert bool(control_filter["enabled"]) is True
    assert sorted(control_filter["removed_control_features"]) == sorted(
        CONTROL_FEATURES
    )
    assert (computed["output"] / platform.lower() / "plots" / "qc").is_dir()
    assert (
        computed["output"]
        / platform.lower()
        / f"{sample_id}_hierarchical"
        / f"{sample_id}_hierarchical_manifest.json"
    ).is_file()


def test_both_sections_record_the_pair_cross_platform_scope(
    computed: dict[str, Any],
) -> None:
    """Cross-platform statements downstream read RESOLVE's pair record (§8.5)."""
    for platform in PLATFORMS:
        sample_id = f"{PAIR}_{platform}"
        clustered = ad.read_h5ad(
            computed["output"] / platform.lower() / f"{sample_id}_clustered.h5ad"
        )
        scope = recorded_cross_platform(clustered.uns)
        assert scope is not None
        assert scope.statistics_level == "broad_only"
        assert scope.allows("broad") and not scope.allows("supercluster")
        assert clustered.uns[HIERARCHICAL_UNS_KEY][
            "cross_platform_statistics_level"
        ] == ("broad_only")
        manifest = json.loads(
            (
                computed["output"]
                / platform.lower()
                / f"{sample_id}_hierarchical"
                / f"{sample_id}_hierarchical_manifest.json"
            ).read_text()
        )
        assert manifest["cross_platform"]["reasons"] == ["intersection_genes:80<100"]
        assert CROSS_PLATFORM_FIELD not in manifest


def test_finalize_writes_the_recorded_suffix_and_leaves_the_legacy_table(
    computed: dict[str, Any], tmp_path: Path
) -> None:
    """FINALIZE's legacy config (no mode, no suffix) writes the *_mapfirst table."""
    import spatialdata as sd

    from .test_map_first_roundtrip import _spatialdata_zarr

    section: Section = computed["sections"]["MERSCOPE"]
    zarr_path = tmp_path / "merscope.zarr"
    _spatialdata_zarr(section, zarr_path)
    config = ClusteringSquidpyConfig(
        pair_id=PAIR,
        output_dir=tmp_path / "clustering_squidpy_out",
        samples=[
            ClusteringSquidpySampleConfig(
                sample_id=f"{PAIR}_MERSCOPE",
                platform="MERSCOPE",
                zarr_path=zarr_path,
                segmentation="proseg_hybrid",
            )
        ],
        write_spatialdata_table=True,
    )
    computed_dir = tmp_path / "computed"
    (computed_dir / "merscope").mkdir(parents=True)
    source = computed["output"] / "merscope" / f"{PAIR}_MERSCOPE_clustered.h5ad"
    (computed_dir / "merscope" / source.name).write_bytes(source.read_bytes())

    results = finalize_clustering_squidpy(config, computed_dir)

    key = results[f"{PAIR}_MERSCOPE"]["spatialdata_table_key"]
    assert key == clustered_table_key("table", "proseg_hybrid", "mapfirst")
    tables = sd.read_zarr(zarr_path).tables
    assert key in tables
    assert clustered_table_key("table", "proseg_hybrid") not in tables


def test_finalize_suffix_rule() -> None:
    """Recorded suffix for map_first tables; explicit config wins; legacy ''."""
    legacy_table = ad.AnnData()
    map_first_table = ad.AnnData(
        uns={
            HIERARCHICAL_UNS_KEY: {"mode": "map_first", "table_key_suffix": "mapfirst"}
        }
    )
    base: dict[str, Any] = {"pair_id": PAIR, "output_dir": Path("out"), "samples": []}

    finalize_config = ClusteringSquidpyConfig(**base)
    assert finalize_table_key_suffix(legacy_table, finalize_config) == ""
    assert finalize_table_key_suffix(map_first_table, finalize_config) == "mapfirst"
    explicit = ClusteringSquidpyConfig(**base, table_key_suffix="")
    assert finalize_table_key_suffix(map_first_table, explicit) == ""
    run_config = ClusteringSquidpyConfig(
        **base, mode="map_first", table_key_suffix="trial"
    )
    assert finalize_table_key_suffix(map_first_table, run_config) == "trial"


def test_mode_overrides_revalidate_and_keep_legacy_untouched(tmp_path: Path) -> None:
    config = ClusteringSquidpyConfig.model_validate(_legacy_config_json(tmp_path))

    assert (
        apply_mode_overrides(config, mode=None, labels_dir=None, table_key_suffix=None)
        is config
    )
    updated = apply_mode_overrides(
        config,
        mode="map_first",
        labels_dir=tmp_path / "labels",
        table_key_suffix="mapfirst",
    )
    assert (updated.mode, updated.table_key_suffix) == ("map_first", "mapfirst")
    assert updated.labels_dir == tmp_path / "labels"
    assert updated.n_pcs == config.n_pcs
    with pytest.raises(ValueError, match="map_first runs only"):
        apply_mode_overrides(
            config, mode="legacy", labels_dir=None, table_key_suffix="mapfirst"
        )


def test_an_empty_suffix_before_the_flip_is_refused(tmp_path: Path) -> None:
    """A map_first compute never writes the unsuffixed (legacy) key (OD-A3)."""
    config_path, prepared, labels, _ = _inputs(tmp_path)
    config = apply_mode_overrides(
        ClusteringSquidpyConfig.model_validate_json(config_path.read_text()),
        mode="map_first",
        labels_dir=labels,
        table_key_suffix="",
    )
    with pytest.raises(ValueError, match="overwrite the legacy one"):
        compute_clustering_squidpy(
            config,
            prepared,
            tmp_path / "out",
            map_first_options=MapFirstComputeOptions(),
        )


def test_map_first_options_apply_to_compute_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_legacy_config_json(tmp_path)))
    with pytest.raises(SystemExit):
        _run_cli(
            monkeypatch,
            "finalize",
            "--config",
            str(config),
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--mode",
            "map_first",
        )
    with pytest.raises(SystemExit):
        _run_cli(
            monkeypatch,
            "compute",
            "--config",
            str(config),
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--mode",
            "map_first",
        )
