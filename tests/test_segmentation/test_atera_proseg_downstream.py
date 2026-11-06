"""ENRICH, MIQ and viewer caches of a ``native_proseg`` ATERA store (plan AT-9).

The store is the synthetic multi-tile latest store with the ProSeg layer
stitched from the identity fake ProSeg, so the ProSeg labels equal the vendor
labels and every per-cell value of the two layers must agree.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd

from merxen.config import EnrichmentConfig, MaskImageQuantificationConfig, MiqLayer
from merxen.enrichment.assignment import run_per_shape_assignment_for_dataset
from merxen.enrichment.enrich import (
    _derived_table_paths,
    _is_native_store,
    enrich_single_latest,
)
from merxen.mask_image_quantification import run_mask_image_quantification
from merxen.segmentation.atera_proseg.plan import write_plan
from merxen.segmentation.atera_proseg.stitch import stitch
from merxen.segmentation.atera_proseg.tile import run_tile
from merxen.viewer_cache.build import (
    ViewerCacheParams,
    build_viewer_caches,
    vendor_label_keys,
)
from test_segmentation.atera_proseg_fixture import (
    DATASET,
    build_multitile_latest,
    fake_proseg_binary,
    plan_config,
    stitch_config,
    tile_configs,
)

_SRC = Path(__file__).resolve().parents[2] / "src" / "merxen"


@pytest.fixture(scope="module")
def stitched(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """``(source, native latest without ProSeg, stitched latest)``."""
    root = tmp_path_factory.mktemp("downstream")
    source, latest = build_multitile_latest(root)
    plain = root / "plain_latest.zarr"
    shutil.copytree(latest, plain)
    plan_path = write_plan(plan_config(source, root / "plan_out"))
    plan = json.loads(plan_path.read_text())
    binary = fake_proseg_binary(root / "bin")
    dirs = []
    for config in tile_configs(plan, source, root / "tiles"):
        run_tile(config, binary)
        dirs.append(config.output_dir)
    stitch(stitch_config(latest, plan_path, dirs, root / "stitch_out"))
    return source, plain, latest


@pytest.fixture
def store(stitched: tuple[Path, Path, Path], tmp_path: Path) -> Path:
    copy = tmp_path / "latest.zarr"
    shutil.copytree(stitched[2], copy)
    return copy


def _enrich_config(store: Path, source: Path, out: Path) -> EnrichmentConfig:
    return EnrichmentConfig(
        dataset_name=DATASET,
        platform="ATERA",
        latest_zarr_path=store,
        mask_path=None,
        nuclei_mask_path=None,
        original_data_path=source,
        output_dir=out,
        segmentation_source="native_proseg",
    )


def test_native_store_detection_with_two_sources() -> None:
    assert _is_native_store("ATERA", {})
    assert _is_native_store(
        "ATERA", {"merxen_schema": {"segmentation_source": "native"}}
    )
    assert not _is_native_store("XENIUM", {})
    assert not _is_native_store("MERSCOPE", {})


def test_partial_cleanup_keeps_the_stitched_table(store: Path) -> None:
    derived = _derived_table_paths(store)
    assert "tables/table_MOSAIK_proseg" not in derived


def test_enrich_checks_the_proseg_layer(
    stitched: tuple[Path, Path, Path], store: Path, tmp_path: Path
) -> None:
    source = stitched[0]
    enrich_single_latest(_enrich_config(store, source, tmp_path / "enrich_out"))
    check = json.loads(
        (tmp_path / "enrich_out" / "atera_proseg_layer_check.json").read_text()
    )
    assert check["passed"] and check["failures"] == []
    sdata = sd.read_zarr(store)
    assert sdata.attrs["merxen_atera_proseg"]["enrich_check"]["passed"]
    assert "merxen_native_enrichment" in sdata.attrs
    assert "table_MOSAIK_proseg" in sdata.tables
    # Per-shape assignment leaves the stitched table and registry alone.
    before = json.dumps(sdata.attrs["merxen_schema"]["segmentations"]["proseg"])
    run_per_shape_assignment_for_dataset(
        dataset_name=DATASET,
        latest_path=store,
        platform="ATERA",
        segmentation_source="native_proseg",
    )
    after = sd.read_zarr(store).attrs["merxen_schema"]["segmentations"]["proseg"]
    assert json.dumps(after) == before
    # A second ENRICH skips the check (same stitch fingerprint).
    enrich_single_latest(_enrich_config(store, source, tmp_path / "enrich_again"))
    assert not (tmp_path / "enrich_again" / "atera_proseg_layer_check.json").exists()


def test_enrich_refuses_a_store_without_the_layer(
    stitched: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source, plain, _ = stitched
    copy = tmp_path / "plain.zarr"
    shutil.copytree(plain, copy)
    with pytest.raises(RuntimeError, match="without a stitched ProSeg layer"):
        enrich_single_latest(_enrich_config(copy, source, tmp_path / "out"))


def test_enrich_fails_when_counts_disagree(
    stitched: tuple[Path, Path, Path], store: Path, tmp_path: Path
) -> None:
    import anndata as ad
    import zarr

    group = zarr.open_group(str(store / "tables" / "table_MOSAIK_proseg"), mode="a")
    matrix = ad.io.read_elem(group["X"])
    matrix.data[0] += 1
    ad.io.write_elem(group, "X", matrix)
    with pytest.raises(RuntimeError, match="points foreground counts"):
        enrich_single_latest(_enrich_config(store, stitched[0], tmp_path / "out"))


def test_viewer_cache_backs_the_proseg_mask_by_its_labels(store: Path) -> None:
    sdata = sd.read_zarr(store)
    keys = vendor_label_keys(sdata, "ATERA")
    assert keys["MOSAIK_proseg"] == "proseg_cell_labels"
    assert keys["atera_cell_boundaries"] == "cell_labels"
    summary = build_viewer_caches(
        store,
        "ATERA",
        store,
        params=ViewerCacheParams(
            min_size=16, base_image_key="morphology_focus", build_image_pyramid=False
        ),
    )
    result = summary["masks"]["MOSAIK_proseg"]
    assert result["source_label_key"] == "proseg_cell_labels"
    assert result["outline"] == "built"


def test_viewer_cache_discovery_is_unchanged_without_label_entries() -> None:
    from types import SimpleNamespace

    xenium = SimpleNamespace(
        labels={"proseg_cell_labels": None},
        shapes={"MOSAIK_proseg": None},
        attrs={
            "merxen_schema": {
                "segmentations": {
                    "proseg": {"shape": "MOSAIK_proseg", "label": "proseg_cell_labels"}
                }
            }
        },
    )
    assert vendor_label_keys(xenium, "XENIUM") == {}


def test_miq_quantifies_the_proseg_layer(store: Path, tmp_path: Path) -> None:
    config = MaskImageQuantificationConfig(
        dataset_name=DATASET,
        platform="ATERA",
        latest_zarr_path=store,
        mask_path=None,
        mask_label_key="cell_labels",
        output_dir=tmp_path / "miq",
        shape_key="atera_cell_boundaries",
        table_key="table_atera_cell_boundaries_image_quantification",
        tile_size=32,
        extra_layers=[
            MiqLayer(
                mask_label_key="proseg_cell_labels",
                shape_key="MOSAIK_proseg",
                table_key="table_MOSAIK_proseg_image_quantification",
            )
        ],
    )
    outputs = run_mask_image_quantification(config)
    assert outputs["MOSAIK_proseg_summary"].is_file()
    sdata = sd.read_zarr(store)
    vendor = sdata.tables["table_atera_cell_boundaries_image_quantification"]
    proseg = sdata.tables["table_MOSAIK_proseg_image_quantification"]
    assert proseg.uns["mask_image_quantification"]["mask_label_key"] == (
        "proseg_cell_labels"
    )
    # The identity fake ProSeg draws the vendor labels: equal per-cell values.
    vendor_frame = vendor.to_df().set_index(vendor.obs["instance_id"].to_numpy())
    proseg_frame = proseg.to_df().set_index(proseg.obs["instance_id"].to_numpy())
    common = proseg_frame.index
    assert np.allclose(
        vendor_frame.loc[common].to_numpy(), proseg_frame.to_numpy(), equal_nan=True
    )
    # A second run skips both layers.
    again = run_mask_image_quantification(config)
    assert again["MOSAIK_proseg_summary"] == outputs["MOSAIK_proseg_summary"]


# --------------------------------------------------------------------------- #
# Generic consumers of an "assignment" column (design §5)
# --------------------------------------------------------------------------- #

#: Every call that picks the first of ``assignment`` / ``cell`` / ``cell_id``.
#: On an ATERA ``native_proseg`` store the points carry ``assignment`` (and
#: ``background``) next to ``original_assignment``; each site must not change
#: an original-layer result:
#: - qc/metrics.py: only without a table key (QC of a layer always has one);
#: - qc/gene_comparison.py: only with ``assigned_only``, which no caller sets;
#: - visualization/sanity_plots.py ``_crop_points``: only without an explicit
#:   column, and ATERA overlays use the rasterised path (registry column).
GENERIC_ASSIGNMENT_SITES = {
    "qc/metrics.py": 1,
    "qc/gene_comparison.py": 1,
    "visualization/sanity_plots.py": 1,
}


def test_generic_assignment_column_sites_are_reviewed() -> None:
    pattern = re.compile(r'\[\s*"assignment",\s*"cell",\s*"cell_id"\s*\]')
    found: dict[str, int] = {}
    for path in sorted(_SRC.rglob("*.py")):
        count = len(pattern.findall(path.read_text()))
        if count:
            found[str(path.relative_to(_SRC))] = count
    assert found == GENERIC_ASSIGNMENT_SITES
