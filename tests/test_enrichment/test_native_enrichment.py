"""ENRICH in native mode on the AT-6 reference latest store of an ATERA row.

The reference store (``test_io.atera_native_fixture``) has the layout
NATIVE_LATEST produces: the vendor shapes renamed to ``atera_cell_boundaries``
and ``atera_nucleus``, the vendor table as ``table_original`` and the
``original``/``original_nucleus`` registry entries. Native enrichment must keep
all of that, add only the label-check counts, and never write a Cellpose layer
or Cellpose provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import spatialdata as sd
from click.testing import CliRunner
from test_io.atera_native_fixture import (
    CELL_KEY,
    CELL_SHAPES_KEY,
    NUCLEUS_SHAPES_KEY,
    TABLE_KEY,
    make_native_reference,
)

from merxen.cli.run_enrichment import enrich_command
from merxen.config import EnrichmentConfig
from merxen.enrichment import enrich as enrich_module
from merxen.enrichment.assignment import run_per_shape_assignment_for_dataset
from merxen.enrichment.enrich import (
    NATIVE_ENRICHMENT_ATTR,
    _is_already_enriched,
    _is_enriched_except_nuclei,
    _partial_enrichment_artifact_paths,
    _remove_partial_enrichment_artifacts_from_zarr_path,
    enrich_single_latest,
    native_label_check_path,
)
from merxen.enrichment.label_lookup import NATIVE_COUNT_COLUMNS
from merxen.io.points_aggregates import fingerprint_matches, points_parquet_dir
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR
from merxen.segmentation import native as native_segmentation

PS = 0.21249222
DATASET = "ATERA_SYNTH"


@dataclass(frozen=True)
class Stores:
    source: Path
    latest: Path
    output_dir: Path


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return make_native_reference(tmp_path_factory.mktemp("native_enrich_ref"))


@pytest.fixture
def stores(reference: tuple[Path, Path], tmp_path: Path) -> Iterator[Stores]:
    """A private copy of the reference latest store; the source is shared."""
    source, latest = reference
    copy = tmp_path / "latest_spatialdata.zarr"
    shutil.copytree(latest, copy)
    yield Stores(source=source, latest=copy, output_dir=tmp_path / "enrich_out")


def _config(stores: Stores, **overrides: Any) -> EnrichmentConfig:
    fields: dict[str, Any] = {
        "dataset_name": DATASET,
        "platform": "ATERA",
        "latest_zarr_path": stores.latest,
        "mask_path": None,
        "nuclei_mask_path": None,
        "original_data_path": stores.source,
        "output_dir": stores.output_dir,
        "segmentation_source": "native",
    }
    fields.update(overrides)
    return EnrichmentConfig(**fields)


def _manifest(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _points(store: Path) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(points_parquet_dir(store).glob("*.parquet"))
        ],
        ignore_index=True,
    )


def _labels(store: Path, key: str) -> np.ndarray:
    sdata = sd.read_zarr(store)
    return np.asarray(sd.get_pyramid_levels(sdata.labels[key], n=0))


def _pixel_labels(store: Path, key: str, points: pd.DataFrame) -> np.ndarray:
    labels = _labels(store, key)
    column = np.floor(points["x"].to_numpy(np.float64) / PS).astype(int)
    row = np.floor(points["y"].to_numpy(np.float64) / PS).astype(int)
    inside = (
        (column >= 0)
        & (row >= 0)
        & (column < labels.shape[1])
        & (row < labels.shape[0])
    )
    out = np.zeros(len(points), dtype=np.int64)
    out[inside] = labels[row[inside], column[inside]]
    return out


# --------------------------------------------------------------------------- #
# The enriched store
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_native_enrichment_adds_only_counts_and_vendor_provenance(
    stores: Stores,
) -> None:
    """No MOSAIK_*/Cellpose layer or identity; table_original only gains counts."""
    before = sd.read_zarr(stores.latest)
    table_before = before.tables[TABLE_KEY].copy()
    shapes_before = {key: before.shapes[key].copy() for key in before.shapes}
    source_manifest = _manifest(stores.source)

    written = enrich_single_latest(_config(stores))

    assert written == stores.latest
    after = sd.read_zarr(stores.latest)
    assert set(after.shapes) == {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY}
    assert set(after.tables) == {TABLE_KEY}
    assert set(after.labels) == {"cell_labels", "nucleus_labels"}
    assert not any(str(key).startswith("MOSAIK") for key in after.shapes)

    schema = after.attrs[MERXEN_SCHEMA_ATTR]
    assert "model_identities" not in schema
    assert schema["segmentation_source"] == "native"
    provenance = schema["native_segmentation"]
    assert provenance["source"] == "vendor"
    assert provenance["analysis_sw_version"] == "atera-1.1.1.0"
    assert sum(provenance["cell_segmentation_methods"].values()) == len(
        after.shapes[CELL_SHAPES_KEY]
    )
    registry = schema["segmentations"]
    assert set(registry) == {"original", "original_nucleus"}
    assert registry["original"]["table"] == TABLE_KEY
    assert registry["original"]["shape"] == CELL_SHAPES_KEY
    assert registry["original_nucleus"]["table"] is None
    assert registry["original_nucleus"]["cell_key"] == CELL_KEY
    assert not any("model_identity" in entry for entry in registry.values())

    for key, shape in shapes_before.items():
        pd.testing.assert_frame_equal(
            pd.DataFrame(after.shapes[key].drop(columns="geometry")),
            pd.DataFrame(shape.drop(columns="geometry")),
        )
        assert after.shapes[key].geometry.equals(shape.geometry)

    table = after.tables[TABLE_KEY]
    added = [column for column in table.obs.columns if column.startswith("merxen_")]
    assert sorted(added) == sorted(NATIVE_COUNT_COLUMNS.values())
    pd.testing.assert_frame_equal(table.obs.drop(columns=added), table_before.obs)
    pd.testing.assert_frame_equal(table.var, table_before.var)
    assert (table.X != table_before.X).nnz == 0
    assert set(table.obsm) == set(table_before.obsm)
    assert table.uns["spatialdata_attrs"] == table_before.uns["spatialdata_attrs"]

    record = after.attrs[NATIVE_ENRICHMENT_ATTR]
    assert record["version"] == enrich_module.NATIVE_ENRICHMENT_VERSION
    assert record["label_check"]["passed"] is True
    assert _manifest(stores.source) == source_manifest
    # QC's fast paths (build aggregates, window pruning) need the points index
    # to still describe the enriched store.
    assert fingerprint_matches(stores.latest)


@pytest.mark.integration
def test_native_label_check_matches_the_vendor_exactly(stores: Stores) -> None:
    """0 mismatches; 100% overlaps_nucleus agreement; counts equal brute force."""
    enrich_single_latest(_config(stores))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    points = _points(stores.latest)
    assert summary["passed"] is True
    assert summary["n_mismatch"] == 0
    assert summary["n_rows"] == len(points)
    assert summary["n_vendor_assigned"] == int(
        points["original_assignment"].notna().sum()
    )
    assert summary["nucleus"]["overlaps_agreement"] == 1.0
    assert summary["nucleus"]["n_nucleus_outside_own_cell"] == 0
    assert summary["store_points_scale_agrees"] is True

    cells = _pixel_labels(stores.latest, "cell_labels", points)
    table = sd.read_zarr(stores.latest).tables[TABLE_KEY]
    ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    expected = np.array([(cells == cell).sum() for cell in ids])
    np.testing.assert_array_equal(
        table.obs[NATIVE_COUNT_COLUMNS["cell"]].to_numpy(), expected
    )
    assigned = points["original_assignment"].fillna(0).to_numpy(np.int64)
    np.testing.assert_array_equal(
        expected, np.array([(assigned == cell).sum() for cell in ids])
    )


@pytest.mark.integration
def test_two_nucleus_cell_gets_both_nuclei_transcripts(stores: Stores) -> None:
    """Nuclear counts go to the owning cell, never to the cell with the nucleus id."""
    enrich_single_latest(_config(stores))
    sdata = sd.read_zarr(stores.latest)
    nuclei = sdata.shapes[NUCLEUS_SHAPES_KEY]
    owners = dict(
        zip(
            nuclei["instance_id"].astype(int),
            nuclei[CELL_KEY].astype(int),
            strict=True,
        )
    )
    points = _points(stores.latest)
    per_nucleus = pd.Series(
        _pixel_labels(stores.latest, "nucleus_labels", points)
    ).value_counts()
    two_nucleus_cells = [
        cell for cell in set(owners.values()) if list(owners.values()).count(cell) > 1
    ]
    assert two_nucleus_cells, "the fixture has a two-nucleus cell"

    table = sdata.tables[TABLE_KEY]
    got = dict(
        zip(
            table.obs["instance_id"].astype(int),
            table.obs[NATIVE_COUNT_COLUMNS["nucleus"]].astype(int),
            strict=True,
        )
    )
    for cell, count in got.items():
        mine = [nucleus for nucleus, owner in owners.items() if owner == cell]
        assert count == sum(int(per_nucleus.get(nucleus, 0)) for nucleus in mine)
    for cell in two_nucleus_cells:
        mine = [nucleus for nucleus, owner in owners.items() if owner == cell]
        assert sum(int(per_nucleus.get(n, 0) > 0) for n in mine) == 2
        assert got[cell] > max(int(per_nucleus.get(n, 0)) for n in mine)


@pytest.mark.integration
def test_a_shifted_pixel_size_fails_the_check(stores: Stores, tmp_path: Path) -> None:
    """A wrong transform is caught before any count is written."""
    transform = tmp_path / "shifted_transform.txt"
    scale = 1.0 / (PS * 1.05)
    np.savetxt(transform, np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))

    with pytest.raises(RuntimeError, match="Native label check failed"):
        enrich_single_latest(_config(stores, transform_path=transform))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    assert summary["passed"] is False
    assert summary["n_mismatch"] > 0
    assert summary["store_points_scale_agrees"] is False
    after = sd.read_zarr(stores.latest)
    assert not any(c.startswith("merxen_") for c in after.tables[TABLE_KEY].obs)
    assert NATIVE_ENRICHMENT_ATTR not in after.attrs


@pytest.mark.integration
def test_a_second_run_is_a_skip_and_rewrites_the_summary(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    enrich_single_latest(_config(stores))
    summary_path = native_label_check_path(stores.output_dir, DATASET)
    first = summary_path.read_text()
    summary_path.unlink()

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the label check must not run again")

    monkeypatch.setattr(enrich_module, "native_label_check", _fail)
    enrich_single_latest(_config(stores))
    assert summary_path.read_text() == first

    with pytest.raises(AssertionError, match="must not run again"):
        enrich_single_latest(_config(stores), force_rerun=True)


def _edge_transcript_y(labels: np.ndarray) -> tuple[float, int, int, int]:
    """Find a float32 y on a row edge where the float32 and float64 rules differ.

    Returns ``(y_um, column, row_float32, row_float64)`` for a column whose
    labels differ between the two rows, so only the vendor's float32 rule
    puts the transcript on the vendor's cell.
    """
    scale32 = np.float32(1.0 / PS)
    for edge in range(1, labels.shape[0]):
        centre = np.float32(edge * PS)
        candidates = [centre]
        up = down = centre
        for _ in range(64):
            up = np.nextafter(up, np.float32(np.inf))
            down = np.nextafter(down, np.float32(-np.inf))
            candidates += [up, down]
        for y32 in candidates:
            row32 = int(np.floor(np.float32(y32) * scale32))
            row64 = int(np.floor(float(y32) / PS))
            lowest, highest = min(row32, row64), max(row32, row64)
            if row32 == row64 or lowest < 0 or highest >= labels.shape[0]:
                continue
            differs = np.flatnonzero(labels[row32] != labels[row64])
            if differs.size:
                return float(y32), int(differs[0]), row32, row64
    raise AssertionError("no float32/float64 edge disagreement in the fixture")


def _append_edge_transcript(store: Path) -> tuple[int, int]:
    """Append one transcript the float64 rule puts on the wrong pixel.

    Its vendor columns follow the vendor's float32 rule. Returns the vendor
    cell (0 = unassigned) and the cell the float64 rule would give.
    """
    cells = _labels(store, "cell_labels")
    nuclei = _labels(store, "nucleus_labels")
    y_um, column, row32, row64 = _edge_transcript_y(cells)
    x_um = float(np.float32((column + 0.5) * PS))
    vendor_cell = int(cells[row32, column])
    parts = sorted(points_parquet_dir(store).glob("*.parquet"))
    last = parts[-1]
    table = pq.read_table(last)
    everything = _points(store)
    row = table.slice(table.num_rows - 1, 1).to_pylist()[0]
    row.update(
        x=x_um,
        y=y_um,
        overlaps_nucleus=bool(nuclei[row32, column] > 0),
        original_assignment=vendor_cell or None,
        transcript_id=int(everything["transcript_id"].max()) + 1,
    )
    if "__null_dask_index__" in row:
        row["__null_dask_index__"] = int(len(everything))
    extra = pa.Table.from_pylist([row], schema=table.schema)
    pq.write_table(pa.concat_tables([table, extra]), last)
    return vendor_cell, int(cells[row64, column])


@pytest.mark.integration
def test_the_vendor_float32_rule_is_the_enrich_default(stores: Stores) -> None:
    """A transcript on a pixel edge passes only with the vendor's float32 rule.

    Under float64 division it lands on the neighbouring pixel, which is another
    cell (or background): one mismatch, above the 0 the tolerance allows on a
    store this small.
    """
    vendor_cell, float64_cell = _append_edge_transcript(stores.latest)
    assert vendor_cell != float64_cell

    enrich_single_latest(_config(stores))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    assert summary["pixel_precision"] == "float32"
    assert summary["mismatch_allowed"] == 0
    assert summary["n_mismatch"] == 0
    assert summary["passed"] is True
    points = _points(stores.latest)
    assert summary["n_rows"] == len(points)
    table = sd.read_zarr(stores.latest).tables[TABLE_KEY]
    ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    assigned = points["original_assignment"].fillna(0).to_numpy(np.int64)
    np.testing.assert_array_equal(
        table.obs[NATIVE_COUNT_COLUMNS["cell"]].to_numpy(),
        np.array([(assigned == cell).sum() for cell in ids]),
    )


@pytest.mark.integration
def test_a_broken_nucleus_to_cell_join_fails_the_check(stores: Stores) -> None:
    """Nuclei pointed at the wrong cells fail the check before any write (R6)."""
    sdata = sd.read_zarr(stores.latest)
    nuclei = sdata.shapes[NUCLEUS_SHAPES_KEY].copy()
    owners = nuclei[CELL_KEY].to_numpy()
    nuclei[CELL_KEY] = np.roll(owners, 1)
    assert (nuclei[CELL_KEY].to_numpy() != owners).any()
    sdata.delete_element_from_disk(NUCLEUS_SHAPES_KEY)
    sdata.shapes[NUCLEUS_SHAPES_KEY] = nuclei
    sdata.write_element(NUCLEUS_SHAPES_KEY)
    sdata.write_consolidated_metadata()

    with pytest.raises(RuntimeError, match=r"failed \(.*nuclei_inside_own_cell"):
        enrich_single_latest(_config(stores))

    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    assert summary["checks"]["cell_assignment"] is True
    assert summary["checks"]["nuclei_inside_own_cell"] is False
    assert summary["nucleus"]["n_nucleus_outside_own_cell"] > 0
    assert summary["passed"] is False
    after = sd.read_zarr(stores.latest)
    assert not any(c.startswith("merxen_") for c in after.tables[TABLE_KEY].obs)
    assert NATIVE_ENRICHMENT_ATTR not in after.attrs


@pytest.mark.integration
def test_a_points_scale_that_disagrees_fails_the_check(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lookup pixel size must be the one the store's points record."""
    monkeypatch.setattr(enrich_module, "_points_scale_um_per_px", lambda _: PS * 2)
    with pytest.raises(RuntimeError, match=r"failed \(points_scale\)"):
        enrich_single_latest(_config(stores))
    summary = json.loads(
        native_label_check_path(stores.output_dir, DATASET).read_text()
    )
    assert summary["n_mismatch"] == 0
    assert summary["failures"] == ["points_scale"]
    assert NATIVE_ENRICHMENT_ATTR not in sd.read_zarr(stores.latest).attrs


@pytest.mark.integration
def test_the_label_check_can_be_switched_off(stores: Stores) -> None:
    enrich_single_latest(_config(stores, native_label_check=False))
    after = sd.read_zarr(stores.latest)
    assert after.attrs[NATIVE_ENRICHMENT_ATTR]["label_check"] is None
    assert not any(c.startswith("merxen_") for c in after.tables[TABLE_KEY].obs)
    assert not native_label_check_path(stores.output_dir, DATASET).exists()
    # Switching it on later runs it.
    enrich_single_latest(_config(stores))
    assert native_label_check_path(stores.output_dir, DATASET).exists()


# --------------------------------------------------------------------------- #
# Partial stores
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_missing_renamed_elements_are_recreated_from_the_source(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENRICH calls ensure_native_elements, re-reads, and completes."""
    sdata = sd.read_zarr(stores.latest)
    nuclei = sdata.shapes[NUCLEUS_SHAPES_KEY].copy()
    sdata.delete_element_from_disk(NUCLEUS_SHAPES_KEY)
    calls: list[tuple[Path, Path]] = []

    def _ensure(latest_path: Path, source_path: Path) -> list[str]:
        calls.append((Path(latest_path), Path(source_path)))
        store = sd.read_zarr(latest_path)
        store.shapes[NUCLEUS_SHAPES_KEY] = nuclei
        store.write_element(NUCLEUS_SHAPES_KEY)
        store.write_consolidated_metadata()
        return [f"shapes/{NUCLEUS_SHAPES_KEY}"]

    monkeypatch.setattr(native_segmentation, "ensure_native_elements", _ensure)
    enrich_single_latest(_config(stores))

    assert calls == [(stores.latest, stores.source)]
    after = sd.read_zarr(stores.latest)
    assert NUCLEUS_SHAPES_KEY in after.shapes
    assert after.attrs[NATIVE_ENRICHMENT_ATTR]["label_check"]["passed"] is True


@pytest.mark.integration
def test_missing_payload_elements_are_refused(stores: Stores) -> None:
    """Images, labels and points are shared payloads only NATIVE_LATEST restores."""
    shutil.rmtree(stores.latest / "labels" / "nucleus_labels")
    enrich_module._prune_zarr_consolidated_metadata(
        stores.latest / "zarr.json", {"labels/nucleus_labels"}
    )
    enrich_module._prune_zarr_consolidated_metadata(
        stores.latest / "labels" / "zarr.json", {"nucleus_labels"}
    )
    with pytest.raises(RuntimeError, match="payload elements.*nucleus_labels"):
        enrich_single_latest(_config(stores))


@pytest.mark.integration
def test_partial_cleanup_of_a_native_store_removes_only_derived_tables(
    stores: Stores,
) -> None:
    """Payloads, renamed shapes and table_original survive; the source too."""
    derived = stores.latest / "tables" / "table_atera_cell_boundaries_extra"
    derived.mkdir(parents=True)
    source_manifest = _manifest(stores.source)

    assert _partial_enrichment_artifact_paths(stores.latest, "ATERA") == {
        "tables/table_atera_cell_boundaries_extra"
    }
    _remove_partial_enrichment_artifacts_from_zarr_path(stores.latest, "ATERA")

    assert not derived.exists()
    for rel in (
        f"shapes/{CELL_SHAPES_KEY}",
        f"shapes/{NUCLEUS_SHAPES_KEY}",
        f"tables/{TABLE_KEY}",
        "images/morphology_focus",
        "labels/cell_labels",
        "labels/nucleus_labels",
        "points/transcripts",
    ):
        assert (stores.latest / rel).is_dir(), rel
    assert _manifest(stores.source) == source_manifest
    sdata = sd.read_zarr(stores.latest)
    assert set(sdata.shapes) == {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY}


@pytest.mark.integration
def test_an_unreadable_native_raster_is_refused_without_cleanup(
    stores: Stores, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native store's rasters are payloads: no cleanup, a NATIVE_LATEST hint."""
    derived = stores.latest / "tables" / "table_atera_cell_boundaries_image_x"
    derived.mkdir(parents=True)

    def _unreadable(path: Path) -> None:
        raise KeyError("ome")

    monkeypatch.setattr(enrich_module.sd, "read_zarr", _unreadable)
    with pytest.raises(RuntimeError, match="unreadable raster.*NATIVE_LATEST"):
        enrich_single_latest(_config(stores))
    assert derived.is_dir()


def _is_payload(relative: Path) -> bool:
    """Whether a store file is a payload NATIVE_LATEST hardlinks to the source."""
    text = relative.as_posix()
    if text.startswith("points/"):
        return relative.suffix == ".parquet"
    if text.startswith(("images/", "labels/")):
        return relative.name not in {"zarr.json", ".zattrs", ".zgroup", ".zarray"}
    return False


def _hardlinked_latest(payload_source: Path, latest: Path) -> int:
    """Lay out ``latest`` like NATIVE_LATEST: payloads linked read-only."""
    n_linked = 0
    for path in sorted(payload_source.rglob("*")):
        relative = path.relative_to(payload_source)
        target = latest / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _is_payload(relative):
            os.chmod(path, 0o444)
            os.link(path, target)
            n_linked += 1
        else:
            shutil.copy2(path, target)
    return n_linked


def _payload_manifest(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.integration
def test_enrich_assignment_and_miq_keep_hardlinked_payloads_intact(
    reference: tuple[Path, Path], tmp_path: Path
) -> None:
    """The NATIVE_LATEST contract: shared payloads are never written.

    The latest store's points, image and label chunks are read-only hardlinks
    to a payload source; ENRICH, the per-shape assignment and label-backed MIQ
    must leave every one linked and the source unchanged.
    """
    from merxen.config import MaskImageQuantificationConfig
    from merxen.mask_image_quantification import run_mask_image_quantification

    source, reference_latest = reference
    payload_source = tmp_path / "payload_source.zarr"
    shutil.copytree(reference_latest, payload_source)
    latest = tmp_path / "latest" / "latest_spatialdata.zarr"
    try:
        assert _hardlinked_latest(payload_source, latest) > 0
        before = _payload_manifest(payload_source)
        stores = Stores(source=source, latest=latest, output_dir=tmp_path / "out")

        enrich_single_latest(_config(stores))
        run_per_shape_assignment_for_dataset(
            DATASET, latest, platform="ATERA", segmentation_source="native"
        )
        run_mask_image_quantification(
            MaskImageQuantificationConfig(
                dataset_name=DATASET,
                platform="ATERA",
                latest_zarr_path=latest,
                output_dir=tmp_path / "miq",
                shape_key=CELL_SHAPES_KEY,
                table_key=f"table_{CELL_SHAPES_KEY}_image_quantification",
                tile_size=32,
                mask_path=None,
                mask_label_key="cell_labels",
            )
        )

        assert _payload_manifest(payload_source) == before
        unlinked = [
            str(path.relative_to(latest))
            for path in latest.rglob("*")
            if path.is_file()
            and _is_payload(path.relative_to(latest))
            and path.stat().st_nlink < 2
        ]
        assert unlinked == []
        after = sd.read_zarr(latest)
        assert after.attrs[NATIVE_ENRICHMENT_ATTR]["label_check"]["passed"] is True
        assert f"table_{CELL_SHAPES_KEY}_image_quantification" in after.tables
    finally:
        for path in payload_source.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o644)


def _namespace(attrs: dict[str, Any] | None = None, **missing: str) -> Any:
    elements = {
        "shapes": {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY},
        "tables": {TABLE_KEY},
        "images": {"morphology_focus"},
        "labels": {"cell_labels", "nucleus_labels"},
        "points": {"transcripts"},
    }
    for kind, name in missing.items():
        elements[kind].discard(name)
    return SimpleNamespace(
        attrs=attrs or {},
        **{kind: dict.fromkeys(names, object()) for kind, names in elements.items()},
    )


@pytest.mark.parametrize(
    "missing",
    [
        {},
        {"shapes": NUCLEUS_SHAPES_KEY},
        {"tables": TABLE_KEY},
        {"images": "morphology_focus"},
        {"labels": "cell_labels"},
        {"points": "transcripts"},
    ],
)
def test_native_completeness_needs_vendor_elements_not_cellpose_layers(
    missing: dict[str, str],
) -> None:
    store = _namespace(**missing)
    assert _is_already_enriched(store, "ATERA") is (not missing)
    assert _is_enriched_except_nuclei(store, "ATERA") is False


# --------------------------------------------------------------------------- #
# Per-shape assignment and the CLI
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_native_assignment_registers_table_original_without_a_clone(
    stores: Stores,
) -> None:
    enrich_single_latest(_config(stores))
    summaries = run_per_shape_assignment_for_dataset(
        DATASET,
        stores.latest,
        platform="ATERA",
        segmentation_source="native",
    )

    assert [(s["shape_key"], s["table_key"], s["mode"]) for s in summaries] == [
        (CELL_SHAPES_KEY, TABLE_KEY, "native_table_original")
    ]
    after = sd.read_zarr(stores.latest)
    assert set(after.tables) == {TABLE_KEY}
    registry = after.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert registry["original"]["table"] == TABLE_KEY
    assert registry["original"]["assignment_column"] == "original_assignment"
    assert registry["original_nucleus"]["cell_key"] == CELL_KEY
    assert registry["original_nucleus"]["table"] is None


@pytest.mark.integration
def test_native_assignment_needs_a_native_platform(stores: Stores) -> None:
    with pytest.raises(ValueError, match="native segmentation"):
        run_per_shape_assignment_for_dataset(
            DATASET, stores.latest, platform="XENIUM", segmentation_source="native"
        )


@pytest.mark.integration
def test_enrich_cli_native_row_end_to_end(stores: Stores, tmp_path: Path) -> None:
    """The ENRICH task command on a native row: counts, registry, summaries."""
    config = tmp_path / "enrich_config.json"
    config.write_text(_config(stores).model_dump_json(indent=2))

    result = CliRunner().invoke(enrich_command, ["--config", str(config)])
    assert result.exit_code == 0, result.output

    summary_csv = stores.output_dir / "atera_synth_per_shape_assignment_summary.csv"
    rows = pd.read_csv(summary_csv)
    assert rows[["shape_key", "table_key", "mode"]].values.tolist() == [
        [CELL_SHAPES_KEY, TABLE_KEY, "native_table_original"]
    ]
    assert native_label_check_path(stores.output_dir, DATASET).exists()
    after = sd.read_zarr(stores.latest)
    assert set(after.tables) == {TABLE_KEY}
    assert {"MOSAIK_proseg", "MOSAIK_cellpose", "cellpose_nuclei"}.isdisjoint(
        after.shapes
    )

    # A second task run on the enriched store is a no-op apart from summaries.
    result = CliRunner().invoke(enrich_command, ["--config", str(config)])
    assert result.exit_code == 0, result.output
    assert set(sd.read_zarr(stores.latest).tables) == {TABLE_KEY}


def test_native_partial_paths_hold_only_derived_tables(tmp_path: Path) -> None:
    """ATERA (native only): no shapes, images, labels or points are rebuildable."""
    store = tmp_path / "store.zarr"
    assert _partial_enrichment_artifact_paths(store, "ATERA") == set()
    for name in ("table_original", "table_atera_cell_boundaries_x", "not_a_table"):
        (store / "tables" / name).mkdir(parents=True)
    assert _partial_enrichment_artifact_paths(store, "ATERA") == {
        "tables/table_atera_cell_boundaries_x"
    }


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_a_native_record_never_makes_xenium_or_merscope_native(platform: str) -> None:
    attrs = {MERXEN_SCHEMA_ATTR: {"segmentation_source": "native"}}
    assert enrich_module._is_native_store(platform, attrs) is False
    assert enrich_module._is_native_store(platform, {}) is False
    assert enrich_module._is_native_store("ATERA", {}) is True
