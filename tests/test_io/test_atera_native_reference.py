"""The AT-6 reference layout of a native ATERA latest store and its aggregates.

The reference store (``atera_native_fixture.make_native_reference``) is the
layout NATIVE_LATEST (WP-A) must produce and the downstream packages test
against. ``merxen.io.points_aggregates`` reads the build aggregates and the
points index that both stores carry.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import spatialdata as sd

from merxen.io import points_aggregates as pa
from merxen.io.builders import atera as atera_builder
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR
from merxen.platforms import find_um_per_px

from .atera_native_fixture import (
    CELL_KEY,
    CELL_SHAPES_KEY,
    NUCLEUS_SHAPES_KEY,
    SOURCE_NAMES,
    TABLE_KEY,
    make_native_reference,
)


@dataclass(frozen=True)
class Reference:
    source: Path
    latest: Path


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Reference]:
    source, latest = make_native_reference(tmp_path_factory.mktemp("native_ref"))
    yield Reference(source, latest)


def _all_points(store: Path) -> pd.DataFrame:
    directory = pa.points_parquet_dir(store)
    return pd.concat(
        [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# Reference layout
# --------------------------------------------------------------------------- #


def test_reference_elements_are_renamed(reference: Reference) -> None:
    sdata = sd.read_zarr(reference.latest)
    assert set(sdata.shapes) == {CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY}
    assert set(sdata.tables) == {TABLE_KEY}
    assert set(sdata.labels) == {"cell_labels", "nucleus_labels"}
    assert "morphology_focus" in sdata.images
    assert set(sdata.points) == {"transcripts"}
    names = {str(name) for name in sdata.shapes} | set(sdata.tables)
    assert not names & SOURCE_NAMES
    for kind in ("shapes", "tables"):
        on_disk = {path.name for path in (reference.latest / kind).iterdir()}
        assert not on_disk & SOURCE_NAMES, kind


def test_reference_table_annotates_the_renamed_cells(reference: Reference) -> None:
    sdata = sd.read_zarr(reference.latest)
    source = sd.read_zarr(reference.source)
    table = sdata.tables[TABLE_KEY]
    assert table.uns["spatialdata_attrs"]["region"] == CELL_SHAPES_KEY
    assert set(table.obs["region"].astype(str)) == {CELL_SHAPES_KEY}
    cells = sdata.shapes[CELL_SHAPES_KEY]
    assert set(table.obs["instance_id"].astype(np.uint64)) == set(
        cells["instance_id"].astype(np.uint64)
    )
    np.testing.assert_array_equal(table.obs_names, source.tables["table"].obs_names)
    assert (table.X != source.tables["table"].X).nnz == 0


def test_reference_nuclei_join_cells_through_cell_instance_id(
    reference: Reference,
) -> None:
    """The two-nucleus cell keeps both nuclei; nucleus ids are not cell ids."""
    nuclei = sd.read_zarr(reference.latest).shapes[NUCLEUS_SHAPES_KEY]
    per_cell = nuclei.groupby(CELL_KEY).size()
    assert per_cell.max() == 2
    assert int(per_cell.idxmax()) == 2  # cell row 1 of the synthetic bundle
    np.testing.assert_array_equal(
        nuclei["instance_id"].to_numpy(np.uint64),
        np.arange(1, len(nuclei) + 1, dtype=np.uint64),
    )


def test_reference_registry(reference: Reference) -> None:
    attrs = pa.read_root_attrs(reference.latest)
    schema = attrs[MERXEN_SCHEMA_ATTR]
    assert schema["segmentation_source"] == "native"
    registry = schema["segmentations"]
    assert set(registry) == {"original", "original_nucleus"}
    original = registry["original"]
    assert original["shape"] == CELL_SHAPES_KEY
    assert original["table"] == TABLE_KEY
    assert original["assignment_column"] == "original_assignment"
    nucleus = registry["original_nucleus"]
    assert nucleus["shape"] == NUCLEUS_SHAPES_KEY
    assert nucleus["table"] is None and nucleus["assignment_column"] is None
    assert nucleus["cell_key"] == CELL_KEY
    assert nucleus["id_namespace"] != original["id_namespace"]


def test_reference_keeps_the_vendor_metadata(reference: Reference) -> None:
    source_attrs = pa.read_root_attrs(reference.source)
    latest_attrs = pa.read_root_attrs(reference.latest)
    assert latest_attrs[pa.ATERA_ATTRS_KEY] == source_attrs[pa.ATERA_ATTRS_KEY]
    assert (reference.latest / pa.POINTS_INDEX_NAME).read_bytes() == (
        reference.source / pa.POINTS_INDEX_NAME
    ).read_bytes()
    assert find_um_per_px(reference.latest, "ATERA") == find_um_per_px(
        reference.source, "ATERA"
    )


# --------------------------------------------------------------------------- #
# points_aggregates
# --------------------------------------------------------------------------- #


def test_constants_match_the_builder() -> None:
    assert pa.ATERA_ATTRS_KEY == atera_builder.ATERA_ATTRS_KEY
    assert pa.POINTS_INDEX_NAME == atera_builder.POINTS_INDEX_NAME


@pytest.mark.parametrize("which", ["source", "latest"])
def test_aggregates_equal_a_full_count(reference: Reference, which: str) -> None:
    store = getattr(reference, which)
    aggregates = pa.read_points_aggregates(store)
    assert aggregates is not None
    assert aggregates.fingerprint_matches()
    assert pa.fingerprint_matches(store)
    points = _all_points(store)
    assert aggregates.n_points_written == len(points)
    counts = points["feature_name"].astype(str).value_counts()
    totals = aggregates.gene_counts()
    assert {name: n for name, n in totals.items() if n} == counts.to_dict()
    assert aggregates.density.sum() == len(points)
    # Rows outside the grid (negative coordinates) count in the edge bins.
    x0, y0, x1, y1 = aggregates.density_extent_um()
    inside = points["x"].between(x0, x1) & points["y"].between(y0, y1)
    assert inside.mean() > 0.9
    assert aggregates.bounds_um["x_min"] <= points["x"].min()


def test_files_intersecting_prunes_without_losing_rows(reference: Reference) -> None:
    aggregates = pa.read_points_aggregates(reference.latest)
    assert aggregates is not None
    points = _all_points(reference.latest)
    window = (0.0, 0.0, 8.0, 8.0)
    selected = aggregates.files_intersecting(window)
    assert 0 < len(selected) < len(aggregates.files)
    in_window = points[
        points["x"].between(window[0], window[2])
        & points["y"].between(window[1], window[3])
    ]
    pruned = pd.concat([pd.read_parquet(path) for path in selected], ignore_index=True)
    pruned = pruned[
        pruned["x"].between(window[0], window[2])
        & pruned["y"].between(window[1], window[3])
    ]
    assert len(pruned) == len(in_window) > 0
    assert pa.files_intersecting(reference.latest, window) == selected
    with pytest.raises(ValueError, match="empty window"):
        aggregates.files_intersecting((1.0, 0.0, 0.0, 1.0))


def test_a_stale_index_is_not_trusted(reference: Reference, tmp_path: Path) -> None:
    store = tmp_path / "stale.zarr"
    shutil.copytree(reference.latest, store)
    directory = pa.points_parquet_dir(store)
    first = sorted(directory.glob("*.parquet"))[0]
    table = pq.read_table(first)
    pq.write_table(table.slice(0, max(table.num_rows - 1, 0)), first)
    assert not pa.fingerprint_matches(store)
    assert pa.files_intersecting(store, (0.0, 0.0, 1.0, 1.0)) is None

    extra = tmp_path / "extra.zarr"
    shutil.copytree(reference.latest, extra)
    shutil.copy(first, pa.points_parquet_dir(extra) / "part.999.parquet")
    assert not pa.fingerprint_matches(extra)


def test_stores_without_aggregates_return_none(tmp_path: Path) -> None:
    """XENIUM and MERSCOPE stores keep their current path."""
    store = tmp_path / "xenium.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    assert pa.read_points_aggregates(store) is None
    assert not pa.fingerprint_matches(store)
    assert pa.files_intersecting(store, (0.0, 0.0, 1.0, 1.0)) is None
    v2 = tmp_path / "v2.zarr"
    v2.mkdir()
    (v2 / ".zattrs").write_text(json.dumps({"spatialdata_attrs": {}}))
    assert pa.read_root_attrs(v2) == {"spatialdata_attrs": {}}
    assert pa.read_points_aggregates(v2) is None


# --------------------------------------------------------------------------- #
# The ATERA arm of the enrichment pixel transform
# --------------------------------------------------------------------------- #


def test_enrichment_transform_reads_the_atera_sidecar(reference: Reference) -> None:
    from merxen.enrichment.enrich import resolve_px_to_um_transform

    mpp = find_um_per_px(reference.latest, "ATERA")
    x_row, y_row = resolve_px_to_um_transform(
        platform="ATERA", original_data_path=reference.latest, transform_path=None
    )
    assert x_row[0] == pytest.approx(mpp) and y_row[1] == pytest.approx(mpp)
    assert x_row[1] == 0.0 and y_row[0] == 0.0


def test_enrichment_transform_prefers_the_transform_path(tmp_path: Path) -> None:
    from merxen.enrichment.enrich import resolve_px_to_um_transform

    spec = tmp_path / "experiment.spatial"
    spec.write_text(json.dumps({"pixel_size": 0.5}))
    x_row, _ = resolve_px_to_um_transform(
        platform="ATERA", original_data_path=tmp_path / "missing", transform_path=spec
    )
    assert x_row[0] == pytest.approx(0.5)
    with pytest.raises(FileNotFoundError, match="Atera pixel_size"):
        resolve_px_to_um_transform(
            platform="ATERA",
            original_data_path=tmp_path / "missing",
            transform_path=None,
        )
