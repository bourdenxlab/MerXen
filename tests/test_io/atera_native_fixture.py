"""Reference layout of a native (vendor-segmentation) ATERA latest store.

:func:`make_native_reference` writes the synthetic Atera bundle
(:mod:`atera_synth`, including its two-nucleus cell), builds the source store
with ``write_atera_spatialdata``, then makes a test-only *reference* latest
store from it: a plain copy with the renames, ``table_original`` and the
segmentation registry entries of the AT-6 work order (§1.4). It does not
hardlink anything; NATIVE_LATEST (WP-A) must produce the same layout with
linked payloads, and the enrichment, QC, clustering, viewer-cache and
visualization packages (WP-B to WP-E) test against this reference.

Layout of the reference latest store:

- ``images/morphology_focus``, ``labels/cell_labels``, ``labels/nucleus_labels``
  and ``points/transcripts`` as in the source store;
- ``shapes/atera_cell_boundaries`` (source ``cell_boundaries``) and
  ``shapes/atera_nucleus`` (source ``nucleus_boundaries``; ``instance_id`` is
  the nucleus row + 1, ``cell_instance_id`` the owning cell);
- ``tables/table_original`` (source ``table``) annotating
  ``atera_cell_boundaries``;
- no ``cell_boundaries``, ``nucleus_boundaries`` or ``table`` element;
- ``merxen_schema.segmentations``: ``original`` (cells, ``table_original``,
  ``original_assignment``) and ``original_nucleus`` (nuclei, no table, no
  assignment, ``cell_key="cell_instance_id"``);
- ``merxen_schema.segmentation_source == "native"``;
- the root ``merxen_atera`` attrs, the ``atera/`` sidecars and the points index
  copied unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Final

import pandas as pd
import spatialdata as sd

from merxen.config import AteraBuildConfig
from merxen.io.builders import atera as atera_builder
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    ORIGINAL_ID_NAMESPACE,
    register_segmentation_branch,
    stamp_merxen_schema,
)
from merxen.platforms import platform_spec

from .atera_synth import write_atera_bundle

#: Small partitions and pools keep the fixture build at a few seconds.
FAST_OPTIONS: Final[dict[str, Any]] = {
    "transcript_partition_rows": 1_048_576,
    "polygon_n_jobs": 2,
    "morphology_levels": 3,
}

SPEC = platform_spec("ATERA")
CELL_SHAPES_KEY: Final = SPEC.original_cell_shape_key
NUCLEUS_SHAPES_KEY: Final = SPEC.original_nucleus_shape_key or "atera_nucleus"
TABLE_KEY: Final = "table_original"
POINTS_KEY: Final = "transcripts"
CELL_KEY: Final = "cell_instance_id"
NUCLEUS_ID_NAMESPACE: Final = "original_nucleus_row_v1"
SOURCE_NAMES: Final = frozenset({"cell_boundaries", "nucleus_boundaries", "table"})


def build_atera_source(tmp_path: Path, **bundle_options: Any) -> Path:
    """Write the synthetic bundle and build its source store.

    Args:
        tmp_path: Test directory; ``roi/`` and ``source.zarr`` are written in it.
        **bundle_options: Passed to ``write_atera_bundle``.

    Returns:
        The built source store.
    """
    synth = write_atera_bundle(tmp_path / "roi", **bundle_options)
    return atera_builder.write_atera_spatialdata(
        input_path=synth.root,
        output_path=tmp_path / "source.zarr",
        build_config=AteraBuildConfig(**FAST_OPTIONS),
        n_workers=2,
        dataset_name="ATERA_SYNTH",
    )


def register_native_branches(sdata: Any) -> None:
    """Add the native registry entries (``original``, ``original_nucleus``).

    Args:
        sdata: A store with the renamed native elements.
    """
    stamp_merxen_schema(sdata, primary_points_key=POINTS_KEY, platform="ATERA")
    register_segmentation_branch(
        sdata,
        "original",
        points_key=POINTS_KEY,
        assignment_column="original_assignment",
        shape_key=CELL_SHAPES_KEY,
        table_key=TABLE_KEY,
        id_namespace=ORIGINAL_ID_NAMESPACE,
        legacy_aliases=("original_seg",),
    )
    register_segmentation_branch(
        sdata,
        "original_nucleus",
        points_key=POINTS_KEY,
        assignment_column=None,
        shape_key=NUCLEUS_SHAPES_KEY,
        table_key=None,
        id_namespace=NUCLEUS_ID_NAMESPACE,
    )
    schema = dict(sdata.attrs[MERXEN_SCHEMA_ATTR])
    registry = dict(schema["segmentations"])
    registry["original_nucleus"] = {
        **registry["original_nucleus"],
        "cell_key": CELL_KEY,
    }
    schema["segmentations"] = registry
    schema["segmentation_source"] = "native"
    sdata.attrs[MERXEN_SCHEMA_ATTR] = schema


def make_reference_latest(source_store: Path, latest_store: Path) -> Path:
    """Copy a source store into the reference native latest layout.

    Args:
        source_store: A built ATERA source store.
        latest_store: Destination (must not exist).

    Returns:
        ``latest_store``.
    """
    shutil.copytree(source_store, latest_store)
    sdata = sd.read_zarr(latest_store)
    cells = sdata.shapes["cell_boundaries"]
    nuclei = sdata.shapes["nucleus_boundaries"]
    table = sdata.tables["table"].copy()
    table.obs["region"] = pd.Categorical([CELL_SHAPES_KEY] * table.n_obs)
    attrs = dict(table.uns["spatialdata_attrs"])
    attrs["region"] = CELL_SHAPES_KEY
    table.uns["spatialdata_attrs"] = attrs

    for name in sorted(SOURCE_NAMES):
        sdata.delete_element_from_disk(name)
    del sdata.shapes["cell_boundaries"]
    del sdata.shapes["nucleus_boundaries"]
    del sdata.tables["table"]

    sdata.shapes[CELL_SHAPES_KEY] = cells
    sdata.shapes[NUCLEUS_SHAPES_KEY] = nuclei
    sdata.tables[TABLE_KEY] = table
    for name in (CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY, TABLE_KEY):
        sdata.write_element(name)
    register_native_branches(sdata)
    sdata.write_attrs()
    sdata.write_consolidated_metadata()
    return latest_store


def make_native_reference(tmp_path: Path, **bundle_options: Any) -> tuple[Path, Path]:
    """Build the synthetic source store and its reference native latest store.

    Args:
        tmp_path: Test directory.
        **bundle_options: Passed to ``write_atera_bundle``.

    Returns:
        ``(source_store, latest_store)``.
    """
    source = build_atera_source(tmp_path, **bundle_options)
    latest = make_reference_latest(
        source, tmp_path / "latest" / "latest_spatialdata.zarr"
    )
    return source, latest
