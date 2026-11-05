"""10x Atera ROI folder to SpatialData writer (plan §3, §4.2).

:func:`write_atera_spatialdata` assembles the elements the Atera readers
(:mod:`merxen.io.atera_reader`) produce into MerXen's source store:

1. start the memory watchdog (:class:`RssWatchdog`);
2. set dask and zarr concurrency explicitly;
3. resolve the bundle;
4. polygonise the vendor masks (a process pool, started before dask runs);
5. run the readers (labels, image, table, transcripts);
6. parse every element with its spatialdata model (points with ``sort=True``);
7. apply the source contract with ``points_canonical=True``;
8. write the store to ``<target>.partial-<uuid>``; the transcript statistics
   are recorded by the partition reads of that same write;
9. copy the vendor metadata files verbatim under ``atera/``, write the points
   index, the full header in ``attrs["merxen_atera"]`` and the completion
   marker ``merxen_build_complete.json`` with the build fingerprint;
10. rename the partial store onto the target;
11. release memory.

A store without the marker is incomplete (a killed or failed build) and is
never reused; a complete store is reused only when its fingerprint matches
the requested build (:func:`check_cached_atera_build`).

Vendor metadata is kept whole in the store (plan §3.11, user decision of
2026-11-05). Run, slide, region and instrument values are sample names, so log
lines and error messages here name the dataset by its ``dataset_name`` alias
and never print a manifest value or an input path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import spatialdata as sd
import zarr
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations import Identity, Scale

from merxen.config import AteraBuildConfig
from merxen.io import atera_reader as ar
from merxen.io.spatialdata_io import (
    prepare_source_spatialdata_contract,
    write_spatialdata_zarr,
)
from merxen.io.spatialdata_schema import INSTANCE_ID_COLUMN
from merxen.memory import force_release
from merxen.path_utils import remove_path
from merxen.platforms import platform_spec

logger = logging.getLogger(__name__)

#: Bump when the builder writes a different store for the same inputs and
#: options; it is part of the reuse fingerprint.
ATERA_BUILD_FORMAT_VERSION: Final = 1
#: Completion marker at the store root; a store without it is incomplete.
BUILD_MARKER_NAME: Final = "merxen_build_complete.json"
#: Partition index of the transcripts parquet files, at the store root.
POINTS_INDEX_NAME: Final = "merxen_atera_points_index.json"
#: Root attrs key of the vendor header and the build record.
ATERA_ATTRS_KEY: Final = "merxen_atera"
POINTS_KEY: Final = "transcripts"
IMAGE_KEY: Final = platform_spec("ATERA").native_image_key
CELL_SHAPES_KEY: Final = "cell_boundaries"
NUCLEUS_SHAPES_KEY: Final = "nucleus_boundaries"
TABLE_KEY: Final = "table"
#: Exit status of a build stopped by the watchdog. It differs from 137 (a
#: SIGKILL from the kernel or the scheduler), so the workflow retries an
#: external kill with more memory but not a build that hit its own cap.
RSS_LIMIT_EXIT_CODE: Final = 99
#: AteraBuildConfig fields that change how the build runs, not what it writes.
RUNTIME_ONLY_FIELDS: Final = frozenset({"max_rss_gb", "polygon_n_jobs"})
#: AteraBuildConfig fields naming input files; the fingerprint records the
#: files themselves (name, size, mtime) instead of host paths.
INPUT_PATH_FIELDS: Final = frozenset(
    {"transcripts_path", "cells_path", "cell_feature_matrix_path"}
)
_GIB: Final = float(1 << 30)


# --------------------------------------------------------------------------- #
# Memory watchdog
# --------------------------------------------------------------------------- #


def _anonymous_rss_bytes(process: psutil.Process) -> int:
    """Anonymous resident memory of one process (``RssAnon``, else USS).

    File-backed pages (a memory-mapped mask touched by every polygonisation
    worker) are shared page cache, so they are not counted.
    """
    try:
        with open(f"/proc/{process.pid}/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(process.memory_full_info().uss)


def process_tree_anonymous_rss_bytes(pid: int | None = None) -> int:
    """Anonymous resident memory of a process and all its descendants.

    Args:
        pid: Root process; this process when ``None``.

    Returns:
        Bytes summed over the tree; processes that exit while being read are
        skipped.
    """
    root = psutil.Process(pid)
    total = 0
    for process in (root, *root.children(recursive=True)):
        try:
            total += _anonymous_rss_bytes(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return total


def _stop_build(used_gb: float, limit_gb: float) -> None:
    """Kill the process tree of a build that exceeded its memory cap."""
    logger.error(
        "[ATERA] Build stopped by the memory watchdog: %.1f GB anonymous RSS "
        "> %.1f GB cap (atera max_rss_gb). Raise the cap and the task memory.",
        used_gb,
        limit_gb,
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    for child in psutil.Process().children(recursive=True):
        try:
            child.send_signal(signal.SIGKILL)
        except psutil.Error:
            continue
    os._exit(RSS_LIMIT_EXIT_CODE)


class RssWatchdog:
    """Stop the build when its process tree exceeds an anonymous-memory cap.

    A background thread polls the anonymous RSS of this process and its
    descendants (``RssAnon`` from ``/proc``, else USS; never ``rss``, which
    counts shared page cache once per process). The local executor only
    schedules memory and does not cap it, so the watchdog is the cap that
    keeps a runaway build from the kernel OOM killer, which may pick another
    session's process. With no cap it only records the peak.

    Use it as a context manager, or call :meth:`start` and :meth:`stop`.
    """

    def __init__(
        self,
        limit_gb: float | None,
        *,
        interval_s: float = 2.0,
        on_exceed: Callable[[float, float], None] | None = None,
        measure: Callable[[], int] = process_tree_anonymous_rss_bytes,
    ) -> None:
        """Prepare a watchdog; nothing runs until :meth:`start`.

        Args:
            limit_gb: Cap in GiB; ``None`` only records the peak.
            interval_s: Seconds between two measurements.
            on_exceed: Called with ``(used_gb, limit_gb)`` when the cap is
                exceeded; the default logs, kills the child processes and
                exits with :data:`RSS_LIMIT_EXIT_CODE`.
            measure: Returns the bytes to compare with the cap.
        """
        self.limit_gb = limit_gb
        self.interval_s = interval_s
        self._on_exceed = on_exceed or _stop_build
        self._measure = measure
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_gb = 0.0

    def check(self) -> float:
        """Measure once, update the peak and act on the cap.

        Returns:
            The measured anonymous RSS in GiB.
        """
        used_gb = self._measure() / _GIB
        self.peak_gb = max(self.peak_gb, used_gb)
        if self.limit_gb is not None and used_gb > self.limit_gb:
            self._on_exceed(used_gb, self.limit_gb)
        return used_gb

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.check()

    def start(self) -> RssWatchdog:
        """Start polling in a daemon thread.

        Returns:
            This watchdog.
        """
        if self._thread is None:
            self.check()
            self._thread = threading.Thread(
                target=self._run, name="atera-rss-watchdog", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling (idempotent) after a last measurement."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            self.check()

    def __enter__(self) -> RssWatchdog:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def resolve_worker_count(n_workers: int | None = None) -> int:
    """Return the number of worker threads the build uses.

    Args:
        n_workers: Explicit count; else ``DASK_NUM_WORKERS`` (the workflow
            exports ``task.cpus``), else the CPUs this process may run on.

    Returns:
        A positive count.

    Raises:
        ValueError: If the count or ``DASK_NUM_WORKERS`` is not a positive
            integer.
    """
    if n_workers is None:
        raw = os.environ.get("DASK_NUM_WORKERS", "").strip()
        if raw:
            try:
                n_workers = int(raw)
            except ValueError:
                raise ValueError(
                    f"DASK_NUM_WORKERS must be a positive integer, got {raw!r}"
                ) from None
        else:
            n_workers = len(os.sched_getaffinity(0))
    if n_workers < 1:
        raise ValueError(f"the worker count must be positive, got {n_workers}")
    return n_workers


@contextmanager
def atera_build_concurrency(n_workers: int) -> Iterator[None]:
    """Run dask on ``n_workers`` threads and cap zarr's decode pool alike.

    The transcript statistics are recorded in this process, so the threaded
    scheduler is required; zarr's async pool does not follow dask's setting.

    Args:
        n_workers: Worker threads for both.

    Yields:
        Nothing; the previous settings come back on exit.
    """
    with (
        dask.config.set(scheduler="threads", num_workers=n_workers),
        zarr.config.set({"threading.max_workers": n_workers}),
    ):
        yield


# --------------------------------------------------------------------------- #
# Fingerprint, completion marker and reuse
# --------------------------------------------------------------------------- #


def resolve_atera_input(
    input_path: Path,
    build_config: AteraBuildConfig,
    *,
    atera_spec_path: Path | None = None,
) -> ar.AteraBundle:
    """Resolve the bundle of one ROI folder with the build's explicit paths.

    Args:
        input_path: The extracted ROI folder.
        build_config: Atera build options (explicit zip paths, species).
        atera_spec_path: Metadata-only replacement of ``experiment.spatial``.

    Returns:
        The resolved bundle.
    """
    return ar.resolve_atera_bundle(
        Path(input_path),
        spec_override=atera_spec_path,
        transcripts_path=build_config.transcripts_path,
        cells_path=build_config.cells_path,
        cfm_path=build_config.cell_feature_matrix_path,
        species=build_config.species,
    )


def build_options(build_config: AteraBuildConfig) -> dict[str, Any]:
    """Return the options that decide what the build writes (JSON values).

    Args:
        build_config: Atera build options.

    Returns:
        Every field but the runtime-only ones and the input paths.
    """
    return build_config.model_dump(
        mode="json", exclude=set(RUNTIME_ONLY_FIELDS | INPUT_PATH_FIELDS)
    )


def _input_files(bundle: ar.AteraBundle) -> list[Path]:
    files = {
        bundle.transcripts_zip,
        bundle.cells_zip,
        bundle.cell_feature_matrix_zip,
        *bundle.morphology_2d_files,
        *(sidecar.source for sidecar in bundle.sidecars),
    }
    return sorted(files)


def _file_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def atera_build_fingerprint(
    bundle: ar.AteraBundle, build_config: AteraBuildConfig
) -> str:
    """Hash what decides the store: options, input files and builder format.

    Input files are identified by their bundle-relative name, size and
    modification time, so a re-extracted or edited bundle changes the
    fingerprint without hashing 16 GB of zips.

    Args:
        bundle: Resolved bundle (with its sidecars and any spec override).
        build_config: Atera build options.

    Returns:
        A SHA-256 hex digest.
    """
    files = []
    for path in _input_files(bundle):
        stat = path.stat()
        files.append(
            {
                "name": _file_label(path, bundle.root),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "format_version": ATERA_BUILD_FORMAT_VERSION,
        "options": build_options(build_config),
        "files": files,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_build_marker(store: Path) -> dict[str, Any] | None:
    """Return the completion marker of a store, or ``None`` if it has none.

    Args:
        store: A SpatialData store.

    Returns:
        The parsed marker; ``None`` when it is missing or unreadable (an
        incomplete store).
    """
    marker = Path(store) / BUILD_MARKER_NAME
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_cached_atera_build(
    path: Path, fingerprint: str | None, *, force_rerun: bool
) -> bool:
    """Decide whether an existing Atera store may be reused.

    Args:
        path: The existing store.
        fingerprint: Fingerprint of the requested build; ``None`` when no raw
            input is available to compute it (then a complete store is reused
            as it is).
        force_rerun: Rebuild a complete store whose fingerprint differs
            instead of refusing.

    Returns:
        True to reuse the store; False to rebuild it (it is incomplete, or
        ``force_rerun`` overrides a mismatch).

    Raises:
        ValueError: If the store is complete but was built from other inputs
            or options, and ``force_rerun`` is not set.
    """
    marker = read_build_marker(path)
    if marker is None:
        return False
    recorded = marker.get("build_fingerprint")
    if fingerprint is None or recorded == fingerprint:
        return True
    if force_rerun:
        return False
    raise ValueError(
        "The existing Atera SpatialData store was built from other inputs or "
        "options (build_fingerprint mismatch). Rebuild it with "
        "--force_spatialdata_build true (build-spatialdata --force-rerun), or "
        "point atera_spatialdata_path at a new location."
    )


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def _remove_stale_partials(target: Path) -> None:
    """Remove partial stores a killed build left beside ``target``."""
    for stale in target.parent.glob(f"{target.name}.partial-*"):
        logger.warning("[ATERA] Removing an incomplete earlier build: %s", stale.name)
        remove_path(stale)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


@dataclass
class AteraAssembly:
    """The in-memory store and what the write fills in afterwards.

    Attributes:
        sdata: The SpatialData object to write.
        partition_specs: Transcript partitions in order (empty without points).
        aggregates: Statistics the partition reads record during the write.
        feature_names: Every transcript feature, in vendor order.
        shape_stats: Polygon vertex statistics per shapes element.
    """

    sdata: sd.SpatialData
    partition_specs: list[ar.PartitionSpec] = field(default_factory=list)
    aggregates: ar.TranscriptAggregates | None = None
    feature_names: tuple[str, ...] = ()
    shape_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)


def _microns_to_pixels(pixel_size_um: float) -> Scale:
    scale = 1.0 / pixel_size_um
    return Scale([scale, scale], axes=("x", "y"))


def _atera_shapes(
    bundle: ar.AteraBundle,
    cells: ar.AteraCells,
    build_config: AteraBuildConfig,
    *,
    n_jobs: int,
    workdir: Path,
) -> dict[str, gpd.GeoDataFrame]:
    wanted = {
        CELL_SHAPES_KEY: build_config.cells_boundaries,
        NUCLEUS_SHAPES_KEY: build_config.nucleus_boundaries,
    }
    if not any(wanted.values()):
        return {}
    if build_config.boundary_source == "masks":
        shapes = ar.atera_shapes_from_masks(
            bundle,
            cells,
            n_jobs=n_jobs,
            workdir=workdir,
            simplify_px=build_config.boundary_simplify_px,
        )
    else:
        shapes = ar.atera_shapes_from_vendor_polygons(cells, bundle.pixel_size_um)
    return {key: frame for key, frame in shapes.items() if wanted[key]}


def _check_table_cells_have_shapes(
    shapes: gpd.GeoDataFrame, cells: ar.AteraCells
) -> None:
    missing = np.setdiff1d(
        cells.instance_id, shapes[INSTANCE_ID_COLUMN].to_numpy(dtype=np.uint64)
    )
    if missing.size:
        raise ar.AteraFormatError(
            f"{missing.size} cell(s) have no cell polygon (e.g. instance_id "
            f"{int(missing[0])}); the table needs a shape per cell. Use "
            "boundary_source=masks."
        )


def _label_elements(
    bundle: ar.AteraBundle,
    build_config: AteraBuildConfig,
    image_shapes: Sequence[tuple[int, int]] | None,
) -> dict[str, Any]:
    keys = [
        key
        for key, wanted in (
            ("cell_labels", build_config.cells_labels),
            ("nucleus_labels", build_config.nucleus_labels),
        )
        if wanted
    ]
    if not keys:
        return {}
    arrays = ar.read_atera_labels(bundle)
    n_levels = len(image_shapes) if image_shapes else build_config.morphology_levels
    labels = {}
    for key in keys:
        parsed = Labels2DModel.parse(
            arrays[key],
            dims=("y", "x"),
            scale_factors=[2] * (n_levels - 1) if n_levels > 1 else None,
            transformations={"global": Identity()},
        )
        if image_shapes is not None:
            label_shapes = _level_shapes(parsed)
            if label_shapes != list(image_shapes):
                raise ar.AteraFormatError(
                    f"{key} pyramid levels {label_shapes} differ from the "
                    f"morphology levels {list(image_shapes)}"
                )
        labels[key] = parsed
    return labels


def _level_shapes(element: Any) -> list[tuple[int, int]]:
    """``(y, x)`` shape of every level of an image or labels element."""
    if not hasattr(element, "children"):
        return [(int(element.shape[-2]), int(element.shape[-1]))]
    shapes = []
    for name in sorted(
        element.children, key=lambda key: int(key.removeprefix("scale"))
    ):
        array = next(iter(element[name].data_vars.values()))
        shapes.append((int(array.shape[-2]), int(array.shape[-1])))
    return shapes


def _points_element(
    bundle: ar.AteraBundle, build_config: AteraBuildConfig
) -> tuple[Any, list[ar.PartitionSpec], ar.TranscriptAggregates]:
    points, specs, aggregates = ar.read_atera_transcripts(
        bundle,
        min_qv=build_config.min_qv,
        partition_rows=build_config.transcript_partition_rows,
        spatial_order=build_config.spatial_order,
        block_um=build_config.spatial_block_um,
        min_block_rows=build_config.min_block_rows,
        max_tiles_in_flight=build_config.max_tiles_in_flight,
        emit_codeword_index=build_config.emit_codeword_index,
        tile_allowlist=build_config.tile_allowlist,
    )
    # sort=True skips spatialdata's monotonic-index check, a full decode pass.
    parsed = PointsModel.parse(
        points,
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="feature_name",
        instance_key="original_assignment",
        transformations={"global": _microns_to_pixels(bundle.pixel_size_um)},
        sort=True,
    )
    return parsed, specs, aggregates


def assemble_atera_sdata(
    bundle: ar.AteraBundle,
    build_config: AteraBuildConfig,
    *,
    n_jobs: int,
    workdir: Path,
) -> AteraAssembly:
    """Read every requested element and assemble the source store in memory.

    Nothing heavy is computed here except the shapes, the table and the cell
    records: images, labels and points stay lazy until the write.

    Args:
        bundle: Resolved bundle.
        build_config: Atera build options.
        n_jobs: Processes for the mask polygonisation (and threads for its
            simplification).
        workdir: Directory for the temporary memory-mapped masks.

    Returns:
        The assembly; its points are parsed and its contract applied.

    Raises:
        AteraFormatError: If the elements disagree (shapes per cell, pyramid
            levels, cell and feature order).
    """
    cells = ar.read_atera_cells(bundle)
    shapes = _atera_shapes(bundle, cells, build_config, n_jobs=n_jobs, workdir=workdir)
    transform = _microns_to_pixels(bundle.pixel_size_um)
    shape_elements = {
        key: ShapesModel.parse(frame, transformations={"global": transform})
        for key, frame in shapes.items()
    }
    shape_stats = {key: ar.polygon_vertex_stats(frame) for key, frame in shapes.items()}

    images: dict[str, Any] = {}
    image_shapes: list[tuple[int, int]] | None = None
    if build_config.morphology_2d:
        tree = ar.read_atera_morphology_2d(
            bundle, n_levels=build_config.morphology_levels
        )
        Image2DModel().validate(tree)
        images[IMAGE_KEY] = tree
        image_shapes = _level_shapes(tree)
    labels = _label_elements(bundle, build_config, image_shapes)

    tables: dict[str, Any] = {}
    if build_config.cells_table:
        _check_table_cells_have_shapes(shapes[CELL_SHAPES_KEY], cells)
        table = ar.read_atera_table(
            bundle,
            cells,
            gex_only=build_config.table_gex_only,
            keep_vendor_annotations=build_config.keep_vendor_annotations,
        )
        table.obs["region"] = pd.Categorical(
            np.repeat(CELL_SHAPES_KEY, table.n_obs), categories=[CELL_SHAPES_KEY]
        )
        tables[TABLE_KEY] = TableModel.parse(
            table,
            region=CELL_SHAPES_KEY,
            region_key="region",
            instance_key=INSTANCE_ID_COLUMN,
        )

    assembly = AteraAssembly(
        sdata=sd.SpatialData(),
        feature_names=ar.read_atera_gene_names(bundle),
        shape_stats=shape_stats,
    )
    points: dict[str, Any] = {}
    if build_config.transcripts:
        parsed, specs, aggregates = _points_element(bundle, build_config)
        points[POINTS_KEY] = parsed
        assembly.partition_specs = specs
        assembly.aggregates = aggregates

    # The native image must come first: segmentation takes images[0].
    assembly.sdata = sd.SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shape_elements,
        tables=tables,
    )
    prepare_source_spatialdata_contract(
        assembly.sdata, platform="ATERA", points_canonical=True
    )
    return assembly


# --------------------------------------------------------------------------- #
# Points index
# --------------------------------------------------------------------------- #


def _column_range(metadata: pq.FileMetaData, column: str) -> list[float] | None:
    """Min and max of one column over a parquet file's row-group statistics."""
    names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
    if column not in names:
        return None
    position = names.index(column)
    lows, highs = [], []
    for group in range(metadata.num_row_groups):
        chunk = metadata.row_group(group)
        if chunk.num_rows == 0:
            continue
        stats = chunk.column(position).statistics
        if stats is None or not stats.has_min_max:
            return None
        lows.append(stats.min)
        highs.append(stats.max)
    if not lows:
        return None
    return [min(lows), max(highs)]


def write_points_index(
    store: Path,
    specs: Sequence[ar.PartitionSpec],
    *,
    points_key: str = POINTS_KEY,
) -> dict[str, Any]:
    """Index the written transcript parquet files by tile, bbox and ID range.

    The index sits at the store root (store rewrites keep it). It is keyed
    by parquet file name, never by dask partition number: a reader may split
    or merge files. Only the footers are read.

    Args:
        store: The written store.
        specs: Partition specs in partition order (file ``part.<k>`` holds
            partition ``k``).
        points_key: Points element.

    Returns:
        The index that was written.

    Raises:
        RuntimeError: If the parquet files do not match the partitions.
    """
    directory = Path(store, "points", points_key, "points.parquet")
    files = sorted(
        directory.glob("part.*.parquet"), key=lambda path: int(path.name.split(".")[1])
    )
    if specs and len(files) != len(specs):
        raise RuntimeError(
            f"{len(files)} transcript parquet files for {len(specs)} partitions"
        )
    entries = []
    total = 0
    for position, path in enumerate(files):
        metadata = pq.read_metadata(path)
        total += metadata.num_rows
        spec = specs[position] if specs else None
        x_range = _column_range(metadata, "x")
        y_range = _column_range(metadata, "y")
        entries.append(
            {
                "file": path.name,
                "tile": spec.tile_key if spec else None,
                "tile_rows": [spec.row_start, spec.row_stop] if spec else None,
                "block_bounds_um": (
                    list(spec.block_bounds_um)
                    if spec and spec.block_bounds_um is not None
                    else None
                ),
                "n_rows": metadata.num_rows,
                "n_row_groups": metadata.num_row_groups,
                "bbox_um": ([*x_range, *y_range] if x_range and y_range else None),
                "transcript_id_range": _column_range(metadata, "transcript_id"),
            }
        )
    index = {
        "format_version": 1,
        "points": {
            points_key: {
                "directory": f"points/{points_key}/points.parquet",
                "fingerprint": {"n_files": len(files), "n_rows": total},
                "files": entries,
            }
        },
    }
    _write_json(Path(store) / POINTS_INDEX_NAME, index)
    return index


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats (not JSON) with ``None``, recursively."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


def write_atera_spatialdata(
    *,
    input_path: Path,
    output_path: Path,
    build_config: AteraBuildConfig,
    atera_spec_path: Path | None = None,
    n_workers: int | None = None,
    dataset_name: str = "ATERA",
    workdir: Path | None = None,
) -> Path:
    """Build an Atera SpatialData store from one extracted ROI folder.

    The store is written beside the target as ``<name>.partial-<uuid>`` and
    renamed onto it once complete, so the target is either absent or whole.
    An existing target is replaced.

    Args:
        input_path: The extracted ROI folder.
        output_path: Destination store.
        build_config: Atera build options.
        atera_spec_path: Metadata-only replacement of ``experiment.spatial``.
        n_workers: Dask and zarr worker threads (default: ``DASK_NUM_WORKERS``,
            else the available CPUs); also the polygonisation pool size unless
            ``build_config.polygon_n_jobs`` is set.
        dataset_name: Alias of the dataset, used in log lines.
        workdir: Directory for the temporary masks (default: beside the
            target).

    Returns:
        ``output_path``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workers = resolve_worker_count(n_workers)
    started = time.monotonic()
    with (
        RssWatchdog(build_config.max_rss_gb) as watchdog,
        atera_build_concurrency(workers),
    ):
        bundle = resolve_atera_input(
            input_path, build_config, atera_spec_path=atera_spec_path
        )
        fingerprint = atera_build_fingerprint(bundle, build_config)
        logger.info(
            "[%s] Building Atera SpatialData (%d workers, min_qv=%s, %d channels)",
            dataset_name,
            workers,
            build_config.min_qv,
            len(bundle.morphology_channel_names),
        )
        _remove_stale_partials(output_path)
        partial = output_path.with_name(
            f"{output_path.name}.partial-{uuid.uuid4().hex}"
        )
        scratch = Path(workdir or output_path.parent) / f".{partial.name}.work"
        try:
            assembly = assemble_atera_sdata(
                bundle,
                build_config,
                n_jobs=build_config.polygon_n_jobs or workers,
                workdir=scratch,
            )
            write_spatialdata_zarr(assembly.sdata, partial, overwrite=False)
            watchdog.check()
            _finish_store(
                partial,
                assembly,
                bundle,
                build_config,
                fingerprint=fingerprint,
                peak_gb=watchdog.peak_gb,
                elapsed_s=time.monotonic() - started,
            )
        except BaseException:
            remove_path(partial)
            raise
        finally:
            remove_path(scratch)
        remove_path(output_path)
        os.replace(partial, output_path)
        del assembly
    force_release("atera build")
    logger.info(
        "[%s] Atera SpatialData complete in %.0f s (peak anonymous RSS %.1f GB)",
        dataset_name,
        time.monotonic() - started,
        watchdog.peak_gb,
    )
    return output_path


def _finish_store(
    store: Path,
    assembly: AteraAssembly,
    bundle: ar.AteraBundle,
    build_config: AteraBuildConfig,
    *,
    fingerprint: str,
    peak_gb: float,
    elapsed_s: float,
) -> None:
    """Add sidecars, the index, the header attrs and the completion marker."""
    sidecars = ar.copy_atera_sidecars(bundle, store)
    header = ar.atera_metadata_attrs(bundle, sidecars)
    header["build"] = {
        "format_version": ATERA_BUILD_FORMAT_VERSION,
        "build_fingerprint": fingerprint,
        "options": build_options(build_config),
        "pixel_size_um": bundle.pixel_size_um,
        "analysis_sw_version": bundle.analysis_sw_version,
        "morphology_channels": list(bundle.morphology_channel_names),
        "elements": {
            kind: sorted(map(str, getattr(assembly.sdata, kind)))
            for kind in ("images", "labels", "points", "shapes", "tables")
        },
    }
    header["shapes"] = assembly.shape_stats
    if assembly.aggregates is not None:
        index = write_points_index(store, assembly.partition_specs)
        transcripts = assembly.aggregates.summary().to_attrs()
        transcripts["feature_names"] = list(assembly.feature_names)
        transcripts["points_index"] = POINTS_INDEX_NAME
        transcripts["n_parquet_files"] = index["points"][POINTS_KEY]["fingerprint"][
            "n_files"
        ]
        header["transcripts"] = transcripts
    sdata = assembly.sdata
    sdata.attrs[ATERA_ATTRS_KEY] = _json_safe(header)
    sdata.write_attrs()
    if sdata.has_consolidated_metadata():
        sdata.write_consolidated_metadata()
    _write_json(
        store / BUILD_MARKER_NAME,
        {
            "platform": "ATERA",
            "format_version": ATERA_BUILD_FORMAT_VERSION,
            "build_fingerprint": fingerprint,
            "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed_s, 1),
            "peak_anonymous_rss_gb": round(peak_gb, 3),
        },
    )
