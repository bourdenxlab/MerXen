"""Segmentation-independent Atera gene density on the native morphology grid.

Read parquet columns in bounded batches, count every eligible gene (including
unassigned rows), sum over z, and smooth in microns. Integer bins are anchored
to native pixel edges. Reconstruction samples their centres at native pixel
centres, without per-tile normalization or smoothing seams.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import dask.array as da
import numpy as np
import pyarrow.parquet as pq
import spatialdata as sd
import xarray as xr
from scipy import ndimage as ndi
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

from merxen.config import AteraTranscriptDensityConfig
from merxen.enrichment.label_lookup import PixelFrame, pixel_indices
from merxen.io.image_source import image_to_cyx
from merxen.io.points_aggregates import read_root_attrs
from merxen.platforms import find_um_per_px, platform_spec
from merxen.visualization.raster_density import points_files

if TYPE_CHECKING:
    from merxen.segmentation.model_identity import TranscriptDensityInputIdentity

logger = logging.getLogger(__name__)
DENSITY_IMAGE_KEY = "atera_transcript_density"
DENSITY_CHANNEL = "transcript_density"
DENSITY_MANIFEST = "transcript_density.json"
DENSITY_VERSION = 1


@dataclass(frozen=True)
class DensityGeometry:
    """The native morphology dimensions and physical pixel pitch."""

    height: int
    width: int
    pixel_size_um: float


@dataclass
class DensityCounts:
    """Coarse integer counts, global native-pixel origin and filtering totals."""

    counts: np.ndarray
    y0: int
    x0: int
    n_rows: int = 0
    n_eligible: int = 0
    n_binned: int = 0
    n_invalid_coordinates: int = 0
    n_outside_image: int = 0


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_tree_identity(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name != DENSITY_MANIFEST
            and not any(
                part.startswith("_napari_compare_")
                for part in path.relative_to(root).parts
            )
        ):
            stat = path.stat()
            entries.append(
                (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
            )
    return _json_hash(entries)


def _native_image_identity(root: Path) -> str:
    """Bind image chunks and semantic metadata without metadata timestamps."""
    metadata_files = {"zarr.json", ".zattrs", ".zarray", ".zgroup", ".zmetadata"}
    entries: list[tuple[str, str] | tuple[str, int, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.name in metadata_files:
            entries.append((relative, _json_hash(json.loads(path.read_text()))))
        else:
            stat = path.stat()
            entries.append((relative, stat.st_size, stat.st_mtime_ns))
    return _json_hash(entries)


def _density_matches_native_image(
    source: Path, density_path: Path, record: dict[str, Any]
) -> bool:
    """Verify an original image or a proven equivalent enriched copy."""
    from merxen.segmentation.density_storage import (
        DENSITY_ATTR,
        embedded_density_record,
        has_embedded_density,
    )

    key = platform_spec("ATERA").native_image_key
    image = source / "images" / key
    if _file_tree_identity(image) == record["image_fingerprint"]:
        return True
    current = _native_image_identity(image)
    if has_embedded_density(source):
        installed = embedded_density_record(source)
        binding = read_root_attrs(source)[DENSITY_ATTR]
        if (
            installed["fingerprint"] == record["fingerprint"]
            and binding.get("native_image_fingerprint") == current
        ):
            return True
    # Legacy manifests included metadata timestamps. An untouched original
    # proves that only metadata serialization changed in the enriched copy.
    for candidate in (density_path, Path(record["source_zarr"])):
        original = candidate / "images" / key
        if (
            original.is_dir()
            and _file_tree_identity(original) == record["image_fingerprint"]
            and _native_image_identity(original) == current
        ):
            return True
    return False


def native_density_geometry(source: Path) -> DensityGeometry:
    """Read Atera geometry from the native image and retained vendor metadata.

    Args:
        source: A built/latest/review Atera SpatialData store.

    Returns:
        Native dimensions and positive pixel size, without loading points/tables.
    """
    sdata = sd.read_zarr(source, selection=("images",))
    image = image_to_cyx(sdata.images[platform_spec("ATERA").native_image_key])
    try:
        pixel_size = find_um_per_px(source, "ATERA")
    except FileNotFoundError:
        pixel_size = float(
            sdata.attrs.get("merxen_atera", {})
            .get("experiment", {})
            .get("pixel_size", 0)
        )
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("Atera transcript density needs a native pixel size")
    return DensityGeometry(int(image.sizes["y"]), int(image.sizes["x"]), pixel_size)


def density_profile(config: AteraTranscriptDensityConfig) -> dict[str, Any]:
    """Return output-relevant parameters, independent of paths or batch sizes.

    Args:
        config: Requested density settings.

    Returns:
        Portable recipe for checkpoint metadata and input provenance.
    """
    return {
        "version": DENSITY_VERSION,
        "min_qv": config.min_qv,
        "bin_size_px": config.bin_size_px,
        "sigma_um": config.sigma_um,
        "gene_filter": "is_gene",
        "assignment_filter": "none",
        "z_projection": "sum_all",
        "units": "transcripts_per_um2",
        "boundary_mode": "reflect",
        "gaussian_truncate": 4.0,
        "reconstruction": "bilinear_pixel_centres",
    }


def accumulate_density_counts(
    source: Path,
    geometry: DensityGeometry,
    config: AteraTranscriptDensityConfig,
    *,
    window: tuple[int, int, int, int] | None = None,
) -> DensityCounts:
    """Stream every qualifying transcript into integer bins of one window.

    Args:
        source: Store holding raw micron ``x/y``, ``is_gene`` and ``qv`` columns.
        geometry: Full native image bounds.
        config: Quality, integer bin width and batch settings.
        window: Optional global native ``(y0, y1, x0, x1)`` window. Callers
            requesting a preview must include the smoothing halo.

    Returns:
        Exact uint64 counts and row totals. No assignment or nuclear mask is read.
    """
    y0, y1, x0, x1 = window or (0, geometry.height, 0, geometry.width)
    factor = config.bin_size_px
    if not (0 <= y0 < y1 <= geometry.height and 0 <= x0 < x1 <= geometry.width):
        raise ValueError("Density window must lie inside the native image")
    y0, x0 = (y0 // factor) * factor, (x0 // factor) * factor
    y1 = min(geometry.height, math.ceil(y1 / factor) * factor)
    x1 = min(geometry.width, math.ceil(x1 / factor) * factor)
    shape = (math.ceil((y1 - y0) / factor), math.ceil((x1 - x0) / factor))
    result = DensityCounts(np.zeros(shape, dtype=np.uint64), y0, x0)
    ps = geometry.pixel_size_um
    bbox = (x0 * ps, y0 * ps, x1 * ps, y1 * ps)
    files, total, _pruned = points_files(source, bbox=bbox if window else None)
    for index, path in enumerate(files, 1):
        with pq.ParquetFile(path) as parquet:
            required = {"x", "y", "is_gene", "qv"}
            if not required.issubset(parquet.schema_arrow.names):
                raise ValueError("Density requires x/y micron, is_gene and qv columns")
            for batch in parquet.iter_batches(
                batch_size=config.batch_rows, columns=sorted(required)
            ):
                columns = {
                    name: batch.column(name).to_numpy(zero_copy_only=False)
                    for name in required
                }
                result.n_rows += batch.num_rows
                eligible = np.asarray(columns["is_gene"], bool) & (
                    columns["qv"] >= config.min_qv
                )
                result.n_eligible += int(eligible.sum())
                x, y = columns["x"][eligible], columns["y"][eligible]
                finite = np.isfinite(x) & np.isfinite(y)
                result.n_invalid_coordinates += int((~finite).sum())
                col, row = pixel_indices(
                    x[finite],
                    y[finite],
                    PixelFrame(um_per_px=geometry.pixel_size_um),
                    convention="corner",
                )
                inside = (
                    (row >= 0)
                    & (row < geometry.height)
                    & (col >= 0)
                    & (col < geometry.width)
                )
                result.n_outside_image += int((~inside).sum())
                keep = inside & (row >= y0) & (row < y1) & (col >= x0) & (col < x1)
                np.add.at(
                    result.counts,
                    ((row[keep] - y0) // factor, (col[keep] - x0) // factor),
                    1,
                )
                result.n_binned += int(keep.sum())
        if index % 20 == 0 or index == len(files):
            logger.info(
                "Density: %d/%d parquet files (%d total), %d transcripts binned",
                index,
                len(files),
                total,
                result.n_binned,
            )
    if int(result.counts.sum(dtype=np.uint64)) != result.n_binned:
        raise RuntimeError("Transcript density count conservation failed")
    return result


def smooth_density_counts(
    counts: DensityCounts,
    geometry: DensityGeometry,
    config: AteraTranscriptDensityConfig,
) -> np.ndarray:
    """Convert counts to transcripts/µm² and apply one global Gaussian.

    Args:
        counts: Native-anchored coarse counts, including any preview halo.
        geometry: Full native geometry.
        config: Bin width and physical Gaussian standard deviation.

    Returns:
        Float32 physical density on coarse bin centres.
    """
    factor = config.bin_size_px
    density = counts.counts.astype(np.float32)
    # Only edge bins have a smaller area; an outer-product area array would
    # allocate several unnecessary full slides (tens of GB) at native resolution.
    density /= np.float32((factor * geometry.pixel_size_um) ** 2)
    last_height = min(
        factor, geometry.height - counts.y0 - (density.shape[0] - 1) * factor
    )
    last_width = min(
        factor, geometry.width - counts.x0 - (density.shape[1] - 1) * factor
    )
    if last_height < factor:
        density[-1, :] *= np.float32(factor / last_height)
    if last_width < factor:
        density[:, -1] *= np.float32(factor / last_width)
    ndi.gaussian_filter(
        density,
        config.sigma_um / (factor * geometry.pixel_size_um),
        output=density,
        mode="reflect",
        truncate=4.0,
    )
    return density


def reconstruct_density_window(
    density: np.ndarray,
    *,
    origin: tuple[int, int],
    bin_size_px: int,
    window: tuple[int, int, int, int],
) -> np.ndarray:
    """Sample coarse bin centres on native pixel centres, without tile seams.

    Args:
        density: Smoothed coarse raster.
        origin: Global native ``(y0, x0)`` of its first bin corner.
        bin_size_px: Native pixels per coarse bin edge.
        window: Native ``(y0, y1, x0, x1)`` requested output.

    Returns:
        Float32 native-resolution density in precisely the morphology grid.
    """
    y0, y1, x0, x1 = window
    rows = (np.arange(y0, y1) - origin[0] + 0.5) / bin_size_px - 0.5
    cols = (np.arange(x0, x1) - origin[1] + 0.5) / bin_size_px - 0.5
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    return np.asarray(
        ndi.map_coordinates(
            density, [yy, xx], order=1, mode="nearest", prefilter=False
        ),
        dtype=np.float32,
    )


def create_transcript_density(config: AteraTranscriptDensityConfig) -> Path:
    """Write an independent, complete density store before Atera segmentation.

    Args:
        config: Native source, fresh output directory and raster parameters.

    Returns:
        SpatialData density store. Source data and previous outputs are untouched.
    """
    source, output = config.source_zarr_path.resolve(), config.output_dir.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Density output must be separate from the source store")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Density output must be new or empty")
    geometry = native_density_geometry(source)
    points = sd.read_zarr(source, selection=("points",)).points["transcripts"]
    transform = get_transformation(
        points, to_coordinate_system="global"
    ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    if not np.allclose(
        transform,
        np.diag([1 / geometry.pixel_size_um, 1 / geometry.pixel_size_um, 1]),
        rtol=1e-6,
        atol=1e-10,
    ):
        raise ValueError(
            "Density requires Atera micron transcripts in the native frame"
        )
    counts = accumulate_density_counts(source, geometry, config)
    if not counts.n_binned:
        raise ValueError("No eligible gene transcripts lie in the native image")
    density = smooth_density_counts(counts, geometry, config)
    totals = {key: value for key, value in vars(counts).items() if key != "counts"}
    del counts
    template = da.empty(
        (1, geometry.height, geometry.width),
        chunks=(1, config.chunk_size_px, config.chunk_size_px),
        dtype=np.float32,
    )

    def block(_unused: np.ndarray, block_info: Any = None) -> np.ndarray:
        location = block_info[None]["array-location"]
        return reconstruct_density_window(
            density,
            origin=(0, 0),
            bin_size_px=config.bin_size_px,
            window=(*location[1], *location[2]),
        )[None]

    name = "atera-density-" + _json_hash(density_profile(config))
    array = (
        da.from_array(density[None], chunks=template.chunks, name=name, asarray=False)
        if config.bin_size_px == 1
        else template.map_blocks(
            block, name=name, dtype=np.float32, meta=np.empty((0, 0, 0), np.float32)
        )
    )
    original = sd.read_zarr(source, selection=("images",))
    native = original.images[platform_spec("ATERA").native_image_key]
    image = Image2DModel.parse(
        array,
        dims=("c", "y", "x"),
        c_coords=[DENSITY_CHANNEL],
        transformations={
            "global": get_transformation(native, to_coordinate_system="global")
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "transcript_density.zarr"
    with dask.config.set(scheduler="threads", num_workers=4):
        sd.SpatialData(images={DENSITY_IMAGE_KEY: image}).write(target)
    files, _, _ = points_files(source)
    record = {
        "schema_version": 1,
        "source_zarr": str(source),
        "geometry": asdict(geometry),
        "profile": density_profile(config),
        "totals": totals,
        "image_fingerprint": _file_tree_identity(
            source / "images" / platform_spec("ATERA").native_image_key
        ),
        "points_fingerprint": _json_hash(
            [
                (
                    str(path.relative_to(source)),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in files
            ]
        ),
        "raster_fingerprint": _file_tree_identity(target),
    }
    record["fingerprint"] = _json_hash(record)
    (target / DENSITY_MANIFEST).write_text(json.dumps(record, indent=2) + "\n")
    (output / DENSITY_MANIFEST).write_text(json.dumps(record, indent=2) + "\n")
    logger.info("Wrote density for %d gene transcripts", totals["n_binned"])
    return target


def read_density_record(path: Path) -> dict[str, Any]:
    """Read and verify the recipe/content identity of a completed density raster.

    Args:
        path: Completed density SpatialData store.

    Returns:
        Manifest, rejecting incomplete or modified metadata.
    """
    from merxen.segmentation.density_storage import (
        embedded_density_record,
        has_embedded_density,
    )

    if has_embedded_density(path):
        return embedded_density_record(path)
    record: dict[str, Any] = json.loads((path / DENSITY_MANIFEST).read_text())
    if record.get("fingerprint") != _json_hash(
        {key: value for key, value in record.items() if key != "fingerprint"}
    ):
        raise ValueError("Transcript density manifest identity mismatch")
    if record["raster_fingerprint"] != _file_tree_identity(path):
        raise ValueError("Transcript density raster changed after generation")
    return record


def density_input_identity(path: Path | None) -> TranscriptDensityInputIdentity | None:
    """Bind a model input to its verified raster and reusable method recipe.

    Args:
        path: Density store, or ``None`` for historical two-stain inputs.

    Returns:
        Typed inference identity, or ``None`` when density is absent.
    """
    from merxen.segmentation.model_identity import TranscriptDensityInputIdentity

    if path is None:
        return None
    record = read_density_record(path)
    return TranscriptDensityInputIdentity(
        fingerprint=record["fingerprint"],
        profile=record["profile"],
        raster_path=path.resolve(),
    )


def image_with_density(sdata: Any, source: Path, density_path: Path | None) -> Any:
    """Append a verified auxiliary density channel to the native Atera image.

    Args:
        sdata: Source SpatialData, with images loaded.
        source: Source store path, also usable for an enriched copy of its image.
        density_path: Separate pre-segmentation raster; ``None`` keeps native stains.

    Returns:
        Lazy CYX array, retaining native stains and adding density when requested.
    """
    key = platform_spec("ATERA").native_image_key
    native = image_to_cyx(sdata.images[key])
    if density_path is None:
        return native
    record = read_density_record(density_path)
    if not _density_matches_native_image(source, density_path, record):
        raise ValueError("Transcript density belongs to a different native image")
    if source.resolve() == Path(record["source_zarr"]).resolve():
        files, _, _ = points_files(source)
        found = _json_hash(
            [
                (
                    str(path.relative_to(source)),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in files
            ]
        )
        if found != record["points_fingerprint"]:
            raise ValueError("Source transcripts changed since density generation")
    density_store = sd.read_zarr(density_path, selection=("images",))
    element = density_store.images[DENSITY_IMAGE_KEY]
    density = image_to_cyx(element)
    if density.shape[1:] != native.shape[1:]:
        raise ValueError("Transcript density shape differs from native morphology")
    left = get_transformation(
        sdata.images[key], to_coordinate_system="global"
    ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    right = get_transformation(element, to_coordinate_system="global").to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    if not np.allclose(left, right, rtol=0, atol=1e-10):
        raise ValueError("Transcript density coordinate grid differs from morphology")
    return xr.concat([native, density], dim="c", join="exact")
