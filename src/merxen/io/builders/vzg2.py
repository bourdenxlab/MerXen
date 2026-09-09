"""Read Vizgen ``.vzg2`` archives into SpatialData.

VZG2 is a POSIX tar archive containing a manifest, JPEG-12 compressed OME-Zarr
image chunks, cell-level Parquet tables, and packed cell polygons.  Vizgen does
not publish a decoder for the packed transcript tiles, so this reader preserves
every recoverable source element without synthesizing transcript coordinates.

The cell polygon decoding follows the Apache-2.0 licensed implementation in
Vizgen's ``vpt-segmentation-packing`` package.
"""

from __future__ import annotations

import json
import logging
import math
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata
import geopandas
import imagecodecs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import xarray
from dask import array as da
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation

from merxen.config import MerscopeBuildConfig
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME
from merxen.io.spatialdata_io import (
    prepare_source_spatialdata_contract,
    write_spatialdata_zarr,
)

logger = logging.getLogger(__name__)

_MANIFEST_PATH = "manifest.json"
_CELL_POLYGON_BYTES = 136
_CELL_POLYGON_VERTICES = 44


@dataclass(frozen=True)
class Vzg2Member:
    """Byte range for one regular file in a local VZG2 tar archive."""

    offset: int
    size: int


@dataclass(frozen=True)
class Vzg2Archive:
    """Random-access index over a local VZG2 tar archive."""

    path: Path
    members: dict[str, Vzg2Member]

    @classmethod
    def open(cls, path: Path) -> Vzg2Archive:
        """Validate and index a VZG2 archive without extracting it."""
        archive_path = Path(path).absolute()
        if archive_path.suffix.lower() != ".vzg2":
            raise ValueError(f"Expected a .vzg2 file, got: {archive_path}")
        if not archive_path.is_file():
            raise FileNotFoundError(f"VZG2 archive not found: {archive_path}")

        members: dict[str, Vzg2Member] = {}
        try:
            with tarfile.open(archive_path, mode="r:") as archive:
                for member in archive:
                    if member.isfile():
                        members[member.name] = Vzg2Member(
                            offset=int(member.offset_data),
                            size=int(member.size),
                        )
        except tarfile.TarError as exc:
            raise ValueError(f"Invalid VZG2 tar archive: {archive_path}") from exc

        if _MANIFEST_PATH not in members:
            raise ValueError(f"VZG2 archive has no {_MANIFEST_PATH!r}: {archive_path}")
        return cls(path=archive_path, members=members)

    def read_bytes(self, name: str) -> bytes:
        """Read one indexed archive member by byte range."""
        try:
            member = self.members[name]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Required VZG2 member {name!r} is missing from {self.path}"
            ) from exc
        with self.path.open("rb") as handle:
            handle.seek(member.offset)
            value = handle.read(member.size)
        if len(value) != member.size:
            raise OSError(
                f"Short read for VZG2 member {name!r}: "
                f"expected {member.size} bytes, got {len(value)}"
            )
        return value

    def read_json(self, name: str) -> dict[str, Any] | list[Any]:
        """Decode one JSON member."""
        return cast(dict[str, Any] | list[Any], json.loads(self.read_bytes(name)))

    def read_parquet(self, name: str) -> pa.Table:
        """Read one Parquet member without extracting the archive."""
        return pq.read_table(pa.BufferReader(self.read_bytes(name)))


class Vzg2Jpeg12Array:
    """Sliceable array backed by JPEG-12 OME-Zarr chunks inside a VZG2."""

    def __init__(
        self,
        *,
        archive_path: Path,
        member_prefix: str,
        shape: tuple[int, int, int],
        chunks: tuple[int, int, int],
        dtype: np.dtype[Any],
        fill_value: int | float,
        members: dict[tuple[int, int, int], Vzg2Member],
    ) -> None:
        self.archive_path = Path(archive_path)
        self.member_prefix = member_prefix
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        self.fill_value = fill_value
        self.members = members
        self.ndim = 3

    def __dask_tokenize__(self) -> tuple[Any, ...]:
        """Provide a compact stable token instead of hashing the chunk index."""
        stat = self.archive_path.stat()
        return (
            type(self).__name__,
            str(self.archive_path),
            stat.st_size,
            stat.st_mtime_ns,
            self.member_prefix,
            self.shape,
            self.chunks,
            self.dtype.str,
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        """Decode only the JPEG chunks intersecting a basic 3D slice."""
        slices = _normalize_basic_slices(key, self.shape)
        starts = tuple(value.start for value in slices)
        stops = tuple(value.stop for value in slices)
        output_shape = tuple(
            stop - start for start, stop in zip(starts, stops, strict=True)
        )
        output = np.full(output_shape, self.fill_value, dtype=self.dtype)
        if any(size == 0 for size in output_shape):
            return output

        chunk_ranges = [
            range(start // chunk, (stop - 1) // chunk + 1)
            for start, stop, chunk in zip(starts, stops, self.chunks, strict=True)
        ]
        for channel_chunk in chunk_ranges[0]:
            for y_chunk in chunk_ranges[1]:
                for x_chunk in chunk_ranges[2]:
                    member = self.members.get((channel_chunk, y_chunk, x_chunk))
                    if member is None:
                        continue
                    tile = self._decode_member(member)
                    _copy_chunk_intersection(
                        output=output,
                        tile=tile,
                        chunk_index=(channel_chunk, y_chunk, x_chunk),
                        chunks=self.chunks,
                        starts=starts,
                        stops=stops,
                    )
        return output

    def _decode_member(self, member: Vzg2Member) -> np.ndarray:
        with self.archive_path.open("rb") as handle:
            handle.seek(member.offset)
            encoded = handle.read(member.size)
        if len(encoded) != member.size:
            raise OSError(
                f"Short VZG2 image-chunk read at offset {member.offset} in "
                f"{self.archive_path}"
            )
        decoded = np.asarray(imagecodecs.jpeg_decode(encoded)).squeeze()
        if decoded.ndim != 2:
            raise ValueError(
                f"Expected a 2D JPEG-12 VZG2 chunk, got shape={decoded.shape}"
            )
        return decoded.astype(self.dtype, copy=False)[np.newaxis, :, :]


def resolve_vzg2_path(input_path: Path) -> Path | None:
    """Resolve a direct VZG2 path or an unambiguous VZG2 inside a folder."""
    path = Path(input_path)
    if path.is_file():
        return path if path.suffix.lower() == ".vzg2" else None
    if not path.is_dir():
        return None
    images_dir = path / "images"
    if images_dir.is_dir() and any(images_dir.glob("mosaic_*_z*.tif")):
        return None
    candidates = sorted(path.glob("*.vzg2"))
    if len(candidates) > 1:
        joined = ", ".join(candidate.name for candidate in candidates[:5])
        raise ValueError(
            f"Multiple .vzg2 files found in {path}; pass one archive directly. "
            f"Candidates: {joined}"
        )
    return candidates[0] if candidates else None


def write_vzg2_spatialdata(
    *,
    input_path: Path,
    output_path: Path,
    build_config: MerscopeBuildConfig,
    transform_path_override: Path | None = None,
) -> Path:
    """Build a MERSCOPE SpatialData zarr directly from a VZG2 archive."""
    archive_path = resolve_vzg2_path(input_path)
    if archive_path is None:
        raise FileNotFoundError(f"No .vzg2 archive found at: {input_path}")
    archive = Vzg2Archive.open(archive_path)
    transform_matrix = _load_transform_override(transform_path_override)
    source_dir = Path(input_path) if Path(input_path).is_dir() else archive_path.parent
    sdata = read_vzg2_spatialdata(
        archive,
        build_config=build_config,
        transform_matrix=transform_matrix,
        external_source_dir=source_dir,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_source_spatialdata_contract(sdata, platform="MERSCOPE")
    write_spatialdata_zarr(sdata, output_path, overwrite=True)

    matrix = cast(list[list[float]], sdata.attrs["merxen_vzg2"]["transform"])
    np.savetxt(
        output_path / "micron_to_mosaic_pixel_transform.csv",
        np.asarray(matrix, dtype=float),
    )
    return output_path


def read_vzg2_spatialdata(
    archive: Vzg2Archive,
    *,
    build_config: MerscopeBuildConfig,
    transform_matrix: np.ndarray | None = None,
    external_source_dir: Path | None = None,
) -> SpatialData:
    """Read VZG2 content and optional sibling transcript coordinates."""
    manifest_raw = archive.read_json(_MANIFEST_PATH)
    if not isinstance(manifest_raw, dict):
        raise ValueError("VZG2 manifest.json must contain a JSON object")
    manifest = manifest_raw
    matrix = (
        _manifest_transform(manifest)
        if transform_matrix is None
        else _validate_transform(transform_matrix)
    )

    region_name = build_config.region_name or str(
        manifest.get("name") or archive.path.stem
    )
    slide_name = build_config.slide_name or archive.path.parent.name
    dataset_id = f"{slide_name}_{region_name}"
    shape_key = f"{dataset_id}_polygons"

    image, image_details = _read_vzg2_image(
        archive,
        manifest=manifest,
        z_layers=build_config.z_layers,
    )
    feature_path = _resolve_feature_path(archive, manifest)
    shapes = _read_external_cell_shapes(
        external_source_dir,
        transform_matrix=matrix,
    )
    if shapes is None:
        shapes = _read_vzg2_shapes(
            archive,
            manifest=manifest,
            feature_path=feature_path,
            transform_matrix=matrix,
        )
    table = _read_vzg2_table(
        archive,
        feature_path=feature_path,
        region_name=region_name,
        slide_name=slide_name,
        dataset_id=dataset_id,
        shape_key=shape_key,
    )

    points, transcript_source = _read_external_transcript_points(
        external_source_dir,
        dataset_id=dataset_id,
        transform_matrix=matrix,
    )

    if transcript_source is None:
        logger.warning(
            "[MERSCOPE] %s contains packed transcript tiles, but Vizgen publishes no "
            "VZG2 transcript decoder. The SpatialData source will contain no "
            "transcript points; transcript-dependent segmentation requires the "
            "original detected_transcripts CSV/Parquet.",
            archive.path,
        )
    else:
        logger.info(
            "[MERSCOPE] Using transcript coordinates alongside VZG2 image: %s",
            transcript_source,
        )
    attrs = {
        "merxen_vzg2": {
            "source_archive": str(archive.path),
            "archive_manifest_version": str(manifest.get("version", "")),
            "transform": matrix.astype(float).tolist(),
            "image": image_details,
            "transcript_points_available": transcript_source is not None,
            "transcript_source": (
                str(transcript_source) if transcript_source is not None else None
            ),
            "transcript_limitation": (
                None
                if transcript_source is not None
                else (
                    "The VZG2 transcript tile codec has no public Vizgen decoder; "
                    "coordinates were not synthesized."
                )
            ),
        }
    }
    return SpatialData(
        images={MERSCOPE_ZPROJ_IMAGE_NAME: image},
        shapes={shape_key: shapes},
        points=points,
        tables={"table": table},
        attrs=attrs,
    )


def _read_external_transcript_points(
    source_dir: Path | None,
    *,
    dataset_id: str,
    transform_matrix: np.ndarray,
) -> tuple[dict[str, Any], Path | None]:
    """Load canonical transcript coordinates stored beside a VZG2 archive."""
    if source_dir is None:
        return {}, None

    from merxen.io.builders.merscope import (
        MerscopeKeys,
        _get_parquet_points,
        _get_points,
    )

    root = Path(source_dir)
    parquet_path = root / MerscopeKeys.TRANSCRIPTS_FILE_PARQUET
    csv_path = root / MerscopeKeys.TRANSCRIPTS_FILE_CSV
    transform: dict[str, BaseTransformation] = {
        "global": Affine(
            transform_matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
    }

    if parquet_path.is_file():
        points = _get_parquet_points(parquet_path, transform)
        return {f"{dataset_id}_transcripts": points}, parquet_path
    if csv_path.is_file():
        points = _get_points(csv_path, transform)
        return {f"{dataset_id}_transcripts": points}, csv_path
    return {}, None


def _read_external_cell_shapes(
    source_dir: Path | None,
    *,
    transform_matrix: np.ndarray,
) -> geopandas.GeoDataFrame | None:
    """Prefer lossless canonical boundaries when they accompany a VZG2."""
    if source_dir is None:
        return None

    from merxen.io.builders.merscope import MerscopeKeys, _get_polygons

    boundaries_path = Path(source_dir) / MerscopeKeys.BOUNDARIES_FILE
    if not boundaries_path.is_file():
        return None
    transform: dict[str, BaseTransformation] = {
        "global": Affine(
            transform_matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
    }
    logger.info(
        "[MERSCOPE] Using canonical cell boundaries alongside VZG2 image: %s",
        boundaries_path,
    )
    return _get_polygons(boundaries_path, transform)


def _read_vzg2_image(
    archive: Vzg2Archive,
    *,
    manifest: dict[str, Any],
    z_layers: list[int] | None,
) -> tuple[Any, dict[str, Any]]:
    images = manifest.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("VZG2 manifest does not describe any images")

    selected: list[tuple[str, dict[str, Any]]] = []
    allowed_layers = set(z_layers) if z_layers is not None else None
    for layer, value in sorted(images.items(), key=lambda item: _layer_sort(item[0])):
        if not isinstance(value, dict):
            continue
        if allowed_layers is not None:
            try:
                if int(layer) not in allowed_layers:
                    continue
            except ValueError:
                pass
        selected.append((str(layer), value))
    if not selected:
        raise ValueError(
            f"None of requested MERSCOPE z_layers={z_layers} are present in the "
            f"VZG2 image manifest (available={list(images)})."
        )

    arrays: list[da.Array] = []
    channels: list[str] | None = None
    selected_layers: list[str] = []
    for layer, image_spec in selected:
        image_path = str(image_spec.get("path", "")).strip("/")
        channel_values = [str(value) for value in image_spec.get("channels", [])]
        if not image_path or not channel_values:
            raise ValueError(f"Invalid VZG2 image manifest entry for layer {layer!r}")
        source = _open_vzg2_image_array(archive, image_path=image_path)
        if len(channel_values) != source.shape[0]:
            raise ValueError(
                f"VZG2 image channel mismatch for {image_path}: manifest has "
                f"{len(channel_values)}, array has {source.shape[0]}"
            )
        if channels is None:
            channels = channel_values
        elif channels != channel_values or arrays[0].shape != source.shape:
            raise ValueError("VZG2 image planes do not share shape and channel order")
        arrays.append(
            da.from_array(
                source,
                chunks=source.chunks,
                asarray=False,
                fancy=False,
                name=False,
            )
        )
        selected_layers.append(layer)

    assert channels is not None
    projected = arrays[0]
    for plane in arrays[1:]:
        projected = da.maximum(projected, plane)
    data_array = xarray.DataArray(
        projected,
        dims=("c", "y", "x"),
        coords={"c": channels},
    )
    min_dimension = min(int(projected.shape[1]), int(projected.shape[2]))
    pyramid_levels = min(4, max(0, int(math.log2(max(1, min_dimension)))))
    parse_kwargs: dict[str, Any] = {
        "chunks": (1, 4096, 4096),
    }
    if pyramid_levels:
        parse_kwargs["scale_factors"] = [2] * pyramid_levels
    image = Image2DModel.parse(
        data_array,
        c_coords=channels,
        rgb=None,
        **parse_kwargs,
    )
    details = {
        "source_z_layers": selected_layers,
        "channels": channels,
        "is_max_projection": len(arrays) > 1,
    }
    return image, details


def _open_vzg2_image_array(
    archive: Vzg2Archive,
    *,
    image_path: str,
) -> Vzg2Jpeg12Array:
    array_prefix = f"{image_path}/data.zarr/0/0"
    metadata_raw = archive.read_json(f"{array_prefix}/.zarray")
    if not isinstance(metadata_raw, dict):
        raise ValueError(f"Invalid .zarray metadata under {array_prefix}")
    metadata = metadata_raw
    shape_5d = tuple(int(value) for value in metadata["shape"])
    chunks_5d = tuple(int(value) for value in metadata["chunks"])
    if len(shape_5d) != 5 or len(chunks_5d) != 5 or shape_5d[:2] != (1, 1):
        raise ValueError(
            f"Unsupported VZG2 image array shape/chunks: {shape_5d}/{chunks_5d}"
        )
    compressor = metadata.get("compressor") or {}
    if compressor.get("id") != "jpeg12":
        raise ValueError(f"Unsupported VZG2 image compressor: {compressor.get('id')!r}")
    if metadata.get("dimension_separator", ".") != ".":
        raise ValueError("Only dot-separated VZG2 Zarr chunks are supported")

    members: dict[tuple[int, int, int], Vzg2Member] = {}
    member_prefix = f"{array_prefix}/0.0."
    for name, member in archive.members.items():
        if not name.startswith(member_prefix):
            continue
        indices = name[len(member_prefix) :].split(".")
        if len(indices) != 3 or not all(value.isdigit() for value in indices):
            continue
        chunk_index = cast(tuple[int, int, int], tuple(int(value) for value in indices))
        members[chunk_index] = member
    dtype = np.dtype(str(metadata["dtype"]))
    fill_value = metadata.get("fill_value")
    if fill_value is None:
        fill_value = 0
    return Vzg2Jpeg12Array(
        archive_path=archive.path,
        member_prefix=array_prefix,
        shape=cast(tuple[int, int, int], shape_5d[2:]),
        chunks=cast(tuple[int, int, int], chunks_5d[2:]),
        dtype=dtype,
        fill_value=fill_value,
        members=members,
    )


def _read_vzg2_shapes(
    archive: Vzg2Archive,
    *,
    manifest: dict[str, Any],
    feature_path: str,
    transform_matrix: np.ndarray,
) -> geopandas.GeoDataFrame:
    cells_manifest_raw = archive.read_json(f"{feature_path}/manifest_cells.json")
    cell_names_raw = archive.read_json(f"{feature_path}/cells_names_array.json")
    if not isinstance(cells_manifest_raw, dict) or not isinstance(cell_names_raw, list):
        raise ValueError("Invalid VZG2 cell manifest or cell-name array")
    grids = cells_manifest_raw.get("tiles")
    if not isinstance(grids, list) or not grids:
        raise ValueError("VZG2 cell manifest contains no tile grids")
    grid_x, grid_y = (int(value) for value in grids[0])
    width = int(manifest["mosaic_width_pixels"])
    height = int(manifest["mosaic_height_pixels"])

    coordinate_parts: list[np.ndarray] = []
    cell_ordinals: list[np.ndarray] = []
    for tile_number in range(grid_x * grid_y):
        name = f"{feature_path}/z-plane0/Lod0/tile{tile_number}.bin"
        tile_data = archive.read_bytes(name)
        coordinates, ordinals = _decode_vzg2_cell_tile(
            tile_data,
            tile_number=tile_number,
            grid=(grid_x, grid_y),
            image_shape=(height, width),
        )
        if len(ordinals):
            coordinate_parts.append(coordinates)
            cell_ordinals.append(ordinals)

    if not coordinate_parts:
        raise ValueError("VZG2 contains no LOD0 cell polygons in z-plane 0")
    pixel_coordinates = np.concatenate(coordinate_parts, axis=0).astype(
        np.float64, copy=False
    )
    ordinals = np.concatenate(cell_ordinals)
    if int(ordinals.max()) >= len(cell_names_raw):
        raise ValueError("VZG2 polygon cell map exceeds the cell-name array")

    inverse = np.linalg.inv(transform_matrix)
    pixel_x = pixel_coordinates[..., 0].copy()
    pixel_y = pixel_coordinates[..., 1].copy()
    pixel_coordinates[..., 0] = (
        pixel_x * inverse[0, 0] + pixel_y * inverse[0, 1] + inverse[0, 2]
    )
    pixel_coordinates[..., 1] = (
        pixel_x * inverse[1, 0] + pixel_y * inverse[1, 1] + inverse[1, 2]
    )
    geometries = shapely.polygons(pixel_coordinates)
    invalid = np.asarray(~shapely.is_valid(geometries))
    if bool(invalid.any()):
        logger.warning(
            "[MERSCOPE] Repairing %d self-intersecting VZG2 cell polygons",
            int(invalid.sum()),
        )
        geometries[invalid] = shapely.buffer(geometries[invalid], 0)
    valid = np.asarray(shapely.is_valid(geometries) & ~shapely.is_empty(geometries))
    if not bool(valid.all()):
        logger.warning(
            "[MERSCOPE] Dropping %d invalid VZG2 cell polygons",
            int((~valid).sum()),
        )
        geometries = geometries[valid]
        ordinals = ordinals[valid]

    source_ids = np.asarray(cell_names_raw, dtype=object)[ordinals].astype(str)
    frame = geopandas.GeoDataFrame(
        {"EntityID": source_ids, "geometry": geometries},
        geometry="geometry",
    )
    frame.index = pd.Index(source_ids, name="EntityID")
    transform = Affine(
        transform_matrix,
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    return ShapesModel.parse(frame, transformations={"global": transform})


def _decode_vzg2_cell_tile(
    tile_data: bytes,
    *,
    tile_number: int,
    grid: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one VPT-compatible LOD0 cell polygon tile."""
    # Encoding layout adapted from Vizgen's Apache-2.0 licensed
    # vpt-segmentation-packing 1.0.1 (Copyright 2024 Vizgen, Inc.).
    if len(tile_data) < 12:
        raise ValueError(f"Truncated VZG2 cell tile {tile_number}")
    count = struct.unpack_from("<I", tile_data, 8)[0]
    polygon_end = 12 + count * _CELL_POLYGON_BYTES
    map_end = polygon_end + count * 4
    if len(tile_data) < map_end:
        raise ValueError(
            f"Truncated VZG2 cell tile {tile_number}: expected >= {map_end} bytes, "
            f"got {len(tile_data)}"
        )
    if count == 0:
        return (
            np.empty((0, _CELL_POLYGON_VERTICES, 2), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )

    records = np.frombuffer(
        tile_data,
        dtype=np.dtype([("base_y", "<u2"), ("base_x", "<u2"), ("words", "<u4", 33)]),
        count=count,
        offset=12,
    )
    words = records["words"].reshape(count, 11, 3)
    packed = np.empty((count, 11, 4), dtype=np.uint32)
    packed[..., 0] = words[..., 0] >> 8
    packed[..., 1] = ((words[..., 0] & 0xFF) << 16) | (words[..., 1] >> 16)
    packed[..., 2] = ((words[..., 1] & 0xFFFF) << 8) | (words[..., 2] >> 24)
    packed[..., 3] = words[..., 2] & 0xFFFFFF
    packed = packed.reshape(count, _CELL_POLYGON_VERTICES)

    grid_x, grid_y = grid
    height, width = image_shape
    tile_x = tile_number % grid_x
    tile_y = tile_number // grid_x
    corner_x = int(width / grid_x * tile_x)
    corner_y = int(height / grid_y * tile_y)

    coordinates = np.empty((count, _CELL_POLYGON_VERTICES, 2), dtype=np.float32)
    coordinates[..., 0] = records["base_x"][:, np.newaxis] + (packed >> 12) + corner_x
    coordinates[..., 1] = records["base_y"][:, np.newaxis] + (packed & 0xFFF) + corner_y
    ordinals = np.frombuffer(
        tile_data,
        dtype="<u4",
        count=count,
        offset=polygon_end,
    ).copy()
    return coordinates, ordinals


def _read_vzg2_table(
    archive: Vzg2Archive,
    *,
    feature_path: str,
    region_name: str,
    slide_name: str,
    dataset_id: str,
    shape_key: str,
) -> anndata.AnnData:
    counts = archive.read_parquet(f"{feature_path}/cell_by_gene/cell_by_gene.parquet")
    cell_indices = archive.read_parquet(f"{feature_path}/cells_indices.parquet").column(
        0
    )
    centers = archive.read_parquet(f"{feature_path}/cells_centers.parquet")
    volumes = archive.read_parquet(f"{feature_path}/cells_volume.parquet")
    n_cells = len(cell_indices)
    if counts.num_rows != n_cells or centers.num_rows != n_cells:
        raise ValueError(
            "VZG2 cell table row mismatch: "
            f"counts={counts.num_rows}, centers={centers.num_rows}, cells={n_cells}"
        )

    feature_names = [
        name for name in counts.column_names if not name.startswith("__index_level_")
    ]
    gene_names = [name for name in feature_names if "blank" not in name.lower()]
    blank_names = [name for name in feature_names if "blank" in name.lower()]
    expression = _arrow_columns_to_sparse(counts, gene_names)
    blank_expression = _arrow_columns_to_sparse(counts, blank_names)

    ids = np.asarray(cell_indices.combine_chunks().to_numpy()).astype(str)
    center_frame = centers.to_pandas().reset_index(drop=True)
    volume_frame = volumes.to_pandas().reset_index(drop=True)
    obs = pd.concat([center_frame, volume_frame], axis=1)
    obs.index = pd.Index(ids)
    obs["EntityID"] = ids
    obs["region"] = pd.Categorical(np.repeat(region_name, n_cells))
    obs["slide"] = pd.Categorical(np.repeat(slide_name, n_cells))
    obs["dataset_id"] = pd.Categorical(np.repeat(dataset_id, n_cells))
    obs["cells_region"] = pd.Categorical(np.repeat(shape_key, n_cells))

    adata = anndata.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obsm["blank"] = blank_expression
    adata.uns["blank_gene_names"] = blank_names
    adata.obsm["spatial"] = obs[["center_x", "center_y"]].to_numpy(dtype=float)
    return TableModel.parse(
        adata,
        region_key="cells_region",
        region=shape_key,
        instance_key="EntityID",
    )


def _arrow_columns_to_sparse(
    table: pa.Table,
    columns: list[str],
) -> sparse.csr_matrix:
    n_rows = table.num_rows
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    indptr = [0]
    for name in columns:
        values = np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        nonzero = np.flatnonzero(values)
        row_parts.append(nonzero.astype(np.int32, copy=False))
        data_parts.append(values[nonzero])
        indptr.append(indptr[-1] + len(nonzero))
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int32)
    matrix = sparse.csc_matrix(
        (data, rows, np.asarray(indptr, dtype=np.int64)),
        shape=(n_rows, len(columns)),
        dtype=np.float32,
    )
    return matrix.tocsr()


def _resolve_feature_path(
    archive: Vzg2Archive,
    manifest: dict[str, Any],
) -> str:
    features = manifest.get("features")
    if not isinstance(features, list):
        raise ValueError("VZG2 manifest contains no feature sets")
    for feature in features:
        if not isinstance(feature, dict) or "path" not in feature:
            continue
        path = str(feature["path"]).strip("/")
        required = (
            f"{path}/manifest_cells.json",
            f"{path}/cell_by_gene/cell_by_gene.parquet",
        )
        if all(name in archive.members for name in required):
            return path
    raise ValueError("VZG2 contains no supported cell feature set")


def _manifest_transform(manifest: dict[str, Any]) -> np.ndarray:
    width = float(manifest["mosaic_width_pixels"])
    height = float(manifest["mosaic_height_pixels"])
    bbox = manifest.get("bbox_microns")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("VZG2 manifest bbox_microns must have four values")
    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid VZG2 micron bounding box: {bbox}")
    scale_x = width / (x_max - x_min)
    scale_y = height / (y_max - y_min)
    # Canonical transcript and boundary coordinates use the mosaic-local micron
    # origin. VZG2 bounding-box minima can include Vizualizer viewport padding;
    # treating them as an affine origin shifts images and masks away from points.
    return np.asarray(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_transform_override(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"MERSCOPE transform override not found: {source}")
    try:
        matrix = np.loadtxt(source)
    except ValueError:
        matrix = np.loadtxt(source, delimiter=",")
    return _validate_transform(matrix)


def _validate_transform(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"MERSCOPE transform must be a finite 3x3 matrix: {value}")
    if not np.allclose(value[2], [0.0, 0.0, 1.0]):
        raise ValueError("MERSCOPE transform must be a 2D homogeneous affine")
    if np.isclose(np.linalg.det(value), 0.0):
        raise ValueError("MERSCOPE transform is singular")
    return value


def _normalize_basic_slices(
    key: Any,
    shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    if not isinstance(key, tuple):
        key = (key,)
    expanded: list[Any] = []
    for value in key:
        if value is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - len(key) + 1))
        else:
            expanded.append(value)
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))
    if len(expanded) != len(shape) or not all(
        isinstance(value, slice) for value in expanded
    ):
        raise IndexError("VZG2 image arrays support basic 3D slices only")
    normalized = []
    for value, size in zip(expanded, shape, strict=True):
        start, stop, step = value.indices(size)
        if step != 1:
            raise IndexError("VZG2 image-array slices must have step=1")
        normalized.append(slice(start, stop, 1))
    return cast(tuple[slice, slice, slice], tuple(normalized))


def _copy_chunk_intersection(
    *,
    output: np.ndarray,
    tile: np.ndarray,
    chunk_index: tuple[int, int, int],
    chunks: tuple[int, int, int],
    starts: tuple[int, int, int],
    stops: tuple[int, int, int],
) -> None:
    chunk_starts = tuple(
        index * chunk for index, chunk in zip(chunk_index, chunks, strict=True)
    )
    global_starts = tuple(
        max(requested, chunk_start)
        for requested, chunk_start in zip(starts, chunk_starts, strict=True)
    )
    global_stops = tuple(
        min(requested, chunk_start + chunk)
        for requested, chunk_start, chunk in zip(
            stops, chunk_starts, chunks, strict=True
        )
    )
    source_slices = tuple(
        slice(global_start - chunk_start, global_stop - chunk_start)
        for global_start, global_stop, chunk_start in zip(
            global_starts, global_stops, chunk_starts, strict=True
        )
    )
    output_slices = tuple(
        slice(global_start - requested, global_stop - requested)
        for global_start, global_stop, requested in zip(
            global_starts, global_stops, starts, strict=True
        )
    )
    output[output_slices] = tile[source_slices]


def _layer_sort(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, str(value))
