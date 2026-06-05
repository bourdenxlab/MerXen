"""MERSCOPE raw-folder to SpatialData writer."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import anndata
import dask.dataframe as dd
import geopandas
import numpy as np
import pandas as pd
import xarray
import zarr
from dask import array as da
from dask_image.imread import imread
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation

from merxen.config import MerscopeBuildConfig
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    build_merscope_z_projection,
)
from merxen.io.spatialdata_io import write_spatialdata_zarr
from merxen.memory import force_release
from merxen.path_utils import remove_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectionBuildResult:
    """Projection image plus the temporary store backing its lazy array."""

    image: Any
    temp_path: Path | None = None


class MerscopeKeys:
    """Constants describing the expected MERSCOPE folder layout."""

    IMAGES_DIR = "images"
    TRANSFORMATION_FILE = "micron_to_mosaic_pixel_transform.csv"
    BOUNDARIES_FILE = "cell_boundaries.parquet"
    COUNTS_FILE = "cell_by_gene.csv"
    CELL_METADATA_FILE = "cell_metadata.csv"
    TRANSCRIPTS_FILE_CSV = "detected_transcripts.csv"
    TRANSCRIPTS_FILE_PARQUET = "detected_transcripts.parquet"
    METADATA_CELL_KEY = "EntityID"
    COUNTS_CELL_KEY = "cell"
    CELL_X = "center_x"
    CELL_Y = "center_y"
    GLOBAL_X = "global_x"
    GLOBAL_Y = "global_y"
    GLOBAL_Z = "global_z"
    Z_INDEX = "ZIndex"
    REGION_KEY = "cells_region"
    GENE_KEY = "gene"
    CELL_ID = "cell_id"


SUPPORTED_BACKENDS = ["dask_image", "rioxarray"]


def write_merscope_spatialdata(
    *,
    input_path: Path,
    output_path: Path,
    build_config: MerscopeBuildConfig,
    transform_path_override: Path | None = None,
) -> Path:
    """Build a MERSCOPE SpatialData zarr from a raw Vizgen output folder.

    Args:
        input_path: MERSCOPE folder containing transcripts, boundaries, counts,
            metadata, and the ``images/`` directory.
        output_path: Destination zarr path.
        build_config: Platform-specific reader options.
        transform_path_override: Optional explicit transform CSV to copy into the
            built zarr. When omitted, the transform is copied from the raw
            ``images/`` directory.

    Returns:
        The written zarr path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_layers = (
        list(build_config.z_layers)
        if build_config.z_layers is not None
        else discover_merscope_z_layers(input_path)
    )
    if len(z_layers) == 0:
        raise FileNotFoundError(
            "[MERSCOPE] No image z-layers found in"
            f" {input_path / MerscopeKeys.IMAGES_DIR}"
        )

    try:
        from spatialdata_io import merscope as merscope_reader
    except Exception:
        merscope_reader = None

    sdata = None
    if merscope_reader is not None:
        try:
            sdata = merscope_reader(
                input_path,
                z_layers=z_layers,
                region_name=build_config.region_name,
                slide_name=build_config.slide_name,
                mosaic_images=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MERSCOPE] spatialdata_io.merscope failed (%s)."
                " Falling back to local reader.",
                exc,
            )

    if sdata is None:
        sdata = read_merscope_spatialdata_local(
            path=input_path,
            z_layers=z_layers,
            region_name=build_config.region_name,
            slide_name=build_config.slide_name,
            mosaic_images=False,
        )

    for key in list(sdata.images.keys()):
        del sdata.images[key]
    projection = _load_merscope_projection_from_raw(
        input_path=input_path,
        z_layers=z_layers,
        temp_root=output_path.parent,
    )
    sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME] = projection.image
    try:
        write_spatialdata_zarr(sdata, output_path, overwrite=True)
        _copy_merscope_sidecars(
            input_path=input_path,
            output_path=output_path,
            transform_path_override=transform_path_override,
        )
    finally:
        if projection.temp_path is not None:
            remove_path(projection.temp_path)
        del sdata, projection
        force_release(note="[MERSCOPE] after SpatialData build write")
    return output_path


def _replace_merscope_images_with_projection(sdata: SpatialData) -> None:
    """Keep only the max-projected MERSCOPE image layer before writing."""
    if len(sdata.images) == 0:
        return
    projection = build_merscope_z_projection(sdata.images)
    for key in list(sdata.images.keys()):
        del sdata.images[key]
    sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME] = projection


def _load_merscope_projection_from_raw(
    *,
    input_path: Path,
    z_layers: list[int],
    temp_root: Path | None = None,
) -> ProjectionBuildResult:
    """Build the final MERSCOPE projection image without storing z planes."""
    images_dir = Path(input_path) / MerscopeKeys.IMAGES_DIR
    stainings = _get_channel_names(images_dir)
    if not stainings:
        raise FileNotFoundError(f"No MERSCOPE mosaic images found in {images_dir}")

    image_models_kwargs: dict[str, Any] = {
        "chunks": (1, 4096, 4096),
        "scale_factors": [2, 2, 2, 2],
    }
    projection = _build_iterative_projection_store(
        images_dir=images_dir,
        stainings=stainings,
        z_layers=z_layers,
        image_models_kwargs=image_models_kwargs,
        temp_root=temp_root,
    )
    image = Image2DModel.parse(
        projection.image,
        c_coords=stainings,
        rgb=None,
        **image_models_kwargs,
    )
    return ProjectionBuildResult(image=image, temp_path=projection.temp_path)


def _build_iterative_projection_store(
    *,
    images_dir: Path,
    stainings: list[str],
    z_layers: list[int],
    image_models_kwargs: Mapping[str, Any],
    temp_root: Path | None,
) -> ProjectionBuildResult:
    """Max-project raw MERSCOPE planes into a temporary zarr store."""
    if len(z_layers) == 0:
        raise ValueError("z_layers is empty")

    chunk_y, chunk_x = tuple(image_models_kwargs["chunks"])[-2:]
    first_plane = _open_merscope_channel_plane_array(
        images_dir=images_dir,
        stain=stainings[0],
        z_layer=z_layers[0],
        chunks=(int(chunk_y), int(chunk_x)),
    )
    height, width = (int(first_plane.shape[0]), int(first_plane.shape[1]))
    dtype = np.dtype(first_plane.dtype)
    del first_plane
    store_chunks = (
        1,
        min(int(chunk_y), height),
        min(int(chunk_x), width),
    )

    temp_parent = Path(temp_root) if temp_root is not None else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="merscope_projection_", dir=temp_parent))
    projection_path = temp_dir / "projection.zarr"
    projection_store = zarr.open_array(
        projection_path,
        mode="w",
        shape=(len(stainings), height, width),
        chunks=store_chunks,
        dtype=dtype,
    )

    try:
        for channel_index, stain in enumerate(stainings):
            logger.info(
                "[MERSCOPE] Iteratively max-projecting channel %s (%d/%d)",
                stain,
                channel_index + 1,
                len(stainings),
            )
            current = _open_merscope_channel_plane_array(
                images_dir=images_dir,
                stain=stain,
                z_layer=z_layers[0],
                chunks=(int(chunk_y), int(chunk_x)),
            )
            _store_projection_channel(
                current,
                projection_store,
                channel_index=channel_index,
                shape=(height, width),
            )
            del current

            for z_layer in z_layers[1:]:
                previous = da.from_zarr(projection_path)[channel_index, :, :]
                plane = _open_merscope_channel_plane_array(
                    images_dir=images_dir,
                    stain=stain,
                    z_layer=z_layer,
                    chunks=(int(chunk_y), int(chunk_x)),
                )
                projected = da.maximum(previous, plane)
                _store_projection_channel(
                    projected,
                    projection_store,
                    channel_index=channel_index,
                    shape=(height, width),
                )
                del previous, plane, projected
                force_release(
                    note=(
                        f"[MERSCOPE] after projecting channel={stain} " f"z={z_layer}"
                    )
                )

        projection_da = xarray.DataArray(
            da.from_zarr(projection_path),
            dims=("c", "y", "x"),
            coords={"c": stainings},
        )
        return ProjectionBuildResult(image=projection_da, temp_path=temp_dir)
    except Exception:
        remove_path(temp_dir)
        raise


def _store_projection_channel(
    source: da.Array,
    projection_store: Any,
    *,
    channel_index: int,
    shape: tuple[int, int],
) -> None:
    """Store one projected channel into the projection zarr."""
    source_3d = source.reshape((1, int(shape[0]), int(shape[1])))
    da.store(
        source_3d,
        projection_store,
        regions=(slice(channel_index, channel_index + 1), slice(None), slice(None)),
        compute=True,
        lock=True,
        scheduler="single-threaded",
    )


def _open_merscope_channel_plane_array(
    *,
    images_dir: Path,
    stain: str,
    z_layer: int,
    chunks: tuple[int, int],
) -> da.Array:
    """Open one MERSCOPE channel/z plane as a 2D Dask array."""
    tif_path = Path(images_dir) / f"mosaic_{stain}_z{z_layer}.tif"
    if not tif_path.exists():
        raise FileNotFoundError(f"Missing MERSCOPE image plane: {tif_path}")

    try:
        import rioxarray
    except ModuleNotFoundError:
        arr = imread(tif_path).squeeze()
        return da.asarray(arr).rechunk(chunks)

    from rasterio.errors import NotGeoreferencedWarning

    warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
    data = (
        rioxarray.open_rasterio(
            tif_path,
            chunks=(1, int(chunks[0]), int(chunks[1])),
        )
        .reset_coords("spatial_ref", drop=True)
        .data
    )
    arr = da.asarray(data).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D MERSCOPE plane, got shape={arr.shape}")
    return arr.rechunk(chunks)


def discover_merscope_z_layers(path: Path) -> list[int]:
    """Discover all available z-layer indices from the raw MERSCOPE images."""
    images_dir = Path(path) / MerscopeKeys.IMAGES_DIR
    if not images_dir.exists():
        return []
    pattern = re.compile(r"^mosaic_[\w|-]+[0-9]?_z(?P<z>[0-9]+)\.tif$")
    values = set()
    for tif_path in images_dir.iterdir():
        match = pattern.match(tif_path.name)
        if match is not None:
            values.add(int(match.group("z")))
    return sorted(values)


def read_merscope_spatialdata_local(
    *,
    path: Path,
    z_layers: list[int],
    region_name: str | None,
    slide_name: str | None,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    cells_boundaries: bool = True,
    cells_table: bool = True,
    mosaic_images: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read MERSCOPE raw files into a SpatialData object.

    This is a lightweight local fallback used when ``spatialdata-io`` does not
    expose a MERSCOPE reader in the current environment.
    """
    if "chunks" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["chunks"] = (1, 4096, 4096)
    if "scale_factors" not in image_models_kwargs:
        if isinstance(image_models_kwargs, MappingProxyType):
            image_models_kwargs = {}
        assert isinstance(image_models_kwargs, dict)
        image_models_kwargs["scale_factors"] = [2, 2, 2, 2]

    if backend is not None and backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' not supported. Expected one of: {SUPPORTED_BACKENDS}"
        )

    path = Path(path).absolute()
    images_dir = path / MerscopeKeys.IMAGES_DIR
    count_path = path / MerscopeKeys.COUNTS_FILE
    obs_path = path / MerscopeKeys.CELL_METADATA_FILE
    boundaries_path = path / MerscopeKeys.BOUNDARIES_FILE

    microns_to_pixels = Affine(
        np.genfromtxt(images_dir / MerscopeKeys.TRANSFORMATION_FILE),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    transform = {"global": microns_to_pixels}

    vizgen_region = path.name if region_name is None else region_name
    slide = path.parent.name if slide_name is None else slide_name
    dataset_id = f"{slide}_{vizgen_region}"
    region = f"{dataset_id}_polygons"

    images: dict[str, Any] = {}
    if mosaic_images:
        reader = _get_reader(backend)
        stainings = _get_channel_names(images_dir)
        for z_layer in z_layers:
            images[f"{dataset_id}_z{z_layer}"] = reader(
                images_dir,
                stainings,
                z_layer,
                image_models_kwargs,
                **imread_kwargs,
            )

    points: dict[str, Any] = {}
    if transcripts:
        transcript_path_parquet = path / MerscopeKeys.TRANSCRIPTS_FILE_PARQUET
        transcript_path_csv = path / MerscopeKeys.TRANSCRIPTS_FILE_CSV
        if transcript_path_parquet.exists():
            table = pd.read_parquet(transcript_path_parquet, engine="pyarrow")
            points[f"{dataset_id}_transcripts"] = _parse_transcript_table(
                table,
                transform,
            )
        elif transcript_path_csv.exists():
            points[f"{dataset_id}_transcripts"] = _get_points(
                transcript_path_csv,
                transform,
            )

    shapes: dict[str, Any] = {}
    if cells_boundaries and boundaries_path.exists():
        shapes[f"{dataset_id}_polygons"] = _get_polygons(boundaries_path, transform)

    tables: dict[str, Any] = {}
    if cells_table and count_path.exists() and obs_path.exists():
        tables["table"] = _get_table(
            count_path,
            obs_path,
            vizgen_region,
            slide,
            dataset_id,
            region,
        )

    return SpatialData(shapes=shapes, points=points, images=images, tables=tables)


def _copy_merscope_sidecars(
    *,
    input_path: Path,
    output_path: Path,
    transform_path_override: Path | None,
) -> None:
    """Copy transform metadata into the built zarr for downstream inference."""
    if transform_path_override is not None:
        source_path = Path(transform_path_override)
    else:
        source_path = (
            Path(input_path)
            / MerscopeKeys.IMAGES_DIR
            / MerscopeKeys.TRANSFORMATION_FILE
        )
    if not source_path.exists():
        return
    dest_path = output_path / MerscopeKeys.TRANSFORMATION_FILE
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)


def _get_channel_names(images_dir: Path) -> list[str]:
    """Return all stain/channel names encoded in MERSCOPE mosaic filenames."""
    exp = r"mosaic_(?P<stain>[\w|-]+[0-9]?)_z(?P<z>[0-9]+).tif"
    matches = [re.search(exp, file.name) for file in images_dir.iterdir()]
    stainings = {match.group("stain") for match in matches if match}
    return list(stainings)


def _get_reader(backend: str | None) -> Callable[..., Image2DModel]:
    """Choose the preferred raster backend for reading MERSCOPE planes."""
    if backend is not None:
        if backend == "rioxarray":
            return _rioxarray_load_merscope
        return _dask_image_load_merscope
    try:
        import rioxarray  # noqa: F401

        return _rioxarray_load_merscope
    except ModuleNotFoundError:
        return _dask_image_load_merscope


def _parse_transcript_table(
    table: pd.DataFrame,
    transformations: dict[str, BaseTransformation],
) -> dd.DataFrame:
    """Normalize transcript types and parse them as a SpatialData points model."""
    if MerscopeKeys.GENE_KEY not in table.columns and "feature_name" in table.columns:
        table = table.rename(columns={"feature_name": MerscopeKeys.GENE_KEY})

    required = [
        MerscopeKeys.GLOBAL_X,
        MerscopeKeys.GLOBAL_Y,
        MerscopeKeys.GLOBAL_Z,
        MerscopeKeys.GENE_KEY,
    ]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Transcript table is missing required columns: {missing}")

    for col in (
        MerscopeKeys.GLOBAL_X,
        MerscopeKeys.GLOBAL_Y,
        MerscopeKeys.GLOBAL_Z,
    ):
        table[col] = pd.to_numeric(table[col], errors="coerce").astype("float64")

    table[MerscopeKeys.GENE_KEY] = table[MerscopeKeys.GENE_KEY].astype(str)
    table = table.dropna(subset=required)

    if MerscopeKeys.CELL_ID in table.columns:
        table[MerscopeKeys.CELL_ID] = table[MerscopeKeys.CELL_ID].astype(str)

    npartitions = max(1, min(16, len(table) // 1_000_000 + 1))
    ddf = dd.from_pandas(table, npartitions=npartitions)

    parse_kwargs: dict[str, Any] = {
        "coordinates": {
            "x": MerscopeKeys.GLOBAL_X,
            "y": MerscopeKeys.GLOBAL_Y,
            "z": MerscopeKeys.GLOBAL_Z,
        },
        "feature_key": MerscopeKeys.GENE_KEY,
        "transformations": transformations,
        "sort": True,
    }
    if MerscopeKeys.CELL_ID in ddf.columns:
        parse_kwargs["instance_key"] = MerscopeKeys.CELL_ID

    transcripts = PointsModel.parse(ddf, **parse_kwargs)
    transcripts["gene"] = transcripts["gene"].astype("category")
    return cast(dd.DataFrame, transcripts)


def _rioxarray_load_merscope(
    images_dir: Path,
    stainings: list[str],
    z_layer: int,
    image_models_kwargs: Mapping[str, Any],
    **kwargs: Any,
) -> Image2DModel:
    """Read one MERSCOPE z-plane using rioxarray."""
    try:
        import rioxarray
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Using rioxarray backend requires the rioxarray package."
        ) from exc
    from rasterio.errors import NotGeoreferencedWarning

    warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
    im = xarray.concat(
        [
            rioxarray.open_rasterio(
                images_dir / f"mosaic_{stain}_z{z_layer}.tif",
                chunks=image_models_kwargs["chunks"],
                **kwargs,
            )
            .rename({"band": "c"})
            .reset_coords("spatial_ref", drop=True)
            for stain in stainings
        ],
        dim="c",
    )
    return Image2DModel.parse(im, c_coords=stainings, rgb=None, **image_models_kwargs)


def _dask_image_load_merscope(
    images_dir: Path,
    stainings: list[str],
    z_layer: int,
    image_models_kwargs: Mapping[str, Any],
    **kwargs: Any,
) -> Image2DModel:
    """Read one MERSCOPE z-plane using dask-image."""
    im = da.stack(
        [
            imread(images_dir / f"mosaic_{stain}_z{z_layer}.tif", **kwargs).squeeze()
            for stain in stainings
        ],
        axis=0,
    )
    return Image2DModel.parse(
        im,
        dims=("c", "y", "x"),
        c_coords=stainings,
        rgb=None,
        **image_models_kwargs,
    )


def _get_points(
    transcript_path: Path,
    transformations: dict[str, BaseTransformation],
) -> dd.DataFrame:
    """Read the transcript CSV and parse it into a points element."""
    transcript_df = pd.read_csv(transcript_path)
    return _parse_transcript_table(transcript_df, transformations)


def _get_polygons(
    boundaries_path: Path,
    transformations: dict[str, BaseTransformation],
) -> geopandas.GeoDataFrame:
    """Read the cell boundary parquet file and parse it into shapes."""
    geo_df = geopandas.read_parquet(boundaries_path)
    geo_df = geo_df.rename_geometry("geometry")
    geo_df = geo_df[geo_df[MerscopeKeys.Z_INDEX] == 0]
    geo_df = geo_df[geo_df.geometry.is_valid]
    geo_df.geometry = geo_df.geometry.map(_to_multi_polygon)
    geo_df.index = geo_df[MerscopeKeys.METADATA_CELL_KEY].astype(str)
    return ShapesModel.parse(geo_df, transformations=transformations)


def _to_multi_polygon(geometry: Polygon | MultiPolygon) -> MultiPolygon:
    """Normalize a shapely polygon geometry to MultiPolygon."""
    if isinstance(geometry, MultiPolygon):
        return geometry
    return MultiPolygon([geometry])


def _get_table(
    count_path: Path,
    obs_path: Path,
    vizgen_region: str,
    slide_name: str,
    dataset_id: str,
    region: str,
) -> anndata.AnnData:
    """Build a cell-by-gene AnnData table from raw MERSCOPE CSV files."""
    data = pd.read_csv(
        count_path, index_col=0, dtype={MerscopeKeys.COUNTS_CELL_KEY: str}
    )
    obs = pd.read_csv(
        obs_path, index_col=0, dtype={MerscopeKeys.METADATA_CELL_KEY: str}
    )

    is_gene = ~data.columns.str.lower().str.contains("blank")
    adata = anndata.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)
    adata.obsm["blank"] = data.loc[:, ~is_gene]
    adata.obsm["spatial"] = adata.obs[[MerscopeKeys.CELL_X, MerscopeKeys.CELL_Y]].values
    adata.obs["region"] = pd.Series(
        vizgen_region, index=adata.obs_names, dtype="category"
    )
    adata.obs["slide"] = pd.Series(slide_name, index=adata.obs_names, dtype="category")
    adata.obs["dataset_id"] = pd.Series(
        dataset_id, index=adata.obs_names, dtype="category"
    )
    adata.obs[MerscopeKeys.REGION_KEY] = pd.Series(
        region, index=adata.obs_names, dtype="category"
    )
    adata.obs[MerscopeKeys.METADATA_CELL_KEY] = adata.obs.index

    return TableModel.parse(
        adata,
        region_key=MerscopeKeys.REGION_KEY,
        region=region,
        instance_key=MerscopeKeys.METADATA_CELL_KEY,
    )
