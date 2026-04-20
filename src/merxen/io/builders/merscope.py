"""MERSCOPE raw-folder to SpatialData writer."""

from __future__ import annotations

import logging
import re
import shutil
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import anndata
import dask.dataframe as dd
import geopandas
import numpy as np
import pandas as pd
import xarray
from dask import array as da
from dask_image.imread import imread
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, BaseTransformation

from merxen.config import MerscopeBuildConfig
from merxen.io.spatialdata_io import write_spatialdata_zarr

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


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
            f"[MERSCOPE] No image z-layers found in {input_path / MerscopeKeys.IMAGES_DIR}"
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
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MERSCOPE] spatialdata_io.merscope failed (%s). Falling back to local reader.",
                exc,
            )

    if sdata is None:
        sdata = read_merscope_spatialdata_local(
            path=input_path,
            z_layers=z_layers,
            region_name=build_config.region_name,
            slide_name=build_config.slide_name,
        )

    write_spatialdata_zarr(sdata, output_path, overwrite=True)
    _copy_merscope_sidecars(
        input_path=input_path,
        output_path=output_path,
        transform_path_override=transform_path_override,
    )
    return output_path


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
        return _rioxarray_load_merscope if backend == "rioxarray" else _dask_image_load_merscope
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
    return transcripts


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
    data = pd.read_csv(count_path, index_col=0, dtype={MerscopeKeys.COUNTS_CELL_KEY: str})
    obs = pd.read_csv(obs_path, index_col=0, dtype={MerscopeKeys.METADATA_CELL_KEY: str})

    is_gene = ~data.columns.str.lower().str.contains("blank")
    adata = anndata.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)
    adata.obsm["blank"] = data.loc[:, ~is_gene]
    adata.obsm["spatial"] = adata.obs[[MerscopeKeys.CELL_X, MerscopeKeys.CELL_Y]].values
    adata.obs["region"] = pd.Series(vizgen_region, index=adata.obs_names, dtype="category")
    adata.obs["slide"] = pd.Series(slide_name, index=adata.obs_names, dtype="category")
    adata.obs["dataset_id"] = pd.Series(dataset_id, index=adata.obs_names, dtype="category")
    adata.obs[MerscopeKeys.REGION_KEY] = pd.Series(region, index=adata.obs_names, dtype="category")
    adata.obs[MerscopeKeys.METADATA_CELL_KEY] = adata.obs.index

    return TableModel.parse(
        adata,
        region_key=MerscopeKeys.REGION_KEY,
        region=region,
        instance_key=MerscopeKeys.METADATA_CELL_KEY,
    )
