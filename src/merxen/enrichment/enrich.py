"""SpatialData enrichment utilities."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse as sps
from scipy.io import mmread
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel
from spatialdata_io import xenium as xenium_reader
from tqdm.auto import tqdm

from merxen.alignment.manifest import invalidate_alignment_materialization
from merxen.config import EnrichmentConfig
from merxen.enrichment.assignment import clone_table_for_region
from merxen.enrichment.label_lookup import (
    NATIVE_COUNT_COLUMNS,
    native_label_check,
    natural_partition_order,
)
from merxen.io.bead_removal.provenance import (
    IMAGE_PREPROCESSING_ATTR,
    check_store_segmentation_sidecar,
    read_image_preprocessing_attr,
)
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    build_merscope_z_projection,
    image_to_cyx,
)
from merxen.io.points_aggregates import points_parquet_dir, read_root_attrs
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    MERXEN_SCHEMA_ATTR,
    SOURCE_CELL_ID_COLUMN,
    choose_primary_points_key,
)
from merxen.masks import register_missing_mask_branches
from merxen.memory import force_release, log_status
from merxen.path_utils import remove_path, stage_existing_output
from merxen.platforms import (
    PlatformSpec,
    find_platform_spec,
    find_um_per_px,
    platform_spec,
    spec_file_pixel_size,
)
from merxen.segmentation import native as native_segmentation
from merxen.segmentation.cellpose import build_cellpose_affine_to_microns
from merxen.segmentation.mask_geometry import masks_to_labeled_polygons
from merxen.segmentation.model_identity import (
    find_model_identity_path,
    read_model_identity,
)

logger = logging.getLogger(__name__)

MOSAIK_PROSEG_SHAPE_NAME = "MOSAIK_proseg"
MOSAIK_CELLPOSE_SHAPE_NAME = "MOSAIK_cellpose"
CELLPOSE_NUCLEI_SHAPE_NAME = "cellpose_nuclei"
MERSCOPE_OLD_SHAPE_NAME = platform_spec("MERSCOPE").original_cell_shape_key
XENIUM_OLD_CELL_SHAPE_NAME, XENIUM_OLD_NUCLEUS_SHAPE_NAME = platform_spec(
    "XENIUM"
).original_shape_keys
ORIGINAL_TABLE_NAME = "table_original"
NATIVE_ENRICHMENT_ATTR = "merxen_native_enrichment"
"""Root attr recording a completed native (vendor-segmentation) enrichment."""
NATIVE_ENRICHMENT_VERSION = 1
"""Bumped when the native enrichment's outputs change."""
NATIVE_PROVENANCE_KEY = "native_segmentation"
"""``merxen_schema`` key holding the vendor segmentation provenance."""


def _remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory tree if it exists."""
    remove_path(path)


def _to_cyx(image_like: Any) -> Any:
    """Convert image-like input to (c, y, x) ordering."""
    return image_to_cyx(image_like)


def _resolve_enrichment_write_path(latest_path: Path, target_path: Path) -> Path:
    """Choose the backed zarr path that enrichment should update in place."""
    if target_path == latest_path:
        return latest_path
    if target_path.exists() or target_path.is_symlink():
        return target_path
    return latest_path


def _parse_shapes_with_template(
    gdf: gpd.GeoDataFrame,
    template_shape: Any | None = None,
) -> Any:
    """Parse a GeoDataFrame to a ShapesModel, inheriting transforms when possible."""
    transformations = None
    if template_shape is not None and hasattr(template_shape, "attrs"):
        transformations = template_shape.attrs.get("transform", None)

    if transformations is not None:
        try:
            return ShapesModel.parse(gdf, transformations=transformations)
        except TypeError:
            pass
    return ShapesModel.parse(gdf)


def _prepare_original_table(adata: ad.AnnData, target_region_name: str) -> ad.AnnData:
    """Clone and normalize a platform-original AnnData table for enrichment output."""
    return clone_table_for_region(adata, target_region_name)


def _load_xenium_original_table_from_matrix(xenium_dir: Path) -> ad.AnnData | None:
    """Load Xenium matrix files when an original table is otherwise unavailable."""
    matrix_dir = xenium_dir / "cell_feature_matrix"
    mtx_path = matrix_dir / "matrix.mtx.gz"
    features_path = matrix_dir / "features.tsv.gz"
    barcodes_path = matrix_dir / "barcodes.tsv.gz"
    if not (mtx_path.exists() and features_path.exists() and barcodes_path.exists()):
        return None

    barcodes = (
        pd.read_csv(barcodes_path, sep="\t", header=None, compression="gzip")
        .iloc[:, 0]
        .astype(str)
        .to_numpy()
    )
    feats = pd.read_csv(features_path, sep="\t", header=None, compression="gzip")
    feature_ids = feats.iloc[:, 0].astype(str).to_numpy()
    gene_names = (
        feats.iloc[:, 1].astype(str).to_numpy() if feats.shape[1] > 1 else feature_ids
    )

    x_matrix = mmread(str(mtx_path))
    if not sps.issparse(x_matrix):
        x_matrix = sps.csr_matrix(x_matrix)
    x_matrix = x_matrix.tocsr()

    # Xenium matrix usually arrives as genes x cells.
    if x_matrix.shape[1] == len(barcodes):
        x_matrix = x_matrix.T.tocsr()
    if x_matrix.shape[0] != len(barcodes):
        n = min(x_matrix.shape[0], len(barcodes))
        x_matrix = x_matrix[:n, :]
        barcodes = barcodes[:n]
    if x_matrix.shape[1] != len(gene_names):
        n = min(x_matrix.shape[1], len(gene_names))
        x_matrix = x_matrix[:, :n]
        gene_names = gene_names[:n]
        feature_ids = feature_ids[:n]

    adata = ad.AnnData(X=x_matrix)
    adata.obs_names = pd.Index(barcodes.astype(str), name="cell_id")
    adata.var_names = pd.Index(
        pd.Series(gene_names).astype(str).to_numpy(),
        name="gene",
    )
    adata.var_names_make_unique()
    adata.var["gene"] = adata.var_names.astype(str)
    if len(feature_ids) >= adata.n_vars:
        adata.var["feature_id"] = np.array(feature_ids[: adata.n_vars], dtype=object)
    if feats.shape[1] > 2 and len(feats) >= adata.n_vars:
        adata.var["feature_type"] = feats.iloc[: adata.n_vars, 2].astype(str).to_numpy()
    return adata


def _load_xenium_boundary_shapes_from_csv(
    xenium_dir: Path,
    which: str = "cell",
) -> gpd.GeoDataFrame:
    """Load Xenium cell or nucleus boundary polygons from compressed CSV."""
    which_norm = str(which).lower()
    if which_norm.startswith("nuc"):
        csv_path = xenium_dir / "nucleus_boundaries.csv.gz"
        desc = "nucleus"
    else:
        csv_path = xenium_dir / "cell_boundaries.csv.gz"
        desc = "cell"

    if not csv_path.exists():
        log_status(f"[XENIUM] Missing {desc} boundary file: {csv_path}")
        return gpd.GeoDataFrame({"cell_id": [], "geometry": []}, geometry="geometry")

    log_status(f"[XENIUM] Loading {desc} boundaries from {csv_path.name}")
    df = pd.read_csv(
        csv_path,
        compression="gzip",
        usecols=["cell_id", "vertex_x", "vertex_y", "label_id"],
    )
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df["vertex_x"] = pd.to_numeric(df["vertex_x"], errors="coerce")
    df["vertex_y"] = pd.to_numeric(df["vertex_y"], errors="coerce")
    df = df.dropna(subset=["label_id", "vertex_x", "vertex_y"])

    polygons: list[Polygon] = []
    cell_ids: list[str] = []
    grouped = df.groupby("label_id", sort=False)
    for _, grp in tqdm(
        grouped,
        total=grouped.ngroups,
        desc=f"[XENIUM] {desc} polygons",
    ):
        xy = grp[["vertex_x", "vertex_y"]].to_numpy(dtype=float)
        if xy.shape[0] < 3:
            continue
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly is None or poly.is_empty:
            continue
        polygons.append(poly)
        cell_ids.append(str(grp["cell_id"].iloc[0]))

    gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "geometry": polygons},
        geometry="geometry",
    )
    log_status(f"[XENIUM] {desc} boundaries loaded: {len(gdf):,} polygons")
    return gdf


def _cellpose_gdf_from_mask(
    mask_path: Path,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    dataset_name: str,
    *,
    polygon_n_jobs: int,
    polygon_show_progress: bool,
    shape_name: str = MOSAIK_CELLPOSE_SHAPE_NAME,
    id_prefix: str = "cellpose",
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of Cellpose polygons in micron space."""
    if not mask_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Missing Cellpose mask file: {mask_path}"
        )

    log_status(f"[{dataset_name}] Building {shape_name} from {mask_path}")
    mask_mmap = np.load(mask_path, mmap_mode="r")
    labeled_polygons_px = masks_to_labeled_polygons(
        mask_mmap,
        factor_rescale=0,
        n_jobs=int(polygon_n_jobs),
        show_progress=bool(polygon_show_progress),
    )

    coeffs = [
        float(x_transform[0]),
        float(x_transform[1]),
        float(y_transform[0]),
        float(y_transform[1]),
        float(x_transform[2]),
        float(y_transform[2]),
    ]

    labeled_polygons_um: list[tuple[int, Polygon]] = []
    for label_id, poly in tqdm(
        labeled_polygons_px,
        desc=f"[{dataset_name}] cellpose polygons->microns",
        unit="poly",
    ):
        if poly is None or poly.is_empty:
            continue
        transformed = affine_transform(poly, coeffs)
        if transformed is not None and not transformed.is_empty:
            labeled_polygons_um.append((int(label_id), transformed))

    del mask_mmap, labeled_polygons_px
    force_release(note=f"after {dataset_name} cellpose polygon conversion")

    if len(labeled_polygons_um) == 0:
        raise RuntimeError(
            f"[{dataset_name}] No valid polygons created from Cellpose masks."
        )

    labels = np.asarray(
        [label_id for label_id, _ in labeled_polygons_um],
        dtype=np.uint64,
    )
    gdf = gpd.GeoDataFrame(
        {
            INSTANCE_ID_COLUMN: labels,
            SOURCE_CELL_ID_COLUMN: [f"{id_prefix}_{label_id}" for label_id in labels],
            "geometry": [polygon for _, polygon in labeled_polygons_um],
        },
        geometry="geometry",
    )
    gdf.index = pd.Index(
        labels,
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    return gdf


def _load_merscope_transform(config: EnrichmentConfig | _TransformInputs) -> np.ndarray:
    """Load MERSCOPE transform matrix from config."""
    candidates: list[Path] = []
    if config.transform_path is not None:
        candidates.append(Path(config.transform_path))
    candidates.append(
        Path(config.original_data_path) / "micron_to_mosaic_pixel_transform.csv"
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        matrix = np.loadtxt(candidate)
        if matrix.shape == (3, 3):
            return matrix

    raise FileNotFoundError(
        "Could not determine MERSCOPE transform from transform_path/original_data_path."
    )


def _find_xenium_pixel_size(config: EnrichmentConfig | _TransformInputs) -> float:
    """Find Xenium micron-per-pixel value from config transform sources.

    An explicit ``transform_path`` (a micron->pixel matrix or a spec file) comes
    first; then the platform's spec sidecars (``find_um_per_px``).
    """
    return _find_10x_pixel_size(config, "XENIUM")


def _find_10x_pixel_size(
    config: EnrichmentConfig | _TransformInputs, platform: str
) -> float:
    """Find the micron-per-pixel value of a 10x-family platform.

    An explicit ``transform_path`` (a micron->pixel matrix or a spec file) comes
    first; then the platform's spec sidecars under ``original_data_path``
    (``find_um_per_px``: ``experiment.xenium`` for XENIUM, the
    ``atera/experiment*.spatial`` sidecars of a built store for ATERA).
    """
    if config.transform_path is not None:
        candidate = Path(config.transform_path)
        if candidate.exists():
            if candidate.suffix.lower() in {".txt", ".csv"}:
                matrix = np.loadtxt(candidate)
                if matrix.shape == (3, 3):
                    # candidate stores micron->pixel, derive mpp from inverse scale.
                    sx = float(matrix[0, 0])
                    if sx != 0:
                        return 1.0 / sx
            pixel_size = spec_file_pixel_size(candidate)
            if pixel_size is not None:
                return pixel_size
    try:
        return find_um_per_px(Path(config.original_data_path), platform)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not determine {platform.title()} pixel_size from "
            "transform_path/original_data_path."
        ) from None


@dataclass(frozen=True)
class _TransformInputs:
    """Minimal fields the pixel->micron transform loaders read off a config.

    Lets :func:`resolve_px_to_um_transform` reuse the merscope/xenium loaders
    below (which only touch these three attributes) without constructing a full
    :class:`EnrichmentConfig`.
    """

    platform: str
    transform_path: Path | None
    original_data_path: Path


def resolve_px_to_um_transform(
    platform: str,
    original_data_path: Path,
    transform_path: Path | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the image pixel->micron affine ``(x_transform, y_transform)``.

    Each tuple is ``(a, b, offset)`` such that ``x_um = a*x_px + b*y_px + offset``
    (and likewise for y). This is the transform that maps morphology-image pixels
    (and thus Cellpose mask pixels) into the micron coordinate system the shapes
    live in. Shared by enrichment and the viewer-cache stage.
    """
    return _dataset_cellpose_transform(
        _TransformInputs(
            platform=str(platform),
            transform_path=transform_path,
            original_data_path=Path(original_data_path),
        )
    )


def _dataset_cellpose_transform(
    config: EnrichmentConfig | _TransformInputs,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build Cellpose pixel->micron affine tuples for enrichment."""
    platform = config.platform.upper()
    if platform == "MERSCOPE":
        matrix = _load_merscope_transform(config)
        return build_cellpose_affine_to_microns(
            matrix,
            scale_factor=1.0,
            x0=0.0,
            y0=0.0,
        )

    if platform == "XENIUM":
        mpp = _find_xenium_pixel_size(config)
        matrix = np.array(
            [
                [1.0 / mpp, 0.0, 0.0],
                [0.0, 1.0 / mpp, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return build_cellpose_affine_to_microns(
            matrix,
            scale_factor=1.0,
            x0=0.0,
            y0=0.0,
        )

    # Other 10x-family platforms (ATERA): the pixel size of the spec sidecars
    # the build copied into the store, with transform_path still first.
    spec = find_platform_spec(platform)
    if spec is not None and spec.family == "10X":
        mpp = _find_10x_pixel_size(config, spec.name)
        matrix = np.array(
            [
                [1.0 / mpp, 0.0, 0.0],
                [0.0, 1.0 / mpp, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return build_cellpose_affine_to_microns(
            matrix,
            scale_factor=1.0,
            x0=0.0,
            y0=0.0,
        )

    raise ValueError(f"Unknown platform: {config.platform}")


def _load_original_source(config: EnrichmentConfig) -> Any:
    """Load original platform source data for enrichment copy steps."""
    original_path = Path(config.original_data_path)
    if original_path.is_dir() and original_path.suffix == ".zarr":
        return sd.read_zarr(original_path)
    if original_path.is_file() and original_path.suffix == ".zarr":
        return sd.read_zarr(original_path)

    if config.platform.upper() == "XENIUM":
        return xenium_reader(
            original_path,
            cells_table=False,
            cells_as_circles=False,
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_labels=False,
            nucleus_labels=False,
            transcripts=False,
            morphology_focus=True,
            morphology_mip=False,
            aligned_images=False,
        )

    return sd.read_zarr(original_path)


def _copy_xenium_shapes_from_sdata(
    dst_sdata: Any,
    src_sdata: Any,
    *,
    force: bool,
) -> int:
    """Copy Xenium cell and nucleus boundary layers from a built SpatialData source."""
    copied = 0
    cell_key = _find_first_existing_key(
        src_sdata.shapes,
        ["cell_boundaries", XENIUM_OLD_CELL_SHAPE_NAME],
    )
    if cell_key is not None:
        copied += int(
            write_or_replace_element(
                dst_sdata,
                XENIUM_OLD_CELL_SHAPE_NAME,
                "shapes",
                src_sdata.shapes[cell_key].copy(),
                overwrite=force,
            )
        )

    nucleus_key = _find_first_existing_key(
        src_sdata.shapes,
        ["nucleus_boundaries", XENIUM_OLD_NUCLEUS_SHAPE_NAME],
    )
    if nucleus_key is not None:
        copied += int(
            write_or_replace_element(
                dst_sdata,
                XENIUM_OLD_NUCLEUS_SHAPE_NAME,
                "shapes",
                src_sdata.shapes[nucleus_key].copy(),
                overwrite=force,
            )
        )
    return copied


def _find_first_existing_key(mapping: Any, candidates: list[str]) -> str | None:
    """Return the first candidate key present in a SpatialData mapping."""
    for candidate in candidates:
        if candidate in mapping:
            return candidate
    return None


def _copy_merscope_images(
    dst_sdata: Any,
    src_sdata: Any,
    *,
    force: bool,
    image_prefix: str | None = None,
) -> int:
    """Copy only the MERSCOPE max-projection image into the destination."""
    if len(src_sdata.images) == 0:
        return 0
    projection = build_merscope_z_projection(
        src_sdata.images,
        image_prefix=image_prefix,
    )
    return int(
        write_or_replace_element(
            dst_sdata,
            MERSCOPE_ZPROJ_IMAGE_NAME,
            "images",
            projection,
            overwrite=force,
        )
    )


def _copy_xenium_images(dst_sdata: Any, src_sdata: Any, *, force: bool) -> int:
    """Copy Xenium image elements from source to destination SpatialData."""
    copied = 0
    for key in list(src_sdata.images.keys()):
        copied += int(
            write_or_replace_element(
                dst_sdata,
                key,
                "images",
                src_sdata.images[key],
                overwrite=force,
            )
        )
    return copied


def _is_native_store(platform: str, attrs: Any) -> bool:
    """Whether a store of ``platform`` holds a native (vendor) segmentation.

    A native latest store records ``merxen_schema.segmentation_source ==
    "native"`` (also when a ``native_proseg`` row added its ProSeg layer); a
    platform whose default segmentation source is ``native`` (ATERA) is native
    even when the record is missing. XENIUM and MERSCOPE stores are never
    native.

    Args:
        platform: Platform name of the store.
        attrs: The store's root attrs (``None`` or empty when unknown).

    Returns:
        ``True`` for a native store.

    Raises:
        UnknownPlatformError: If the platform is not registered.
    """
    spec = platform_spec(platform)
    schema = dict(attrs or {}).get(MERXEN_SCHEMA_ATTR)
    if isinstance(schema, dict) and schema.get("segmentation_source") == "native":
        return "native" in spec.segmentation_sources
    return spec.default_segmentation_source == "native"


def _missing_native_elements(
    dst_sdata: Any, spec: PlatformSpec
) -> dict[str, list[str]]:
    """Return the required native elements a store lacks, by element type.

    Args:
        dst_sdata: The latest store.
        spec: Its platform spec.

    Returns:
        ``{"shapes": [...], "tables": [...], "images": [...], "labels": [...],
        "points": [...]}`` with only the missing names (empty lists when all
        are present).
    """
    shapes = set(map(str, dst_sdata.shapes.keys()))
    tables = set(map(str, dst_sdata.tables.keys()))
    labels = set(map(str, getattr(dst_sdata, "labels", {}).keys()))
    images = set(map(str, dst_sdata.images.keys()))
    points = set(map(str, getattr(dst_sdata, "points", {}).keys()))
    return {
        "shapes": [key for key in spec.original_shape_keys if key not in shapes],
        "tables": [] if ORIGINAL_TABLE_NAME in tables else [ORIGINAL_TABLE_NAME],
        "images": [] if spec.native_image_key in images else [spec.native_image_key],
        "labels": [key for key in spec.original_label_keys if key not in labels],
        "points": [] if points else ["<any points element>"],
    }


def _has_native_elements(dst_sdata: Any, platform: str) -> bool:
    """Whether a native store has every vendor element enrichment needs."""
    missing = _missing_native_elements(dst_sdata, platform_spec(platform))
    return not any(missing.values())


def _is_already_enriched(dst_sdata: Any, platform: str) -> bool:
    """Check whether required enrichment artifacts already exist.

    A native store needs the vendor shapes, ``table_original``, the native
    image and the vendor label elements (no ``MOSAIK_*`` or Cellpose layer).
    """
    if _is_native_store(platform, getattr(dst_sdata, "attrs", None)):
        return _has_native_elements(dst_sdata, platform)
    req_shapes = {
        MOSAIK_PROSEG_SHAPE_NAME,
        MOSAIK_CELLPOSE_SHAPE_NAME,
        CELLPOSE_NUCLEI_SHAPE_NAME,
    }
    req_tables = {ORIGINAL_TABLE_NAME}
    spec = platform_spec(platform)
    req_shapes.update(spec.original_shape_keys)
    if spec.family == "MERSCOPE":
        req_images = {spec.native_image_key}
        has_images = req_images.issubset(set(map(str, dst_sdata.images.keys())))
    else:
        # Viewer-cache label grids and image pyramids require the real Xenium
        # morphology image. Derived cache images alone are not a valid source.
        has_images = _has_non_derived_image(dst_sdata.images)
    has_shapes = req_shapes.issubset(set(map(str, dst_sdata.shapes.keys())))
    has_tables = req_tables.issubset(set(map(str, dst_sdata.tables.keys())))
    return has_shapes and has_tables and has_images


def _is_enriched_except_nuclei(dst_sdata: Any, platform: str) -> bool:
    """Check whether a pre-nuclei enrichment has every legacy artifact.

    A native store has no Cellpose nuclei layer to add, so this is never true
    for it: its enrichment is complete or not (:func:`_is_already_enriched`).
    """
    if _is_native_store(platform, getattr(dst_sdata, "attrs", None)):
        return False
    req_shapes = {
        MOSAIK_PROSEG_SHAPE_NAME,
        MOSAIK_CELLPOSE_SHAPE_NAME,
    }
    req_tables = {ORIGINAL_TABLE_NAME}
    spec = platform_spec(platform)
    req_shapes.update(spec.original_shape_keys)
    if spec.family == "MERSCOPE":
        req_images = {spec.native_image_key}
        has_images = req_images.issubset(set(map(str, dst_sdata.images.keys())))
    else:
        has_images = _has_non_derived_image(dst_sdata.images)
    has_shapes = req_shapes.issubset(set(map(str, dst_sdata.shapes.keys())))
    has_tables = req_tables.issubset(set(map(str, dst_sdata.tables.keys())))
    return has_shapes and has_tables and has_images


def _has_non_derived_image(images: Any) -> bool:
    """Return whether an image mapping contains a real source image."""
    return any(not str(key).startswith("_napari_compare_") for key in images)


def _remove_per_shape_assignment_tables(dst_sdata: Any) -> None:
    """Remove derived per-shape tables before rebuilding partial enrichment."""
    protected_tables = {
        "table_MOSAIK_proseg_hybrid",
        "table_MOSAIK_proseg_geometry_assignment",
    }
    table_keys = [
        str(key)
        for key in list(dst_sdata.tables.keys())
        if str(key).startswith("table_")
        and str(key) != ORIGINAL_TABLE_NAME
        and str(key) not in protected_tables
    ]
    for table_key in table_keys:
        try:
            dst_sdata.delete_element_from_disk(table_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "delete_element_from_disk('%s') warning during enrichment rebuild: %s",
                table_key,
                exc,
            )
        with suppress(Exception):
            del dst_sdata.tables[table_key]


def _segmentation_result_tables(zarr_path: Path) -> set[str]:
    """Return the tables of registry branches refined from a prior.

    A branch with a ``prior`` (the stitched ATERA ProSeg layer,
    ``table_MOSAIK_proseg``) is a segmentation result that enrichment cannot
    rebuild, so its table is never a derived table. No XENIUM or MERSCOPE
    branch records a prior.
    """
    try:
        schema = read_root_attrs(Path(zarr_path)).get(MERXEN_SCHEMA_ATTR) or {}
    except (OSError, ValueError):
        return set()
    registry = dict(schema).get("segmentations") or {}
    return {
        str(entry["table"])
        for entry in registry.values()
        if isinstance(entry, dict) and entry.get("prior") and entry.get("table")
    }


def _derived_table_paths(zarr_path: Path) -> set[str]:
    """Return the ``tables/table_*`` paths of a store other than ``table_original``.

    Tables of branches refined from a prior (:func:`_segmentation_result_tables`)
    are left out.
    """
    rel_paths: set[str] = set()
    tables_dir = Path(zarr_path) / "tables"
    kept = _segmentation_result_tables(Path(zarr_path))
    if tables_dir.exists():
        for table_path in tables_dir.iterdir():
            table_name = table_path.name
            if (
                table_name.startswith("table_")
                and table_name != ORIGINAL_TABLE_NAME
                and table_name not in kept
            ):
                rel_paths.add(f"tables/{table_name}")
    return rel_paths


def _partial_enrichment_artifact_paths(zarr_path: Path, platform: str) -> set[str]:
    """Return zarr-relative paths for enrichment artifacts that can be rebuilt.

    A native store returns only its derived tables: its images, labels and
    points are payloads shared with the source store (hardlinked, read-only),
    and its renamed vendor shapes and ``table_original`` are re-created by
    :func:`merxen.segmentation.native.ensure_native_elements` when missing.
    """
    try:
        attrs = read_root_attrs(Path(zarr_path))
    except (OSError, ValueError):
        attrs = {}
    if _is_native_store(platform, attrs):
        return _derived_table_paths(Path(zarr_path))
    rel_paths = {
        f"shapes/{MOSAIK_PROSEG_SHAPE_NAME}",
        f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}",
        f"shapes/{CELLPOSE_NUCLEI_SHAPE_NAME}",
        f"tables/{ORIGINAL_TABLE_NAME}",
    }
    spec = platform_spec(platform)
    rel_paths.update(f"shapes/{key}" for key in spec.original_shape_keys)
    if spec.family == "MERSCOPE":
        rel_paths.add(f"images/{spec.native_image_key}")

    rel_paths.update(_derived_table_paths(Path(zarr_path)))
    return rel_paths


def _prune_zarr_consolidated_metadata(
    zarr_json_path: Path,
    rel_paths: set[str],
) -> None:
    """Remove stale consolidated metadata entries for deleted zarr elements."""
    if not rel_paths or not zarr_json_path.exists():
        return
    try:
        data = json.loads(zarr_json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read zarr metadata %s: %s", zarr_json_path, exc)
        return

    consolidated = data.get("consolidated_metadata")
    metadata = consolidated.get("metadata") if isinstance(consolidated, dict) else None
    if not isinstance(metadata, dict):
        return

    def _should_remove(key: str) -> bool:
        return any(
            key == rel_path or key.startswith(f"{rel_path}/") for rel_path in rel_paths
        )

    pruned = {
        key: value for key, value in metadata.items() if not _should_remove(str(key))
    }
    if len(pruned) == len(metadata):
        return
    consolidated["metadata"] = pruned
    zarr_json_path.write_text(json.dumps(data, indent=2) + "\n")


def _remove_partial_enrichment_artifacts_from_zarr_path(
    zarr_path: Path,
    platform: str,
) -> None:
    """Delete known rebuildable enrichment artifacts from an on-disk zarr."""
    zarr_path = Path(zarr_path)
    rel_paths = _partial_enrichment_artifact_paths(zarr_path, platform)
    for rel_path in sorted(rel_paths):
        remove_path(zarr_path / rel_path)

    _prune_zarr_consolidated_metadata(zarr_path / "zarr.json", rel_paths)
    for container in ["images", "shapes", "tables"]:
        container_rel_paths = {
            rel_path.split("/", 1)[1]
            for rel_path in rel_paths
            if rel_path.startswith(f"{container}/")
        }
        _prune_zarr_consolidated_metadata(
            zarr_path / container / "zarr.json",
            container_rel_paths,
        )


def _is_partial_enrichment_read_error(exc: Exception) -> bool:
    """Return True when a partial enrichment raster blocks SpatialData reads."""
    return isinstance(exc, KeyError) and exc.args == ("ome",)


def _read_latest_zarr_for_enrichment(
    write_path: Path,
    platform: str,
    dataset_name: str,
) -> Any:
    """Read latest zarr, recovering from interrupted partial enrichment writes.

    Raises:
        RuntimeError: If a native store has an unreadable raster: its rasters
            are payloads only NATIVE_LATEST can restore, so nothing is removed.
    """
    try:
        return sd.read_zarr(write_path)
    except Exception as exc:
        if not _is_partial_enrichment_read_error(exc):
            raise
        try:
            attrs = read_root_attrs(Path(write_path))
        except (OSError, ValueError):
            attrs = {}
        if _is_native_store(platform, attrs):
            # A native store's rasters are payloads shared with the source
            # store; deleting derived tables cannot make them readable.
            raise RuntimeError(
                f"[{dataset_name}] Native latest store has an unreadable raster "
                f"element ({exc!r}); its images and labels are payloads shared "
                "with the source store, so rebuild it with NATIVE_LATEST "
                "(segment stage)."
            ) from exc
        log_status(
            f"[{dataset_name}] Latest zarr contains an unreadable partial "
            "enrichment artifact; removing rebuildable enrichment artifacts."
        )
        _remove_partial_enrichment_artifacts_from_zarr_path(write_path, platform)
        force_release(note=f"after partial enrichment cleanup {dataset_name}")
        return sd.read_zarr(write_path)


def _source_image_preprocessing(config: EnrichmentConfig) -> dict[str, Any] | None:
    """Read the preprocessing record of the source SpatialData (``None`` if none)."""
    original_path = Path(config.original_data_path)
    if not original_path.exists():
        return None
    return read_image_preprocessing_attr(original_path)


def _check_store_matches_source(
    write_path: Path, config: EnrichmentConfig, dataset_name: str
) -> None:
    """Refuse to enrich a ProSeg store that was built from another image.

    The segmentation step records which image the store's masks came from. A
    store whose record disagrees with the source's (or has none while the
    source is bead-corrected) holds cells from an image other than the one that
    is quantified, and stamping the source's record onto it would hide that.

    Raises:
        ValueError: The store and the source disagree about bead removal.
    """
    check_store_segmentation_sidecar(
        write_path,
        _source_image_preprocessing(config),
        dataset_name=dataset_name,
    )


def _copy_image_preprocessing_attr(dst: Any, config: EnrichmentConfig) -> bool:
    """Carry the source build's preprocessing record onto the latest zarr root.

    The latest zarr is rebuilt from the ProSeg output and so never inherits the
    attributes of the SpatialData it was segmented from. Without the record,
    downstream stages could not tell that PolyT/DAPI intensities came from a
    bead-corrected projection. A source without a record (legacy build, raw
    Xenium bundle) leaves the store untouched.

    Args:
        dst: The latest SpatialData store being enriched; its metadata is
            persisted by the caller, which holds the store write lock.
        config: Enrichment configuration.

    Returns:
        True when the attribute was added or changed.
    """
    recorded = _source_image_preprocessing(config)
    if recorded is None or dst.attrs.get(IMAGE_PREPROCESSING_ATTR) == recorded:
        return False
    dst.attrs[IMAGE_PREPROCESSING_ATTR] = recorded
    return True


def enrich_single_latest(
    config: EnrichmentConfig,
    *,
    force_rerun: bool = False,
    keep_backup: bool = False,
    polygon_n_jobs: int = 16,
    polygon_show_progress: bool = True,
) -> Path:
    """Enrich latest ProSeg output with explicit shape/image/table layers."""
    latest_path = Path(config.latest_zarr_path)
    target_path = (
        Path(config.persistent_output_path)
        if config.persistent_output_path is not None
        else latest_path
    )
    dataset_name = str(config.dataset_name).upper()

    if not latest_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Latest zarr not found: {latest_path}"
        )

    write_path = _resolve_enrichment_write_path(latest_path, target_path)
    # Enrichment adds and replaces elements and rewrites the store's attrs, so
    # the store lock is held from the read until the metadata is persisted.
    with spatialdata_write_lock(write_path):
        if config.segmentation_source in ("native", "native_proseg"):
            _enrich_native_in_place(
                config,
                write_path=write_path,
                force_rerun=force_rerun,
            )
            if config.segmentation_source == "native_proseg":
                _check_atera_proseg_layer(config, write_path=write_path)
        else:
            _enrich_store_in_place(
                config,
                write_path=write_path,
                force_rerun=force_rerun,
                polygon_n_jobs=polygon_n_jobs,
                polygon_show_progress=polygon_show_progress,
            )

    if keep_backup:
        log_status(
            f"[{dataset_name}] keep_backup ignored; "
            "enrichment writes elements in place."
        )
    if latest_path != write_path:
        stage_existing_output(write_path, latest_path)
    return write_path


def _check_atera_proseg_layer(config: EnrichmentConfig, *, write_path: Path) -> None:
    """Check the stitched ProSeg layer of a ``native_proseg`` row (AT-9).

    The layer is written by ATERA_PROSEG_STITCH, not by enrichment; this runs
    its streaming check (table and shape ids, assignments, per-cell counts,
    registry) once per stitch and records it in
    ``attrs["merxen_atera_proseg"]["enrich_check"]``. The caller holds the
    store's write lock.

    Raises:
        RuntimeError: If the layer is missing or the check fails.
    """
    from merxen.segmentation.atera_proseg import PROVENANCE_ATTR, AteraProsegError
    from merxen.segmentation.atera_proseg.stitch import check_proseg_layer

    dataset_name = str(config.dataset_name).upper()
    provenance = dict(read_root_attrs(write_path).get(PROVENANCE_ATTR) or {})
    previous = dict(provenance.get("enrich_check") or {})
    if (
        previous.get("passed")
        and previous.get("fingerprint") is not None
        and previous.get("fingerprint") == provenance.get("fingerprint")
    ):
        log_status(f"[{dataset_name}] ProSeg layer already checked; skipping.")
        return
    try:
        result = check_proseg_layer(write_path)
    except AteraProsegError as error:
        raise RuntimeError(f"[{dataset_name}] {error}") from error
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atera_proseg_layer_check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if not result["passed"]:
        raise RuntimeError(
            f"[{dataset_name}] ProSeg layer check failed: "
            f"{'; '.join(result['failures'])}"
        )
    dst = sd.read_zarr(write_path, selection=("tables",))
    record = dict(dst.attrs.get(PROVENANCE_ATTR) or {})
    record["enrich_check"] = result
    dst.attrs[PROVENANCE_ATTR] = record
    dst.write_attrs()
    del dst
    log_status(
        f"[{dataset_name}] ProSeg layer check passed: {result['n_cells']:,} cells, "
        f"{result['n_foreground_gene_rows']:,} foreground gene transcripts."
    )


def _record_mask_model_identities(dst: Any, config: EnrichmentConfig) -> bool:
    """Copy mask provenance into the registry without changing writer_versions."""
    schema = dst.attrs.setdefault(MERXEN_SCHEMA_ATTR, {})
    # Cell tables are registered in the later assignment stage. Retain the
    # identities at schema level so that registration can copy them then.
    identities = schema.setdefault("model_identities", {})
    registry = schema.get("segmentations", {})
    changed = False
    for branch, mask_path in (
        ("cellpose", config.mask_path),
        ("cellpose_nuclei", config.nuclei_mask_path),
    ):
        if mask_path is None:
            raise ValueError(
                f"{branch} needs a Cellpose mask; native rows record no mask model"
            )
        existing = registry.get(branch, {}).get(
            "model_identity", identities.get(branch)
        )
        if find_model_identity_path(Path(mask_path)) is None and existing is not None:
            identity = existing
        else:
            identity = read_model_identity(Path(mask_path)).model_dump(mode="json")
        if identities.get(branch) != identity:
            identities[branch] = identity
            changed = True
        if branch in registry and registry[branch].get("model_identity") != identity:
            registry[branch]["model_identity"] = identity
            changed = True
    return changed


def _enrich_store_in_place(
    config: EnrichmentConfig,
    *,
    write_path: Path,
    force_rerun: bool,
    polygon_n_jobs: int,
    polygon_show_progress: bool,
) -> None:
    """Enrich the store at ``write_path``; the caller holds its write lock."""
    if config.mask_path is None or config.nuclei_mask_path is None:
        # The config validator allows this only for segmentation_source=native,
        # which enrich_single_latest sends to _enrich_native_in_place.
        raise ValueError(
            f"segmentation_source={config.segmentation_source} enrichment needs "
            "mask_path and nuclei_mask_path"
        )
    mask_path = Path(config.mask_path)
    nuclei_mask_path = Path(config.nuclei_mask_path)
    dataset_name = str(config.dataset_name).upper()
    platform = config.platform.upper()

    # Before anything is read or written, also when the store is skipped as
    # already enriched.
    _check_store_matches_source(write_path, config, dataset_name)
    log_status(f"[{dataset_name}] Loading latest zarr for enrichment: {write_path}")
    dst = _read_latest_zarr_for_enrichment(write_path, platform, dataset_name)

    already_enriched = _is_already_enriched(dst, platform)
    if already_enriched and (not force_rerun):
        # The record is refreshed here too: a store enriched before the source
        # was corrected (or by an older version) has none, and quantification
        # reads it from this store.
        changed = _record_mask_model_identities(dst, config)
        changed = _copy_image_preprocessing_attr(dst, config) or changed
        if changed:
            write_spatialdata_metadata(dst, write_attrs=True)
        log_status(f"[{dataset_name}] Already enriched; skipping.")
        del dst
        force_release(note=f"after enrichment skip {dataset_name}")
        return

    nuclei_missing = CELLPOSE_NUCLEI_SHAPE_NAME not in dst.shapes
    if not force_rerun and nuclei_missing and _is_enriched_except_nuclei(dst, platform):
        log_status(
            f"[{dataset_name}] Existing enrichment predates the nuclei layer; "
            "adding nuclei without rebuilding existing shapes or tables."
        )
        proseg_template = dst.shapes[MOSAIK_PROSEG_SHAPE_NAME]
        x_transform, y_transform = _dataset_cellpose_transform(config)
        nuclei_gdf = _cellpose_gdf_from_mask(
            nuclei_mask_path,
            x_transform,
            y_transform,
            dataset_name=dataset_name,
            polygon_n_jobs=polygon_n_jobs,
            polygon_show_progress=polygon_show_progress,
            shape_name=CELLPOSE_NUCLEI_SHAPE_NAME,
            id_prefix="nucleus",
        )
        nuclei_shapes = _parse_shapes_with_template(
            nuclei_gdf,
            template_shape=proseg_template,
        )
        write_or_replace_element(
            dst,
            CELLPOSE_NUCLEI_SHAPE_NAME,
            "shapes",
            nuclei_shapes,
            overwrite=False,
        )
        register_missing_mask_branches(dst)
        _record_mask_model_identities(dst, config)
        invalidate_alignment_materialization(
            dst,
            reason="incremental Cellpose nucleus shapes added",
        )
        _copy_image_preprocessing_attr(dst, config)
        write_spatialdata_metadata(dst, write_attrs=True)
        del dst, nuclei_gdf
        force_release(note=f"after incremental nuclei enrichment ({dataset_name})")
        log_status(
            f"[{dataset_name}] Incremental nuclei enrichment complete: {write_path}"
        )
        return

    overwrite_existing = bool(force_rerun or not already_enriched)
    if overwrite_existing and not force_rerun:
        log_status(
            f"[{dataset_name}] Existing enrichment is incomplete; "
            "rebuilding partial enrichment artifacts."
        )
        _remove_per_shape_assignment_tables(dst)

    # 1. Ensure explicit MOSAIK_proseg layer exists.
    proseg_src_key = None
    for cand in [
        "cell_boundaries",
        "cell_boundaries_refined",
        "shapes",
        MOSAIK_PROSEG_SHAPE_NAME,
    ]:
        if cand in dst.shapes:
            proseg_src_key = cand
            break
    if proseg_src_key is None and len(dst.shapes) > 0:
        proseg_src_key = list(dst.shapes.keys())[0]
    if proseg_src_key is None:
        raise RuntimeError(f"[{dataset_name}] No shapes found in latest zarr.")

    proseg_template = dst.shapes[proseg_src_key]
    write_or_replace_element(
        dst,
        MOSAIK_PROSEG_SHAPE_NAME,
        "shapes",
        proseg_template.copy(),
        overwrite=overwrite_existing,
    )

    # 2. Add explicit Cellpose polygons.
    x_transform, y_transform = _dataset_cellpose_transform(config)
    cp_gdf = _cellpose_gdf_from_mask(
        mask_path,
        x_transform,
        y_transform,
        dataset_name=dataset_name,
        polygon_n_jobs=polygon_n_jobs,
        polygon_show_progress=polygon_show_progress,
    )
    cp_shapes = _parse_shapes_with_template(cp_gdf, template_shape=proseg_template)
    write_or_replace_element(
        dst,
        MOSAIK_CELLPOSE_SHAPE_NAME,
        "shapes",
        cp_shapes,
        overwrite=overwrite_existing,
    )

    nuclei_gdf = _cellpose_gdf_from_mask(
        nuclei_mask_path,
        x_transform,
        y_transform,
        dataset_name=dataset_name,
        polygon_n_jobs=polygon_n_jobs,
        polygon_show_progress=polygon_show_progress,
        shape_name=CELLPOSE_NUCLEI_SHAPE_NAME,
        id_prefix="nucleus",
    )
    nuclei_shapes = _parse_shapes_with_template(
        nuclei_gdf,
        template_shape=proseg_template,
    )
    write_or_replace_element(
        dst,
        CELLPOSE_NUCLEI_SHAPE_NAME,
        "shapes",
        nuclei_shapes,
        overwrite=overwrite_existing,
    )

    # 3. Load original source data and copy boundaries/images/table.
    src = _load_original_source(config)

    if platform == "MERSCOPE":
        if len(src.shapes) == 0:
            raise RuntimeError("[MERSCOPE] No original shapes found to copy.")
        old_key = list(src.shapes.keys())[0]
        write_or_replace_element(
            dst,
            MERSCOPE_OLD_SHAPE_NAME,
            "shapes",
            src.shapes[old_key].copy(),
            overwrite=overwrite_existing,
        )
        added = _copy_merscope_images(
            dst,
            src,
            force=overwrite_existing,
            image_prefix=None,
        )
        log_status(f"[MERSCOPE] Added/updated {added} image entries")

        if len(src.tables) > 0:
            old_tbl_key = list(src.tables.keys())[0]
            old_tbl = _prepare_original_table(
                src.tables[old_tbl_key],
                MERSCOPE_OLD_SHAPE_NAME,
            )
            write_or_replace_element(
                dst,
                ORIGINAL_TABLE_NAME,
                "tables",
                old_tbl,
                overwrite=overwrite_existing,
            )
        else:
            log_status("[MERSCOPE] WARNING: original source has no tables.")

    elif platform == "XENIUM":
        original_path = Path(config.original_data_path)
        xenium_dir = original_path
        if original_path.suffix == ".zarr":
            copied_shapes = _copy_xenium_shapes_from_sdata(
                dst,
                src,
                force=overwrite_existing,
            )
            if copied_shapes == 0:
                raise RuntimeError(
                    "[XENIUM] No cell/nucleus shapes found in the built"
                    " SpatialData source."
                )
        else:
            xenium_dir = original_path
            cell_gdf = _load_xenium_boundary_shapes_from_csv(xenium_dir, which="cell")
            if len(cell_gdf) == 0:
                raise RuntimeError(
                    "[XENIUM] No cell boundaries parsed from cell_boundaries.csv.gz"
                )
            cell_shapes = _parse_shapes_with_template(
                cell_gdf,
                template_shape=proseg_template,
            )
            write_or_replace_element(
                dst,
                XENIUM_OLD_CELL_SHAPE_NAME,
                "shapes",
                cell_shapes,
                overwrite=overwrite_existing,
            )

            nuc_gdf = _load_xenium_boundary_shapes_from_csv(xenium_dir, which="nucleus")
            if len(nuc_gdf) > 0:
                nuc_shapes = _parse_shapes_with_template(
                    nuc_gdf,
                    template_shape=proseg_template,
                )
                write_or_replace_element(
                    dst,
                    XENIUM_OLD_NUCLEUS_SHAPE_NAME,
                    "shapes",
                    nuc_shapes,
                    overwrite=overwrite_existing,
                )

        added = _copy_xenium_images(dst, src, force=overwrite_existing)
        log_status(f"[XENIUM] Added/updated {added} image entries")

        if len(src.tables) > 0:
            old_tbl_src = src.tables[list(src.tables.keys())[0]]
        else:
            old_tbl_src = None
        if old_tbl_src is None:
            old_tbl_src = _load_xenium_original_table_from_matrix(xenium_dir)
        if old_tbl_src is not None:
            old_tbl = _prepare_original_table(old_tbl_src, XENIUM_OLD_CELL_SHAPE_NAME)
            write_or_replace_element(
                dst,
                ORIGINAL_TABLE_NAME,
                "tables",
                old_tbl,
                overwrite=overwrite_existing,
            )
        else:
            log_status("[XENIUM] WARNING: no original Xenium table available.")
    else:
        raise ValueError(f"Unsupported platform: {config.platform}")

    register_missing_mask_branches(dst)
    _record_mask_model_identities(dst, config)
    invalidate_alignment_materialization(
        dst,
        reason="native enrichment points, shapes, tables, or images replaced",
    )
    _copy_image_preprocessing_attr(dst, config)
    write_spatialdata_metadata(dst, write_attrs=True)
    del dst, src, cp_gdf, nuclei_gdf
    force_release(note=f"after in-place enrichment ({dataset_name})")
    log_status(f"[{dataset_name}] Enrichment complete: {write_path}")


def native_label_check_path(output_dir: Path, dataset_name: str) -> Path:
    """Return the JSON path of a dataset's native label check summary."""
    return Path(output_dir) / f"{str(dataset_name).lower()}_native_label_check.json"


def _native_pixel_size(config: EnrichmentConfig, spec: PlatformSpec) -> float:
    """Return the microns per vendor-mask pixel of a native store.

    ``transform_path`` comes first, then the spec sidecars of the source store
    (the same lookup as the pixel->micron transform of the store's images).
    """
    return _find_10x_pixel_size(config, spec.name)


def _points_scale_um_per_px(points: Any) -> float | None:
    """Return the microns per ``global`` unit the points element records.

    Native stores map microns to the ``global`` pixel frame with
    ``Scale(1/ps)``; ``None`` when the transform is not a pure x/y scale.
    """
    from spatialdata.transformations import Scale, get_transformation

    try:
        transform = get_transformation(points, to_coordinate_system="global")
    except (KeyError, ValueError, AssertionError):
        return None
    if not isinstance(transform, Scale):
        return None
    scales = dict(zip(transform.axes, (float(v) for v in transform.scale), strict=True))
    sx, sy = scales.get("x"), scales.get("y")
    if not sx or not sy or not bool(np.isclose(sx, sy, rtol=1e-9)):
        return None
    return float(1.0 / sx)


def _points_source(write_path: Path, sdata_obj: Any, points_key: str) -> Any:
    """Return the parquet files of a points element, or the element itself.

    The files come in natural name order (the writer's tile order), so the
    partitions in flight together share decoded label blocks.
    """
    parquet_dir = points_parquet_dir(write_path, points_key)
    files = (
        natural_partition_order(list(parquet_dir.glob("*.parquet")))
        if parquet_dir.is_dir()
        else []
    )
    if files:
        return files
    return sdata_obj.points[points_key]


def _level0(element: Any) -> Any:
    """Return the level-0 array (lazy) of a single- or multi-scale raster."""
    from spatialdata import get_pyramid_levels

    level = get_pyramid_levels(element, n=0)
    return level.data


def _run_native_label_check(
    dst: Any,
    config: EnrichmentConfig,
    *,
    write_path: Path,
    spec: PlatformSpec,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the native label check of a store (one streaming pass)."""
    dataset_name = str(config.dataset_name).upper()
    points_key = choose_primary_points_key(dst)
    if points_key is None:
        raise RuntimeError(f"[{dataset_name}] Native store has no points element")
    points = dst.points[points_key]
    point_columns = [str(column) for column in points.columns]
    assignment_column = "original_assignment"
    if assignment_column not in point_columns:
        raise RuntimeError(
            f"[{dataset_name}] Native points '{points_key}' lack the vendor "
            f"assignment column '{assignment_column}'"
        )
    cell_label_key, nucleus_label_key = spec.original_label_keys
    nucleus_shapes = dst.shapes[spec.original_nucleus_shape_key]
    cell_key = native_segmentation.NUCLEUS_CELL_KEY
    if cell_key not in nucleus_shapes.columns:
        raise RuntimeError(
            f"[{dataset_name}] Nucleus shapes '{spec.original_nucleus_shape_key}' "
            f"lack '{cell_key}'; nuclei cannot be joined to their cells"
        )
    table = dst.tables[ORIGINAL_TABLE_NAME]
    um_per_px = _native_pixel_size(config, spec)
    log_status(
        f"[{dataset_name}] Native label check: {points_key} against "
        f"{cell_label_key}/{nucleus_label_key} at {um_per_px:.8f} um/px (corner)"
    )
    summary, per_cell = native_label_check(
        _points_source(write_path, dst, points_key),
        cell_labels=_level0(dst.labels[cell_label_key]),
        nucleus_labels=_level0(dst.labels[nucleus_label_key]),
        nucleus_ids=nucleus_shapes[INSTANCE_ID_COLUMN].to_numpy(dtype=np.int64),
        nucleus_cell_ids=nucleus_shapes[cell_key].to_numpy(dtype=np.int64),
        table_cell_ids=table.obs[INSTANCE_ID_COLUMN].to_numpy(dtype=np.int64),
        um_per_px=um_per_px,
        assignment_column=assignment_column,
        available_columns=point_columns,
    )
    summary["dataset_name"] = dataset_name
    summary["points_key"] = points_key
    summary["cell_labels"] = cell_label_key
    summary["nucleus_labels"] = nucleus_label_key
    summary["nucleus_shapes"] = spec.original_nucleus_shape_key
    summary["nucleus_cell_key"] = cell_key
    store_um_per_px = _points_scale_um_per_px(points)
    summary["store_points_um_per_px"] = store_um_per_px
    summary["store_points_scale_agrees"] = bool(
        store_um_per_px is not None
        and np.isclose(store_um_per_px, um_per_px, rtol=1e-6, atol=0.0)
    )
    # The pixel size of the lookup must be the one the store's points record.
    checks = dict(summary["checks"])
    checks["points_scale"] = summary["store_points_scale_agrees"]
    summary["checks"] = checks
    summary["failures"] = sorted(name for name, ok in checks.items() if not ok)
    summary["passed"] = all(checks.values())
    summary["count_columns"] = {
        key: NATIVE_COUNT_COLUMNS[key] for key in sorted(per_cell)
    }
    return summary, per_cell


def _vendor_segmentation_provenance(dst: Any, spec: PlatformSpec) -> dict[str, Any]:
    """Return the vendor segmentation provenance recorded for a native store."""
    header = dict(getattr(dst, "attrs", {}).get("merxen_atera", {}) or {})
    experiment = dict(header.get("experiment", {}) or {})
    provenance: dict[str, Any] = {
        "source": "vendor",
        "platform": spec.name,
        "analysis_sw_version": experiment.get("analysis_sw_version"),
        "segmentation_stain": experiment.get("segmentation_stain"),
        "cell_shapes": spec.original_cell_shape_key,
        "nucleus_shapes": spec.original_nucleus_shape_key,
        "label_elements": list(spec.original_label_keys),
    }
    cells = dst.shapes.get(spec.original_cell_shape_key)
    if cells is not None and "segmentation_method" in cells.columns:
        counts = cells["segmentation_method"].astype(str).value_counts(sort=False)
        provenance["cell_segmentation_methods"] = {
            str(method): int(count) for method, count in sorted(counts.items())
        }
    return provenance


def _record_native_provenance(dst: Any, spec: PlatformSpec) -> bool:
    """Record ``segmentation_source=native`` and the vendor provenance.

    Unlike :func:`_record_mask_model_identities`, nothing Cellpose-shaped is
    written: a native store's geometry comes from the vendor.

    Returns:
        True when the attrs changed.
    """
    schema = dict(dst.attrs.get(MERXEN_SCHEMA_ATTR, {}))
    provenance = _vendor_segmentation_provenance(dst, spec)
    changed = (
        schema.get("segmentation_source") != "native"
        or schema.get(NATIVE_PROVENANCE_KEY) != provenance
    )
    schema["segmentation_source"] = "native"
    schema[NATIVE_PROVENANCE_KEY] = provenance
    dst.attrs[MERXEN_SCHEMA_ATTR] = schema
    return changed


def _native_enrichment_is_current(dst: Any, config: EnrichmentConfig) -> bool:
    """Whether a native store's enrichment is complete for this config."""
    record = dict(getattr(dst, "attrs", {}).get(NATIVE_ENRICHMENT_ATTR, {}) or {})
    if record.get("version") != NATIVE_ENRICHMENT_VERSION:
        return False
    if not _has_native_elements(dst, config.platform):
        return False
    if not config.native_label_check:
        return True
    check = record.get("label_check")
    if not isinstance(check, dict) or not check.get("passed"):
        return False
    columns = dict(check.get("count_columns", {}) or {})
    obs_columns = set(map(str, dst.tables[ORIGINAL_TABLE_NAME].obs.columns))
    return bool(columns) and set(columns.values()).issubset(obs_columns)


def _write_native_label_check(
    summary: dict[str, Any], config: EnrichmentConfig
) -> Path:
    """Write the label check summary to the task output folder."""
    path = native_label_check_path(config.output_dir, config.dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def _ensure_native_elements_present(
    dst: Any,
    config: EnrichmentConfig,
    *,
    write_path: Path,
    spec: PlatformSpec,
) -> Any:
    """Re-create missing renamed vendor elements, or refuse a broken store.

    Shapes and ``table_original`` are re-created from the source store by
    :func:`merxen.segmentation.native.ensure_native_elements`. Missing images,
    labels or points are payloads shared with the source store; only
    NATIVE_LATEST can restore them.

    Returns:
        The (re-read) store.

    Raises:
        RuntimeError: If payload elements are missing, or elements are still
            missing after the re-creation.
    """
    dataset_name = str(config.dataset_name).upper()
    missing = _missing_native_elements(dst, spec)
    payloads = {key: missing[key] for key in ("images", "labels", "points")}
    if any(payloads.values()):
        raise RuntimeError(
            f"[{dataset_name}] Native latest store lacks payload elements "
            f"{ {key: value for key, value in payloads.items() if value} }; "
            "rebuild it with NATIVE_LATEST (segment stage)."
        )
    if not (missing["shapes"] or missing["tables"]):
        return dst
    log_status(
        f"[{dataset_name}] Native latest store lacks "
        f"{missing['shapes'] + missing['tables']}; re-creating them from the "
        "source store."
    )
    created = native_segmentation.ensure_native_elements(
        write_path, Path(config.original_data_path)
    )
    log_status(f"[{dataset_name}] Re-created native elements: {created}")
    del dst
    force_release(note=f"after native element re-creation ({dataset_name})")
    dst = sd.read_zarr(write_path)
    still_missing = _missing_native_elements(dst, spec)
    if any(still_missing.values()):
        raise RuntimeError(
            f"[{dataset_name}] Native elements still missing after re-creation: "
            f"{ {key: value for key, value in still_missing.items() if value} }"
        )
    return dst


def _write_native_count_columns(
    dst: Any, per_cell: dict[str, np.ndarray], dataset_name: str
) -> None:
    """Add the native per-cell counts to ``table_original`` and persist it."""
    table = dst.tables[ORIGINAL_TABLE_NAME]
    for key, values in per_cell.items():
        table.obs[NATIVE_COUNT_COLUMNS[key]] = np.asarray(values, dtype=np.int64)
    write_or_replace_element(
        dst,
        ORIGINAL_TABLE_NAME,
        "tables",
        table,
        overwrite=True,
    )
    log_status(
        f"[{dataset_name}] Added {sorted(NATIVE_COUNT_COLUMNS[k] for k in per_cell)} "
        f"to {ORIGINAL_TABLE_NAME}"
    )


def _enrich_native_in_place(
    config: EnrichmentConfig,
    *,
    write_path: Path,
    force_rerun: bool,
) -> None:
    """Enrich a native (vendor-segmentation) store; the caller holds its lock.

    The store comes from NATIVE_LATEST: the vendor shapes and table are already
    registered under their latest names. This verifies and completes those
    elements, runs the streaming label check (vendor assignments against the
    cell labels; nuclear transcripts attributed to cells through the nucleus
    labels and ``cell_instance_id``), adds the per-cell counts to
    ``table_original`` and records the vendor provenance. No Cellpose mask is
    read, and no ``MOSAIK_*`` layer or Cellpose model identity is written.

    Raises:
        RuntimeError: If the label check fails or required elements are
            missing.
    """
    dataset_name = str(config.dataset_name).upper()
    spec = platform_spec(config.platform)
    if not spec.original_label_keys or spec.original_nucleus_shape_key is None:
        raise ValueError(
            f"[{dataset_name}] {spec.name} has no vendor labels for native enrichment"
        )

    _check_store_matches_source(write_path, config, dataset_name)
    log_status(
        f"[{dataset_name}] Loading native latest zarr for enrichment: {write_path}"
    )
    dst = _read_latest_zarr_for_enrichment(write_path, spec.name, dataset_name)
    dst = _ensure_native_elements_present(dst, config, write_path=write_path, spec=spec)

    if not force_rerun and _native_enrichment_is_current(dst, config):
        changed = _record_native_provenance(dst, spec)
        changed = _copy_image_preprocessing_attr(dst, config) or changed
        changed = bool(register_missing_mask_branches(dst)) or changed
        if changed:
            write_spatialdata_metadata(dst, write_attrs=True)
        record = dict(dst.attrs.get(NATIVE_ENRICHMENT_ATTR, {}))
        if isinstance(record.get("label_check"), dict):
            _write_native_label_check(record["label_check"], config)
        log_status(f"[{dataset_name}] Native store already enriched; skipping.")
        del dst
        force_release(note=f"after native enrichment skip {dataset_name}")
        return

    label_check: dict[str, Any] | None = None
    if config.native_label_check:
        label_check, per_cell = _run_native_label_check(
            dst, config, write_path=write_path, spec=spec
        )
        check_path = _write_native_label_check(label_check, config)
        if not label_check["passed"]:
            nucleus = dict(label_check.get("nucleus", {}) or {})
            raise RuntimeError(
                f"[{dataset_name}] Native label check failed "
                f"({', '.join(label_check['failures'])}): "
                f"{label_check['n_mismatch']:,} of {label_check['n_rows']:,} "
                "transcripts disagree with the vendor assignment (at most "
                f"{label_check['mismatch_allowed']:,} allowed); "
                f"{nucleus.get('n_unknown_nucleus_labels', 0):,} on unknown "
                f"nucleus labels; {nucleus.get('n_nucleus_outside_own_cell', 0):,} "
                "nuclear transcripts outside their own cell; store points "
                f"scale {label_check['store_points_um_per_px']} um/px against "
                f"{label_check['um_per_px']} um/px. The pixel size, the masks or "
                f"the nucleus-to-cell join do not match the points; see "
                f"{check_path.name}."
            )
        nucleus = label_check.get("nucleus", {})
        log_status(
            f"[{dataset_name}] Native label check passed: "
            f"{label_check['n_mismatch']} mismatches over "
            f"{label_check['n_rows']:,} transcripts; nucleus overlap agreement "
            f"{nucleus.get('overlaps_agreement')}"
        )
        _write_native_count_columns(dst, per_cell, dataset_name)
        del per_cell
        invalidate_alignment_materialization(
            dst, reason="native enrichment added per-cell counts to table_original"
        )

    _record_native_provenance(dst, spec)
    register_missing_mask_branches(dst)
    _copy_image_preprocessing_attr(dst, config)
    dst.attrs[NATIVE_ENRICHMENT_ATTR] = {
        "version": NATIVE_ENRICHMENT_VERSION,
        "label_check": label_check,
    }
    write_spatialdata_metadata(dst, write_attrs=True)
    del dst
    force_release(note=f"after native enrichment ({dataset_name})")
    log_status(f"[{dataset_name}] Native enrichment complete: {write_path}")
