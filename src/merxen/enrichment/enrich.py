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

from merxen.config import EnrichmentConfig
from merxen.enrichment.assignment import clone_table_for_region
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    build_merscope_z_projection,
    image_to_cyx,
)
from merxen.io.spatialdata_io import write_or_replace_element
from merxen.memory import force_release, log_status
from merxen.path_utils import remove_path, stage_existing_output
from merxen.segmentation.cellpose import build_cellpose_affine_to_microns
from merxen.segmentation.mask_geometry import masks_to_polygons

logger = logging.getLogger(__name__)

MOSAIK_PROSEG_SHAPE_NAME = "MOSAIK_proseg"
MOSAIK_CELLPOSE_SHAPE_NAME = "MOSAIK_cellpose"
CELLPOSE_NUCLEI_SHAPE_NAME = "cellpose_nuclei"
MERSCOPE_OLD_SHAPE_NAME = "merscope_cell_boundaries"
XENIUM_OLD_CELL_SHAPE_NAME = "xenium_cell_boundaries"
XENIUM_OLD_NUCLEUS_SHAPE_NAME = "xenium_nucleus"
ORIGINAL_TABLE_NAME = "table_original"


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
    polygons_px = masks_to_polygons(
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

    polygons_um: list[Polygon] = []
    for poly in tqdm(
        polygons_px,
        desc=f"[{dataset_name}] cellpose polygons->microns",
        unit="poly",
    ):
        if poly is None or poly.is_empty:
            continue
        transformed = affine_transform(poly, coeffs)
        if transformed is not None and not transformed.is_empty:
            polygons_um.append(transformed)

    del mask_mmap, polygons_px
    force_release(note=f"after {dataset_name} cellpose polygon conversion")

    if len(polygons_um) == 0:
        raise RuntimeError(
            f"[{dataset_name}] No valid polygons created from Cellpose masks."
        )

    return gpd.GeoDataFrame(
        {
            "cell_id": [f"{id_prefix}_{i + 1}" for i in range(len(polygons_um))],
            "geometry": polygons_um,
        },
        geometry="geometry",
    )


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
    """Find Xenium micron-per-pixel value from config transform sources."""
    candidates: list[Path] = []
    if config.transform_path is not None:
        candidates.append(Path(config.transform_path))
    candidates.extend(
        [
            Path(config.original_data_path) / "experiment.xenium",
            Path(config.original_data_path) / "specs.json",
            Path(config.original_data_path) / "specs" / "specs.json",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in {".txt", ".csv"}:
            matrix = np.loadtxt(candidate)
            if matrix.shape == (3, 3):
                # candidate stores micron->pixel, derive mpp from inverse scale.
                sx = float(matrix[0, 0])
                if sx != 0:
                    return 1.0 / sx
        try:
            data = json.loads(candidate.read_text())
            if "pixel_size" in data:
                return float(data["pixel_size"])
            if "microns_per_pixel" in data:
                return float(data["microns_per_pixel"])
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        "Could not determine Xenium pixel_size from transform_path/original_data_path."
    )


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
        ["cell_boundaries", "xenium_cell_boundaries"],
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
        ["nucleus_boundaries", "xenium_nucleus"],
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


def _is_already_enriched(dst_sdata: Any, platform: str) -> bool:
    """Check whether required enrichment artifacts already exist."""
    req_shapes = {
        MOSAIK_PROSEG_SHAPE_NAME,
        MOSAIK_CELLPOSE_SHAPE_NAME,
        CELLPOSE_NUCLEI_SHAPE_NAME,
    }
    req_tables = {ORIGINAL_TABLE_NAME}
    if platform.upper() == "MERSCOPE":
        req_shapes.add(MERSCOPE_OLD_SHAPE_NAME)
        req_images = {MERSCOPE_ZPROJ_IMAGE_NAME}
    else:
        req_shapes.update({XENIUM_OLD_CELL_SHAPE_NAME, XENIUM_OLD_NUCLEUS_SHAPE_NAME})
        req_images = set()
    has_shapes = req_shapes.issubset(set(map(str, dst_sdata.shapes.keys())))
    has_tables = req_tables.issubset(set(map(str, dst_sdata.tables.keys())))
    has_images = req_images.issubset(set(map(str, dst_sdata.images.keys())))
    return has_shapes and has_tables and has_images


def _is_enriched_except_nuclei(dst_sdata: Any, platform: str) -> bool:
    """Check whether a pre-nuclei enrichment has every legacy artifact."""
    req_shapes = {
        MOSAIK_PROSEG_SHAPE_NAME,
        MOSAIK_CELLPOSE_SHAPE_NAME,
    }
    req_tables = {ORIGINAL_TABLE_NAME}
    if platform.upper() == "MERSCOPE":
        req_shapes.add(MERSCOPE_OLD_SHAPE_NAME)
        req_images = {MERSCOPE_ZPROJ_IMAGE_NAME}
    else:
        req_shapes.update({XENIUM_OLD_CELL_SHAPE_NAME, XENIUM_OLD_NUCLEUS_SHAPE_NAME})
        req_images = set()
    has_shapes = req_shapes.issubset(set(map(str, dst_sdata.shapes.keys())))
    has_tables = req_tables.issubset(set(map(str, dst_sdata.tables.keys())))
    has_images = req_images.issubset(set(map(str, dst_sdata.images.keys())))
    return has_shapes and has_tables and has_images


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


def _partial_enrichment_artifact_paths(zarr_path: Path, platform: str) -> set[str]:
    """Return zarr-relative paths for enrichment artifacts that can be rebuilt."""
    rel_paths = {
        f"shapes/{MOSAIK_PROSEG_SHAPE_NAME}",
        f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}",
        f"shapes/{CELLPOSE_NUCLEI_SHAPE_NAME}",
        f"tables/{ORIGINAL_TABLE_NAME}",
    }
    if platform.upper() == "MERSCOPE":
        rel_paths.update(
            {
                f"shapes/{MERSCOPE_OLD_SHAPE_NAME}",
                f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}",
            }
        )
    else:
        rel_paths.update(
            {
                f"shapes/{XENIUM_OLD_CELL_SHAPE_NAME}",
                f"shapes/{XENIUM_OLD_NUCLEUS_SHAPE_NAME}",
            }
        )

    tables_dir = Path(zarr_path) / "tables"
    if tables_dir.exists():
        for table_path in tables_dir.iterdir():
            table_name = table_path.name
            if table_name.startswith("table_") and table_name != ORIGINAL_TABLE_NAME:
                rel_paths.add(f"tables/{table_name}")
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
    """Read latest zarr, recovering from interrupted partial enrichment writes."""
    try:
        return sd.read_zarr(write_path)
    except Exception as exc:
        if not _is_partial_enrichment_read_error(exc):
            raise
        log_status(
            f"[{dataset_name}] Latest zarr contains an unreadable partial "
            "enrichment artifact; removing rebuildable enrichment artifacts."
        )
        _remove_partial_enrichment_artifacts_from_zarr_path(write_path, platform)
        force_release(note=f"after partial enrichment cleanup {dataset_name}")
        return sd.read_zarr(write_path)


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
    mask_path = Path(config.mask_path)
    nuclei_mask_path = Path(config.nuclei_mask_path)
    dataset_name = str(config.dataset_name).upper()
    platform = config.platform.upper()

    if not latest_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Latest zarr not found: {latest_path}"
        )

    write_path = _resolve_enrichment_write_path(latest_path, target_path)
    log_status(f"[{dataset_name}] Loading latest zarr for enrichment: {write_path}")
    dst = _read_latest_zarr_for_enrichment(write_path, platform, dataset_name)

    already_enriched = _is_already_enriched(dst, platform)
    if already_enriched and (not force_rerun):
        log_status(f"[{dataset_name}] Already enriched; skipping.")
        del dst
        force_release(note=f"after enrichment skip {dataset_name}")
        if latest_path != write_path:
            stage_existing_output(write_path, latest_path)
        return write_path

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
        del dst, nuclei_gdf
        force_release(note=f"after incremental nuclei enrichment ({dataset_name})")
        if latest_path != write_path:
            stage_existing_output(write_path, latest_path)
        log_status(
            f"[{dataset_name}] Incremental nuclei enrichment complete: {write_path}"
        )
        return write_path

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

    del dst, src, cp_gdf, nuclei_gdf
    force_release(note=f"after in-place enrichment ({dataset_name})")

    if keep_backup:
        log_status(
            f"[{dataset_name}] keep_backup ignored; "
            "enrichment writes elements in place."
        )
    if latest_path != write_path:
        stage_existing_output(write_path, latest_path)
    log_status(f"[{dataset_name}] Enrichment complete: {write_path}")
    return write_path
