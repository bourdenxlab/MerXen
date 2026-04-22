"""SpatialData enrichment utilities."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from scipy import sparse as sps
from scipy.io import mmread
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel
from spatialdata_io import xenium as xenium_reader
from tqdm.auto import tqdm

from merxen.config import EnrichmentConfig
from merxen.io.image_source import _get_image_dataarray, list_plane_keys
from merxen.io.spatialdata_io import write_spatialdata_zarr
from merxen.memory import force_release, log_status
from merxen.segmentation.cellpose import build_cellpose_affine_to_microns
from merxen.segmentation.mask_geometry import masks_to_polygons

logger = logging.getLogger(__name__)

MOSAIK_PROSEG_SHAPE_NAME = "MOSAIK_proseg"
MOSAIK_CELLPOSE_SHAPE_NAME = "MOSAIK_cellpose"
MERSCOPE_OLD_SHAPE_NAME = "merscope_cell_boundaries"
XENIUM_OLD_CELL_SHAPE_NAME = "xenium_cell_boundaries"
XENIUM_OLD_NUCLEUS_SHAPE_NAME = "xenium_nucleus"
ORIGINAL_TABLE_NAME = "table_original"
MERSCOPE_ZPROJ_IMAGE_NAME = "MERSCOPE_z_projection"


def _delete_if_exists(mapping: Any, key: str) -> None:
    """Delete an element from a SpatialData mapping if present."""
    try:
        if key in mapping:
            del mapping[key]
    except Exception:  # noqa: BLE001
        pass


def _set_element(mapping: Any, key: str, value: Any, *, force: bool = False) -> bool:
    """Set element in a SpatialData mapping with optional overwrite semantics."""
    if (key in mapping) and (not force):
        return False
    if key in mapping and force:
        _delete_if_exists(mapping, key)
    mapping[key] = value
    return True


def _remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory tree if it exists."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _to_cyx(image_like: Any) -> Any:
    """Convert image-like input to (c, y, x) ordering."""
    da = _get_image_dataarray(image_like)
    dims = tuple(str(d) for d in da.dims)
    if all(d in dims for d in ("c", "y", "x")):
        return da.transpose("c", "y", "x")
    if all(d in dims for d in ("y", "x", "c")):
        return da.transpose("c", "y", "x")
    if all(d in dims for d in ("y", "x")):
        return da.expand_dims(c=["c0"]).transpose("c", "y", "x")
    raise ValueError(f"Unsupported image dims for conversion to (c,y,x): {dims}")


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
    tbl = adata.copy()
    spatial_attrs = dict(tbl.uns.get("spatialdata_attrs", {}))
    region_key = str(spatial_attrs.get("region_key", "region"))
    instance_key = spatial_attrs.get("instance_key")

    if region_key not in tbl.obs.columns:
        region_key = "region"
        tbl.obs[region_key] = target_region_name
    tbl.obs[region_key] = pd.Categorical([target_region_name] * tbl.n_obs)

    if (instance_key is None) or (instance_key not in tbl.obs.columns):
        for cand in ["cell_id", "cell", "cell_ID", "region"]:
            if cand in tbl.obs.columns:
                instance_key = cand
                break

    if (instance_key is None) or (instance_key not in tbl.obs.columns):
        instance_key = "cell_id"
        tbl.obs[instance_key] = tbl.obs_names.astype(str)

    tbl.uns["spatialdata_attrs"] = {
        "region": target_region_name,
        "region_key": region_key,
        "instance_key": instance_key,
    }
    return tbl


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
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of Cellpose polygons in micron space."""
    if not mask_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Missing Cellpose mask file: {mask_path}"
        )

    log_status(
        f"[{dataset_name}] Building {MOSAIK_CELLPOSE_SHAPE_NAME} from {mask_path}"
    )
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
            "cell_id": [f"cellpose_{i + 1}" for i in range(len(polygons_um))],
            "geometry": polygons_um,
        },
        geometry="geometry",
    )


def _load_merscope_transform(config: EnrichmentConfig) -> np.ndarray:
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


def _find_xenium_pixel_size(config: EnrichmentConfig) -> float:
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


def _dataset_cellpose_transform(
    config: EnrichmentConfig,
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
            _set_element(
                dst_sdata.shapes,
                XENIUM_OLD_CELL_SHAPE_NAME,
                src_sdata.shapes[cell_key].copy(),
                force=force,
            )
        )

    nucleus_key = _find_first_existing_key(
        src_sdata.shapes,
        ["nucleus_boundaries", "xenium_nucleus"],
    )
    if nucleus_key is not None:
        copied += int(
            _set_element(
                dst_sdata.shapes,
                XENIUM_OLD_NUCLEUS_SHAPE_NAME,
                src_sdata.shapes[nucleus_key].copy(),
                force=force,
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
    """Copy MERSCOPE image planes and add a lazy z-projection."""
    plane_pairs = list_plane_keys(src_sdata.images, prefix=image_prefix)
    plane_keys = [k for _, k in plane_pairs]
    if len(plane_keys) == 0:
        plane_keys = list(src_sdata.images.keys())

    copied = 0
    for key in plane_keys:
        copied += int(
            _set_element(
                dst_sdata.images,
                key,
                src_sdata.images[key],
                force=force,
            )
        )

    if len(plane_keys) > 0:
        log_status(
            f"[MERSCOPE] Building lazy z-projection from {len(plane_keys)} planes"
        )
        plane_arrays = [_to_cyx(src_sdata.images[k]) for k in plane_keys]
        projection = xr.concat(plane_arrays, dim="z").max(dim="z", keep_attrs=True)
        copied += int(
            _set_element(
                dst_sdata.images,
                MERSCOPE_ZPROJ_IMAGE_NAME,
                projection,
                force=force,
            )
        )
    return copied


def _copy_xenium_images(dst_sdata: Any, src_sdata: Any, *, force: bool) -> int:
    """Copy Xenium image elements from source to destination SpatialData."""
    copied = 0
    for key in list(src_sdata.images.keys()):
        copied += int(
            _set_element(
                dst_sdata.images,
                key,
                src_sdata.images[key],
                force=force,
            )
        )
    return copied


def _is_already_enriched(dst_sdata: Any, dataset_name: str) -> bool:
    """Check whether required enrichment artifacts already exist."""
    req_shapes = {MOSAIK_PROSEG_SHAPE_NAME, MOSAIK_CELLPOSE_SHAPE_NAME}
    req_tables = {ORIGINAL_TABLE_NAME}
    if dataset_name.upper() == "MERSCOPE":
        req_shapes.add(MERSCOPE_OLD_SHAPE_NAME)
        req_images = {MERSCOPE_ZPROJ_IMAGE_NAME}
    else:
        req_shapes.update({XENIUM_OLD_CELL_SHAPE_NAME, XENIUM_OLD_NUCLEUS_SHAPE_NAME})
        req_images = set()
    has_shapes = req_shapes.issubset(set(map(str, dst_sdata.shapes.keys())))
    has_tables = req_tables.issubset(set(map(str, dst_sdata.tables.keys())))
    has_images = req_images.issubset(set(map(str, dst_sdata.images.keys())))
    return has_shapes and has_tables and has_images


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
    mask_path = Path(config.mask_path)
    dataset_name = str(config.dataset_name).upper()
    platform = config.platform.upper()

    if not latest_path.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Latest zarr not found: {latest_path}"
        )

    log_status(f"[{dataset_name}] Loading latest zarr for enrichment: {latest_path}")
    dst = sd.read_zarr(latest_path)

    if _is_already_enriched(dst, dataset_name) and (not force_rerun):
        log_status(f"[{dataset_name}] Already enriched; skipping.")
        del dst
        force_release(note=f"after enrichment skip {dataset_name}")
        return latest_path

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
    _set_element(
        dst.shapes,
        MOSAIK_PROSEG_SHAPE_NAME,
        proseg_template.copy(),
        force=force_rerun,
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
    _set_element(dst.shapes, MOSAIK_CELLPOSE_SHAPE_NAME, cp_shapes, force=force_rerun)

    # 3. Load original source data and copy boundaries/images/table.
    src = _load_original_source(config)

    if platform == "MERSCOPE":
        if len(src.shapes) == 0:
            raise RuntimeError("[MERSCOPE] No original shapes found to copy.")
        old_key = list(src.shapes.keys())[0]
        _set_element(
            dst.shapes,
            MERSCOPE_OLD_SHAPE_NAME,
            src.shapes[old_key].copy(),
            force=force_rerun,
        )
        added = _copy_merscope_images(
            dst,
            src,
            force=force_rerun,
            image_prefix=None,
        )
        log_status(f"[MERSCOPE] Added/updated {added} image entries")

        if len(src.tables) > 0:
            old_tbl_key = list(src.tables.keys())[0]
            old_tbl = _prepare_original_table(
                src.tables[old_tbl_key],
                MERSCOPE_OLD_SHAPE_NAME,
            )
            _set_element(dst.tables, ORIGINAL_TABLE_NAME, old_tbl, force=force_rerun)
        else:
            log_status("[MERSCOPE] WARNING: original source has no tables.")

    elif platform == "XENIUM":
        original_path = Path(config.original_data_path)
        if original_path.suffix == ".zarr":
            copied_shapes = _copy_xenium_shapes_from_sdata(
                dst,
                src,
                force=force_rerun,
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
            _set_element(
                dst.shapes,
                XENIUM_OLD_CELL_SHAPE_NAME,
                cell_shapes,
                force=force_rerun,
            )

            nuc_gdf = _load_xenium_boundary_shapes_from_csv(xenium_dir, which="nucleus")
            if len(nuc_gdf) > 0:
                nuc_shapes = _parse_shapes_with_template(
                    nuc_gdf,
                    template_shape=proseg_template,
                )
                _set_element(
                    dst.shapes,
                    XENIUM_OLD_NUCLEUS_SHAPE_NAME,
                    nuc_shapes,
                    force=force_rerun,
                )

        added = _copy_xenium_images(dst, src, force=force_rerun)
        log_status(f"[XENIUM] Added/updated {added} image entries")

        if len(src.tables) > 0:
            old_tbl_src = src.tables[list(src.tables.keys())[0]]
        else:
            old_tbl_src = None
        if old_tbl_src is None:
            old_tbl_src = _load_xenium_original_table_from_matrix(xenium_dir)
        if old_tbl_src is not None:
            old_tbl = _prepare_original_table(old_tbl_src, XENIUM_OLD_CELL_SHAPE_NAME)
            _set_element(dst.tables, ORIGINAL_TABLE_NAME, old_tbl, force=force_rerun)
        else:
            log_status("[XENIUM] WARNING: no original Xenium table available.")
    else:
        raise ValueError(f"Unsupported platform: {config.platform}")

    # 4. Safe write + atomic replace.
    tmp_out = latest_path.parent / f"{latest_path.stem}__enrich_tmp.zarr"
    backup_out = latest_path.parent / f"{latest_path.stem}__pre_enrich_backup.zarr"
    _remove_path(tmp_out)

    log_status(f"[{dataset_name}] Writing enriched zarr to temp path: {tmp_out}")
    write_spatialdata_zarr(dst, tmp_out, overwrite=True)

    del dst, src, cp_gdf
    force_release(note=f"after writing enriched temp zarr ({dataset_name})")

    if keep_backup:
        _remove_path(backup_out)
        if latest_path.exists() or latest_path.is_symlink():
            shutil.move(str(latest_path), str(backup_out))
            log_status(f"[{dataset_name}] Backup saved: {backup_out}")
    elif latest_path.exists() or latest_path.is_symlink():
        _remove_path(latest_path)

    shutil.move(str(tmp_out), str(latest_path))
    log_status(f"[{dataset_name}] Enrichment complete: {latest_path}")
    return latest_path
