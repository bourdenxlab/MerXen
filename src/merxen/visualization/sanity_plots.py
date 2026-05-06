"""Sanity overlay plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import box as shapely_box

SANITY_CROP_SIZE_UM = 250.0
SANITY_MAX_TRANSCRIPTS_PER_PANEL = 2_000_000
SANITY_RANDOM_STATE = 42
SANITY_ASSIGNMENT_SHAPE_KEY = "MOSAIK_proseg"


def _prepare_overlay_image(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    """Convert microscopy image data into a display-safe grayscale or RGB array."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        return _normalize_channel(arr), "gray"

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image data, got shape {arr.shape}")

    if arr.shape[-1] == 1:
        return _normalize_channel(arr[..., 0]), "gray"

    arr = arr.astype(np.float32, copy=False)
    if arr.shape[-1] == 2:
        arr = np.concatenate([arr, np.zeros_like(arr[..., :1])], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    return _normalize_channel(arr), None


def _normalize_channel(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize an array to the [0, 1] range for plotting."""
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    values = arr[finite]
    lo, hi = np.percentile(values, (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.asarray(np.clip((arr - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)
    scaled[~finite] = 0.0
    return scaled


def plot_sanity_overlay(
    image: np.ndarray,
    output_path: Path | str,
    *,
    shapes: gpd.GeoDataFrame | None = None,
    points: pd.DataFrame | None = None,
    x_col: str = "x",
    y_col: str = "y",
    title: str = "Sanity Overlay",
    point_size: float = 1.5,
) -> Path:
    """Overlay shape boundaries and transcript points on an image crop."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_image, cmap = _prepare_overlay_image(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_image, cmap=cmap)

    if shapes is not None and len(shapes) > 0:
        shapes.boundary.plot(ax=ax, linewidth=0.5, color="#F97316", alpha=0.85)

    if points is not None and len(points) > 0:
        ax.scatter(
            points[x_col].to_numpy(float),
            points[y_col].to_numpy(float),
            s=point_size,
            c="#06B6D4",
            alpha=0.6,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_pair_sanity_crops(
    merscope_sdata: Any,
    xenium_sdata: Any,
    output_path: Path | str,
    *,
    merscope_zarr_path: Path | str | None = None,
    xenium_zarr_path: Path | str | None = None,
    crop_size_um: float = SANITY_CROP_SIZE_UM,
    assignment_shape_key: str | None = SANITY_ASSIGNMENT_SHAPE_KEY,
) -> Path:
    """Plot paired MOSAIK-style 250 um sanity crops for MERSCOPE and Xenium."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    plot_sanity_crop_panel(
        axes[0],
        merscope_sdata,
        "MERSCOPE",
        crop_size_um=crop_size_um,
        assignment_shape_key=assignment_shape_key,
        zarr_path=merscope_zarr_path,
    )
    plot_sanity_crop_panel(
        axes[1],
        xenium_sdata,
        "XENIUM",
        crop_size_um=crop_size_um,
        assignment_shape_key=assignment_shape_key,
        zarr_path=xenium_zarr_path,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_sanity_crop_panel(
    ax: plt.Axes,
    sdata_obj: Any,
    dataset_name: str,
    *,
    crop_size_um: float = SANITY_CROP_SIZE_UM,
    center_xy: tuple[float, float] | None = None,
    assignment_shape_key: str | None = SANITY_ASSIGNMENT_SHAPE_KEY,
    zarr_path: Path | str | None = None,
) -> None:
    """Draw one MOSAIK-style image, shape, and assignment sanity crop panel."""
    bbox, ref_shape_key = _choose_crop_bbox(
        sdata_obj,
        size_um=crop_size_um,
        center_xy=center_xy,
    )
    x0, y0, x1, y1 = bbox

    bg = _get_background_image_crop(sdata_obj, dataset_name, bbox, zarr_path=zarr_path)
    if bg is not None:
        ax.imshow(
            bg["rgb"],
            extent=bg["extent_um"],
            origin="lower",
            interpolation="nearest",
            alpha=0.95,
        )

    aligned_shape_keys = [
        key for key in sdata_obj.shapes if str(key).endswith("_aligned_nonrigid")
    ]
    shape_keys = aligned_shape_keys or list(sdata_obj.shapes.keys())
    cmap = plt.get_cmap("tab10")
    shape_handles = []

    for i, shape_key in enumerate(shape_keys):
        try:
            shp_crop = _crop_single_shape(sdata_obj, shape_key, bbox)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{dataset_name}] Warning: failed to crop shapes[{shape_key}] ({exc})"
            )
            continue

        if len(shp_crop) == 0:
            continue

        color = cmap(i % 10)
        shp_crop.boundary.plot(ax=ax, linewidth=0.75, color=color, alpha=0.95)
        shape_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
                label=f"{shape_key} ({len(shp_crop):,})",
            )
        )

    tx_crop, _points_key, _assign_col = _crop_points(
        sdata_obj,
        bbox,
        max_points=SANITY_MAX_TRANSCRIPTS_PER_PANEL,
        random_state=SANITY_RANDOM_STATE,
        assignment_shape_key=assignment_shape_key,
    )

    tx_handles = []
    if len(tx_crop) > 0:
        unassigned = tx_crop[~tx_crop["assigned"]]
        assigned = tx_crop[tx_crop["assigned"]]

        if len(unassigned) > 0:
            ax.scatter(
                unassigned["x_um"],
                unassigned["y_um"],
                s=4,
                c="#d62728",
                alpha=0.50,
                rasterized=True,
            )
            tx_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="#d62728",
                    label=f"Unassigned tx ({len(unassigned):,})",
                    markersize=5,
                )
            )

        if len(assigned) > 0:
            ax.scatter(
                assigned["x_um"],
                assigned["y_um"],
                s=4,
                c="yellow",
                alpha=0.50,
                rasterized=True,
            )
            tx_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color="yellow",
                    label=f"Assigned tx ({len(assigned):,})",
                    markersize=5,
                )
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (microns)")
    ax.set_ylabel("y (microns)")
    ax.set_title(
        f"{dataset_name} sanity crop ({crop_size_um:.0f} x {crop_size_um:.0f} um)"
    )

    handles = tx_handles + shape_handles
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)

    if ref_shape_key not in shape_keys:
        print(f"[{dataset_name}] Warning: reference shape {ref_shape_key} not plotted")


def _shape_geometry_only(shape_obj: Any) -> gpd.GeoDataFrame:
    if "geometry" in shape_obj.columns:
        gdf = shape_obj[["geometry"]].copy()
    else:
        gdf = gpd.GeoDataFrame({"geometry": shape_obj.geometry}, index=shape_obj.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def _reference_shape_key(sdata_obj: Any) -> str:
    for key in [
        "MOSAIK_proseg_aligned_nonrigid",
        "cell_boundaries_aligned_nonrigid",
        "MOSAIK_cellpose_aligned_nonrigid",
        "MOSAIK_proseg",
        "cell_boundaries",
        "MOSAIK_cellpose",
    ]:
        if key in sdata_obj.shapes:
            return key
    if len(sdata_obj.shapes) == 0:
        raise RuntimeError("No shapes found in SpatialData object.")
    return str(list(sdata_obj.shapes.keys())[0])


def _bounded_interval(
    center: float,
    size: float,
    min_v: float,
    max_v: float,
) -> tuple[float, float]:
    span = max_v - min_v
    if span <= size:
        return float(min_v), float(max_v)
    half = size / 2.0
    lo = center - half
    hi = center + half
    if lo < min_v:
        hi += min_v - lo
        lo = min_v
    if hi > max_v:
        lo -= hi - max_v
        hi = max_v
    return float(lo), float(hi)


def _choose_crop_bbox(
    sdata_obj: Any,
    *,
    size_um: float,
    center_xy: tuple[float, float] | None,
) -> tuple[tuple[float, float, float, float], str]:
    ref_key = _reference_shape_key(sdata_obj)
    gdf = _shape_geometry_only(sdata_obj.shapes[ref_key])
    if len(gdf) == 0:
        raise RuntimeError(f"No non-empty geometries in shapes[{ref_key}]")

    minx, miny, maxx, maxy = gdf.total_bounds
    if center_xy is None:
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
    else:
        cx, cy = center_xy

    x0, x1 = _bounded_interval(cx, size_um, minx, maxx)
    y0, y1 = _bounded_interval(cy, size_um, miny, maxy)
    return (x0, y0, x1, y1), ref_key


def _crop_single_shape(
    sdata_obj: Any,
    shape_key: str,
    bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    gdf = _shape_geometry_only(sdata_obj.shapes[shape_key])
    crop_poly = shapely_box(*bbox)
    keep = gdf.geometry.intersects(crop_poly)
    return gdf.loc[keep].copy()


def _assign_points_by_shape(
    points_pdf: pd.DataFrame,
    shapes_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    n_points = len(points_pdf)
    if n_points == 0 or len(shapes_gdf) == 0:
        return np.zeros(n_points, dtype=bool)

    pts_gdf = gpd.GeoDataFrame(
        {"_row_id": np.arange(n_points, dtype=np.int64)},
        geometry=gpd.points_from_xy(
            points_pdf["x_um"].to_numpy(),
            points_pdf["y_um"].to_numpy(),
        ),
    )
    shp = shapes_gdf[["geometry"]].copy()

    try:
        joined = gpd.sjoin(
            pts_gdf[["_row_id", "geometry"]],
            shp,
            how="left",
            predicate="within",
        )
        matched = joined.loc[
            joined["index_right"].notna(),
            "_row_id",
        ].to_numpy(dtype=np.int64, copy=False)
        assigned = np.zeros(n_points, dtype=bool)
        if matched.size:
            assigned[np.unique(matched)] = True
        return assigned
    except Exception:  # noqa: BLE001
        return _assign_points_by_shape_fallback(pts_gdf, shp)


def _assign_points_by_shape_fallback(
    pts_gdf: gpd.GeoDataFrame,
    shapes_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    assigned = np.zeros(len(pts_gdf), dtype=bool)
    try:
        sindex = shapes_gdf.sindex
        use_sindex = sindex is not None
    except Exception:  # noqa: BLE001
        sindex = None
        use_sindex = False

    for i, point in enumerate(pts_gdf.geometry.values):
        candidates = (
            list(sindex.intersection(point.bounds))
            if use_sindex
            else range(len(shapes_gdf))
        )
        if not candidates:
            continue
        assigned[i] = any(shapes_gdf.geometry.iloc[j].covers(point) for j in candidates)
    return assigned


def _crop_points(
    sdata_obj: Any,
    bbox: tuple[float, float, float, float],
    *,
    max_points: int | None,
    random_state: int,
    assignment_shape_key: str | None,
) -> tuple[pd.DataFrame, str, str | None]:
    if len(sdata_obj.points) == 0:
        raise RuntimeError("No points found in SpatialData object.")

    points_key = _reference_points_key(sdata_obj)
    pts = sdata_obj.points[points_key]
    x_col = _first_existing_col(
        pts,
        ["x", "global_x", "x_location", "x_micron", "observed_x", "x_global_px"],
    )
    y_col = _first_existing_col(
        pts,
        ["y", "global_y", "y_location", "y_micron", "observed_y", "y_global_px"],
    )
    assign_col = _first_existing_col(pts, ["assignment", "cell", "cell_id"])
    if x_col is None or y_col is None:
        raise KeyError(f"Could not resolve x/y columns in points[{points_key}]")
    if (
        assignment_shape_key == SANITY_ASSIGNMENT_SHAPE_KEY
        and f"{assignment_shape_key}_aligned_nonrigid" in sdata_obj.shapes
    ):
        assignment_shape_key = f"{assignment_shape_key}_aligned_nonrigid"

    x0, y0, x1, y1 = bbox
    cols = [x_col, y_col] + ([assign_col] if assign_col is not None else [])

    if hasattr(pts, "npartitions") and hasattr(pts, "partitions"):
        work = pts[cols]
        work = work[
            (work[x_col] >= x0)
            & (work[x_col] <= x1)
            & (work[y_col] >= y0)
            & (work[y_col] <= y1)
        ]
        pdf = work.compute()
    else:
        pdf = pd.DataFrame(pts[cols]).copy()
        pdf = pdf[
            (pdf[x_col] >= x0)
            & (pdf[x_col] <= x1)
            & (pdf[y_col] >= y0)
            & (pdf[y_col] <= y1)
        ].copy()

    pdf = pdf.rename(columns={x_col: "x_um", y_col: "y_um"})

    if assignment_shape_key is not None:
        if assignment_shape_key not in sdata_obj.shapes:
            raise KeyError(
                f"assignment_shape_key='{assignment_shape_key}' not found in shapes. "
                f"Available: {list(sdata_obj.shapes.keys())}"
            )
        shape_crop = _crop_single_shape(sdata_obj, assignment_shape_key, bbox)
        pdf["assigned"] = _assign_points_by_shape(pdf, shape_crop)
        assign_col = f"shape:{assignment_shape_key}"
    elif assign_col is not None and assign_col in pdf.columns:
        pdf["assigned"] = _assignment_mask(pdf[assign_col]).values
    else:
        pdf["assigned"] = True

    if max_points is not None and len(pdf) > max_points:
        pdf = pdf.sample(n=max_points, random_state=random_state)

    return pdf, points_key, assign_col


def _reference_points_key(sdata_obj: Any) -> str:
    for key in sdata_obj.points:
        if str(key).endswith("_aligned_nonrigid"):
            return str(key)
    return str(list(sdata_obj.points.keys())[0])


def _get_scale0_dataarray(image_elem: Any) -> Any:
    if hasattr(image_elem, "keys") and "scale0" in image_elem:
        node = image_elem["scale0"]
        if hasattr(node, "ds"):
            if "image" in node.ds:
                return node.ds["image"]
            if len(node.ds.data_vars) > 0:
                return next(iter(node.ds.data_vars.values()))
    if hasattr(image_elem, "ds"):
        if "image" in image_elem.ds:
            return image_elem.ds["image"]
        if len(image_elem.ds.data_vars) > 0:
            return next(iter(image_elem.ds.data_vars.values()))
    return image_elem


def _pick_channel_name(channel_labels: list[str], preferred: list[str]) -> str | None:
    lower = [str(label).lower() for label in channel_labels]
    for preferred_name in preferred:
        preferred_lower = preferred_name.lower()
        for i, label in enumerate(lower):
            if label == preferred_lower or preferred_lower in label:
                return channel_labels[i]
    return None


def _norm01(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(values[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    return np.asarray(np.clip((values - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def _get_background_image_crop(
    sdata_obj: Any,
    dataset_name: str,
    bbox: tuple[float, float, float, float],
    *,
    zarr_path: Path | str | None,
) -> dict[str, Any] | None:
    if len(sdata_obj.images) == 0:
        return None

    if dataset_name.upper() == "MERSCOPE":
        image_key = _pick_merscope_image_key(sdata_obj.images)
        ch2_pref = ["PolyT", "18S"]
    else:
        image_key = (
            "morphology_focus"
            if "morphology_focus" in sdata_obj.images
            else list(sdata_obj.images.keys())[0]
        )
        ch2_pref = ["18S", "PolyT"]

    try:
        da = _get_scale0_dataarray(sdata_obj.images[image_key])
        x_transform, y_transform = _resolve_dataset_mask_affine(
            dataset_name,
            zarr_path=zarr_path,
        )
        crop = _crop_image_dataarray_to_bbox(da, bbox, x_transform, y_transform)
    except Exception as exc:  # noqa: BLE001
        print(f"[{dataset_name}] Warning: failed to crop background image ({exc})")
        return None

    if crop is None:
        return None

    channels = [str(c) for c in crop.coords["c"].values] if "c" in crop.coords else []
    ch_dapi = _pick_channel_name(channels, ["DAPI"]) if channels else None
    ch_rna = _pick_channel_name(channels, ch2_pref) if channels else None

    if "c" in crop.dims:
        dapi_da = crop.sel(c=ch_dapi) if ch_dapi is not None else crop.isel(c=0)
        if ch_rna is not None:
            rna_da = crop.sel(c=ch_rna)
        else:
            rna_da = crop.isel(c=1 if crop.sizes["c"] > 1 else 0)
    else:
        dapi_da = crop
        rna_da = crop

    dapi = _norm01(_to_numpy(dapi_da))
    rna = _norm01(_to_numpy(rna_da))
    rgb = np.zeros((dapi.shape[0], dapi.shape[1], 3), dtype=np.float32)
    rgb[..., 2] = dapi
    rgb[..., 1] = rna

    extent_um = _crop_extent_um(crop, x_transform, y_transform)
    return {
        "image_key": image_key,
        "channels_used": {"dapi": ch_dapi, "rna_like": ch_rna},
        "rgb": rgb,
        "extent_um": extent_um,
        "transform": {"x_transform": x_transform, "y_transform": y_transform},
    }


def _pick_merscope_image_key(images: Any) -> str:
    if "MERSCOPE_z_projection" in images:
        return "MERSCOPE_z_projection"
    projection_key = next(
        (key for key in images if "projection" in str(key).lower()),
        None,
    )
    if projection_key is not None:
        return str(projection_key)
    return str(list(images.keys())[0])


def _crop_image_dataarray_to_bbox(
    da: Any,
    bbox: tuple[float, float, float, float],
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> Any | None:
    x0_um, y0_um, x1_um, y1_um = bbox
    corners_um = [
        (x0_um, y0_um),
        (x0_um, y1_um),
        (x1_um, y0_um),
        (x1_um, y1_um),
    ]
    corners_px = [
        _affine_um_to_px(xu, yu, x_transform, y_transform) for xu, yu in corners_um
    ]
    px_vals = np.array([point[0] for point in corners_px], dtype=float)
    py_vals = np.array([point[1] for point in corners_px], dtype=float)
    px0, px1 = float(px_vals.min() - 1.0), float(px_vals.max() + 1.0)
    py0, py1 = float(py_vals.min() - 1.0), float(py_vals.max() + 1.0)

    crop = da.sel(x=_coord_slice(da, "x", px0, px1), y=_coord_slice(da, "y", py0, py1))
    if crop.sizes.get("x", 0) == 0 or crop.sizes.get("y", 0) == 0:
        return None
    return crop


def _coord_slice(da: Any, dim: str, lo: float, hi: float) -> slice:
    coord = np.asarray(da.coords[dim].values)
    if coord.size >= 2 and coord[0] > coord[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def _crop_extent_um(
    crop: Any,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    xv = np.asarray(crop.coords["x"].values)
    yv = np.asarray(crop.coords["y"].values)
    px_corners = [
        (float(xv.min()), float(yv.min())),
        (float(xv.min()), float(yv.max())),
        (float(xv.max()), float(yv.min())),
        (float(xv.max()), float(yv.max())),
    ]
    um_corners = [
        _affine_px_to_um(px, py, x_transform, y_transform) for px, py in px_corners
    ]
    x_um_vals = [corner[0] for corner in um_corners]
    y_um_vals = [corner[1] for corner in um_corners]
    return (min(x_um_vals), max(x_um_vals), min(y_um_vals), max(y_um_vals))


def _resolve_dataset_mask_affine(
    dataset_name: str,
    *,
    zarr_path: Path | str | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    dataset = dataset_name.upper()
    if dataset == "MERSCOPE":
        matrix = _load_merscope_transform_matrix(zarr_path)
        if matrix is not None:
            return _matrix_pixel_to_micron_affine(matrix)
        return (0.108, 0.0, 0.0), (0.0, 0.108, 0.0)

    if dataset == "XENIUM":
        mpp = _find_xenium_microns_per_pixel(zarr_path)
        if mpp is None:
            mpp = 0.2125
        return (float(mpp), 0.0, 0.0), (0.0, float(mpp), 0.0)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_merscope_transform_matrix(zarr_path: Path | str | None) -> np.ndarray | None:
    candidates = _sidecar_candidates(zarr_path, "micron_to_mosaic_pixel_transform.csv")
    for candidate in candidates:
        if not candidate.exists():
            continue
        matrix = np.loadtxt(candidate)
        if matrix.shape == (3, 3):
            return matrix
    return None


def _find_xenium_microns_per_pixel(zarr_path: Path | str | None) -> float | None:
    candidates = []
    candidates.extend(_sidecar_candidates(zarr_path, "experiment.xenium"))
    candidates.extend(_sidecar_candidates(zarr_path, "specs.json"))
    if zarr_path is not None:
        base = Path(zarr_path)
        candidates.extend(
            [
                base / "specs" / "specs.json",
                base.parent / "specs" / "specs.json",
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() in {".txt", ".csv"}:
            matrix = np.loadtxt(candidate)
            if matrix.shape == (3, 3) and float(matrix[0, 0]) != 0.0:
                return 1.0 / float(matrix[0, 0])
        try:
            data = json.loads(candidate.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "pixel_size" in data:
            return float(data["pixel_size"])
        if "microns_per_pixel" in data:
            return float(data["microns_per_pixel"])
    return None


def _sidecar_candidates(zarr_path: Path | str | None, name: str) -> list[Path]:
    if zarr_path is None:
        return []
    base = Path(zarr_path)
    return [
        base / name,
        base.parent / name,
        base.parent.parent / name,
    ]


def _matrix_pixel_to_micron_affine(
    micron_to_pixel_matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    inverse = np.linalg.inv(micron_to_pixel_matrix)
    x_transform = (float(inverse[0, 0]), float(inverse[0, 1]), float(inverse[0, 2]))
    y_transform = (float(inverse[1, 0]), float(inverse[1, 1]), float(inverse[1, 2]))
    return x_transform, y_transform


def _affine_um_to_px(
    x_um: float,
    y_um: float,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float]:
    matrix = np.array(
        [
            [float(x_transform[0]), float(x_transform[1])],
            [float(y_transform[0]), float(y_transform[1])],
        ],
        dtype=float,
    )
    offset = np.array([float(x_transform[2]), float(y_transform[2])], dtype=float)
    px, py = np.linalg.inv(matrix) @ (np.array([x_um, y_um], dtype=float) - offset)
    return float(px), float(py)


def _affine_px_to_um(
    x_px: float,
    y_px: float,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> tuple[float, float]:
    x_um = (
        float(x_transform[0]) * float(x_px)
        + float(x_transform[1]) * float(y_px)
        + float(x_transform[2])
    )
    y_um = (
        float(y_transform[0]) * float(x_px)
        + float(y_transform[1]) * float(y_px)
        + float(y_transform[2])
    )
    return float(x_um), float(y_um)


def _assignment_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    values = series.astype("string")
    bad = {"", "0", "-1", "nan", "None", "<NA>"}
    return values.notna() & ~values.isin(bad)


def _first_existing_col(df_like: Any, candidates: list[str]) -> str | None:
    cols = set(map(str, list(df_like.columns)))
    for col in candidates:
        if col in cols:
            return col
    return None


def _to_numpy(data_array: Any) -> np.ndarray:
    data = data_array.compute() if hasattr(data_array, "compute") else data_array
    return np.asarray(data)
