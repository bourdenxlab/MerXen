"""Prepare native spatial transcripts for pciSeq and attach cell typing results."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from shapely import STRtree
from shapely.affinity import affine_transform
from skimage.draw import polygon

from merxen.config import PciSeqConfig
from merxen.control_features import classify_control_transcripts
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    MERXEN_SCHEMA_VERSION,
    ORIGINAL_ID_NAMESPACE,
    canonical_instance_series,
    canonical_shape_instance_series,
    choose_primary_points_key,
)
from merxen.io.transcript_io import iter_points_chunks, resolve_col

_BRANCHES = {
    "reseg": "proseg",
    "original_seg": "original",
    "proseg_mask": "cellpose",
    "proseg_hybrid": "proseg_hybrid",
}
_FALLBACK_SHAPES = {
    "reseg": ("MOSAIK_proseg", "cell_boundaries"),
    "original_seg": ("xenium_cell_boundaries", "merscope_cell_boundaries"),
    "proseg_mask": ("MOSAIK_cellpose",),
    "proseg_hybrid": ("MOSAIK_proseg_hybrid",),
}
_PROBABILITY_SEMANTICS = (
    "pciSeq reported summaries retain classes with probability >0.001 and "
    "truncate probabilities to three decimals. Missing entries are recorded "
    "as zero in the reported matrix; they are not exact posterior zeros. "
    "Rows need not sum to one."
)


@dataclass
class PciSeqInputs:
    """Task-local model input together with stable MerXen identifier mappings."""

    spots: pd.DataFrame
    planes: list[sparse.coo_matrix]
    cell_id_map: pd.DataFrame
    spot_id_map: pd.DataFrame
    metadata: dict[str, Any]


def _positive_ids(values: Any, field: str) -> pd.Series:
    original = pd.Series(values, copy=False)
    numeric = pd.to_numeric(original, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{field} must contain finite positive integer identifiers")
    if (numeric != np.floor(numeric)).any():
        raise ValueError(f"{field} must contain integer identifiers")
    result = canonical_instance_series(numeric, field_name=field)
    if result.duplicated().any():
        raise ValueError(f"{field} contains duplicate identifiers")
    return result


def _table_ids(table: ad.AnnData) -> np.ndarray:
    attrs = table.uns.get("spatialdata_attrs", {})
    instance_key = attrs.get("instance_key", "instance_id")
    if instance_key not in table.obs:
        raise ValueError(
            "The clustered table needs canonical instance_id metadata; "
            "rerun enrichment and clustering for this segmentation"
        )
    return np.array(
        _positive_ids(table.obs[instance_key], "clustered table instances"),
        dtype=np.uint64,
    )


def _read_reference(path: Path) -> pd.DataFrame:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    if len(header) < 2 or len(set(header[1:])) != len(header[1:]):
        raise ValueError("Reference requires unique cell type column names")
    reference = pd.read_csv(path, sep=delimiter, index_col=0, dtype={header[0]: str})
    if reference.empty or reference.index.isna().any():
        raise ValueError("Reference requires gene rows and cell type columns")
    reference.index = reference.index.astype(str).str.strip()
    reference.columns = reference.columns.astype(str).str.strip()
    if (
        not reference.index.is_unique
        or not reference.columns.is_unique
        or (reference.index == "").any()
        or (reference.columns == "").any()
        or "Zero" in reference.columns
    ):
        raise ValueError(
            "Reference genes and classes must be nonempty and unique; "
            "Zero is reserved by pciSeq"
        )
    try:
        reference = reference.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Reference expression must be numeric") from error
    expression = reference.to_numpy()
    if not np.isfinite(expression).all() or (expression < 0).any():
        raise ValueError("Reference expression must be finite and nonnegative")
    if (expression.sum(axis=0) <= 0).any():
        raise ValueError("Each reference cell type needs positive expression")
    return reference


def _resolve_native_keys(
    sdata_obj: Any, config: PciSeqConfig
) -> tuple[str, str, str | None]:
    schema = getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {})
    registry = schema.get("segmentations", {})
    branch = _BRANCHES.get(config.segmentation, config.segmentation)
    entry = registry.get(branch, {})
    shape_key = config.shape_key or entry.get("shape")
    if shape_key is None:
        shape_key = next(
            (
                key
                for key in _FALLBACK_SHAPES.get(config.segmentation, ())
                if key in sdata_obj.shapes
            ),
            None,
        )
    points_key = config.points_key or entry.get("points")
    if points_key is None:
        points_key = choose_primary_points_key(sdata_obj)
    if not shape_key or shape_key not in sdata_obj.shapes:
        raise ValueError("pciSeq requires the selected native segmentation shapes")
    if not points_key or points_key not in sdata_obj.points:
        raise ValueError("pciSeq requires the selected native transcript points")
    if str(shape_key).endswith("_aligned_nonrigid") or str(points_key).endswith(
        "_aligned_nonrigid"
    ):
        raise ValueError("pciSeq must use paired native shapes and transcript points")
    if entry and (shape_key != entry.get("shape") or points_key != entry.get("points")):
        raise ValueError(
            "Explicit shape/points keys do not match the selected segmentation registry"
        )
    table_key = config.table_key or entry.get("table")
    return str(shape_key), str(points_key), table_key


def _affine(element: Any, coordinate_system: str, dimensions: int) -> np.ndarray:
    # Native MerXen coordinates are microns even when vendor shape metadata
    # maps them into an image pixel grid. Selecting intrinsic is explicit.
    if coordinate_system == "intrinsic":
        return np.eye(dimensions + 1)

    from spatialdata.transformations import get_transformation

    axes = ("x", "y", "z")[:dimensions]
    try:
        transform = get_transformation(element, to_coordinate_system=coordinate_system)
        matrix = np.asarray(
            transform.to_affine_matrix(input_axes=axes, output_axes=axes), dtype=float
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Cannot map pciSeq inputs into coordinate system {coordinate_system!r}"
        ) from error
    if (
        matrix.shape != (dimensions + 1, dimensions + 1)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[-1], np.eye(dimensions + 1)[-1])
        or abs(np.linalg.det(matrix[:-1, :-1])) < 1e-12
    ):
        raise ValueError("pciSeq requires a finite invertible spatial affine")
    return matrix


def _check_native_vendor_frame(
    sdata_obj: Any, config: PciSeqConfig, shape_key: str, points_key: str
) -> None:
    if (
        config.geometry_mode != "project_2d"
        or config.coordinate_system != "global"
        or config.segmentation != "original_seg"
    ):
        return
    schema = getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {})
    original = schema.get("segmentations", {}).get("original", {})
    if (
        schema.get("schema_version") != MERXEN_SCHEMA_VERSION
        or schema.get("platform") != config.platform
        or schema.get("primary_points") != points_key
        or original.get("points") != points_key
        or original.get("shape") != shape_key
        or original.get("id_namespace") != ORIGINAL_ID_NAMESPACE
        or original.get("coordinate_variant_of") is not None
        or shape_key != f"{config.platform.lower()}_cell_boundaries"
    ):
        return
    point_matrix = _affine(sdata_obj.points[points_key], "global", 2)
    shape_matrix = _affine(sdata_obj.shapes[shape_key], "global", 2)
    if np.allclose(point_matrix, np.eye(3)) and not np.allclose(
        shape_matrix, point_matrix
    ):
        raise ValueError(
            "Registered MerXen native vendor boundaries and transcript points "
            "declare different 'global' transforms. Vendor boundaries can retain "
            "a micron-to-image-pixel transform while native points remain in "
            "microns. Select coordinate_system='intrinsic' "
            "(Nextflow: pciseq_coordinate_system=intrinsic) when both stored "
            "coordinates are native microns, or select a verified shared named "
            "physical coordinate system. Refusing to rescale this segmentation "
            "against unchanged transcript coordinates."
        )


def _load_spots(
    points_obj: Any,
    *,
    config: PciSeqConfig,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], np.ndarray]:
    axes = ["x", "y"]
    candidates = [["x", "global_x", "x_location"], ["y", "global_y", "y_location"]]
    if config.geometry_mode == "labels_3d":
        axes.append("z")
        candidates.append(["z", "global_z", "z_location"])
    coord_cols: list[str] = []
    for options in candidates:
        coordinate = resolve_col(points_obj, options)
        assert coordinate is not None
        coord_cols.append(coordinate)
    gene_col = resolve_col(points_obj, ["gene", "feature_name", "target", "gene_name"])
    assert gene_col is not None
    if "transcript_id" not in points_obj.columns:
        raise ValueError("Native points need stable transcript_id; rerun enrichment")
    matrix = _affine(points_obj, config.coordinate_system, len(axes))
    columns = [*coord_cols, gene_col, "transcript_id"]
    columns.extend(
        name for name in ("is_gene", "codeword_category") if name in points_obj.columns
    )
    parts: list[pd.DataFrame] = []
    counts = {
        "input_spots": 0,
        "invalid_spots": 0,
        "control_spots": 0,
        "excluded_spots": 0,
        "unmatched_gene_spots": 0,
    }
    excluded = config.options.get("exclude_genes", [])
    for chunk in iter_points_chunks(points_obj, columns, desc="pciSeq transcripts"):
        counts["input_spots"] += len(chunk)
        coordinates = chunk[coord_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        coordinates = coordinates @ matrix[:-1, :-1].T + matrix[:-1, -1]
        genes = chunk[gene_col].astype("string").str.strip()
        valid = np.isfinite(coordinates).all(axis=1) & genes.notna().to_numpy()
        valid &= genes.fillna("").ne("").to_numpy(dtype=bool)
        counts["invalid_spots"] += int((~valid).sum())
        controls = classify_control_transcripts(
            genes.fillna("").to_numpy(dtype=str),
            platform=config.platform,
            is_gene=chunk.get("is_gene"),
            codeword_category=chunk.get("codeword_category"),
        ).control
        counts["control_spots"] += int((valid & controls).sum())
        valid &= ~controls
        exclude_mask = genes.isin(excluded).to_numpy()
        counts["excluded_spots"] += int((valid & exclude_mask).sum())
        valid &= ~exclude_mask
        matched = genes.isin(reference.index).to_numpy()
        counts["unmatched_gene_spots"] += int((valid & ~matched).sum())
        valid &= matched
        if not valid.any():
            continue
        part = pd.DataFrame(coordinates[valid], columns=axes)
        part["gene_name"] = genes.iloc[np.flatnonzero(valid)].to_numpy(dtype=str)
        part["transcript_id"] = chunk["transcript_id"].to_numpy()[valid]
        parts.append(part)
    if not parts:
        raise ValueError("No non-control transcripts overlap the reference genes")
    spots = pd.concat(parts, ignore_index=True)
    spots["transcript_id"] = _positive_ids(
        spots["transcript_id"], "native transcript_id"
    ).to_numpy(dtype=np.uint64)
    return spots, counts, matrix


def _project_shapes(
    shapes: Any, matrix: np.ndarray, overlap_policy: str
) -> tuple[list[Any], np.ndarray, list[Any]]:
    ids = canonical_shape_instance_series(shapes, field_name="pciSeq shapes")
    ids = _positive_ids(ids, "pciSeq shape instances").to_numpy(dtype=np.uint64)
    coefficients = [
        matrix[0, 0],
        matrix[0, 1],
        matrix[1, 0],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    ]
    order = np.argsort(ids)
    geometries = [shapes.geometry.iloc[i] for i in order]
    if any(geometry is None for geometry in geometries):
        raise ValueError("pciSeq requires nonempty valid cell polygons")
    geometries = [affine_transform(geometry, coefficients) for geometry in geometries]
    for geometry in geometries:
        if (
            geometry is None
            or geometry.is_empty
            or not geometry.is_valid
            or geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not np.isfinite(geometry.bounds).all()
        ):
            raise ValueError("pciSeq requires nonempty valid cell polygons")
    tree = STRtree(geometries)
    overlaps: list[Any] = []
    for index, geometry in enumerate(geometries):
        for other in tree.query(geometry, predicate="intersects"):
            if other <= index:
                continue
            intersection = geometry.intersection(geometries[other])
            if intersection.area <= 1e-9:
                continue
            if overlap_policy == "error":
                raise ValueError(
                    "pciSeq overlap_policy='error' requires nonoverlapping polygons; "
                    f"instances {int(ids[order[index]])} and "
                    f"{int(ids[order[other]])} overlap"
                )
            overlaps.append(intersection)
    return geometries, ids[order], overlaps


def _polygon_parts(geometry: Any) -> list[Any]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    return (
        [part for piece in geometry.geoms for part in _polygon_parts(piece)]
        if hasattr(geometry, "geoms")
        else []
    )


def _local_geometry_mask(
    geometry: Any,
    *,
    origin: np.ndarray,
    pixel_size: float,
    width: int,
    height: int,
) -> tuple[int, int, np.ndarray]:
    minx, miny, maxx, maxy = geometry.bounds
    x0 = max(0, int(np.floor((minx - origin[0]) / pixel_size)))
    y0 = max(0, int(np.floor((miny - origin[1]) / pixel_size)))
    x1 = min(width, int(np.ceil((maxx - origin[0]) / pixel_size)) + 1)
    y1 = min(height, int(np.ceil((maxy - origin[1]) / pixel_size)) + 1)
    local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for piece in _polygon_parts(geometry):
        rings = [(piece.exterior, True), *((ring, False) for ring in piece.interiors)]
        for ring, fill in rings:
            coords = np.asarray(ring.coords)
            rows = (coords[:, 1] - origin[1]) / pixel_size - y0
            cols = (coords[:, 0] - origin[0]) / pixel_size - x0
            rr, cc = polygon(rows, cols, shape=local.shape)
            local[rr, cc] = fill
    return x0, y0, local


def _rasterize(
    geometries: list[Any],
    ids: np.ndarray,
    spots: pd.DataFrame,
    pixel_size: float,
    max_pixels: int,
    overlaps: list[Any],
) -> tuple[sparse.coo_matrix, list[float], int, int]:
    bounds = np.asarray([geometry.bounds for geometry in geometries])
    xy = spots[["x", "y"]].to_numpy()
    minimum = np.minimum(bounds[:, :2].min(axis=0), xy.min(axis=0))
    maximum = np.maximum(bounds[:, 2:].max(axis=0), xy.max(axis=0))
    origin = np.floor(minimum / pixel_size) * pixel_size - pixel_size
    size_xy = np.ceil((maximum - origin) / pixel_size) + 2
    if not np.isfinite(size_xy).all() or (size_xy > max_pixels).any():
        raise ValueError(
            "pciSeq raster dimensions exceed max_pixels; check coordinate outliers "
            "or increase pixel_size_um"
        )
    width, height = map(int, size_xy)
    if width * height > max_pixels:
        raise ValueError(
            f"pciSeq raster needs {width * height:,} pixels, above max_pixels="
            f"{max_pixels:,}; increase pixel_size_um or max_pixels"
        )
    mask = np.zeros((height, width), dtype=np.uint32)
    # A hybrid segmentation can deliberately overlap. Ambiguous prior pixels
    # remain background so the model, rather than polygon order, assigns spots.
    ambiguous = np.zeros((height, width), dtype=bool) if overlaps else None
    for geometry in overlaps:
        x0, y0, local = _local_geometry_mask(
            geometry, origin=origin, pixel_size=pixel_size, width=width, height=height
        )
        assert ambiguous is not None
        ambiguous[y0 : y0 + local.shape[0], x0 : x0 + local.shape[1]] |= local
    boundary_ties = 0
    for label, geometry in enumerate(geometries, start=1):
        x0, y0, local = _local_geometry_mask(
            geometry, origin=origin, pixel_size=pixel_size, width=width, height=height
        )
        y1, x1 = y0 + local.shape[0], x0 + local.shape[1]
        view = mask[y0:y1, x0:x1]
        if ambiguous is not None:
            local &= ~ambiguous[y0:y1, x0:x1]
        boundary_ties += int((local & (view > 0)).sum())
        available = local & (view == 0)
        if not available.any():
            raise ValueError(
                f"Instance {int(ids[label - 1])} has no unambiguous seed pixels; "
                "reduce pixel_size_um or resolve fully overlapping polygons"
            )
        view[available] = label
    spots.loc[:, ["x", "y"]] = (xy - origin) / pixel_size
    ambiguous_pixels = int(ambiguous.sum()) if ambiguous is not None else 0
    return sparse.coo_matrix(mask), origin.tolist(), boundary_ties, ambiguous_pixels


def prepare_pciseq_inputs(
    sdata_obj: Any, *, config: PciSeqConfig, reference: pd.DataFrame
) -> PciSeqInputs:
    """Build native-coordinate model arrays with explicit stable ID mappings.

    Args:
        sdata_obj: Open MerXen SpatialData object.
        config: One platform and segmentation's pciSeq settings.
        reference: Validated genes-by-cell-types mean expression matrix.

    Returns:
        Sparse masks, filtered model coordinates, ID maps, and provenance.
    """
    shape_key, points_key, table_key = _resolve_native_keys(sdata_obj, config)
    shapes = sdata_obj.shapes[shape_key]
    _check_native_vendor_frame(sdata_obj, config, shape_key, points_key)
    spots, counts, point_matrix = _load_spots(
        sdata_obj.points[points_key], config=config, reference=reference
    )
    shape_ids = _positive_ids(
        canonical_shape_instance_series(shapes), "pciSeq shape instances"
    ).to_numpy(dtype=np.uint64)
    if not len(shape_ids):
        raise ValueError("pciSeq requires at least one segmented cell")
    metadata: dict[str, Any] = {
        **counts,
        "shape_key": shape_key,
        "points_key": points_key,
        "source_table_key": table_key,
        "coordinate_system": config.coordinate_system,
        "coordinate_policy": (
            "intrinsic_native_microns"
            if config.coordinate_system == "intrinsic"
            else "declared_element_transforms"
        ),
        "point_affine": point_matrix.tolist(),
        "geometry_mode": config.geometry_mode,
    }
    if config.geometry_mode == "project_2d":
        shape_matrix = _affine(shapes, config.coordinate_system, 2)
        geometries, instance_ids, overlaps = _project_shapes(
            shapes, shape_matrix, config.overlap_policy
        )
        plane, origin, ties, ambiguous_pixels = _rasterize(
            geometries,
            instance_ids,
            spots,
            config.pixel_size_um,
            config.max_pixels,
            overlaps,
        )
        planes = [plane]
        voxel_size = [config.pixel_size_um, config.pixel_size_um, 1.0]
        metadata.update(
            shape_affine=shape_matrix.tolist(),
            origin_um=[*origin, 0.0],
            raster_boundary_ties=ties,
            overlap_policy=config.overlap_policy,
            geometric_overlap_pair_count=len(overlaps),
            ambiguous_raster_pixels=ambiguous_pixels,
            projection="Measured x/y projected onto native 2D cell polygons; z unused",
            out_of_bounds_spots=0,
        )
    else:
        assert config.labels_path is not None
        volume = np.load(config.labels_path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 3 or volume.shape[0] < 2:
            raise ValueError(
                "labels_3d requires a genuine multi-plane (z,y,x) label NPY"
            )
        if np.prod(volume.shape, dtype=object) > config.max_pixels:
            raise ValueError("3D label volume exceeds max_pixels")
        if volume.dtype.kind not in {"u", "i"} or (volume < 0).any():
            raise ValueError("3D labels must be nonnegative integer canonical IDs")
        present = np.unique(volume)
        instance_ids = present[present > 0].astype(np.uint64)
        if set(map(int, instance_ids)) != set(map(int, shape_ids)):
            raise ValueError(
                "3D label IDs must exactly match selected segmentation IDs"
            )
        planes = []
        for plane in volume:
            rows, cols = np.nonzero(plane)
            labels = np.searchsorted(instance_ids, plane[rows, cols]) + 1
            planes.append(
                sparse.coo_matrix(
                    (labels.astype(np.uint32), (rows, cols)), shape=plane.shape
                )
            )
        origin_xyz = np.asarray(config.origin_um, dtype=float)
        voxel_size = list(config.voxel_size_um)
        xyz = (spots[["x", "y", "z"]].to_numpy() - origin_xyz) / voxel_size
        maxima = np.asarray(
            [volume.shape[2] - 1, volume.shape[1] - 1, volume.shape[0] - 1]
        )
        valid = ((xyz >= 0) & (xyz <= maxima)).all(axis=1)
        metadata.update(
            origin_um=origin_xyz.tolist(),
            out_of_bounds_spots=int((~valid).sum()),
            labels_shape=list(volume.shape),
        )
        spots = spots.loc[valid].copy().reset_index(drop=True)
        spots.loc[:, ["x", "y", "z"]] = xyz[valid]
        spots = spots.rename(columns={"z": "z_plane"})
    if spots.empty:
        raise ValueError("No pciSeq transcripts remain inside the label volume")
    spots = spots.reset_index(drop=True)
    spot_map = pd.DataFrame(
        {
            "spot_index": np.arange(len(spots), dtype=np.int64),
            "transcript_id": spots.pop("transcript_id").to_numpy(dtype=np.uint64),
        }
    )
    metadata.update(
        voxel_size=voxel_size,
        cell_count=len(instance_ids),
        spot_count=len(spots),
        raster_shape=list(planes[0].shape),
    )
    return PciSeqInputs(
        spots=spots,
        planes=planes,
        cell_id_map=pd.DataFrame(
            {
                "label": np.arange(1, len(instance_ids) + 1, dtype=np.uint32),
                "instance_id": instance_ids,
            }
        ),
        spot_id_map=spot_map,
        metadata=metadata,
    )


def prepare_pciseq(config: PciSeqConfig, output_dir: Path) -> Path:
    """Export one clustered sample's native model inputs for isolated compute.

    Args:
        config: Validated per-sample settings.
        output_dir: Destination for a self-contained prepared input directory.

    Returns:
        Path to the prepared manifest.
    """
    import spatialdata as sd

    output_dir = Path(output_dir)
    reference = _read_reference(config.reference_path)
    with config.reference_path.open("rb") as handle:
        reference_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    reference_origin = None
    if config.reference_manifest_path is not None:
        reference_origin = json.loads(config.reference_manifest_path.read_text())
        if not isinstance(reference_origin, dict):
            raise ValueError("The prepared reference manifest must be a JSON object")
        if reference_origin.get("reference_sha256") != reference_sha256:
            raise ValueError(
                "The reference manifest checksum does not match the supplied matrix"
            )
    source = ad.read_h5ad(config.source_h5ad)
    source_ids = _table_ids(source)
    sdata_obj = sd.read_zarr(config.spatialdata_path)
    inputs = prepare_pciseq_inputs(sdata_obj, config=config, reference=reference)
    observed_genes = inputs.spots["gene_name"].unique()
    unsupported_classes = reference.columns[
        reference.loc[observed_genes].sum(axis=0) <= 0
    ].tolist()
    if unsupported_classes:
        raise ValueError(
            "Reference cell types need positive expression over retained spot genes: "
            + ", ".join(unsupported_classes)
        )
    if not set(map(int, source_ids)).issubset(
        set(map(int, inputs.cell_id_map.instance_id))
    ):
        raise ValueError(
            "Clustered cell IDs do not belong to the selected segmentation"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    for old_plane in mask_dir.glob("plane_*.npz"):
        old_plane.unlink()
    for index, plane in enumerate(inputs.planes):
        sparse.save_npz(mask_dir / f"plane_{index:04d}.npz", plane)
    inputs.spots.to_csv(output_dir / "spots.csv", index=False)
    inputs.cell_id_map.to_csv(output_dir / "cell_id_map.csv", index=False)
    inputs.spot_id_map.to_csv(output_dir / "spot_id_map.csv", index=False)
    reference.to_csv(output_dir / "reference.csv")
    source.write_h5ad(output_dir / "source.h5ad")
    options = {
        "exclude_genes": [],
        "max_iter": config.max_iter,
        "nNeighbors": config.n_neighbors,
        **config.options,
    }
    manifest = {
        **inputs.metadata,
        "pair_id": config.pair_id,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "options": options,
        "reference_classes": reference.columns.tolist(),
        "reference_gene_count": len(reference),
        "reference_sha256": reference_sha256,
        "reference_path": str(config.reference_path.resolve()),
        "source_h5ad": str(config.source_h5ad.resolve()),
        "spatialdata_path": str(config.spatialdata_path.resolve()),
        "clustered_cell_count": source.n_obs,
    }
    if config.reference_manifest_path is not None:
        manifest["reference_preparation"] = reference_origin
        shutil.copy2(
            config.reference_manifest_path, output_dir / "reference_manifest.json"
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _read_id_maps(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_map = pd.read_csv(
        prepared_dir / "cell_id_map.csv",
        dtype={"label": "uint32", "instance_id": "uint64"},
    )
    spot_map = pd.read_csv(
        prepared_dir / "spot_id_map.csv",
        dtype={"spot_index": "int64", "transcript_id": "uint64"},
    )
    _positive_ids(cell_map.instance_id, "cell map instances")
    _positive_ids(cell_map.label, "cell map labels")
    _positive_ids(spot_map.transcript_id, "spot map transcript_id")
    if not np.array_equal(np.sort(cell_map.label), np.arange(1, len(cell_map) + 1)):
        raise ValueError("Prepared cell map labels must be sequential")
    if not np.array_equal(np.sort(spot_map.spot_index), np.arange(len(spot_map))):
        raise ValueError("Prepared spot map indices must be sequential")
    return cell_map, spot_map


def _summarize_cells(
    cells: pd.DataFrame, cell_map: pd.DataFrame, classes: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    if not {"Cell_Num", "ClassName", "Prob"}.issubset(cells):
        raise ValueError("pciSeq cell output lacks required columns")
    labels = _positive_ids(cells.Cell_Num, "pciSeq Cell_Num")
    if set(labels) != set(cell_map.label):
        raise ValueError("pciSeq output changed the prepared cell identifiers")
    probabilities = np.zeros((len(cells), len(classes)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(classes)}
    rows: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(cells.iterrows()):
        names, values = row.ClassName, row.Prob
        if (
            not isinstance(names, list)
            or not isinstance(values, list)
            or len(names) != len(values)
        ):
            raise ValueError("pciSeq returned invalid class probability summaries")
        values_array = np.asarray(values, dtype=float)
        if (
            len(set(names)) != len(names)
            or not set(names).issubset(lookup)
            or not np.isfinite(values_array).all()
            or (values_array < 0).any()
            or (values_array > 1).any()
        ):
            raise ValueError("pciSeq returned invalid class probabilities")
        for name, value in zip(names, values_array, strict=True):
            probabilities[row_index, lookup[name]] = value
        best = int(np.argmax(values_array)) if len(values_array) else None
        cell_type = names[best] if best is not None else "Uncertain"
        rows.append(
            {
                "label": int(row.Cell_Num),
                "pciseq_cell_type": cell_type,
                "pciseq_probability": float(values_array[best])
                if best is not None
                else np.nan,
                "pciseq_status": "uncertain"
                if best is None
                else ("zero_expression" if cell_type == "Zero" else "assigned"),
                "reported_class_names": json.dumps(names),
                "reported_probabilities": json.dumps(values),
            }
        )
    assignments = pd.DataFrame(rows).merge(
        cell_map, on="label", how="left", validate="one_to_one"
    )
    return assignments, probabilities


def _summarize_spots(
    genes: pd.DataFrame, spot_map: pd.DataFrame, cell_map: pd.DataFrame
) -> pd.DataFrame:
    if not {"spot_id", "neighbour", "gene_name"}.issubset(genes):
        raise ValueError("pciSeq spot output lacks required columns")
    if genes.spot_id.duplicated().any() or set(genes.spot_id) != set(
        spot_map.spot_index
    ):
        raise ValueError("pciSeq output changed the prepared transcript identifiers")
    mapping = {
        int(label): int(instance)
        for label, instance in zip(cell_map.label, cell_map.instance_id, strict=True)
    }
    if not genes.neighbour.isin([0, *mapping]).all():
        raise ValueError("pciSeq returned an unknown assigned cell")
    result = genes.copy()
    result["instance_id"] = pd.array(
        [mapping.get(int(label)) for label in genes.neighbour], dtype="UInt64"
    )
    result = result.merge(
        spot_map, left_on="spot_id", right_on="spot_index", validate="one_to_one"
    )
    for column in result:
        if result[column].map(lambda value: isinstance(value, (list, dict))).any():
            result[column] = result[column].map(
                lambda value: (
                    json.dumps(value) if isinstance(value, (list, dict)) else value
                )
            )
    return result


def _write_derived_table(
    config: PciSeqConfig, annotated: ad.AnnData, metadata: dict[str, Any]
) -> str:
    import spatialdata as sd
    from spatialdata.models import TableModel

    from merxen.io.spatialdata_io import (
        spatialdata_write_lock,
        write_or_replace_element,
    )

    attrs = dict(annotated.uns.get("spatialdata_attrs", {}))
    region = attrs.get("region")
    if isinstance(region, (list, tuple, np.ndarray)):
        region = region[0] if len(region) == 1 else None
    if not isinstance(region, str) or not region:
        raise ValueError("Annotated table requires exactly one SpatialData region")
    table_key = str(
        metadata.get("source_table_key")
        or config.table_key
        or f"table_{config.segmentation}"
    )
    table_key = f"{table_key}_pciseq"
    table = annotated.copy()
    table.uns.pop("spatialdata_attrs", None)
    table = TableModel.parse(
        table,
        region=region,
        region_key=attrs.get("region_key", "region"),
        instance_key=attrs.get("instance_key", "instance_id"),
    )
    with spatialdata_write_lock(config.spatialdata_path):
        sdata_obj = sd.read_zarr(config.spatialdata_path)
        if region not in sdata_obj.shapes and region not in sdata_obj.labels:
            raise ValueError("Annotated table's linked SpatialData region is missing")
        write_or_replace_element(sdata_obj, table_key, "tables", table, overwrite=True)
    return table_key


def finalize_pciseq(
    config: PciSeqConfig,
    prepared_dir: Path,
    computed_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Map reported pciSeq results to cells and preserve clustered annotations.

    Args:
        config: Validated per-sample settings.
        prepared_dir: Prepared arrays and stable ID maps.
        computed_dir: Raw JSON results from the isolated pciSeq worker.
        output_dir: Destination for annotated data and reviewable assignments.

    Returns:
        Paths to the annotated H5AD, assignments, and provenance summary.
    """
    prepared_dir, computed_dir, output_dir = map(
        Path, (prepared_dir, computed_dir, output_dir)
    )
    metadata = json.loads((prepared_dir / "manifest.json").read_text())
    compute = json.loads((computed_dir / "compute_manifest.json").read_text())
    for field in ("pair_id", "sample_id", "platform", "segmentation"):
        if metadata.get(field) != getattr(config, field):
            raise ValueError(
                f"Prepared pciSeq {field} does not match finalization config"
            )
    fingerprint = compute.get("prepared_manifest_sha256")
    if fingerprint is not None:
        with (prepared_dir / "manifest.json").open("rb") as handle:
            actual_fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
        if fingerprint != actual_fingerprint:
            raise ValueError(
                "Computed pciSeq results belong to different prepared inputs"
            )
    cell_map, spot_map = _read_id_maps(prepared_dir)
    cells = pd.DataFrame(json.loads((computed_dir / "cell_data.json").read_text()))
    genes = pd.DataFrame(json.loads((computed_dir / "gene_data.json").read_text()))
    classes = [*metadata["reference_classes"], "Zero"]
    assignments, probabilities = _summarize_cells(cells, cell_map, classes)
    spot_assignments = _summarize_spots(genes, spot_map, cell_map)
    prepared_genes = pd.read_csv(
        prepared_dir / "spots.csv", usecols=["gene_name"], dtype={"gene_name": str}
    )["gene_name"]
    reported_genes = genes.set_index("spot_id")["gene_name"].reindex(
        np.arange(len(prepared_genes))
    )
    if not np.array_equal(prepared_genes.to_numpy(), reported_genes.to_numpy()):
        raise ValueError("pciSeq output changed the prepared transcript gene mapping")
    annotated = ad.read_h5ad(prepared_dir / "source.h5ad")
    ids = _table_ids(annotated)
    positions = pd.Index(assignments.instance_id).get_indexer(ids)
    if (positions < 0).any():
        raise ValueError("pciSeq results do not cover all clustered cell identifiers")
    for column in ("pciseq_cell_type", "pciseq_probability", "pciseq_status"):
        annotated.obs[column] = assignments.iloc[positions][column].to_numpy()
    annotated.obsm["pciseq_reported_probabilities"] = probabilities[positions]
    annotated.uns["merxen_pciseq"] = {
        "geometry_mode": metadata["geometry_mode"],
        "coordinate_system": metadata["coordinate_system"],
        "shape_key": metadata["shape_key"],
        "points_key": metadata["points_key"],
        "reported_probability_classes": classes,
        "probability_semantics": _PROBABILITY_SEMANTICS,
        "prepared_manifest_json": json.dumps(metadata),
        "compute_manifest_json": json.dumps(compute),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.h5ad"
    annotated.write_h5ad(annotated_path)
    cell_path, spot_path = (
        output_dir / "cell_assignments.csv",
        output_dir / "spot_assignments.csv",
    )
    assignments.to_csv(cell_path, index=False)
    spot_assignments.to_csv(spot_path, index=False)
    for filename in ("cell_data.json", "gene_data.json", "compute_manifest.json"):
        shutil.copy2(computed_dir / filename, output_dir / filename)
    for filename in ("manifest.json", "cell_id_map.csv", "spot_id_map.csv"):
        shutil.copy2(prepared_dir / filename, output_dir / filename)
    reference_manifest = prepared_dir / "reference_manifest.json"
    if reference_manifest.exists():
        shutil.copy2(reference_manifest, output_dir / "reference_manifest.json")
    summary = {
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "annotated_cells": annotated.n_obs,
        "model_cells": len(cell_map),
        "model_spots": len(spot_map),
        "probability_semantics": _PROBABILITY_SEMANTICS,
        "compute": compute,
    }
    if config.write_spatialdata_table:
        summary["spatialdata_table_key"] = _write_derived_table(
            config, annotated, metadata
        )
    summary_path = output_dir / "pciseq_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "annotated_h5ad": annotated_path,
        "cell_assignments": cell_path,
        "spot_assignments": spot_path,
        "summary": summary_path,
    }
