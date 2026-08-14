"""Annotation-derived tissue support for VALIS registration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from scipy import ndimage as ndi
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from skimage.draw import polygon as draw_polygon

from merxen.alignment.transforms import apply_affine_matrix
from merxen.cortical_depth.boundaries import load_boundary_annotations
from merxen.cortical_depth.tissue import build_full_tissue_polygon


@dataclass(frozen=True)
class AlignmentTissueAnnotation:
    """Validated full-tissue geometry and source provenance for one platform."""

    platform: str
    path: Path
    sha256: str
    polygon: Polygon | MultiPolygon
    pial_piece_count: int
    exclusion_count: int

    def metadata(self: Self) -> dict[str, Any]:
        """Return JSON-safe annotation provenance."""
        return {
            "source": "manual_annotation",
            "platform": self.platform,
            "path": str(self.path),
            "sha256": self.sha256,
            "pial_piece_count": int(self.pial_piece_count),
            "exclusion_count": int(self.exclusion_count),
            "geometry_type": self.polygon.geom_type,
            "area_dataset_um2": float(self.polygon.area),
            "bounds_dataset_um": [float(value) for value in self.polygon.bounds],
        }


@dataclass(frozen=True)
class RasterizedTissueMask:
    """Exact annotation raster before and after acquisition-support clipping."""

    unclipped: np.ndarray
    tissue: np.ndarray
    metadata: dict[str, Any]


def annotation_sha256(path: Path | str) -> str:
    """Return a stable content digest for an annotation file."""
    annotation_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with annotation_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_alignment_tissue_annotation(
    path: Path | str,
    *,
    platform: str,
) -> AlignmentTissueAnnotation:
    """Load and validate the required manual VALIS tissue annotation."""
    normalized_platform = str(platform).upper()
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"{normalized_platform} VALIS tissue annotation is missing: "
            f"{annotation_path}"
        )
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"{normalized_platform} VALIS tissue annotation is not a file: "
            f"{annotation_path}"
        )
    try:
        annotations = load_boundary_annotations(
            annotation_path=annotation_path,
            smoothing_window=0,
            tissue_mask_only=True,
        )
        polygon = build_full_tissue_polygon(
            annotations,
            require_tissue_edge=True,
            allow_ribbon_fallback=False,
        )
    except (OSError, TypeError, ValueError, GEOSException) as exc:
        raise ValueError(
            f"Invalid {normalized_platform} VALIS tissue annotation "
            f"{annotation_path}: {exc}"
        ) from exc
    return AlignmentTissueAnnotation(
        platform=normalized_platform,
        path=annotation_path,
        sha256=annotation_sha256(annotation_path),
        polygon=polygon,
        pial_piece_count=len(annotations.pieces),
        exclusion_count=len(annotations.exclusions),
    )


def rasterize_alignment_tissue_annotation(
    annotation: AlignmentTissueAnnotation,
    *,
    dataset_to_registration_matrix: np.ndarray,
    shape_rc: tuple[int, int],
    acquired_support_mask: Any,
    registration_pixel_size_um: float,
) -> RasterizedTissueMask:
    """Rasterize exact annotation geometry and clip only to acquired support."""
    matrix = np.asarray(dataset_to_registration_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("dataset_to_registration_matrix must be a finite 3x3 matrix")
    height, width = (int(shape_rc[0]), int(shape_rc[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Registration raster shape must be positive, got {shape_rc}")
    support = np.asarray(acquired_support_mask) > 0
    if support.shape != (height, width):
        raise ValueError(
            "Acquired support and tissue raster shapes differ: "
            f"{support.shape} vs {(height, width)}"
        )

    raster = np.zeros((height, width), dtype=bool)
    transformed_interiors: list[np.ndarray] = []
    for polygon in _polygon_parts(annotation.polygon):
        exterior_xy = apply_affine_matrix(
            np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2],
            matrix,
        )
        rows, cols = draw_polygon(
            exterior_xy[:, 1],
            exterior_xy[:, 0],
            shape=raster.shape,
        )
        raster[rows, cols] = True
        transformed_interiors.extend(
            apply_affine_matrix(
                np.asarray(interior.coords, dtype=np.float64)[:, :2],
                matrix,
            )
            for interior in polygon.interiors
        )
    for interior_xy in transformed_interiors:
        rows, cols = draw_polygon(
            interior_xy[:, 1],
            interior_xy[:, 0],
            shape=raster.shape,
        )
        raster[rows, cols] = False

    unclipped = np.asarray(raster.astype(np.uint8) * 255, dtype=np.uint8)
    tissue_binary = raster & support
    if not np.any(tissue_binary):
        raise ValueError(
            f"{annotation.platform} tissue annotation has no overlap with the "
            "acquired DAPI support; check its coordinate system."
        )
    tissue = np.asarray(tissue_binary.astype(np.uint8) * 255, dtype=np.uint8)
    _, component_count = ndi.label(tissue_binary)
    unclipped_count = int(np.count_nonzero(raster))
    clipped_count = int(np.count_nonzero(raster & ~support))
    metadata = annotation.metadata() | {
        "raster_shape_rc": [height, width],
        "registration_pixel_size_um": float(registration_pixel_size_um),
        "raster_component_count": int(component_count),
        "raster_area_um2": float(np.count_nonzero(tissue_binary))
        * float(registration_pixel_size_um) ** 2,
        "pixels_before_support_clip": unclipped_count,
        "pixels_after_support_clip": int(np.count_nonzero(tissue_binary)),
        "fraction_clipped_by_acquired_support": float(
            clipped_count / max(1, unclipped_count)
        ),
        "morphology_applied": False,
    }
    return RasterizedTissueMask(
        unclipped=unclipped,
        tissue=tissue,
        metadata=metadata,
    )


def _polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise TypeError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
