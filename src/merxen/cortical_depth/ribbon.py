"""Cortical ribbon construction and rasterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from merxen.cortical_depth.boundaries import BoundaryAnnotations


@dataclass(frozen=True)
class RasterSpec:
    """Mapping between source coordinates and raster indices."""

    x_min: float
    y_min: float
    width: int
    height: int
    step: float
    resolution_um: float
    coordinate_unit_um: float

    @property
    def x_centers(self: Self) -> np.ndarray:
        """Grid x coordinates at pixel centers."""
        return self.x_min + (np.arange(self.width, dtype=float) + 0.5) * self.step

    @property
    def y_centers(self: Self) -> np.ndarray:
        """Grid y coordinates at pixel centers."""
        return self.y_min + (np.arange(self.height, dtype=float) + 0.5) * self.step

    def points_to_indices(
        self: Self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert source coordinates to nearest integer row/column indices."""
        arr = np.asarray(points, dtype=float)
        cols = np.floor((arr[:, 0] - self.x_min) / self.step).astype(int)
        rows = np.floor((arr[:, 1] - self.y_min) / self.step).astype(int)
        return rows, cols

    def points_to_fractional_indices(
        self: Self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert source coordinates to fractional row/column pixel centers."""
        arr = np.asarray(points, dtype=float)
        cols = (arr[:, 0] - self.x_min) / self.step - 0.5
        rows = (arr[:, 1] - self.y_min) / self.step - 0.5
        return rows, cols

    def indices_to_points(
        self: Self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Convert row/column indices to source coordinates at pixel centers."""
        x = self.x_min + (np.asarray(cols, dtype=float) + 0.5) * self.step
        y = self.y_min + (np.asarray(rows, dtype=float) + 0.5) * self.step
        return np.column_stack([x, y])


@dataclass(frozen=True)
class RibbonGrid:
    """Rasterized cortical ribbon and boundary masks."""

    mask: np.ndarray
    pial_boundary: np.ndarray
    wm_boundary: np.ndarray
    side_boundary: np.ndarray
    spec: RasterSpec
    polygon: Polygon | MultiPolygon
    pial_line: LineString
    wm_line: LineString
    side_lines: tuple[LineString, ...]

    @property
    def shape(self: Self) -> tuple[int, int]:
        """Raster shape as ``(height, width)``."""
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))


def build_cortical_ribbon_polygon(
    annotations: BoundaryAnnotations,
) -> tuple[Polygon | MultiPolygon, tuple[LineString, ...]]:
    """Build a ribbon polygon from annotation geometry.

    If a complete ribbon polygon was supplied it is used directly. Otherwise,
    the pial boundary and gray/white boundary are connected at their endpoints;
    those connecting segments are treated as artificial side boundaries.
    """
    pial, wm = orient_boundary_pair(annotations.pial, annotations.wm)
    if annotations.ribbon is not None:
        polygon: BaseGeometry = annotations.ribbon
    else:
        pial_coords = list(pial.coords)
        wm_coords = list(wm.coords)
        polygon = Polygon(pial_coords + list(reversed(wm_coords)))

    if annotations.exclusions:
        polygon = polygon.difference(MultiPolygon(list(annotations.exclusions)))

    if polygon.is_empty:
        raise ValueError("Cortical ribbon polygon is empty after exclusions.")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon | MultiPolygon) or polygon.is_empty:
        raise ValueError("Could not construct a valid cortical ribbon polygon.")

    side_lines = tuple(annotations.side_boundaries) or (
        LineString([pial.coords[0], wm.coords[0]]),
        LineString([pial.coords[-1], wm.coords[-1]]),
    )
    return polygon, side_lines


def orient_boundary_pair(
    pial: LineString,
    wm: LineString,
) -> tuple[LineString, LineString]:
    """Orient two boundary lines so their endpoints correspond."""
    pial_coords = np.asarray(pial.coords, dtype=float)[:, :2]
    wm_coords = np.asarray(wm.coords, dtype=float)[:, :2]
    same = np.linalg.norm(pial_coords[0] - wm_coords[0]) + np.linalg.norm(
        pial_coords[-1] - wm_coords[-1]
    )
    flipped = np.linalg.norm(pial_coords[0] - wm_coords[-1]) + np.linalg.norm(
        pial_coords[-1] - wm_coords[0]
    )
    if flipped < same:
        wm_coords = wm_coords[::-1]
    return LineString(pial_coords), LineString(wm_coords)


def rasterize_cortical_ribbon(
    annotations: BoundaryAnnotations,
    *,
    resolution_um: float,
    coordinate_unit_um: float = 1.0,
    padding_um: float | None = None,
    boundary_band_um: float | None = None,
) -> RibbonGrid:
    """Rasterize the cortical ribbon and boundary conditions."""
    if resolution_um <= 0:
        raise ValueError("resolution_um must be positive.")
    if coordinate_unit_um <= 0:
        raise ValueError("coordinate_unit_um must be positive.")

    polygon, side_lines = build_cortical_ribbon_polygon(annotations)
    pial, wm = orient_boundary_pair(annotations.pial, annotations.wm)
    step = float(resolution_um) / float(coordinate_unit_um)
    padding = (
        2.0 * step
        if padding_um is None
        else max(float(padding_um) / float(coordinate_unit_um), step)
    )
    minx, miny, maxx, maxy = polygon.bounds
    x_min = float(minx) - padding
    y_min = float(miny) - padding
    width = int(np.ceil((float(maxx) - x_min + padding) / step))
    height = int(np.ceil((float(maxy) - y_min + padding) / step))
    if width <= 2 or height <= 2:
        raise ValueError(
            "Rasterized cortical ribbon is too small; check resolution and bounds."
        )

    spec = RasterSpec(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        step=step,
        resolution_um=float(resolution_um),
        coordinate_unit_um=float(coordinate_unit_um),
    )
    x_grid, y_grid = np.meshgrid(spec.x_centers, spec.y_centers)
    mask = np.asarray(contains_xy(polygon, x_grid, y_grid), dtype=bool)
    if not mask.any():
        raise ValueError("Rasterized cortical ribbon mask contains no pixels.")

    pial_pixels = rasterize_lines([pial], spec)
    wm_pixels = rasterize_lines([wm], spec)
    side_pixels = rasterize_lines(list(side_lines), spec)

    band_um = (
        1.5 * float(resolution_um)
        if boundary_band_um is None
        else max(float(boundary_band_um), float(resolution_um))
    )
    pial_boundary = _expand_boundary(pial_pixels, mask, band_um, spec.resolution_um)
    wm_boundary = _expand_boundary(wm_pixels, mask, band_um, spec.resolution_um)
    side_boundary = _expand_boundary(side_pixels, mask, band_um, spec.resolution_um)

    overlap = pial_boundary & wm_boundary
    if overlap.any():
        pial_dist = ndimage.distance_transform_edt(~pial_pixels) * spec.resolution_um
        wm_dist = ndimage.distance_transform_edt(~wm_pixels) * spec.resolution_um
        pial_boundary[overlap & (wm_dist < pial_dist)] = False
        wm_boundary[overlap & (pial_dist <= wm_dist)] = False

    if not pial_boundary.any():
        raise ValueError("No pial Dirichlet pixels were found in the ribbon mask.")
    if not wm_boundary.any():
        raise ValueError(
            "No gray/white Dirichlet pixels were found in the ribbon mask."
        )
    if np.count_nonzero(mask & ~(pial_boundary | wm_boundary)) == 0:
        raise ValueError("Ribbon has no interior pixels after boundary rasterization.")

    return RibbonGrid(
        mask=mask,
        pial_boundary=pial_boundary,
        wm_boundary=wm_boundary,
        side_boundary=side_boundary,
        spec=spec,
        polygon=polygon,
        pial_line=pial,
        wm_line=wm,
        side_lines=tuple(side_lines),
    )


def rasterize_lines(lines: list[LineString], spec: RasterSpec) -> np.ndarray:
    """Rasterize one or more lines by dense arc-length sampling."""
    out = np.zeros((spec.height, spec.width), dtype=bool)
    for line in lines:
        if line.is_empty or line.length <= 0:
            continue
        spacing = max(spec.step / 2.0, np.finfo(float).eps)
        n_points = max(2, int(np.ceil(float(line.length) / spacing)) + 1)
        distances = np.linspace(0.0, float(line.length), n_points)
        coords = np.asarray(
            [line.interpolate(distance).coords[0][:2] for distance in distances],
            dtype=float,
        )
        rows, cols = spec.points_to_indices(coords)
        keep = (rows >= 0) & (rows < spec.height) & (cols >= 0) & (cols < spec.width)
        out[rows[keep], cols[keep]] = True
    return out


def points_inside_mask(points: np.ndarray, grid: RibbonGrid) -> np.ndarray:
    """Return whether source-coordinate points fall inside the raster ribbon."""
    arr = np.asarray(points, dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    out = np.zeros(arr.shape[0], dtype=bool)
    if not finite.any():
        return out
    work = arr[finite]
    rows, cols = grid.spec.points_to_indices(work)
    inside_bounds = (
        (rows >= 0) & (rows < grid.spec.height) & (cols >= 0) & (cols < grid.spec.width)
    )
    finite_positions = np.flatnonzero(finite)
    out[finite_positions[inside_bounds]] = grid.mask[
        rows[inside_bounds], cols[inside_bounds]
    ]
    return out


def _expand_boundary(
    line_pixels: np.ndarray,
    mask: np.ndarray,
    band_um: float,
    resolution_um: float,
) -> np.ndarray:
    if not line_pixels.any():
        return np.zeros_like(mask, dtype=bool)
    distance_um = ndimage.distance_transform_edt(~line_pixels) * float(resolution_um)
    return np.asarray(mask & (distance_um <= float(band_um)), dtype=bool)
