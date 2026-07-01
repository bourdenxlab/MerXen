"""Cortical ribbon construction and rasterization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from merxen.cortical_depth.boundaries import (
    BoundaryAnnotations,
    BoundaryAnnotationSet,
    BoundaryPieceAnnotations,
)


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
    wm_line: LineString | None
    side_lines: tuple[LineString, ...]
    tissue_piece_id: str = "piece_1"
    piece_mode: str = "depth"

    @property
    def shape(self: Self) -> tuple[int, int]:
        """Raster shape as ``(height, width)``."""
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))


def build_cortical_ribbon_polygon(
    annotations: BoundaryAnnotations | BoundaryPieceAnnotations,
    *,
    edge_line: LineString | None = None,
) -> tuple[Polygon | MultiPolygon, tuple[LineString, ...]]:
    """Build a ribbon polygon from annotation geometry.

    If a complete ribbon polygon was supplied it is used directly. Otherwise,
    piece-aware annotations are polygonized from the tissue edge plus pia and
    optional WM lines. Legacy pia+WM annotations without an edge still use
    straight endpoint closures.
    """
    wm_line = annotations.wm
    pial = annotations.pial
    wm = None
    if wm_line is not None:
        pial, wm = orient_boundary_pair(annotations.pial, wm_line)
    if annotations.ribbon is not None:
        polygon: BaseGeometry = annotations.ribbon
    elif edge_line is not None:
        polygon = _polygon_from_edge_boundaries(edge_line=edge_line, pial=pial, wm=wm)
    else:
        if wm is None:
            raise ValueError(
                "Pial-only pieces require a tissue edge or explicit "
                "cortical-ribbon polygon."
            )
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

    side_lines = tuple(getattr(annotations, "side_boundaries", ())) or (
        (edge_line,) if edge_line is not None else ()
    )
    if not side_lines and wm is not None:
        side_lines = (
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


def _polygon_from_edge_boundaries(
    *,
    edge_line: LineString,
    pial: LineString,
    wm: LineString | None,
) -> Polygon:
    candidates = _edge_subchain_candidate_piece_polygons(
        edge_line=edge_line, pial=pial, wm=wm
    )

    lines = [edge_line, pial]
    if wm is not None:
        lines.append(wm)
    polygonized = polygonize(unary_union(lines))
    geoms = getattr(polygonized, "geoms", polygonized)
    for geom in geoms:
        if not isinstance(geom, Polygon) or geom.is_empty or geom.area <= 0:
            continue
        boundary = geom.boundary
        if not boundary.intersects(pial):
            continue
        if wm is not None and not boundary.intersects(wm):
            continue
        candidates.append(geom)
    candidates = [
        polygon
        for polygon in _unique_valid_polygons(candidates)
        if _polygon_boundary_line_coverage(polygon, pial) >= 0.95
        and (wm is None or _polygon_boundary_line_coverage(polygon, wm) >= 0.95)
    ]
    if not candidates:
        raise ValueError(
            "Tissue edge and cortical boundaries do not form a valid ribbon polygon; "
            "snap line endpoints to the edge or provide an explicit "
            "cortical-ribbon polygon."
        )
    if wm is None and len(candidates) != 1:
        raise ValueError(
            "Pial-only pieces produced ambiguous polygons; provide an explicit "
            "cortical-ribbon polygon."
        )
    return min(candidates, key=lambda polygon: float(polygon.area))


def _edge_subchain_candidate_piece_polygons(
    *,
    edge_line: LineString,
    pial: LineString,
    wm: LineString | None,
) -> list[Polygon]:
    pial_coords = _line_xy_array(pial)
    if pial_coords.shape[0] < 2:
        return []

    pial_distances = _line_endpoint_edge_distances(edge_line, pial)
    if wm is None:
        pial_only_candidates: list[Polygon] = []
        for edge_path in _edge_paths_between(
            edge_line, pial_distances[1], pial_distances[0]
        ):
            ring = _join_coordinate_parts([pial_coords, edge_path[1:]])
            pial_only_candidates.extend(_valid_polygons_from_ring_coordinates(ring))
        return _unique_valid_polygons(pial_only_candidates)

    wm_coords = _line_xy_array(wm)
    if wm_coords.shape[0] < 2:
        return []
    wm_distances = _line_endpoint_edge_distances(edge_line, wm)

    pairings = (
        # pial start -> WM start, pial end -> WM end.
        (wm_coords[::-1], wm_distances[1], wm_distances[0]),
        # pial start -> WM end, pial end -> WM start.
        (wm_coords, wm_distances[0], wm_distances[1]),
    )
    candidates: list[Polygon] = []
    for wm_path_coords, pial_end_wm_distance, wm_start_pial_distance in pairings:
        for edge_to_wm in _edge_paths_between(
            edge_line, pial_distances[1], pial_end_wm_distance
        ):
            for edge_to_pia in _edge_paths_between(
                edge_line, wm_start_pial_distance, pial_distances[0]
            ):
                ring = _join_coordinate_parts(
                    [
                        pial_coords,
                        edge_to_wm[1:],
                        wm_path_coords[1:],
                        edge_to_pia[1:],
                    ]
                )
                candidates.extend(_valid_polygons_from_ring_coordinates(ring))
    return _unique_valid_polygons(candidates)


def _line_xy_array(line: LineString) -> np.ndarray:
    return np.asarray(line.coords, dtype=float)[:, :2]


def _line_endpoint_edge_distances(
    edge_line: LineString, line: LineString
) -> tuple[float, float]:
    coords = _line_xy_array(line)
    start = Point(float(coords[0, 0]), float(coords[0, 1]))
    end = Point(float(coords[-1, 0]), float(coords[-1, 1]))
    return float(edge_line.project(start)), float(edge_line.project(end))


def _edge_paths_between(
    edge_line: LineString, start_distance: float, end_distance: float
) -> list[np.ndarray]:
    if _is_closed_line(edge_line):
        if np.isclose(start_distance, end_distance, rtol=0.0, atol=1e-9):
            return [_edge_path_forward(edge_line, start_distance, end_distance)]
        forward = _edge_path_forward(edge_line, start_distance, end_distance)
        backward = _edge_path_forward(edge_line, end_distance, start_distance)[::-1]
        return _unique_coordinate_paths([forward, backward])
    return [_edge_path_no_wrap(edge_line, start_distance, end_distance)]


def _edge_path_forward(
    edge_line: LineString, start_distance: float, end_distance: float
) -> np.ndarray:
    length = float(edge_line.length)
    start = float(np.clip(start_distance, 0.0, length))
    end = float(np.clip(end_distance, 0.0, length))
    if start <= end:
        return _edge_path_no_wrap(edge_line, start, end)
    first = _edge_path_no_wrap(edge_line, start, length)
    second = _edge_path_no_wrap(edge_line, 0.0, end)
    return _drop_consecutive_duplicate_points(np.vstack([first, second[1:]]))


def _edge_path_no_wrap(
    edge_line: LineString, start_distance: float, end_distance: float
) -> np.ndarray:
    length = float(edge_line.length)
    start = float(np.clip(start_distance, 0.0, length))
    end = float(np.clip(end_distance, 0.0, length))
    if start > end:
        return _edge_path_no_wrap(edge_line, end, start)[::-1]

    edge_coords = _line_xy_array(edge_line)
    points: list[np.ndarray] = [_point_on_line(edge_line, start)]
    cumulative = 0.0
    for idx in range(edge_coords.shape[0] - 1):
        segment_length = float(np.linalg.norm(edge_coords[idx + 1] - edge_coords[idx]))
        next_cumulative = cumulative + segment_length
        if segment_length > 0 and start < next_cumulative < end:
            points.append(edge_coords[idx + 1].astype(float, copy=True))
        cumulative = next_cumulative
    points.append(_point_on_line(edge_line, end))
    return _drop_consecutive_duplicate_points(np.vstack(points))


def _point_on_line(line: LineString, distance: float) -> np.ndarray:
    point = line.interpolate(float(distance))
    return np.asarray(point.coords[0][:2], dtype=float)


def _is_closed_line(line: LineString) -> bool:
    coords = _line_xy_array(line)
    return coords.shape[0] > 2 and bool(
        np.allclose(coords[0], coords[-1], rtol=0.0, atol=1e-9)
    )


def _drop_consecutive_duplicate_points(xy: np.ndarray) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy
    keep = np.ones(xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0
    return xy[keep]


def _join_coordinate_parts(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [
        np.asarray(part, dtype=float)[:, :2] for part in parts if np.asarray(part).size
    ]
    if not arrays:
        return np.empty((0, 2), dtype=float)
    return _drop_consecutive_duplicate_points(np.vstack(arrays))


def _valid_polygons_from_ring_coordinates(coords: np.ndarray) -> list[Polygon]:
    coords = _drop_consecutive_duplicate_points(np.asarray(coords, dtype=float)[:, :2])
    if coords.shape[0] < 3:
        return []
    if not np.allclose(coords[0], coords[-1], rtol=0.0, atol=1e-9):
        coords = np.vstack([coords, coords[0]])
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, Polygon):
        return [polygon] if not polygon.is_empty and polygon.area > 0 else []
    if isinstance(polygon, MultiPolygon):
        return [part for part in polygon.geoms if not part.is_empty and part.area > 0]
    if isinstance(polygon, GeometryCollection):
        return [
            part
            for part in polygon.geoms
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0
        ]
    return []


def _unique_valid_polygons(polygons: Iterable[Polygon]) -> list[Polygon]:
    out: list[Polygon] = []
    seen: set[tuple[float, float, float]] = set()
    for polygon in polygons:
        if not isinstance(polygon, Polygon) or polygon.is_empty or polygon.area <= 0:
            continue
        candidate = polygon if polygon.is_valid else polygon.buffer(0)
        for part in _geometry_polygon_parts(candidate):
            if any(_polygons_are_near_duplicates(part, existing) for existing in out):
                continue
            key = (
                round(float(part.area), 6),
                round(float(part.centroid.x), 6),
                round(float(part.centroid.y), 6),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(part)
    return out


def _polygons_are_near_duplicates(left: Polygon, right: Polygon) -> bool:
    area_scale = max(float(left.area), float(right.area), 1.0)
    return float(left.symmetric_difference(right).area) <= 1e-6 * area_scale


def _polygon_boundary_line_coverage(polygon: Polygon, line: LineString) -> float:
    tolerance = max(1e-6, 1e-9 * max(float(polygon.length), float(line.length)))
    missing = line.difference(polygon.boundary.buffer(tolerance))
    missing_length = 0.0 if missing.is_empty else float(getattr(missing, "length", 0.0))
    line_length = float(line.length)
    if line_length <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - missing_length / line_length))


def _geometry_polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty and geom.area > 0 else []
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty and part.area > 0]
    if isinstance(geom, GeometryCollection):
        return [
            part
            for part in geom.geoms
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0
        ]
    return []


def _unique_coordinate_paths(paths: Iterable[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for path in paths:
        path = _drop_consecutive_duplicate_points(np.asarray(path, dtype=float)[:, :2])
        key = tuple((round(float(x), 6), round(float(y), 6)) for x, y in path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def rasterize_cortical_ribbon(
    annotations: BoundaryAnnotationSet | BoundaryAnnotations | BoundaryPieceAnnotations,
    *,
    resolution_um: float,
    coordinate_unit_um: float = 1.0,
    padding_um: float | None = None,
    boundary_band_um: float | None = None,
    edge_line: LineString | None = None,
    require_wm: bool = True,
) -> RibbonGrid:
    """Rasterize the cortical ribbon and boundary conditions."""
    if isinstance(annotations, BoundaryAnnotationSet):
        if len(annotations.pieces) != 1:
            raise ValueError(
                "Use one BoundaryPieceAnnotations at a time for rasterization."
            )
        edge_line = annotations.edge
        annotations = annotations.pieces[0]
    tissue_piece_id = getattr(annotations, "tissue_piece_id", "piece_1")
    piece_mode = "depth" if annotations.wm is not None else "mask_qc_only"
    if require_wm and annotations.wm is None:
        raise ValueError(
            f"{tissue_piece_id} has no gray/white boundary for depth computation."
        )
    if resolution_um <= 0:
        raise ValueError("resolution_um must be positive.")
    if coordinate_unit_um <= 0:
        raise ValueError("coordinate_unit_um must be positive.")

    polygon, side_lines = build_cortical_ribbon_polygon(
        annotations, edge_line=edge_line
    )
    pial = annotations.pial
    wm = annotations.wm
    if wm is not None:
        pial, wm = orient_boundary_pair(annotations.pial, wm)
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
    wm_pixels = (
        rasterize_lines([wm], spec)
        if wm is not None
        else np.zeros_like(mask, dtype=bool)
    )
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
    if require_wm and not wm_boundary.any():
        raise ValueError(
            "No gray/white Dirichlet pixels were found in the ribbon mask."
        )
    if require_wm and np.count_nonzero(mask & ~(pial_boundary | wm_boundary)) == 0:
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
        tissue_piece_id=str(tissue_piece_id),
        piece_mode=piece_mode,
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
