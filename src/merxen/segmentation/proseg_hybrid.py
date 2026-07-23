"""Local-convex refinement of Cellpose and Proseg segmentation."""

from __future__ import annotations

import json
import logging
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.affinity import affine_transform
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation

from merxen.config import ProsegHybridConfig
from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    PROSEG_ASSIGNMENT_COLUMN,
    PROSEG_ID_NAMESPACE,
    PROSEG_INTERNAL_ID_COLUMN,
    SOURCE_CELL_ID_COLUMN,
    TRANSCRIPT_ID_COLUMN,
    choose_primary_points_key,
    register_segmentation_branch,
    stamp_merxen_schema,
    validate_merxen_schema,
    with_stable_transcript_ids,
)
from merxen.io.transcript_io import background_mask, first_existing_col
from merxen.memory import force_release, log_status
from merxen.segmentation.mask_geometry import masks_to_labeled_polygons

logger = logging.getLogger(__name__)

PROSEG_HYBRID_SHAPE_NAME = "MOSAIK_proseg_hybrid"
PROSEG_HYBRID_TABLE_NAME = "table_MOSAIK_proseg_hybrid"
HYBRID_ASSIGNMENT_COLUMN = "hybrid_assignment"
HYBRID_BACKGROUND_COLUMN = "hybrid_background"
HYBRID_CANDIDATE_COUNT_COLUMN = "hybrid_candidate_count"
HYBRID_ASSIGNMENT_SOURCE_COLUMN = "hybrid_assignment_source"
PROSEG_HYBRID_ALGORITHM = "local_convex_growth_only_v1"

_LOCAL_CAP_QUAD_SEGS = 64
_LOCAL_ROUNDING_QUAD_SEGS = 12
_ATTACHMENT_ARC_SAMPLES = 9


@dataclass(frozen=True)
class BulkSelection:
    """Robustly selected bulk transcript component for one cell."""

    retained_points: np.ndarray
    outlier_count: int
    neighbor_scale: float
    is_cellpose_anchored: bool


@dataclass(frozen=True)
class HybridCellGeometry:
    """Final geometry and construction diagnostics for one cell."""

    geometry: BaseGeometry
    retained_count: int
    outlier_count: int
    neighbor_scale: float
    candidate_external: int
    cap_rejected_external: int
    unsupported_external: int
    near_surface_accepted: int
    chain_accepted: int
    supported_external: int
    accepted_groups: int
    pre_smoothing_area: float
    smoothing_added_area: float
    area_growth_fraction: float
    perimeter_change_fraction: float
    holes_filled: int
    fallback_reason: str


@dataclass(frozen=True)
class LocalConvexExpansion:
    """Transcript-supported local-convex expansion before final smoothing."""

    geometry: BaseGeometry
    retained_count: int
    outlier_count: int
    neighbor_scale: float
    candidate_external: int
    cap_rejected_external: int
    unsupported_external: int
    near_surface_accepted: int
    chain_accepted: int
    supported_external: int
    accepted_groups: int
    supported_coordinates: np.ndarray
    fallback_reason: str


@dataclass(frozen=True)
class GrowthOnlySmoothing:
    """Growth-only smoothing result and its geometric diagnostics."""

    geometry: BaseGeometry
    cap_region: BaseGeometry
    original_area: float
    smoothed_area: float
    added_area: float
    area_growth_fraction: float
    original_perimeter: float
    smoothed_perimeter: float
    perimeter_change_fraction: float
    holes_filled: int
    missing_original_area: float
    cap_violation_area: float


def select_bulk_transcripts(
    points_xy: np.ndarray,
    cellpose_polygon: BaseGeometry,
    *,
    neighbors: int = 2,
    mad_multiplier: float = 2.0,
) -> BulkSelection:
    """Select the dominant density component of transcript coordinates.

    Args:
        points_xy: Observed transcript coordinates with shape ``(n, 2)``.
        cellpose_polygon: Cellpose polygon for the same cell.
        neighbors: Nearest-neighbour rank used to estimate local density.
        mad_multiplier: Robust MAD multiplier used as the graph link distance.

    Returns:
        The retained component and diagnostics. Component size is primary;
        Cellpose overlap and distance to the Cellpose center break ties.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    if len(points) == 0:
        return BulkSelection(points, 0, 0.0, False)
    if len(points) == 1:
        is_anchored = bool(cellpose_polygon.covers(Point(points[0])))
        retained = points if is_anchored else np.empty((0, 2), dtype=np.float64)
        return BulkSelection(retained, len(points) - len(retained), 0.0, is_anchored)

    neighbor_rank = min(max(1, int(neighbors)), len(points) - 1)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=neighbor_rank + 1)
    kth_distances = np.asarray(distances[:, neighbor_rank], dtype=np.float64)
    finite_distances = kth_distances[np.isfinite(kth_distances)]
    if len(finite_distances) == 0:
        return BulkSelection(np.empty((0, 2)), len(points), 0.0, False)

    distance_median = float(np.median(finite_distances))
    distance_mad = float(np.median(np.abs(finite_distances - distance_median)))
    robust_sigma = 1.4826 * distance_mad
    link_distance = distance_median + (float(mad_multiplier) * robust_sigma)
    if link_distance <= 0.0:
        positive = finite_distances[finite_distances > 0.0]
        link_distance = float(np.median(positive)) if len(positive) else 0.0

    pairs = tree.query_pairs(link_distance, output_type="ndarray")
    if len(pairs) == 0:
        component_labels = np.arange(len(points), dtype=np.int64)
        n_components = len(points)
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(points), len(points)),
        ).tocsr()
        n_components, component_labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

    anchored = np.fromiter(
        (cellpose_polygon.covers(Point(xy)) for xy in points),
        dtype=bool,
        count=len(points),
    )
    cellpose_center = cellpose_polygon.representative_point()
    center_xy = np.asarray([cellpose_center.x, cellpose_center.y], dtype=np.float64)
    best_component = -1
    best_score = (-1, -1, float("-inf"))
    for component in range(int(n_components)):
        component_mask = component_labels == component
        component_points = points[component_mask]
        center_distance = float(
            np.median(np.linalg.norm(component_points - center_xy, axis=1))
        )
        score = (
            int(np.count_nonzero(component_mask)),
            int(np.count_nonzero(anchored & component_mask)),
            -center_distance,
        )
        if score > best_score:
            best_component = component
            best_score = score

    primary_mask = component_labels == best_component
    primary_points = points[primary_mask]
    primary_center = np.median(primary_points, axis=0)
    primary_radii = np.linalg.norm(primary_points - primary_center, axis=1)
    radius_median = float(np.median(primary_radii))
    radius_mad = float(np.median(np.abs(primary_radii - radius_median)))
    radius_sigma = 1.4826 * radius_mad
    epsilon = float(np.finfo(np.float64).eps)
    remote_cutoff = (
        radius_median
        + (float(mad_multiplier) * radius_sigma)
        + max(float(link_distance), epsilon)
    )
    low_component_limit = max(
        neighbor_rank,
        int(np.ceil(np.sqrt(len(primary_points)))),
    )
    retained_mask = primary_mask.copy()
    for component in range(int(n_components)):
        if component == best_component:
            continue
        component_mask = component_labels == component
        component_points = points[component_mask]
        center_distance = float(
            np.median(np.linalg.norm(component_points - primary_center, axis=1))
        )
        is_low_count = len(component_points) <= low_component_limit
        is_remote = center_distance > remote_cutoff
        if not (is_low_count and is_remote):
            retained_mask |= component_mask

    retained = points[retained_mask]
    is_anchored = bool(np.any(anchored & retained_mask))
    return BulkSelection(
        retained,
        len(points) - len(retained),
        link_distance,
        is_anchored,
    )


def _polygonal_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Return every polygonal component from a Shapely geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        parts: list[Polygon] = []
        for part in geometry.geoms:  # type: ignore[attr-defined]
            parts.extend(_polygonal_parts(part))
        return parts
    return []


def _fill_all_polygon_holes(geometry: BaseGeometry) -> BaseGeometry:
    """Return polygonal geometry with every interior ring filled."""
    parts = _polygonal_parts(geometry)
    if not parts:
        return Polygon()
    filled = unary_union([Polygon(part.exterior) for part in parts])
    if not filled.is_valid:
        filled = filled.buffer(0)
    return filled


def _count_polygon_holes(geometry: BaseGeometry) -> int:
    """Count interior rings across every polygonal component."""
    return int(sum(len(part.interiors) for part in _polygonal_parts(geometry)))


def _graph_components(neighbor_sets: list[set[int]]) -> list[np.ndarray]:
    """Return connected components for an undirected adjacency list."""
    unseen = set(range(len(neighbor_sets)))
    components: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        component = [seed]
        queue = deque([seed])
        while queue:
            index = queue.popleft()
            discovered = neighbor_sets[index].intersection(unseen)
            if not discovered:
                continue
            unseen.difference_update(discovered)
            component.extend(discovered)
            queue.extend(discovered)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _subset_graph_components(
    neighbor_sets: list[set[int]],
    selected_indices: np.ndarray,
) -> list[np.ndarray]:
    """Split selected nodes without allowing excluded nodes to bridge them."""
    selected = set(int(index) for index in selected_indices)
    groups: list[np.ndarray] = []
    while selected:
        seed = selected.pop()
        group = [seed]
        queue = deque([seed])
        while queue:
            index = queue.popleft()
            discovered = neighbor_sets[index].intersection(selected)
            if not discovered:
                continue
            selected.difference_update(discovered)
            group.extend(discovered)
            queue.extend(discovered)
        groups.append(np.asarray(group, dtype=np.int64))
    return groups


def _attachment_arc_coordinates(
    cellpose_polygon: BaseGeometry,
    group_points: np.ndarray,
    *,
    neighbor_scale: float,
    width_scale: float,
) -> np.ndarray:
    """Sample a local Cellpose boundary arc nearest an external group."""
    group_geometry = MultiPoint(group_points)
    cellpose_parts = _polygonal_parts(cellpose_polygon)
    if not cellpose_parts:
        return np.empty((0, 2), dtype=np.float64)
    ring = min(
        (part.exterior for part in cellpose_parts),
        key=lambda candidate: candidate.distance(group_geometry),
    )
    ring_length = float(ring.length)
    if ring_length <= np.finfo(np.float64).eps:
        return np.empty((0, 2), dtype=np.float64)

    nearest_surface = nearest_points(group_geometry, ring)[1]
    center_distance = float(ring.project(nearest_surface))
    tangent_step = min(
        max(float(neighbor_scale), 0.01 * ring_length),
        0.1 * ring_length,
    )
    before = np.asarray(
        ring.interpolate((center_distance - tangent_step) % ring_length).coords[0],
        dtype=np.float64,
    )
    after = np.asarray(
        ring.interpolate((center_distance + tangent_step) % ring_length).coords[0],
        dtype=np.float64,
    )
    tangent = after - before
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(np.float64).eps:
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        tangent /= tangent_norm

    surface_xy = np.asarray(nearest_surface.coords[0], dtype=np.float64)
    projections = (group_points - surface_xy) @ tangent
    tangential_span = float(np.ptp(projections)) if len(projections) > 1 else 0.0
    group_width = max(
        2.0 * float(neighbor_scale),
        tangential_span + (2.0 * float(neighbor_scale)),
    )
    arc_width = min(
        0.45 * ring_length,
        max(float(neighbor_scale), float(width_scale) * group_width),
    )
    offsets = np.linspace(
        -0.5 * arc_width,
        0.5 * arc_width,
        _ATTACHMENT_ARC_SAMPLES,
    )
    return np.asarray(
        [
            ring.interpolate((center_distance + offset) % ring_length).coords[0]
            for offset in offsets
        ],
        dtype=np.float64,
    )


def _empty_local_expansion(
    cellpose_polygon: BaseGeometry,
    *,
    fallback_reason: str,
    retained_count: int = 0,
    outlier_count: int = 0,
    neighbor_scale: float = 0.0,
    candidate_external: int = 0,
    cap_rejected_external: int = 0,
    unsupported_external: int = 0,
) -> LocalConvexExpansion:
    """Build a Cellpose-only local expansion result."""
    return LocalConvexExpansion(
        geometry=cellpose_polygon,
        retained_count=retained_count,
        outlier_count=outlier_count,
        neighbor_scale=neighbor_scale,
        candidate_external=candidate_external,
        cap_rejected_external=cap_rejected_external,
        unsupported_external=unsupported_external,
        near_surface_accepted=0,
        chain_accepted=0,
        supported_external=0,
        accepted_groups=0,
        supported_coordinates=np.empty((0, 2), dtype=np.float64),
        fallback_reason=fallback_reason,
    )


def build_local_convex_expansion_geometry(
    points_xy: np.ndarray,
    cellpose_polygon: BaseGeometry,
    config: ProsegHybridConfig,
) -> LocalConvexExpansion:
    """Add capped local convex wedges driven by supported Proseg transcripts.

    Only retained Proseg-foreground transcripts outside Cellpose can expand the
    mask. Near-surface transcripts are accepted directly. More distant groups
    must contain at least ``minimum_external_group`` transcripts and form a
    nearest-neighbour chain back to the Cellpose surface. Every expansion is
    attached to a short Cellpose boundary arc and clipped to the configured
    equivalent-radius cap.

    Args:
        points_xy: Proseg-foreground observed transcript coordinates.
        cellpose_polygon: Cellpose polygon for the same cell.
        config: Local-convex selection and construction parameters.

    Returns:
        The expanded geometry, supported coordinates, and selection diagnostics.

    Raises:
        ValueError: If coordinates are malformed or Cellpose has no area.
        AssertionError: If strict transcript or Cellpose containment fails.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    if cellpose_polygon.is_empty or cellpose_polygon.area <= 0.0:
        raise ValueError("cellpose_polygon must have positive area")
    if len(points) < config.min_transcripts:
        return _empty_local_expansion(
            cellpose_polygon,
            fallback_reason="low_transcript_count",
        )

    selection = select_bulk_transcripts(
        points,
        cellpose_polygon,
        neighbors=config.outlier_neighbors,
        mad_multiplier=config.outlier_mad_multiplier,
    )
    retained = selection.retained_points
    neighbor_scale = float(selection.neighbor_scale)
    if len(retained) < config.min_transcripts:
        return _empty_local_expansion(
            cellpose_polygon,
            fallback_reason="low_retained_transcript_count",
            retained_count=len(retained),
            outlier_count=selection.outlier_count,
            neighbor_scale=neighbor_scale,
        )

    outside = np.fromiter(
        (not cellpose_polygon.covers(Point(coordinate)) for coordinate in retained),
        dtype=bool,
        count=len(retained),
    )
    external = retained[outside]
    equivalent_radius = float(np.sqrt(cellpose_polygon.area / np.pi))
    cap_distance = float(config.maximum_expansion_radius_fraction) * equivalent_radius
    near_distance = float(config.near_surface_radius_fraction) * equivalent_radius
    epsilon = max(float(np.finfo(np.float64).eps), equivalent_radius * 1.0e-7)
    containment_epsilon = max(1.0e-4, equivalent_radius * 1.0e-5)
    if len(external) == 0 or neighbor_scale <= epsilon or cap_distance <= epsilon:
        return _empty_local_expansion(
            cellpose_polygon,
            fallback_reason="",
            retained_count=len(retained),
            outlier_count=selection.outlier_count,
            neighbor_scale=neighbor_scale,
            candidate_external=len(external),
            unsupported_external=len(external),
        )

    external_distances = np.fromiter(
        (cellpose_polygon.distance(Point(coordinate)) for coordinate in external),
        dtype=np.float64,
        count=len(external),
    )
    within_cap = external_distances <= cap_distance
    candidates = external[within_cap]
    candidate_distances = external_distances[within_cap]
    cap_rejected = int((~within_cap).sum())
    if len(candidates) == 0:
        return _empty_local_expansion(
            cellpose_polygon,
            fallback_reason="",
            retained_count=len(retained),
            outlier_count=selection.outlier_count,
            neighbor_scale=neighbor_scale,
            candidate_external=len(external),
            cap_rejected_external=cap_rejected,
            unsupported_external=len(external),
        )

    chain_radius = float(config.chain_radius_scale) * neighbor_scale
    candidate_tree = cKDTree(candidates)
    neighbor_sets = [
        set(int(value) for value in neighbors if int(value) != index)
        for index, neighbors in enumerate(
            candidate_tree.query_ball_point(candidates, r=chain_radius)
        )
    ]
    components = _graph_components(neighbor_sets)
    near_surface = candidate_distances <= near_distance + epsilon
    chain_anchors = candidate_distances <= chain_radius + epsilon
    accepted = near_surface.copy()
    for component in components:
        if len(component) >= config.minimum_external_group and bool(
            chain_anchors[component].any()
        ):
            accepted[component] = True

    accepted_indices = np.flatnonzero(accepted)
    if len(accepted_indices) == 0:
        return _empty_local_expansion(
            cellpose_polygon,
            fallback_reason="",
            retained_count=len(retained),
            outlier_count=selection.outlier_count,
            neighbor_scale=neighbor_scale,
            candidate_external=len(external),
            cap_rejected_external=cap_rejected,
            unsupported_external=len(external),
        )

    accepted_groups = _subset_graph_components(neighbor_sets, accepted_indices)
    cap_buffer_distance = (cap_distance + containment_epsilon) / np.cos(
        np.pi / (4.0 * _LOCAL_CAP_QUAD_SEGS)
    )
    cap_region = cellpose_polygon.buffer(
        cap_buffer_distance,
        quad_segs=_LOCAL_CAP_QUAD_SEGS,
    )
    rounding_radius = float(config.rounding_radius_fraction) * equivalent_radius
    extensions: list[BaseGeometry] = []
    for group_indices in accepted_groups:
        group_points = candidates[group_indices]
        arc_points = _attachment_arc_coordinates(
            cellpose_polygon,
            group_points,
            neighbor_scale=neighbor_scale,
            width_scale=float(config.attachment_arc_width_scale),
        )
        if len(arc_points) == 0:
            continue
        local_hull = MultiPoint(np.vstack([group_points, arc_points])).convex_hull
        rounded_hull = (
            local_hull.buffer(
                rounding_radius,
                quad_segs=_LOCAL_ROUNDING_QUAD_SEGS,
            )
            if rounding_radius > 0.0
            else local_hull
        )
        clipped_hull = rounded_hull.intersection(cap_region)
        if not clipped_hull.is_empty:
            extensions.append(clipped_hull)

    supported = candidates[accepted_indices]
    containment_support = MultiPoint(supported).buffer(
        containment_epsilon,
        quad_segs=4,
    )
    containment_support = containment_support.intersection(cap_region)
    geometry = unary_union([cellpose_polygon, containment_support, *extensions])
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    if not all(geometry.covers(Point(coordinate)) for coordinate in supported):
        raise AssertionError(
            "Local convex expansion excluded a geometry-driving transcript"
        )
    area_tolerance = max(1.0e-8, cellpose_polygon.area * 1.0e-10)
    if cellpose_polygon.difference(geometry).area > area_tolerance:
        raise AssertionError("Local convex geometry no longer covers Cellpose")

    chain_only = accepted & ~near_surface
    return LocalConvexExpansion(
        geometry=geometry,
        retained_count=len(retained),
        outlier_count=selection.outlier_count,
        neighbor_scale=neighbor_scale,
        candidate_external=len(external),
        cap_rejected_external=cap_rejected,
        unsupported_external=len(external) - len(supported),
        near_surface_accepted=int((accepted & near_surface).sum()),
        chain_accepted=int(chain_only.sum()),
        supported_external=len(supported),
        accepted_groups=len(accepted_groups),
        supported_coordinates=supported,
        fallback_reason="",
    )


def smooth_growth_only_geometry(
    current_geometry: BaseGeometry,
    cellpose_polygon: BaseGeometry,
    config: ProsegHybridConfig,
) -> GrowthOnlySmoothing:
    """Smooth a mask only by adding area and respect its Cellpose cap.

    Args:
        current_geometry: Assembled local-convex mask to smooth.
        cellpose_polygon: Original Cellpose polygon that defines the hard cap.
        config: Fixed-micron smoothing and numerical containment parameters.

    Returns:
        The smoothed geometry, cap, and growth/topology diagnostics.

    Raises:
        ValueError: If either input is empty or Cellpose has no positive area.
        AssertionError: If smoothing shrinks the mask, escapes the cap, or leaves
            an internal hole.
    """
    if current_geometry.is_empty or cellpose_polygon.is_empty:
        raise ValueError("Cannot smooth an empty current or Cellpose geometry")

    original = current_geometry
    holes_before = _count_polygon_holes(original)
    base = _fill_all_polygon_holes(original)
    equivalent_radius = float(np.sqrt(cellpose_polygon.area / np.pi))
    if not np.isfinite(equivalent_radius) or equivalent_radius <= 0.0:
        raise ValueError("Cellpose geometry must have positive finite area")

    segments = int(config.smoothing_quad_segs)
    tolerance = float(config.containment_tolerance_um)
    cap_distance = float(config.maximum_expansion_radius_fraction) * equivalent_radius
    cap_buffer_distance = (cap_distance + tolerance) / np.cos(np.pi / (4.0 * segments))
    cap_region = _fill_all_polygon_holes(
        cellpose_polygon.buffer(cap_buffer_distance, quad_segs=segments)
    )

    smoothing_radius = float(config.smoothing_radius_um)
    if smoothing_radius > 0.0:
        closed = base.buffer(smoothing_radius, quad_segs=segments).buffer(
            -smoothing_radius,
            quad_segs=segments,
        )
        if closed.is_empty:
            closed = base
    else:
        closed = base
    closed = _fill_all_polygon_holes(closed)
    outward_rounding = float(config.outward_rounding_um)
    rounded = (
        closed.buffer(outward_rounding, quad_segs=segments)
        if outward_rounding > 0.0
        else closed
    )
    capped_candidate = rounded.intersection(cap_region)
    final_geometry = _fill_all_polygon_holes(unary_union([base, capped_candidate]))
    if not final_geometry.is_valid:
        final_geometry = final_geometry.buffer(0)
    final_geometry = _fill_all_polygon_holes(unary_union([original, final_geometry]))

    area_tolerance = max(tolerance * tolerance, original.area * 1.0e-10)
    missing_original_area = float(original.difference(final_geometry).area)
    if missing_original_area > area_tolerance:
        raise AssertionError(
            "Growth-only smoothing failed to contain the original mask: "
            f"missing area={missing_original_area:.6g} um^2"
        )
    newly_added = final_geometry.difference(base)
    cap_violation_area = float(newly_added.difference(cap_region).area)
    if cap_violation_area > area_tolerance:
        raise AssertionError(
            "Smoothing added geometry beyond the Cellpose expansion cap: "
            f"area={cap_violation_area:.6g} um^2"
        )
    holes_after = _count_polygon_holes(final_geometry)
    if holes_after:
        raise AssertionError(f"Smoothed geometry still contains {holes_after} hole(s)")

    original_area = float(original.area)
    final_area = float(final_geometry.area)
    original_perimeter = float(original.length)
    final_perimeter = float(final_geometry.length)
    return GrowthOnlySmoothing(
        geometry=final_geometry,
        cap_region=cap_region,
        original_area=original_area,
        smoothed_area=final_area,
        added_area=float(final_geometry.difference(original).area),
        area_growth_fraction=float(final_area / original_area - 1.0),
        original_perimeter=original_perimeter,
        smoothed_perimeter=final_perimeter,
        perimeter_change_fraction=float(final_perimeter / original_perimeter - 1.0),
        holes_filled=holes_before,
        missing_original_area=missing_original_area,
        cap_violation_area=cap_violation_area,
    )


def build_hybrid_cell_geometry(
    points_xy: np.ndarray,
    cellpose_polygon: BaseGeometry,
    config: ProsegHybridConfig,
) -> HybridCellGeometry:
    """Build one final local-convex, growth-only-smoothed hybrid mask.

    Args:
        points_xy: Proseg-foreground observed transcript coordinates.
        cellpose_polygon: Cellpose polygon for the same cell.
        config: Complete hybrid construction and smoothing parameters.

    Returns:
        Final cell geometry and per-cell expansion/smoothing diagnostics.

    Raises:
        AssertionError: If final smoothing excludes a supported transcript.
    """
    expansion = build_local_convex_expansion_geometry(
        points_xy,
        cellpose_polygon,
        config,
    )
    smoothing = smooth_growth_only_geometry(
        expansion.geometry,
        cellpose_polygon,
        config,
    )
    if not all(
        smoothing.geometry.covers(Point(coordinate))
        for coordinate in expansion.supported_coordinates
    ):
        raise AssertionError("Final smoothing excluded a supported transcript")
    return HybridCellGeometry(
        geometry=smoothing.geometry,
        retained_count=expansion.retained_count,
        outlier_count=expansion.outlier_count,
        neighbor_scale=expansion.neighbor_scale,
        candidate_external=expansion.candidate_external,
        cap_rejected_external=expansion.cap_rejected_external,
        unsupported_external=expansion.unsupported_external,
        near_surface_accepted=expansion.near_surface_accepted,
        chain_accepted=expansion.chain_accepted,
        supported_external=expansion.supported_external,
        accepted_groups=expansion.accepted_groups,
        pre_smoothing_area=smoothing.original_area,
        smoothing_added_area=smoothing.added_area,
        area_growth_fraction=smoothing.area_growth_fraction,
        perimeter_change_fraction=smoothing.perimeter_change_fraction,
        holes_filled=smoothing.holes_filled,
        fallback_reason=expansion.fallback_reason,
    )


def _cellpose_polygons_in_microns(
    mask_path: Path,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
) -> dict[int, BaseGeometry]:
    """Convert every Cellpose label to a micron-space polygon."""
    mask = np.load(mask_path, mmap_mode="r")
    labeled_polygons = masks_to_labeled_polygons(mask, n_jobs=1)
    coefficients = [
        float(x_transform[0]),
        float(x_transform[1]),
        float(y_transform[0]),
        float(y_transform[1]),
        float(x_transform[2]),
        float(y_transform[2]),
    ]
    polygons = {
        int(label_id): affine_transform(polygon, coefficients)
        for label_id, polygon in labeled_polygons
    }
    del mask
    return polygons


def _proseg_assignment_to_instance_ids(sdata_obj: Any) -> dict[int, int]:
    """Map stored ProSeg assignments to canonical positive instance IDs."""
    if "table" not in sdata_obj.tables:
        raise KeyError("Proseg SpatialData output has no base 'table'.")
    obs = sdata_obj.tables["table"].obs
    if INSTANCE_ID_COLUMN in obs.columns:
        instance_ids = pd.to_numeric(
            obs[INSTANCE_ID_COLUMN],
            errors="raise",
        ).astype(np.uint64)
        return {
            int(instance_id): int(instance_id) for instance_id in instance_ids.tolist()
        }

    source_col = (
        SOURCE_CELL_ID_COLUMN
        if SOURCE_CELL_ID_COLUMN in obs.columns
        else "original_cell_id"
    )
    if source_col not in obs.columns:
        raise KeyError("Proseg table has no canonical or original cell identifier.")
    internal_col = (
        PROSEG_INTERNAL_ID_COLUMN
        if PROSEG_INTERNAL_ID_COLUMN in obs.columns
        else "cell"
    )
    if internal_col in obs.columns:
        cells = pd.to_numeric(obs[internal_col], errors="raise").astype(np.int64)
    else:
        cells = pd.Series(np.arange(len(obs), dtype=np.int64), index=obs.index)
    original = pd.to_numeric(obs[source_col], errors="raise").astype(np.uint64)
    return {
        int(cell): int(original_cell)
        for cell, original_cell in zip(cells, original, strict=True)
    }


def _points_columns(points_obj: Any) -> tuple[str, str, str, str, str]:
    """Resolve required Proseg point columns for hybrid refinement."""
    x_col = first_existing_col(points_obj, ["x", "observed_x", "x_micron"])
    y_col = first_existing_col(points_obj, ["y", "observed_y", "y_micron"])
    gene_col = first_existing_col(points_obj, ["gene", "feature_name", "target"])
    assignment_col = first_existing_col(points_obj, [PROSEG_ASSIGNMENT_COLUMN])
    background_col = first_existing_col(points_obj, ["background"])
    values = (x_col, y_col, gene_col, assignment_col, background_col)
    if any(value is None for value in values):
        raise KeyError(
            "Hybrid refinement requires x, y, gene, assignment, and background "
            f"columns; available={list(points_obj.columns)}"
        )
    return tuple(str(value) for value in values)  # type: ignore[return-value]


def _iter_point_partitions(points_obj: Any) -> Iterator[pd.DataFrame]:
    """Materialize point partitions one at a time."""
    if hasattr(points_obj, "npartitions") and hasattr(points_obj, "partitions"):
        for index in range(int(points_obj.npartitions)):
            yield points_obj.partitions[index].compute()
        return
    if hasattr(points_obj, "compute") and not isinstance(points_obj, pd.DataFrame):
        yield points_obj.compute()
        return
    yield points_obj.copy()


def _collect_foreground_coordinates(
    points_obj: Any,
    assignment_map: dict[int, int],
) -> dict[int, np.ndarray]:
    """Collect observed foreground coordinates by canonical Cellpose cell ID."""
    x_col, y_col, _, assignment_col, background_col = _points_columns(points_obj)
    coordinate_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
    for partition in _iter_point_partitions(points_obj):
        assignments = pd.to_numeric(partition[assignment_col], errors="coerce")
        foreground = ~background_mask(partition[background_col])
        valid = assignments.notna() & foreground
        valid &= pd.to_numeric(partition[x_col], errors="coerce").notna()
        valid &= pd.to_numeric(partition[y_col], errors="coerce").notna()
        if not bool(valid.any()):
            continue
        selected = partition.loc[valid, [x_col, y_col]].copy()
        selected["_assignment"] = assignments.loc[valid].astype(np.int64)
        for assignment, group in selected.groupby("_assignment", sort=False):
            cell_id = assignment_map.get(int(assignment))
            if cell_id is None:
                continue
            coordinate_chunks[cell_id].append(
                group[[x_col, y_col]].to_numpy(dtype=np.float64)
            )
    return {
        cell_id: np.concatenate(chunks, axis=0)
        for cell_id, chunks in coordinate_chunks.items()
    }


def assign_transcripts_to_hybrid_masks(
    points: pd.DataFrame,
    polygons: list[BaseGeometry],
    cell_ids: list[int],
    assignment_map: dict[int, int],
    *,
    x_col: str,
    y_col: str,
    assignment_col: str,
) -> pd.DataFrame:
    """Assign one transcript partition using single-mask and overlap rules.

    Args:
        points: Transcript partition containing observed coordinates and Proseg
            assignments.
        polygons: Hybrid polygons in the same order as ``cell_ids``.
        cell_ids: Canonical Cellpose cell identifiers.
        assignment_map: Proseg internal assignment to canonical cell identifier.
        x_col: Observed x-coordinate column.
        y_col: Observed y-coordinate column.
        assignment_col: Proseg assignment column.

    Returns:
        Copy of ``points`` with hybrid assignment provenance columns.
    """
    result = points.copy()
    x_values = pd.to_numeric(result[x_col], errors="coerce").to_numpy(np.float64)
    y_values = pd.to_numeric(result[y_col], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    point_geometries = np.asarray(
        [
            Point(x, y) if is_valid else Point()
            for x, y, is_valid in zip(
                x_values,
                y_values,
                valid,
                strict=True,
            )
        ],
        dtype=object,
    )
    tree = STRtree(polygons)
    matches = tree.query(point_geometries, predicate="covered_by")

    candidates: list[list[int]] = [[] for _ in range(len(result))]
    if matches.size:
        for point_index, polygon_index in zip(matches[0], matches[1], strict=True):
            candidates[int(point_index)].append(cell_ids[int(polygon_index)])

    proseg_values = pd.to_numeric(result[assignment_col], errors="coerce")
    assignments: list[int | None] = []
    sources: list[str] = []
    candidate_counts = np.zeros(len(result), dtype=np.uint16)
    for index, candidate_cells in enumerate(candidates):
        candidate_counts[index] = min(len(candidate_cells), np.iinfo(np.uint16).max)
        if len(candidate_cells) == 0:
            assignments.append(None)
            sources.append("outside")
            continue
        if len(candidate_cells) == 1:
            assignments.append(candidate_cells[0])
            sources.append("single_mask")
            continue
        proseg_value = proseg_values.iloc[index]
        proseg_cell = (
            assignment_map.get(int(proseg_value)) if pd.notna(proseg_value) else None
        )
        if proseg_cell is not None and proseg_cell in candidate_cells:
            assignments.append(proseg_cell)
            sources.append("proseg_overlap")
        else:
            assignments.append(None)
            sources.append("ambiguous_overlap")

    assignment_series = pd.Series(assignments, index=result.index, dtype="UInt64")
    result[HYBRID_ASSIGNMENT_COLUMN] = assignment_series
    result[HYBRID_BACKGROUND_COLUMN] = assignment_series.isna().to_numpy()
    result[HYBRID_CANDIDATE_COUNT_COLUMN] = candidate_counts
    result[HYBRID_ASSIGNMENT_SOURCE_COLUMN] = pd.Series(
        sources,
        index=result.index,
        dtype="category",
    )
    return result


def _config_json(config: ProsegHybridConfig) -> str:
    """Return a stable serialization for hybrid output reuse checks."""
    return json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))


def has_proseg_hybrid_refinement(
    zarr_path: Path | str,
    config: ProsegHybridConfig | None = None,
) -> bool:
    """Return whether a store contains the current complete hybrid branch."""
    path = Path(zarr_path)
    if not path.exists():
        return False
    sdata_obj = sd.read_zarr(path)
    if (
        PROSEG_HYBRID_SHAPE_NAME not in sdata_obj.shapes
        or PROSEG_HYBRID_TABLE_NAME not in sdata_obj.tables
        or len(sdata_obj.points) == 0
    ):
        return False
    metadata = sdata_obj.tables[PROSEG_HYBRID_TABLE_NAME].uns.get(
        "proseg_hybrid",
        {},
    )
    if metadata.get("algorithm") != PROSEG_HYBRID_ALGORITHM:
        return False
    if config is not None and metadata.get("config_json") != _config_json(config):
        return False
    points_key = choose_primary_points_key(sdata_obj)
    if points_key is None:
        return False
    points_obj = sdata_obj.points[points_key]
    return HYBRID_ASSIGNMENT_COLUMN in points_obj.columns


def run_proseg_hybrid_refinement(
    zarr_path: Path | str,
    cellpose_mask_path: Path | str,
    transforms_path: Path | str,
    config: ProsegHybridConfig,
) -> dict[str, int | float | str]:
    """Build and persist the hybrid shape, assignments, and count table.

    Args:
        zarr_path: Latest Proseg SpatialData zarr.
        cellpose_mask_path: Original Cellpose label image used as the prior.
        transforms_path: JSON file containing pixel-to-micron affine terms.
        config: Hybrid refinement configuration.

    Returns:
        Run-level refinement summary.
    """
    zarr_path = Path(zarr_path)
    transforms = json.loads(Path(transforms_path).read_text())
    x_values = tuple(float(value) for value in transforms["x_transform"])
    y_values = tuple(float(value) for value in transforms["y_transform"])
    if len(x_values) != 3 or len(y_values) != 3:
        raise ValueError("Cellpose transforms must each contain three values.")
    x_transform = cast(tuple[float, float, float], x_values)
    y_transform = cast(tuple[float, float, float], y_values)
    cellpose_polygons = _cellpose_polygons_in_microns(
        Path(cellpose_mask_path),
        x_transform,
        y_transform,
    )
    if not cellpose_polygons:
        raise RuntimeError("No Cellpose polygons were available for hybrid refinement.")

    sdata_obj = sd.read_zarr(zarr_path)
    points_key = choose_primary_points_key(sdata_obj)
    if points_key is None:
        raise RuntimeError("No transcript points element is available.")
    points_obj = sdata_obj.points[points_key]
    assignment_map = _proseg_assignment_to_instance_ids(sdata_obj)
    foreground_coordinates = _collect_foreground_coordinates(points_obj, assignment_map)

    cell_ids = sorted(cellpose_polygons)
    cell_results: dict[int, HybridCellGeometry] = {}
    records: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        coordinates = foreground_coordinates.get(
            cell_id,
            np.empty((0, 2), dtype=np.float64),
        )
        cellpose_polygon = cellpose_polygons[cell_id]
        cell_result = build_hybrid_cell_geometry(
            coordinates,
            cellpose_polygon,
            config,
        )
        cell_results[cell_id] = cell_result
        records.append(
            {
                INSTANCE_ID_COLUMN: np.uint64(cell_id),
                SOURCE_CELL_ID_COLUMN: f"cellpose_{cell_id}",
                "cellpose_label": int(cell_id),
                "proseg_foreground_transcripts": int(len(coordinates)),
                "retained_transcripts": int(cell_result.retained_count),
                "outlier_transcripts": int(cell_result.outlier_count),
                "neighbor_scale_um": float(cell_result.neighbor_scale),
                "candidate_external": int(cell_result.candidate_external),
                "cap_rejected_external": int(cell_result.cap_rejected_external),
                "unsupported_external": int(cell_result.unsupported_external),
                "near_surface_accepted": int(cell_result.near_surface_accepted),
                "chain_accepted": int(cell_result.chain_accepted),
                "supported_external": int(cell_result.supported_external),
                "accepted_groups": int(cell_result.accepted_groups),
                "fallback_reason": cell_result.fallback_reason,
                "cellpose_area_um2": float(cellpose_polygon.area),
                "pre_smoothing_area_um2": float(cell_result.pre_smoothing_area),
                "smoothing_added_area_um2": float(cell_result.smoothing_added_area),
                "smoothing_area_growth_fraction": float(
                    cell_result.area_growth_fraction
                ),
                "smoothing_perimeter_change_fraction": float(
                    cell_result.perimeter_change_fraction
                ),
                "holes_filled": int(cell_result.holes_filled),
                "hybrid_area_um2": float(cell_result.geometry.area),
                "geometry": cell_result.geometry,
            }
        )

    shape_gdf = gpd.GeoDataFrame(records, geometry="geometry")
    shape_gdf.index = pd.Index(
        shape_gdf[INSTANCE_ID_COLUMN].to_numpy(dtype=np.uint64),
        dtype="uint64",
        name=INSTANCE_ID_COLUMN,
    )
    shape_transformations = None
    if len(sdata_obj.shapes) > 0:
        template_shape = sdata_obj.shapes[list(sdata_obj.shapes.keys())[0]]
        shape_transformations = get_transformation(template_shape, get_all=True)
    hybrid_shapes = ShapesModel.parse(
        shape_gdf,
        transformations=shape_transformations,
    )

    x_col, y_col, gene_col, assignment_col, _ = _points_columns(points_obj)
    base_table = sdata_obj.tables["table"]
    genes = (
        base_table.var["gene"].astype(str).tolist()
        if "gene" in base_table.var.columns
        else base_table.var_names.astype(str).tolist()
    )
    cell_to_index = {int(cell_id): index for index, cell_id in enumerate(cell_ids)}
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    counts = sparse.csr_matrix((len(cell_ids), len(genes)), dtype=np.int64)
    hybrid_counts: defaultdict[int, int] = defaultdict(int)
    polygons = [cell_results[cell_id].geometry for cell_id in cell_ids]

    with tempfile.TemporaryDirectory(prefix="merxen-proseg-hybrid-") as temp_dir:
        temp_path = Path(temp_dir)
        for partition_index, partition in enumerate(_iter_point_partitions(points_obj)):
            augmented = assign_transcripts_to_hybrid_masks(
                partition,
                polygons,
                cell_ids,
                assignment_map,
                x_col=x_col,
                y_col=y_col,
                assignment_col=assignment_col,
            )
            assigned = augmented[HYBRID_ASSIGNMENT_COLUMN].notna()
            if bool(assigned.any()):
                assigned_cells = augmented.loc[
                    assigned,
                    HYBRID_ASSIGNMENT_COLUMN,
                ].astype("uint64")
                assigned_genes = augmented.loc[assigned, gene_col].astype(str)
                cell_indices = assigned_cells.map(cell_to_index).fillna(-1).astype(int)
                gene_indices = assigned_genes.map(gene_to_index).fillna(-1).astype(int)
                keep = (cell_indices >= 0) & (gene_indices >= 0)
                if bool(keep.any()):
                    chunk_counts = sparse.coo_matrix(
                        (
                            np.ones(int(keep.sum()), dtype=np.int64),
                            (
                                cell_indices.loc[keep].to_numpy(),
                                gene_indices.loc[keep].to_numpy(),
                            ),
                        ),
                        shape=counts.shape,
                    ).tocsr()
                    counts = counts + chunk_counts
                for cell_id, count in assigned_cells.value_counts().items():
                    hybrid_counts[int(cell_id)] += int(count)
            augmented.to_parquet(
                temp_path / f"part-{partition_index:05d}.parquet",
                index=False,
            )

        augmented_points = dd.read_parquet(temp_path)
        augmented_points = augmented_points.reset_index(drop=True)
        if TRANSCRIPT_ID_COLUMN not in augmented_points.columns:
            augmented_points = with_stable_transcript_ids(augmented_points)
        augmented_points[gene_col] = augmented_points[gene_col].astype(
            pd.CategoricalDtype(categories=genes)
        )
        point_transformations = get_transformation(points_obj, get_all=True)
        coordinate_mapping = {"x": x_col, "y": y_col}
        if "z" in augmented_points.columns:
            coordinate_mapping["z"] = "z"
        parsed_points = PointsModel.parse(
            augmented_points,
            coordinates=coordinate_mapping,
            feature_key=gene_col,
            transformations=point_transformations,
            instance_key=PROSEG_ASSIGNMENT_COLUMN,
            sort=True,
        )
        write_or_replace_element(
            sdata_obj,
            points_key,
            "points",
            parsed_points,
            overwrite=True,
        )

    obs = shape_gdf.drop(columns="geometry").copy()
    obs.index = pd.Index(
        np.asarray(cell_ids, dtype=np.uint64).astype(str),
        dtype=str,
        name="obs_id",
    )
    obs[INSTANCE_ID_COLUMN] = np.asarray(cell_ids, dtype=np.uint64)
    obs["hybrid_assigned_transcripts"] = [
        hybrid_counts.get(cell_id, 0) for cell_id in cell_ids
    ]
    obs["region"] = pd.Categorical(
        [PROSEG_HYBRID_SHAPE_NAME] * len(obs),
        categories=[PROSEG_HYBRID_SHAPE_NAME],
    )
    var = pd.DataFrame(index=pd.Index(genes, dtype=str, name="gene"))
    var["gene"] = var.index.astype(str)
    table_adata = ad.AnnData(X=counts, obs=obs, var=var)
    table_adata.uns["proseg_hybrid"] = {
        "algorithm": PROSEG_HYBRID_ALGORITHM,
        "config_json": _config_json(config),
        **config.model_dump(),
    }
    hybrid_table = TableModel.parse(
        table_adata,
        region=PROSEG_HYBRID_SHAPE_NAME,
        region_key="region",
        instance_key=INSTANCE_ID_COLUMN,
    )
    write_or_replace_element(
        sdata_obj,
        PROSEG_HYBRID_SHAPE_NAME,
        "shapes",
        hybrid_shapes,
        overwrite=True,
    )
    write_or_replace_element(
        sdata_obj,
        PROSEG_HYBRID_TABLE_NAME,
        "tables",
        hybrid_table,
        overwrite=True,
    )
    stamp_merxen_schema(sdata_obj, primary_points_key=points_key)
    register_segmentation_branch(
        sdata_obj,
        "proseg_hybrid",
        points_key=points_key,
        assignment_column=HYBRID_ASSIGNMENT_COLUMN,
        background_column=HYBRID_BACKGROUND_COLUMN,
        assignment_source_column=HYBRID_ASSIGNMENT_SOURCE_COLUMN,
        shape_key=PROSEG_HYBRID_SHAPE_NAME,
        table_key=PROSEG_HYBRID_TABLE_NAME,
        id_namespace=PROSEG_ID_NAMESPACE,
    )
    validate_merxen_schema(sdata_obj, deep=False)
    write_spatialdata_metadata(sdata_obj, write_attrs=True)

    n_fallback = int(
        sum(bool(result.fallback_reason) for result in cell_results.values())
    )
    n_supported_external = int(
        sum(result.supported_external for result in cell_results.values())
    )
    summary: dict[str, int | float | str] = {
        "shape_key": PROSEG_HYBRID_SHAPE_NAME,
        "table_key": PROSEG_HYBRID_TABLE_NAME,
        "algorithm": PROSEG_HYBRID_ALGORITHM,
        "n_cells": len(cell_ids),
        "n_fallback_cellpose": n_fallback,
        "n_supported_external": n_supported_external,
        "n_hybrid_assigned_transcripts": int(sum(hybrid_counts.values())),
    }
    log_status(
        "[Proseg hybrid] wrote "
        f"{len(cell_ids):,} cells; Cellpose fallbacks={n_fallback:,}; "
        f"supported external transcripts={n_supported_external:,}; "
        f"assigned transcripts={sum(hybrid_counts.values()):,}"
    )
    del sdata_obj, points_obj, counts
    force_release(note="after Proseg hybrid refinement")
    return summary
