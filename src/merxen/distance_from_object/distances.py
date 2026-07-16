"""Nearest polygon-edge distance assignment for cell centroids."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

from merxen.cortical_depth.assign_cells import CellCoordinateTable
from merxen.distance_from_object.annotations import ObjectAnnotation

DISTANCE_FROM_OBJECT_COLUMNS = [
    "nearest_object_id",
    "nearest_object_type",
    "distance_to_object_edge_um",
    "signed_distance_to_object_edge_um",
    "inside_object",
    "nearest_object_edge_x",
    "nearest_object_edge_y",
    "object_proximity",
    "distance_from_object_qc_flag",
]


def assign_distances_to_objects(
    coordinates: CellCoordinateTable,
    annotations: list[ObjectAnnotation],
    *,
    coordinate_unit_um: float = 1.0,
    near_distance_um: float = 50.0,
    far_distance_um: float = 100.0,
    max_distance_um: float = 200.0,
) -> pd.DataFrame:
    """Assign each cell centroid to its nearest annotated polygon edge.

    Cells covered by an object polygon receive a negative signed distance and
    are included in the ``near`` proximity class. Unsigned edge distance remains
    available for direct interpretation.

    Args:
        coordinates: Cell IDs and registered x/y centroid coordinates.
        annotations: Named polygon objects in the same coordinate system.
        coordinate_unit_um: Micrometres represented by one coordinate unit.
        near_distance_um: Upper exclusive distance for ``near``.
        far_distance_um: Lower inclusive distance for ``far``.
        max_distance_um: Upper inclusive distance retained as ``far``.

    Returns:
        Per-cell nearest-object assignments indexed by normalized cell ID.

    Raises:
        ValueError: If no objects are supplied or thresholds are inconsistent.
    """
    if not annotations:
        raise ValueError("At least one object annotation is required.")
    if coordinate_unit_um <= 0:
        raise ValueError("coordinate_unit_um must be positive.")
    if not 0 <= near_distance_um < far_distance_um < max_distance_um:
        raise ValueError("Distance thresholds must satisfy 0 <= near < far < max.")

    polygons = [annotation.geometry for annotation in annotations]
    boundaries = [polygon.boundary for polygon in polygons]
    polygon_tree = STRtree(polygons)
    boundary_tree = STRtree(boundaries)
    coordinate_values = np.asarray(coordinates.coordinates, dtype=float)
    output = _empty_assignments(
        coordinates.cell_ids.astype(str),
        len(coordinate_values),
    )
    finite = np.isfinite(coordinate_values[:, :2]).all(axis=1)
    if not finite.any():
        return output

    finite_positions = np.flatnonzero(finite)
    point_geometries = shapely.points(coordinate_values[finite, :2])
    boundary_array = np.asarray(boundaries, dtype=object)
    nearest_pairs, raw_distances = boundary_tree.query_nearest(
        point_geometries,
        all_matches=False,
        return_distance=True,
    )
    nearest_pairs = np.asarray(nearest_pairs, dtype=int)
    object_indices = np.empty(len(point_geometries), dtype=int)
    edge_distances = np.empty(len(point_geometries), dtype=float)
    object_indices[nearest_pairs[0]] = nearest_pairs[1]
    edge_distances[nearest_pairs[0]] = np.asarray(raw_distances, dtype=float)

    # For overlapping polygons, an interior cell belongs to the containing
    # object whose own boundary is closest, not an unrelated nearby boundary.
    containing_pairs = np.asarray(
        polygon_tree.query(point_geometries, predicate="covered_by"),
        dtype=int,
    )
    is_inside = np.zeros(len(point_geometries), dtype=bool)
    if containing_pairs.size:
        candidate_distances = np.asarray(
            shapely.distance(
                point_geometries[containing_pairs[0]],
                boundary_array[containing_pairs[1]],
            ),
            dtype=float,
        )
        order = np.lexsort((candidate_distances, containing_pairs[0]))
        sorted_points = containing_pairs[0, order]
        first_for_point = np.concatenate(
            ([True], sorted_points[1:] != sorted_points[:-1])
        )
        chosen = order[first_for_point]
        inside_points = containing_pairs[0, chosen]
        object_indices[inside_points] = containing_pairs[1, chosen]
        edge_distances[inside_points] = candidate_distances[chosen]
        is_inside[inside_points] = True

    selected_boundaries = boundary_array[object_indices]
    shortest_lines = shapely.shortest_line(point_geometries, selected_boundaries)
    edge_points = shapely.get_point(shortest_lines, 1)
    distance_um = edge_distances * float(coordinate_unit_um)
    proximity = _label_object_proximity_array(
        distance_um,
        is_inside=is_inside,
        near_distance_um=near_distance_um,
        far_distance_um=far_distance_um,
        max_distance_um=max_distance_um,
    )
    object_ids = np.asarray(
        [annotation.object_id for annotation in annotations],
        dtype=object,
    )[object_indices]
    object_types = np.asarray(
        [annotation.object_type for annotation in annotations],
        dtype=object,
    )[object_indices]
    _set_finite_values(output, "nearest_object_id", finite_positions, object_ids)
    _set_finite_values(output, "nearest_object_type", finite_positions, object_types)
    _set_finite_values(
        output,
        "distance_to_object_edge_um",
        finite_positions,
        distance_um,
    )
    _set_finite_values(
        output,
        "signed_distance_to_object_edge_um",
        finite_positions,
        np.where(is_inside, -distance_um, distance_um),
    )
    _set_finite_values(output, "inside_object", finite_positions, is_inside)
    _set_finite_values(
        output,
        "nearest_object_edge_x",
        finite_positions,
        np.asarray(shapely.get_x(edge_points), dtype=float),
    )
    _set_finite_values(
        output,
        "nearest_object_edge_y",
        finite_positions,
        np.asarray(shapely.get_y(edge_points), dtype=float),
    )
    _set_finite_values(output, "object_proximity", finite_positions, proximity)
    _set_finite_values(
        output,
        "distance_from_object_qc_flag",
        finite_positions,
        np.full(len(finite_positions), "assigned", dtype=object),
    )
    output["inside_object"] = output["inside_object"].astype(bool)
    return output


def label_object_proximity(
    distance_um: float,
    *,
    is_inside: bool,
    near_distance_um: float,
    far_distance_um: float,
    max_distance_um: float,
) -> str:
    """Return the configured near/middle/far/beyond distance label."""
    if is_inside or distance_um < near_distance_um:
        return "near"
    if distance_um < far_distance_um:
        return "middle"
    if distance_um <= max_distance_um:
        return "far"
    return "beyond_max"


def apply_distance_columns(table: Any, assignments: pd.DataFrame) -> Any:
    """Return a copy of an AnnData table with distance columns in ``obs``."""
    out = table.copy()
    cell_ids = _table_cell_ids(out)
    aligned = assignments.reindex(cell_ids)
    for column in DISTANCE_FROM_OBJECT_COLUMNS:
        if column in aligned.columns:
            out.obs[column] = aligned[column].to_numpy()
    return out


def _label_object_proximity_array(
    distance_um: np.ndarray,
    *,
    is_inside: np.ndarray,
    near_distance_um: float,
    far_distance_um: float,
    max_distance_um: float,
) -> np.ndarray:
    labels = np.full(len(distance_um), "beyond_max", dtype=object)
    labels[distance_um <= max_distance_um] = "far"
    labels[distance_um < far_distance_um] = "middle"
    labels[(distance_um < near_distance_um) | is_inside] = "near"
    return labels


def _set_finite_values(
    output: pd.DataFrame,
    column: str,
    positions: np.ndarray,
    values: np.ndarray,
) -> None:
    all_values = output[column].to_numpy(copy=True)
    all_values[positions] = values
    output[column] = all_values


def _empty_assignments(cell_ids: pd.Index, n_cells: int) -> pd.DataFrame:
    output = pd.DataFrame(index=pd.Index(cell_ids, dtype=str, name="cell_id"))
    output["nearest_object_id"] = pd.Series(
        [None] * n_cells, index=output.index, dtype=object
    )
    output["nearest_object_type"] = pd.Series(
        [None] * n_cells, index=output.index, dtype=object
    )
    for column in (
        "distance_to_object_edge_um",
        "signed_distance_to_object_edge_um",
        "nearest_object_edge_x",
        "nearest_object_edge_y",
    ):
        output[column] = np.nan
    output["inside_object"] = np.zeros(n_cells, dtype=bool)
    output["object_proximity"] = pd.Series(
        [None] * n_cells, index=output.index, dtype=object
    )
    output["distance_from_object_qc_flag"] = "missing_coordinate"
    return output


def _table_cell_ids(table: Any) -> pd.Index:
    if "cell_id" in table.obs.columns:
        return pd.Index(table.obs["cell_id"].astype(str), dtype=str)
    return pd.Index(table.obs_names.astype(str), dtype=str)
