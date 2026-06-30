"""Boundary annotation readers for cortical-depth computation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

ROLE_ALIASES: dict[str, set[str]] = {
    "pia": {"pia", "pial", "pial_boundary", "surface_pia", "depth_0"},
    "wm": {
        "wm",
        "white",
        "white_matter",
        "grey_white",
        "gray_white",
        "gm_wm",
        "grey_white_boundary",
        "gray_white_boundary",
        "depth_1",
    },
    "side": {"side", "edge", "tissue_edge", "side_boundary", "artificial"},
    "exclusion": {"exclude", "exclusion", "mask", "tear", "fold", "vessel"},
    "ribbon": {"ribbon", "cortical_ribbon", "cortex", "cortical_mask"},
}


@dataclass(frozen=True)
class BoundaryAnnotations:
    """Boundary geometry needed to build a 2D cortical ribbon."""

    pial: LineString
    wm: LineString
    side_boundaries: tuple[LineString, ...] = ()
    exclusions: tuple[Polygon, ...] = ()
    ribbon: Polygon | MultiPolygon | None = None


def load_boundary_annotations(
    *,
    pial_path: Path | str | None = None,
    wm_path: Path | str | None = None,
    side_boundary_path: Path | str | None = None,
    exclusion_path: Path | str | None = None,
    ribbon_path: Path | str | None = None,
    annotation_path: Path | str | None = None,
    smoothing_window: int = 0,
) -> BoundaryAnnotations:
    """Load pial/white-matter boundaries and optional masks from GeoJSON files.

    Args:
        pial_path: GeoJSON containing a pial boundary line. Required unless
            ``annotation_path`` contains a feature with a pial role.
        wm_path: GeoJSON containing a gray/white matter boundary line. Required
            unless ``annotation_path`` contains a feature with a white-matter role.
        side_boundary_path: Optional GeoJSON with artificial side boundaries.
        exclusion_path: Optional GeoJSON with exclusion polygons.
        ribbon_path: Optional GeoJSON with a complete cortical ribbon polygon.
        annotation_path: Optional combined GeoJSON. Feature properties named
            ``role``, ``type``, ``name``, ``label``, or ``classification`` are
            matched against documented role aliases.
        smoothing_window: Optional moving-average window for line coordinates.

    Returns:
        Parsed and validated boundary annotations.

    Raises:
        FileNotFoundError: If a requested path is missing.
        ValueError: If required pial or white-matter lines cannot be resolved.
    """
    combined = _read_feature_geometries(annotation_path) if annotation_path else []

    pial = (
        read_line_annotation(pial_path, role="pia")
        if pial_path is not None
        else _line_from_features(combined, "pia")
    )
    wm = (
        read_line_annotation(wm_path, role="wm")
        if wm_path is not None
        else _line_from_features(combined, "wm")
    )
    if pial is None:
        raise ValueError("Missing pial boundary annotation.")
    if wm is None:
        raise ValueError("Missing gray/white matter boundary annotation.")

    pial = smooth_line(pial, smoothing_window)
    wm = smooth_line(wm, smoothing_window)
    _validate_line(pial, "pial boundary")
    _validate_line(wm, "gray/white matter boundary")

    side_lines = []
    if side_boundary_path is not None:
        side_lines.extend(read_line_annotations(side_boundary_path, role="side"))
    side_lines.extend(_lines_from_features(combined, "side"))
    side_lines = [smooth_line(line, smoothing_window) for line in side_lines]

    exclusions = []
    if exclusion_path is not None:
        exclusions.extend(read_polygon_annotations(exclusion_path, role="exclusion"))
    exclusions.extend(_polygons_from_features(combined, "exclusion"))

    ribbon = None
    if ribbon_path is not None:
        ribbon_polygons = read_polygon_annotations(ribbon_path, role="ribbon")
        ribbon = _union_polygons(ribbon_polygons) if ribbon_polygons else None
    if ribbon is None:
        ribbon_polygons = _polygons_from_features(combined, "ribbon")
        ribbon = _union_polygons(ribbon_polygons) if ribbon_polygons else None

    return BoundaryAnnotations(
        pial=pial,
        wm=wm,
        side_boundaries=tuple(side_lines),
        exclusions=tuple(exclusions),
        ribbon=ribbon,
    )


def read_line_annotation(
    path: Path | str | None,
    *,
    role: str | None = None,
) -> LineString | None:
    """Read the longest line annotation from a GeoJSON file."""
    lines = read_line_annotations(path, role=role)
    if not lines:
        return None
    return max(lines, key=lambda line: float(line.length))


def read_line_annotations(
    path: Path | str | None,
    *,
    role: str | None = None,
) -> list[LineString]:
    """Read all line annotations from a GeoJSON file."""
    if path is None:
        return []
    features = _read_feature_geometries(path)
    if role is not None:
        role_matches = _lines_from_features(features, role)
        if role_matches:
            return role_matches
    lines: list[LineString] = []
    for geom, _properties in features:
        lines.extend(_coerce_lines(geom))
    return [_validate_line(line, str(path)) for line in lines if line.length > 0]


def read_polygon_annotations(
    path: Path | str | None,
    *,
    role: str | None = None,
) -> list[Polygon]:
    """Read polygon annotations from a GeoJSON file."""
    if path is None:
        return []
    features = _read_feature_geometries(path)
    if role is not None:
        role_matches = _polygons_from_features(features, role)
        if role_matches:
            return role_matches
    polygons: list[Polygon] = []
    for geom, _properties in features:
        polygons.extend(_coerce_polygons(geom))
    return [_validate_polygon(poly, str(path)) for poly in polygons if poly.area > 0]


def smooth_line(line: LineString, window: int = 0) -> LineString:
    """Smooth a polyline with a centered moving average while preserving endpoints."""
    width = int(window)
    if width <= 1:
        return line
    if width % 2 == 0:
        width += 1
    coords = np.asarray(line.coords, dtype=float)
    if coords.shape[0] <= width:
        return line

    pad = width // 2
    padded = np.pad(coords, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    smoothed = np.column_stack(
        [
            np.convolve(padded[:, dim], kernel, mode="valid")
            for dim in range(coords.shape[1])
        ]
    )
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return LineString(smoothed)


def _read_feature_geometries(
    path: Path | str | None,
) -> list[tuple[BaseGeometry, dict[str, Any]]]:
    if path is None:
        return []
    geojson_path = Path(path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"Missing GeoJSON annotation: {geojson_path}")

    data = json.loads(geojson_path.read_text())
    if data.get("type") == "FeatureCollection":
        raw_features = data.get("features", [])
    elif data.get("type") == "Feature":
        raw_features = [data]
    else:
        raw_features = [{"type": "Feature", "properties": {}, "geometry": data}]

    features: list[tuple[BaseGeometry, dict[str, Any]]] = []
    for feature in raw_features:
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        geom = shape(geometry)
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        features.append((geom, properties))
    return features


def _line_from_features(
    features: list[tuple[BaseGeometry, dict[str, Any]]],
    role: str,
) -> LineString | None:
    lines = _lines_from_features(features, role)
    if not lines:
        return None
    return max(lines, key=lambda line: float(line.length))


def _lines_from_features(
    features: list[tuple[BaseGeometry, dict[str, Any]]],
    role: str,
) -> list[LineString]:
    lines: list[LineString] = []
    for geom, properties in features:
        if _feature_matches_role(properties, role):
            lines.extend(_coerce_lines(geom))
    return [_validate_line(line, role) for line in lines if line.length > 0]


def _polygons_from_features(
    features: list[tuple[BaseGeometry, dict[str, Any]]],
    role: str,
) -> list[Polygon]:
    polygons: list[Polygon] = []
    for geom, properties in features:
        if _feature_matches_role(properties, role):
            polygons.extend(_coerce_polygons(geom))
    return [_validate_polygon(poly, role) for poly in polygons if poly.area > 0]


def _feature_matches_role(properties: dict[str, Any], role: str) -> bool:
    aliases = ROLE_ALIASES.get(role, {role})
    values: list[str] = []
    for key in ("role", "type", "name", "label", "boundary", "boundary_type"):
        value = properties.get(key)
        if value is not None:
            values.append(str(value))
    classification = properties.get("classification")
    if isinstance(classification, dict):
        for key in ("name", "label", "role"):
            value = classification.get(key)
            if value is not None:
                values.append(str(value))
    elif classification is not None:
        values.append(str(classification))

    normalized_values = {_normalize_role(value) for value in values}
    normalized_aliases = {_normalize_role(alias) for alias in aliases}
    return bool(normalized_values & normalized_aliases)


def _normalize_role(value: str) -> str:
    return (
        str(value).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    )


def _coerce_lines(geom: BaseGeometry) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return [merged]
        if isinstance(merged, MultiLineString):
            return list(merged.geoms)
    if isinstance(geom, Polygon):
        return [LineString(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [LineString(poly.exterior.coords) for poly in geom.geoms]
    if isinstance(geom, GeometryCollection):
        lines: list[LineString] = []
        for subgeom in geom.geoms:
            lines.extend(_coerce_lines(subgeom))
        return lines
    return []


def _coerce_polygons(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [_validate_polygon(geom, "polygon")]
    if isinstance(geom, MultiPolygon):
        return [_validate_polygon(poly, "multipolygon") for poly in geom.geoms]
    if isinstance(geom, GeometryCollection):
        polygons: list[Polygon] = []
        for subgeom in geom.geoms:
            polygons.extend(_coerce_polygons(subgeom))
        return polygons
    return []


def _validate_line(line: LineString, label: str) -> LineString:
    if line.is_empty or line.length <= 0:
        raise ValueError(f"{label} is empty or has zero length.")
    coords = np.asarray(line.coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        raise ValueError(f"{label} must contain at least two 2D coordinates.")
    if not np.isfinite(coords[:, :2]).all():
        raise ValueError(f"{label} contains non-finite coordinates.")
    return LineString(coords[:, :2])


def _validate_polygon(poly: Polygon, label: str) -> Polygon:
    if poly.is_empty or poly.area <= 0:
        raise ValueError(f"{label} polygon is empty or has zero area.")
    if poly.is_valid:
        return poly
    repaired = poly.buffer(0)
    if repaired.is_empty or repaired.area <= 0:
        raise ValueError(f"{label} polygon is invalid and could not be repaired.")
    if isinstance(repaired, Polygon):
        return repaired
    if isinstance(repaired, MultiPolygon):
        return max(repaired.geoms, key=lambda geom: float(geom.area))
    raise ValueError(f"{label} polygon repair produced {repaired.geom_type}.")


def _union_polygons(
    polygons: list[Polygon],
) -> Polygon | MultiPolygon | None:
    if not polygons:
        return None
    merged = unary_union(polygons)
    if isinstance(merged, Polygon | MultiPolygon):
        return merged
    polygon_parts = _coerce_polygons(merged)
    if not polygon_parts:
        return None
    return unary_union(polygon_parts)
