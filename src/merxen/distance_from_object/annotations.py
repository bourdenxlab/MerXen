"""Readers and writers for registered polygon object annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, mapping, shape


@dataclass(frozen=True)
class ObjectAnnotation:
    """One named polygon object used for nearest-edge assignment."""

    object_id: str
    object_type: str
    geometry: Polygon


def load_object_annotations(
    path: Path | str,
    *,
    object_types: list[str] | None = None,
) -> list[ObjectAnnotation]:
    """Load validated polygon objects from a registered GeoJSON file.

    Args:
        path: GeoJSON FeatureCollection containing Polygon or MultiPolygon
            features in the same coordinate system as the cell tables.
        object_types: Optional object types to retain, matched case-insensitively.

    Returns:
        Polygon objects with unique IDs and non-empty type names.

    Raises:
        FileNotFoundError: If the annotation path does not exist.
        ValueError: If no usable polygons remain or geometry/properties are invalid.
    """
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing object annotation: {annotation_path}")
    payload = json.loads(annotation_path.read_text())
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": payload}]

    selected = (
        None
        if object_types is None
        else {str(value).strip().casefold() for value in object_types}
    )
    annotations: list[ObjectAnnotation] = []
    seen_ids: set[str] = set()
    type_counts: dict[str, int] = {}
    for feature_index, feature in enumerate(features, start=1):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        object_type = _object_type(properties)
        if selected is not None and object_type.casefold() not in selected:
            continue
        geom = shape(geometry)
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            raise ValueError(
                f"Object feature {feature_index} must be Polygon or MultiPolygon, "
                f"not {geom.geom_type}."
            )
        base_id = str(properties.get("object_id") or "").strip()
        for polygon_index, polygon in enumerate(polygons, start=1):
            _validate_polygon(polygon, feature_index)
            type_counts[object_type] = type_counts.get(object_type, 0) + 1
            object_id = base_id
            if len(polygons) > 1 and object_id:
                object_id = f"{object_id}_{polygon_index}"
            if not object_id:
                object_id = (
                    f"{_safe_identifier(object_type)}_{type_counts[object_type]:04d}"
                )
            if object_id in seen_ids:
                raise ValueError(f"Duplicate object_id {object_id!r}.")
            seen_ids.add(object_id)
            annotations.append(
                ObjectAnnotation(
                    object_id=object_id,
                    object_type=object_type,
                    geometry=polygon,
                )
            )
    if not annotations:
        selected_message = "" if object_types is None else f" for types {object_types}"
        raise ValueError(
            f"No usable polygon object annotations found in {annotation_path}"
            f"{selected_message}."
        )
    return annotations


def object_annotations_to_geojson(
    annotations: list[ObjectAnnotation],
) -> dict[str, Any]:
    """Return normalized GeoJSON for validated object annotations."""
    return {
        "type": "FeatureCollection",
        "object_annotation_schema": "merxen.distance_from_object/v1",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "role": "analysis_object",
                    "annotation_role": "object",
                    "object_id": annotation.object_id,
                    "object_type": annotation.object_type,
                    "name": annotation.object_type,
                },
                "geometry": mapping(annotation.geometry),
            }
            for annotation in annotations
        ],
    }


def write_object_annotations(
    path: Path | str,
    annotations: list[ObjectAnnotation],
) -> Path:
    """Write normalized object annotations and return the output path."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(object_annotations_to_geojson(annotations), indent=2) + "\n"
    )
    return output_path


def _object_type(properties: dict[str, Any]) -> str:
    classification = properties.get("classification")
    classification_name = (
        classification.get("name") if isinstance(classification, dict) else None
    )
    value = (
        properties.get("object_type")
        or properties.get("name")
        or properties.get("label")
        or classification_name
        or "objects"
    )
    object_type = str(value).strip()
    if not object_type:
        raise ValueError("Object annotation type/name must not be blank.")
    return object_type


def _validate_polygon(polygon: Polygon, feature_index: int) -> None:
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError(f"Object feature {feature_index} has zero polygon area.")
    if not polygon.is_valid:
        raise ValueError(
            f"Object feature {feature_index} is invalid or self-intersecting."
        )


def _safe_identifier(value: str) -> str:
    safe = (
        "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(value)
        )
        .strip("_")
        .lower()
    )
    return safe or "object"
