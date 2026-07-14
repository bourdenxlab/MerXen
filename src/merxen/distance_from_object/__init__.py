"""Distance-from-object analysis for registered polygon annotations."""

from merxen.distance_from_object.annotations import (
    ObjectAnnotation,
    load_object_annotations,
)
from merxen.distance_from_object.distances import assign_distances_to_objects

__all__ = [
    "ObjectAnnotation",
    "assign_distances_to_objects",
    "load_object_annotations",
]
