"""Canonical segmentation-mask inventory shared by caches and alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    ORIGINAL_ID_NAMESPACE,
    choose_primary_points_key,
    register_segmentation_branch,
)

BoundaryType = Literal["cell", "nucleus"]


@dataclass(frozen=True)
class MaskSpec:
    """One explicitly registered polygon mask and its assignment semantics."""

    branch: str
    shape_key: str
    boundary_type: BoundaryType
    points_key: str
    assignment_column: str | None
    table_key: str | None
    instance_key: str
    id_namespace: str
    coordinate_variant_of: str | None = None

    @property
    def has_assignment(self: MaskSpec) -> bool:
        """Return whether transcripts carry an assignment for this mask."""
        return self.assignment_column is not None

    @property
    def has_table(self: MaskSpec) -> bool:
        """Return whether this mask has a registered expression table."""
        return self.table_key is not None


_NUCLEUS_BRANCHES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("cellpose_nuclei", "cellpose_nuclei", ("nuclei",)),
    ("original_nucleus", "xenium_nucleus", ("nucleus_boundaries",)),
)


def register_missing_mask_branches(sdata_obj: Any) -> tuple[str, ...]:
    """Register displayable nucleus masks omitted by older schema writers.

    Cell masks are already registered by enrichment's assignment stage. Nucleus
    masks intentionally carry neither a transcript assignment nor a count table;
    registering those facts makes them discoverable without inventing cell
    semantics for nuclear boundaries.

    Args:
        sdata_obj: SpatialData-like object whose schema should be updated.

    Returns:
        Branch names added to ``merxen_schema.segmentations``.
    """
    points_key = choose_primary_points_key(sdata_obj)
    if points_key is None:
        return ()
    schema = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    registry = dict(schema.get("segmentations", {}))
    added: list[str] = []
    for branch, shape_key, aliases in _NUCLEUS_BRANCHES:
        if shape_key not in getattr(sdata_obj, "shapes", {}):
            continue
        if branch in registry:
            continue
        register_segmentation_branch(
            sdata_obj,
            branch,
            points_key=points_key,
            assignment_column=None,
            shape_key=shape_key,
            table_key=None,
            id_namespace=(
                "cellpose_nucleus_label_v1"
                if branch == "cellpose_nuclei"
                else f"{ORIGINAL_ID_NAMESPACE}_nucleus"
            ),
            legacy_aliases=aliases,
        )
        registry = dict(
            sdata_obj.attrs.get(MERXEN_SCHEMA_ATTR, {}).get("segmentations", {})
        )
        added.append(branch)
    return tuple(added)


def discover_masks(
    sdata_obj: Any,
    *,
    include_coordinate_variants: bool = False,
) -> tuple[MaskSpec, ...]:
    """Return de-duplicated masks declared by ``merxen_schema.segmentations``.

    Only explicit registry entries are returned. This deliberately differs from
    vector alignment, which continues to transform unknown native shapes for
    backwards compatibility but must not make them viewer-visible by accident.

    Args:
        sdata_obj: SpatialData-like object containing shapes and MerXen attrs.
        include_coordinate_variants: Include aligned coordinate variants.

    Returns:
        Stable branch-ordered mask specifications.
    """
    schema = dict(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    registry = dict(schema.get("segmentations", {}))
    shapes = getattr(sdata_obj, "shapes", {})
    registered_shape_keys = {
        str(entry.get("shape", ""))
        for entry in registry.values()
        if isinstance(entry, dict)
    }
    seen: set[str] = set()
    masks: list[MaskSpec] = []
    for branch, raw_entry in sorted(registry.items()):
        entry = dict(raw_entry)
        coordinate_variant_of = entry.get("coordinate_variant_of")
        if coordinate_variant_of is not None and not include_coordinate_variants:
            continue
        shape_key = str(entry.get("shape", ""))
        if not shape_key or shape_key not in shapes:
            continue
        deduplication_key = _canonical_shape_key(shape_key, registered_shape_keys)
        if shape_key != deduplication_key:
            continue
        if deduplication_key in seen:
            continue
        seen.add(deduplication_key)
        masks.append(
            MaskSpec(
                branch=str(branch),
                shape_key=shape_key,
                boundary_type=_boundary_type(branch=str(branch), shape_key=shape_key),
                points_key=str(entry.get("points", "")),
                assignment_column=entry.get("assignment_column"),
                table_key=entry.get("table"),
                instance_key=str(entry.get("instance_key", "instance_id")),
                id_namespace=str(entry.get("id_namespace", branch)),
                coordinate_variant_of=(
                    None
                    if coordinate_variant_of is None
                    else str(coordinate_variant_of)
                ),
            )
        )
    return tuple(masks)


def _canonical_shape_key(shape_key: str, available: set[str]) -> str:
    """Collapse the legacy ProSeg ``cell_boundaries`` duplicate when possible."""
    if shape_key == "cell_boundaries" and "MOSAIK_proseg" in available:
        return "MOSAIK_proseg"
    return shape_key


def _boundary_type(*, branch: str, shape_key: str) -> BoundaryType:
    token = f"{branch} {shape_key}".lower()
    return "nucleus" if "nucle" in token else "cell"
