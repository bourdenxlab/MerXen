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
from merxen.platforms import PLATFORM_SPECS

BoundaryType = Literal["cell", "nucleus"]

ORIGINAL_NUCLEUS_ROW_ID_NAMESPACE = "original_nucleus_row_v1"
"""Id namespace of vendor nuclei keyed by nucleus row (``instance_id`` = row + 1).

Used by platforms whose nucleus ids are not cell ids (ATERA); their nuclei join
cells through :data:`NUCLEUS_CELL_KEY`.
"""

NUCLEUS_CELL_KEY = "cell_instance_id"
"""Nucleus-shape column naming the owning cell when nucleus ids are not cell ids."""


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
    cell_key: str | None = None

    @property
    def has_assignment(self: MaskSpec) -> bool:
        """Return whether transcripts carry an assignment for this mask."""
        return self.assignment_column is not None

    @property
    def has_table(self: MaskSpec) -> bool:
        """Return whether this mask has a registered expression table."""
        return self.table_key is not None


@dataclass(frozen=True)
class _NucleusBranch:
    """A nucleus mask branch registered when its shapes exist but it is missing.

    Attributes:
        branch: Registry name of the branch.
        shape_key: Shapes element of the nuclei.
        aliases: Legacy names of the branch.
        id_namespace: Namespace of the nucleus ids.
        cell_key: Shape column naming the owning cell, or ``None`` when the
            nucleus ids are the cell ids.
    """

    branch: str
    shape_key: str
    aliases: tuple[str, ...]
    id_namespace: str
    cell_key: str | None = None


def _original_nucleus_branches() -> tuple[_NucleusBranch, ...]:
    """Return the vendor nucleus branches of the registry, in registry order."""
    branches: list[_NucleusBranch] = []
    for spec in PLATFORM_SPECS.values():
        if spec.original_nucleus_shape_key is None:
            continue
        if spec.nucleus_ids_are_cell_ids:
            # Xenium nuclei share the cell ids; keep the namespace and alias
            # older stores were registered with.
            branches.append(
                _NucleusBranch(
                    "original_nucleus",
                    spec.original_nucleus_shape_key,
                    ("nucleus_boundaries",),
                    f"{ORIGINAL_ID_NAMESPACE}_nucleus",
                )
            )
        else:
            branches.append(
                _NucleusBranch(
                    "original_nucleus",
                    spec.original_nucleus_shape_key,
                    (),
                    ORIGINAL_NUCLEUS_ROW_ID_NAMESPACE,
                    cell_key=NUCLEUS_CELL_KEY,
                )
            )
    return tuple(branches)


_NUCLEUS_BRANCHES: tuple[_NucleusBranch, ...] = (
    _NucleusBranch(
        "cellpose_nuclei", "cellpose_nuclei", ("nuclei",), "cellpose_nucleus_label_v1"
    ),
    *_original_nucleus_branches(),
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
    for candidate in _NUCLEUS_BRANCHES:
        if candidate.shape_key not in getattr(sdata_obj, "shapes", {}):
            continue
        if candidate.branch in registry:
            continue
        register_segmentation_branch(
            sdata_obj,
            candidate.branch,
            points_key=points_key,
            assignment_column=None,
            shape_key=candidate.shape_key,
            table_key=None,
            id_namespace=candidate.id_namespace,
            legacy_aliases=candidate.aliases,
            cell_key=candidate.cell_key,
        )
        registry = dict(
            sdata_obj.attrs.get(MERXEN_SCHEMA_ATTR, {}).get("segmentations", {})
        )
        added.append(candidate.branch)
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
                cell_key=(
                    None if entry.get("cell_key") is None else str(entry["cell_key"])
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
