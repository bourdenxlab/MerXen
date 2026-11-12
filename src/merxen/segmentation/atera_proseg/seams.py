"""Seam clipping of the stitched ProSeg polygons (AT-9 follow-up, plan AT-7).

Each tile's ProSeg run draws the polygons of the cells it owns without seeing
the cells another tile owns, so two cells of neighbouring tiles can overlap
across a seam (AT-9's full section: 174 cross-tile pairs above 1 um^2, at most
91.5 um^2). The transcripts in such an overlap already follow the core rule
of :func:`merxen.segmentation.atera_proseg.stitch.resolve_claims`: a row
claimed by two tiles goes to the tile whose core holds it, else to the tile
earlier in plan order. :func:`clip_seam_overlaps` applies the same rule to the
area: of two overlapping cells of different tiles, each keeps the part of the
overlap its tile would win and loses the rest (``shapely`` difference). In the
usual case, an overlap lying wholly in one tile's core, that is the winner's
polygon subtracted from the loser's.

Only cells that lose area change; every other geometry is passed through as it
is. Clipped geometries stay MultiPolygons (ProSeg's polygon type), keeping
every polygonal part of positive area. A cell left with no area is reported as
emptied; the stitch leaves it out of the layer and its transcripts become
background, as for an owned cell ProSeg gave no polygon. After clipping no
cross-tile pair may overlap by more than :data:`SEAM_OVERLAP_TOLERANCE_UM2`.
Overlaps of two cells of one tile are ProSeg's own and are not touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from merxen.segmentation.atera_proseg import AteraProsegError

SEAM_OVERLAP_TOLERANCE_UM2: Final = 0.01
"""Largest overlap (um^2) two cells of different tiles may keep after clipping."""

SEAM_CLIP_RULE: Final = "core_then_plan_order"
"""The clip rule's name in the stitch report (the rule of the transcripts)."""

_LARGE_OVERLAP_UM2: Final = 1.0


@dataclass(frozen=True)
class OverlapStats:
    """Overlaps between cells of different tiles.

    Attributes:
        n_pairs: Cross-tile pairs whose overlap has positive area.
        n_pairs_over_1um2: Those above 1 um^2.
        n_pairs_over_tolerance: Those above :data:`SEAM_OVERLAP_TOLERANCE_UM2`.
        max_area_um2: Largest overlap area.
        total_area_um2: Summed overlap area.
    """

    n_pairs: int
    n_pairs_over_1um2: int
    n_pairs_over_tolerance: int
    max_area_um2: float
    total_area_um2: float

    def as_dict(self) -> dict[str, Any]:
        """Return the stats as a JSON-ready dict."""
        return {
            "n_pairs": self.n_pairs,
            "n_pairs_over_1um2": self.n_pairs_over_1um2,
            "n_pairs_over_tolerance": self.n_pairs_over_tolerance,
            "max_area_um2": self.max_area_um2,
            "total_area_um2": self.total_area_um2,
        }


@dataclass
class SeamClip:
    """Result of :func:`clip_seam_overlaps`.

    Attributes:
        geometries: The geometries, clipped where a cell lost area (``None``
            for an emptied cell); the others are the input objects.
        emptied: Positions of the cells left with no area.
        clipped: One record per clipped (or emptied) cell: ``position``,
            ``area_before_um2``, ``area_after_um2``, ``clipped_area_um2`` and
            ``against`` (positions of the cells it ceded area to).
        before: Cross-tile overlaps of the input.
        after: Cross-tile overlaps of the result.
    """

    geometries: list[Any]
    emptied: list[int]
    clipped: list[dict[str, Any]] = field(default_factory=list)
    before: OverlapStats | None = None
    after: OverlapStats | None = None


def _valid(geometry: Any) -> Any:
    import shapely

    return geometry if geometry.is_valid else shapely.make_valid(geometry)


def _usable(geometries: list[Any]) -> np.ndarray:
    return np.asarray(
        [g is not None and not g.is_empty for g in geometries], dtype=bool
    )


def cross_tile_overlaps(
    geometries: list[Any], groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every pair of cells of different groups whose overlap has area.

    Args:
        geometries: Shapely geometries in microns (``None`` or empty skipped).
        groups: Group (tile) of each geometry.

    Returns:
        ``(left, right, area)``: positions with ``left < right`` and the
        overlap area in um^2, for pairs of positive area only.
    """
    import shapely

    group_of = np.asarray(groups, dtype=np.int64)
    usable = _usable(geometries)
    array = np.empty(len(geometries), dtype=object)
    array[:] = [g if ok else None for g, ok in zip(geometries, usable, strict=True)]
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64))
    if not usable.any():
        return empty
    tree = shapely.STRtree(array)
    left, right = tree.query(array, predicate="intersects")
    keep = (left < right) & (group_of[left] != group_of[right])
    left, right = left[keep], right[keep]
    if not left.size:
        return empty
    involved = np.unique(np.concatenate([left, right]))
    fixed = np.empty(len(geometries), dtype=object)
    fixed[involved] = [_valid(array[i]) for i in involved]
    area = np.asarray(
        shapely.area(shapely.intersection(fixed[left], fixed[right])),
        dtype=np.float64,
    )
    positive = area > 0
    return left[positive], right[positive], area[positive]


def _stats(area: np.ndarray, tolerance: float) -> OverlapStats:
    return OverlapStats(
        n_pairs=int(area.size),
        n_pairs_over_1um2=int((area > _LARGE_OVERLAP_UM2).sum()),
        n_pairs_over_tolerance=int((area > tolerance).sum()),
        max_area_um2=float(area.max()) if area.size else 0.0,
        total_area_um2=float(area.sum()),
    )


def _as_multipolygon(geometry: Any) -> Any:
    """Return the polygonal parts of positive area as a MultiPolygon (or None)."""
    from shapely.geometry import MultiPolygon, Polygon

    parts: list[Any] = []
    stack = [geometry]
    while stack:
        item = stack.pop()
        if isinstance(item, Polygon):
            if not item.is_empty and item.area > 0:
                parts.append(item)
        elif hasattr(item, "geoms"):
            stack.extend(item.geoms)
    if not parts:
        return None
    parts.sort(key=lambda part: part.area, reverse=True)
    return MultiPolygon(parts)


def core_box(grid: tuple[int, int], tile_um: float) -> Any:
    """Return the core ``[c T, (c + 1) T) x [r T, (r + 1) T)`` of grid ``(c, r)``."""
    from shapely.geometry import box

    column, row = grid
    size = float(tile_um)
    return box(column * size, row * size, (column + 1) * size, (row + 1) * size)


def clip_seam_overlaps(
    geometries: list[Any],
    groups: np.ndarray,
    cores: list[Any],
    *,
    tolerance_um2: float = SEAM_OVERLAP_TOLERANCE_UM2,
) -> SeamClip:
    """Clip overlapping cells of different tiles by the core rule.

    For a pair of overlapping cells ``a`` (the tile earlier in plan order) and
    ``b``, ``b`` wins the overlap inside its own core and ``a`` wins the rest
    (inside ``a``'s core, or in neither core, where the earlier tile wins), as
    for a transcript two tiles claim. Each cell loses the union of what it
    cedes to every neighbour, always measured against the neighbours' input
    polygons, so every pair ends disjoint whatever the order.

    Args:
        geometries: Shapely polygons or multipolygons in microns.
        groups: Plan position of each geometry's tile (lower is earlier).
        cores: Core polygon of each group, indexed by group.
        tolerance_um2: Largest cross-tile overlap allowed after clipping.

    Returns:
        The clipped geometries and the clip report.

    Raises:
        AteraProsegError: If a cross-tile overlap above the tolerance remains.
    """
    import shapely

    group_of = np.asarray(groups, dtype=np.int64)
    left, right, area = cross_tile_overlaps(geometries, group_of)
    result = SeamClip(geometries=list(geometries), emptied=[])
    result.before = _stats(area, tolerance_um2)
    losses: dict[int, list[Any]] = {}
    against: dict[int, set[int]] = {}
    valid: dict[int, Any] = {}

    def valid_of(position: int) -> Any:
        if position not in valid:
            valid[position] = _valid(geometries[position])
        return valid[position]

    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        early, late = (i, j) if group_of[i] < group_of[j] else (j, i)
        late_core = cores[int(group_of[late])]
        # ``early`` cedes the overlap inside the late tile's core, ``late``
        # cedes the rest of it.
        early_loss = shapely.intersection(valid_of(late), late_core)
        late_loss = shapely.difference(valid_of(early), late_core)
        for cell, loss, other in ((early, early_loss, late), (late, late_loss, early)):
            if shapely.area(shapely.intersection(valid_of(cell), loss)) > 0:
                losses.setdefault(cell, []).append(loss)
                against.setdefault(cell, set()).add(other)
    for cell in sorted(losses):
        before = valid_of(cell)
        clipped = _as_multipolygon(
            shapely.difference(before, shapely.union_all(losses[cell]))
        )
        area_before = float(shapely.area(before))
        area_after = 0.0 if clipped is None else float(clipped.area)
        result.geometries[cell] = clipped
        if clipped is None:
            result.emptied.append(cell)
        result.clipped.append(
            {
                "position": cell,
                "area_before_um2": area_before,
                "area_after_um2": area_after,
                "clipped_area_um2": area_before - area_after,
                "against": sorted(against[cell]),
            }
        )
    _, _, after = cross_tile_overlaps(result.geometries, group_of)
    result.after = _stats(after, tolerance_um2)
    if result.after.n_pairs_over_tolerance:
        raise AteraProsegError(
            f"{result.after.n_pairs_over_tolerance} cross-tile cell pairs still "
            f"overlap by more than {tolerance_um2} um^2 after the seam clip "
            f"(largest {result.after.max_area_um2:.4f} um^2)"
        )
    return result
