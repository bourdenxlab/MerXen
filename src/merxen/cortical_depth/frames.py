"""Put cell coordinates in the coordinate frame of the cortical-depth boundaries.

The pia / white-matter / tissue-edge GeoJSONs of a platform are drawn on that
platform's own section, in its dataset microns (the ``native`` frame). The
VALIS alignment reads the same combined annotation files as native-frame
tissue masks, before any aligned element exists, and the spatial-gene stage
pairs them with native transcripts. After ``MATERIALIZE_ALIGNMENT`` the moving
section's store (MERSCOPE by default) also holds ``*_aligned_nonrigid``
elements in the fixed section's frame (the ``aligned`` frame). Depth is only
meaningful when the cells and the boundaries share a frame, so the frame of the
boundaries is an explicit setting and the cells are read from the element in
that frame, or the combination is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import anndata as ad

# Kept in sync with merxen.alignment.manifest (NONRIGID_ELEMENT_SUFFIX,
# ALIGNMENT_MANIFEST_ATTR, ALIGNMENT_PAIR_REFERENCE_ATTR). Not imported from
# there: the merxen.alignment package __init__ loads the registration stack.
ALIGNED_ELEMENT_SUFFIX: Final = "_aligned_nonrigid"
ALIGNMENT_MANIFEST_ATTR: Final = "merxen_alignment"
ALIGNMENT_PAIR_REFERENCE_ATTR: Final = "merxen_alignment_pair_reference"

BoundaryFrame = Literal["native", "aligned"]
BOUNDARY_FRAMES: Final[tuple[str, ...]] = ("native", "aligned")

# How the cell element was chosen (recorded in the QC summary).
NATIVE_ELEMENT: Final = "native_element"
ALIGNED_ELEMENT: Final = "aligned_element"
FIXED_REFERENCE_NATIVE_ELEMENT: Final = "fixed_reference_native_element"
TABLE_SPATIAL_ONLY: Final = "table_spatial_only"


class BoundaryFrameMismatchError(ValueError):
    """The cell coordinates cannot be put in the frame of the boundaries."""


@dataclass(frozen=True)
class CellCoordinateFrame:
    """The cell element and coordinates that match the boundary frame.

    Attributes:
        boundary_frame: Frame the boundary GeoJSONs are drawn in.
        shape_key: Shape element whose centroids (or frame) the cells take, or
            ``None`` when the store has no shapes.
        region_key: Native shape element the table annotates; used as the table
            region when depth columns are written back, so a native table never
            gets retargeted to an aligned element.
        use_table_spatial: Whether the table's ``obsm['spatial']`` is in the
            boundary frame and may be used instead of shape centroids.
        resolution: How the element was chosen (``native_element``,
            ``aligned_element``, ``fixed_reference_native_element`` or
            ``table_spatial_only``).
    """

    boundary_frame: str
    shape_key: str | None
    region_key: str | None
    use_table_spatial: bool
    resolution: str

    def provenance(self: CellCoordinateFrame) -> dict[str, Any]:
        """Return the frame fields recorded in the cortical-depth QC summary."""
        return {
            "boundary_frame": self.boundary_frame,
            "cell_coordinate_frame": self.boundary_frame,
            "frame_resolution": self.resolution,
        }


def is_aligned_element_key(key: str | None) -> bool:
    """Return whether a SpatialData element key names an aligned derivative."""
    return key is not None and str(key).endswith(ALIGNED_ELEMENT_SUFFIX)


def is_alignment_fixed_reference(sdata_obj: Any, platform: str) -> bool:
    """Return whether this store is the fixed section of a materialized pair.

    The fixed section's native frame is the pair's aligned frame. The store
    says so itself: ``MATERIALIZE_ALIGNMENT`` writes a pair reference into the
    fixed store and a manifest naming the fixed platform into the moving one.

    Args:
        sdata_obj: SpatialData object (anything with an ``attrs`` mapping).
        platform: Platform of this store (``MERSCOPE`` or ``XENIUM``).

    Returns:
        True when the store carries the pair reference, or a manifest whose
        fixed role is ``platform``.
    """
    attrs = getattr(sdata_obj, "attrs", None) or {}
    if ALIGNMENT_PAIR_REFERENCE_ATTR in attrs:
        return True
    manifest = attrs.get(ALIGNMENT_MANIFEST_ATTR)
    if not isinstance(manifest, dict):
        return False
    roles = manifest.get("roles")
    return isinstance(roles, dict) and (
        str(roles.get("fixed", "")).upper() == str(platform).upper()
    )


def resolve_cell_coordinate_frame(
    sdata_obj: Any,
    *,
    table: ad.AnnData,
    table_key: str,
    requested_shape_key: str | None,
    platform: str,
    boundary_frame: str,
    dataset_name: str = "",
) -> CellCoordinateFrame:
    """Choose the cell element whose coordinates are in the boundary frame.

    Truth table (``requested`` is the configured native shape key,
    ``aligned`` is ``requested + "_aligned_nonrigid"``):

    - ``native``: always ``requested``, on every platform, whether or not an
      aligned variant exists; the table's ``obsm['spatial']`` (native tables
      are never modified by alignment) may be used. Refused when only the
      aligned variant exists.
    - ``aligned``: ``aligned`` when present (the moving section; its native
      ``obsm['spatial']`` is then not used). Without it, ``requested`` only on
      the pair's fixed section, whose native frame is the aligned frame.
      Refused otherwise (no materialized alignment for this store).
    - Configured keys that already name an aligned element (table or shape)
      are refused in both frames: the stage reads and writes native tables
      and picks the aligned variant itself.

    Args:
        sdata_obj: SpatialData object holding the table and its shapes.
        table: The cell table depth is assigned to.
        table_key: Key of ``table`` in ``sdata_obj.tables``.
        requested_shape_key: Configured native shape key, or ``None`` to use
            the table's region.
        platform: Platform of this store (``MERSCOPE`` or ``XENIUM``).
        boundary_frame: Frame the boundary GeoJSONs are drawn in.
        dataset_name: Dataset label for error messages.

    Returns:
        The chosen element, table region and provenance.

    Raises:
        ValueError: If ``boundary_frame`` is not a known frame.
        BoundaryFrameMismatchError: If no element of the store is in
            ``boundary_frame``.
        KeyError: If the requested shape element does not exist in any frame.
    """
    frame = str(boundary_frame).strip().lower()
    if frame not in BOUNDARY_FRAMES:
        raise ValueError(
            f"Unknown cortical-depth boundary_frame {boundary_frame!r}; "
            f"expected one of {list(BOUNDARY_FRAMES)}."
        )
    label = f"[{dataset_name}] " if dataset_name else ""
    shapes = [str(key) for key in sdata_obj.shapes]
    if is_aligned_element_key(table_key):
        raise BoundaryFrameMismatchError(
            f"{label}table_key={table_key!r} is an aligned table. Cortical depth "
            "reads and writes the native table and selects the aligned element "
            "itself; configure the native table key and "
            f"boundary_frame={frame!r}."
        )
    if is_aligned_element_key(requested_shape_key):
        raise BoundaryFrameMismatchError(
            f"{label}shape_key={requested_shape_key!r} names an aligned element. "
            "Configure the native shape key; the boundary frame "
            f"(boundary_frame={frame!r}) selects the element."
        )
    is_fixed_reference = is_alignment_fixed_reference(sdata_obj, platform)
    if not shapes:
        if frame == "native" or is_fixed_reference:
            return CellCoordinateFrame(
                boundary_frame=frame,
                shape_key=None,
                region_key=None,
                use_table_spatial=True,
                resolution=TABLE_SPATIAL_ONLY,
            )
        raise BoundaryFrameMismatchError(
            f"{label}boundary_frame='aligned' but the store has no shapes and is "
            "not the fixed section of a materialized alignment, so its table "
            "coordinates are native."
        )

    native_key = (
        str(requested_shape_key)
        if requested_shape_key is not None
        else _native_region_key(table, shapes, label=label)
    )
    aligned_key = f"{native_key}{ALIGNED_ELEMENT_SUFFIX}"
    if frame == "native":
        if native_key in shapes:
            return CellCoordinateFrame(
                boundary_frame=frame,
                shape_key=native_key,
                region_key=native_key,
                use_table_spatial=True,
                resolution=NATIVE_ELEMENT,
            )
        if aligned_key in shapes:
            raise BoundaryFrameMismatchError(
                f"{label}boundary_frame='native' but only the aligned element "
                f"{aligned_key!r} of {native_key!r} exists. Native boundaries "
                "cannot be matched to aligned cells; restore the native shapes, "
                "or set boundary_frame='aligned' if the boundaries were drawn in "
                "the aligned frame."
            )
        raise KeyError(
            f"{label}Requested shape_key={native_key!r} not found. "
            f"Available shapes: {shapes}"
        )

    if aligned_key in shapes:
        return CellCoordinateFrame(
            boundary_frame=frame,
            shape_key=aligned_key,
            region_key=native_key if native_key in shapes else aligned_key,
            use_table_spatial=False,
            resolution=ALIGNED_ELEMENT,
        )
    if is_fixed_reference:
        if native_key not in shapes:
            raise KeyError(
                f"{label}Requested shape_key={native_key!r} not found. "
                f"Available shapes: {shapes}"
            )
        return CellCoordinateFrame(
            boundary_frame=frame,
            shape_key=native_key,
            region_key=native_key,
            use_table_spatial=True,
            resolution=FIXED_REFERENCE_NATIVE_ELEMENT,
        )
    raise BoundaryFrameMismatchError(
        f"{label}boundary_frame='aligned' but {platform} has no aligned element "
        f"{aligned_key!r} and is not the fixed section of a materialized "
        "alignment. Run MATERIALIZE_ALIGNMENT, or set boundary_frame='native' "
        "if the boundaries were drawn on this section."
    )


def _native_region_key(table: ad.AnnData, shapes: list[str], *, label: str) -> str:
    """Return the native shape key a table annotates when none is configured."""
    region = _table_region(table)
    if region is not None:
        if is_aligned_element_key(region):
            native = region[: -len(ALIGNED_ELEMENT_SUFFIX)]
            if native in shapes:
                return native
            raise BoundaryFrameMismatchError(
                f"{label}The table annotates the aligned element {region!r} and "
                f"its native element {native!r} is missing; configure shape_key."
            )
        if region in shapes or f"{region}{ALIGNED_ELEMENT_SUFFIX}" in shapes:
            return region
    native_shapes = [key for key in shapes if not is_aligned_element_key(key)]
    if not native_shapes:
        raise BoundaryFrameMismatchError(
            f"{label}The store holds only aligned shape elements ({shapes}); "
            "configure shape_key."
        )
    return native_shapes[0]


def _table_region(table: ad.AnnData) -> str | None:
    region = dict(table.uns.get("spatialdata_attrs", {})).get("region")
    if isinstance(region, str):
        return region
    if isinstance(region, list | tuple) and region:
        return str(region[0])
    return None
