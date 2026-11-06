"""Tiled ProSeg for ATERA rows (``segmentation_source=native_proseg``, plan AT-9).

The vendor cell masks are ProSeg's prior; no Cellpose runs. A section is cut
into square tiles (:mod:`.plan`); each tile is one ProSeg run on the
transcripts of its window (:mod:`.export`, :mod:`.runner`) reduced to the cells
it owns and their transcript claims (:mod:`.tile`). The stitch
(:mod:`.stitch`) writes the stitched cells into the latest store as the
analysis layer ``reseg``: registry branch ``proseg`` with the shapes
``MOSAIK_proseg``, the table ``table_MOSAIK_proseg``, the label element
``proseg_cell_labels`` and the transcript columns ``assignment`` and
``background``, as on XENIUM and MERSCOPE.

Ownership: a vendor label belongs to the tile whose half-open core holds its
vendor centroid; every label has exactly one owner. A tile's window is its core
plus the halo plus the bounding box of every cell it owns (with a margin), so
an owned cell is never cut by the window. A ProSeg cell keeps the vendor label
as its ``instance_id`` (the original id namespace); a transcript assigned by
two tiles to different cells goes to the tile whose core holds it.
"""

from __future__ import annotations

from typing import Final

PROSEG_BRANCH: Final = "proseg"
"""Registry branch of the ProSeg layer (``analysis_segmentation=reseg``)."""

PROSEG_SHAPES_KEY: Final = "MOSAIK_proseg"
PROSEG_TABLE_KEY: Final = "table_MOSAIK_proseg"
PROSEG_LABEL_KEY: Final = "proseg_cell_labels"
ASSIGNMENT_COLUMN: Final = "assignment"
BACKGROUND_COLUMN: Final = "background"
PRIOR_NAME: Final = "vendor_cell_labels"
"""Registry ``prior`` field: what the ProSeg cells were refined from."""

PROVENANCE_ATTR: Final = "merxen_atera_proseg"
"""Root attribute holding the provenance of the stitched ProSeg layer."""

PLAN_FILE_NAME: Final = "tile_plan.json"
TILE_MANIFEST_NAME: Final = "tile_manifest.json"
TILE_CELLS_NAME: Final = "cells.parquet"
TILE_CLAIMS_DIR: Final = "claims"
REQUIRED_PROSEG_VERSION: Final = "3.2.0"
"""The ProSeg release whose Xenium-layout reader and flags were verified."""

PLAN_FORMAT_VERSION: Final = 1
TILE_FORMAT_VERSION: Final = 1
STITCH_FORMAT_VERSION: Final = 1


def tile_id(column: int, row: int) -> str:
    """Return the id of the tile at grid column ``column`` and row ``row``."""
    return f"c{int(column)}_r{int(row)}"


class AteraProsegError(RuntimeError):
    """Raised when a plan, a tile or the stitch cannot proceed safely."""


class ProsegRunError(AteraProsegError):
    """Raised when the ProSeg process itself exits non-zero.

    Unlike the other errors of a tile (inputs, version, self-checks), this
    can be transient, so the tile CLI exits with the plain failure status that
    the workflow retries once.
    """


class PriorWeightMismatchError(AteraProsegError):
    """Raised when ProSeg reports another effective prior weight than requested.

    Attributes:
        reported: The weight ProSeg printed (``None`` when it printed none).
        requested: The requested effective weight.
    """

    def __init__(
        self, message: str, *, reported: float | None, requested: float
    ) -> None:
        super().__init__(message)
        self.reported = reported
        self.requested = requested
