"""Tiled ProSeg for ATERA rows (plans AT-9 and AT-7).

ProSeg refines a prior segmentation of the morphology grid. Two priors exist,
one per segmentation source, and each writes its own analysis layer
(:data:`PROSEG_LAYERS`):

- ``native_proseg`` (AT-9): the vendor cell masks are the prior; the layer is
  ``proseg_vendor`` (shapes ``MOSAIK_proseg_vendor``, table
  ``table_MOSAIK_proseg_vendor``, labels ``proseg_vendor_cell_labels``) and
  keeps the vendor ids (namespace ``source_segmentation_v1``).
- ``cellpose_proseg`` (AT-7): MerXen's Cellpose masks (DAPI + 18S) are the
  prior; the layer is ``reseg`` with the XENIUM names (shapes
  ``MOSAIK_proseg``, table ``table_MOSAIK_proseg``, labels
  ``proseg_cell_labels``) and the Cellpose label namespace
  ``cellpose_label_v1``.

Both write the transcript columns ``assignment`` and ``background``; a row has
one source, so one store never holds both layers.

A section is cut into square tiles (:mod:`.plan`); each tile is one ProSeg run
on the transcripts of its window (:mod:`.export`, :mod:`.runner`) reduced to
the cells it owns and their transcript claims (:mod:`.tile`). The stitch
(:mod:`.stitch`) writes the stitched cells into the latest store.

Ownership: a prior label belongs to the tile whose half-open core holds its
centroid; every label has exactly one owner. A tile's window is its core plus
the halo plus the bounding box of every cell it owns (with a margin), so an
owned cell is never cut by the window. A ProSeg cell keeps the prior label as
its ``instance_id``; a transcript assigned by two tiles to different cells goes
to the tile whose core holds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from merxen.io.spatialdata_schema import ORIGINAL_ID_NAMESPACE, PROSEG_ID_NAMESPACE

ASSIGNMENT_COLUMN: Final = "assignment"
BACKGROUND_COLUMN: Final = "background"
CELLPOSE_ASSIGNMENT_COLUMN: Final = "cellpose_assignment"
STOCK_CELLPOSE_ASSIGNMENT_COLUMN: Final = "stock_cellpose_assignment"
"""Points column of a Cellpose-prior store: the Cellpose label at each row."""

PriorKind = Literal["vendor", "cellpose"]
"""What a tiled ProSeg run takes as its prior."""


@dataclass(frozen=True)
class ProsegLayer:
    """Element names and registry fields of one ProSeg layer.

    Attributes:
        prior_kind: The prior the layer is refined from.
        analysis_segmentation: Its ``analysis_segmentation`` value.
        branch: Registry branch name.
        shapes_key: Shapes element.
        table_key: Table element.
        label_key: Label element (cells rasterised on the morphology grid).
        id_namespace: Namespace of the cell ids (the prior's labels).
        prior_name: Registry ``prior`` field.
        legacy_aliases: Registry aliases.
    """

    prior_kind: PriorKind
    analysis_segmentation: str
    branch: str
    shapes_key: str
    table_key: str
    label_key: str
    id_namespace: str
    prior_name: str
    legacy_aliases: tuple[str, ...]


PROSEG_LAYERS: Final[Mapping[str, ProsegLayer]] = MappingProxyType(
    {
        "vendor": ProsegLayer(
            prior_kind="vendor",
            analysis_segmentation="proseg_vendor",
            branch="proseg_vendor",
            shapes_key="MOSAIK_proseg_vendor",
            table_key="table_MOSAIK_proseg_vendor",
            label_key="proseg_vendor_cell_labels",
            id_namespace=ORIGINAL_ID_NAMESPACE,
            prior_name="vendor_cell_labels",
            legacy_aliases=("vendor_proseg", "native_proseg"),
        ),
        "cellpose": ProsegLayer(
            prior_kind="cellpose",
            analysis_segmentation="reseg",
            branch="proseg",
            shapes_key="MOSAIK_proseg",
            table_key="table_MOSAIK_proseg",
            label_key="proseg_cell_labels",
            id_namespace=PROSEG_ID_NAMESPACE,
            prior_name="cellpose_cell_labels",
            legacy_aliases=("reseg",),
        ),
    }
)
"""The ProSeg layer of each prior kind (twin: ``AteraProsegLayers.groovy``)."""

SOURCE_PRIOR_KIND: Final[Mapping[str, PriorKind]] = MappingProxyType(
    {
        "native_proseg": "vendor",
        "cellpose_proseg": "cellpose",
        "cellpose_comparison": "cellpose",
    }
)
"""Prior kind of each segmentation source that runs the tiled ProSeg route."""


def proseg_layer(prior_kind: str) -> ProsegLayer:
    """Return the ProSeg layer of a prior kind.

    Args:
        prior_kind: ``"vendor"`` or ``"cellpose"``.

    Returns:
        The layer's names.

    Raises:
        AteraProsegError: For an unknown prior kind.
    """
    layer = PROSEG_LAYERS.get(str(prior_kind))
    if layer is None:
        raise AteraProsegError(
            f"Unknown ProSeg prior kind {prior_kind!r}; known: "
            f"{', '.join(PROSEG_LAYERS)}"
        )
    return layer


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
STITCH_FORMAT_VERSION: Final = 2
"""2 (AT-7): cross-tile polygon overlaps are clipped by the core rule."""


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
