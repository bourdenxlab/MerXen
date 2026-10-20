"""Full-tissue polygon construction from cortical boundary annotations."""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from merxen.cortical_depth.boundaries import (
    BoundaryAnnotations,
    BoundaryAnnotationSet,
)
from merxen.cortical_depth.ribbon import build_cortical_ribbon_polygon


def build_full_tissue_polygon(
    annotations: BoundaryAnnotationSet,
    *,
    require_tissue_edge: bool = True,
    allow_ribbon_fallback: bool = False,
) -> Polygon | MultiPolygon:
    """Build full tissue support from every pia and the shared tissue edge.

    Gray/white boundaries are internal cortical landmarks and are deliberately
    ignored. Piece polygons are unioned before every exclusion polygon is
    subtracted, so exclusions apply consistently across the full tissue.
    """
    edge = annotations.edge
    if edge is None and require_tissue_edge:
        raise ValueError(
            "Expected exactly one global tissue-edge annotation; "
            f"found {len(annotations.side_boundaries)}."
        )

    piece_polygons: list[Polygon | MultiPolygon] = []
    for piece in annotations.pieces:
        tissue_piece = BoundaryAnnotations(
            pial=piece.pial,
            wm=None,
            side_boundaries=(),
            exclusions=(),
            ribbon=(piece.ribbon if edge is None and allow_ribbon_fallback else None),
        )
        try:
            polygon, _ = build_cortical_ribbon_polygon(
                tissue_piece,
                edge_line=edge,
            )
        except ValueError as exc:
            raise ValueError(
                f"Tissue piece {piece.tissue_piece_id!r} could not be closed "
                f"against the tissue edge: {exc}"
            ) from exc
        piece_polygons.append(polygon)

    tissue_polygon = unary_union(piece_polygons)
    if annotations.exclusions:
        tissue_polygon = tissue_polygon.difference(
            unary_union(list(annotations.exclusions))
        )
    if (
        tissue_polygon.is_empty
        or not tissue_polygon.is_valid
        or not isinstance(tissue_polygon, Polygon | MultiPolygon)
    ):
        raise ValueError(
            "Pial/tissue-edge annotations did not produce a valid non-empty "
            "Polygon or MultiPolygon after exclusions."
        )
    return tissue_polygon
