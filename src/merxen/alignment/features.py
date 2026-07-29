"""Compatibility exports for the explicitly legacy Spateo feature module."""

from merxen.alignment.legacy_features import (
    _robust_centroid_xy,
    build_alignment_adata,
    prepare_spateo_features,
    shared_gene_subset,
)

__all__ = [
    "_robust_centroid_xy",
    "build_alignment_adata",
    "prepare_spateo_features",
    "shared_gene_subset",
]
