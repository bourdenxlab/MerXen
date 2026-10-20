"""Utilities for recognizing species-specific Ensembl gene identifiers."""

from __future__ import annotations

import re

ENSEMBL_GENE_ID_PATTERN = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$", re.IGNORECASE)


def is_ensembl_gene_id(value: object) -> bool:
    """Return whether a value is a human or non-human Ensembl gene ID.

    Examples include human ``ENSG...``, mouse ``ENSMUSG...``, and rat
    ``ENSRNOG...`` identifiers, with an optional version suffix.

    Args:
        value: Candidate identifier.

    Returns:
        ``True`` when the normalized value matches an Ensembl gene ID.
    """
    return ENSEMBL_GENE_ID_PATTERN.fullmatch(str(value).strip()) is not None
