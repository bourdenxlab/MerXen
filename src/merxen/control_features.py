"""Registry of platform control features (negative controls, blanks, codewords).

Xenium and MERSCOPE report control features alongside real genes. This module
is the single place that recognises them, so segmentation inputs, clustering
and transcript analyses drop the same features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Case-insensitive substrings used by clustering and transcript analyses to
# recognise control-like features when no platform-specific rule applies.
CONTROL_TOKENS: tuple[str, ...] = (
    "blank",
    "control",
    "negative",
    "negcontrol",
    "unassigned",
    "deprecated",
)


def has_control_token(name: str) -> bool:
    """Return whether a name contains one of :data:`CONTROL_TOKENS`.

    Args:
        name: Feature, column or key name.

    Returns:
        ``True`` when a control token occurs anywhere in the lower-cased name.
    """
    lower = str(name).lower()
    return any(token in lower for token in CONTROL_TOKENS)


def control_token_mask(values: Any) -> np.ndarray:
    """Vectorised :func:`has_control_token` over feature names.

    Args:
        values: One-dimensional array-like of names. Missing values are
            converted with ``str`` and so never match.

    Returns:
        Boolean array, ``True`` where a name contains a control token.
    """
    names = pd.Series(values, dtype=object).astype(str)
    codes, uniques = pd.factorize(names)
    unique_mask = np.fromiter(
        (has_control_token(name) for name in uniques),
        dtype=bool,
        count=len(uniques),
    )
    return np.asarray(unique_mask[codes], dtype=bool)
