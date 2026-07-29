"""Narrow compatibility helpers for VALIS 1.2 in MerXen's NumPy 2 runtime."""

from __future__ import annotations

import numpy as np


def apply_valis_numpy_compatibility() -> None:
    """Restore the two removed scalar aliases still referenced by VALIS 1.2."""
    aliases = {"float": float, "int": int}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
