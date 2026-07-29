"""Shared alignment result models used by VALIS and the legacy backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from merxen.alignment.bundle import ValisTransformBundle
from merxen.alignment.transforms import NonRigidTransform


@dataclass(frozen=True)
class TransformResult:
    """Container for a selected moving-to-fixed registration."""

    merscope_to_common: dict[str, Any]
    xenium_to_common: dict[str, Any]
    metadata: dict[str, Any]
    coordinate_tables: dict[str, pd.DataFrame] | None = None
    nonrigid_transform: NonRigidTransform | None = None
    valis_transform: ValisTransformBundle | None = None
    valid_domain_mask: Any | None = None
