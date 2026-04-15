"""Cross-section registration placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformResult:
    """Container for registration outputs.

    Attributes:
        merscope_to_common: Transform object mapping MERSCOPE coordinates to a
            shared reference space.
        xenium_to_common: Transform object mapping Xenium coordinates to the
            same shared reference space.
        metadata: Additional implementation-specific information.
    """

    merscope_to_common: Any
    xenium_to_common: Any
    metadata: dict[str, Any]


def register_pair(
    merscope_sdata: Any,
    xenium_sdata: Any,
    config: Any,
) -> TransformResult:
    """Register paired MERSCOPE and Xenium sections to a common coordinate system.

    Raises:
        NotImplementedError: Registration is intentionally deferred.
    """
    del merscope_sdata, xenium_sdata, config
    raise NotImplementedError("Cross-section registration not yet implemented")
