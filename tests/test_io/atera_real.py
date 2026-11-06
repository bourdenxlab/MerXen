"""Locate the public 10x Atera dataset for the slow real-data tests.

``$MERXEN_ATERA_RAW_ROOT`` (``.env``) holds one folder per downloaded dataset,
each with an extracted ``outs/`` bundle. The tests take the single extracted
bundle there, so no dataset or sample name is written into the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: Written by the extraction once ``outs/`` is complete.
EXTRACT_MARKER = ".extract_complete"


def extracted_outs(raw_root: Path) -> Path:
    """Return the one extracted ``outs/`` folder under the Atera data root.

    Args:
        raw_root: The ``$MERXEN_ATERA_RAW_ROOT`` folder.

    Returns:
        The ``outs/`` folder.

    Raises:
        pytest.skip.Exception: If no bundle, or more than one, is extracted.
    """
    found = sorted(path.parent for path in raw_root.glob(f"*/outs/{EXTRACT_MARKER}"))
    if len(found) != 1:
        pytest.skip(
            f"{len(found)} extracted Atera bundles under MERXEN_ATERA_RAW_ROOT; "
            "these tests need exactly one"
        )
    return found[0]
