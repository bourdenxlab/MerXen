"""Shared test fixtures for MerXen."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_mask() -> np.ndarray:
    """A 32x32 labeled mask with 3 cell regions."""
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[2:8, 2:8] = 1
    mask[12:20, 12:20] = 2
    mask[24:30, 5:15] = 3
    return mask


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path / "output"
