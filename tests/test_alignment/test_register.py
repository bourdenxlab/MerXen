"""Tests for alignment registration placeholder API."""

from __future__ import annotations

import pytest

from merxen.alignment.register import register_pair


def test_register_pair_raises_not_implemented() -> None:
    """Registration placeholder should raise until implemented."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        register_pair(None, None, None)
