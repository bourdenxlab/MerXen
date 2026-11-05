"""Tests of the original-shape choice in ``frames._choose_dataset_element``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from merxen.alignment import frames
from merxen.platforms import UnknownPlatformError

_BOTH_ORIGINALS = ("merscope_cell_boundaries", "xenium_cell_boundaries")


def _fake_sdata(shape_keys: tuple[str, ...], schema_shape: str | None = None) -> Any:
    """Return an object with the attributes the chooser reads."""
    attrs: dict[str, Any] = {}
    if schema_shape is not None:
        attrs["merxen_schema"] = {
            "segmentations": {"original": {"shape": schema_shape}}
        }
    return SimpleNamespace(
        shapes={key: object() for key in shape_keys},
        points={},
        labels={},
        attrs=attrs,
    )


@pytest.fixture(autouse=True)
def _identity_affine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real transform read; the tests only check which key is chosen."""
    monkeypatch.setattr(frames, "_element_affine_to_global", lambda _: np.eye(3))


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("MERSCOPE", "merscope_cell_boundaries"),
        ("XENIUM", "xenium_cell_boundaries"),
    ],
)
def test_pre_schema_store_uses_the_platform_original_shape(
    platform: str, expected: str
) -> None:
    """Without a schema entry, the platform's own original layer is chosen."""
    key, element, trusted = frames._choose_dataset_element(
        _fake_sdata(_BOTH_ORIGINALS), platform=platform
    )
    assert key == expected
    assert element is not None
    assert trusted is True


def test_schema_registered_shape_wins_over_the_registry() -> None:
    """The schema's original shape is used even if another platform's is present."""
    key, _, trusted = frames._choose_dataset_element(
        _fake_sdata(_BOTH_ORIGINALS, schema_shape="xenium_cell_boundaries"),
        platform="MERSCOPE",
    )
    assert key == "xenium_cell_boundaries"
    assert trusted is True


def test_schema_registered_shape_needs_no_registry_lookup() -> None:
    """A store that names its original shape is read without consulting the registry."""
    key, _, _ = frames._choose_dataset_element(
        _fake_sdata(("custom_original",), schema_shape="custom_original"),
        platform="COSMX",
    )
    assert key == "custom_original"


@pytest.mark.parametrize("schema_shape", [None, "missing_original"])
def test_unknown_platform_raises_when_the_registry_is_needed(
    schema_shape: str | None,
) -> None:
    """An unregistered platform raises instead of using another platform's layer."""
    with pytest.raises(UnknownPlatformError, match="Unknown platform"):
        frames._choose_dataset_element(
            _fake_sdata(_BOTH_ORIGINALS, schema_shape=schema_shape),
            platform="COSMX",
        )
