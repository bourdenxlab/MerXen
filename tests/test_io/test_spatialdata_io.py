"""Tests for SpatialData write helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import spatialdata as sd
from spatialdata import datasets

from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
    write_spatialdata_zarr,
)


def test_write_spatialdata_zarr_writes_blobs_dataset(tmp_path: Path) -> None:
    """Writing a multiscale SpatialData object should succeed without local shims."""
    out = tmp_path / "blobs.zarr"

    write_spatialdata_zarr(datasets.blobs(), out)

    assert out.exists()
    reloaded = sd.read_zarr(out)
    assert "blobs_image" in reloaded.images
    assert "blobs_labels" in reloaded.labels


def test_write_spatialdata_zarr_supports_overwrite(tmp_path: Path) -> None:
    """The helper should pass through SpatialData's overwrite flag."""
    out = tmp_path / "blobs.zarr"

    write_spatialdata_zarr(datasets.blobs(), out)
    write_spatialdata_zarr(datasets.blobs(), out, overwrite=True)

    assert out.exists()


def test_write_spatialdata_zarr_passes_overwrite_flag(tmp_path: Path) -> None:
    """write_spatialdata_zarr should forward the overwrite kwarg when supplied."""
    sdata = MagicMock()
    out = tmp_path / "out.zarr"

    write_spatialdata_zarr(sdata, out, overwrite=True)

    sdata.write.assert_called_once_with(out, overwrite=True)


def test_write_spatialdata_zarr_omits_overwrite_when_none(tmp_path: Path) -> None:
    """write_spatialdata_zarr should not pass overwrite when it is None."""
    sdata = MagicMock()
    out = tmp_path / "out.zarr"

    write_spatialdata_zarr(sdata, out, overwrite=None)

    sdata.write.assert_called_once_with(out)


def test_write_or_replace_element_writes_new_element() -> None:
    """New elements should be assigned in memory and persisted without overwrite."""
    value = object()
    sdata = SimpleNamespace(shapes={}, write_element=MagicMock())

    wrote = write_or_replace_element(sdata, "cells", "shapes", value)

    assert wrote
    assert sdata.shapes["cells"] is value
    sdata.write_element.assert_called_once_with("cells", overwrite=False)


def test_write_or_replace_element_skips_existing_without_overwrite() -> None:
    """Existing elements should be left untouched when overwrite is disabled."""
    old_value = object()
    sdata = SimpleNamespace(shapes={"cells": old_value}, write_element=MagicMock())

    wrote = write_or_replace_element(
        sdata,
        "cells",
        "shapes",
        object(),
        overwrite=False,
    )

    assert not wrote
    assert sdata.shapes["cells"] is old_value
    sdata.write_element.assert_not_called()


def test_write_or_replace_element_overwrites_without_disk_delete() -> None:
    """Replacement should rely on write_element(overwrite=True), not disk deletion."""
    new_value = object()
    sdata = SimpleNamespace(
        shapes={"cells": object()},
        write_element=MagicMock(),
        delete_element_from_disk=MagicMock(),
    )

    wrote = write_or_replace_element(
        sdata,
        "cells",
        "shapes",
        new_value,
        overwrite=True,
    )

    assert wrote
    assert sdata.shapes["cells"] is new_value
    sdata.write_element.assert_called_once_with("cells", overwrite=True)
    sdata.delete_element_from_disk.assert_not_called()


def test_write_or_replace_element_deletes_only_after_overwrite_refusal() -> None:
    """Fallback deletion should happen only after SpatialData rejects overwrite."""
    calls: list[tuple[str, object]] = []

    def _write_element(key: str, *, overwrite: bool) -> None:
        calls.append(("write", overwrite))
        if len(calls) == 1:
            raise ValueError("Cannot overwrite. The target path is in use.")

    def _delete_element(key: str) -> None:
        calls.append(("delete", key))

    sdata = SimpleNamespace(
        shapes={"cells": object()},
        write_element=_write_element,
        delete_element_from_disk=_delete_element,
    )

    wrote = write_or_replace_element(
        sdata,
        "cells",
        "shapes",
        object(),
        overwrite=True,
    )

    assert wrote
    assert calls == [("write", True), ("delete", "cells"), ("write", False)]


def test_write_spatialdata_metadata_persists_metadata_and_transforms() -> None:
    """Metadata helper should delegate to SpatialData's narrow write APIs."""
    sdata = SimpleNamespace(
        write_metadata=MagicMock(),
        write_transformations=MagicMock(),
    )

    write_spatialdata_metadata(
        sdata,
        write_attrs=True,
        write_transformations=True,
    )

    sdata.write_transformations.assert_called_once_with()
    sdata.write_metadata.assert_called_once_with(write_attrs=True)
