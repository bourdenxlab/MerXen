"""Tests for SpatialData write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import zarr
from spatialdata import datasets
from spatialdata.models import Labels2DModel

from merxen.io.spatialdata_io import (
    convert_to_latest_zarr,
    deduplicate_ome_labels_metadata,
    normalize_points_for_latest_write,
    write_or_replace_element,
    write_spatialdata_metadata,
    write_spatialdata_zarr,
)


def test_convert_to_latest_retains_merscope_transcript_score_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MERSCOPE scores should survive ProSeg's canonical qv output name."""
    sdata_obj = SimpleNamespace(
        points={
            "transcripts": pd.DataFrame(
                {
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                    "gene": ["A", "B"],
                    "qv": [0.91, 0.82],
                }
            )
        }
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "merxen.io.spatialdata_io.sd.read_zarr",
        lambda path: sdata_obj,
    )

    def _capture_write(obj: object, path: Path, **kwargs: object) -> None:
        captured["obj"] = obj
        captured["path"] = path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "merxen.io.spatialdata_io.write_spatialdata_zarr",
        _capture_write,
    )

    output = convert_to_latest_zarr(
        tmp_path / "raw.zarr",
        tmp_path / "latest.zarr",
        quality_column_alias="transcript_score",
    )

    points = sdata_obj.points["transcripts"]
    assert output == tmp_path / "latest.zarr"
    assert points["transcript_score"].tolist() == [0.91, 0.82]
    assert points["qv"].tolist() == [0.91, 0.82]


def test_convert_to_latest_preserves_non_spatialdata_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ordinary conversion should retain vendor metadata outside element roots."""
    raw_path = tmp_path / "raw.zarr"
    raw_path.mkdir()
    transform_text = "1,0,0\n0,1,0\n"
    (raw_path / "micron_to_mosaic_pixel_transform.csv").write_text(transform_text)
    payload_dir = raw_path / "vendor_payload"
    payload_dir.mkdir()
    (payload_dir / "specs.json").write_text('{"pixel_size": 0.2125}\n')

    sdata_obj = SimpleNamespace(points={})
    monkeypatch.setattr(
        "merxen.io.spatialdata_io.sd.read_zarr",
        lambda _: sdata_obj,
    )

    def _write_zarr(_obj: object, path: Path, **_kwargs: object) -> None:
        Path(path).mkdir(parents=True)

    monkeypatch.setattr(
        "merxen.io.spatialdata_io.write_spatialdata_zarr",
        _write_zarr,
    )

    latest_path = tmp_path / "latest.zarr"
    output = convert_to_latest_zarr(raw_path, latest_path)

    assert output == latest_path
    assert (
        latest_path / "micron_to_mosaic_pixel_transform.csv"
    ).read_text() == transform_text
    assert (latest_path / "vendor_payload" / "specs.json").read_text() == (
        '{"pixel_size": 0.2125}\n'
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


def test_write_or_replace_element_retries_newer_zarr_store_refusal() -> None:
    """Newer SpatialData same-store messages should use the same fallback."""
    calls: list[tuple[str, object]] = []

    def _write_element(key: str, *, overwrite: bool) -> None:
        calls.append(("write", overwrite))
        if len(calls) == 1:
            raise ValueError(
                "The Zarr store already exists. Use `overwrite=True` to try "
                "overwriting the store. Please note that only Zarr stores not "
                "currently in use by the current SpatialData object can be "
                "overwritten."
            )

    def _delete_element(key: str) -> None:
        calls.append(("delete", key))

    sdata = SimpleNamespace(
        images={"MERSCOPE_z_projection": object()},
        write_element=_write_element,
        delete_element_from_disk=_delete_element,
    )

    wrote = write_or_replace_element(
        sdata,
        "MERSCOPE_z_projection",
        "images",
        object(),
        overwrite=True,
    )

    assert wrote
    assert calls == [
        ("write", True),
        ("delete", "MERSCOPE_z_projection"),
        ("write", False),
    ]


def test_write_or_replace_element_retries_orphaned_zarr_store_refusal() -> None:
    """Orphaned on-disk stores should be deleted even when metadata missed them."""
    calls: list[tuple[str, object]] = []

    def _write_element(key: str, *, overwrite: bool) -> None:
        calls.append(("write", overwrite))
        if len(calls) == 1:
            raise ValueError(
                "The Zarr store already exists. Use `overwrite=True` to try "
                "overwriting the store. Please note that only Zarr stores not "
                "currently in use by the current SpatialData object can be "
                "overwritten."
            )

    def _delete_element(key: str) -> None:
        calls.append(("delete", key))

    sdata = SimpleNamespace(
        images={},
        write_element=_write_element,
        delete_element_from_disk=_delete_element,
    )

    wrote = write_or_replace_element(
        sdata,
        "MERSCOPE_z_projection",
        "images",
        object(),
        overwrite=True,
    )

    assert wrote
    assert calls == [
        ("write", False),
        ("delete", "MERSCOPE_z_projection"),
        ("write", False),
    ]


def test_write_or_replace_element_removes_orphaned_store_by_path(
    tmp_path: Path,
) -> None:
    """Path cleanup should handle stores not deletable through SpatialData."""
    zarr_path = tmp_path / "latest.zarr"
    orphan = zarr_path / "images" / "MERSCOPE_z_projection"
    orphan.mkdir(parents=True)
    calls: list[tuple[str, object]] = []

    def _write_element(key: str, *, overwrite: bool) -> None:
        calls.append(("write", overwrite))
        if len(calls) == 1:
            raise ValueError(
                "The Zarr store already exists. Use `overwrite=True` to try "
                "overwriting the store. Please note that only Zarr stores not "
                "currently in use by the current SpatialData object can be "
                "overwritten."
            )

    def _delete_element(key: str) -> None:
        calls.append(("delete", key))
        raise KeyError(key)

    sdata = SimpleNamespace(
        images={},
        path=zarr_path,
        write_element=_write_element,
        delete_element_from_disk=_delete_element,
    )

    wrote = write_or_replace_element(
        sdata,
        "MERSCOPE_z_projection",
        "images",
        object(),
        overwrite=True,
    )

    assert wrote
    assert not orphan.exists()
    assert calls == [("write", False), ("write", False)]


def _consolidated_keys(zarr_path: Path) -> list[str]:
    root = json.loads((zarr_path / "zarr.json").read_text())
    return list(root["consolidated_metadata"]["metadata"])


def test_write_or_replace_element_backup_leaves_no_stale_consolidated_entries(
    tmp_path: Path,
) -> None:
    """Replacing a table read from the store must not keep its backup in zarr.json."""
    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    sdata = sd.read_zarr(zarr_path)
    table = sdata.tables["table"].copy()
    table.obs["replaced"] = 1

    wrote = write_or_replace_element(sdata, "table", "tables", table, overwrite=True)

    assert wrote
    assert sorted(p.name for p in (zarr_path / "tables").iterdir()) == [
        "table",
        "zarr.json",
    ]
    keys = _consolidated_keys(zarr_path)
    assert "tables/table" in keys
    assert not [key for key in keys if "merxen-backup" in key]
    assert "replaced" in sd.read_zarr(zarr_path).tables["table"].obs


def test_recoverable_backup_reconsolidates_after_failed_write(
    tmp_path: Path,
) -> None:
    """A failed replacement restores the element and drops the backup's entries."""
    from merxen.io.spatialdata_io import _write_element_with_recoverable_backup

    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    sdata = sd.read_zarr(zarr_path)

    def _failing_write(key: str, *, overwrite: bool) -> None:
        # write_element consolidates after writing; fail after that point
        sdata.write_consolidated_metadata()
        raise RuntimeError("write failed")

    sdata.write_element = _failing_write  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="write failed"):
        _write_element_with_recoverable_backup(sdata, "table", "tables")

    assert (zarr_path / "tables" / "table").is_dir()
    keys = _consolidated_keys(zarr_path)
    assert "tables/table" in keys
    assert not [key for key in keys if "merxen-backup" in key]


def _ome_labels(zarr_path: Path) -> list[str]:
    group = json.loads((zarr_path / "labels" / "zarr.json").read_text())
    return list(group["attributes"]["ome"]["labels"])


def _consolidated_ome_labels(zarr_path: Path) -> list[str]:
    root = json.loads((zarr_path / "zarr.json").read_text())
    group = root["consolidated_metadata"]["metadata"]["labels"]
    return list(group["attributes"]["ome"]["labels"])


@pytest.mark.parametrize("scale_factors", [None, [2]])
def test_write_or_replace_element_replacing_labels_keeps_ome_labels_unique(
    tmp_path: Path,
    scale_factors: list[int] | None,
) -> None:
    """Replacing a label element twice must not repeat it in ``ome.labels``.

    ome-zarr's label writer appends the element name to the labels group list
    on every write, so each replacement of a stored label used to add a copy.
    """
    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    sdata = sd.read_zarr(zarr_path)
    before = _ome_labels(zarr_path)
    pixels = np.asarray(sdata.labels["blobs_labels"].data)

    for _ in range(2):
        replacement = Labels2DModel.parse(
            pixels,
            dims=("y", "x"),
            scale_factors=scale_factors,
        )
        write_or_replace_element(
            sdata, "blobs_labels", "labels", replacement, overwrite=True
        )

    assert _ome_labels(zarr_path) == before
    assert _consolidated_ome_labels(zarr_path) == before
    reloaded = sd.read_zarr(zarr_path)
    assert sorted(reloaded.labels) == sorted(before)


def test_write_or_replace_element_new_label_drops_existing_duplicates(
    tmp_path: Path,
) -> None:
    """Writing a label repairs earlier repeats, keeping first-seen order."""
    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    labels_group = zarr.open_group(str(zarr_path / "labels"), mode="r+")
    ome = dict(labels_group.attrs["ome"])
    ome["labels"] = [
        "blobs_multiscale_labels",
        "blobs_labels",
        "blobs_multiscale_labels",
        "blobs_labels",
    ]
    labels_group.attrs["ome"] = ome
    sdata = sd.read_zarr(zarr_path)
    pixels = np.asarray(sdata.labels["blobs_labels"].data)

    write_or_replace_element(
        sdata,
        "added_labels",
        "labels",
        Labels2DModel.parse(pixels, dims=("y", "x")),
    )

    expected = ["blobs_multiscale_labels", "blobs_labels", "added_labels"]
    assert _ome_labels(zarr_path) == expected
    assert _consolidated_ome_labels(zarr_path) == expected


def test_deduplicate_ome_labels_metadata_reports_repeats_and_dry_run(
    tmp_path: Path,
) -> None:
    """The repair helper lists repeats and only rewrites when asked to."""
    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    labels_group = zarr.open_group(str(zarr_path / "labels"), mode="r+")
    ome = dict(labels_group.attrs["ome"])
    ome["labels"] = ["blobs_labels", "blobs_multiscale_labels", "blobs_labels"]
    labels_group.attrs["ome"] = ome

    removed = deduplicate_ome_labels_metadata(zarr_path, dry_run=True)

    assert removed == ["blobs_labels"]
    assert _ome_labels(zarr_path) == ome["labels"]

    removed = deduplicate_ome_labels_metadata(zarr_path)

    assert removed == ["blobs_labels"]
    assert _ome_labels(zarr_path) == ["blobs_labels", "blobs_multiscale_labels"]
    assert deduplicate_ome_labels_metadata(zarr_path) == []


def test_deduplicate_ome_labels_metadata_handles_zarr_v2_layout(
    tmp_path: Path,
) -> None:
    """NGFF 0.4 stores keep the list at the top level of ``.zattrs``."""
    zarr_path = tmp_path / "legacy.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
    labels_group = root.create_group("labels")
    labels_group.attrs["labels"] = ["cells", "nuclei", "cells"]

    removed = deduplicate_ome_labels_metadata(zarr_path)

    assert removed == ["cells"]
    reread = zarr.open_group(str(zarr_path / "labels"), mode="r")
    assert reread.attrs["labels"] == ["cells", "nuclei"]


def test_deduplicate_ome_labels_metadata_ignores_store_without_labels(
    tmp_path: Path,
) -> None:
    """Stores with no labels group need no repair."""
    zarr_path = tmp_path / "points_only.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    assert deduplicate_ome_labels_metadata(zarr_path) == []


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


def test_normalize_points_preserves_proseg_assignment_zero_and_raw_xy() -> None:
    """ProSeg nullable assignment and observed coordinates should survive."""
    points = pd.DataFrame(
        {
            "x": [10.0, 20.0],
            "y": [30.0, 40.0],
            "z": [0.2, 0.8],
            "observed_x": [1.0, 2.0],
            "observed_y": [3.0, 4.0],
            "observed_z": [0.0, 1.0],
            "gene": ["A", "B"],
            "assignment": [0.0, None],
            "background": [False, True],
        }
    )

    out = normalize_points_for_latest_write(points)

    assert out["x"].tolist() == [1.0, 2.0]
    assert out["y"].tolist() == [3.0, 4.0]
    assert out["z"].tolist() == [0.0, 1.0]
    assert out["proseg_moved_x"].tolist() == [10.0, 20.0]
    assert out["proseg_moved_y"].tolist() == [30.0, 40.0]
    assert out["proseg_moved_z"].tolist() == [0.2, 0.8]
    assert int(out.loc[0, "assignment"]) == 0
    assert pd.isna(out.loc[1, "assignment"])


def test_normalize_points_handles_dask_proseg_points() -> None:
    """Dask point partitions should keep nullable ProSeg assignments."""
    points = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [11.0, 21.0, 31.0],
            "observed_x": [1.0, 2.0, 3.0],
            "observed_y": [4.0, 5.0, 6.0],
            "gene": ["A", "B", "C"],
            "assignment": [0.0, 1.0, None],
            "background": [False, False, True],
        }
    )
    ddf = dd.from_pandas(points, npartitions=2)

    out = normalize_points_for_latest_write(ddf).compute()

    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    assert out["y"].tolist() == [4.0, 5.0, 6.0]
    assert out["proseg_moved_x"].tolist() == [10.0, 20.0, 30.0]
    assert out["assignment"].isna().tolist() == [False, False, True]
    assert int(out.loc[0, "assignment"]) == 0
