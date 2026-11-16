"""An element replacement killed part-way is rolled back, never half kept.

:func:`write_or_replace_element` moves the stored element aside as a hidden
backup while it writes the new one. A writer killed in between (SIGKILL, so no
``except`` runs) leaves the backup and a missing or partial element; the next
store lock or the next write of that element must put the old element back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import spatialdata as sd
from spatialdata import datasets

from merxen.io import spatialdata_io
from merxen.io.spatialdata_io import (
    recover_interrupted_element_writes,
    spatialdata_write_lock,
    write_or_replace_element,
)


class Killed(BaseException):
    """Stands in for SIGKILL: nothing below the raise runs, no handler cleans up."""


def _hidden(zarr_path: Path) -> list[str]:
    return sorted(p.name for p in (zarr_path / "tables").iterdir() if p.name[0] == ".")


def _consolidated_keys(zarr_path: Path) -> list[str]:
    root = json.loads((zarr_path / "zarr.json").read_text())
    return list(root["consolidated_metadata"]["metadata"])


@pytest.fixture
def store(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "latest.zarr"
    datasets.blobs().write(zarr_path)
    return zarr_path


def _replacement(sdata: Any, column: str) -> Any:
    table = sdata.tables["table"].copy()
    table.obs[column] = 1
    return table


def _kill_mid_write(sdata: Any, *, after_full_write: bool) -> None:
    """Make the next element write stop part-way (or just after it finishes)."""
    real_write = sdata.write_element

    def _write(key: str, *, overwrite: bool) -> None:
        if overwrite:
            # SpatialData refuses to overwrite a stored element in place, which
            # sends the replacement to the backup route.
            real_write(key, overwrite=overwrite)
        if after_full_write:
            real_write(key, overwrite=overwrite)
        else:
            # The group exists but its arrays were never written.
            partial = Path(sdata.path) / "tables" / key
            partial.mkdir()
            (partial / "zarr.json").write_text('{"zarr_format": 3, "node_')
        raise Killed

    sdata.write_element = _write


@pytest.mark.parametrize("after_full_write", [False, True])
def test_a_replacement_killed_mid_write_is_rolled_back_by_the_next_lock(
    store: Path, after_full_write: bool
) -> None:
    sdata = sd.read_zarr(store)
    _kill_mid_write(sdata, after_full_write=after_full_write)

    with pytest.raises(Killed):
        write_or_replace_element(sdata, "table", "tables", _replacement(sdata, "new"))
    assert [name for name in _hidden(store) if ".merxen-backup-" in name]

    with spatialdata_write_lock(store):
        table = sd.read_zarr(store).tables["table"]

    assert "new" not in table.obs
    assert _hidden(store) == []
    assert not [key for key in _consolidated_keys(store) if "merxen-" in key]


def test_the_next_write_of_the_element_restores_it_first(store: Path) -> None:
    """A writer that takes no store lock still never builds on a partial element."""
    sdata = sd.read_zarr(store)
    first, second = _replacement(sdata, "new"), _replacement(sdata, "again")
    real_write = sdata.write_element
    _kill_mid_write(sdata, after_full_write=False)
    with pytest.raises(Killed):
        write_or_replace_element(sdata, "table", "tables", first)
    sdata.write_element = real_write

    write_or_replace_element(sdata, "table", "tables", second)

    assert _hidden(store) == []
    table = sd.read_zarr(store).tables["table"]
    assert "again" in table.obs and "new" not in table.obs


def test_a_kill_while_deleting_the_backup_keeps_the_new_element(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backup is renamed before it is deleted, so it is never restored half
    deleted over the complete new element."""
    real_remove = spatialdata_io.remove_path

    def _remove(path: Path) -> None:
        if ".merxen-discard-" in Path(path).name:
            # Part of the backup is gone when the kill lands.
            for child in sorted(Path(path).iterdir())[:1]:
                real_remove(child)
            raise Killed
        real_remove(path)

    sdata = sd.read_zarr(store)
    monkeypatch.setattr(spatialdata_io, "remove_path", _remove)
    with pytest.raises(Killed):
        write_or_replace_element(sdata, "table", "tables", _replacement(sdata, "new"))
    monkeypatch.setattr(spatialdata_io, "remove_path", real_remove)
    assert [name for name in _hidden(store) if ".merxen-discard-" in name]

    with spatialdata_write_lock(store):
        table = sd.read_zarr(store).tables["table"]

    assert "new" in table.obs
    assert _hidden(store) == []


def test_a_backup_without_a_journal_is_left_alone(store: Path) -> None:
    """Older versions deleted the backup in place, so one they left may be half
    deleted; it is never restored over the element."""
    legacy = store / "tables" / ".table.merxen-backup-0123"
    legacy.mkdir()
    with spatialdata_write_lock(store):
        pass
    assert legacy.is_dir()
    assert "table" in sd.read_zarr(store).tables


def test_a_journal_naming_another_backup_is_refused(store: Path) -> None:
    journal = store / "tables" / ".table.merxen-replacing.json"
    journal.write_text(json.dumps({"backup": "../../elsewhere"}))
    with pytest.raises(RuntimeError, match="does not name a backup of table"):
        recover_interrupted_element_writes(store)
    assert (store / "tables" / "table").is_dir()


def test_recovery_scoped_to_one_element_leaves_the_others(store: Path) -> None:
    tables = store / "tables"
    other = tables / ".other.merxen-backup-a"
    other.mkdir()
    assert (
        recover_interrupted_element_writes(
            store, element_type="tables", element_key="table"
        )
        == []
    )
    assert other.is_dir()
