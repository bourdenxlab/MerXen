"""Tests for ``scripts/annotation/remove_mapfirst_outputs.py``.

A synthetic results directory with one SpatialData zarr holds a base table, a
legacy clustered table and a map_first table written by the pipeline's own
writer (``write_or_replace_element``), plus map_first and legacy output
directories. The removal must take away exactly the map_first table (and any
backup group of it), their consolidated-metadata entries and the map_first
output directories, and leave every legacy byte in place.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Point

sd = pytest.importorskip("spatialdata")
from spatialdata.models import ShapesModel, TableModel  # noqa: E402

from merxen.io.spatialdata_io import write_or_replace_element  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "annotation" / "remove_mapfirst_outputs.py"
SHAPE = "MOSAIK_proseg_hybrid"
BASE = "table_MOSAIK_proseg_hybrid"
LEGACY = "table_MOSAIK_proseg_hybrid_clustering_squidpy"
MAPFIRST = "table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst"
N_CELLS = 40


@pytest.fixture(scope="module")
def remover() -> ModuleType:
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location(
        "remove_mapfirst_outputs", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table(label: str) -> ad.AnnData:
    """Return a small table over the shape with one label column."""
    rng = np.random.default_rng(0)
    ids = np.arange(1, N_CELLS + 1)
    table = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(2.0, size=(N_CELLS, 5)).astype(np.float32)),
        obs=pd.DataFrame(
            {
                "instance_id": ids,
                "region": pd.Categorical([SHAPE] * N_CELLS),
                "broad_class": pd.Categorical([label] * N_CELLS),
            },
            index=pd.Index([str(value) for value in ids]),
        ),
        var=pd.DataFrame(index=pd.Index([f"G{index}" for index in range(5)])),
    )
    return TableModel.parse(
        table, region=SHAPE, region_key="region", instance_key="instance_id"
    )


def _tree_digest(root: Path) -> dict[str, str]:
    """Return {relative path: sha256} of every file under root."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture()
def results(tmp_path: Path) -> dict[str, object]:
    """Build a results root with legacy and map_first content for pair P1."""
    outdir = tmp_path / "results"
    zarr_path = outdir / "P1" / "merscope" / "latest" / "latest_spatialdata.zarr"
    ids = np.arange(1, N_CELLS + 1)
    shapes = gpd.GeoDataFrame(
        {"radius": np.full(N_CELLS, 5.0), "instance_id": ids},
        geometry=[Point(float(i), float(i)) for i in ids],
        index=ids,
    )
    sdata = sd.SpatialData(
        shapes={SHAPE: ShapesModel.parse(shapes)},
        tables={BASE: _table("base"), LEGACY: _table("legacy")},
    )
    zarr_path.parent.mkdir(parents=True)
    sdata.write(zarr_path)
    root_before = (zarr_path / "zarr.json").read_bytes()
    legacy_before = {
        name: _tree_digest(zarr_path / "tables" / name) for name in (BASE, LEGACY)
    }
    # The pipeline's writer, twice: FINALIZE adds the table, MENDER_IMPORT
    # replaces it (the replacement goes through a backup group).
    for label in ("mapfirst", "mapfirst_with_mender"):
        backed = sd.read_zarr(zarr_path)
        write_or_replace_element(
            backed, MAPFIRST, "tables", _table(label), overwrite=True
        )
    # A stale consolidated entry of a removed backup group of the map_first
    # table, as an interrupted replacement can leave.
    root_json = json.loads((zarr_path / "zarr.json").read_text())
    metadata = root_json["consolidated_metadata"]["metadata"]
    stale = f"tables/.{MAPFIRST}.merxen-backup-0123456789abcdef"
    metadata[stale] = metadata[f"tables/{MAPFIRST}"]
    (zarr_path / "zarr.json").write_text(json.dumps(root_json, indent=2))
    for name in (
        "annotation_panel",
        "annotation_map",
        "annotation_resolve",
        "clustering_squidpy_mapfirst",
        "mender_mapfirst",
    ):
        path = outdir / "P1" / "proseg_hybrid" / name / "out.txt"
        path.parent.mkdir(parents=True)
        path.write_text(name)
    depth = outdir / "P1" / "merscope" / "compute_cortical_depth_mapfirst" / "out.txt"
    depth.parent.mkdir(parents=True)
    depth.write_text("depth")
    for legacy_dir in (
        "P1/proseg_hybrid/clustering_squidpy",
        "P1/proseg_hybrid/mender",
        "P1/merscope/compute_cortical_depth",
    ):
        keep = outdir / legacy_dir / "keep.txt"
        keep.parent.mkdir(parents=True)
        keep.write_text("legacy")
    # What the run created, as the before/after comparison lists it
    # (type, path relative to the results root, size).
    new_paths = tmp_path / "new_paths.txt"
    created = [
        f"d\tP1/proseg_hybrid/{name}\t4096"
        for name in (*MAP_FIRST_DIRS, *ANNOTATION_DIRS)
    ]
    created += [
        "d\tP1/merscope/compute_cortical_depth_mapfirst\t4096",
        f"d\t{zarr_path.relative_to(outdir)}/tables/{MAPFIRST}\t4096",
    ]
    new_paths.write_text("\n".join(created) + "\n")
    return {
        "outdir": outdir,
        "zarr": zarr_path,
        "root_before": root_before,
        "legacy_before": legacy_before,
        "quarantine": tmp_path / "quarantine",
        "new_paths": new_paths,
    }


ANNOTATION_DIRS = ("annotation_panel", "annotation_map", "annotation_resolve")
MAP_FIRST_DIRS = ("clustering_squidpy_mapfirst", "mender_mapfirst")


def _args(results: dict[str, object], *extra: str) -> list[str]:
    return [
        "--outdir",
        str(results["outdir"]),
        "--pair",
        "P1",
        "--platform",
        "merscope",
        "--quarantine",
        str(results["quarantine"]),
        *extra,
    ]


def _with_annotation_dirs(results: dict[str, object], *extra: str) -> list[str]:
    return _args(
        results,
        "--include-annotation-dirs",
        "--new-paths",
        str(results["new_paths"]),
        *extra,
    )


def test_name_rules(remover: ModuleType) -> None:
    """Only suffixed clustered tables and their backups are removal targets."""
    assert remover.is_suffixed_table(MAPFIRST, "mapfirst")
    assert not remover.is_suffixed_table(LEGACY, "mapfirst")
    assert not remover.is_suffixed_table(BASE, "mapfirst")
    assert not remover.is_suffixed_table(
        "table_MOSAIK_proseg_hybrid_mapfirst", "mapfirst"
    )
    assert remover.is_suffixed_backup(f".{MAPFIRST}.merxen-backup-abc", "mapfirst")
    assert not remover.is_suffixed_backup(f".{BASE}.merxen-backup-abc", "mapfirst")


@pytest.mark.parametrize("suffix", ["", " ", "a/b"])
def test_refuses_an_empty_or_nested_suffix(
    remover: ModuleType, results: dict[str, object], suffix: str
) -> None:
    """An empty suffix would match legacy tables; it is refused."""
    with pytest.raises(ValueError, match="suffix"):
        remover.main(_args(results, "--suffix", suffix))


def test_dry_run_changes_nothing(
    remover: ModuleType, results: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --apply the plan is printed and nothing moves."""
    outdir = results["outdir"]
    assert isinstance(outdir, Path)
    before = _tree_digest(outdir)
    assert remover.main(_args(results)) == 0
    assert _tree_digest(outdir) == before
    printed = capsys.readouterr().out
    assert f"move table {MAPFIRST}" in printed
    assert "compute_cortical_depth_mapfirst" in printed
    assert not Path(str(results["quarantine"])).exists()


def test_apply_removes_only_map_first_content(
    remover: ModuleType, results: dict[str, object]
) -> None:
    """--apply moves the map_first table and outputs; legacy bytes stay."""
    zarr_path = results["zarr"]
    outdir = results["outdir"]
    quarantine = results["quarantine"]
    assert isinstance(zarr_path, Path) and isinstance(outdir, Path)
    assert isinstance(quarantine, Path)
    assert remover.main(_with_annotation_dirs(results, "--apply")) == 0

    assert not (zarr_path / "tables" / MAPFIRST).exists()
    assert not any(
        ".merxen-backup-" in p.name for p in (zarr_path / "tables").iterdir()
    )
    for name, digest in results["legacy_before"].items():  # type: ignore[attr-defined]
        assert _tree_digest(zarr_path / "tables" / name) == digest
    root = json.loads((zarr_path / "zarr.json").read_text())
    keys = root["consolidated_metadata"]["metadata"]
    assert not [key for key in keys if "mapfirst" in key]
    # Only the map_first entries were dropped: the root metadata is the one
    # the zarr had before the map_first table was written.
    assert json.loads(results["root_before"]) == root  # type: ignore[arg-type]

    for name in (
        "annotation_panel",
        "annotation_map",
        "annotation_resolve",
        "clustering_squidpy_mapfirst",
        "mender_mapfirst",
    ):
        assert not (outdir / "P1" / "proseg_hybrid" / name).exists()
        moved = quarantine / "outputs" / "P1" / "proseg_hybrid" / name / "out.txt"
        assert moved.read_text() == name
    assert not (outdir / "P1" / "merscope" / "compute_cortical_depth_mapfirst").exists()
    for legacy_dir in (
        "P1/proseg_hybrid/clustering_squidpy",
        "P1/proseg_hybrid/mender",
        "P1/merscope/compute_cortical_depth",
    ):
        assert (outdir / legacy_dir / "keep.txt").read_text() == "legacy"
    moved_table = quarantine / "zarr_tables" / "P1" / "merscope" / "latest"
    assert (moved_table / "latest_spatialdata.zarr" / "tables" / MAPFIRST).is_dir()

    record = json.loads((quarantine / remover.RECORD_NAME).read_text())
    dropped = record["zarrs"][0]["dropped_consolidated_entries"]
    assert f"tables/{MAPFIRST}" in dropped
    assert f"tables/.{MAPFIRST}.merxen-backup-0123456789abcdef" in dropped
    assert len(record["output_moves"]) == 6

    reread = sd.read_zarr(zarr_path)
    assert set(reread.tables) == {BASE, LEGACY}
    assert remover.main(_args(results)) == 0  # a second plan is empty


def test_annotation_dirs_stay_without_the_opt_in(
    remover: ModuleType, results: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Unsuffixed annotation directories do not say which run wrote them."""
    outdir = results["outdir"]
    assert isinstance(outdir, Path)
    assert remover.main(_args(results, "--apply")) == 0
    for name in ANNOTATION_DIRS:
        assert (outdir / "P1" / "proseg_hybrid" / name / "out.txt").read_text() == name
    for name in MAP_FIRST_DIRS:
        assert not (outdir / "P1" / "proseg_hybrid" / name).exists()
    assert "needs --include-annotation-dirs" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        remover.main(_args(results, "--include-annotation-dirs"))


def test_a_pre_existing_annotation_map_and_table_survive(
    remover: ModuleType, results: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """With --new-paths, only what the run created moves.

    Here annotation_map and the map_first table existed before the run (an
    earlier run published them; a later MAP reuses annotation_map), so both
    are kept, with the table's consolidated entries.
    """
    outdir = results["outdir"]
    zarr_path = results["zarr"]
    new_paths = results["new_paths"]
    assert isinstance(outdir, Path) and isinstance(zarr_path, Path)
    assert isinstance(new_paths, Path)
    new_paths.write_text(
        "".join(
            line + "\n"
            for line in new_paths.read_text().splitlines()
            if "annotation_map" not in line and MAPFIRST not in line
        )
    )
    root_before = (zarr_path / "zarr.json").read_bytes()
    assert remover.main(_with_annotation_dirs(results, "--apply")) == 0

    assert (outdir / "P1/proseg_hybrid/annotation_map/out.txt").read_text() == (
        "annotation_map"
    )
    assert (zarr_path / "tables" / MAPFIRST).is_dir()
    assert (zarr_path / "zarr.json").read_bytes() == root_before
    for name in ("annotation_panel", "annotation_resolve", *MAP_FIRST_DIRS):
        assert not (outdir / "P1" / "proseg_hybrid" / name).exists()
    printed = capsys.readouterr().out
    assert "annotation_map (not created by the run)" in printed
    assert f"{MAPFIRST} (not created by the run)" in printed


def test_existing_quarantine_target_is_refused_before_any_move(
    remover: ModuleType, results: dict[str, object]
) -> None:
    """A clash is found before the first move, so nothing is half removed."""
    outdir = results["outdir"]
    zarr_path = results["zarr"]
    quarantine = results["quarantine"]
    assert isinstance(outdir, Path) and isinstance(zarr_path, Path)
    assert isinstance(quarantine, Path)
    clash = quarantine / "outputs" / "P1" / "proseg_hybrid" / "annotation_map"
    clash.mkdir(parents=True)
    root_before = (zarr_path / "zarr.json").read_bytes()
    tree_before = _tree_digest(outdir)

    with pytest.raises(FileExistsError):
        remover.main(_with_annotation_dirs(results, "--apply"))

    # The table moves come first in the plan: they must not have happened.
    assert (zarr_path / "tables" / MAPFIRST).is_dir()
    assert (zarr_path / "zarr.json").read_bytes() == root_before
    assert _tree_digest(outdir) == tree_before
    for name in (*ANNOTATION_DIRS, *MAP_FIRST_DIRS):
        assert (outdir / "P1" / "proseg_hybrid" / name).is_dir(), name
    assert (outdir / "P1" / "merscope" / "compute_cortical_depth_mapfirst").is_dir()
    held = sorted(str(path.relative_to(quarantine)) for path in quarantine.rglob("*"))
    assert held == [
        "outputs",
        "outputs/P1",
        "outputs/P1/proseg_hybrid",
        "outputs/P1/proseg_hybrid/annotation_map",
    ]


def test_each_store_changes_under_its_writer_lock(
    remover: ModuleType, results: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table moves and the root prune of a zarr happen inside one lock."""
    zarr_path = results["zarr"]
    assert isinstance(zarr_path, Path)
    root_before = (zarr_path / "zarr.json").read_bytes()
    events: list[tuple[str, str, bool, bool]] = []

    @contextmanager
    def recording_lock(path: Path) -> Iterator[Path]:
        def state(kind: str) -> tuple[str, str, bool, bool]:
            return (
                kind,
                str(path),
                (zarr_path / "tables" / MAPFIRST).exists(),
                (zarr_path / "zarr.json").read_bytes() == root_before,
            )

        events.append(state("enter"))
        yield path
        events.append(state("exit"))

    monkeypatch.setattr(remover, "spatialdata_write_lock", recording_lock)
    assert remover.main(_with_annotation_dirs(results, "--apply")) == 0

    assert events == [
        ("enter", str(zarr_path), True, True),
        ("exit", str(zarr_path), False, False),
    ]
