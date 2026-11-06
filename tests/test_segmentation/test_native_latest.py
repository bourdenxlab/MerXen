"""NATIVE_LATEST: the latest store of a native ATERA row (AT-6 WP-A).

The store built by :func:`merxen.segmentation.native.build_native_latest` must
match the reference layout of ``tests/test_io/atera_native_fixture.py`` while
sharing only the immutable payloads with the source store, and writes to the
latest store must never reach the source.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from click.testing import CliRunner
from spatialdata.transformations import get_transformation
from test_io.atera_native_fixture import (
    CELL_KEY,
    CELL_SHAPES_KEY,
    NUCLEUS_SHAPES_KEY,
    SOURCE_NAMES,
    TABLE_KEY,
    make_native_reference,
)

from merxen.cli import main
from merxen.config import NativeLatestConfig
from merxen.io.builders import atera as atera_builder
from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    register_segmentation_branch,
)
from merxen.masks import discover_masks, register_missing_mask_branches
from merxen.platforms import platform_spec
from merxen.segmentation import native

DATASET = "ATERA_SYNTH"


@dataclass(frozen=True)
class Stores:
    source: Path
    reference: Path


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stores]:
    source, reference = make_native_reference(tmp_path_factory.mktemp("native"))
    yield Stores(source, reference)


@pytest.fixture
def source(stores: Stores, tmp_path: Path) -> Path:
    """A private copy of the source store (hardlinking changes file modes)."""
    copy = tmp_path / "source.zarr"
    shutil.copytree(stores.source, copy)
    return copy


def _config(source: Path, tmp_path: Path, **overrides: Any) -> NativeLatestConfig:
    values: dict[str, Any] = {
        "dataset_name": DATASET,
        "platform": "ATERA",
        "source_zarr_path": source,
        "output_dir": tmp_path / "native_out",
        "persistent_latest_zarr_path": tmp_path
        / "results"
        / "atera"
        / "latest"
        / "latest_spatialdata.zarr",
    }
    values.update(overrides)
    return NativeLatestConfig(**values)


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _manifest(root: Path) -> dict[str, tuple[int, int, int, str]]:
    """Inode, size, mtime and sha256 of every file below ``root``."""
    manifest = {}
    for path in _files(root):
        info = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[str(path.relative_to(root))] = (
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            digest,
        )
    return manifest


def _is_payload(rel: str) -> bool:
    parts = rel.split("/")
    if parts[-1] in {"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}:
        return False
    if parts[0] in {"images", "labels"} and len(parts) > 2:
        return True
    return parts[0] == "points" and rel.endswith(".parquet")


def _schema(store: Path) -> dict[str, Any]:
    root = json.loads((store / "zarr.json").read_text())
    return dict(root["attributes"][MERXEN_SCHEMA_ATTR])


# --------------------------------------------------------------------------- #
# Signatures and names
# --------------------------------------------------------------------------- #


def test_signatures_are_final() -> None:
    build = inspect.signature(native.build_native_latest)
    assert list(build.parameters) == ["cfg"]
    ensure = inspect.signature(native.ensure_native_elements)
    assert list(ensure.parameters) == ["latest_path", "source_path"]


def test_renames_match_the_registry() -> None:
    spec = platform_spec("ATERA")
    assert spec.original_cell_shape_key == native.NATIVE_CELL_SHAPES_KEY
    assert spec.original_nucleus_shape_key == native.NATIVE_NUCLEUS_SHAPES_KEY
    renamed = {target for _, target in native.SOURCE_ELEMENT_RENAMES.values()}
    assert renamed == {
        spec.original_cell_shape_key,
        spec.original_nucleus_shape_key,
        native.NATIVE_TABLE_KEY,
    }
    assert native.NUCLEUS_CELL_KEY == CELL_KEY


# --------------------------------------------------------------------------- #
# Layout equals the WP-F reference
# --------------------------------------------------------------------------- #


def test_layout_equals_the_reference(
    stores: Stores, source: Path, tmp_path: Path
) -> None:
    report = native.build_native_latest(_config(source, tmp_path))
    latest = report.latest_path
    assert not report.reused
    assert sorted(report.elements_written) == [
        f"shapes/{CELL_SHAPES_KEY}",
        f"shapes/{NUCLEUS_SHAPES_KEY}",
        f"tables/{TABLE_KEY}",
    ]
    built = sd.read_zarr(latest)
    reference = sd.read_zarr(stores.reference)
    for kind in ("images", "labels", "points", "shapes", "tables"):
        assert set(getattr(built, kind)) == set(getattr(reference, kind)), kind
        on_disk = {path.name for path in (latest / kind).iterdir() if path.is_dir()}
        assert not on_disk & SOURCE_NAMES, kind

    # Ids: shapes, nucleus -> cell join and table rows.
    for key in (CELL_SHAPES_KEY, NUCLEUS_SHAPES_KEY):
        pd.testing.assert_frame_equal(
            pd.DataFrame(built.shapes[key].drop(columns="geometry")),
            pd.DataFrame(reference.shapes[key].drop(columns="geometry")),
        )
        assert built.shapes[key].geometry.equals(reference.shapes[key].geometry)
    nuclei = built.shapes[NUCLEUS_SHAPES_KEY]
    assert nuclei.groupby(CELL_KEY).size().max() == 2
    table = built.tables[TABLE_KEY]
    ref_table = reference.tables[TABLE_KEY]
    pd.testing.assert_frame_equal(table.obs, ref_table.obs)
    assert table.uns["spatialdata_attrs"] == ref_table.uns["spatialdata_attrs"]
    assert (table.X != ref_table.X).nnz == 0
    assert set(table.obs["instance_id"].astype(np.uint64)) == set(
        built.shapes[CELL_SHAPES_KEY]["instance_id"].astype(np.uint64)
    )

    # Transforms.
    for kind in ("images", "labels", "points", "shapes"):
        for name, element in getattr(built, kind).items():
            assert get_transformation(element, get_all=True) == get_transformation(
                getattr(reference, kind)[name], get_all=True
            ), f"{kind}/{name}"

    # Registry, source flag and vendor metadata.
    built_schema = built.attrs[MERXEN_SCHEMA_ATTR]
    ref_schema = reference.attrs[MERXEN_SCHEMA_ATTR]
    assert built_schema["segmentations"] == ref_schema["segmentations"]
    assert built_schema["segmentation_source"] == "native"
    assert built_schema["platform"] == "ATERA"
    assert built.attrs["merxen_atera"] == reference.attrs["merxen_atera"]
    index = atera_builder.POINTS_INDEX_NAME
    assert (latest / index).read_bytes() == (source / index).read_bytes()
    assert (latest / "atera").is_dir()
    assert {p.relative_to(latest) for p in _files(latest / "atera")} == {
        p.relative_to(source) for p in _files(source / "atera")
    }

    # Provenance; the latest store is not a build output.
    assert not (latest / atera_builder.BUILD_MARKER_NAME).exists()
    marker = json.loads((latest / native.NATIVE_LATEST_MARKER_NAME).read_text())
    source_marker = atera_builder.read_build_marker(source)
    assert source_marker is not None
    assert marker["source_build_fingerprint"] == source_marker["build_fingerprint"]
    assert marker["link_mode"] == "hardlink"
    assert marker["files_linked"] == report.files_linked > 0
    assert built.attrs[native.PROVENANCE_ATTR]["files_linked"] == report.files_linked
    assert report.apparent_size_delta_bytes is not None
    assert 0 < report.apparent_size_delta_bytes < report.bytes_copied + 10_000_000

    # The task output is a symlink to the persistent store.
    staged = tmp_path / "native_out" / "native_latest.zarr"
    assert staged.is_symlink() and staged.resolve() == latest.resolve()
    assert not list(latest.parent.glob("*.partial-*"))


def test_discover_masks_on_the_native_store(source: Path, tmp_path: Path) -> None:
    report = native.build_native_latest(_config(source, tmp_path))
    sdata = sd.read_zarr(report.latest_path)
    masks = {mask.branch: mask for mask in discover_masks(sdata)}
    assert set(masks) == {"original", "original_nucleus"}
    assert masks["original"].shape_key == CELL_SHAPES_KEY
    assert masks["original"].table_key == TABLE_KEY
    assert masks["original"].boundary_type == "cell"
    assert masks["original"].cell_key is None
    nucleus = masks["original_nucleus"]
    assert nucleus.shape_key == NUCLEUS_SHAPES_KEY
    assert nucleus.boundary_type == "nucleus"
    assert not nucleus.has_table and not nucleus.has_assignment
    assert nucleus.cell_key == CELL_KEY
    assert nucleus.id_namespace == native.NUCLEUS_ID_NAMESPACE
    assert register_missing_mask_branches(sdata) == ()


# --------------------------------------------------------------------------- #
# The source store is never written
# --------------------------------------------------------------------------- #


def test_writes_to_the_latest_store_never_reach_the_source(
    source: Path, tmp_path: Path
) -> None:
    from merxen.enrichment import enrich

    before = _manifest(source)
    report = native.build_native_latest(_config(source, tmp_path))
    latest = report.latest_path

    source_inodes = {
        (path.stat().st_dev, path.stat().st_ino): str(path.relative_to(source))
        for path in _files(source)
    }
    for path in _files(latest):
        rel = str(path.relative_to(latest))
        key = (path.stat().st_dev, path.stat().st_ino)
        if _is_payload(rel):
            assert source_inodes.get(key) == rel, rel
            assert not path.stat().st_mode & 0o222, rel
        else:
            assert key not in source_inodes, rel

    # ENRICH's partial cleanup, a metadata write and a table rewrite in place.
    # The cleanup keeps the vendor elements of a native store (only derived
    # tables go); ensure_native_elements re-creates any that went missing.
    enrich._remove_partial_enrichment_artifacts_from_zarr_path(latest, "ATERA")
    assert native.ensure_native_elements(latest, source) == []
    for element in report.elements_written:
        shutil.rmtree(latest / element)
    recreated = native.ensure_native_elements(latest, source)
    assert sorted(recreated) == sorted(report.elements_written)
    sdata = sd.read_zarr(latest)
    write_spatialdata_metadata(sdata)
    table = sdata.tables[TABLE_KEY].copy()
    table.obs["merxen_test_column"] = np.arange(table.n_obs)
    write_or_replace_element(sdata, TABLE_KEY, "tables", table)
    sdata.write_consolidated_metadata()

    assert _manifest(source) == before
    payload = next(
        path for path in _files(latest) if _is_payload(str(path.relative_to(latest)))
    )
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            payload.open("r+b")
    reread = sd.read_zarr(latest)
    assert "merxen_test_column" in reread.tables[TABLE_KEY].obs


def test_copy_mode_shares_no_inode(source: Path, tmp_path: Path) -> None:
    before = _manifest(source)
    report = native.build_native_latest(_config(source, tmp_path, link_mode="copy"))
    assert report.files_linked == 0 and report.bytes_linked == 0
    source_inodes = {path.stat().st_ino for path in _files(source)}
    assert not {path.stat().st_ino for path in _files(report.latest_path)} & (
        source_inodes
    )
    assert _manifest(source) == before


# --------------------------------------------------------------------------- #
# Reuse, force, atomicity and refusals
# --------------------------------------------------------------------------- #


def test_a_second_run_reuses_the_store(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    first = native.build_native_latest(cfg)
    marker = first.latest_path / native.NATIVE_LATEST_MARKER_NAME
    root = first.latest_path / "zarr.json"
    stamps = (marker.stat().st_ino, root.stat().st_ino, root.stat().st_mtime_ns)
    (tmp_path / "native_out" / "native_latest.zarr").unlink()

    second = native.build_native_latest(cfg)
    assert second.reused and second.elements_written == []
    assert second.source_build_fingerprint == first.source_build_fingerprint
    assert (marker.stat().st_ino, root.stat().st_ino, root.stat().st_mtime_ns) == (
        stamps
    )
    assert (tmp_path / "native_out" / "native_latest.zarr").is_symlink()


def test_reuse_repairs_a_missing_element(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    latest = native.build_native_latest(cfg).latest_path
    shutil.rmtree(latest / "shapes" / NUCLEUS_SHAPES_KEY)
    report = native.build_native_latest(cfg)
    assert report.reused
    assert report.elements_written == [f"shapes/{NUCLEUS_SHAPES_KEY}"]
    assert NUCLEUS_SHAPES_KEY in sd.read_zarr(latest).shapes


def test_a_stale_native_store_is_rebuilt(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    latest = native.build_native_latest(cfg).latest_path
    marker_path = latest / native.NATIVE_LATEST_MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker["source_build_fingerprint"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    (latest / "tables" / "stale_leftover").mkdir()

    report = native.build_native_latest(cfg)
    assert not report.reused
    assert not (latest / "tables" / "stale_leftover").exists()
    rebuilt = json.loads(marker_path.read_text())
    assert rebuilt["source_build_fingerprint"] == report.source_build_fingerprint
    assert not list(latest.parent.glob("*.stale-*"))


def test_a_foreign_target_needs_force(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    target = cfg.persistent_latest_zarr_path
    target.mkdir(parents=True)
    (target / "something.txt").write_text("not a native store")
    with pytest.raises(native.NativeLatestError, match="force"):
        native.build_native_latest(cfg)
    assert (target / "something.txt").exists()

    report = native.build_native_latest(_config(source, tmp_path, force=True))
    assert not report.reused
    assert not (target / "something.txt").exists()
    assert (target / native.NATIVE_LATEST_MARKER_NAME).is_file()


def test_an_empty_target_directory_is_used(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    cfg.persistent_latest_zarr_path.mkdir(parents=True)
    report = native.build_native_latest(cfg)
    assert (report.latest_path / native.NATIVE_LATEST_MARKER_NAME).is_file()


def test_a_failure_leaves_no_partial_store(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(source, tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> list[str]:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(native, "_write_renamed_elements", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        native.build_native_latest(cfg)
    parent = cfg.persistent_latest_zarr_path.parent
    assert not cfg.persistent_latest_zarr_path.exists()
    assert list(parent.iterdir()) == []


def test_cross_filesystem_hardlinks_are_refused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link
    source_root = str(source.resolve())

    def _exdev(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        # Only links out of the source store cross the (simulated) boundary;
        # zarr's own atomic writes keep working.
        if str(src).startswith(source_root):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", _exdev)
    cfg = _config(source, tmp_path)
    with pytest.raises(native.NativeLatestError, match="link_mode=copy"):
        native.build_native_latest(cfg)
    assert list(cfg.persistent_latest_zarr_path.parent.iterdir()) == []

    report = native.build_native_latest(_config(source, tmp_path, link_mode="copy"))
    assert report.link_mode == "copy" and report.files_linked == 0


def test_a_staged_symlink_source_is_resolved(source: Path, tmp_path: Path) -> None:
    staged = tmp_path / "work" / "native_source.zarr"
    staged.parent.mkdir()
    staged.symlink_to(source, target_is_directory=True)
    report = native.build_native_latest(_config(staged, tmp_path))
    linked = [
        path
        for path in _files(report.latest_path)
        if _is_payload(str(path.relative_to(report.latest_path)))
    ]
    assert linked
    for path in linked:
        rel = path.relative_to(report.latest_path)
        assert path.stat().st_ino == (source / rel).stat().st_ino


def test_the_source_must_be_a_complete_atera_build(
    source: Path, tmp_path: Path
) -> None:
    marker = source / atera_builder.BUILD_MARKER_NAME
    data = json.loads(marker.read_text())
    marker.write_text(json.dumps({**data, "platform": "XENIUM"}))
    with pytest.raises(native.NativeLatestError, match="XENIUM"):
        native.build_native_latest(_config(source, tmp_path))
    marker.unlink()
    with pytest.raises(native.NativeLatestError, match="build marker"):
        native.build_native_latest(_config(source, tmp_path))
    marker.write_text(json.dumps(data))
    shutil.rmtree(source / "labels" / "nucleus_labels")
    with pytest.raises(native.NativeLatestError, match="labels/nucleus_labels"):
        native.build_native_latest(_config(source, tmp_path))


def test_the_target_must_not_overlap_the_source(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path, persistent_latest_zarr_path=source / "latest")
    with pytest.raises(native.NativeLatestError, match="overlaps"):
        native.build_native_latest(cfg)


def test_the_self_check_refuses_a_shared_non_payload(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_copy_tree = native._copy_tree
    shared: list[str] = []

    def _leaky_copy_tree(src: Path, dst: Path, **kwargs: Any) -> Any:
        # A copy step that wrongly hardlinks one copied (non-payload) file.
        tree = real_copy_tree(src, dst, **kwargs)
        rel = next(
            rel for rel in tree.copied if not rel.endswith(("zarr.json", ".zattrs"))
        )
        (dst / rel).unlink()
        os.link(src / rel, dst / rel)
        shared.append(rel)
        return tree

    monkeypatch.setattr(native, "_copy_tree", _leaky_copy_tree)
    cfg = _config(source, tmp_path)
    with pytest.raises(native.NativeLatestError, match="self-check failed") as info:
        native.build_native_latest(cfg)
    assert shared and shared[0] in str(info.value)
    assert list(cfg.persistent_latest_zarr_path.parent.iterdir()) == []


def test_leftovers_of_a_killed_run_are_removed(source: Path, tmp_path: Path) -> None:
    cfg = _config(source, tmp_path)
    target = cfg.persistent_latest_zarr_path
    target.parent.mkdir(parents=True)
    leftovers = [
        target.with_name(f"{target.name}.partial-0123"),
        target.with_name(f"{target.name}.stale-4567"),
    ]
    for path in leftovers:
        (path / "points").mkdir(parents=True)
        (path / "points" / "part.0.parquet").write_bytes(b"x")
    unrelated = target.with_name("other_spatialdata.zarr.partial-89ab")
    unrelated.mkdir()

    native.build_native_latest(cfg)
    assert not any(path.exists() for path in leftovers)
    assert unrelated.is_dir()
    assert (target / native.NATIVE_LATEST_MARKER_NAME).is_file()


def test_reuse_makes_writable_payloads_read_only(source: Path, tmp_path: Path) -> None:
    first = native.build_native_latest(
        _config(source, tmp_path, readonly_payloads=False)
    )
    latest = first.latest_path
    payloads = [
        path for path in _files(latest) if _is_payload(str(path.relative_to(latest)))
    ]
    assert payloads and all(path.stat().st_mode & 0o200 for path in payloads)

    second = native.build_native_latest(_config(source, tmp_path))
    assert second.reused
    assert second.extra["payloads_made_readonly"] == len(payloads)
    assert not any(path.stat().st_mode & 0o222 for path in payloads)
    marker = json.loads((latest / native.NATIVE_LATEST_MARKER_NAME).read_text())
    assert marker["readonly_payloads"] is True
    # The payloads still share their source inodes: nothing was rebuilt.
    for path in payloads:
        rel = path.relative_to(latest)
        assert path.stat().st_ino == (source / rel).stat().st_ino

    third = native.build_native_latest(_config(source, tmp_path))
    assert third.reused and third.extra["payloads_made_readonly"] == 0


def test_a_symlinked_source_directory_is_refused(source: Path, tmp_path: Path) -> None:
    points = source / "points" / "transcripts"
    moved = tmp_path / "elsewhere" / "transcripts"
    moved.parent.mkdir()
    shutil.move(points, moved)
    points.symlink_to(moved, target_is_directory=True)
    cfg = _config(source, tmp_path)
    with pytest.raises(native.NativeLatestError, match="symlinked directory"):
        native.build_native_latest(cfg)
    assert list(cfg.persistent_latest_zarr_path.parent.iterdir()) == []


# --------------------------------------------------------------------------- #
# ensure_native_elements
# --------------------------------------------------------------------------- #


def test_ensure_native_elements_is_idempotent(source: Path, tmp_path: Path) -> None:
    latest = native.build_native_latest(_config(source, tmp_path)).latest_path
    root = latest / "zarr.json"
    stamp = (root.stat().st_ino, root.stat().st_mtime_ns)
    assert native.ensure_native_elements(latest, source) == []
    assert (root.stat().st_ino, root.stat().st_mtime_ns) == stamp


def test_ensure_native_elements_restores_the_registry(
    source: Path, tmp_path: Path
) -> None:
    latest = native.build_native_latest(_config(source, tmp_path)).latest_path
    expected = _schema(latest)["segmentations"]
    sdata = sd.read_zarr(latest)
    schema = dict(sdata.attrs[MERXEN_SCHEMA_ATTR])
    schema["segmentations"] = {}
    schema.pop("segmentation_source")
    sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
    sdata.write_attrs()
    sdata.write_consolidated_metadata()

    assert native.ensure_native_elements(latest, source) == []
    restored = _schema(latest)
    assert restored["segmentations"] == expected
    assert restored["segmentation_source"] == "native"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_writes_the_report_and_stages_the_store(
    source: Path, tmp_path: Path
) -> None:
    assert "native-latest" in main.commands
    cfg = _config(source, tmp_path)
    config = tmp_path / "native_latest_config.json"
    config.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(main, ["native-latest", "--config", str(config)])
    assert result.exit_code == 0, result.output
    report = json.loads(
        (tmp_path / "native_out" / "native_latest_report.json").read_text()
    )
    assert report["dataset_name"] == DATASET
    assert report["reused"] is False
    assert (tmp_path / "native_out" / "native_latest.zarr").is_symlink()


# --------------------------------------------------------------------------- #
# XENIUM registry output is unchanged
# --------------------------------------------------------------------------- #


def _registry_store(*shape_keys: str, platform: str | None = None) -> Any:
    attrs: dict[str, Any] = {}
    if platform is not None:
        attrs[MERXEN_SCHEMA_ATTR] = {"platform": platform, "segmentations": {}}
    return SimpleNamespace(
        attrs=attrs,
        shapes={key: object() for key in shape_keys},
        points={"transcripts": object()},
    )


def test_xenium_nucleus_registration_is_unchanged() -> None:
    store = _registry_store("cellpose_nuclei", "xenium_nucleus", platform="XENIUM")
    assert register_missing_mask_branches(store) == (
        "cellpose_nuclei",
        "original_nucleus",
    )
    registry = store.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    common = {
        "points": "transcripts",
        "assignment_column": None,
        "table": None,
        "instance_key": "instance_id",
        "background_column": None,
        "assignment_source_column": None,
    }
    assert registry == {
        "cellpose_nuclei": {
            **common,
            "shape": "cellpose_nuclei",
            "id_namespace": "cellpose_nucleus_label_v1",
            "legacy_aliases": ["nuclei"],
        },
        "original_nucleus": {
            **common,
            "shape": "xenium_nucleus",
            "id_namespace": "source_segmentation_v1_nucleus",
            "legacy_aliases": ["nucleus_boundaries"],
        },
    }
    assert list(registry["original_nucleus"]) == [
        "points",
        "assignment_column",
        "shape",
        "table",
        "instance_key",
        "id_namespace",
        "background_column",
        "assignment_source_column",
        "legacy_aliases",
    ]
    assert all(mask.cell_key is None for mask in discover_masks(store))


def test_atera_nucleus_registration_uses_the_nucleus_row_namespace() -> None:
    store = _registry_store(NUCLEUS_SHAPES_KEY, platform="ATERA")
    assert register_missing_mask_branches(store) == ("original_nucleus",)
    entry = store.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["original_nucleus"]
    assert entry["shape"] == NUCLEUS_SHAPES_KEY
    assert entry["id_namespace"] == native.NUCLEUS_ID_NAMESPACE
    assert entry["cell_key"] == CELL_KEY
    assert entry["legacy_aliases"] == []


def test_cell_key_is_stored_only_when_set() -> None:
    store = _registry_store("cells")
    register_segmentation_branch(
        store,
        "plain",
        points_key="transcripts",
        assignment_column=None,
        shape_key="cells",
        table_key=None,
        id_namespace="ns",
    )
    register_segmentation_branch(
        store,
        "keyed",
        points_key="transcripts",
        assignment_column=None,
        shape_key="cells",
        table_key=None,
        id_namespace="ns",
        cell_key=CELL_KEY,
    )
    registry = store.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert "cell_key" not in registry["plain"]
    assert registry["keyed"]["cell_key"] == CELL_KEY
