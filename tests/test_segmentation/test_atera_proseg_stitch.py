"""ATERA_PROSEG_STITCH: claim resolution and the stitched layer (plan AT-9)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import spatialdata as sd

from merxen.analysis_layers import validate_analysis_layer
from merxen.io.points_aggregates import fingerprint_matches
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR
from merxen.segmentation.atera_proseg import AteraProsegError
from merxen.segmentation.atera_proseg.plan import write_plan
from merxen.segmentation.atera_proseg.stitch import (
    ClaimSet,
    rasterise_cells,
    resolve_claims,
    stitch,
)
from merxen.segmentation.atera_proseg.tile import run_tile
from test_segmentation.atera_proseg_fixture import (
    TILE_UM,
    build_multitile_latest,
    fake_proseg_binary,
    plan_config,
    stitch_config,
    tile_configs,
)


@dataclass(frozen=True)
class Built:
    source: Path
    latest: Path
    plan_path: Path
    tile_dirs: list[Path]


def _build(root: Path, mode: str) -> Built:
    source, latest = build_multitile_latest(root)
    plan_path = write_plan(plan_config(source, root / "plan_out"))
    plan = json.loads(plan_path.read_text())
    binary = fake_proseg_binary(root / "bin", FAKE_PROSEG_MODE=mode)
    dirs = []
    for config in tile_configs(plan, source, root / "tiles"):
        run_tile(config, binary)
        dirs.append(config.output_dir)
    return Built(source, latest, plan_path, dirs)


@pytest.fixture(scope="module")
def identity(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return _build(tmp_path_factory.mktemp("identity"), "identity")


@pytest.fixture(scope="module")
def greedy(tmp_path_factory: pytest.TempPathFactory) -> Built:
    return _build(tmp_path_factory.mktemp("greedy"), "greedy_halo")


def _copy(built: Built, tmp_path: Path) -> Built:
    """A private copy of the latest store (stitching writes into it)."""
    latest = tmp_path / "latest.zarr"
    shutil.copytree(built.latest, latest)
    return Built(built.source, latest, built.plan_path, built.tile_dirs)


def _manifest(root: Path) -> dict[str, tuple[int, int, int, str]]:
    found = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        info = path.stat()
        found[str(path.relative_to(root))] = (
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return found


# --------------------------------------------------------------------------- #
# Claim resolution
# --------------------------------------------------------------------------- #


def _claims(
    tile: str,
    index: int,
    grid: tuple[int, int],
    rows: list[int],
    cells: list[int],
    background: list[bool] | None = None,
) -> ClaimSet:
    return ClaimSet(
        tile_id=tile,
        plan_index=index,
        grid=grid,
        rows=np.asarray(rows),
        instance_id=np.asarray(cells, dtype=np.uint64),
        background=np.asarray(background or [False] * len(rows)),
    )


def test_one_claim_none_and_conflicts() -> None:
    # Row 0: one claim. Row 1: none. Row 2: two tiles agree. Row 3: conflict in
    # the core of the second tile. Row 4: conflict in neither core (lowest
    # plan index wins).
    x = np.asarray([1.0, 1.0, 7.9, 8.5, 20.0])
    y = np.asarray([1.0, 1.0, 1.0, 1.0, 20.0])
    first = _claims("c0_r0", 0, (0, 0), [0, 2, 3, 4], [5, 6, 6, 6])
    second = _claims("c1_r0", 1, (1, 0), [2, 3, 4], [6, 9, 9], [False, True, False])
    result = resolve_claims(5, x, y, 8.0, [second, first])
    assert result.assignment.tolist() == [5, 0, 6, 9, 6]
    assert result.background.tolist() == [False, True, False, True, False]
    assert result.claimed.tolist() == [True, False, True, True, True]
    assert result.conflicts == {"c1_r0|c0_r0": 1, "c0_r0|c1_r0": 1}
    assert result.losing_cells == {6: 1, 9: 1}


def test_rasterise_uses_pixel_centres_and_last_wins() -> None:
    from shapely.geometry import box

    out = np.zeros((6, 6), dtype=np.uint32)
    # Box 1 covers pixel centres 0.5..2.5 (x, y in [0, 3)); box 2 overlaps it.
    overlap = rasterise_cells(
        [box(0.0, 0.0, 3.0, 3.0), box(2.0, 2.0, 4.9, 4.9)],
        np.asarray([1, 2]),
        (6, 6),
        1.0,
        out,
    )
    assert out[0, 0] == 1 and out[2, 2] == 2 and out[4, 4] == 2
    assert out[5, 5] == 0 and out[0, 4] == 0
    assert overlap == 1


# --------------------------------------------------------------------------- #
# Stitched store
# --------------------------------------------------------------------------- #


def test_identity_stitch_reproduces_the_vendor_counts(
    identity: Built, tmp_path: Path
) -> None:
    built = _copy(identity, tmp_path)
    source_before = _manifest(built.source)
    points_dir = built.latest / "points" / "transcripts" / "points.parquet"
    before = {
        path.name: pq.read_table(path) for path in sorted(points_dir.glob("*.parquet"))
    }
    report = stitch(
        stitch_config(built.latest, built.plan_path, built.tile_dirs, tmp_path / "out")
    )
    assert report["skipped"] is False
    assert report["n_claim_conflicts"] == 0
    assert report["n_zero_transcript_cells"] == 0
    assert (tmp_path / "out" / "proseg_latest.zarr").exists()
    assert json.loads((tmp_path / "out" / "atera_proseg_report.json").read_text())

    # Source untouched; points keep names, rows, order and columns (+2).
    assert _manifest(built.source) == source_before
    after = {
        path.name: pq.read_table(path) for path in sorted(points_dir.glob("*.parquet"))
    }
    assert list(after) == list(before)
    for name, table in after.items():
        old = before[name]
        assert table.num_rows == old.num_rows
        data = [c for c in old.column_names if c != "__null_dask_index__"]
        assert table.column_names == [
            *data,
            "assignment",
            "background",
            "__null_dask_index__",
        ]
        assert table.column("transcript_id").equals(old.column("transcript_id"))
        assert table.column("x").equals(old.column("x"))
    assert fingerprint_matches(built.latest)

    sdata = sd.read_zarr(built.latest)
    table = sdata.tables["table_MOSAIK_proseg"]
    original = sdata.tables["table_original"]
    assert list(table.var_names) == list(original.var_names)
    assert table.var.equals(original.var)
    vendor = dict(
        zip(
            original.obs["instance_id"].astype(int),
            np.asarray(original.X.sum(axis=1)).ravel(),
            strict=True,
        )
    )
    proseg = np.asarray(table.X.sum(axis=1)).ravel()
    for cell, count in zip(table.obs["instance_id"].astype(int), proseg, strict=True):
        assert count == vendor[cell]
    assert (table.obs["transcript_counts"].to_numpy() == proseg).all()
    points = sdata.points["transcripts"].compute()
    genes = points[points["is_gene"]]
    assert (
        genes["assignment"].fillna(0).astype("int64")
        == genes["original_assignment"].fillna(0).astype("int64")
    ).all()
    controls = points[~points["is_gene"]]
    assert controls["assignment"].isna().all() and controls["background"].all()
    # Controls are attributed through the raster (here equal to the vendor mask).
    control_columns = [
        "genomic_control_counts",
        "unassigned_codeword_counts",
        "deprecated_codeword_counts",
        "control_codeword_counts",
        "control_probe_counts",
        "other_control_counts",
    ]
    per_cell = table.obs[control_columns].sum(axis=1)
    vendor_controls = (
        controls["original_assignment"].dropna().astype(int).value_counts()
    )
    for cell, count in zip(table.obs["instance_id"].astype(int), per_cell, strict=True):
        assert count == vendor_controls.get(cell, 0)
    assert np.array_equal(
        np.asarray(sdata.labels["proseg_cell_labels"]["scale0"]["image"]),
        np.asarray(sdata.labels["cell_labels"]["scale0"]["image"]),
    )
    entry = sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]["proseg"]
    assert entry["shape"] == "MOSAIK_proseg"
    assert entry["table"] == "table_MOSAIK_proseg"
    assert entry["assignment_column"] == "assignment"
    assert entry["background_column"] == "background"
    assert entry["id_namespace"] == "source_segmentation_v1"
    assert entry["legacy_aliases"] == ["reseg"]
    assert entry["label"] == "proseg_cell_labels"
    assert entry["prior"] == "vendor_cell_labels"
    assert entry["model_identity"]["tool"] == "proseg"
    assert sdata.attrs[MERXEN_SCHEMA_ATTR]["segmentation_source"] == "native"
    provenance = sdata.attrs["merxen_atera_proseg"]
    assert len(provenance["tiles"]) == len(built.tile_dirs)
    assert provenance["stitch"]["control_assignment"] == "proseg_label_raster_corner"
    assert (
        validate_analysis_layer(
            built.latest,
            platform="ATERA",
            segmentation="reseg",
            table_key="table_MOSAIK_proseg",
            shape_key="MOSAIK_proseg",
        )["n_cells"]
        == table.n_obs
    )

    # A second stitch of the same tiles is a skip.
    again = stitch(
        stitch_config(built.latest, built.plan_path, built.tile_dirs, tmp_path / "out2")
    )
    assert again["skipped"] is True


def test_greedy_halo_stitch_counts_conflicts(greedy: Built, tmp_path: Path) -> None:
    built = _copy(greedy, tmp_path)
    report = stitch(
        stitch_config(built.latest, built.plan_path, built.tile_dirs, tmp_path / "out")
    )
    assert report["n_claim_conflicts"] > 0
    assert report["n_claim_conflicts"] == sum(
        report["claim_conflicts_by_tile_pair"].values()
    )
    sdata = sd.read_zarr(built.latest)
    table = sdata.tables["table_MOSAIK_proseg"]
    points = sdata.points["transcripts"].compute()
    counted = points[
        points["is_gene"] & ~points["background"] & points["assignment"].notna()
    ]
    expected = counted["assignment"].astype(int).value_counts()
    for cell, count in zip(
        table.obs["instance_id"].astype(int),
        np.asarray(table.X.sum(axis=1)).ravel(),
        strict=True,
    ):
        assert count == expected.get(cell, 0)
    # Each conflicted row went to the tile whose core holds it when one does.
    assert report["n_rows_assigned"] == int(points["assignment"].notna().sum())


def test_missing_tile_and_subset_plans_are_refused(
    identity: Built, tmp_path: Path
) -> None:
    built = _copy(identity, tmp_path)
    with pytest.raises(AteraProsegError, match="missing"):
        stitch(
            stitch_config(
                built.latest, built.plan_path, built.tile_dirs[1:], tmp_path / "out"
            )
        )
    plan = json.loads(built.plan_path.read_text())
    plan["tile_subset"] = ["c0_r0"]
    subset_plan = tmp_path / "tile_plan.json"
    subset_plan.write_text(json.dumps(plan))
    with pytest.raises(AteraProsegError, match="cannot be stitched"):
        stitch(
            stitch_config(built.latest, subset_plan, built.tile_dirs, tmp_path / "o")
        )


def test_a_latest_store_of_another_build_is_refused(
    identity: Built, tmp_path: Path
) -> None:
    built = _copy(identity, tmp_path)
    marker = built.latest / "merxen_native_latest.json"
    data = json.loads(marker.read_text())
    data["source_build_fingerprint"] = "0" * 64
    marker.write_text(json.dumps(data))
    with pytest.raises(AteraProsegError, match="another source build"):
        stitch(
            stitch_config(
                built.latest, built.plan_path, built.tile_dirs, tmp_path / "o"
            )
        )


def test_tile_grid_constant_matches_the_fixture() -> None:
    assert TILE_UM == 8.0
