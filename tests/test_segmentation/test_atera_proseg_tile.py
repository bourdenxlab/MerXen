"""ATERA_PROSEG_TILE: export, ProSeg command, run and reduction (plan AT-9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from merxen.config import AteraProsegModelParams
from merxen.segmentation.atera_proseg import AteraProsegError
from merxen.segmentation.atera_proseg.export import (
    XENIUM_SCHEMA,
    export_tile,
    window_row_mask,
    xenium_layout_table,
)
from merxen.segmentation.atera_proseg.plan import build_plan
from merxen.segmentation.atera_proseg.runner import (
    build_command,
    check_prior_weight,
    check_version,
    parse_prior_weight,
    parse_timings,
    prior_weight_multiplier,
)
from merxen.segmentation.atera_proseg.tile import (
    cells_near_interior_edges,
    interior_edges,
    run_tile,
)
from test_segmentation.atera_proseg_fixture import (
    build_multitile_latest,
    fake_proseg_binary,
    plan_config,
    tile_configs,
)

PS = 0.21249222


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    root = tmp_path_factory.mktemp("tile")
    source, latest = build_multitile_latest(root)
    plan = build_plan(plan_config(source, root / "plan"))
    return source, latest, plan


def _points(source: Path) -> pa.Table:
    files = sorted(
        (source / "points" / "transcripts" / "points.parquet").glob("*.parquet")
    )
    return pa.concat_tables([pq.read_table(path) for path in files])


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_window_rule_is_the_float32_floor_and_end_exclusive() -> None:
    x = np.asarray([0.0, 10 * PS - 1e-4, 10 * PS, 3 * PS], dtype=np.float32)
    y = np.asarray([0.0, 0.0, 0.0, 5 * PS + 1e-3], dtype=np.float32)
    keep = window_row_mask(x, y, [0, 10, 0, 10], PS)
    # x = col1 * ps falls on column 10, outside [0, 10).
    assert keep.tolist() == [True, True, False, True]


def test_xenium_layout_schema_and_values() -> None:
    table = pa.table(
        {
            "x": pa.array([1.0, 2.0], pa.float32()),
            "y": pa.array([3.0, 4.0], pa.float32()),
            "z": pa.array([0.5, 0.5], pa.float32()),
            "feature_name": pa.array(["A", "B"]).dictionary_encode(),
            "is_gene": pa.array([True, True]),
            "qv": pa.array([30.0, 20.0], pa.float32()),
            "overlaps_nucleus": pa.array([True, False]),
            "original_assignment": pa.array([7, None], pa.uint64()),
            "transcript_id": pa.array([11, 12], pa.uint64()),
        }
    )
    out = xenium_layout_table(table)
    assert out.schema == XENIUM_SCHEMA
    assert out.column("cell_id").to_pylist() == ["7", "0"]
    assert out.column("overlaps_nucleus").to_pylist() == [1, 0]
    assert out.column("fov_name").to_pylist() == ["0", "0"]
    assert out.column("feature_name").to_pylist() == ["A", "B"]
    off = xenium_layout_table(table, nuclear_compartment=False)
    assert off.column("overlaps_nucleus").to_pylist() == [0, 0]


def test_export_writes_window_genes_only_and_the_prior(
    stores: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    source, _, plan = stores
    tile = next(t for t in plan["tiles"] if t["tile_id"] == "c1_r1")
    record = export_tile(source, tile["window_px"], PS, tmp_path, min_qv=20.0)
    written = pq.read_table(record.parquet_path)
    assert written.schema == XENIUM_SCHEMA
    points = _points(source)
    keep = (
        window_row_mask(
            points.column("x").to_numpy(),
            points.column("y").to_numpy(),
            tile["window_px"],
            PS,
        )
        & points.column("is_gene").to_numpy()
    )
    expected = np.sort(points.column("transcript_id").to_numpy()[keep])
    assert np.array_equal(np.sort(written.column("transcript_id").to_numpy()), expected)
    assert record.n_rows == expected.size
    original = points.column("original_assignment").to_pylist()
    by_id = dict(zip(points.column("transcript_id").to_pylist(), original, strict=True))
    for tid, cell in zip(
        written.column("transcript_id").to_pylist(),
        written.column("cell_id").to_pylist(),
        strict=True,
    ):
        assert cell == (str(by_id[tid]) if by_id[tid] is not None else "0")
    col0, col1, row0, row1 = tile["window_px"]
    labels = zarr.open_group(str(source / "labels" / "cell_labels"), mode="r")["s0"]
    assert np.array_equal(np.load(record.prior_path), labels[row0:row1, col0:col1])
    exported = sum(np.load(path).size for path in record.exported_dir.glob("*.npy"))
    assert exported == record.n_rows


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def test_command_golden() -> None:
    command = build_command(
        "proseg",
        parquet_path="in/tile.parquet",
        prior_path="in/prior.npy",
        output_dir="out",
        model=AteraProsegModelParams(),
        prior_weight=56.0,
        pixel_size_um=0.25,
        col0=8,
        row0=4,
        num_threads=16,
    )
    assert command == [
        "proseg", "in/tile.parquet", "xenium",
        "-x", "x_location", "-y", "y_location", "-z", "z_location",
        "--gene-column", "feature_name", "--cell-id-column", "cell_id",
        "--cell-id-unassigned", "0", "--transcript-id-column", "transcript_id",
        "--qv-column", "qv", "--min-qv", "20.0",
        "--compartment-column", "overlaps_nucleus", "--compartment-nuclear", "1",
        "--fov-column", "fov_name",
        "--voxel-size", "0.5", "--burnin-voxel-size", "1.0", "--voxel-layers", "2",
        "--nuclear-reassignment-prob", "0.2", "--diffusion-probability", "0.2",
        "--cell-compactness", "0.04",
        "--burnin-samples", "100", "--samples", "200", "--hillclimb", "20",
        "--recorded-samples", "100", "--prior-weight", "56.0",
        "--cellpose-masks", "in/prior.npy",
        "--cellpose-x-transform", "0.25", "0", "2.0",
        "--cellpose-y-transform", "0", "0.25", "1.0",
        "--nthreads", "16", "--output-path", "out",
        "--output-spatialdata", "proseg.zarr", "--exclude-spatialdata-transcripts",
        "--output-cell-metadata", "cell-metadata.parquet",
        "--output-transcript-metadata", "transcript-metadata.parquet",
        "--output-cell-polygons", "cell-polygons.geojson.gz", "--overwrite",
    ]  # fmt: skip
    extra = build_command(
        "proseg",
        parquet_path="a",
        prior_path="b",
        output_dir="c",
        model=AteraProsegModelParams(
            prior_seg_reassignment_prob=0.2, use_cell_initialization=True
        ),
        prior_weight=1.0,
        pixel_size_um=0.25,
        col0=0,
        row0=0,
        num_threads=1,
    )
    assert extra[extra.index("--prior-seg-reassignment-prob") + 1] == "0.2"
    assert "--use-cell-initialization" in extra


@pytest.mark.parametrize(
    ("rows", "labels", "expected"),
    [(102_400_000, 7006, 140.0 / (102_400_000 / 7006 / 100)), (500, 10, 140.0)],
)
def test_prior_weight_multiplier(rows: int, labels: int, expected: float) -> None:
    assert prior_weight_multiplier(140.0, rows, labels) == pytest.approx(expected)


def test_prior_weight_is_parsed_and_checked() -> None:
    stdout = (
        "Read dataset:\n 102389935 transcripts\n     7005 cells\n"
        "   146.17 prior weight\n"
    )
    assert parse_prior_weight(stdout) == 146.17
    assert parse_prior_weight("no weight") is None
    check_prior_weight(141.0, 140.0, 0.02)
    with pytest.raises(AteraProsegError, match="differs"):
        check_prior_weight(146.17, 140.0, 0.02)
    with pytest.raises(AteraProsegError, match="printed no prior weight"):
        check_prior_weight(None, 140.0, 0.02)


def test_timings_are_parsed() -> None:
    lines = [
        "[2026-11-05T20:48:05Z INFO  proseg] loaded transcripts: 66.86s",
        "[2026-11-05T20:50:31Z INFO  proseg] initialized voxels: 129.7ms",
        *(
            f"[2026-11-05T20:51:{second:02d}Z INFO  proseg] sampling times: "
            "morphology: 1.0ms, repo: 2.0s, params: 3.0µs"
            for second in (0, 2, 4, 10, 16)
        ),
    ]
    timings = parse_timings("\n".join(lines), burnin_samples=3)
    assert timings["phases_s"] == {
        "loaded transcripts": 66.86,
        "initialized voxels": 0.13,
    }
    assert timings["iterations_logged"] == 5
    assert timings["burnin_s_per_iter"] == 2.0
    assert timings["sampling_s_per_iter"] == 6.0


def test_version_refusal(tmp_path: Path) -> None:
    assert check_version(fake_proseg_binary(tmp_path / "ok")) == "3.2.0"
    other = fake_proseg_binary(tmp_path / "other", FAKE_PROSEG_VERSION="3.3.0")
    with pytest.raises(AteraProsegError, match="needs ProSeg 3.2.0"):
        check_version(other)


def test_watchdog_stops_the_task_with_exit_99(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = (
        "from pathlib import Path\n"
        "from merxen.segmentation.atera_proseg.runner import run_proseg\n"
        f"run_proseg(['sleep', '30'], log_dir=Path(r'{tmp_path}'), burnin_samples=0,"
        " max_rss_gb=1e-6)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 99, result.stderr


# --------------------------------------------------------------------------- #
# Tile (fake ProSeg)
# --------------------------------------------------------------------------- #


def _run(
    stores: tuple[Path, Path, dict], tmp_path: Path, tile: str, **env: str
) -> dict:
    source, _, plan = stores
    binary = fake_proseg_binary(tmp_path / "bin", **env)
    config = next(
        c for c in tile_configs(plan, source, tmp_path / "tiles") if c.tile_id == tile
    )
    return run_tile(config, binary) | {"_dir": str(config.output_dir)}


def test_tile_claims_only_owned_cells(
    stores: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    _, _, plan = stores
    tile = next(t for t in plan["tiles"] if t["tile_id"] == "c1_r0")
    manifest = _run(stores, tmp_path, "c1_r0")
    out = Path(manifest["_dir"])
    cells = pq.read_table(out / "cells.parquet").to_pandas()
    assert sorted(cells["instance_id"].tolist()) == tile["owned_labels"]
    claims = pa.concat_tables(
        [pq.read_table(path) for path in sorted((out / "claims").glob("*.parquet"))]
    )
    assert set(claims.column("instance_id").to_pylist()) <= set(tile["owned_labels"])
    assert claims.num_rows == manifest["rows"]["claimed"]
    assert manifest["prior_weight"]["reported"] == pytest.approx(140.0, rel=0.02)
    assert manifest["proseg"]["arguments"][0] == "proseg"
    assert not any(arg.startswith("/") for arg in manifest["proseg"]["arguments"])
    assert not (out / "work" / "inputs").exists()
    assert set(manifest["result_sha256"]) == {
        "cells.parquet",
        *(f"claims/{p.name}" for p in (out / "claims").glob("*.parquet")),
    }
    json.dumps(manifest)


def test_tile_recount_mismatch_fails(
    stores: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    with pytest.raises(AteraProsegError, match="differ from ProSeg's count matrix"):
        _run(stores, tmp_path, "c0_r0", FAKE_PROSEG_CORRUPT_X="1")


def test_tile_wrong_prior_weight_stops_the_run(
    stores: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    with pytest.raises(AteraProsegError, match="effective prior weight"):
        _run(stores, tmp_path, "c0_r0", FAKE_PROSEG_WEIGHT_FACTOR="1.5")


def test_interior_edge_check_ignores_mask_boundary_edges() -> None:
    edges = interior_edges([0, 50, 10, 97], [97, 81])
    assert edges == {"left": False, "right": True, "top": True, "bottom": False}
    bounds = np.asarray(
        [
            [0.0, 5.0, 2.0, 6.0],  # touches the left (mask) edge: ignored
            [3.0, 5.0, 50 * PS, 6.0],  # touches the right (interior) edge
            [3.0, 10 * PS + 0.1, 4.0, 6.0],  # within tolerance of the top
            [3.0, 4.0, 4.0, 5.0],
        ]
    )
    window = (0.0, 10 * PS, 50 * PS, 97 * PS)
    found = cells_near_interior_edges(
        bounds, np.asarray([1, 2, 3, 4]), window, edges, 0.5
    )
    assert found == [2, 3]


def _manifest_keys(manifest: dict[str, Any]) -> set[str]:
    return set(manifest)


def test_manifest_records_identity_and_cells(
    stores: tuple[Path, Path, dict], tmp_path: Path
) -> None:
    manifest = _run(stores, tmp_path, "c0_r1")
    expected = {"tile_id", "window_px", "source_build_fingerprint", "proseg", "rows"}
    expected |= {"cells", "prior_weight", "run", "edge_check"}
    assert expected <= _manifest_keys(manifest)
    assert manifest["cells"]["owned_labels_without_proseg_cell"] == []
    assert manifest["cells"]["owned_cells_zero_transcripts"] == []
    assert manifest["proseg"]["version"] == "3.2.0"
