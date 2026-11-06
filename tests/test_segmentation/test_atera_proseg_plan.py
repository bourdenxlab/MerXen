"""ATERA_PROSEG_PLAN: grid, label ownership and windows (plan AT-9)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from merxen.cli import main
from merxen.segmentation.atera_proseg import AteraProsegError, tile_id
from merxen.segmentation.atera_proseg.plan import (
    assign_owners,
    build_plan,
    read_vendor_cells,
)
from test_segmentation.atera_proseg_fixture import (
    HALO_UM,
    MARGIN_UM,
    TILE_UM,
    build_multitile_latest,
    plan_config,
)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    store, _ = build_multitile_latest(tmp_path_factory.mktemp("plan"))
    return store


def test_centroid_on_a_core_edge_goes_to_the_higher_tile() -> None:
    centroids = np.asarray([[8.0, 0.0], [7.999999, 15.9999], [0.0, 16.0], [-0.5, 3.0]])
    columns, rows = assign_owners(centroids, 8.0)
    assert columns.tolist() == [1, 0, 0, -1]
    assert rows.tolist() == [0, 1, 2, 0]
    assert tile_id(-1, 0) == "c-1_r0"


def test_every_label_has_exactly_one_owner(source: Path, tmp_path: Path) -> None:
    plan = build_plan(plan_config(source, tmp_path))
    cells = read_vendor_cells(source)
    owned = [label for tile in plan["tiles"] for label in tile["owned_labels"]]
    assert sorted(owned) == sorted(int(v) for v in cells["instance_id"])
    assert len(owned) == len(set(owned)) == plan["n_labels"]
    for tile in plan["tiles"]:
        column, row = tile["grid"]
        assert tile["tile_id"] == tile_id(column, row)
        assert tile["run"] == bool(tile["owned_labels"])
        if not tile["run"]:
            assert tile["reason"] == "no vendor cell centroid in the core"


def test_windows_cover_owned_cells_and_snap_to_pixels(
    source: Path, tmp_path: Path
) -> None:
    plan = build_plan(plan_config(source, tmp_path))
    ps = plan["pixel_size_um"]
    n_rows, n_cols = plan["mask_shape_px"]
    cells = read_vendor_cells(source)
    bounds = dict(
        zip((int(v) for v in cells["instance_id"]), cells["bounds"], strict=True)
    )
    for tile in plan["tiles"]:
        col0, col1, row0, row1 = tile["window_px"]
        assert 0 <= col0 < col1 <= n_cols and 0 <= row0 < row1 <= n_rows
        x0, y0, x1, y1 = tile["core_um"]
        # The window holds the core plus the halo, clipped to the mask.
        assert col0 <= max(math.floor((x0 - HALO_UM) / ps), 0)
        assert row0 <= max(math.floor((y0 - HALO_UM) / ps), 0)
        assert col1 >= min(math.ceil((x1 + HALO_UM) / ps), n_cols)
        assert row1 >= min(math.ceil((y1 + HALO_UM) / ps), n_rows)
        for label in tile["owned_labels"]:
            bx0, by0, bx1, by1 = bounds[label]
            assert col0 * ps <= max(bx0 - MARGIN_UM, 0) + 1e-9
            assert row0 * ps <= max(by0 - MARGIN_UM, 0) + 1e-9
            assert col1 * ps >= min(bx1 + MARGIN_UM, n_cols * ps) - 1e-9
            assert row1 * ps >= min(by1 + MARGIN_UM, n_rows * ps) - 1e-9
    assert plan["grid"] == {
        "tile_um": TILE_UM,
        "halo_um": HALO_UM,
        "owned_margin_um": MARGIN_UM,
        "max_window_extension_um": HALO_UM,
        "origin_um": [0.0, 0.0],
    }


def test_a_window_beyond_the_extension_cap_fails(source: Path, tmp_path: Path) -> None:
    with pytest.raises(AteraProsegError, match="beyond the halo"):
        build_plan(
            plan_config(
                source,
                tmp_path,
                owned_margin_um=4.0,
                max_window_extension_um=0.0,
            )
        )


def test_tile_ids_are_deterministic(source: Path, tmp_path: Path) -> None:
    first = build_plan(plan_config(source, tmp_path))
    second = build_plan(plan_config(source, tmp_path))
    assert first == second


def test_subset_marks_other_tiles_and_refuses_unknown_ids(
    source: Path, tmp_path: Path
) -> None:
    plan = build_plan(plan_config(source, tmp_path, tile_subset=["c0_r0"]))
    assert [t["tile_id"] for t in plan["tiles"] if t["run"]] == ["c0_r0"]
    assert plan["tile_subset"] == ["c0_r0"]
    with pytest.raises(AteraProsegError, match="outside the grid"):
        build_plan(plan_config(source, tmp_path, tile_subset=["c9_r9"]))


def test_plan_cli_writes_the_plan(source: Path, tmp_path: Path) -> None:
    config = plan_config(source, tmp_path / "plan_out")
    path = tmp_path / "plan_config.json"
    path.write_text(config.model_dump_json())
    result = CliRunner().invoke(main, ["atera-proseg-plan", "--config", str(path)])
    assert result.exit_code == 0, result.output
    written = json.loads((tmp_path / "plan_out" / "tile_plan.json").read_text())
    assert written["n_tiles_run"] == sum(t["run"] for t in written["tiles"])
