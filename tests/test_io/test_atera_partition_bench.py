"""Atera partition-read benchmark script on the synthetic bundle (plan §3.3, §7.4)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from merxen.config import AteraBuildConfig
from merxen.io.builders import atera as atera_builder

from .atera_synth import write_atera_bundle

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/acceptance/atera_partition_bench.py"
)
#: The synthetic section spans about 18 x 20 µm in one 100 µm density bin, so
#: a window around the bin centre must be large to reach it.
WINDOW_UM = 120.0
FAST = {
    "transcript_partition_rows": 1_048_576,
    "polygon_n_jobs": 2,
    "morphology_levels": 3,
}


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    """Load the standalone benchmark script without running it."""
    spec = importlib.util.spec_from_file_location("atera_partition_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One store per partition order, built from the same synthetic bundle."""
    root = tmp_path_factory.mktemp("atera_bench")
    synth = write_atera_bundle(root / "roi")
    built = {}
    for order, extra in (
        ("row_range", {}),
        ("tile_blocks", {"spatial_block_um": 2.5, "min_block_rows": 1}),
    ):
        built[order] = atera_builder.write_atera_spatialdata(
            input_path=synth.root,
            output_path=root / f"{order}.zarr",
            build_config=AteraBuildConfig(spatial_order=order, **FAST, **extra),
            n_workers=2,
        )
    return built


@pytest.mark.parametrize("order", ["row_range", "tile_blocks"])
def test_benchmark_reads_check_out_on_both_orders(
    bench: ModuleType, stores: dict[str, Path], order: str
) -> None:
    report = bench.run_benchmark(
        stores[order], alias="SYN1_ATERA", window_um=WINDOW_UM, n_windows=3, repeats=1
    )
    assert report["spatial_order"] == order
    assert report["alias"] == "SYN1_ATERA"
    assert len(report["windows"]) == 3
    for window in report["windows"]:
        assert 0 < window["files_read"] <= window["files_total"]
        assert window["rows"] <= window["rows_in_files_read"]
    assert "full_scan_seconds" in report["windows"][0]
    assert report["genes"]
    assert all(gene["rows"] > 0 for gene in report["genes"])


def test_both_orders_read_the_same_windows_and_rows(
    bench: ModuleType, stores: dict[str, Path]
) -> None:
    reports = {
        order: bench.run_benchmark(
            store, alias="SYN1_ATERA", window_um=WINDOW_UM, n_windows=3, repeats=1
        )
        for order, store in stores.items()
    }
    rows = {
        order: [(w["centre_um"], w["rows"]) for w in report["windows"]]
        for order, report in reports.items()
    }
    assert rows["row_range"] == rows["tile_blocks"]
    genes = {
        order: [(g["gene"], g["rows"]) for g in report["genes"]]
        for order, report in reports.items()
    }
    assert genes["row_range"] == genes["tile_blocks"]


def test_a_gene_count_mismatch_fails(
    bench: ModuleType, stores: dict[str, Path], tmp_path: Path
) -> None:
    import shutil

    store = tmp_path / "store.zarr"
    shutil.copytree(stores["row_range"], store)
    root = json.loads((store / "zarr.json").read_text())
    transcripts = root["attributes"]["merxen_atera"]["transcripts"]
    kept = transcripts["gene_counts_kept"]
    top = max(range(len(kept)), key=kept.__getitem__)
    kept[top] += 1
    (store / "zarr.json").write_text(json.dumps(root))
    with pytest.raises(RuntimeError, match="gene_counts_kept"):
        bench.run_benchmark(
            store, alias="SYN1_ATERA", window_um=WINDOW_UM, n_windows=1, repeats=1
        )


def test_cli_writes_a_report_without_paths(
    bench: ModuleType, stores: dict[str, Path], tmp_path: Path
) -> None:
    report_path = tmp_path / "bench.json"
    status = bench.main(
        [
            "--store",
            str(stores["row_range"]),
            "--alias",
            "SYN1_ATERA",
            "--report",
            str(report_path),
            "--window-um",
            str(WINDOW_UM),
            "--windows",
            "2",
            "--repeats",
            "1",
        ]
    )
    assert status == 0
    text = report_path.read_text()
    report = json.loads(text)
    assert report["summary"]["window_median_files_read"] > 0
    assert str(tmp_path) not in text


def test_window_pruning_uses_the_file_bounding_boxes(bench: ModuleType) -> None:
    entries = [
        {"file": "a", "bbox_um": [0.0, 10.0, 0.0, 10.0]},
        {"file": "b", "bbox_um": [20.0, 30.0, 0.0, 10.0]},
        {"file": "c", "bbox_um": None},
    ]
    picked = bench._files_for_window(entries, (5.0, 15.0, 5.0, 15.0))
    assert [e["file"] for e in picked] == ["a", "c"]
    assert bench._read([], ["x"], bench._window_filter((0, 1, 0, 1)))[0] == 0
