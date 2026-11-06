"""Atera crop-bundle script on the synthetic bundle (plan AT-5a).

A crop must be a bundle of its own: the builder reads it and the acceptance
script passes on the store built from it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import zarr
from scipy import sparse

from merxen.config import AteraBuildConfig
from merxen.io import atera_reader as ar
from merxen.io.builders import atera as atera_builder

from .atera_synth import AteraSynth, rewrite_zip_store, write_atera_bundle

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "acceptance"
FAST = AteraBuildConfig(
    transcript_partition_rows=1_048_576, polygon_n_jobs=2, morphology_levels=3
)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def crop() -> ModuleType:
    return _load("atera_crop", SCRIPTS / "make_atera_crop.py")


@pytest.fixture(scope="module")
def acceptance() -> ModuleType:
    return _load("atera_acceptance_for_crop", SCRIPTS / "atera_build.py")


@pytest.fixture
def synth(tmp_path: Path) -> AteraSynth:
    return write_atera_bundle(tmp_path / "roi")


def _tile(key: str, n_rows: int = 10) -> ar.AteraTile:
    x, y = (int(part) for part in key.split(","))
    return ar.AteraTile(key, x, y, n_rows, 0, 10.0)


def test_window_covers_whole_pixels_of_the_selected_tiles(crop: ModuleType) -> None:
    tiles = [_tile("0,0"), _tile("0,1"), _tile("-1,1"), _tile("1,0")]
    ps = 0.21249222
    window = crop.crop_window(tiles, ["0,0", "0,1", "-1,1"], ps, (500, 500))
    # Pixels lie wholly inside [0, 10) x [0, 20) microns; negative x is clipped.
    assert (window.row0, window.row1, window.col0, window.col1) == (0, 94, 0, 47)
    assert (window.row1 * ps <= 20.0) and ((window.row1 + 1) * ps > 20.0)
    with pytest.raises(ValueError, match="overlaps the crop window"):
        crop.crop_window([*tiles, _tile("2,0")], ["0,0", "2,0"], ps, (500, 500))
    with pytest.raises(ValueError, match="unknown tiles"):
        crop.crop_window(tiles, ["9,9"], ps, (500, 500))
    # An empty tile may be left out even when it overlaps the window.
    window = crop.crop_window(
        [_tile("0,0"), _tile("1,0", 0), _tile("2,0")], ["0,0", "2,0"], ps, (500, 500)
    )
    assert window.col1 == 141


def test_vendor_excess_names_the_columns_without_transcripts(
    crop: ModuleType,
) -> None:
    matrix = sparse.csr_matrix(
        np.array([[1, 0, 2, 0, 0], [0, 3, 0, 0, 4]], dtype=np.int64)
    )
    gene_counts = np.array([5, 0, 7, 0, 0])
    names = ["A", "B", "C", "D", "E"]
    assert crop.vendor_excess(matrix, gene_counts, names) == {"B": 3, "E": 4}


def test_crop_builds_and_passes_acceptance(
    crop: ModuleType, acceptance: ModuleType, synth: AteraSynth, tmp_path: Path
) -> None:
    # A count on a column with transcripts that no crop transcript explains:
    # the crop must drop that cell and unassign its transcripts.
    cfm = synth.zip_path("cell_feature_matrix")
    target_cell = 0
    assert synth.cfm_counts[target_cell].sum() > 0

    def bump(group: zarr.Group) -> None:
        indptr = np.asarray(group["X/indptr"][:])
        data = np.asarray(group["X/data"][:])
        data[indptr[target_cell]] += 1
        group["X/data"][:] = data

    rewrite_zip_store(cfm, bump)
    manifest = crop.make_crop(synth.root, tmp_path / "crop", ["0,0"], n_workers=2)
    cells = manifest["cells"]
    assert cells["dropped_matrix_row_not_reproduced"] == 1
    assert cells["kept"] == cells["touching_the_window"] - 1
    tile = synth.transcripts[synth.transcripts["tile"] == "0,0"]
    moved = int((tile["cell_row"] == target_cell).sum())
    assert manifest["transcripts"]["rows_unassigned_from_dropped_cells"] == {
        "0,0": moved
    }
    assert manifest["transcripts"]["n_rows"] == len(tile)
    assert (tmp_path / "crop" / crop.COMPLETE_MARKER).is_file()
    assert json.loads((tmp_path / "crop" / crop.MANIFEST_NAME).read_text()) == manifest
    assert not list(tmp_path.glob("crop.partial-*"))

    outs = tmp_path / "crop" / "outs"
    bundle = ar.resolve_atera_bundle(outs)
    assert bundle.morphology_shape_yx == tuple(manifest["image_shape_yx"])
    assert [t.key for t in ar.read_atera_tile_index(bundle)] == ["0,0"]
    # Metadata files are copied byte for byte.
    for name in ("experiment.spatial", "panel_config.json"):
        assert (outs / name).read_bytes() == (synth.root / name).read_bytes()

    store = atera_builder.write_atera_spatialdata(
        input_path=outs,
        output_path=tmp_path / "crop.zarr",
        build_config=FAST,
        n_workers=2,
    )
    report = acceptance.run_acceptance(
        store,
        outs,
        alias="SYN1_ATERA_CROP",
        n_workers=2,
        expected_vendor_excess=manifest["matrix"]["expected_vendor_excess"],
    )
    assert report["passed"], json.dumps(report["criteria"], indent=1, default=str)
    points = report["criteria"]["points_vs_matrix"]["details"]
    assert points["vendor_total"] == manifest["matrix"]["total"]


def test_an_existing_crop_is_kept_without_force(
    crop: ModuleType, synth: AteraSynth, tmp_path: Path
) -> None:
    out = tmp_path / "crop"
    out.mkdir()
    with pytest.raises(FileExistsError):
        crop.make_crop(synth.root, out, ["0,0"], n_workers=1)
    crop.make_crop(synth.root, out, ["0,0"], n_workers=1, force=True)
    assert (out / crop.COMPLETE_MARKER).is_file()
    shutil.rmtree(out)
