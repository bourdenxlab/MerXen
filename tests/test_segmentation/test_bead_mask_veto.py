"""Tests of the bead-mask veto of the Cellpose cell masks (synthetic data only).

V1 a ring mask centred on a modelled bead without a DAPI nucleus is dropped, V2 a
mask with a DAPI nucleus on a bead is kept, V3 masks far from beads (or near a bead
below the amplitude limit, or just outside the distance limit) are kept, V4 the kept
labels are dense and their pixels unchanged, V5 the counts sidecar and the INFO
line, V6 pruned and blob beads are used, V7 nothing is rewritten when nothing is
vetoed, V8 the veto is a no-op (no identity, nothing prepared) without bead removal,
without a bead table or when disabled, V9 the DAPI reader, V10 a ring that cannot
be estimated keeps the mask, V11 the config, V14 the tissue DAPI floor keeps a
cell whose ring holds brighter neighbouring nuclei, V15 the floor is the Otsu level
of the coarsest projection level. The Cellpose stage: S1 the veto runs
before the transcripts are seeded and is recorded in the mask sidecar, S2 reuse of
those masks, S3 masks from before the veto are refused, S4 no-op runs keep their
sidecar and write no veto file.
"""

from __future__ import annotations

import gzip
import json
import logging
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from merxen.config import BeadMaskVetoConfig, SegmentationConfig
from merxen.io.bead_removal import qc
from merxen.io.bead_removal.provenance import segmentation_sidecar_path
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME
from merxen.segmentation import bead_mask_veto as veto_module
from merxen.segmentation.bead_mask_veto import (
    BEAD_QC_DIRNAME,
    VETO_SIDECAR_NAME,
    PreparedBeadMaskVeto,
    apply_bead_mask_veto,
    bead_mask_veto_identity,
    bead_table_path,
    has_dapi_nucleus,
    make_dapi_reader,
    prepare_bead_mask_veto,
    read_bead_table,
    tissue_dapi_floor,
    veto_sidecar_path,
)
from merxen.segmentation.pipeline import run_cellpose_segmentation

from .test_pipeline_bead_sidecar import _config, _record, _source_zarr

SIZE = 240
BACKGROUND = 100.0
# (y, x, A0, blob, pruned) of the modelled beads.
BEADS = (
    (60, 60, 1000, 0, 0),  # under the ring mask: no nucleus -> vetoed
    (60, 160, 900, 1, 1),  # pruned blob bead under a mask with a nucleus -> kept
    (160, 60, 100, 0, 0),  # below the amplitude limit -> not a bead for the veto
    (110, 110, 800, 0, 0),  # 15 px from its mask centre, beyond the limit of 8 px
    (200, 200, 800, 0, 0),  # 4 px from its mask centre, no nucleus -> vetoed
)
# Mask label -> (centre row, centre column, inner radius, outer radius, has nucleus).
MASKS = {
    1: (60, 60, 6, 12, False),
    2: (60, 160, 0, 10, True),
    3: (160, 60, 0, 10, False),
    4: (160, 160, 0, 10, False),
    5: (110, 125, 0, 10, False),
    6: (200, 204, 0, 10, False),
}
VETOED = {1, 6}


def _disc(
    yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, inner: float, outer: float
) -> np.ndarray:
    radius = np.hypot(yy - cy, xx - cx)
    return (radius >= inner) & (radius < outer)


def make_scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Label mask (uint32) and DAPI image (float32) of the six masks."""
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    mask = np.zeros((SIZE, SIZE), np.uint32)
    dapi = BACKGROUND + np.random.default_rng(seed).normal(0, 4, (SIZE, SIZE))
    for label, (cy, cx, inner, outer, nucleus) in MASKS.items():
        mask[_disc(yy, xx, cy, cx, inner, outer)] = label
        if nucleus:
            dapi[_disc(yy, xx, cy, cx, 0, 7)] += 400.0
    return mask, dapi.astype(np.float32)


def write_bead_table(source: Path, beads: tuple[tuple[int, ...], ...] = BEADS) -> Path:
    """Write ``bead_removal_qc/beads.csv.gz`` the way the QC does."""
    path = bead_table_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        handle.write("y,x,A0,a,blob,gamma,ell,pruned,chunk\n")
        for y, x, a0, blob, pruned in beads:
            handle.write(f"{y}.0,{x}.0,{a0},{a0 // 2},{blob},1.000,0.500,{pruned},0\n")
    return path


def make_veto(
    config: BeadMaskVetoConfig,
    dapi: np.ndarray,
    beads: tuple[tuple[int, ...], ...] = BEADS,
    dapi_floor: float | None = None,
) -> PreparedBeadMaskVeto:
    """A prepared veto that reads its DAPI from an array."""
    table = pd.DataFrame(beads, columns=["y", "x", "A0", "blob", "pruned"])
    used = table[table["A0"] >= config.a0_min]
    return PreparedBeadMaskVeto(
        config=config,
        identity={"version": veto_module.VETO_ALGORITHM_VERSION},
        bead_y=used["y"].to_numpy(np.float64),
        bead_x=used["x"].to_numpy(np.float64),
        n_table_beads=len(table),
        read_dapi=lambda y0, y1, x0, x1: dapi[y0:y1, x0:x1],
        dapi_floor=dapi_floor,
    )


def make_sdata(dapi: np.ndarray) -> Any:
    """A stand-in SpatialData whose projection holds a PolyT and a DAPI channel."""
    image = xr.DataArray(
        np.stack([np.zeros_like(dapi), dapi]),
        dims=("c", "y", "x"),
        coords={"c": ["PolyT", "DAPI"]},
    )
    return types.SimpleNamespace(images={MERSCOPE_ZPROJ_IMAGE_NAME: image})


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the veto once on the six-mask scene."""
    folder = tmp_path_factory.mktemp("veto")
    mask, dapi = make_scene()
    path = folder / "cellpose_masks_tiled.npy"
    np.save(path, mask)
    result = apply_bead_mask_veto(
        path, make_veto(BeadMaskVetoConfig(), dapi), dataset_name="D"
    )
    return {"before": mask, "after": np.load(path), "result": result, "path": path}


def test_v1_a_ring_mask_on_a_bead_without_dapi_is_dropped(
    scene: dict[str, Any],
) -> None:
    before, after = scene["before"], scene["after"]
    ring = before == 1
    assert ring.sum() > 100 and not ring[60, 60]  # a ring: hollow at the bead
    assert (after[ring] == 0).all()
    # The disc on the other vetoed bead goes too.
    assert (after[before == 6] == 0).all()
    assert scene["result"].n_vetoed == len(VETOED)


def test_v2_a_mask_with_a_dapi_nucleus_on_a_bead_is_kept(
    scene: dict[str, Any],
) -> None:
    before, after = scene["before"], scene["after"]
    nucleus_mask = before == 2
    kept = after[nucleus_mask]
    assert (kept > 0).all() and len(np.unique(kept)) == 1
    assert scene["result"].n_kept_with_nucleus == 1


def test_v3_masks_far_from_beads_or_on_faint_beads_are_kept(
    scene: dict[str, Any],
) -> None:
    before, after = scene["before"], scene["after"]
    for label in (3, 4, 5):  # faint bead, no bead, bead beyond the distance limit
        assert (after[before == label] > 0).all(), label
    assert scene["result"].n_near_bead == 3  # masks 1, 2 and 6


def test_v4_kept_labels_are_dense_and_their_pixels_unchanged(
    scene: dict[str, Any],
) -> None:
    before, after = scene["before"], scene["after"]
    assert after.dtype == before.dtype
    assert sorted(np.unique(after)) == [0, 1, 2, 3, 4]
    kept = [label for label in MASKS if label not in VETOED]
    for new, old in enumerate(kept, start=1):
        assert np.array_equal(after == new, before == old)
    assert scene["result"].n_masks == 6 and scene["result"].n_masks_after == 4
    assert scene["result"].written is True


def test_v5_the_counts_sidecar_and_the_info_line(
    scene: dict[str, Any], caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    sidecar = veto_sidecar_path(scene["path"])
    assert sidecar.name == VETO_SIDECAR_NAME == "bead_mask_veto.json"
    payload = json.loads(sidecar.read_text())
    assert payload["counts"] == scene["result"].counts()
    assert payload["counts"]["n_vetoed"] == 2
    assert payload["counts"]["n_beads_used"] == 4
    assert payload["counts"]["n_table_beads"] == 5
    assert payload["parameters"] == {"version": veto_module.VETO_ALGORITHM_VERSION}

    # The line is logged at INFO with the counts (run again on a scratch copy).
    mask, dapi = make_scene()
    path = tmp_path / "again.npy"
    np.save(path, mask)
    with caplog.at_level(logging.INFO, logger=veto_module.logger.name):
        apply_bead_mask_veto(
            path, make_veto(BeadMaskVetoConfig(), dapi), dataset_name="D"
        )
    lines = [r for r in caplog.records if "Bead mask veto" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "dropped 2 of 6 Cellpose masks" in lines[0].getMessage()


def test_v6_pruned_and_blob_beads_are_used(tmp_path: Path) -> None:
    """The bead under mask 2 is a pruned blob; only its nucleus keeps the mask."""
    mask, dapi = make_scene()
    no_nucleus = dapi.copy()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    no_nucleus[_disc(yy, xx, 60, 160, 0, 7)] -= 400.0
    path = tmp_path / "pruned.npy"
    np.save(path, mask)
    result = apply_bead_mask_veto(path, make_veto(BeadMaskVetoConfig(), no_nucleus))
    assert result.n_vetoed == 3
    assert (np.load(path)[mask == 2] == 0).all()


def test_v7_nothing_is_rewritten_when_nothing_is_vetoed(tmp_path: Path) -> None:
    mask, dapi = make_scene()
    path = tmp_path / "cellpose_masks_tiled.npy"
    np.save(path, mask)
    before = path.read_bytes()

    result = apply_bead_mask_veto(path, make_veto(BeadMaskVetoConfig(), dapi, beads=()))

    assert path.read_bytes() == before
    assert result.n_vetoed == 0 and result.written is False
    # The veto ran, so the record exists.
    assert json.loads(veto_sidecar_path(path).read_text())["counts"]["n_vetoed"] == 0


def test_v8_the_veto_does_nothing_without_bead_removal_a_table_or_when_disabled(
    tmp_path: Path,
) -> None:
    source = _source_zarr(tmp_path, _record(enabled=True))
    config = BeadMaskVetoConfig()
    sdata = make_sdata(make_scene()[1])
    shape = (SIZE, SIZE)

    # No bead table in the source: a no-op.
    assert bead_mask_veto_identity(config, source, _record(enabled=True)) is None
    assert (
        prepare_bead_mask_veto(
            config,
            source=source,
            recorded=_record(enabled=True),
            sdata=sdata,
            shape=shape,
        )
        is None
    )

    write_bead_table(source)
    identity = bead_mask_veto_identity(config, source, _record(enabled=True))
    assert identity is not None
    assert identity["version"] == veto_module.VETO_ALGORITHM_VERSION
    assert identity["a0_min"] == 400.0 and identity["ring_px"] == [10.0, 25.0]
    prepared = prepare_bead_mask_veto(
        config, source=source, recorded=_record(enabled=True), sdata=sdata, shape=shape
    )
    assert prepared is not None and prepared.identity == identity
    assert prepared.n_table_beads == len(BEADS) and prepared.bead_y.size == 4
    # The background (100) and the nuclei (500) give a floor between them.
    assert prepared.dapi_floor is not None and 100.0 < prepared.dapi_floor < 500.0
    no_floor = prepare_bead_mask_veto(
        BeadMaskVetoConfig(tissue_floor=False),
        source=source,
        recorded=_record(enabled=True),
        sdata=sdata,
        shape=shape,
    )
    assert no_floor is not None and no_floor.dapi_floor is None

    # A source built without bead removal (or before the record), or a disabled
    # veto, is a no-op even with a table present.
    for recorded in (_record(enabled=False), None):
        assert bead_mask_veto_identity(config, source, recorded) is None
    assert (
        bead_mask_veto_identity(
            BeadMaskVetoConfig(enabled=False), source, _record(enabled=True)
        )
        is None
    )


def test_v8b_the_identity_holds_the_decision_parameters_only() -> None:
    config = BeadMaskVetoConfig(a0_min=500.0, chunk_mb=8)
    other = BeadMaskVetoConfig(a0_min=500.0, chunk_mb=128)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(veto_module, "bead_table_path", lambda source: Path(__file__))
        patch.setattr(veto_module, "segmentation_identity", lambda recorded: {"x": 1})
        first = bead_mask_veto_identity(config, "s", {})
        assert first == bead_mask_veto_identity(other, "s", {})
        assert first is not None and first["a0_min"] == 500.0
        assert first["tissue_floor"] is True and first["version"] == 2
        assert first != bead_mask_veto_identity(BeadMaskVetoConfig(), "s", {})
        without_floor = BeadMaskVetoConfig(a0_min=500.0, tissue_floor=False)
        assert first != bead_mask_veto_identity(without_floor, "s", {})


def test_v9_the_dapi_reader_reads_the_dapi_channel_of_the_projection() -> None:
    dapi = make_scene()[1]
    read = make_dapi_reader(make_sdata(dapi), (SIZE, SIZE))
    window = read(10, 40, 20, 90)
    assert window.dtype == np.float32 and window.shape == (30, 70)
    assert np.array_equal(window, dapi[10:40, 20:90])
    with pytest.raises(ValueError, match="MERSCOPE_z_projection image"):
        make_dapi_reader(types.SimpleNamespace(images={}), (SIZE, SIZE))
    with pytest.raises(ValueError, match="px but the Cellpose masks"):
        make_dapi_reader(make_sdata(dapi), (SIZE, SIZE + 1))
    no_dapi = make_sdata(dapi)
    image = no_dapi.images[MERSCOPE_ZPROJ_IMAGE_NAME]
    no_dapi.images[MERSCOPE_ZPROJ_IMAGE_NAME] = image.isel(c=[0])
    with pytest.raises(ValueError, match="DAPI"):
        make_dapi_reader(no_dapi, (SIZE, SIZE))


def test_v10_a_mask_whose_ring_cannot_be_estimated_is_kept() -> None:
    mask, dapi = make_scene()
    ys, xs = np.nonzero(mask == 1)

    def read(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return dapi[y0:y1, x0:x1]

    shape = (SIZE, SIZE)
    assert has_dapi_nucleus(ys, xs, read, shape, BeadMaskVetoConfig()) is False
    # With a demand of more ring pixels than exist, nothing is decided.
    strict = BeadMaskVetoConfig(min_ring_px=10**7)
    assert has_dapi_nucleus(ys, xs, read, shape, strict) is None
    # A nucleus is found where there is one.
    ys, xs = np.nonzero(mask == 2)
    assert has_dapi_nucleus(ys, xs, read, shape, BeadMaskVetoConfig()) is True


def test_v11_the_config_defaults_and_validation() -> None:
    config = BeadMaskVetoConfig()
    assert config.enabled is True
    assert (config.a0_min, config.min_distance_px, config.radius_fraction) == (
        400.0,
        8.0,
        0.35,
    )
    assert (config.ring_inner_px, config.ring_outer_px) == (10.0, 25.0)
    assert (config.dapi_percentile, config.dapi_noise_multiple) == (90.0, 3.0)
    assert config.tissue_floor is True
    assert SegmentationConfig.model_fields["bead_mask_veto"].default == config
    with pytest.raises(ValidationError, match="ring_outer_px"):
        BeadMaskVetoConfig(ring_inner_px=30.0, ring_outer_px=25.0)
    with pytest.raises(ValidationError):
        BeadMaskVetoConfig(unknown=1)  # type: ignore[call-arg]


def test_v12_the_bead_qc_folder_name_matches_the_qc_module() -> None:
    assert BEAD_QC_DIRNAME == qc.QC_DIRNAME
    assert bead_table_path("zarr").as_posix() == "zarr/bead_removal_qc/beads.csv.gz"


def test_v13_the_bead_table_of_the_qc_is_read(tmp_path: Path) -> None:
    path = qc_written_table(tmp_path)
    table = read_bead_table(path)
    assert list(table.columns)[:3] == ["y", "x", "A0"]
    assert len(table) == len(BEADS) and table["pruned"].sum() == 1
    broken = tmp_path / "broken.csv"
    broken.write_text("y,x\n1,2\n")
    with pytest.raises(ValueError, match="A0"):
        read_bead_table(broken)


def tissue_scene() -> tuple[np.ndarray, np.ndarray]:
    """One cell (label 1) with its own nucleus on a bead at the centre, crowded by
    brighter neighbouring nuclei that fill most of its DAPI background ring."""
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    mask = np.zeros((SIZE, SIZE), np.uint32)
    cell = _disc(yy, xx, 120, 120, 0, 12)
    mask[cell] = 1
    dapi = BACKGROUND + np.random.default_rng(1).normal(0, 4, (SIZE, SIZE))
    dapi[cell] = 600.0
    angle = np.arctan2(yy - 120, xx - 120)
    neighbours = _disc(yy, xx, 120, 120, 12, 40) & (np.sin(4 * angle) > -0.3)
    dapi[neighbours] = 1500.0
    return mask, dapi.astype(np.float32)


def test_v14_the_tissue_floor_keeps_a_cell_whose_ring_holds_brighter_nuclei(
    tmp_path: Path,
) -> None:
    mask, dapi = tissue_scene()
    ys, xs = np.nonzero(mask == 1)

    def read(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return dapi[y0:y1, x0:x1]

    config, shape = BeadMaskVetoConfig(), (SIZE, SIZE)
    # The ring test alone sees no nucleus: the neighbours outshine the cell.
    assert has_dapi_nucleus(ys, xs, read, shape, config) is False
    assert has_dapi_nucleus(ys, xs, read, shape, config, 300.0) is True
    # A bead "cell" on the glass stays below the floor and is still dropped.
    ring_mask, glass = make_scene()
    ys, xs = np.nonzero(ring_mask == 1)

    def read_glass(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return glass[y0:y1, x0:x1]

    assert has_dapi_nucleus(ys, xs, read_glass, shape, config, 300.0) is False

    # Through the veto: dropped without the floor, kept (and recorded) with it.
    bead = ((120, 120, 1000, 0, 0),)
    for floor, n_vetoed in ((None, 1), (300.0, 0)):
        path = tmp_path / f"tissue_{floor}.npy"
        np.save(path, mask)
        result = apply_bead_mask_veto(
            path, make_veto(config, dapi, beads=bead, dapi_floor=floor)
        )
        assert result.n_vetoed == n_vetoed
        assert result.n_kept_with_nucleus == 1 - n_vetoed
        counts = json.loads(veto_sidecar_path(path).read_text())["counts"]
        assert counts["dapi_floor"] == floor


def _projection(dapi: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        np.stack([np.zeros_like(dapi), dapi]),
        dims=("c", "y", "x"),
        coords={"c": ["PolyT", "DAPI"]},
    )


def test_v15_the_floor_is_the_otsu_level_of_the_coarsest_projection_level() -> None:
    rng = np.random.default_rng(2)
    half = np.full((64, 64), 100.0) + rng.normal(0, 5, (64, 64))
    half[:, 32:] += 900.0
    floor = tissue_dapi_floor(make_sdata(half.astype(np.float32)))
    # It lies above the glass (left half) and below the tissue (right half).
    assert floor is not None
    assert np.median(half[:, :32]) < floor < half[:, 32:].min()

    # Multiscale: the coarsest level decides (here with other values than scale0).
    coarse = (half[::2, ::2] / 10.0).astype(np.float32)
    tree = xr.DataTree.from_dict(
        {
            "scale0": xr.Dataset({"image": _projection(half.astype(np.float32))}),
            "scale1": xr.Dataset({"image": _projection(coarse)}),
        }
    )
    multiscale = types.SimpleNamespace(images={MERSCOPE_ZPROJ_IMAGE_NAME: tree})
    coarse_floor = tissue_dapi_floor(multiscale)
    assert coarse_floor is not None
    assert np.median(coarse[:, :16]) < coarse_floor < coarse[:, 16:].min()

    # No contrast: no floor. No DAPI channel: an error.
    assert tissue_dapi_floor(make_sdata(np.full((8, 8), 7.0, np.float32))) is None
    no_dapi = make_sdata(half.astype(np.float32))
    image = no_dapi.images[MERSCOPE_ZPROJ_IMAGE_NAME]
    no_dapi.images[MERSCOPE_ZPROJ_IMAGE_NAME] = image.isel(c=[0])
    with pytest.raises(ValueError, match="DAPI"):
        tissue_dapi_floor(no_dapi)


def qc_written_table(folder: Path) -> Path:
    """A table written by the QC writer itself (the format the veto reads)."""
    beads = np.array(
        [[y, x, a0, a0 / 2, blob, 1.0, 0.5, pruned] for y, x, a0, blob, pruned in BEADS]
    )
    path = folder / "beads.csv.gz"
    qc.write_beads_table(path, beads, np.zeros(len(beads), int))
    return path


# ---------------------------------------------------------------------------
# The Cellpose stage
POINTS = pd.DataFrame(
    {
        # Pixels of the transcripts: two on the ring mask (8 px from its centre),
        # two on the mask with the nucleus.
        "global_x": [68.0, 52.0, 160.0, 163.0],
        "global_y": [60.0, 60.0, 60.0, 62.0],
        "gene": ["Gad1", "Nkx6-1", "Gad1", "Nkx6-1"],
        "transcript_score": np.full(4, 0.9),
    }
)


class StageFixture:
    """The Cellpose stage with a fake Cellpose that writes the six-mask scene."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.calls = 0
        self.mask, self.dapi = make_scene()
        sdata = make_sdata(self.dapi)

        def fake_cellpose(
            *,
            output_mask_path: Path,
            output_cellprob_path: Path | None = None,
            output_stitching_stats_path: Path | None = None,
            before_overwrite: Any = None,
            **kwargs: object,
        ) -> Path:
            self.calls += 1
            if before_overwrite is not None:
                before_overwrite()
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_mask_path, self.mask)
            if output_cellprob_path is not None:
                np.save(output_cellprob_path, np.ones(self.mask.shape, np.float32))
            if output_stitching_stats_path is not None:
                output_stitching_stats_path.write_text('{"final_labels": 6}\n')
            return output_mask_path

        monkeypatch.setattr(
            "merxen.segmentation.pipeline._load_dataset_sdata",
            lambda config: (sdata, object(), SIZE, SIZE, np.eye(3), POINTS),
        )
        monkeypatch.setattr(
            "merxen.segmentation.pipeline.run_tiled_cellpose", fake_cellpose
        )
        monkeypatch.setattr(
            "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
            lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        )

    def config(
        self, source: Path, **veto: Any
    ) -> tuple[SegmentationConfig, Path, Path]:
        config = _config(self.tmp_path, source)
        if veto:
            config = config.model_copy(
                update={"bead_mask_veto": BeadMaskVetoConfig(**veto)}
            )
        mask_path = Path(str(config.dataset.persistent_mask_path))
        return config, mask_path, mask_path.with_name("transcripts_for_proseg.csv")


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StageFixture:
    return StageFixture(tmp_path, monkeypatch)


def test_s1_the_veto_runs_before_seeding_and_is_recorded_in_the_mask_sidecar(
    stage: StageFixture,
) -> None:
    source = _source_zarr(stage.tmp_path, _record(enabled=True))
    write_bead_table(source)
    config, mask_path, transcripts = stage.config(source)

    run_cellpose_segmentation(config)

    saved = np.load(mask_path)
    assert (saved[stage.mask == 1] == 0).all()  # the ring on the bead is gone
    assert sorted(np.unique(saved)) == [0, 1, 2, 3, 4]
    sidecar = json.loads(segmentation_sidecar_path(mask_path).read_text())
    identity = bead_mask_veto_identity(
        config.bead_mask_veto, source, _record(enabled=True)
    )
    assert sidecar["bead_mask_veto"] == identity
    counts = json.loads(veto_sidecar_path(mask_path).read_text())
    assert counts["parameters"] == identity
    assert counts["counts"]["n_vetoed"] == 2
    # The logits follow the vetoed mask (zero on the background).
    probabilities = np.load(mask_path.with_name("cellpose_cellprobs_tiled.npy"))
    assert (probabilities[saved == 0] == 0).all()
    assert (probabilities[saved > 0] == 1).all()
    # Seeded from the saved masks: the transcripts of the vetoed ring mask are
    # background (label 0), those of the kept mask carry its new label (1).
    seeded = pd.read_csv(transcripts)
    column = next(c for c in seeded.columns if "cell" in c.lower())
    assert seeded[column].tolist() == [0, 0, 1, 1]


def test_s2_the_masks_are_reused_while_the_veto_setting_is_unchanged(
    stage: StageFixture,
) -> None:
    source = _source_zarr(stage.tmp_path, _record(enabled=True))
    write_bead_table(source)
    config, mask_path, _ = stage.config(source)
    run_cellpose_segmentation(config)
    saved = mask_path.read_bytes()

    run_cellpose_segmentation(config)

    assert stage.calls == 1
    assert mask_path.read_bytes() == saved
    # Another veto setting is not the masks that were computed.
    off, _, _ = stage.config(source, enabled=False)
    with pytest.raises(ValueError, match="without the bead-mask veto") as caught:
        run_cellpose_segmentation(off)
    assert "new --outdir" in str(caught.value)
    other, _, _ = stage.config(source, a0_min=500.0)
    with pytest.raises(ValueError, match="bead-mask veto"):
        run_cellpose_segmentation(other)
    assert stage.calls == 1


def test_s3_masks_from_before_the_veto_are_refused_and_not_silently_reused(
    stage: StageFixture,
) -> None:
    source = _source_zarr(stage.tmp_path, _record(enabled=True))
    write_bead_table(source)
    config, mask_path, _ = stage.config(source, enabled=False)
    run_cellpose_segmentation(config)  # as before the veto existed
    sidecar = json.loads(segmentation_sidecar_path(mask_path).read_text())
    assert "bead_mask_veto" not in sidecar
    assert not veto_sidecar_path(mask_path).exists()

    on, _, _ = stage.config(source)
    with pytest.raises(ValueError, match="with the bead-mask veto") as caught:
        run_cellpose_segmentation(on)
    assert "merxen cellpose-segment --force-rerun" in str(caught.value)
    assert stage.calls == 1
    # Recomputing records the veto and drops the stale counts file with the masks.
    run_cellpose_segmentation(on, force_rerun=True)
    assert stage.calls == 2 and veto_sidecar_path(mask_path).is_file()
    run_cellpose_segmentation(on)
    assert stage.calls == 2


@pytest.mark.parametrize("case", ["no_table", "bead_removal_off", "disabled"])
def test_s4_a_no_op_veto_keeps_the_masks_and_the_sidecar_as_they_were(
    stage: StageFixture, case: str
) -> None:
    source = _source_zarr(stage.tmp_path, _record(enabled=case != "bead_removal_off"))
    if case != "no_table":
        write_bead_table(source)
    config, mask_path, _ = stage.config(
        source, **({"enabled": False} if case == "disabled" else {})
    )

    run_cellpose_segmentation(config)

    assert np.array_equal(np.load(mask_path), stage.mask)
    assert not veto_sidecar_path(mask_path).exists()
    sidecar = json.loads(segmentation_sidecar_path(mask_path).read_text())
    assert set(sidecar) == {"schema", "bead_removal"}
    run_cellpose_segmentation(config)
    assert stage.calls == 1
