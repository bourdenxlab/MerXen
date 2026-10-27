"""Synthetic end to end: prepare, label, register, rerun, compare.

Test-only setting overrides keep each register run short (final strings: ``FAST``
= ``same_section:nonrigid_pixel_size_um=8.0``; the serial-like pair uses
``serial_section:refinement_window_um=400`` and
``same_section:nonrigid_pixel_size_um=12.0``); no guard default is changed.
Pairs are rendered by ``make_synthetic_pair`` (2048 px reference at 2 um, 1.6 um
moving, known similarity plus a smooth warp).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spatialdata as sd
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from spatialdata.transformations import get_transformation

from merxen.image_registration import (
    check_matrix_scale,
    load_alignment_matrix,
    push_points_through_registration,
)
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.io.spatialdata_schema import validate_merxen_schema
from merxen.manual_registration import register as register_module
from merxen.manual_registration.compare import _overview, compare_runs
from merxen.manual_registration.config import (
    ManualRegistrationCompareConfig,
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
)
from merxen.manual_registration.geometry import (
    coarse_from_matrix,
    points_outside_crop,
)
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.register import run_manual_registration
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    RESULT_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LandmarkPair,
    read_root_attrs,
)
from merxen.manual_registration.simulate import (
    SyntheticPair,
    inject_labelling,
    make_synthetic_pair,
    sample_tissue_points,
    simulate_landmark_pairs,
)
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

from ._mr_helpers import LANDMARKS_CSV_HEADER, a2_store

pytestmark = pytest.mark.integration

# Test-only overrides that keep a register run well under a minute on a CI runner:
# the optical flow runs on a coarser grid (8 um instead of the default 3.4 um).
FAST = "same_section:nonrigid_pixel_size_um=8.0"
P_REF = 2.0
P_MOV = 1.6
WORKING_TO_LEVEL0 = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])


@dataclass
class Prepared:
    """A prepared synthetic pair with its labelling."""

    name: str
    root: Path
    pair: SyntheticPair
    store: Path
    pairs: list[LandmarkPair]
    coarse: CoarseTransform

    def run(
        self,
        preset: str,
        *,
        row_settings: str = FAST,
        run_label: str | None = None,
        output: str = "out",
        force: bool = False,
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        run_name = preset if run_label is None else f"{preset}_{run_label}"
        config = ManualRegistrationRunConfig(
            dataset_name=self.name,
            preset=preset,  # type: ignore[arg-type]
            run_label=run_label,
            prepared_store_path=self.store,
            result_dir=self.root / self.name / "registered" / run_name,
            output_dir=self.root / f"register_{output}_{run_name}",
            row_settings=row_settings,
            force=force,
        )
        outputs = run_manual_registration(config)
        summary = json.loads(outputs["summary_json"].read_text())
        return outputs, summary


def _moving_level(pair: SyntheticPair, level: int) -> np.ndarray:
    info = read_ome_tiff_info(pair.moving_path)
    return np.asarray(open_ome_tiff_level(info, level).values[0], dtype=np.float64)


def _untiled_copy(pair: SyntheticPair) -> Path:
    """The moving image as an untiled single-level OME-TIFF.

    Prepare cannot use such a file in place: it writes a tiled pyramidal copy, so
    the prepared moving image has exact (integer-mean) level factors, as every
    CZI-derived image does.
    """
    info = read_ome_tiff_info(pair.moving_path)
    path = pair.moving_path.with_name("moving_untiled.ome.tif")
    tifffile.imwrite(
        path,
        np.asarray(open_ome_tiff_level(info, 0).values),
        ome=True,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": P_MOV,
            "PhysicalSizeY": P_MOV,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": list(info.channel_names)},
        },
    )
    return path


def _prepare(
    root: Path,
    name: str,
    *,
    noise_um: float,
    outlier_um: float,
    untiled_moving: bool = False,
    **pair_kwargs: Any,
) -> Prepared:
    pair = make_synthetic_pair(root / f"images_{name}", **pair_kwargs)
    moving_path = _untiled_copy(pair) if untiled_moving else pair.moving_path
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name=name,
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=moving_path),
            prepared_dir=root / name / "prepared",
            output_dir=root / f"prepare_out_{name}",
            workers=1,
        )
    )
    points = sample_tissue_points(
        _moving_level(pair, 1),
        WORKING_TO_LEVEL0,
        n=20,
        seed=1,
        min_separation_px=150.0 / P_MOV,
    )
    # 16 enabled pairs (one outlier) and 4 held out. The master design asks for
    # 10 + 3; with the 3 um warp acting as landmark noise, 9 used pairs leave about
    # 0.65 px (1 sigma) corner error, too close to the 1.0 px gate; 15 give 0.45.
    pairs = simulate_landmark_pairs(
        points,
        pair.moving_to_reference,
        noise_um=noise_um,
        reference_pixel_size_um=P_REF,
        seed=1,
        outliers={5: (outlier_um, 0.0)},
        disabled=(17, 18, 19, 20),
    )
    fitted = coarse_from_matrix(
        pair.true_matrix,
        moving_shape_yx=pair.moving_shape_yx,
        reference_shape_yx=pair.reference_shape_yx,
        moving_pixel_size_um=P_MOV,
        reference_pixel_size_um=P_REF,
    )[0]
    # A rough manual coarse alignment: 2 degrees and (50, -30) um off.
    coarse = CoarseTransform(
        rotation_deg=fitted.rotation_deg + 2.0,
        mirror_x=fitted.mirror_x,
        offset_um=(fitted.offset_um[0] + 50.0, fitted.offset_um[1] - 30.0),
    )
    inject_labelling(outputs["store"], pairs, coarse=coarse)
    return Prepared(name, root, pair, outputs["store"], pairs, coarse)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("manual_registration_e2e")


@pytest.fixture(scope="module")
def pair_a(root: Path) -> Prepared:
    return _prepare(
        root,
        "SYN1",
        noise_um=0.5,
        outlier_um=30.0,
        rotation_deg=178.0,
        mirror_x=False,
        offset_um=(40.0, -25.0),
        seed=0,
    )


@pytest.fixture(scope="module")
def pair_mirrored(root: Path) -> Prepared:
    return _prepare(
        root,
        "SYN2",
        noise_um=0.5,
        outlier_um=30.0,
        rotation_deg=178.0,
        mirror_x=True,
        offset_um=(40.0, -25.0),
        seed=0,
    )


@pytest.fixture(scope="module")
def pair_serial(root: Path) -> Prepared:
    return _prepare(
        root,
        "SYN3",
        noise_um=3.0,
        outlier_um=90.0,
        rotation_deg=178.0,
        mirror_x=False,
        offset_um=(40.0, -25.0),
        warp_amplitude_um=8.0,
        warp_wavelength_um=800.0,
        seed=5,
        independent_nuclei=True,
    )


@pytest.fixture(scope="module")
def pair_moving_crop(root: Path) -> Prepared:
    return _prepare(
        root,
        "SYN4",
        noise_um=0.5,
        outlier_um=30.0,
        untiled_moving=True,
        rotation_deg=178.0,
        mirror_x=False,
        offset_um=(40.0, -25.0),
        seed=0,
    )


@pytest.fixture(scope="module")
def run_a(pair_a: Prepared) -> tuple[dict[str, Path], dict[str, Any]]:
    return pair_a.run("same_section")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tissue_corners(pair: SyntheticPair) -> np.ndarray:
    """The four corners of the moving tissue bounding box (level-0 xy)."""
    level = _moving_level(pair, 1)
    smoothed = gaussian_filter(np.log1p(level), 2.0)
    rows, cols = np.nonzero(smoothed > threshold_otsu(smoothed))
    x0, x1 = 2.0 * cols.min(), 2.0 * cols.max()
    y0, y1 = 2.0 * rows.min(), 2.0 * rows.max()
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _corner_error(matrix: np.ndarray, pair: SyntheticPair) -> float:
    corners = _tissue_corners(pair)
    return float(
        np.linalg.norm(
            _apply(matrix, corners) - _apply(pair.true_matrix, corners), axis=1
        ).max()
    )


def _landmark_rows(outputs: dict[str, Path]) -> Any:
    import pandas as pd

    return pd.read_csv(outputs["landmarks_csv"])


def _moving_points(rows: Any) -> np.ndarray:
    return rows[["moving_x_px", "moving_y_px"]].to_numpy()


def _stage_error(rows: Any, stage: str, pair: SyntheticPair) -> np.ndarray:
    truth = pair.moving_to_reference(_moving_points(rows))
    position = rows[[f"{stage}_x_px", f"{stage}_y_px"]].to_numpy()
    return np.asarray(np.linalg.norm(position - truth, axis=1))


# --------------------------------------------------------------------------
# 1. same_section, no crop
# --------------------------------------------------------------------------


def test_same_section_no_crop(
    pair_a: Prepared, run_a: tuple[dict[str, Path], dict[str, Any]]
) -> None:
    outputs, summary = run_a
    pair = pair_a.pair
    assert summary["status"] == "complete"
    assert outputs["landmarks_csv"].read_text().splitlines()[0] == LANDMARKS_CSV_HEADER
    rows = _landmark_rows(outputs)
    outlier = rows[rows["id"] == 5].iloc[0]
    assert bool(outlier["outlier"]) and not bool(outlier["used_in_fit"])

    landmark = load_alignment_matrix(outputs["landmark_matrix_csv"])
    final = load_alignment_matrix(outputs["final_matrix_csv"])
    assert _corner_error(landmark, pair) <= 1.0
    assert _corner_error(final, pair) <= 1.0
    for matrix in (landmark, final):
        check_matrix_scale(
            matrix,
            image_pixel_size_um=P_MOV,
            reference_pixel_size_um=P_REF,
            tolerance=0.02,
        )
    assert summary["nonrigid"]["status"] == "applied"

    held_out = rows[~rows["enabled"]]
    assert len(held_out) == 4
    final_error = np.median(_stage_error(held_out, "final", pair))
    landmark_error = np.median(_stage_error(held_out, "landmark", pair))
    assert final_error < landmark_error

    # Dense check through the flow.
    points = sample_tissue_points(
        _moving_level(pair, 1),
        WORKING_TO_LEVEL0,
        n=200,
        seed=7,
        min_separation_px=15.0,
    )
    flow = np.load(summary["outputs"]["core_qc"]["nonrigid_flow_npz"])
    pushed, converged = push_points_through_registration(
        points,
        final_matrix=np.asarray(summary["matrices"]["final_affine_crop"]),
        flow_yx=flow["flow_yx"],
        working_to_reference=flow["working_to_reference"],
    )
    assert converged.all()
    truth = pair.moving_to_reference(points)
    final_dense = np.median(np.linalg.norm(pushed - truth, axis=1))
    landmark_dense = np.median(np.linalg.norm(_apply(landmark, points) - truth, axis=1))
    assert final_dense < landmark_dense


def test_same_section_result_store(
    pair_a: Prepared, run_a: tuple[dict[str, Path], dict[str, Any]]
) -> None:
    outputs, summary = run_a
    store = outputs["result_store"]
    sdata = sd.read_zarr(store)
    assert set(sdata.images) == {
        "reference",
        "moving_registered",
        derived_image_pyramid_cache_key("reference", 4),
        derived_image_pyramid_cache_key("moving_registered", 4),
    }
    attrs = read_root_attrs(store)
    assert attrs[STORE_ATTR]["kind"] == "result"
    assert attrs[RESULT_ATTR]["status"] == "complete"
    assert attrs[RESULT_ATTR] == summary
    assert LABELLING_ATTR not in attrs
    validate_merxen_schema(sdata)
    for key in ("reference", "moving_registered"):
        matrix = get_transformation(sdata.images[key], "global").to_affine_matrix(
            ("x", "y"), ("x", "y")
        )
        np.testing.assert_allclose(matrix, np.diag([P_REF, P_REF, 1.0]), atol=1e-12)
    moving = sdata.images["moving_registered"]
    assert [str(c) for c in moving.coords["c"].values] == ["DAPI", "AF568"]
    assert moving.dtype == np.uint16
    assert tuple(moving.shape[1:]) == pair_a.pair.reference_shape_yx
    assert summary["outputs"]["result_store"] == str(store)
    assert not list(store.parent.glob("*.partial-*"))
    assert not list(store.parent.glob("*.superseded-*"))


# --------------------------------------------------------------------------
# 2. same_section with a crop
# --------------------------------------------------------------------------


def test_same_section_with_a_crop(
    pair_a: Prepared, run_a: tuple[dict[str, Path], dict[str, Any]]
) -> None:
    _, full_summary = run_a
    crop = CropRect(x0=205, y0=205, x1=1843, y1=1843)
    inject_labelling(
        pair_a.store, pair_a.pairs, coarse=pair_a.coarse, crops={"reference": crop}
    )
    outputs, summary = pair_a.run("same_section", run_label="crop")
    assert outputs["result_store"].parent.name == "same_section_crop"
    assert summary["crops"]["reference"]["effective"] == {
        "x0": 256,
        "y0": 256,
        "x1": 1792,
        "y1": 1792,
    }
    assert summary["crops"]["reference"]["requested"] == crop.model_dump()
    assert summary["crops"]["moving"]["requested"] is None
    attrs = read_root_attrs(outputs["result_store"])
    effective = {"x0": 256, "y0": 256, "x1": 1792, "y1": 1792}
    images = attrs[STORE_ATTR]["images"]
    assert images["reference"]["shape_yx"] == [1536, 1536]
    # The result variant of STORE_ATTR (what the viewer reads).
    assert images["reference"]["crop"] == effective
    assert images["reference"]["full_shape_yx"] == list(pair_a.pair.reference_shape_yx)
    assert images["reference"]["pixel_size_um"] == P_REF
    assert images["moving"]["element"] == "moving_registered"
    assert images["moving"]["shape_yx"] == [1536, 1536]
    assert images["moving"]["pixel_size_um"] == P_REF
    assert images["moving"]["source_shape_yx"] == list(pair_a.pair.moving_shape_yx)
    assert images["moving"]["source_pixel_size_um"] == P_MOV
    moving_height, moving_width = pair_a.pair.moving_shape_yx
    assert images["moving"]["source_crop"] == {
        "x0": 0,
        "y0": 0,
        "x1": moving_width,
        "y1": moving_height,
    }
    sdata = sd.read_zarr(outputs["result_store"], selection=("images",))
    for key in ("reference", "moving_registered"):
        matrix = get_transformation(sdata.images[key], "global").to_affine_matrix(
            ("x", "y"), ("x", "y")
        )
        np.testing.assert_allclose(
            matrix, [[P_REF, 0, P_REF * 256], [0, P_REF, P_REF * 256], [0, 0, 1]]
        )
        assert tuple(sdata.images[key].shape[1:]) == (1536, 1536)

    reference_xy = np.array([p.reference_xy_px for p in pair_a.pairs])
    moving_xy = np.array([p.moving_xy_px for p in pair_a.pairs])
    outside = points_outside_crop(reference_xy, crop) | points_outside_crop(
        moving_xy, None
    )
    enabled = np.array([p.enabled for p in pair_a.pairs])
    assert summary["fit"]["n_outside_crop"] == int((outside & enabled).sum())

    first = np.asarray(full_summary["matrices"]["final_affine_full"])
    second = np.asarray(summary["matrices"]["final_affine_full"])
    corners = np.array([[256, 256], [1792, 256], [1792, 1792], [256, 1792]], float)
    moving_corners = _apply(np.linalg.inv(first), corners)
    difference = np.linalg.norm(_apply(second, moving_corners) - corners, axis=1)
    assert difference.max() <= 0.5


# --------------------------------------------------------------------------
# 3 and 4. mirrored pair, rerun skip
# --------------------------------------------------------------------------


def test_mirrored_pair_and_rerun_behaviour(pair_mirrored: Prepared) -> None:
    settings = FAST + ";nonrigid_enabled=false"
    outputs, summary = pair_mirrored.run("same_section", row_settings=settings)
    assert summary["fit"]["reflected"] is True
    landmark = load_alignment_matrix(outputs["landmark_matrix_csv"])
    final = load_alignment_matrix(outputs["final_matrix_csv"])
    assert _corner_error(landmark, pair_mirrored.pair) <= 1.0
    assert _corner_error(final, pair_mirrored.pair) <= 1.0
    assert np.linalg.det(final[:2, :2]) < 0

    # Second call: nothing changed, nothing durable is touched.
    store = outputs["result_store"]
    root_json = store / "zarr.json"
    before = (root_json.read_bytes(), root_json.stat().st_mtime_ns)
    durable = outputs["summary_json"].read_bytes()
    again, reused = pair_mirrored.run(
        "same_section", row_settings=settings, output="again"
    )
    task_summary = json.loads(again["task_summary_json"].read_text())
    assert task_summary["reused"] is True
    assert "reused" not in json.loads(again["summary_json"].read_text())
    assert (root_json.read_bytes(), root_json.stat().st_mtime_ns) == before
    assert outputs["summary_json"].read_bytes() == durable
    assert reused["provenance_digest"] == summary["provenance_digest"]

    # A durable output deleted since: the same inputs are registered again.
    outputs["final_matrix_csv"].unlink()
    restored, restored_summary = pair_mirrored.run(
        "same_section", row_settings=settings, output="restored"
    )
    assert "reused" not in json.loads(restored["task_summary_json"].read_text())
    assert restored["final_matrix_csv"].exists()
    assert restored_summary["provenance_digest"] == summary["provenance_digest"]

    # One landmark moved by 1 px: a different digest, a re-run.
    moved = list(pair_mirrored.pairs)
    x, y = moved[1].reference_xy_px
    moved[1] = moved[1].model_copy(update={"reference_xy_px": (x + 1.0, y)})
    inject_labelling(pair_mirrored.store, moved, coarse=pair_mirrored.coarse)
    third, third_summary = pair_mirrored.run(
        "same_section", row_settings=settings, output="third"
    )
    assert "reused" not in json.loads(third["task_summary_json"].read_text())
    assert third_summary["provenance_digest"] != summary["provenance_digest"]
    assert (
        third_summary["labelling"]["revision"] == summary["labelling"]["revision"] + 1
    )
    siblings = list(store.parent.glob("*.partial-*")) + list(
        store.parent.glob("*.superseded-*")
    )
    assert siblings == []
    assert read_root_attrs(store)[RESULT_ATTR] == third_summary


# --------------------------------------------------------------------------
# 3b. a crop of the moving image, on an image that prepare rewrote
# --------------------------------------------------------------------------

MOVING_CROP = CropRect(x0=300, y0=260, x1=2300, y1=2350)
REFERENCE_CROP = CropRect(x0=205, y0=205, x1=1843, y1=1843)


def test_moving_crop_and_a_rerun_without_the_flow(
    pair_moving_crop: Prepared,
) -> None:
    pair = pair_moving_crop
    prepared = read_root_attrs(pair.store)[STORE_ATTR]["images"]
    # The prepared moving image has exact level factors (every CZI pair does),
    # the reference was used in place.
    assert prepared["moving"]["ome_tiff"]["exact_level_factors"] is True
    assert prepared["reference"]["ome_tiff"]["exact_level_factors"] is False
    inject_labelling(
        pair.store,
        pair.pairs,
        coarse=pair.coarse,
        crops={"reference": REFERENCE_CROP, "moving": MOVING_CROP},
    )
    outputs, summary = pair.run("same_section", run_label="mcrop")

    # Effective crops: origins round up, ends round down to multiples of 64.
    effective_moving = {"x0": 320, "y0": 320, "x1": 2240, "y1": 2304}
    assert summary["crops"]["moving"]["effective"] == effective_moving
    assert summary["crops"]["moving"]["requested"] == MOVING_CROP.model_dump()
    assert summary["crops"]["reference"]["effective"] == {
        "x0": 256,
        "y0": 256,
        "x1": 1792,
        "y1": 1792,
    }
    images = read_root_attrs(outputs["result_store"])[STORE_ATTR]["images"]
    assert images["moving"]["source_crop"] == effective_moving
    assert images["moving"]["source_shape_yx"] == list(pair.pair.moving_shape_yx)
    assert images["moving"]["source_pixel_size_um"] == P_MOV
    assert images["moving"]["shape_yx"] == [1536, 1536]
    assert images["reference"]["crop"] == summary["crops"]["reference"]["effective"]
    reference_xy = np.array([q.reference_xy_px for q in pair.pairs])
    moving_xy = np.array([q.moving_xy_px for q in pair.pairs])
    enabled = np.array([q.enabled for q in pair.pairs])
    outside = points_outside_crop(reference_xy, REFERENCE_CROP) | points_outside_crop(
        moving_xy, MOVING_CROP
    )
    assert summary["fit"]["n_outside_crop"] == int((outside & enabled).sum())

    # The crop matrix has the moving origin added on the moving side.
    full = np.asarray(summary["matrices"]["landmark_full"])
    crop = np.asarray(summary["matrices"]["landmark_crop"])
    np.testing.assert_allclose(crop[:2, :2], full[:2, :2])
    np.testing.assert_allclose(
        crop[:2, 2], full[:2, :2] @ [320, 320] + full[:2, 2] - [256, 256]
    )

    # The full-frame results are right whatever the crops.
    landmark = load_alignment_matrix(outputs["landmark_matrix_csv"])
    final = load_alignment_matrix(outputs["final_matrix_csv"])
    assert _corner_error(landmark, pair.pair) <= 1.0
    assert _corner_error(final, pair.pair) <= 1.0
    rows = _landmark_rows(outputs)
    held_out = rows[~rows["enabled"]]
    final_error = np.median(_stage_error(held_out, "final", pair.pair))
    assert final_error < np.median(_stage_error(held_out, "landmark", pair.pair))
    assert final_error < 1.5
    assert summary["tissue_qc"]["pearson"]["final"] > 0.9

    # Pixels: the resampled moving image lines up with the reference crop.
    sdata = sd.read_zarr(outputs["result_store"], selection=("images",))
    reference = np.asarray(sdata.images["reference"].isel(c=0).values, dtype=float)
    registered = np.asarray(
        sdata.images["moving_registered"].isel(c=0).values, dtype=float
    )
    footprint = registered > 0
    assert footprint.mean() > 0.5
    correlation = np.corrcoef(
        gaussian_filter(reference, 1.0)[footprint],
        gaussian_filter(registered, 1.0)[footprint],
    )[0, 1]
    assert correlation > 0.9

    # The same run id again, with the flow off: the files of the earlier flow are
    # gone and no longer listed (they are in the crop frame of the earlier run).
    core_before = summary["outputs"]["core_qc"]
    assert summary["nonrigid"]["status"] == "applied"
    flow_files = [
        Path(core_before[k]) for k in ("nonrigid_flow_npz", "nonrigid_qc_png")
    ]
    assert all(path.exists() for path in flow_files)
    _, no_flow = pair.run(
        "same_section",
        run_label="mcrop",
        row_settings=FAST + ";nonrigid_enabled=false",
        output="noflow",
    )
    assert no_flow["nonrigid"]["status"] == "disabled"
    assert no_flow["provenance_digest"] != summary["provenance_digest"]
    core_after = no_flow["outputs"]["core_qc"]
    assert "nonrigid_flow_npz" not in core_after
    assert "nonrigid_qc_png" not in core_after
    assert Path(core_after["registration_qc_png"]).exists()
    assert not any(path.exists() for path in flow_files)
    assert not flow_files[1].with_suffix(".pdf").exists()


# --------------------------------------------------------------------------
# 4b. interrupted runs never leave a stale result
# --------------------------------------------------------------------------


def test_interrupted_runs_never_leave_a_stale_result(
    pair_mirrored: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FAST + ";nonrigid_enabled=false"
    outputs, baseline = pair_mirrored.run(
        "same_section", row_settings=settings, output="base"
    )
    store = outputs["result_store"]
    result_dir = store.parent
    kept = {
        key: outputs[key].read_bytes()
        for key in (
            "summary_json",
            "landmark_matrix_csv",
            "final_matrix_csv",
            "landmarks_csv",
        )
    }
    moved = list(pair_mirrored.pairs)
    x, y = moved[2].reference_xy_px
    moved[2] = moved[2].model_copy(update={"reference_xy_px": (x, y + 1.0)})
    inject_labelling(pair_mirrored.store, moved, coarse=pair_mirrored.coarse)

    # 1. The core fails: the previous store and durable files are untouched, the
    # unfinished temporary store stays for the next run to remove.
    def fail(**_: Any) -> None:
        raise RuntimeError("simulated core failure")

    with monkeypatch.context() as patch:
        patch.setattr(register_module, "register_onto_reference", fail)
        with pytest.raises(RuntimeError, match="simulated core failure"):
            pair_mirrored.run("same_section", row_settings=settings, output="fail")
    stored = read_root_attrs(store)[RESULT_ATTR]
    assert stored["provenance_digest"] == baseline["provenance_digest"]
    assert all(outputs[key].read_bytes() == content for key, content in kept.items())
    assert list(result_dir.glob(".*.zarr.partial-*"))

    # 2. The publish after the swap fails once (disk full, killed task): the new
    # store is in place, the durable files are still the previous run's.
    real_publish = register_module._publish
    calls: list[Path] = []

    def flaky_publish(source: Path, target: Path) -> Path:
        calls.append(target)
        if len(calls) == 1:
            raise OSError("simulated disk full")
        return real_publish(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(register_module, "_publish", flaky_publish)
        with pytest.raises(OSError, match="simulated disk full"):
            pair_mirrored.run("same_section", row_settings=settings, output="flaky")
    new_digest = read_root_attrs(store)[RESULT_ATTR]["provenance_digest"]
    assert new_digest != baseline["provenance_digest"]
    assert not list(result_dir.glob(".*partial-*"))  # the failed run's temp is gone
    assert (
        json.loads(outputs["summary_json"].read_text())["provenance_digest"]
        == baseline["provenance_digest"]
    )
    # A run that died between the two renames of a swap leaves a full store.
    leftover = store.with_name(store.name + ".superseded-20200101T000000Z")
    (leftover / "images").mkdir(parents=True)

    # 3. The retry does not reuse the store: it publishes the files of its run.
    retried, summary = pair_mirrored.run(
        "same_section", row_settings=settings, output="retry"
    )
    assert "reused" not in json.loads(retried["task_summary_json"].read_text())
    assert summary["provenance_digest"] == new_digest
    np.testing.assert_allclose(
        load_alignment_matrix(retried["final_matrix_csv"]),
        np.asarray(summary["matrices"]["final_affine_full"]),
    )
    assert read_root_attrs(store)[RESULT_ATTR] == summary
    assert not leftover.exists()
    assert not list(result_dir.glob("*.partial-*"))
    assert not list(result_dir.glob("*.superseded-*"))

    # 4. Everything is published now, so the next call is a no-op.
    done, again = pair_mirrored.run(
        "same_section", row_settings=settings, output="done"
    )
    assert again["provenance_digest"] == new_digest
    task = json.loads(done["task_summary_json"].read_text())
    assert task["reused"] is True

    # 5. A forced re-run that fails in the core has deleted the current result's
    # core QC files (the store and the durable files stay): the next plain run
    # does not report the result as reused but registers again and restores them.
    core_qc = {k: Path(v) for k, v in again["outputs"]["core_qc"].items()}
    assert core_qc and all(path.exists() for path in core_qc.values())
    with monkeypatch.context() as patch:
        patch.setattr(register_module, "register_onto_reference", fail)
        with pytest.raises(RuntimeError, match="simulated core failure"):
            pair_mirrored.run(
                "same_section", row_settings=settings, output="fail2", force=True
            )
    assert read_root_attrs(store)[RESULT_ATTR]["provenance_digest"] == new_digest
    assert not any(path.exists() for path in core_qc.values())
    healed, healed_summary = pair_mirrored.run(
        "same_section", row_settings=settings, output="healed"
    )
    assert "reused" not in json.loads(healed["task_summary_json"].read_text())
    assert healed_summary["provenance_digest"] == new_digest
    healed_qc = {k: Path(v) for k, v in healed_summary["outputs"]["core_qc"].items()}
    assert healed_qc == core_qc
    assert all(path.exists() for path in healed_qc.values())
    # Complete again, so the following call is a no-op once more.
    last, _ = pair_mirrored.run("same_section", row_settings=settings, output="last")
    assert json.loads(last["task_summary_json"].read_text())["reused"] is True

    # 6. A run with other inputs (not forced) that fails after the core rewrites
    # the listed core QC files in place and leaves the store and the durable files
    # of the current result: the next plain run of the current inputs must not
    # report it as reused with the failed run's files.
    def fail_figure(*_: Any, **__: Any) -> None:
        raise OSError("simulated disk full")

    with monkeypatch.context() as patch:
        patch.setattr(register_module, "write_tissue_qc_figure", fail_figure)
        with pytest.raises(OSError, match="simulated disk full"):
            pair_mirrored.run(
                "same_section",
                row_settings=settings + ";tissue_qc_sigma_um=25.0",
                output="fail3",
            )
    assert read_root_attrs(store)[RESULT_ATTR]["provenance_digest"] == new_digest
    summary_mtime = healed["summary_json"].stat().st_mtime_ns
    assert all(path.stat().st_mtime_ns > summary_mtime for path in healed_qc.values())
    recovered, recovered_summary = pair_mirrored.run(
        "same_section", row_settings=settings, output="recovered"
    )
    assert "reused" not in json.loads(recovered["task_summary_json"].read_text())
    assert recovered_summary["provenance_digest"] == new_digest
    assert {
        k: Path(v) for k, v in recovered_summary["outputs"]["core_qc"].items()
    } == healed_qc
    summary_mtime = recovered["summary_json"].stat().st_mtime_ns
    assert all(path.stat().st_mtime_ns <= summary_mtime for path in healed_qc.values())
    final, _ = pair_mirrored.run("same_section", row_settings=settings, output="final")
    assert json.loads(final["task_summary_json"].read_text())["reused"] is True


# --------------------------------------------------------------------------
# 5. serial-like pair, both presets, compare
# --------------------------------------------------------------------------


def test_serial_pair_with_both_presets_and_compare(
    pair_serial: Prepared, root: Path
) -> None:
    # 400 um windows: with 600 um only 9 usable windows exist on this 4 mm image,
    # fewer than the preset's min_refinement_windows (12), and the refinement is
    # skipped. The flow of the same_section run uses a coarser grid to stay short.
    settings = (
        "serial_section:refinement_window_um=400;"
        "same_section:nonrigid_pixel_size_um=12.0"
    )
    serial_out, serial = pair_serial.run("serial_section", row_settings=settings)
    same_out, same = pair_serial.run("same_section", row_settings=settings)
    assert serial_out["result_store"].name == "SYN3_serial_section_registered.zarr"
    assert serial_out["result_store"].parent.name == "serial_section"
    assert same_out["result_store"].name == "SYN3_same_section_registered.zarr"
    assert same_out["result_store"].parent.name == "same_section"
    assert serial_out["landmarks_csv"].name != same_out["landmarks_csv"].name
    assert serial["status"] == same["status"] == "complete"
    pearson = serial["tissue_qc"]["pearson"]
    assert pearson["final"] >= pearson["landmark"] - 0.01
    assert serial["nucleus_qc"]["status"] == "measured"
    assert serial["refinement"]["status"] == "applied"
    assert same["refinement"]["status"]  # reported, not gated
    assert same["nonrigid"]["status"]

    overview = _overview(serial)
    assert overview is not None and overview.ndim == 3 and overview.shape[2] == 3
    comparison = compare_runs(
        ManualRegistrationCompareConfig(
            dataset_name="SYN3",
            summary_paths=[serial_out["summary_json"], same_out["summary_json"]],
            output_dir=root / "SYN3" / "registered",
            task_output_dir=root / "compare_out",
        )
    )
    lines = comparison["comparison_csv"].read_text().splitlines()
    assert len(lines) == 3
    assert comparison["comparison_png"].stat().st_size > 0


# --------------------------------------------------------------------------
# 6. a draft labelling is refused before anything is written
# --------------------------------------------------------------------------


def test_draft_labelling_is_refused_before_any_write(tmp_path: Path) -> None:
    store = a2_store(tmp_path, status="draft")
    result_dir = tmp_path / "registered" / "same_section"
    config = ManualRegistrationRunConfig(
        dataset_name="SYN1",
        preset="same_section",
        prepared_store_path=store,
        result_dir=result_dir,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="Finish labelling"):
        run_manual_registration(config)
    assert not result_dir.exists()
    assert not list(tmp_path.rglob("*.partial-*"))
    shutil.rmtree(tmp_path, ignore_errors=True)
