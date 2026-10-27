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

from ._mr_helpers import a2_store

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


def _prepare(
    root: Path,
    name: str,
    *,
    noise_um: float,
    outlier_um: float,
    **pair_kwargs: Any,
) -> Prepared:
    pair = make_synthetic_pair(root / f"images_{name}", **pair_kwargs)
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name=name,
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=pair.moving_path),
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
    assert attrs[STORE_ATTR]["images"]["reference"]["shape_yx"] == [1536, 1536]
    sdata = sd.read_zarr(outputs["result_store"], selection=("images",))
    matrix = get_transformation(sdata.images["reference"], "global").to_affine_matrix(
        ("x", "y"), ("x", "y")
    )
    assert matrix[0, 2] == pytest.approx(2.0 * 256)
    assert matrix[1, 2] == pytest.approx(2.0 * 256)

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
