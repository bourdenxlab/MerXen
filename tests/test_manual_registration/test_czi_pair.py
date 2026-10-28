"""Test landmark derivation and the Zeiss CZI pair validation (real data, slow).

The derivation finds landmark pairs by SIFT (same section) or, for serial sections,
by tissue-scale window matching.

``test_czi_pair_both_presets`` runs the two manual-registration workflows through
Nextflow on a real Zeiss CZI pair (no ground truth): prepare, landmarks derived by
``scripts/manual_registration/derive_test_landmarks.py``, register with both
presets, compare. The gates of the ``serial_section`` run are plausibility checks
of the fit and the QC; ``same_section`` must complete, its numbers are recorded.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import spatialdata as sd
from scipy import ndimage

from merxen.io.spatialdata_schema import validate_merxen_schema
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.geometry import (
    points_outside_crop,
    rotation_matrix_xy,
    similarity_parameters,
    translation_xy,
    wrap_deg,
)
from merxen.manual_registration.prepare import run_manual_registration_prepare
from merxen.manual_registration.presets import PRESETS
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    read_root_attrs,
)
from merxen.manual_registration.simulate import SyntheticPair, make_synthetic_pair

from ._mr_helpers import A2_COARSE, a2_pairs, a2_store
from ._scripts import load_script

derive_script = load_script("derive_test_landmarks")
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_farthest_point_order_spreads_points() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [5.0, 0.0], [0.0, 9.0]])

    order = derive_script.farthest_point_order(points)

    # Nearest the centroid (3), the farthest from it (4), then the tie between 0 and
    # 2 (both 5 from the chosen set) goes to the lowest index.
    assert order.tolist() == [3, 4, 0, 2, 1]
    assert derive_script.farthest_point_order(np.zeros((0, 2))).size == 0


def test_tissue_mask_finds_the_bright_disc() -> None:
    yy, xx = np.mgrid[:200, :200]
    image = np.where((yy - 100) ** 2 + (xx - 100) ** 2 < 60**2, 500.0, 5.0)
    image = image + np.random.default_rng(0).normal(0, 1, image.shape)

    mask = derive_script.tissue_mask(np.clip(image, 0, None))

    assert 0.25 < mask.mean() < 0.35
    assert mask[100, 100] and not mask[5, 5]


def test_scripts_refuse_an_output_inside_the_repository() -> None:
    inside = REPO_ROOT / "derived_landmarks.json"
    with pytest.raises(SystemExit, match="repository"):
        derive_script.main(["--store", "unused.zarr", "--out", str(inside)])
    with pytest.raises(SystemExit, match="repository"):
        load_script("export_reference_image").main(
            ["--store", "unused.zarr", "--out", str(REPO_ROOT / "export.ome.tif")]
        )


def _prepared_synthetic_pair(
    tmp_path: Path, *, independent_nuclei: bool = False
) -> tuple[SyntheticPair, Path]:
    """Prepare a synthetic pair (178 deg, mirrored, no warp) into a store."""
    pair = make_synthetic_pair(
        tmp_path,
        reference_shape_yx=(2048, 2048),
        rotation_deg=178.0,
        mirror_x=True,
        warp_amplitude_um=0.0,
        independent_nuclei=independent_nuclei,
    )
    outputs = run_manual_registration_prepare(
        ManualRegistrationPrepareConfig(
            dataset_name="SYN1",
            reference=ManualRegistrationImageInput(path=pair.reference_path),
            moving=ManualRegistrationImageInput(path=pair.moving_path),
            prepared_dir=tmp_path / "prepared",
            output_dir=tmp_path / "prepare_out",
            workers=1,
        )
    )
    return pair, outputs["store"]


@pytest.mark.integration
def test_derive_test_landmarks_recovers_a_rotated_mirrored_pair(
    tmp_path: Path,
) -> None:
    pair, store = _prepared_synthetic_pair(tmp_path)

    derived = derive_script.derive(
        store,
        n_pairs=12,
        n_disabled=4,
        working_pixel_size_um=8.0,
        min_inliers=12,
    )

    # The pair shares its nuclei, so SIFT succeeds and the fallback never runs.
    assert derived.diagnostics["method"] == "sift"
    assert derived.coarse.mirror_x is True
    assert abs(wrap_deg(derived.coarse.rotation_deg - 178.0)) <= 10.0
    height, width = pair.moving_shape_yx
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
    )

    def apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    error = np.linalg.norm(
        apply(derived.matrix_full, corners) - apply(pair.true_matrix, corners), axis=1
    ).max()
    assert error <= 4.0, error
    assert len(derived.pairs) == 12
    assert [p.id for p in derived.pairs if not p.enabled] == [3, 6, 9, 12]
    reference = np.array([p.reference_xy_px for p in derived.pairs])
    moving = np.array([p.moving_xy_px for p in derived.pairs])
    assert not points_outside_crop(reference, derived.crops["reference"]).any()
    assert not points_outside_crop(moving, derived.crops["moving"]).any()


# --------------------------------------------------------------------------
# Tissue-scale window matching (the serial-section method)
# --------------------------------------------------------------------------

WINDOW_PIXEL_SIZE_UM = 20.0
PASS_KEYS = {
    "search_um",
    "n_windows",
    "n_confident",
    "n_inliers",
    "scale",
    "rotation_deg",
    "inlier_residual_median_um",
    "inlier_residual_p90_um",
}


def _apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=float) @ matrix[:2, :2].T + matrix[:2, 2]


def _tissue_field(shape_yx: tuple[int, int], seed: int) -> np.ndarray:
    """A smooth tissue-scale pattern of unit variance (structure of 40 to 120 um)."""
    rng = np.random.default_rng(seed)
    coarse = ndimage.gaussian_filter(
        rng.normal(size=shape_yx), 120.0 / WINDOW_PIXEL_SIZE_UM
    )
    fine = ndimage.gaussian_filter(
        rng.normal(size=shape_yx), 40.0 / WINDOW_PIXEL_SIZE_UM
    )
    field = coarse / coarse.std() + 0.5 * fine / fine.std()
    return np.asarray(field / field.std(), dtype=np.float32)


def _intensities(field: np.ndarray, seed: int) -> np.ndarray:
    """Raw intensities: the shared structure plus a texture of the image's own."""
    texture = ndimage.gaussian_filter(
        np.random.default_rng(seed).normal(size=field.shape), 1.0
    )
    return np.asarray(np.exp(6.0 + 0.55 * field + 0.5 * texture / texture.std()))


def _serial_like_pair(
    *, rotation_deg: float, mirror: bool, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference and moving working images of one pattern with different textures.

    Returns:
        ``(reference, moving, truth, coarse)``: the images, the true matrix from
        moving to reference px and a coarse matrix off by 3 deg and about 7 px.
    """
    ref_side, mov_side, canvas_side = 480, 520, 1400
    canvas_offset = (canvas_side - ref_side) // 2
    field = _tissue_field((canvas_side, canvas_side), seed=11)
    flip = np.diag([-1.0 if mirror else 1.0, 1.0, 1.0])
    truth = (
        translation_xy(
            (ref_side - 1) / 2.0 + 37.0,
            (ref_side - 1) / 2.0 - 23.0,
        )
        @ rotation_matrix_xy(rotation_deg)
        @ flip
        @ np.diag([scale, scale, 1.0])
        @ translation_xy(-(mov_side - 1) / 2.0, -(mov_side - 1) / 2.0)
    )
    # moving(m) = canvas(truth(m) + offset): cv2 maps source to destination points.
    to_canvas = translation_xy(canvas_offset, canvas_offset) @ truth
    moving_field = cv2.warpAffine(
        field,
        np.linalg.inv(to_canvas)[:2],
        (mov_side, mov_side),
        flags=cv2.INTER_LINEAR,
    )
    reference_field = field[
        canvas_offset : canvas_offset + ref_side,
        canvas_offset : canvas_offset + ref_side,
    ]
    centre = translation_xy((ref_side - 1) / 2.0, (ref_side - 1) / 2.0)
    coarse = (
        translation_xy(7.0, -5.0)
        @ centre
        @ rotation_matrix_xy(-3.0)
        @ np.linalg.inv(centre)
        @ truth
    )
    return (
        _intensities(reference_field, seed=5),
        _intensities(moving_field, seed=6),
        truth,
        coarse,
    )


@pytest.mark.parametrize(
    ("rotation_deg", "mirror", "scale"), [(30.0, True, 0.9), (-20.0, False, 1.08)]
)
def test_window_ncc_recovers_a_known_similarity(
    rotation_deg: float, mirror: bool, scale: float
) -> None:
    reference, moving, truth, coarse = _serial_like_pair(
        rotation_deg=rotation_deg, mirror=mirror, scale=scale
    )

    result = derive_script.window_ncc_similarity(
        reference,
        moving,
        np.ones(reference.shape, dtype=bool),
        np.ones(moving.shape, dtype=bool),
        coarse_matrix=coarse,
        mirror=mirror,
        ref_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
        mov_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
    )

    matrix = result["matrix_working"]
    true_scale, true_rotation, _ = similarity_parameters(truth)
    fitted_scale, fitted_rotation, reflected = similarity_parameters(matrix)
    assert reflected is mirror
    assert abs(wrap_deg(fitted_rotation - true_rotation)) <= 0.5
    assert abs(fitted_scale / true_scale - 1.0) <= 0.005
    height, width = moving.shape
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
    )
    corner_error = np.linalg.norm(
        _apply(matrix, corners) - _apply(truth, corners), axis=1
    )
    assert corner_error.max() <= 2.0, corner_error
    assert result["n_inliers"] >= 24
    # The inlier correspondences are the last pass's, in the unflipped moving frame.
    assert (
        result["moving_xy"].shape
        == result["reference_xy"].shape
        == (
            result["n_inliers"],
            2,
        )
    )
    threshold_px = 60.0 / WINDOW_PIXEL_SIZE_UM
    residual = np.linalg.norm(
        _apply(matrix, result["moving_xy"]) - result["reference_xy"], axis=1
    )
    assert residual.max() <= threshold_px
    # The counts SIFT's callers read are the last pass's windows.
    first, second = result["passes"]
    assert [first["search_um"], second["search_um"]] == [600.0, 200.0]
    assert all(set(record) == PASS_KEYS for record in result["passes"])
    assert result["n_keypoints_reference"] == result["n_keypoints_moving"]
    assert result["n_keypoints_reference"] == second["n_windows"]
    assert result["n_ratio_matches"] == result["n_gated"] == second["n_confident"]
    assert second["n_inliers"] == result["n_inliers"]
    assert second["inlier_residual_median_um"] <= second["inlier_residual_p90_um"]
    assert json.dumps(result["passes"])  # plain numbers: the diagnostics are JSON


def test_window_ncc_refuses_a_mirror_state_that_disagrees_with_the_matrix() -> None:
    image = np.ones((40, 40))
    mask = np.ones((40, 40), dtype=bool)

    with pytest.raises(ValueError, match="mirror"):
        derive_script.window_ncc_similarity(
            image,
            image,
            mask,
            mask,
            coarse_matrix=np.eye(3),
            mirror=True,
            ref_pixel_size_um=10.0,
            mov_pixel_size_um=10.0,
        )


def test_window_ncc_with_fewer_than_three_confident_windows_has_no_fit() -> None:
    reference, moving, _, coarse = _serial_like_pair(
        rotation_deg=30.0, mirror=True, scale=0.9
    )
    # Tissue only in a 60 px (1.2 mm) block: one 1 mm window fits in it.
    reference_mask = np.zeros(reference.shape, dtype=bool)
    reference_mask[235:295, 235:295] = True

    result = derive_script.window_ncc_similarity(
        reference,
        moving,
        reference_mask,
        np.ones(moving.shape, dtype=bool),
        coarse_matrix=coarse,
        mirror=True,
        ref_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
        mov_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
    )

    assert result["matrix_working"] is None and result["n_inliers"] == 0
    (only,) = result["passes"]  # a pass without a fit ends the passes
    assert 1 <= only["n_windows"] <= 2 and only["n_confident"] < 3
    assert only["scale"] is None and only["inlier_residual_median_um"] is None
    assert result["n_keypoints_reference"] == only["n_windows"]
    assert result["moving_xy"].shape == (0, 2)


def test_window_ncc_keeps_the_last_pass_that_fitted() -> None:
    reference, moving, truth, coarse = _serial_like_pair(
        rotation_deg=30.0, mirror=True, scale=0.9
    )

    # A search of 0 um leaves no room for a peak off the border: the second pass
    # finds nothing and the result is the first pass's.
    result = derive_script.window_ncc_similarity(
        reference,
        moving,
        np.ones(reference.shape, dtype=bool),
        np.ones(moving.shape, dtype=bool),
        coarse_matrix=coarse,
        mirror=True,
        ref_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
        mov_pixel_size_um=WINDOW_PIXEL_SIZE_UM,
        search_um=(600.0, 0.0),
    )

    first, second = result["passes"]
    assert first["n_inliers"] >= 24 and second["n_confident"] == 0
    assert second["scale"] is None and second["n_inliers"] == 0
    assert result["n_inliers"] == first["n_inliers"]
    assert result["n_keypoints_reference"] == first["n_windows"]
    assert result["n_ratio_matches"] == first["n_confident"]
    assert similarity_parameters(result["matrix_working"])[2] is True


def _blob(shape: int, centre_xy: tuple[float, float], sigma: float) -> np.ndarray:
    rows, columns = np.mgrid[:shape, :shape]
    return np.asarray(
        np.exp(
            -((columns - centre_xy[0]) ** 2 + (rows - centre_xy[1]) ** 2)
            / (2 * sigma**2)
        ),
        dtype=np.float32,
    )


def test_a_peak_on_the_search_border_is_not_confident() -> None:
    search = 20
    window = _blob(60, (30.0, 30.0), 12.0)

    # The window's content sits 10 px to the right of the region's centre: inside
    # the search, a sub-pixel peak away from the border.
    inside = _blob(100, (50.0 + 10.0, 50.0), 12.0)
    peak, second, x, y, on_border = derive_script._match_window(
        inside, window, exclusion_px=8
    )
    assert not on_border and peak > 0.9 and second < peak
    assert x - search == pytest.approx(10.0, abs=0.1)
    assert y - search == pytest.approx(0.0, abs=0.1)

    # 35 px to the right is beyond the +-20 px search: the best correlation within
    # the search is a high one on its border, which says nothing about the match.
    beyond = _blob(100, (50.0 + 35.0, 50.0), 12.0)
    peak, _, x, _, on_border = derive_script._match_window(
        beyond, window, exclusion_px=8
    )
    assert on_border and peak > 0.3  # it would pass the peak threshold alone
    assert x == 2 * search


# --- derive with the window method ---------------------------------------------


@pytest.fixture(scope="module")
def serial_like_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path]:
    """A prepared synthetic pair whose nuclei differ (shared tissue structure only)."""
    return _prepared_synthetic_pair(
        tmp_path_factory.mktemp("serial_like"), independent_nuclei=True
    )


def _small_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window matching with windows that fit the 4 mm synthetic tissue."""
    real = derive_script.window_ncc_similarity

    def small(*args: Any, **kwargs: Any) -> Any:
        return real(
            *args,
            **{
                **kwargs,
                "window_um": 400.0,
                "stride_um": 250.0,
                "search_um": (240.0, 80.0),
            },
        )

    monkeypatch.setattr(derive_script, "window_ncc_similarity", small)


def _no_sift(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "matrix_working": None,
        "n_inliers": 0,
        "moving_xy": np.zeros((0, 2)),
        "reference_xy": np.zeros((0, 2)),
        "n_keypoints_reference": 0,
        "n_keypoints_moving": 0,
        "n_ratio_matches": 0,
        "n_gated": 0,
    }


def _corner_error(matrix: np.ndarray, pair: SyntheticPair) -> float:
    height, width = pair.moving_shape_yx
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], float
    )
    return float(
        np.linalg.norm(
            _apply(matrix, corners) - _apply(pair.true_matrix, corners), axis=1
        ).max()
    )


@pytest.mark.integration
def test_derive_window_ncc_recovers_a_serial_like_pair(
    serial_like_store: tuple[SyntheticPair, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pair, store = serial_like_store
    _small_windows(monkeypatch)

    derived = derive_script.derive(
        store,
        n_pairs=12,
        n_disabled=4,
        window_pixel_size_um=8.0,
        min_inliers=12,
        method="window_ncc",
    )

    diagnostics = derived.diagnostics
    assert diagnostics["method"] == "window_ncc"
    assert diagnostics["sift"] is None  # SIFT did not run
    window = diagnostics["window_ncc"]
    assert [record["search_um"] for record in window["passes"]] == [240.0, 80.0]
    assert window["working_pixel_size_um"]["reference"] == pytest.approx(8.0, rel=0.05)
    assert window["passes"][-1]["n_inliers"] >= 12
    assert derived.coarse.mirror_x is True
    assert abs(wrap_deg(derived.coarse.rotation_deg - 178.0)) <= 2.0
    assert _corner_error(derived.matrix_full, pair) <= 4.0
    assert len(derived.pairs) == 12
    assert [p.id for p in derived.pairs if not p.enabled] == [3, 6, 9, 12]
    reference = np.array([p.reference_xy_px for p in derived.pairs])
    moving = np.array([p.moving_xy_px for p in derived.pairs])
    assert not points_outside_crop(reference, derived.crops["reference"]).any()
    assert not points_outside_crop(moving, derived.crops["moving"]).any()
    json.dumps(derived.diagnostics)  # the validation JSON records it


@pytest.mark.integration
def test_derive_auto_falls_back_to_window_matching(
    serial_like_store: tuple[SyntheticPair, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pair, store = serial_like_store
    _small_windows(monkeypatch)
    monkeypatch.setattr(derive_script, "sift_similarity", _no_sift)

    with caplog.at_level("INFO"):
        derived = derive_script.derive(
            store,
            n_pairs=12,
            n_disabled=4,
            working_pixel_size_um=8.0,
            window_pixel_size_um=8.0,
            min_inliers=12,
            method="auto",
        )

    diagnostics = derived.diagnostics
    assert diagnostics["method"] == "window_ncc"
    assert diagnostics["sift"]["n_inliers"] == 0  # the SIFT attempt, with its retry
    assert diagnostics["sift"]["retried_at_um"] is not None
    assert "falling back to window matching" in caplog.text
    assert diagnostics["window_ncc"]["passes"][-1]["n_inliers"] >= 12
    assert derived.coarse.mirror_x is True
    assert _corner_error(derived.matrix_full, pair) <= 4.0
    assert len(derived.pairs) == 12


@pytest.mark.integration
def test_derive_auto_names_both_methods_when_both_fail(
    serial_like_store: tuple[SyntheticPair, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = serial_like_store
    monkeypatch.setattr(derive_script, "sift_similarity", _no_sift)

    # The default windows (1 mm, 600 um search) hardly fit the 4 mm synthetic tissue:
    # far fewer than the 24 inliers that are required.
    with pytest.raises(
        RuntimeError,
        match=(
            r"only 0 RANSAC inliers from SIFT \(keypoints .*\) and \d+ from window "
            r"matching \(\d+ windows, \d+ confident\)"
        ),
    ):
        derive_script.derive(
            store, working_pixel_size_um=8.0, window_pixel_size_um=8.0, method="auto"
        )
    with pytest.raises(RuntimeError, match=r"from window matching \(\d+ windows"):
        derive_script.derive(store, window_pixel_size_um=8.0, method="window_ncc")
    with pytest.raises(RuntimeError, match=r"only 0 RANSAC inliers \(keypoints"):
        derive_script.derive(store, working_pixel_size_um=8.0, method="sift")


def test_derive_refuses_an_unknown_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="method"):
        derive_script.derive(tmp_path / "store.zarr", method="orb")


def test_main_passes_the_method_options_to_derive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_derive(*_args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return _derived()

    monkeypatch.setattr(derive_script, "derive", fake_derive)
    monkeypatch.setattr(
        derive_script, "_output_document", lambda *_a, **_k: {"schema": "x"}
    )
    out = str(tmp_path / "landmarks.json")

    derive_script.main(["--store", "unused.zarr", "--out", out])
    derive_script.main(
        [
            "--store",
            "unused.zarr",
            "--out",
            out,
            "--method",
            "window_ncc",
            "--window-pixel-size-um",
            "12.5",
        ]
    )

    assert (seen[0]["method"], seen[0]["window_pixel_size_um"]) == ("auto", 11.0)
    assert (seen[1]["method"], seen[1]["window_pixel_size_um"]) == ("window_ncc", 12.5)
    with pytest.raises(SystemExit) as parsed:
        derive_script.main(["--store", "s.zarr", "--out", out, "--method", "orb"])
    assert parsed.value.code == 2


# --------------------------------------------------------------------------
# --inject never replaces a labelling that was not written by simulate
# --------------------------------------------------------------------------


def _derived(*, disabled: tuple[int, ...] = (3, 6)) -> Any:
    return derive_script.DerivedLandmarks(
        matrix_full=np.eye(3),
        coarse=A2_COARSE,
        scale_ratio=1.0,
        crops={"reference": None, "moving": None},
        pairs=a2_pairs(disabled=disabled),
        diagnostics={},
    )


def _labelling(store: Path) -> dict[str, Any]:
    return read_root_attrs(store)[LABELLING_ATTR]  # type: ignore[no-any-return]


def _backups(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.previous_labelling.*.json"))


@pytest.mark.parametrize("status", ["finished", "draft"])
def test_inject_refuses_a_labelling_written_elsewhere(
    tmp_path: Path, status: str
) -> None:
    # a2_store writes a labelling with another written_by.software (a viewer save).
    store = a2_store(tmp_path, disabled=(1,), status=status)
    before = _labelling(store)
    out = tmp_path / "landmarks.json"

    with pytest.raises(derive_script.ForeignLabellingError, match="--force"):
        derive_script._inject(store, _derived(), out)

    assert _labelling(store) == before
    assert _backups(tmp_path) == []


def test_inject_force_saves_the_old_labelling_before_replacing_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "runs" / "landmarks.json"

    with caplog.at_level("WARNING"):
        revision = derive_script._inject(store, _derived(), out, force=True)

    (backup,) = _backups(out.parent)
    assert backup.parent == out.parent
    assert backup.name.startswith("landmarks.previous_labelling.20")
    assert backup.name.endswith("Z.json")
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    assert str(backup) in caplog.text
    after = _labelling(store)
    assert revision == before["revision"] + 1 == after["revision"]
    assert after["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    assert after["status"] == "finished"
    assert [p["id"] for p in after["landmarks"]["pairs"] if not p["enabled"]] == [3, 6]


def test_inject_replaces_its_own_labelling_without_force_or_backup(
    tmp_path: Path,
) -> None:
    store = a2_store(tmp_path)
    out = tmp_path / "landmarks.json"
    derive_script._inject(store, _derived(), out, force=True)  # now simulate's
    (first_backup,) = _backups(tmp_path)
    first_backup.unlink()

    assert derive_script._inject(store, _derived(disabled=(2,)), out) == 3
    assert (
        _labelling(store)["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    )
    assert _backups(tmp_path) == []


def test_inject_leaves_an_identical_finished_labelling_alone(tmp_path: Path) -> None:
    store = a2_store(tmp_path, disabled=(3, 6), coarse=A2_COARSE)
    before = _labelling(store)

    # The content equals the derived one, so nothing is written even though the
    # labelling is not simulate's and --force was not given.
    assert derive_script._inject(store, _derived(), tmp_path / "landmarks.json") is None

    assert _labelling(store) == before
    assert _backups(tmp_path) == []


def test_main_refuses_a_foreign_labelling_before_deriving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "landmarks.json"

    def no_derivation(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the derivation must not start")

    monkeypatch.setattr(derive_script, "derive", no_derivation)
    with pytest.raises(SystemExit, match="--force"):
        derive_script.main(["--store", str(store), "--out", str(out), "--inject"])
    assert _labelling(store) == before and not out.exists()

    with pytest.raises(SystemExit) as parsed:
        derive_script.main(["--store", str(store), "--out", str(out), "--force"])
    assert parsed.value.code == 2


def test_main_force_replaces_after_saving_the_old_labelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = a2_store(tmp_path, disabled=(1,))
    before = _labelling(store)
    out = tmp_path / "landmarks.json"
    monkeypatch.setattr(derive_script, "derive", lambda *_a, **_k: _derived())
    monkeypatch.setattr(
        derive_script, "_output_document", lambda *_a, **_k: {"schema": "x"}
    )

    status = derive_script.main(
        ["--store", str(store), "--out", str(out), "--inject", "--force"]
    )

    assert status == 0
    (backup,) = _backups(tmp_path)
    assert json.loads(backup.read_text(encoding="utf-8")) == before
    assert (
        _labelling(store)["written_by"]["software"] == derive_script.SIMULATE_SOFTWARE
    )


# --------------------------------------------------------------------------
# The real CZI pair through Nextflow (slow; never in CI)
# --------------------------------------------------------------------------

WORKFLOWS = REPO_ROOT / "workflows"
SAME_SECTION_OVERRIDE = "same_section:matrix_scale_tolerance=0.05"


def _nextflow(
    script: str,
    *,
    launch: Path,
    sheet: Path,
    outdir: Path,
    czi_python: Path,
    task_tmp: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    nextflow = shutil.which("nextflow")
    if nextflow is None:
        pytest.skip("nextflow is unavailable")
    launch.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "NXF_ANSI_LOG": "false",
        "NXF_DISABLE_CHECK_LATEST": "true",
        "TMPDIR": str(task_tmp),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }
    return subprocess.run(
        [
            nextflow,
            "-log",
            str(launch / "nextflow.log"),
            "run",
            str(WORKFLOWS / script),
            "-profile",
            "conda",
            "-without-conda",
            "--samplesheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--manual_registration_czi_python",
            str(czi_python),
        ],
        cwd=launch,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _held_out(summary: dict[str, Any], stage: str) -> float | None:
    return summary["residual_summary_um"]["held_out"][stage]["median"]  # type: ignore[no-any-return]


@pytest.mark.slow
def test_czi_pair_both_presets(real_path: Callable[[str], Path]) -> None:  # noqa: C901
    test_dir = real_path("MERXEN_MANUAL_REGISTRATION_TEST_DIR")
    evidence = real_path("MERXEN_EVIDENCE_ROOT")
    task_tmp = real_path("MERXEN_TASK_TMP_DIR")
    czi_python = real_path("MERXEN_CZI_PYTHON")
    manifest_path = evidence / "manual_registration" / "czi_test_pairs.json"
    if not manifest_path.is_file():
        pytest.skip(
            "the CZI test manifest does not exist (docs/stages/manual-registration.md)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alias = manifest["default"]
    entry = manifest["pairs"][alias]
    runs = evidence / "manual_registration" / "runs"
    outdir = runs / "out"
    runs.mkdir(parents=True, exist_ok=True)
    sheet = runs / "samplesheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "reference_image",
                "moving_image",
                "presets",
                "registration_settings",
            ]
        )
        writer.writerow(
            [
                alias,
                str(test_dir / entry["reference"]),
                str(test_dir / entry["moving"]),
                "same_section;serial_section",
                SAME_SECTION_OVERRIDE,
            ]
        )

    # 1. Prepare through Nextflow (the real resource requests apply).
    prepared = _nextflow(
        "manual_registration_prepare.nf",
        launch=runs / "launch_prepare",
        sheet=sheet,
        outdir=outdir,
        czi_python=czi_python,
        task_tmp=task_tmp,
        timeout=6 * 3600,
    )
    assert prepared.returncode == 0, prepared.stdout[-3000:] + prepared.stderr[-3000:]
    prepared_dir = outdir / alias / "prepared"
    store = prepared_dir / f"{alias}_prepared.zarr"
    summary = json.loads(
        (prepared_dir / f"{alias}_prepare_summary.json").read_text(encoding="utf-8")
    )
    for role in ("reference", "moving"):
        assert summary["images"][role]["czi"] is not None, role
        provenance = json.loads(
            (prepared_dir / f"{alias}_{role}.prepare.json").read_text(encoding="utf-8")
        )
        assert provenance["conversion"]["converter"].startswith("czifile=="), role

    # 2. Derive and inject test landmarks, unless the store holds someone's own.
    if derive_script.is_foreign_labelling(read_root_attrs(store).get(LABELLING_ATTR)):
        pytest.skip(
            f"the prepared store of {alias} holds a labelling that "
            f"{derive_script.SIMULATE_SOFTWARE} did not write (landmarks placed in "
            "the viewer): this test does not replace it. To run the test on derived "
            "landmarks, run scripts/manual_registration/derive_test_landmarks.py "
            "--inject --force on the store (the old labelling is saved first)"
        )
    derived = derive_script.derive(store, n_pairs=24, n_disabled=8)
    landmarks_json = runs / f"{alias}_test_landmarks.json"
    landmarks_json.write_text(
        json.dumps(derive_script._output_document(store, derived), indent=2) + "\n",
        encoding="utf-8",
    )
    derive_script._inject(store, derived, landmarks_json)

    # 3. Register both presets through Nextflow.
    registered = _nextflow(
        "manual_registration_register.nf",
        launch=runs / "launch_register",
        sheet=sheet,
        outdir=outdir,
        czi_python=czi_python,
        task_tmp=task_tmp,
        timeout=12 * 3600,
    )
    assert registered.returncode == 0, (
        registered.stdout[-3000:] + registered.stderr[-3000:]
    )

    # 4. Read the results.
    summaries: dict[str, dict[str, Any]] = {}
    for preset in ("same_section", "serial_section"):
        path = (
            outdir
            / alias
            / "registered"
            / preset
            / f"{alias}_{preset}_manual_registration_summary.json"
        )
        summaries[preset] = json.loads(path.read_text(encoding="utf-8"))
    comparison = outdir / alias / "registered" / f"{alias}_run_comparison.csv"
    with comparison.open(encoding="utf-8") as handle:
        compared = list(csv.DictReader(handle))

    measured: dict[str, Any] = {
        "alias": alias,
        "derive": derived.diagnostics,
        "scale_ratio_matrix": derived.scale_ratio,
        "runs": {
            preset: {
                "fit": result["fit"],
                "refinement": result["refinement"],
                "nonrigid": result["nonrigid"],
                "tissue_pearson": result["tissue_qc"]["pearson"],
                "held_out_median_um": {
                    stage: _held_out(result, stage)
                    for stage in ("landmark", "affine", "final")
                },
                "nucleus_qc": result["nucleus_qc"],
                "resources": result["resources"],
                "warnings": result["warnings"],
            }
            for preset, result in summaries.items()
        },
    }
    failures: list[str] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        measured.setdefault("gates", {})[name] = {
            "passed": bool(passed),
            "detail": detail,
        }
        if not passed:
            failures.append(f"{name}: {detail}")

    # The override is the documented way to run same_section on a serial pair. These
    # are gates like the others: a failure is recorded and asserted after the
    # numbers are written, so a failed run keeps its measurements.
    gate(
        "0_override",
        all(
            result["settings"]["matrix_scale_tolerance"] == 0.05
            for result in summaries.values()
        )
        and PRESETS["same_section"].get("matrix_scale_tolerance", 0.02) != 0.05,
        {
            preset: result["settings"]["matrix_scale_tolerance"]
            for preset, result in summaries.items()
        },
    )
    gate(
        "0_complete",
        all(result["status"] == "complete" for result in summaries.values()),
        {preset: result["status"] for preset, result in summaries.items()},
    )

    serial = summaries["serial_section"]
    fit = serial["fit"]
    gate(
        "1_fit",
        fit["n_used"] >= 0.75 * fit["n_enabled"]
        and abs(fit["scale_ratio"] - 1.0) <= 0.05
        and fit["reflected"] == derived.coarse.mirror_x
        and fit["other_class_median_um"] >= 2.0 * fit["chosen_class_median_um"]
        and fit["spread_ratio"] >= 0.2,
        {
            key: fit[key]
            for key in (
                "n_used",
                "n_enabled",
                "scale_ratio",
                "reflected",
                "chosen_class_median_um",
                "other_class_median_um",
                "spread_ratio",
            )
        },
    )
    pearson = serial["tissue_qc"]["pearson"]
    gate(
        "2_tissue_pearson",
        pearson["affine"] >= pearson["landmark"] - 0.01
        and pearson["final"] >= pearson["affine"] - 0.01,
        pearson,
    )
    held_final = _held_out(serial, "final")
    held_landmark = _held_out(serial, "landmark")
    gate(
        "3_held_out",
        held_final is not None
        and held_landmark is not None
        and held_final <= held_landmark + 5.0,
        {"final": held_final, "landmark": held_landmark},
    )
    nonrigid = serial["nonrigid"]
    if nonrigid["status"] == "applied":
        flow = nonrigid["final_flow"]
        gate(
            "4_nonrigid",
            flow["min_jacobian"] >= 0.67 and flow["max_jacobian"] <= 1.5,
            flow,
        )
    else:
        gate("4_nonrigid", nonrigid["reason"] is not None, nonrigid)
    stores = {
        preset: outdir
        / alias
        / "registered"
        / preset
        / f"{alias}_{preset}_registered.zarr"
        for preset in summaries
    }
    ok = len(set(stores.values())) == 2
    for result_store in stores.values():
        ok = ok and read_root_attrs(result_store)[STORE_ATTR]["kind"] == "result"
        validate_merxen_schema(sd.read_zarr(result_store, selection=("images",)))
    ok = ok and len(compared) == 2
    ok = ok and comparison.with_suffix(".png").is_file()
    gate("5_stores_and_comparison", ok, {"rows": len(compared)})
    gate(
        "6_resources",
        all(r["resources"]["peak_rss_gb"] <= 48.0 for r in summaries.values()),
        {p: r["resources"] for p, r in summaries.items()},
    )

    (runs / f"{alias}_validation.json").write_text(
        json.dumps(measured, indent=2, default=float) + "\n", encoding="utf-8"
    )
    assert not failures, failures
