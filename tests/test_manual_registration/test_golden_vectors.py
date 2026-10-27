"""Tests for the cross-repo golden vectors of manual landmark registration.

The viewer copies ``golden_v1.json`` byte-for-byte and pins the same SHA-256;
after any regeneration the new digest must be announced to the viewer work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from merxen.manual_registration import geometry, landmark_fit, schema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "manual_registration" / "make_golden_vectors.py"
GOLDEN_PATH = (
    REPO_ROOT / "tests" / "test_data" / "manual_registration" / "golden_v1.json"
)
GOLDEN_SHA256 = "ed1e81a1b02fda322b458a3d3dd980ce1d5d3e99bdc4ee6e501bf8ec0ea34164"

RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_golden_vectors_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data


def _array(values: Any) -> np.ndarray:
    """Read nested lists where ``null`` means NaN."""
    return np.array(values, dtype=float)


def _assert_same(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(left, right, f"{path}[{index}]")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        assert actual is expected, path
    elif isinstance(expected, int) != isinstance(actual, int):
        raise AssertionError(
            f"{path}: int/float type mismatch {actual!r} vs {expected!r}"
        )
    elif isinstance(expected, float) or isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), path
    else:
        assert actual == expected, path


def _coarse(values: dict[str, Any]) -> schema.CoarseTransform:
    return schema.CoarseTransform(**values)


def _crop(values: dict[str, int] | None) -> schema.CropRect | None:
    return None if values is None else schema.CropRect(**values)


def test_golden_file_digest_is_pinned() -> None:
    assert hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest() == GOLDEN_SHA256


def test_golden_file_is_reproducible(
    script: ModuleType, golden: dict[str, Any]
) -> None:
    rebuilt = json.loads(script.render(script.build_golden_vectors()))
    _assert_same(rebuilt, golden)
    assert script.first_difference(rebuilt, golden) is None


def test_first_difference_compares_ints_exactly(script: ModuleType) -> None:
    assert script.first_difference(10, 10) is None
    assert script.first_difference(10, 10.0) == "$"
    assert script.first_difference(10.0, 10) == "$"
    assert script.first_difference({"a": [1.0]}, {"a": [1]}) == "$.a[0]"
    assert script.first_difference(1.0, 1.0 + 1e-12) is None
    assert script.first_difference(True, 1) == "$"
    with pytest.raises(AssertionError, match="int/float"):
        _assert_same(10, 10.0)
    with pytest.raises(AssertionError, match="int/float"):
        _assert_same(10.0, 10)


def test_golden_file_is_canonically_rendered(
    script: ModuleType, golden: dict[str, Any]
) -> None:
    text = GOLDEN_PATH.read_text(encoding="utf-8")
    assert text == json.dumps(golden, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert (
        golden["generated_by"] == "scripts/manual_registration/make_golden_vectors.py"
    )
    assert golden["tolerances"]["matrix_rtol"] == RTOL


def test_golden_constants_match_the_code(golden: dict[str, Any]) -> None:
    assert set(golden["constants"]) == {
        "SCHEMA_VERSION",
        "STORE_ATTR",
        "LABELLING_ATTR",
        "RESULT_ATTR",
        "LAYER_METADATA_KEY",
        "MIN_LANDMARK_PAIRS",
        "DEFAULT_OUTLIER_FLOOR_UM",
        "OUTLIER_K",
        "MAX_OUTLIER_FRACTION",
        "MIN_USED_PAIRS",
        "CROP_SNAP_PX",
        "CROP_MIN_SIZE_PX",
    }
    for name, value in golden["constants"].items():
        assert getattr(schema, name) == value, name
    assert golden["conventions"] == schema.CONVENTIONS


def test_golden_wrap_deg(golden: dict[str, Any]) -> None:
    assert len(golden["wrap_deg"]) == 9
    for entry in golden["wrap_deg"]:
        assert abs(geometry.wrap_deg(entry["input"]) - entry["output"]) < 1e-9


def test_golden_coarse_cases(golden: dict[str, Any]) -> None:
    assert len(golden["coarse_cases"]) == 8
    for case in golden["coarse_cases"]:
        kwargs = {
            "moving_shape_yx": tuple(case["moving_shape_yx"]),
            "reference_shape_yx": tuple(case["reference_shape_yx"]),
            "moving_pixel_size_um": case["moving_pixel_size_um"],
            "reference_pixel_size_um": case["reference_pixel_size_um"],
        }
        matrix = geometry.coarse_matrix_xy(_coarse(case["coarse"]), **kwargs)
        np.testing.assert_allclose(matrix, case["matrix_xy"], rtol=RTOL, atol=ATOL)
        recovered, ratio = geometry.coarse_from_matrix(
            np.array(case["matrix_xy"]), **kwargs
        )
        expected = case["from_matrix"]["coarse"]
        assert (
            abs(geometry.wrap_deg(recovered.rotation_deg - expected["rotation_deg"]))
            < 1e-9
        )
        assert recovered.mirror_x == expected["mirror_x"]
        np.testing.assert_allclose(
            recovered.offset_um, expected["offset_um"], atol=1e-6
        )
        assert ratio == pytest.approx(case["from_matrix"]["scale_ratio"], abs=1e-12)


def test_golden_coarse_transitions(golden: dict[str, Any]) -> None:
    assert len(golden["coarse_transitions"]) == 70
    for entry in golden["coarse_transitions"]:
        start = _coarse(entry["start"])
        argument = entry["argument"]
        action = entry["action"]
        if action == "flip_horizontal":
            result = geometry.flipped_horizontal(start)
        elif action == "flip_vertical":
            result = geometry.flipped_vertical(start)
        elif action == "rotate":
            result = geometry.rotated(start, argument)
        elif action == "set_rotation":
            result = schema.CoarseTransform(
                rotation_deg=argument,
                mirror_x=start.mirror_x,
                offset_um=start.offset_um,
            )
        elif action == "shift":
            result = geometry.shifted(start, argument[0], argument[1])
        else:
            assert action == "reset"
            result = schema.CoarseTransform()
        expected = entry["result"]
        assert (
            abs(geometry.wrap_deg(result.rotation_deg - expected["rotation_deg"]))
            < 1e-9
        )
        assert result.mirror_x == expected["mirror_x"]
        np.testing.assert_allclose(result.offset_um, expected["offset_um"], atol=1e-9)


def test_golden_snap_crop(golden: dict[str, Any]) -> None:
    for entry in golden["snap_crop"]:
        crop = _crop(entry["crop"])
        shape = tuple(entry["shape_yx"])
        if entry["error"]:
            with pytest.raises(ValueError):
                geometry.snap_crop(
                    crop,
                    shape,
                    multiple=entry["multiple"],
                    min_size_px=entry["min_size_px"],
                )
        else:
            effective = geometry.snap_crop(
                crop,
                shape,
                multiple=entry["multiple"],
                min_size_px=entry["min_size_px"],
            )
            assert effective == _crop(entry["effective"])


def test_golden_crop_frames(golden: dict[str, Any]) -> None:
    for entry in golden["crop_frames"]:
        origins = {
            "reference_origin_xy": tuple(entry["reference_origin_xy"]),
            "moving_origin_xy": tuple(entry["moving_origin_xy"]),
        }
        cropped = geometry.crop_matrix(np.array(entry["matrix_full"]), **origins)
        np.testing.assert_allclose(cropped, entry["matrix_crop"], rtol=RTOL, atol=ATOL)
        restored = geometry.full_matrix(np.array(entry["matrix_crop"]), **origins)
        np.testing.assert_allclose(restored, entry["matrix_full"], rtol=RTOL, atol=1e-6)


def test_golden_points_outside_crop(golden: dict[str, Any]) -> None:
    for entry in golden["points_outside_crop"]:
        outside = geometry.points_outside_crop(entry["points_xy"], _crop(entry["crop"]))
        assert outside.tolist() == entry["outside"]


def test_golden_napari_affine(golden: dict[str, Any]) -> None:
    for entry in golden["napari_affine"]:
        affine = geometry.napari_affine_rc(
            np.array(entry["reference_base_rc"]), np.array(entry["matrix_xy"])
        )
        np.testing.assert_allclose(affine, entry["affine_rc"], rtol=RTOL, atol=ATOL)


def test_golden_fits(golden: dict[str, Any]) -> None:
    assert len(golden["fits"]) == 5
    for entry in golden["fits"]:
        moving, reference = _array(entry["moving_xy"]), _array(entry["reference_xy"])
        fit = landmark_fit.fit_similarity(
            moving, reference, reflection=entry["reflection"]
        )
        np.testing.assert_allclose(fit.matrix, entry["matrix"], rtol=RTOL, atol=ATOL)
        assert fit.scale == pytest.approx(entry["scale"], rel=RTOL)
        assert abs(geometry.wrap_deg(fit.rotation_deg - entry["rotation_deg"])) < 1e-9
        assert fit.reflected == entry["reflected"]
        np.testing.assert_allclose(fit.residuals_px, entry["residuals_px"], atol=1e-6)
        loo = landmark_fit.leave_one_out_residuals(
            moving, reference, reflection=entry["reflection"]
        )
        np.testing.assert_allclose(loo, _array(entry["loo_px"]), atol=1e-6)


def test_golden_spread_ratio(golden: dict[str, Any]) -> None:
    for entry in golden["spread_ratio"]:
        assert landmark_fit.spread_ratio(entry["points_xy"]) == pytest.approx(
            entry["value"], abs=1e-9
        )


def test_golden_robust_fits(golden: dict[str, Any]) -> None:
    assert len(golden["robust_fits"]) == 7
    for entry in golden["robust_fits"]:
        options = landmark_fit.RobustFitOptions(**entry["options"])
        moving, reference = _array(entry["moving_xy"]), _array(entry["reference_xy"])
        result = landmark_fit.robust_similarity(
            moving,
            reference,
            reflection=entry["reflection"],
            moving_pixel_size_um=entry["moving_pixel_size_um"],
            reference_pixel_size_um=entry["reference_pixel_size_um"],
            options=options,
        )
        name = entry["name"]
        np.testing.assert_allclose(
            result.fit.matrix, entry["matrix"], rtol=RTOL, atol=ATOL, err_msg=name
        )
        assert result.fit.scale == pytest.approx(entry["scale"], rel=RTOL), name
        assert (
            abs(geometry.wrap_deg(result.fit.rotation_deg - entry["rotation_deg"]))
            < 1e-9
        ), name
        assert result.used.tolist() == entry["used"], name
        assert result.outlier.tolist() == entry["outlier"], name
        for key, actual in (
            ("residuals_um", result.residuals_um),
            ("loo_um", result.loo_um),
        ):
            expected = _array(entry[key])
            assert np.isnan(actual).tolist() == np.isnan(expected).tolist(), (name, key)
            np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)
        first_pass = (
            landmark_fit.leave_one_out_residuals(
                moving, reference, reflection=entry["reflection"]
            )
            * entry["reference_pixel_size_um"]
        )
        expected_first = _array(entry["first_pass_loo_um"])
        assert np.isnan(first_pass).tolist() == np.isnan(expected_first).tolist(), name
        np.testing.assert_allclose(
            first_pass, expected_first, atol=1e-6, equal_nan=True
        )
        for key, actual_value in (
            ("chosen_class_median_um", result.chosen_class_median_um),
            ("other_class_median_um", result.other_class_median_um),
        ):
            assert actual_value == pytest.approx(entry[key], abs=1e-6), (name, key)
        assert result.spread_ratio == pytest.approx(entry["spread_ratio"], abs=1e-9)
        assert result.scale_ratio == pytest.approx(entry["scale_ratio"], rel=RTOL)


def test_golden_nan_loo_case_is_present(golden: dict[str, Any]) -> None:
    """The NaN leave-one-out case pins the finite-only median in both repos."""
    entries = {entry["name"]: entry for entry in golden["robust_fits"]}
    trimmed = entries["nan_loo_trimmed"]
    assert trimmed["used"] == [True, True, True, True, True, False, True]
    assert trimmed["outlier"] == [False, False, False, False, False, True, False]
    assert trimmed["loo_um"][6] is None
    assert trimmed["first_pass_loo_um"][6] is None
    assert entries["nan_loo_no_trimming"]["outlier"] == trimmed["outlier"]


def test_golden_pair_colours(golden: dict[str, Any]) -> None:
    assert len(golden["pair_colour"]) == 24
    for entry in golden["pair_colour"]:
        assert schema.pair_colour(entry["pair_id"], entry["seed"]) == entry["colour"]


def test_golden_canonical(golden: dict[str, Any]) -> None:
    assert len(golden["canonical"]) == 2
    for entry in golden["canonical"]:
        assert schema.canonical_json(entry["object"]) == entry["canonical_json"]
        assert schema.canonical_sha256(entry["object"]) == entry["sha256"]


def test_golden_examples_parse_and_digest(golden: dict[str, Any]) -> None:
    examples = golden["examples"]
    prepared = schema.parse_store_attrs(examples["store_attr_prepared"])
    result = schema.parse_store_attrs(examples["store_attr_result"])
    assert (prepared.kind, result.kind) == ("prepared", "result")
    labelling = schema.parse_labelling_attrs(examples["labelling"])
    assert labelling.status == "finished"
    digest = schema.labelling_content_sha256(examples["labelling"])
    assert (
        digest
        == examples["labelling"]["content_sha256"]
        == examples["labelling_content_sha256"]
    )
