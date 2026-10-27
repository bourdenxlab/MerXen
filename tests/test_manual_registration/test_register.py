"""Landmark fit, store validation and the pure helpers of the register step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from merxen.image_registration import push_points_through_registration
from merxen.manual_registration import register as register_module
from merxen.manual_registration.config import ManualRegistrationSettings
from merxen.manual_registration.presets import resolve_settings
from merxen.manual_registration.register import (
    LandmarkFitError,
    LandmarkFitReport,
    _landmark_records,
    _plain_json,
    _provenance_digest,
    _publish,
    _residual_summary,
    _validate_prepared_store,
    fit_landmarks,
    fit_summary,
)
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LabellingAttrs,
    StoreAttrs,
    UnsupportedSchemaVersionError,
    labelling_content_sha256,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs,
)
from merxen.manual_registration.simulate import inject_labelling

from ._mr_helpers import (
    A2_COARSE,
    A2_MOVING,
    A2_MOVING_SHAPE,
    A2_PIXEL_SIZE,
    A2_REFERENCE_SHAPE,
    a2_pairs,
    a2_store,
    a2_truth_matrix,
    fake_store,
    labelling_dict,
    make_pairs,
    store_attr_dict,
)

RUN_ID = "SYN1_same_section"


def _settings(
    preset: str = "same_section", row_settings: str = ""
) -> ManualRegistrationSettings:
    return resolve_settings(
        preset,
        reference_pixel_size_um=A2_PIXEL_SIZE,
        reference_channel="DAPI",
        row_settings=row_settings,
    )


def _store() -> Any:
    return parse_store_attrs(
        store_attr_dict(
            A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
        )
    )


def _labelling(
    pairs: Any = None, coarse: CoarseTransform = A2_COARSE, **kwargs: Any
) -> LabellingAttrs:
    payload = labelling_dict(a2_pairs() if pairs is None else pairs, coarse, **kwargs)
    return parse_labelling_attrs(payload)


def _fit(
    labelling: Any = None, settings: Any = None, **kwargs: Any
) -> LandmarkFitReport:
    return fit_landmarks(
        labelling or _labelling(),
        _store(),
        settings or _settings(),
        **kwargs,
    )


def _truth_pairs(moving: list[tuple[float, float]], scale: float = 1.0) -> Any:
    matrix = a2_truth_matrix()
    points = np.asarray(moving, dtype=float)
    reference = points @ matrix[:2, :2].T + matrix[:2, 2]
    centroid = reference.mean(axis=0)
    reference = centroid + scale * (reference - centroid)
    return make_pairs(moving, reference)


# --------------------------------------------------------------------------
# fit_landmarks
# --------------------------------------------------------------------------


def test_default_fit_trims_the_planted_outlier() -> None:
    report = _fit(run_id=RUN_ID)
    assert int(report.used.sum()) == 9
    assert list(np.flatnonzero(report.outlier)) == [4]
    np.testing.assert_allclose(
        report.matrix_full[:2, :2],
        [[-0.999861, 0.017468], [-0.017468, -0.999861]],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        report.matrix_full[:2, 2], [76009.005205, 54892.785195], atol=1e-2
    )
    assert report.loo_um[4] == pytest.approx(41.0328, abs=1e-3)
    assert "1 landmark pair(s) flagged as outlier and excluded: 5" in report.warnings
    summary = fit_summary(report, store=_store(), settings=_settings())
    assert summary["n_used"] == 9
    assert summary["rotation_deg"] == pytest.approx(178.99913, abs=1e-4)
    assert summary["n_outliers"] == 1
    assert summary["n_enabled"] == 10
    assert summary["viewer_outlier_floor_um"] is None
    assert all(type(v) in (float, int, bool, str, type(None)) for v in summary.values())


def test_fewer_than_eight_enabled_pairs_fail() -> None:
    labelling = _labelling(a2_pairs(disabled=(1, 2, 3)))
    with pytest.raises(LandmarkFitError, match="7 enabled") as excinfo:
        _fit(labelling, run_id=RUN_ID)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")
    assert excinfo.value.report is None


def test_collinear_pairs_fail_with_a_report() -> None:
    moving = [(5000.0 + 6000.0 * i, 20000.0 + 0.5 * i) for i in range(10)]
    labelling = _labelling(_truth_pairs(moving))
    with pytest.raises(LandmarkFitError, match="collinear") as excinfo:
        _fit(labelling)
    assert excinfo.value.report is not None


def test_wrong_flip_is_detected() -> None:
    labelling = _labelling(
        coarse=CoarseTransform(rotation_deg=179.0, mirror_x=True, offset_um=(150, -80))
    )
    with pytest.raises(LandmarkFitError, match="mirrored"):
        _fit(labelling)


def test_scale_off_by_three_percent_depends_on_the_tolerance() -> None:
    pairs = _truth_pairs(list(A2_MOVING), scale=1.03)
    labelling = _labelling(pairs)
    with pytest.raises(LandmarkFitError, match="pixel-size ratio") as excinfo:
        _fit(labelling)
    assert "same_section:matrix_scale_tolerance=0.05" in str(excinfo.value)
    _fit(labelling, _settings("serial_section"))
    _fit(
        labelling,
        _settings(row_settings="same_section:matrix_scale_tolerance=0.05"),
    )


def test_outlier_policies() -> None:
    with pytest.raises(LandmarkFitError, match="flagged as outliers: 5") as excinfo:
        _fit(settings=_settings(row_settings="landmark_outlier_policy=fail"))
    assert excinfo.value.report is not None
    report = _fit(settings=_settings(row_settings="landmark_outlier_policy=keep"))
    assert int(report.used.sum()) == 10
    assert list(np.flatnonzero(report.outlier)) == [4]
    assert any("kept" in w for w in report.warnings)


def test_outside_crop_policies() -> None:
    crop = {"reference": CropRect(x0=0, y0=0, x1=70000, y1=61000)}
    labelling = _labelling(crops=crop)
    used = _fit(labelling)
    assert used.outside_crop[0] and used.used[0]
    assert any("outside a crop are used" in w and "1" in w for w in used.warnings)
    excluded = _fit(labelling, _settings(row_settings="landmarks_outside_crop=exclude"))
    assert not excluded.used[0]
    assert 0 not in excluded.fit_index
    assert excluded.outside_crop[0]
    with pytest.raises(LandmarkFitError, match="outside a crop"):
        _fit(labelling, _settings(row_settings="landmarks_outside_crop=fail"))


def test_disabled_pairs_stay_out_of_the_fit() -> None:
    report = _fit(_labelling(a2_pairs(disabled=(9, 10))))
    assert 8 not in report.fit_index and 9 not in report.fit_index
    np.testing.assert_array_equal(report.loo_um[8:], report.residuals_um[8:])
    assert not report.used[8:].any()
    assert report.enabled.sum() == 8


def test_large_rotation_difference_warns() -> None:
    labelling = _labelling(
        coarse=CoarseTransform(rotation_deg=150.0, mirror_x=False, offset_um=(150, -80))
    )
    report = _fit(labelling)
    assert any("differs from the coarse rotation" in w for w in report.warnings)


def test_messages_carry_the_run_id() -> None:
    labelling = _labelling(a2_pairs(disabled=(1, 2, 3)))
    with pytest.raises(LandmarkFitError) as excinfo:
        _fit(labelling, run_id="SYN1_same_section")
    assert "[SYN1_same_section]" in str(excinfo.value)
    with pytest.raises(LandmarkFitError) as excinfo:
        _fit(labelling)
    assert str(excinfo.value).startswith("[SYN1]")


def test_viewer_floor_warning() -> None:
    labelling = _labelling(outlier_floor_um=10.0)
    report = _fit(labelling, _settings("serial_section"))
    assert any(
        "viewer's outlier floor was 10 um, this fit uses 30 um" in w
        for w in report.warnings
    )
    summary = fit_summary(
        report,
        store=_store(),
        settings=_settings("serial_section"),
        labelling=labelling,
    )
    assert summary["viewer_outlier_floor_um"] == 10
    same = _fit(labelling)
    assert not any("viewer's outlier floor" in w for w in same.warnings)
    assert not any("viewer's outlier floor" in w for w in _fit().warnings)


# --------------------------------------------------------------------------
# _validate_prepared_store
# --------------------------------------------------------------------------


def _validate(
    store: Path, **kwargs: Any
) -> tuple[StoreAttrs, LabellingAttrs, list[str]]:
    kwargs.setdefault("check_fingerprints", False)
    return _validate_prepared_store(store, run_id=RUN_ID, dataset_name="SYN1", **kwargs)


def test_valid_store_passes(tmp_path: Path) -> None:
    store, labelling, warnings = _validate(a2_store(tmp_path))
    assert store.kind == "prepared" and labelling.status == "finished"
    assert warnings == []


def test_missing_store_and_missing_attrs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manual_registration_prepare"):
        _validate(tmp_path / "absent.zarr")
    empty = fake_store(tmp_path / "empty.zarr", {})
    with pytest.raises(ValueError, match="not a manual-registration store"):
        _validate(empty)


def test_newer_schema_versions_are_refused(tmp_path: Path) -> None:
    attrs = store_attr_dict(
        A2_REFERENCE_SHAPE,
        A2_MOVING_SHAPE,
        A2_PIXEL_SIZE,
        A2_PIXEL_SIZE,
        schema_version=2,
    )
    labelling = labelling_dict(a2_pairs(), A2_COARSE)
    store = fake_store(
        tmp_path / "a.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: labelling}
    )
    with pytest.raises(UnsupportedSchemaVersionError, match="newer") as excinfo:
        _validate(store)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")

    good_store = store_attr_dict(
        A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
    )
    newer = labelling_dict(a2_pairs(), A2_COARSE, schema_version=2)
    store2 = fake_store(
        tmp_path / "b.zarr", {STORE_ATTR: good_store, LABELLING_ATTR: newer}
    )
    with pytest.raises(UnsupportedSchemaVersionError, match="newer") as excinfo:
        _validate(store2)
    assert str(excinfo.value).startswith(f"[{RUN_ID}]")

    before = read_root_attrs(store)
    with pytest.raises(UnsupportedSchemaVersionError):
        inject_labelling(store, a2_pairs())
    assert read_root_attrs(store) == before


def test_store_kind_dataset_and_labelling_state(tmp_path: Path) -> None:
    result = fake_store(
        tmp_path / "r.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE,
                A2_MOVING_SHAPE,
                A2_PIXEL_SIZE,
                A2_PIXEL_SIZE,
                kind="result",
            )
        },
    )
    with pytest.raises(ValueError, match="result store"):
        _validate(result)
    with pytest.raises(ValueError, match="dataset"):
        _validate_prepared_store(
            a2_store(tmp_path),
            run_id=RUN_ID,
            dataset_name="SYN2",
            check_fingerprints=False,
        )
    bare = fake_store(
        tmp_path / "bare.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
            )
        },
    )
    with pytest.raises(ValueError, match="Finish labelling"):
        _validate(bare)
    draft = a2_store(tmp_path, status="draft", name="draft.zarr")
    with pytest.raises(ValueError, match="Finish labelling"):
        _validate(draft)
    _, labelling, _ = _validate(draft, require_finished=False)
    assert labelling.status == "draft"


def test_shape_mismatch_is_refused(tmp_path: Path) -> None:
    labelling = labelling_dict(a2_pairs(), A2_COARSE, ref_shape=(100, 100))
    store = fake_store(
        tmp_path / "s.zarr",
        {
            STORE_ATTR: store_attr_dict(
                A2_REFERENCE_SHAPE, A2_MOVING_SHAPE, A2_PIXEL_SIZE, A2_PIXEL_SIZE
            ),
            LABELLING_ATTR: labelling,
        },
    )
    with pytest.raises(ValueError, match="different reference image"):
        _validate(store)


def test_fingerprint_mismatch_and_content_sha_warning(tmp_path: Path) -> None:
    files = {}
    for role in ("reference", "moving"):
        path = tmp_path / f"{role}.ome.tif"
        path.write_bytes(b"x" * 100)
        files[role] = path
    attrs = store_attr_dict(
        A2_REFERENCE_SHAPE,
        A2_MOVING_SHAPE,
        A2_PIXEL_SIZE,
        A2_PIXEL_SIZE,
        ome_paths=files,
    )
    labelling = labelling_dict(a2_pairs(), A2_COARSE)
    store = fake_store(
        tmp_path / "f.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: labelling}
    )
    _validate(store, check_fingerprints=True)
    files["moving"].write_bytes(b"y" * 100)
    with pytest.raises(ValueError, match="re-run prepare"):
        _validate(store, check_fingerprints=True)
    files["moving"].unlink()
    with pytest.raises(FileNotFoundError, match="re-run prepare"):
        _validate(store, check_fingerprints=True)

    wrong = dict(labelling, content_sha256="sha256:" + "0" * 64)
    store2 = fake_store(tmp_path / "w.zarr", {STORE_ATTR: attrs, LABELLING_ATTR: wrong})
    _, _, warnings = _validate(store2)
    assert any("content_sha256 differs" in w for w in warnings)


# --------------------------------------------------------------------------
# _landmark_records and friends
# --------------------------------------------------------------------------


def _report_for(pairs: Any) -> LandmarkFitReport:
    n = len(pairs)
    return LandmarkFitReport(
        robust=cast(Any, None),
        pairs=pairs,
        enabled=np.array([p.enabled for p in pairs]),
        outside_crop=np.zeros(n, dtype=bool),
        matrix_full=np.eye(3),
        warnings=[],
        fit_index=np.arange(n),
        used=np.array([p.enabled for p in pairs]),
        outlier=np.zeros(n, dtype=bool),
        residuals_um=np.zeros(n),
        loo_um=np.arange(n, dtype=float),
    )


MATRIX = np.array([[1.2, 0.3, 50.0], [-0.3, 1.2, 80.0], [0.0, 0.0, 1.0]])
WORKING = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])


def _flow() -> np.ndarray:
    ys, xs = np.mgrid[0:200, 0:260]
    return np.stack([3.0 * np.sin(xs / 40.0), 2.5 * np.cos(ys / 35.0)]).astype(
        np.float32
    )


def test_records_without_a_flow_equal_the_affine() -> None:
    moving = [(100.0, 100.0), (300.0, 200.0), (500.0, 400.0)]
    pairs = make_pairs(moving, [(10.0, 10.0)] * 3)
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(64, 128),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=MATRIX,
        flow_yx=None,
        flow_working_to_reference=None,
    )
    for record, (mx, my) in zip(records, moving, strict=True):
        affine = MATRIX @ [mx, my, 1.0]
        expected = [affine[0] + 64, affine[1] + 128]
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["affine"], expected
        )
        assert (
            record["moving_in_reference_xy_px"]["final"]
            == record["moving_in_reference_xy_px"]["affine"]
        )
        assert record["final_includes_flow"] is False
        assert record["push_converged"] is True
        assert record["residual_um"]["final"] == record["residual_um"]["affine"]
    assert records[2]["residual_um"]["loo"] == 2.0


def test_records_with_a_flow_push_in_the_crop_frame() -> None:
    inside = [(100.0, 100.0), (300.0, 200.0)]
    outside = [(5000.0, 5000.0)]
    moving = inside + outside
    pairs = make_pairs(moving, [(10.0, 10.0)] * 3)
    flow = _flow()
    records = _landmark_records(
        _report_for(pairs),
        reference_pixel_size_um=0.5,
        reference_origin_xy=(64, 128),
        moving_origin_xy=(0, 0),
        landmark_full=np.eye(3),
        final_affine_crop=MATRIX,
        flow_yx=flow,
        flow_working_to_reference=WORKING,
    )
    pushed, _ = push_points_through_registration(
        np.asarray(inside),
        final_matrix=MATRIX,
        flow_yx=flow,
        working_to_reference=WORKING,
    )
    for record, expected in zip(records[:2], pushed + [64, 128], strict=True):
        np.testing.assert_allclose(
            record["moving_in_reference_xy_px"]["final"], expected, atol=1e-9
        )
        assert record["final_includes_flow"] is True
        assert (
            record["moving_in_reference_xy_px"]["final"]
            != (record["moving_in_reference_xy_px"]["affine"])
        )
    far = records[2]
    assert far["final_includes_flow"] is False
    np.testing.assert_allclose(
        far["moving_in_reference_xy_px"]["final"],
        far["moving_in_reference_xy_px"]["affine"],
    )


def test_residual_summary_groups_and_empty_groups() -> None:
    def record(enabled: bool, used: bool, value: float) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "used_in_fit": used,
            "residual_um": {"landmark": value, "affine": value, "final": value / 2},
        }

    summary = _residual_summary(
        [record(True, True, 2.0), record(True, True, 4.0), record(False, False, 10.0)]
    )
    assert summary["used"]["landmark"] == {
        "n": 2,
        "median": 3.0,
        "p90": pytest.approx(3.8),
        "max": 4.0,
    }
    assert summary["held_out"]["final"]["median"] == 5.0
    empty = _residual_summary([record(True, True, 1.0)])
    assert empty["held_out"]["landmark"] == {
        "n": 0,
        "median": None,
        "p90": None,
        "max": None,
    }


def test_provenance_digest_depends_on_its_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    store = _store()
    payload = labelling_dict(a2_pairs(), A2_COARSE)
    sha = labelling_content_sha256(payload)

    def digest(**overrides: Any) -> str:
        kwargs: dict[str, Any] = {
            "preset": "same_section",
            "run_label": None,
            "settings": settings,
            "labelling_sha": sha,
            "store": store,
        }
        kwargs.update(overrides)
        return _provenance_digest(**kwargs)

    base = digest()
    assert base == digest()
    moved = a2_pairs()
    moved[1] = moved[1].model_copy(update={"reference_xy_px": (46117.596, 48370.654)})
    moved_sha = labelling_content_sha256(labelling_dict(moved, A2_COARSE))
    assert digest(labelling_sha=moved_sha) != base
    assert digest(run_label="x") != base
    monkeypatch.setitem(register_module.PRESET_VERSIONS, "same_section", "v2")
    assert digest() != base


def test_plain_json_and_publish(tmp_path: Path) -> None:
    value = {
        "a": np.float32(1.5),
        "b": np.int64(3),
        "c": float("nan"),
        "d": (1, 2),
        "e": np.array([1.0, np.inf]),
        "f": tmp_path,
        "g": np.bool_(True),
    }
    plain = _plain_json(value)
    assert plain == {
        "a": 1.5,
        "b": 3,
        "c": None,
        "d": [1, 2],
        "e": [1.0, None],
        "f": str(tmp_path),
        "g": True,
    }
    assert type(plain["g"]) is bool
    source = tmp_path / "a.txt"
    source.write_text("hello")
    target = _publish(source, tmp_path / "sub" / "b.txt")
    assert target.read_text() == "hello"
    assert [p.name for p in target.parent.iterdir()] == ["b.txt"]
