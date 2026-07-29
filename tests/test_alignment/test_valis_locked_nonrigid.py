"""Focused tests for VALIS non-rigid registration with a locked global frame."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.alignment.bundle import DisplacementField
from merxen.alignment.frames import RegistrationFrame
from merxen.alignment.transforms import apply_affine_matrix
from merxen.alignment.valis_register import (
    _assert_valis_rigid_map_is_identity,
    _coherent_euclidean_drift_diagnostics,
    _execute_valis,
    _feather_binary_mask,
    _LockedIdentityTransform,
    _shared_non_rigid_domain,
    _taper_displacement_field,
    _valis_locked_global_compatibility,
)
from merxen.config import ValisAlignmentConfig


def test_identity_transform_cannot_estimate_a_second_global_fit() -> None:
    transform = _LockedIdentityTransform()
    source = np.array([[1.0, 2.0], [7.0, 4.0], [3.0, 9.0]])
    target = source + np.array([50.0, -20.0])

    assert transform.estimate(source, target)
    np.testing.assert_array_equal(transform.params, np.eye(3))


def test_locked_valis_supplies_and_restores_missing_identity_transformer() -> None:
    rigid_registrar = SimpleNamespace()

    class FakeValis:
        def rigid_register_partial(
            self: FakeValis,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            del args, kwargs
            return rigid_registrar

    original = FakeValis.rigid_register_partial
    with _valis_locked_global_compatibility(FakeValis):
        registrar = FakeValis().rigid_register_partial()
        assert isinstance(registrar.transformer, _LockedIdentityTransform)
        np.testing.assert_array_equal(registrar.transformer.params, np.eye(3))

    assert FakeValis.rigid_register_partial is original


def test_shared_domain_and_feather_taper_field_to_zero() -> None:
    shape = (41, 45)
    fixed_tissue = np.zeros(shape, dtype=np.uint8)
    moving_tissue = np.zeros(shape, dtype=np.uint8)
    fixed_valid = np.zeros(shape, dtype=np.uint8)
    moving_valid = np.zeros(shape, dtype=np.uint8)
    fixed_tissue[4:37, 5:41] = 255
    moving_tissue[7:39, 2:38] = 255
    fixed_valid[6:35, 7:39] = 255
    moving_valid[9:34, 4:36] = 255

    shared = _shared_non_rigid_domain(
        fixed_tissue_mask=fixed_tissue,
        moving_tissue_mask=moving_tissue,
        fixed_valid_mask=fixed_valid,
        moving_valid_mask=moving_valid,
    )
    expected = (
        (fixed_tissue > 0)
        & (moving_tissue > 0)
        & (fixed_valid > 0)
        & (moving_valid > 0)
    )
    np.testing.assert_array_equal(shared > 0, expected)

    weight = _feather_binary_mask(shared, taper_px=6.0)
    assert weight[0, 0] == 0.0
    assert weight[20, 20] == pytest.approx(1.0)
    assert 0.0 < weight[9, 10] < 1.0

    x = np.arange(0, shape[1], 2, dtype=float)
    y = np.arange(0, shape[0], 2, dtype=float)
    field = DisplacementField(
        x_coordinates=x,
        y_coordinates=y,
        displacement_xy=np.ones((len(y), len(x), 2), dtype=float) * [3.0, -2.0],
    )
    tapered = _taper_displacement_field(field, weight)
    assert np.all(tapered.displacement_xy[0, 0] == 0.0)
    center_y = int(np.argmin(np.abs(y - 20.0)))
    center_x = int(np.argmin(np.abs(x - 20.0)))
    np.testing.assert_allclose(
        tapered.displacement_xy[center_y, center_x],
        [3.0, -2.0],
    )


def test_coherent_euclidean_drift_recovers_rotation_and_center_translation() -> None:
    x = np.arange(0.0, 101.0, 10.0)
    y = np.arange(0.0, 81.0, 10.0)
    xx, yy = np.meshgrid(x, y)
    source = np.column_stack([xx.ravel(), yy.ravel()])
    center = source.mean(axis=0)
    angle = np.deg2rad(2.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    center_translation = np.array([3.0, -4.0])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = center + center_translation - rotation @ center
    displacement = (apply_affine_matrix(source, matrix) - source).reshape(
        len(y), len(x), 2
    )
    field = DisplacementField(
        x_coordinates=x,
        y_coordinates=y,
        displacement_xy=displacement,
    )

    metrics = _coherent_euclidean_drift_diagnostics(
        field,
        active_mask=np.ones((81, 101), dtype=np.uint8) * 255,
        pixel_size_um=2.0,
    )

    assert metrics["coherent_rotation_degrees"] == pytest.approx(2.0)
    assert metrics["coherent_translation_x_um"] == pytest.approx(6.0)
    assert metrics["coherent_translation_y_um"] == pytest.approx(-8.0)
    assert metrics["coherent_translation_magnitude_um"] == pytest.approx(10.0)
    assert metrics["local_residual_p95_um"] == pytest.approx(0.0, abs=1e-10)


def test_identity_map_guard_fails_closed_on_hidden_valis_translation() -> None:
    fixed_slide = object()

    class MovingSlide:
        def warp_xy_from_to(
            self: MovingSlide,
            xy: np.ndarray,
            destination: object,
            **kwargs: Any,
        ) -> np.ndarray:
            del destination, kwargs
            return np.asarray(xy) + np.array([0.01, 0.0])

    with pytest.raises(RuntimeError, match="changed the locked global transform"):
        _assert_valis_rigid_map_is_identity(
            MovingSlide(),
            fixed_slide,
            sample_xy=np.array([[0.0, 0.0], [20.0, 30.0]]),
            shape_rc=(64, 64),
        )


def test_execute_valis_disables_rigid_and_returns_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The VALIS API boundary must be locked, not merely corrected afterward."""
    import merxen.alignment.valis_register as module

    captured: dict[str, Any] = {}

    class FakeMatcher:
        def __init__(
            self: FakeMatcher,
            *,
            feature_detector: Any,
            **kwargs: Any,
        ) -> None:
            del kwargs
            self.feature_detector = feature_detector
            self.lg_matcher = lambda *args, **kwargs: (args, kwargs)

        def match_images(
            self: FakeMatcher,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Any, Any]:
            return args, kwargs

    class FakeSlide:
        def __init__(self: FakeSlide, *, moving: bool) -> None:
            self.moving = moving
            self.fwd_dxdy = np.ones((2, 2, 2)) if moving else None

        def warp_xy_from_to(
            self: FakeSlide,
            xy: np.ndarray,
            destination: FakeSlide,
            *,
            non_rigid: bool,
            **kwargs: Any,
        ) -> np.ndarray:
            del kwargs
            points = np.asarray(xy, dtype=np.float64)
            if not non_rigid:
                return points.copy()
            if self.moving and not destination.moving:
                return points + np.array([0.25, -0.15])
            if not self.moving and destination.moving:
                return points - np.array([0.25, -0.15])
            raise AssertionError("Unexpected fake slide direction")

    class FakeValis:
        def __init__(
            self: FakeValis,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            del args
            captured.update(kwargs)
            self.moving_slide = FakeSlide(moving=True)
            self.fixed_slide = FakeSlide(moving=False)
            self.non_rigid_reg_kwargs = {}

        def rigid_register_partial(
            self: FakeValis,
            *args: Any,
            **kwargs: Any,
        ) -> object:
            del args, kwargs
            return object()

        def register(
            self: FakeValis,
            **kwargs: Any,
        ) -> tuple[object, object, pd.DataFrame]:
            captured["register_kwargs"] = kwargs
            return object(), object(), pd.DataFrame([{"status": "ok"}])

        def get_slide(self: FakeValis, path: str) -> FakeSlide:
            return (
                self.moving_slide
                if Path(path).name.startswith("1_")
                else self.fixed_slide
            )

    fake_registration = SimpleNamespace(
        Valis=FakeValis,
        NON_RIGID_REG_CLASS_KEY="non_rigid_registrar_cls",
        kill_jvm=lambda: None,
    )
    fake_feature_matcher = SimpleNamespace(LightGlueMatcher=FakeMatcher)
    fake_valis = ModuleType("valis")
    fake_valis.feature_detectors = SimpleNamespace()
    fake_valis.feature_matcher = fake_feature_matcher
    fake_valis.non_rigid_registrars = SimpleNamespace(OpticalFlowWarper=object)
    fake_valis.registration = fake_registration
    monkeypatch.setitem(sys.modules, "valis", fake_valis)
    monkeypatch.setattr(module, "_resolve_torch_device", lambda device: "cpu")
    monkeypatch.setattr(
        module,
        "_create_disk_detector",
        lambda feature_detectors, num_features, device: object(),
    )
    monkeypatch.setattr(
        module,
        "_match_registered_features",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        module,
        "_ensure_registrar_pickle",
        lambda registrar, output_dir: Path(output_dir) / "registrar.pickle",
    )

    shape = (64, 64)
    yy, xx = np.indices(shape)
    image = np.clip(20 + xx + yy, 0, 255).astype(np.uint8)
    tissue = np.zeros(shape, dtype=np.uint8)
    tissue[8:56, 9:55] = 255
    valid = np.zeros(shape, dtype=np.uint8)
    valid[5:59, 6:58] = 255
    frame = RegistrationFrame(
        platform="XENIUM",
        image_key="dapi",
        original_shape_rc=shape,
        registration_shape_rc=shape,
        dataset_to_image_matrix=np.eye(3),
        original_to_registration_matrix=np.eye(3),
        registration_pixel_size_um=1.0,
        processed_image=image,
        tissue_mask=tissue,
        support_mask=np.ones(shape, dtype=np.uint8) * 255,
        valid_mask=valid,
        edge_artifact_metrics={},
        coordinate_metadata_source="test",
        coordinate_metadata_trusted=True,
    )
    shared = _shared_non_rigid_domain(
        fixed_tissue_mask=tissue,
        moving_tissue_mask=tissue,
        fixed_valid_mask=valid,
        moving_valid_mask=valid,
    )
    weight = _feather_binary_mask(shared, taper_px=6.0)
    fixed_path = tmp_path / "0_fixed_dapi.tif"
    moving_path = tmp_path / "1_moving_dapi_preoriented.tif"

    attempt = _execute_valis(
        fixed_path=fixed_path,
        moving_path=moving_path,
        fixed_frame=frame,
        moving_pre_image=image,
        moving_pre_mask=tissue,
        moving_pre_valid=valid,
        shared_non_rigid_mask=shared,
        shared_non_rigid_weight=weight,
        config=ValisAlignmentConfig(),
        output_dir=tmp_path / "valis",
    )

    assert captured["do_rigid"] is False
    assert captured["transformer_cls"] is _LockedIdentityTransform
    assert captured["affine_optimizer_cls"] is None
    assert captured["micro_rigid_registrar_cls"] is None
    assert captured["crop_for_rigid_reg"] is False
    assert captured["reference_img_f"] == str(fixed_path)
    assert str(fixed_path) in captured["img_list"]
    np.testing.assert_array_equal(attempt.global_matrix, np.eye(3))
    np.testing.assert_array_equal(attempt.global_image, image)
    np.testing.assert_array_equal(attempt.global_mask, tissue)
    assert attempt.metadata["global_transform_locked"] is True
    assert attempt.metadata["valis_do_rigid"] is False
    assert attempt.forward_displacement is not None
    assert attempt.backward_displacement is not None
    assert not _contains_ndarray(attempt.metadata)


def _contains_ndarray(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_ndarray(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_ndarray(item) for item in value)
    return False
