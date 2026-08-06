"""Unit tests for DAPI-only VALIS alignment components."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tifffile

from merxen.alignment import frames as alignment_frames
from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
from merxen.alignment.dapi import (
    create_dapi_tissue_mask,
    create_registration_validity_mask,
    dapi_edge_artifact_metrics,
    process_dapi_image,
    select_dapi_channel,
    tissue_fragment_count,
)
from merxen.alignment.frames import resolve_dapi_frame
from merxen.alignment.orientation import (
    OrientationResult,
    _best_distinct_by_handedness,
    _distinct_local_peaks,
    _local_maximum_indices,
    _LocalFinePeak,
    _requested_reflections,
    _select_orientation_candidate,
    estimate_pre_orientation,
    warp_image,
)
from merxen.alignment.partial_overlap import refine_partial_overlap_rigid
from merxen.alignment.qc import (
    affine_diagnostics,
    compute_dapi_metrics,
    displacement_diagnostics,
    morphology_supported_global_qc_passes,
    rigid_qc_passes,
    select_non_rigid_result,
)
from merxen.alignment.register import register_pair
from merxen.alignment.transforms import apply_affine_matrix, fit_affine_matrix
from merxen.alignment.valis_register import (
    _HostTensorResultProxy,
    _preorientation_fallback_attempt,
    _valis_lightglue_cuda_compatibility,
)
from merxen.config import (
    AlignmentConfig,
    AlignmentImageConfig,
    AlignmentQCThresholds,
    DAPIProcessingConfig,
    OrientationSearchConfig,
    PartialOverlapRigidConfig,
)


def test_alignment_config_defaults_to_valis_with_validated_reflection_search(
    tmp_path: Path,
) -> None:
    """Paired-platform defaults must search, but not blindly force, reflections."""
    config = AlignmentConfig(
        pair_id="pair",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=tmp_path / "align_out",
    )

    assert config.backend == "valis"
    assert config.fixed_platform == "XENIUM"
    assert config.moving_platform == "MERSCOPE"
    assert config.valis.global_transform == "rigid"
    assert config.valis.preprocessing.background_boundary_mode == "mirror"
    assert config.valis.partial_overlap.enabled is True
    assert config.valis.orientation.allow_reflection is True
    assert config.valis.orientation.reflection_minimum_score_improvement == 0.01
    assert config.valis.orientation.local_fine_angle_radius_degrees == 2.5
    assert config.valis.orientation.local_fine_translation_radius_um == 500.0


def test_dapi_preprocessing_selects_named_channel_and_returns_uint8() -> None:
    """DAPI processing must not depend on multichannel array order."""
    y, x = np.indices((96, 112))
    dapi = 10.0 + 0.03 * x + 0.02 * y
    dapi += 200.0 * np.exp(-((x - 55) ** 2 + (y - 47) ** 2) / 300.0)
    dapi[0, 0] = np.nan
    channels = np.stack([np.ones_like(dapi) * 50.0, dapi], axis=0)

    selected = select_dapi_channel(
        channels,
        channel_names=["PolyT", "DAPI"],
        dapi_channel="dapi",
    )
    processed = process_dapi_image(
        selected,
        pixel_size_um=1.0,
        config=DAPIProcessingConfig(
            background_sigma_um=15.0,
            smoothing_sigma_um=1.0,
        ),
    )

    assert processed.dtype == np.uint8
    assert processed.shape == dapi.shape
    assert np.isfinite(processed).all()
    with pytest.raises(ValueError, match="exactly one"):
        select_dapi_channel(channels, channel_names=["A", "B"])


def test_dapi_preprocessing_tapers_rectangular_edge_artifacts() -> None:
    """Registration-only processing must not preserve a bright source footprint."""
    y, x = np.indices((128, 160))
    image = 50.0 + 0.4 * x
    image += 200.0 * np.exp(-((x - 145) ** 2 + (y - 64) ** 2) / 500.0)
    untapered = process_dapi_image(
        image,
        pixel_size_um=1.0,
        config=DAPIProcessingConfig(
            background_sigma_um=20.0,
            background_boundary_mode="nearest",
            edge_taper_um=0.0,
            smoothing_sigma_um=0.0,
        ),
    )
    processed = process_dapi_image(
        image,
        pixel_size_um=1.0,
        config=DAPIProcessingConfig(
            background_sigma_um=20.0,
            background_boundary_mode="mirror",
            edge_taper_um=24.0,
            smoothing_sigma_um=0.0,
        ),
    )
    untapered_metrics = dapi_edge_artifact_metrics(untapered)
    processed_metrics = dapi_edge_artifact_metrics(processed)
    valid = create_registration_validity_mask(
        image.shape,
        pixel_size_um=1.0,
        edge_exclusion_um=16.0,
    )

    assert processed_metrics["edge_to_interior_p95_ratio"] < 0.25
    assert (
        processed_metrics["edge_to_interior_p95_ratio"]
        < untapered_metrics["edge_to_interior_p95_ratio"]
    )
    assert not np.any(valid[:16])
    assert np.all(valid[16:-16, 16:-16] == 255)


def test_dapi_tissue_mask_retains_multiple_substantial_fragments() -> None:
    """Masking must retain disconnected tissue instead of only the largest piece."""
    image = np.zeros((160, 180), dtype=np.uint8)
    cv2.circle(image, (45, 70), 28, 180, -1)
    cv2.ellipse(image, (130, 95), (30, 18), 20, 0, 360, 220, -1)
    config = DAPIProcessingConfig(
        mask_downsample=2,
        mask_smoothing_sigma_um=2.0,
        mask_closing_radius_um=2.0,
        mask_min_area_um2=100.0,
        mask_hole_area_um2=100.0,
        mask_dilation_um=0.0,
    )

    mask = create_dapi_tissue_mask(image, pixel_size_um=1.0, config=config)

    assert mask.dtype == np.uint8
    assert tissue_fragment_count(mask) == 2


def test_partial_overlap_rigid_refinement_recovers_rotation_and_translation() -> None:
    """Unmatched tissue must not pull the rigid fit toward maximum mask overlap."""
    height = width = 192
    fixed = np.zeros((height, width), dtype=np.uint8)
    fixed_mask = np.zeros_like(fixed)
    cv2.ellipse(fixed_mask, (88, 98), (62, 42), 12, 0, 360, 255, -1)
    cv2.circle(fixed_mask, (161, 72), 28, 255, -1)
    cv2.circle(fixed, (58, 84), 8, 220, -1)
    cv2.circle(fixed, (91, 112), 11, 180, -1)
    cv2.rectangle(fixed, (112, 64), (126, 98), 150, -1)
    cv2.line(fixed, (43, 120), (135, 103), 100, 5)
    fixed = cv2.GaussianBlur(fixed, (0, 0), 2.0)
    fixed = np.where(fixed_mask > 0, fixed + 25, 0).astype(np.uint8)

    shared_target_mask = fixed_mask.copy()
    cv2.circle(shared_target_mask, (161, 72), 30, 0, -1)
    shared_target = np.where(shared_target_mask > 0, fixed, 0).astype(np.uint8)
    expected = np.vstack(
        [
            cv2.getRotationMatrix2D(
                ((width - 1.0) / 2.0, (height - 1.0) / 2.0),
                5.0,
                1.0,
            ),
            [0.0, 0.0, 1.0],
        ]
    )
    expected[:2, 2] += np.array([9.0, -7.0])
    moving = warp_image(
        shared_target,
        np.linalg.inv(expected),
        output_shape_rc=fixed.shape,
    )
    moving_mask = warp_image(
        shared_target_mask,
        np.linalg.inv(expected),
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    result = refine_partial_overlap_rigid(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        initial=OrientationResult(
            matrix=np.eye(3),
            method="synthetic_initial",
            angle_degrees=0.0,
            scale=1.0,
            score=0.0,
            metrics={"reflection": False},
        ),
        config=PartialOverlapRigidConfig(
            max_dimension_px=192,
            angle_search_radius_degrees=8.0,
            coarse_angle_step_degrees=2.0,
            maximum_translation_um=40.0,
            retained_boundary_fraction=0.65,
            boundary_distance_scale_um=10.0,
            density_sigma_um=4.0,
            minimum_fixed_overlap_fraction=0.5,
            minimum_moving_overlap_fraction=0.7,
            boundary_weight=0.45,
            density_weight=0.55,
            candidates_to_refine=3,
            optimizer_max_iterations=35,
            minimum_score_improvement=0.0,
        ),
        pixel_size_um=1.0,
    )
    landmarks = np.array([[58.0, 84.0], [91.0, 112.0], [119.0, 78.0]])

    recovered = apply_affine_matrix(landmarks, result.matrix)
    target = apply_affine_matrix(landmarks, expected)

    assert result.method == "partial_overlap_rigid"
    assert result.metrics["partial_overlap"]["accepted"] is True
    assert np.linalg.det(result.matrix[:2, :2]) == pytest.approx(1.0, abs=1e-8)
    np.testing.assert_allclose(recovered, target, atol=2.5)


def test_partial_overlap_rigid_refinement_retains_an_aligned_baseline(
    tmp_path: Path,
) -> None:
    """An already aligned full-overlap pair must not acquire a spurious transform."""
    image = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(image, (61, 70), (42, 25), 17, 0, 360, 100, -1)
    cv2.circle(image, (42, 59), 7, 230, -1)
    cv2.rectangle(image, (75, 70), (89, 90), 180, -1)
    mask = (image > 0).astype(np.uint8) * 255
    initial = OrientationResult(
        matrix=np.eye(3),
        method="already_aligned",
        angle_degrees=0.0,
        scale=1.0,
        score=1.0,
        metrics={"reflection": False},
    )

    result = refine_partial_overlap_rigid(
        image,
        image,
        mask,
        mask,
        initial=initial,
        config=PartialOverlapRigidConfig(
            max_dimension_px=128,
            angle_search_radius_degrees=4.0,
            coarse_angle_step_degrees=2.0,
            maximum_translation_um=20.0,
            density_sigma_um=3.0,
            boundary_distance_scale_um=8.0,
            candidates_to_refine=2,
            optimizer_max_iterations=20,
            minimum_score_improvement=0.01,
        ),
        pixel_size_um=1.0,
        output_dir=tmp_path / "partial_overlap_qc",
    )

    np.testing.assert_allclose(result.matrix, np.eye(3), atol=1e-12)
    assert result.metrics["partial_overlap"]["accepted"] is False
    assert (
        result.metrics["partial_overlap"]["selection_reason"]
        == "improvement_below_minimum"
    )
    assert (tmp_path / "partial_overlap_qc/partial_overlap_metrics.json").exists()
    assert (tmp_path / "partial_overlap_qc/partial_overlap_candidates.png").exists()
    assert (tmp_path / "partial_overlap_qc/partial_overlap_objective.png").exists()


@pytest.mark.parametrize("angle_degrees", [37.0, 143.0, 278.0])
def test_fallback_orientation_recovers_arbitrary_rotation(
    angle_degrees: float,
) -> None:
    """Full-circle fallback must recover non-right-angle orientations."""
    height = width = 160
    fixed = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(fixed, (65, 80), (36, 16), 25, 0, 360, 180, -1)
    cv2.circle(fixed, (108, 45), 9, 255, -1)
    cv2.rectangle(fixed, (37, 108), (59, 126), 120, -1)
    fixed_mask = (fixed > 0).astype(np.uint8) * 255
    expected = np.vstack(
        [
            cv2.getRotationMatrix2D(
                ((width - 1) / 2.0, (height - 1) / 2.0),
                angle_degrees,
                1.0,
            ),
            [0.0, 0.0, 1.0],
        ]
    )
    expected[:2, 2] += np.array([5.0, -4.0])
    moving = warp_image(
        fixed,
        np.linalg.inv(expected),
        output_shape_rc=fixed.shape,
    )
    moving_mask = warp_image(
        fixed_mask,
        np.linalg.inv(expected),
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    config = OrientationSearchConfig(
        max_dimension_px=160,
        minimum_inliers=1_000,
        coarse_step_degrees=10.0,
        refine_step_degrees=2.0,
        final_step_degrees=0.5,
        candidates_to_refine=3,
        minimum_dice=0.05,
        local_fine_search_enabled=False,
    )

    result = estimate_pre_orientation(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        config=config,
    )
    landmarks = np.array([[65.0, 80.0], [108.0, 45.0], [40.0, 115.0]])
    homogeneous = np.column_stack([landmarks, np.ones(len(landmarks))])
    recovered = (homogeneous @ result.matrix.T)[:, :2]
    target = (homogeneous @ expected.T)[:, :2]

    assert result.method == "joint_angular_translation_search"
    assert result.metrics["reflection"] is False
    assert float(result.metrics["dice"]) > 0.95
    np.testing.assert_allclose(recovered, target, atol=2.0)


def test_reflection_search_is_enabled_by_default_with_a_score_margin() -> None:
    """A mirrored candidate must win by a non-zero margin before selection."""
    config = OrientationSearchConfig()
    assert config.allow_reflection is True
    assert config.reflection_minimum_score_improvement > 0.0


def test_joint_search_retains_independent_handedness_beams() -> None:
    """A strong branch must not consume the other handedness candidate beam."""
    candidates = [
        OrientationResult(
            matrix=np.eye(3),
            method="angular_search",
            angle_degrees=float(index),
            scale=1.0,
            score=1.0 - index * 0.01,
            metrics={
                "reflection": reflected,
                "translation_x_px": float(index),
                "translation_y_px": 0.0,
                "eligible": True,
            },
        )
        for index, reflected in enumerate([False, False, False, True])
    ]

    retained = _best_distinct_by_handedness(
        candidates,
        count=2,
        reflections=[False, True],
    )

    assert sum(not candidate.metrics["reflection"] for candidate in retained) == 2
    assert sum(candidate.metrics["reflection"] for candidate in retained) == 1


def test_handedness_near_tie_proceeds_with_provisional_ambiguity() -> None:
    """A near tie must proceed with an explicit symmetric ambiguity flag."""
    non_reflected = OrientationResult(
        matrix=np.eye(3),
        method="angular_search",
        angle_degrees=0.0,
        scale=1.0,
        score=0.5,
        metrics={"reflection": False},
    )
    reflected = OrientationResult(
        matrix=np.diag([-1.0, 1.0, 1.0]),
        method="angular_search",
        angle_degrees=0.0,
        scale=1.0,
        score=0.505,
        metrics={"reflection": True},
    )

    result = _select_orientation_candidate(
        [non_reflected, reflected],
        config=OrientationSearchConfig(reflection_minimum_score_improvement=0.01),
    )

    assert result.metrics["reflection"] is True
    assert result.metrics["handedness_ambiguous"] is True
    assert result.metrics["provisional_selection"] is True
    assert result.metrics["selection_reason"] == (
        "handedness_ambiguous_reflected_provisional"
    )


def test_ineligible_orientation_cannot_beat_an_eligible_candidate() -> None:
    """Eligibility gates must exclude implausible candidates before ranking."""
    eligible = OrientationResult(
        matrix=np.eye(3),
        method="angular_search",
        angle_degrees=0.0,
        scale=1.0,
        score=0.4,
        metrics={"reflection": False, "eligible": True},
    )
    ineligible = OrientationResult(
        matrix=np.diag([-1.0, 1.0, 1.0]),
        method="angular_search",
        angle_degrees=0.0,
        scale=1.0,
        score=0.9,
        metrics={
            "reflection": True,
            "eligible": False,
            "eligibility_reasons": ["moving_tissue_clipped"],
        },
    )

    result = _select_orientation_candidate(
        [eligible, ineligible],
        config=OrientationSearchConfig(),
    )

    assert result.metrics["reflection"] is False
    assert result.metrics["eligibility_fallback"] is False
    assert result.metrics["candidate_comparison"]["reflected"]["eligible"] is False


def test_generic_adjacent_section_fixture_recovers_joint_reflected_transform(
    tmp_path: Path,
) -> None:
    """Joint search must recover a reflected transform with section differences."""
    fixture_path = (
        Path(__file__).parents[1]
        / "test_data/alignment/challenging_reflected_sections.npz"
    )
    with np.load(fixture_path) as fixture:
        result = estimate_pre_orientation(
            fixture["fixed"],
            fixture["moving"],
            fixture["fixed_mask"],
            fixture["moving_mask"],
            config=OrientationSearchConfig(
                max_dimension_px=128,
                minimum_inliers=1_000,
                coarse_step_degrees=15.0,
                refine_step_degrees=3.0,
                final_step_degrees=1.0,
                candidates_to_refine=3,
                coarse_translation_radius_px=24.0,
                refine_translation_radius_px=6.0,
                final_translation_radius_px=2.0,
                minimum_dice=0.1,
                local_fine_angle_radius_degrees=2.0,
                local_fine_translation_radius_um=16.0,
                local_fine_coarse_translation_step_um=4.0,
                local_fine_refine_translation_step_um=1.0,
            ),
            output_dir=tmp_path / "orientation_qc",
        )
        expected_matrix = fixture["expected_matrix"]

    assert result.metrics["reflection"] is True
    assert result.method == "joint_angular_translation_search"
    np.testing.assert_allclose(result.matrix, expected_matrix, atol=1.2)
    local_search = result.metrics["local_fine_search"]
    assert local_search["enabled"] is True
    assert isinstance(local_search["selected_coordinate_stable"], bool)
    assert isinstance(local_search["has_nearby_stable_maximum"], bool)
    assert (tmp_path / "orientation_qc/orientation_candidates.json").exists()
    assert (tmp_path / "orientation_qc/orientation_candidate_overlays.png").exists()
    assert (
        tmp_path / "orientation_qc/orientation_angle_translation_landscape.png"
    ).exists()
    assert (tmp_path / "orientation_qc/orientation_local_fine_search.json").exists()
    assert (tmp_path / "orientation_qc/orientation_local_fine_overlay.png").exists()
    assert (tmp_path / "orientation_qc/orientation_local_fine_landscape.png").exists()


def test_local_fine_search_detects_and_deduplicates_stable_maxima() -> None:
    """Persistent 3D maxima must remain distinct unless refinement converges."""
    volume = np.zeros((5, 5, 5), dtype=np.float64)
    volume[1, 1, 1] = 0.8
    volume[3, 3, 3] = 0.9

    assert _local_maximum_indices(volume)[:2] == [(3, 3, 3), (1, 1, 1)]

    first = _LocalFinePeak(
        candidate=OrientationResult(
            matrix=np.eye(3),
            method="angular_search",
            angle_degrees=10.0,
            scale=1.0,
            score=0.9,
            metrics={"translation_x_px": 2.0, "translation_y_px": 3.0},
        ),
        coarse_index=(3, 3, 3),
        coarse_score=0.9,
        prominence=0.1,
        coarse_interior=True,
        refinement_interior=True,
        refinement_iterations=1,
    )
    converged_duplicate = _LocalFinePeak(
        candidate=OrientationResult(
            matrix=np.eye(3),
            method="angular_search",
            angle_degrees=10.1,
            scale=1.0,
            score=0.89,
            metrics={"translation_x_px": 2.2, "translation_y_px": 3.1},
        ),
        coarse_index=(2, 3, 3),
        coarse_score=0.88,
        prominence=0.05,
        coarse_interior=True,
        refinement_interior=True,
        refinement_iterations=1,
    )

    retained = _distinct_local_peaks(
        [first, converged_duplicate],
        angle_tolerance_degrees=0.5,
        translation_tolerance_px=1.0,
    )

    assert retained == [first]
    assert retained[0].is_stable is True


def test_orientation_manual_overrides_force_handedness_and_seed_translation() -> None:
    """Manual overrides must constrain handedness and enter the joint seed pool."""
    config = OrientationSearchConfig(
        reflection_mode="force",
        initial_angle_degrees=14.0,
        initial_translation_x_um=18.0,
        initial_translation_y_um=-11.0,
    )

    assert config.reflection_mode == "force"
    assert config.initial_angle_degrees == 14.0
    assert config.initial_translation_x_um == 18.0
    assert _requested_reflections(config) == [True]
    assert _requested_reflections(
        OrientationSearchConfig(reflection_mode="forbid")
    ) == [False]
    with pytest.raises(ValueError, match="must be set together"):
        OrientationSearchConfig(initial_translation_x_um=1.0)


def test_fallback_orientation_selects_required_vertical_reflection() -> None:
    """Opposite-handed asymmetric tissue must select the reflected candidate."""
    fixed = np.zeros((160, 180), dtype=np.uint8)
    cv2.ellipse(fixed, (70, 80), (45, 20), 18, 0, 360, 180, -1)
    cv2.circle(fixed, (132, 43), 11, 255, -1)
    cv2.rectangle(fixed, (35, 115), (58, 135), 120, -1)
    fixed_mask = (fixed > 0).astype(np.uint8) * 255
    moving = cv2.flip(fixed, 0)
    moving_mask = cv2.flip(fixed_mask, 0)

    result = estimate_pre_orientation(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        config=OrientationSearchConfig(
            max_dimension_px=180,
            minimum_inliers=1_000,
            coarse_step_degrees=10.0,
            refine_step_degrees=2.0,
            final_step_degrees=0.5,
            candidates_to_refine=4,
            local_fine_search_enabled=False,
        ),
    )

    assert result.metrics["reflection"] is True
    assert result.metrics["selection_reason"] in {
        "reflected_candidate_exceeded_score_margin",
        "only_reflected_candidate_valid",
    }
    assert float(result.metrics["dice"]) > 0.95
    comparison = result.metrics["candidate_comparison"]
    if comparison["non_reflected"] is not None:
        assert comparison["reflected"]["score"] > comparison["non_reflected"]["score"]


def test_frame_resolution_prefers_schema_original_segmentation_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity transcripts must not hide the vendor micron-to-pixel transform."""
    image = SimpleNamespace(matrix=np.eye(3))
    transcripts = SimpleNamespace(matrix=np.eye(3))
    original = SimpleNamespace(
        matrix=np.array([[9.25937, 0.0, 0.0], [0.0, 9.25934, 0.0], [0.0, 0.0, 1.0]])
    )
    sdata_obj = SimpleNamespace(
        attrs={
            "merxen_schema": {
                "segmentations": {"original": {"shape": "merscope_cell_boundaries"}}
            }
        },
        points={"transcripts": transcripts},
        shapes={"merscope_cell_boundaries": original},
        labels={},
    )
    monkeypatch.setattr(
        alignment_frames,
        "_element_affine_to_global",
        lambda element: np.asarray(element.matrix, dtype=np.float64),
    )

    matrix, source, trusted = alignment_frames._resolve_dataset_to_image_matrix(
        sdata_obj,
        image_element=image,
        configured_matrix=None,
        configured_pixel_size_um=None,
        platform="MERSCOPE",
    )

    np.testing.assert_allclose(matrix, original.matrix)
    assert source == "SpatialData element 'merscope_cell_boundaries'"
    assert trusted is True
    assert 1.0 / matrix[0, 0] == pytest.approx(0.108, rel=1e-4)


def test_explicit_pixel_size_overrides_identity_spatialdata_metadata() -> None:
    """A configured pixel size must not conflict with a rewritten identity image."""
    matrix, source, trusted = alignment_frames._resolve_dataset_to_image_matrix(
        SimpleNamespace(attrs={}, points={}, shapes={}, labels={}),
        image_element=SimpleNamespace(),
        configured_matrix=None,
        configured_pixel_size_um=0.2125,
        platform="XENIUM",
    )

    np.testing.assert_allclose(
        matrix,
        np.diag([1.0 / 0.2125, 1.0 / 0.2125, 1.0]),
    )
    assert source == "configured pixel_size_um"
    assert trusted is True


def test_transform_bundle_composes_frames_once_and_chunks_identically(
    tmp_path: Path,
) -> None:
    """T_pre, VALIS, pixel, and physical matrices must not be inverted or doubled."""
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0, 20.0])
    displacement = np.zeros((3, 3, 2), dtype=np.float64)
    displacement[..., 0] = 1.0
    displacement[..., 1] = -2.0
    field = DisplacementField(x, y, displacement)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=np.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        moving_image_to_registration=np.array(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        ),
        pre_matrix=np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]),
        global_matrix=np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]),
        fixed_image_to_registration=np.array(
            [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]]
        ),
        fixed_dataset_to_image=np.array(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        selected_mode="non_rigid",
        forward_displacement=field,
    )
    points = np.array([[0.0, 0.0], [2.0, 3.0], [8.0, 5.0]])
    expected_global = points + np.array([8.0, 3.0])
    expected_non_rigid = expected_global + np.array([1.0, -2.0])

    np.testing.assert_allclose(bundle.transform_global(points), expected_global)
    np.testing.assert_allclose(bundle.transform(points), expected_non_rigid)
    np.testing.assert_allclose(
        bundle.transform(points, chunk_size=1),
        bundle.transform(points),
    )

    outputs = bundle.save(tmp_path)
    reloaded = ValisTransformBundle.load(outputs["transform_chain"])
    np.testing.assert_allclose(reloaded.transform(points), expected_non_rigid)


def test_identity_transform_preserves_xy_without_row_column_swap() -> None:
    """An identity chain must preserve asymmetric X/Y landmarks exactly."""
    identity = np.eye(3, dtype=np.float64)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=identity,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="global",
    )
    points_xy = np.array([[2.0, 17.0], [31.0, 5.0], [43.5, 11.25]])

    np.testing.assert_allclose(bundle.transform(points_xy), points_xy)


def test_affine_fit_recovers_translation_scale_and_mild_shear() -> None:
    """Global affine helpers must recover modest section-scale distortion."""
    source_xy = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 80.0], [100.0, 80.0], [37.0, 29.0]]
    )
    expected_matrix = np.array(
        [[1.08, 0.06, 13.0], [-0.03, 0.96, -7.0], [0.0, 0.0, 1.0]]
    )
    target_xy = apply_affine_matrix(source_xy, expected_matrix)

    recovered = fit_affine_matrix(source_xy, target_xy)
    diagnostics = affine_diagnostics(recovered)

    np.testing.assert_allclose(recovered, expected_matrix, atol=1e-10)
    assert 1.0 < diagnostics["determinant"] < 1.1
    assert abs(diagnostics["shear"]) < 0.1


def test_valis_lightglue_cuda_results_are_copied_to_host() -> None:
    """VALIS' NumPy conversion must receive detached CPU tensors."""
    torch = pytest.importorskip("torch")

    class FakeLightGlue:
        marker = "preserved"

        def __call__(self: FakeLightGlue) -> tuple[object, object]:
            return (
                torch.tensor([0.25], requires_grad=True),
                torch.tensor([[0, 1]]),
            )

    proxy = _HostTensorResultProxy(FakeLightGlue())
    distances, indices = proxy()

    assert distances.device.type == "cpu"
    assert indices.device.type == "cpu"
    assert distances.requires_grad is False
    assert proxy.marker == "preserved"
    np.testing.assert_allclose(distances.numpy(), [0.25])


def test_valis_lightglue_compatibility_is_scoped_and_restored() -> None:
    """The VALIS method patch must only remain active during registration."""
    torch = pytest.importorskip("torch")

    class FakeMatcher:
        def __init__(self: FakeMatcher) -> None:
            self.lg_matcher = lambda: torch.tensor([1.0], requires_grad=True)

        def match_images(self: FakeMatcher) -> object:
            return self.lg_matcher()

    original_method = FakeMatcher.match_images
    matcher = FakeMatcher()
    with _valis_lightglue_cuda_compatibility(FakeMatcher):
        output = matcher.match_images()
        assert output.device.type == "cpu"
        assert output.requires_grad is False

    assert FakeMatcher.match_images is original_method


def test_dapi_metrics_report_partial_tissue_cropping() -> None:
    """QC must retain asymmetric overlap fractions for partially missing tissue."""
    fixed = np.zeros((80, 100), dtype=np.uint8)
    fixed[10:70, 10:90] = 80 + np.tile(np.arange(80, dtype=np.uint8), (60, 1))
    moving = fixed.copy()
    moving[:, :30] = 0
    fixed_mask = fixed > 0
    moving_mask = moving > 0

    metrics = compute_dapi_metrics(fixed, moving, fixed_mask, moving_mask)

    assert 0.0 < metrics["tissue_dice"] < 1.0
    assert metrics["moving_overlap_fraction"] == pytest.approx(1.0)
    assert metrics["fixed_overlap_fraction"] == pytest.approx(0.75)


def test_smooth_non_rigid_field_maps_known_landmarks() -> None:
    """A smooth sampled deformation must interpolate known landmark offsets."""
    x = np.linspace(0.0, 100.0, 6)
    y = np.linspace(0.0, 80.0, 5)
    xx, yy = np.meshgrid(x, y)
    displacement = np.stack([0.02 * xx, -0.01 * yy], axis=-1)
    field = DisplacementField(x, y, displacement)
    identity = np.eye(3, dtype=np.float64)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=identity,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="non_rigid",
        forward_displacement=field,
    )
    points = np.array([[20.0, 20.0], [50.0, 40.0], [80.0, 60.0]])
    expected = points + np.column_stack([0.02 * points[:, 0], -0.01 * points[:, 1]])

    np.testing.assert_allclose(bundle.transform(points), expected, atol=1e-12)


def test_non_rigid_qc_rejects_folding_and_keeps_global() -> None:
    """Optimizer completion alone must not select an implausible deformation."""
    x = np.linspace(0.0, 100.0, 8)
    y = np.linspace(0.0, 100.0, 8)
    xx, _ = np.meshgrid(x, y)
    displacement = np.zeros((len(y), len(x), 2), dtype=np.float64)
    displacement[..., 0] = -2.0 * xx
    field = DisplacementField(x, y, displacement)
    diagnostics = displacement_diagnostics(field, pixel_size_um=1.0)

    selected, reasons = select_non_rigid_result(
        {"normalized_mutual_information": 1.0},
        {"normalized_mutual_information": 1.2},
        diagnostics,
        thresholds=AlignmentQCThresholds(),
    )

    assert selected is False
    assert "non_rigid_folding" in reasons


def test_morphology_supported_qc_accepts_reflected_cross_platform_dapi() -> None:
    """Strong morphology may substitute for cell-exact features under safeguards."""
    image_metrics = {
        "tissue_dice": 0.81,
        "tissue_iou": 0.68,
        "density_correlation": 0.49,
        "fixed_overlap_fraction": 0.94,
        "moving_overlap_fraction": 0.71,
        "normalized_mutual_information": 0.001,
        "feature_inliers": 0,
        "feature_inlier_coverage": 0.0,
    }
    affine_metrics = affine_diagnostics(
        np.array([[1.01, 0.01, 4.0], [-0.01, 1.01, -2.0], [0.0, 0.0, 1.0]])
    )

    passed, reasons = morphology_supported_global_qc_passes(
        image_metrics,
        affine_metrics,
        preorientation_metrics={"tissue_dice": 0.813},
        thresholds=AlignmentQCThresholds(),
        trusted_coordinate_metadata=True,
        reflection_selected=True,
    )

    assert passed is True
    assert reasons == []


def test_morphology_supported_qc_accepts_locked_nonreflected_preorientation() -> None:
    """A locked morphology transform must not require an unnecessary reflection."""
    image_metrics = {
        "tissue_dice": 0.81,
        "tissue_iou": 0.68,
        "density_correlation": 0.49,
        "fixed_overlap_fraction": 0.94,
        "moving_overlap_fraction": 0.71,
        "normalized_mutual_information": 0.001,
        "feature_inliers": 0,
        "feature_inlier_coverage": 0.0,
    }
    affine_metrics = affine_diagnostics(np.eye(3, dtype=np.float64))

    passed, reasons = morphology_supported_global_qc_passes(
        image_metrics,
        affine_metrics,
        preorientation_metrics={"tissue_dice": 0.813},
        thresholds=AlignmentQCThresholds(),
        trusted_coordinate_metadata=True,
        reflection_selected=False,
        authoritative_preorientation_locked=True,
    )

    assert passed is True
    assert reasons == []


def test_morphology_supported_qc_rejects_unlocked_nonreflected_fallback() -> None:
    """Weak feature evidence still needs reflection or an authoritative lock."""
    image_metrics = {
        "tissue_dice": 0.81,
        "tissue_iou": 0.68,
        "density_correlation": 0.49,
        "fixed_overlap_fraction": 0.94,
        "moving_overlap_fraction": 0.71,
    }

    passed, reasons = morphology_supported_global_qc_passes(
        image_metrics,
        affine_diagnostics(np.eye(3, dtype=np.float64)),
        preorientation_metrics={"tissue_dice": 0.813},
        thresholds=AlignmentQCThresholds(),
        trusted_coordinate_metadata=True,
        reflection_selected=False,
    )

    assert passed is False
    assert "reflection_not_selected" in reasons


def test_morphology_supported_qc_rejects_unexpected_second_reflection() -> None:
    """The VALIS refinement may not introduce another hidden reflection."""
    image_metrics = {
        "tissue_dice": 0.81,
        "tissue_iou": 0.68,
        "density_correlation": 0.49,
        "fixed_overlap_fraction": 0.94,
        "moving_overlap_fraction": 0.71,
    }
    affine_metrics = affine_diagnostics(
        np.array([[-1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )

    passed, reasons = morphology_supported_global_qc_passes(
        image_metrics,
        affine_metrics,
        preorientation_metrics={"tissue_dice": 0.813},
        thresholds=AlignmentQCThresholds(),
        trusted_coordinate_metadata=True,
        reflection_selected=True,
    )

    assert passed is False
    assert "unexpected_reflection_in_valis_refinement" in reasons


def test_rigid_qc_rejects_similarity_scaling() -> None:
    """A similarity matrix with uniform scale is not a valid rigid refinement."""
    angle = np.deg2rad(4.4)
    scale = 1.1267
    matrix = np.array(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), -215.0],
            [scale * np.sin(angle), scale * np.cos(angle), -298.0],
            [0.0, 0.0, 1.0],
        ]
    )

    passed, reasons = rigid_qc_passes(
        affine_diagnostics(matrix),
        thresholds=AlignmentQCThresholds(),
    )

    assert passed is False
    assert "rigid_determinant_deviates_from_one" in reasons
    assert "rigid_scale_detected" in reasons


def test_preorientation_fallback_is_an_identity_valis_refinement(
    tmp_path: Path,
) -> None:
    """A failed VALIS fit must not distort a validated coarse orientation."""
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    mask = (image > 8).astype(np.uint8)
    metrics = {
        "tissue_dice": 0.81,
        "tissue_iou": 0.68,
        "density_correlation": 0.49,
        "fixed_overlap_fraction": 0.94,
        "moving_overlap_fraction": 0.71,
    }

    attempt = _preorientation_fallback_attempt(
        moving_pre_image=image,
        moving_pre_mask=mask,
        preorientation_metrics=metrics,
        moving_feature_xy=None,
        fixed_feature_xy=None,
        registrar_path=tmp_path / "failed_valis_registrar.pickle",
        failed_valis_reasons=["morphology_affine_shear_too_large"],
    )

    np.testing.assert_array_equal(attempt.global_matrix, np.eye(3))
    np.testing.assert_array_equal(attempt.global_image, image)
    np.testing.assert_array_equal(attempt.global_mask, mask)
    assert attempt.transform_name == "PreorientationIdentity"
    assert attempt.forward_displacement is None
    assert attempt.metadata["fallback_after_valis_qc_reasons"] == [
        "morphology_affine_shear_too_large"
    ]


def test_external_dapi_fails_cleanly_without_coordinate_metadata(
    tmp_path: Path,
) -> None:
    """An image path alone cannot establish dataset physical coordinates."""
    image_path = tmp_path / "dapi.tif"
    tifffile.imwrite(image_path, np.ones((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError, match="Cannot reliably determine"):
        resolve_dapi_frame(
            SimpleNamespace(images={}, points={}, shapes={}, labels={}),
            platform="MERSCOPE",
            config=AlignmentImageConfig(image_path=image_path),
        )


def test_register_pair_resumes_complete_parameter_compatible_bundle(
    tmp_path: Path,
) -> None:
    """Direct reruns should load a finished transform without reopening DAPI."""
    output_dir = tmp_path / "align_out"
    config = AlignmentConfig(
        pair_id="pair",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=output_dir,
    )
    identity = np.eye(3, dtype=np.float64)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]),
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="global",
    )
    bundle.save(output_dir)
    np.save(
        output_dir / "shared_tissue_mask_registration.npy",
        np.ones((8, 8), dtype=np.uint8),
    )
    (output_dir / "registration_summary.json").write_text(
        json.dumps(
            {
                "backend": "valis",
                "status": "global_only",
                "selected_mode": "global",
                "parameters": config.valis.model_dump(mode="json"),
                "coordinate_frames": {
                    "fixed_platform": "XENIUM",
                    "moving_platform": "MERSCOPE",
                },
            }
        )
    )
    (output_dir / "resume_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "valis",
                "pair_id": "pair",
                "merscope_zarr_path": str((tmp_path / "merscope.zarr").resolve()),
                "xenium_zarr_path": str((tmp_path / "xenium.zarr").resolve()),
                "fixed_platform": "XENIUM",
                "moving_platform": "MERSCOPE",
                "parameters": config.valis.model_dump(mode="json"),
            }
        )
    )

    result = register_pair(SimpleNamespace(), SimpleNamespace(), config)
    assert result.valis_transform is not None
    transformed = result.valis_transform.transform([[1.0, 2.0]])

    assert result.metadata["status"] == "global_only"
    np.testing.assert_allclose(transformed, [[5.0, -1.0]])
