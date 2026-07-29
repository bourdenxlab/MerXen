"""Focused tests for non-rigid registration selection safeguards."""

from __future__ import annotations

import pytest

from merxen.alignment.qc import select_non_rigid_result
from merxen.config import AlignmentQCThresholds


def _deformation_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "displacement_p95_um": 10.0,
        "jacobian_nonpositive_fraction": 0.0,
        "jacobian_p01": 0.9,
        "jacobian_p99": 1.1,
        "coherent_rotation_degrees": 0.0,
        "coherent_translation_magnitude_um": 0.0,
    }
    metrics.update(overrides)
    return metrics


def _image_metrics(
    *,
    normalized_mutual_information: float,
    density_correlation: float = 0.8,
    tissue_dice: float = 0.8,
    partial_overlap_robust_score: float = 0.7,
) -> dict[str, float]:
    return {
        "normalized_mutual_information": normalized_mutual_information,
        "density_correlation": density_correlation,
        "tissue_dice": tissue_dice,
        "partial_overlap_robust_score": partial_overlap_robust_score,
    }


@pytest.mark.parametrize(
    ("deformation_overrides", "expected_reason"),
    [
        (
            {"coherent_rotation_degrees": 0.3},
            "non_rigid_coherent_rotation_too_large",
        ),
        (
            {"coherent_translation_magnitude_um": 30.0},
            "non_rigid_coherent_translation_too_large",
        ),
    ],
)
def test_non_rigid_selection_rejects_coherent_global_drift(
    deformation_overrides: dict[str, float],
    expected_reason: str,
) -> None:
    """A dense field must not recreate rotation or translation globally."""
    selected, reasons = select_non_rigid_result(
        _image_metrics(normalized_mutual_information=0.1),
        _image_metrics(normalized_mutual_information=0.12),
        _deformation_metrics(**deformation_overrides),
        thresholds=AlignmentQCThresholds(),
    )

    assert selected is False
    assert expected_reason in reasons


@pytest.mark.parametrize(
    ("metric_override", "expected_reason"),
    [
        (
            {"density_correlation": 0.79},
            "non_rigid_density_correlation_degraded",
        ),
        ({"tissue_dice": 0.78}, "non_rigid_tissue_dice_degraded"),
        (
            {"partial_overlap_robust_score": 0.69},
            "non_rigid_robust_score_degraded",
        ),
    ],
)
def test_non_rigid_selection_rejects_morphology_degradation(
    metric_override: dict[str, float],
    expected_reason: str,
) -> None:
    """NMI improvement cannot compensate for worse authoritative morphology."""
    selected, reasons = select_non_rigid_result(
        _image_metrics(normalized_mutual_information=0.1),
        _image_metrics(
            normalized_mutual_information=0.12,
            **metric_override,
        ),
        _deformation_metrics(),
        thresholds=AlignmentQCThresholds(),
    )

    assert selected is False
    assert expected_reason in reasons


def test_non_rigid_selection_accepts_local_improvement_without_global_drift() -> None:
    """A locally improved, topology-preserving field remains selectable."""
    selected, reasons = select_non_rigid_result(
        _image_metrics(normalized_mutual_information=0.1),
        _image_metrics(
            normalized_mutual_information=0.12,
            density_correlation=0.82,
            tissue_dice=0.795,
            partial_overlap_robust_score=0.701,
        ),
        _deformation_metrics(
            coherent_rotation_degrees=0.1,
            coherent_translation_magnitude_um=12.0,
        ),
        thresholds=AlignmentQCThresholds(),
    )

    assert selected is True
    assert reasons == []


def test_non_rigid_selection_accepts_cross_platform_scale_nmi_gain() -> None:
    """Small absolute NMI changes can be meaningful for adjacent platforms."""
    selected, reasons = select_non_rigid_result(
        _image_metrics(
            normalized_mutual_information=0.0004067,
            density_correlation=0.5493,
            partial_overlap_robust_score=0.6901,
        ),
        _image_metrics(
            normalized_mutual_information=0.0005011,
            density_correlation=0.5591,
            partial_overlap_robust_score=0.6926,
        ),
        _deformation_metrics(
            coherent_rotation_degrees=-0.07,
            coherent_translation_magnitude_um=10.2,
        ),
        thresholds=AlignmentQCThresholds(),
    )

    assert selected is True
    assert reasons == []
