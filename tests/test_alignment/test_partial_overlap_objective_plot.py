"""Focused tests for partial-overlap objective-search diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from merxen.alignment import partial_overlap
from merxen.alignment.partial_overlap import (
    _Candidate,
    _coarse_candidates,
    _local_objective_slice,
    _plot_objective_diagnostics,
    evaluate_aligned_partial_overlap_objective,
)
from merxen.config import PartialOverlapRigidConfig


class _QuadraticObjective:
    """Small deterministic objective exposing the interface used by QC helpers."""

    def __init__(self: _QuadraticObjective) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        image[10:22, 8:20] = 1.0
        self.fixed_density = image
        self.moving_density = image.copy()
        self.fixed_mask = image > 0
        self.moving_mask = self.fixed_mask.copy()
        self.fixed_valid = np.ones_like(self.fixed_mask)
        self.moving_valid = np.ones_like(self.fixed_mask)
        self.shape_rc = image.shape
        self.evaluation_count = 0

    def matrix(self: _QuadraticObjective, parameters: Any) -> np.ndarray:
        del parameters
        return np.eye(3, dtype=np.float64)

    def evaluate(
        self: _QuadraticObjective,
        parameters: Any,
        *,
        search_stage: str = "evaluated",
    ) -> _Candidate:
        self.evaluation_count += 1
        angle, translation_x, translation_y = np.asarray(
            parameters,
            dtype=np.float64,
        )
        score = 1.0 - 0.01 * (
            angle**2 + (translation_x - 3.0) ** 2 + (translation_y + 1.0) ** 2
        )
        return _Candidate(
            angle_degrees=float(angle),
            translation_x_px=float(translation_x),
            translation_y_px=float(translation_y),
            score=float(score),
            metrics={"score": float(score), "eligible": True},
            search_stage=search_stage,
        )


def _candidate(
    angle: float,
    translation_x: float,
    translation_y: float,
    score: float,
    search_stage: str,
) -> _Candidate:
    return _Candidate(
        angle_degrees=angle,
        translation_x_px=translation_x,
        translation_y_px=translation_y,
        score=score,
        metrics={"score": score, "eligible": True},
        search_stage=search_stage,
    )


def test_coarse_candidates_record_each_translation_seed_type() -> None:
    """The plot can distinguish zero, phase and centroid samples unambiguously."""
    objective = _QuadraticObjective()

    candidates = _coarse_candidates(
        objective,  # type: ignore[arg-type]
        config=PartialOverlapRigidConfig(
            angle_search_radius_degrees=2.0,
            coarse_angle_step_degrees=2.0,
        ),
        maximum_translation_px=10.0,
    )

    assert Counter(candidate.search_stage for candidate in candidates) == {
        "coarse_zero": 3,
        "coarse_phase": 3,
        "coarse_centroid": 3,
    }


def test_aligned_objective_helper_reuses_physical_identity_score() -> None:
    """Non-rigid QC can score an already-aligned pair without another search."""
    y, x = np.indices((80, 96))
    mask = ((x - 49.0) ** 2 / 34.0**2 + (y - 38.0) ** 2 / 26.0**2) <= 1.0
    image = np.where(
        mask,
        40.0
        + 160.0 * np.exp(-((x - 38.0) ** 2 + (y - 31.0) ** 2) / 120.0)
        + 90.0 * np.exp(-((x - 63.0) ** 2 + (y - 49.0) ** 2) / 80.0),
        0.0,
    ).astype(np.uint8)
    valid = np.ones_like(mask)
    valid[:, :6] = False

    summary = evaluate_aligned_partial_overlap_objective(
        image,
        image,
        mask,
        mask,
        config=PartialOverlapRigidConfig(
            max_dimension_px=128,
            density_sigma_um=4.0,
            boundary_distance_scale_um=12.0,
        ),
        pixel_size_um=2.0,
        fixed_valid_mask=valid,
        moving_valid_mask=valid,
    )

    assert np.isclose(summary["score"], 1.0)
    assert np.isclose(summary["boundary_score"], 1.0)
    assert np.isclose(summary["density_correlation"], 1.0)
    assert summary["fixed_overlap_fraction"] == 1.0
    assert summary["moving_overlap_fraction"] == 1.0
    assert summary["translation_x_um"] == 0.0
    assert summary["translation_y_um"] == 0.0
    assert summary["trimmed_boundary_distance_um"] == 0.0
    assert summary["eligible"] is True


def test_local_objective_slice_is_centred_and_uses_micrometres() -> None:
    """The true local score slice is centred on the selected physical position."""
    objective = _QuadraticObjective()
    selected = objective.evaluate([0.0, 3.0, -1.0], search_stage="refined")

    x_um, y_um, scores = _local_objective_slice(
        objective,  # type: ignore[arg-type]
        selected=selected,
        pixels_to_um=2.0,
        half_width_um=4.0,
        grid_size=5,
    )

    np.testing.assert_allclose(x_um, [2.0, 4.0, 6.0, 8.0, 10.0])
    np.testing.assert_allclose(y_um, [-6.0, -4.0, -2.0, 0.0, 2.0])
    assert scores.shape == (5, 5)
    assert np.unravel_index(np.argmax(scores), scores.shape) == (2, 2)


def test_objective_plot_labels_sampled_and_local_searches(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The QC plot explains its samples and displays the selected score."""
    objective = _QuadraticObjective()
    baseline = _candidate(0.0, 0.0, 0.0, 0.75, "baseline")
    coarse = [
        _candidate(-1.0, 0.0, 0.0, 0.72, "coarse_zero"),
        _candidate(1.0, 0.0, 0.0, 0.73, "coarse_zero"),
        _candidate(-1.0, 2.0, -1.0, 0.88, "coarse_phase"),
        _candidate(1.0, 2.5, -1.0, 0.89, "coarse_phase"),
        _candidate(-1.0, 1.0, 1.0, 0.81, "coarse_centroid"),
        _candidate(1.0, 1.5, 1.0, 0.82, "coarse_centroid"),
    ]
    refined = [
        _candidate(0.2, 2.8, -0.9, 0.96, "refined"),
        _candidate(0.0, 3.0, -1.0, 1.0, "refined"),
    ]
    selected = refined[-1]
    captured: dict[str, Figure] = {}

    def _capture_figure(
        figure: Figure,
        output_path: Path,
        **_: Any,
    ) -> Path:
        captured["figure"] = figure
        return output_path

    monkeypatch.setattr(partial_overlap, "save_figure", _capture_figure)

    _plot_objective_diagnostics(
        tmp_path / "objective.png",
        objective=objective,  # type: ignore[arg-type]
        coarse_candidates=coarse,
        refined_candidates=refined,
        ranked_candidates=refined,
        baseline=baseline,
        selected=selected,
        pixels_to_um=2.0,
    )

    axes = captured["figure"].axes
    assert axes[0].get_title() == "Rotation search samples"
    assert axes[0].get_ylabel() == "robust score"
    assert "selected" in axes[0].get_legend_handles_labels()[1]
    assert axes[1].get_title() == "Sampled translations (not a dense landscape)"
    assert axes[1].get_xlabel() == "translation x (µm)"
    assert {
        "zero-translation seed",
        "phase-correlation seed",
        "centroid seed",
        "refined candidates",
        "selected",
    }.issubset(axes[1].get_legend_handles_labels()[1])
    assert axes[2].get_title().startswith("Local x/y objective slice")
    assert axes[2].get_xlabel() == "translation x (µm)"
