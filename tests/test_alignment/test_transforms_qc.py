"""Tests for transform and alignment-QC helpers."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from merxen.alignment.qc import compute_grid_alignment_metrics, plot_alignment_overlay
from merxen.alignment.transforms import (
    apply_affine_matrix,
    fit_affine_matrix,
    fit_nonrigid_transform,
    fit_rigid_matrix,
)


def test_fit_affine_matrix_recovers_translation() -> None:
    """Affine fitting should recover a simple section translation."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    dst = src + np.array([4.0, -3.0])
    matrix = fit_affine_matrix(src, dst)

    np.testing.assert_allclose(apply_affine_matrix(src, matrix), dst, atol=1e-8)


def test_fit_rigid_matrix_recovers_rotation_translation_without_scale() -> None:
    """Rigid fitting must not introduce scale or shear into metadata-scaled data."""
    source = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 80.0], [100.0, 80.0], [37.0, 29.0]]
    )
    angle = np.deg2rad(17.0)
    expected = np.array(
        [
            [np.cos(angle), -np.sin(angle), 13.0],
            [np.sin(angle), np.cos(angle), -7.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = apply_affine_matrix(source, expected)

    recovered = fit_rigid_matrix(source, target)

    np.testing.assert_allclose(recovered, expected, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.svd(recovered[:2, :2], compute_uv=False),
        [1.0, 1.0],
        atol=1e-12,
    )
    assert np.linalg.det(recovered[:2, :2]) == pytest.approx(1.0)


def test_fit_nonrigid_transform_keeps_affine_when_residuals_are_zero() -> None:
    """The RBF wrapper should behave like affine when no local warp is present."""
    src = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    dst = src + np.array([2.0, 7.0])
    transform = fit_nonrigid_transform(src, dst, neighbors=4)

    np.testing.assert_allclose(transform.transform(src), dst, atol=1e-6)


def test_compute_grid_alignment_metrics_returns_expected_keys() -> None:
    """SABench-style QC should produce finite metrics for overlapping slices."""
    x = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    fixed = ad.AnnData(X=np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 4.0]]))
    moving = fixed.copy()
    fixed.obsm["spatial"] = x
    moving.obsm["spatial"] = x + 0.1

    metrics = compute_grid_alignment_metrics(fixed, moving, grid_shape=(2, 2))

    assert metrics["n_overlap_grids"] == 4
    assert np.isfinite(metrics["grid_cosine"])
    assert metrics["centroid_assd"] < 0.2


def test_plot_alignment_overlay_writes_png_and_pdf(tmp_path: Path) -> None:
    """Alignment overlay plotting should emit PNG and PDF files."""
    fixed = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
    moving = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
    fixed.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    moving.obsm["spatial"] = np.array([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])
    out = tmp_path / "alignment_overlay.png"

    plot_alignment_overlay(fixed, moving, out, title="alignment")

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
