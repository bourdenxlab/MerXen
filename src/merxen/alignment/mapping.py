"""Reloadable forward/inverse coordinate mapping for alignment materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
from merxen.alignment.frames import _resolve_dataset_to_image_matrix
from merxen.alignment.manifest import ALIGNMENT_MANIFEST_ATTR
from merxen.alignment.transforms import (
    NonRigidTransform,
    apply_affine_matrix,
    as_xy_array,
)
from merxen.config import AlignmentConfig


@dataclass(frozen=True)
class AlignmentCoordinateMapper:
    """Common moving/fixed mapping API used by vectors and raster pull warps."""

    backend: str
    selected_mode: str
    moving_dataset_to_image_matrix: np.ndarray
    fixed_dataset_to_image_matrix: np.ndarray
    valis_bundle: ValisTransformBundle | None = None
    legacy_transform: NonRigidTransform | None = None
    legacy_inverse_field: DisplacementField | None = None

    def moving_dataset_to_fixed_dataset(
        self: AlignmentCoordinateMapper,
        xy: Any,
    ) -> np.ndarray:
        """Map native moving dataset coordinates into the fixed dataset frame."""
        if self.valis_bundle is not None:
            return self.valis_bundle.transform(xy, chunk_size=250_000)
        if self.legacy_transform is None:
            raise RuntimeError("Alignment mapper has no forward transform")
        if self.selected_mode != "nonrigid":
            return apply_affine_matrix(xy, self.legacy_transform.affine_matrix)
        return _transform_legacy_in_chunks(self.legacy_transform, xy)

    def fixed_dataset_to_moving_dataset(
        self: AlignmentCoordinateMapper,
        xy: Any,
    ) -> np.ndarray:
        """Map fixed dataset coordinates back into the moving dataset frame."""
        points = as_xy_array(xy)
        if self.valis_bundle is not None:
            return self.valis_bundle.fixed_dataset_to_moving_dataset(points)
        if self.legacy_transform is None:
            raise RuntimeError("Alignment mapper has no inverse transform")
        inverse_affine = np.linalg.inv(
            np.asarray(self.legacy_transform.affine_matrix, dtype=np.float64)
        )
        affine_source = apply_affine_matrix(points, inverse_affine)
        if self.selected_mode != "nonrigid":
            return affine_source
        if self.legacy_inverse_field is None:
            raise ValueError(
                "Legacy non-rigid raster materialization requires an accepted "
                "inverse displacement field"
            )
        return np.asarray(
            affine_source + self.legacy_inverse_field.sample(points),
            dtype=np.float64,
        )

    def fixed_image_to_moving_image(
        self: AlignmentCoordinateMapper,
        xy: Any,
    ) -> np.ndarray:
        """Map fixed scale-0 image pixels into source moving-image pixels."""
        if self.valis_bundle is not None:
            return self.valis_bundle.fixed_image_to_moving_image(xy)
        fixed_dataset = apply_affine_matrix(
            xy,
            np.linalg.inv(
                np.asarray(self.fixed_dataset_to_image_matrix, dtype=np.float64)
            ),
        )
        moving_dataset = self.fixed_dataset_to_moving_dataset(fixed_dataset)
        return apply_affine_matrix(
            moving_dataset,
            np.asarray(self.moving_dataset_to_image_matrix, dtype=np.float64),
        )

    def roundtrip_error(self: AlignmentCoordinateMapper, fixed_xy: Any) -> np.ndarray:
        """Return fixed→moving→fixed Euclidean error for each coordinate."""
        fixed = as_xy_array(fixed_xy)
        moving = self.fixed_dataset_to_moving_dataset(fixed)
        returned = self.moving_dataset_to_fixed_dataset(moving)
        return np.asarray(np.linalg.norm(returned - fixed, axis=1), dtype=np.float64)


def load_alignment_mapper(
    config: AlignmentConfig,
    *,
    moving_sdata: Any,
    fixed_sdata: Any,
    source_image_key: str,
    fixed_image_key: str,
    fixed_shape_rc: tuple[int, int],
    require_raster_inverse: bool,
) -> AlignmentCoordinateMapper:
    """Load the selected transform from align output or the moving Zarr bundle.

    Args:
        config: Complete alignment and materialization configuration.
        moving_sdata: Moving-platform SpatialData object.
        fixed_sdata: Fixed-platform SpatialData object.
        source_image_key: Full-resolution moving image used for pull sampling.
        fixed_image_key: Fixed image defining the destination scale-0 grid.
        fixed_shape_rc: Destination image height and width.
        require_raster_inverse: Reject missing non-rigid inverse data when true.

    Returns:
        A mapper supporting both vector-forward and raster-backward transforms.
    """
    moving_image = moving_sdata.images[source_image_key]
    fixed_image = fixed_sdata.images[fixed_image_key]
    image_configs = {
        "MERSCOPE": config.merscope_image,
        "XENIUM": config.xenium_image,
    }
    moving_image_config = image_configs[config.moving_platform]
    fixed_image_config = image_configs[config.fixed_platform]
    moving_dataset_to_image, _, _ = _resolve_dataset_to_image_matrix(
        moving_sdata,
        image_element=moving_image,
        configured_matrix=moving_image_config.dataset_to_image_matrix,
        configured_pixel_size_um=moving_image_config.pixel_size_um,
        platform=config.moving_platform,
    )
    fixed_dataset_to_image, _, _ = _resolve_dataset_to_image_matrix(
        fixed_sdata,
        image_element=fixed_image,
        configured_matrix=fixed_image_config.dataset_to_image_matrix,
        configured_pixel_size_um=fixed_image_config.pixel_size_um,
        platform=config.fixed_platform,
    )

    if config.backend == "valis":
        bundle = _load_valis_bundle(config, moving_sdata)
        if not np.allclose(
            bundle.moving_dataset_to_image,
            moving_dataset_to_image,
            atol=1e-8,
        ) or not np.allclose(
            bundle.fixed_dataset_to_image,
            fixed_dataset_to_image,
            atol=1e-8,
        ):
            raise ValueError(
                "Current image coordinate frames differ from the saved VALIS bundle"
            )
        if bundle.selected_mode == "non_rigid" and bundle.forward_displacement is None:
            raise ValueError(
                "Non-rigid vector materialization requires the VALIS forward field"
            )
        if (
            require_raster_inverse
            and bundle.selected_mode == "non_rigid"
            and bundle.backward_displacement is None
        ):
            raise ValueError(
                "Non-rigid raster materialization requires the VALIS backward field"
            )
        return AlignmentCoordinateMapper(
            backend="valis",
            selected_mode=bundle.selected_mode,
            moving_dataset_to_image_matrix=np.asarray(
                moving_dataset_to_image,
                dtype=np.float64,
            ),
            fixed_dataset_to_image_matrix=np.asarray(
                fixed_dataset_to_image,
                dtype=np.float64,
            ),
            valis_bundle=bundle,
        )

    legacy_transform, selected_mode = _load_legacy_transform(config, moving_sdata)
    inverse_field = None
    if require_raster_inverse and selected_mode == "nonrigid":
        inverse_field = build_legacy_inverse_field(
            legacy_transform,
            fixed_shape_rc=fixed_shape_rc,
            fixed_dataset_to_image_matrix=fixed_dataset_to_image,
            spacing_um=config.materialization.legacy_inverse_spacing,
            iterations=config.materialization.legacy_inverse_iterations,
            tolerance_um=config.materialization.legacy_inverse_tolerance_um,
        )
    return AlignmentCoordinateMapper(
        backend="legacy_spateo",
        selected_mode=selected_mode,
        moving_dataset_to_image_matrix=np.asarray(
            moving_dataset_to_image,
            dtype=np.float64,
        ),
        fixed_dataset_to_image_matrix=np.asarray(
            fixed_dataset_to_image,
            dtype=np.float64,
        ),
        legacy_transform=legacy_transform,
        legacy_inverse_field=inverse_field,
    )


def build_legacy_inverse_field(
    transform: NonRigidTransform,
    *,
    fixed_shape_rc: tuple[int, int],
    fixed_dataset_to_image_matrix: Any,
    spacing_um: float,
    iterations: int,
    tolerance_um: float,
) -> DisplacementField:
    """Build and validate a bounded-resolution inverse for affine-plus-RBF data.

    Iteration starts from the affine inverse and solves ``A*x + r(x) = y`` as
    ``x <- A^-1(y - r(x))``. The resulting field stores residual corrections to
    the affine inverse on a fixed-coordinate grid, so rasterization remains a
    destination-to-source pull operation.
    """
    target = _fixed_dataset_grid(
        fixed_shape_rc,
        fixed_dataset_to_image_matrix=fixed_dataset_to_image_matrix,
        spacing_um=spacing_um,
    )
    inverse_affine = np.linalg.inv(
        np.asarray(transform.affine_matrix, dtype=np.float64)
    )
    affine_source = apply_affine_matrix(target, inverse_affine)
    source = affine_source.copy()
    for _ in range(int(iterations)):
        affine_forward = apply_affine_matrix(source, transform.affine_matrix)
        residual = _transform_legacy_in_chunks(transform, source) - affine_forward
        updated = apply_affine_matrix(target - residual, inverse_affine)
        delta = np.linalg.norm(updated - source, axis=1)
        source = updated
        if len(delta) == 0 or float(np.nanmax(delta)) <= tolerance_um * 0.05:
            break

    roundtrip = np.linalg.norm(
        _transform_legacy_in_chunks(transform, source) - target,
        axis=1,
    )
    maximum_error = float(np.nanmax(roundtrip)) if len(roundtrip) else 0.0
    if not np.isfinite(maximum_error) or maximum_error > float(tolerance_um):
        raise ValueError(
            "Legacy inverse field failed forward round-trip validation: "
            f"maximum error {maximum_error:.6g} µm exceeds {tolerance_um:.6g} µm"
        )

    x_coordinates = np.unique(target[:, 0])
    y_coordinates = np.unique(target[:, 1])
    return DisplacementField(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        displacement_xy=(source - affine_source).reshape(
            len(y_coordinates),
            len(x_coordinates),
            2,
        ),
    )


def _fixed_dataset_grid(
    shape_rc: tuple[int, int],
    *,
    fixed_dataset_to_image_matrix: Any,
    spacing_um: float,
) -> np.ndarray:
    height, width = (int(shape_rc[0]), int(shape_rc[1]))
    corners_px = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float64,
    )
    corners = apply_affine_matrix(
        corners_px,
        np.linalg.inv(np.asarray(fixed_dataset_to_image_matrix, dtype=np.float64)),
    )
    x = _bounded_axis(
        float(np.min(corners[:, 0])), float(np.max(corners[:, 0])), spacing_um
    )
    y = _bounded_axis(
        float(np.min(corners[:, 1])), float(np.max(corners[:, 1])), spacing_um
    )
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _bounded_axis(start: float, stop: float, spacing: float) -> np.ndarray:
    if np.isclose(start, stop):
        return np.asarray([start, start + float(spacing)], dtype=np.float64)
    values = np.arange(start, stop, float(spacing), dtype=np.float64)
    return np.asarray(np.unique(np.append(values, stop)), dtype=np.float64)


def _transform_legacy_in_chunks(
    transform: NonRigidTransform,
    xy: Any,
    *,
    chunk_size: int = 250_000,
) -> np.ndarray:
    points = as_xy_array(xy)
    chunks = [
        transform.transform(points[start : start + int(chunk_size)])
        for start in range(0, len(points), int(chunk_size))
    ]
    return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)


def _load_valis_bundle(
    config: AlignmentConfig,
    moving_sdata: Any,
) -> ValisTransformBundle:
    output_path = Path(config.output_dir) / "transform_chain.json"
    if output_path.exists():
        return ValisTransformBundle.load(output_path)
    manifest = getattr(moving_sdata, "attrs", {}).get(ALIGNMENT_MANIFEST_ATTR, {})
    transform_metadata = (
        manifest.get("transform", {}) if isinstance(manifest, dict) else {}
    )
    bundle_path = transform_metadata.get("bundle_path")
    if bundle_path is not None:
        candidate = Path(config.merscope_zarr_path) / str(bundle_path)
        if candidate.exists():
            return ValisTransformBundle.load(candidate)
    embedded = transform_metadata.get("transform_chain")
    if isinstance(embedded, dict):
        base = (
            Path(config.merscope_zarr_path) / Path(str(bundle_path)).parent
            if bundle_path is not None
            else Path(config.merscope_zarr_path)
        )
        return ValisTransformBundle.from_metadata(embedded, base_path=base)
    raise FileNotFoundError(
        "VALIS transform bundle was not found in align output or the MERSCOPE Zarr"
    )


def _load_legacy_transform(
    config: AlignmentConfig,
    moving_sdata: Any,
) -> tuple[NonRigidTransform, str]:
    transform_path = Path(config.output_dir) / "alignment_transform.json"
    if transform_path.exists():
        payload = json.loads(transform_path.read_text())
        selected_mode = str(
            payload.get("merscope_to_common", {}).get("selected_mode", "nonrigid")
        )
        transform_payload = payload.get("nonrigid_transform")
        if isinstance(transform_payload, dict):
            return _legacy_transform_from_payload(transform_payload), selected_mode
    attrs = getattr(moving_sdata, "attrs", {})
    manifest = attrs.get(ALIGNMENT_MANIFEST_ATTR, {})
    if isinstance(manifest, dict):
        transform_metadata = manifest.get("transform", {})
        if not isinstance(transform_metadata, dict):
            transform_metadata = {}
        transform_payload = transform_metadata.get("payload")
        if not isinstance(transform_payload, dict):
            transform_payload = transform_metadata.get("nonrigid_transform")
        selected_mode = str(transform_metadata.get("selected_mode", "nonrigid"))
        if not isinstance(transform_payload, dict) and manifest.get("version") == 1:
            transform_payload = manifest.get("nonrigid_transform")
            selected_mode = str(manifest.get("selected_mode", "nonrigid"))
            if not isinstance(transform_payload, dict):
                rigid_affine = manifest.get("rigid_affine_matrix")
                if rigid_affine is not None:
                    transform_payload = {
                        "affine_matrix": rigid_affine,
                        "anchors": [],
                        "residuals": [],
                    }
        if isinstance(transform_payload, dict):
            return (
                _legacy_transform_from_payload(transform_payload),
                selected_mode,
            )
    raise FileNotFoundError(
        "Legacy affine-plus-RBF payload was not found in align output or the "
        "MERSCOPE Zarr"
    )


def _legacy_transform_from_payload(payload: dict[str, Any]) -> NonRigidTransform:
    return NonRigidTransform(
        affine_matrix=np.asarray(payload["affine_matrix"], dtype=np.float64),
        anchors=np.asarray(payload.get("anchors", []), dtype=np.float64).reshape(-1, 2),
        residuals=np.asarray(payload.get("residuals", []), dtype=np.float64).reshape(
            -1, 2
        ),
        neighbors=int(payload.get("neighbors", 64)),
        smoothing=float(payload.get("smoothing", 0.0)),
        support_radius=(
            None
            if payload.get("support_radius") is None
            else float(payload["support_radius"])
        ),
    )
