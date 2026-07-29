"""Serializable VALIS transform chain from moving physical to fixed physical xy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from merxen.alignment.transforms import apply_affine_matrix, as_xy_array


@dataclass(frozen=True)
class DisplacementField:
    """Sampled residual displacement on a regular source-pixel grid."""

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    displacement_xy: np.ndarray

    def __post_init__(self: DisplacementField) -> None:
        x = np.asarray(self.x_coordinates, dtype=np.float64)
        y = np.asarray(self.y_coordinates, dtype=np.float64)
        field = np.asarray(self.displacement_xy, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Displacement grid coordinates must be one-dimensional")
        if field.shape != (len(y), len(x), 2):
            raise ValueError(
                "Displacement field shape must be "
                f"(len(y), len(x), 2), got {field.shape}"
            )
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Displacement field requires at least a 2x2 grid")

    def sample(self: DisplacementField, xy: Any) -> np.ndarray:
        """Interpolate xy displacement, returning zero outside the sampled grid."""
        points = as_xy_array(xy)
        query_yx = np.column_stack([points[:, 1], points[:, 0]])
        interpolator_x = RegularGridInterpolator(
            (
                np.asarray(self.y_coordinates, dtype=np.float64),
                np.asarray(self.x_coordinates, dtype=np.float64),
            ),
            np.asarray(self.displacement_xy, dtype=np.float64)[..., 0],
            bounds_error=False,
            fill_value=0.0,
        )
        interpolator_y = RegularGridInterpolator(
            (
                np.asarray(self.y_coordinates, dtype=np.float64),
                np.asarray(self.x_coordinates, dtype=np.float64),
            ),
            np.asarray(self.displacement_xy, dtype=np.float64)[..., 1],
            bounds_error=False,
            fill_value=0.0,
        )
        return np.column_stack([interpolator_x(query_yx), interpolator_y(query_yx)])


@dataclass(frozen=True)
class ValisTransformBundle:
    """Complete moving-dataset to fixed-dataset transform chain.

    Matrices use homogeneous column-vector convention. ``pre_matrix`` maps
    moving registration pixels into the pre-warped moving image supplied to
    VALIS. ``global_matrix`` then maps those pre-warped pixels into fixed
    registration pixels. The optional displacement is a residual in fixed
    registration pixels, sampled as a function of pre-warped moving pixels.
    """

    moving_dataset_to_image: np.ndarray
    moving_image_to_registration: np.ndarray
    pre_matrix: np.ndarray
    global_matrix: np.ndarray
    fixed_image_to_registration: np.ndarray
    fixed_dataset_to_image: np.ndarray
    selected_mode: str
    forward_displacement: DisplacementField | None = None
    backward_displacement: DisplacementField | None = None

    def transform(
        self: ValisTransformBundle,
        xy: Any,
        *,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Map moving dataset-physical xy into fixed dataset-physical xy."""
        points = as_xy_array(xy)
        if chunk_size is None or int(chunk_size) <= 0 or len(points) <= int(chunk_size):
            return self._transform_chunk(points)
        chunks = [
            self._transform_chunk(points[start : start + int(chunk_size)])
            for start in range(0, len(points), int(chunk_size))
        ]
        return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)

    def transform_global(self: ValisTransformBundle, xy: Any) -> np.ndarray:
        """Map moving to fixed physical xy without non-rigid displacement."""
        points = as_xy_array(xy)
        prewarped = apply_affine_matrix(
            points,
            self.prewarped_from_moving_dataset_matrix,
        )
        fixed_registration = apply_affine_matrix(prewarped, self.global_matrix)
        return apply_affine_matrix(
            fixed_registration,
            self.fixed_dataset_from_registration_matrix,
        )

    @property
    def prewarped_from_moving_dataset_matrix(
        self: ValisTransformBundle,
    ) -> np.ndarray:
        """Return moving dataset-physical to pre-warped registration affine."""
        return (
            np.asarray(self.pre_matrix, dtype=np.float64)
            @ np.asarray(self.moving_image_to_registration, dtype=np.float64)
            @ np.asarray(self.moving_dataset_to_image, dtype=np.float64)
        )

    @property
    def fixed_dataset_from_registration_matrix(
        self: ValisTransformBundle,
    ) -> np.ndarray:
        """Return fixed registration-pixel to fixed dataset-physical affine."""
        return np.linalg.inv(
            np.asarray(self.fixed_image_to_registration, dtype=np.float64)
            @ np.asarray(self.fixed_dataset_to_image, dtype=np.float64)
        )

    @property
    def global_dataset_matrix(self: ValisTransformBundle) -> np.ndarray:
        """Return the composed global moving-to-fixed physical affine."""
        return np.asarray(
            self.fixed_dataset_from_registration_matrix
            @ np.asarray(self.global_matrix, dtype=np.float64)
            @ self.prewarped_from_moving_dataset_matrix,
            dtype=np.float64,
        )

    def _transform_chunk(
        self: ValisTransformBundle,
        points: np.ndarray,
    ) -> np.ndarray:
        prewarped = apply_affine_matrix(
            points,
            self.prewarped_from_moving_dataset_matrix,
        )
        fixed_registration = apply_affine_matrix(prewarped, self.global_matrix)
        if self.selected_mode == "non_rigid" and self.forward_displacement is not None:
            fixed_registration = fixed_registration + self.forward_displacement.sample(
                prewarped
            )
        return apply_affine_matrix(
            fixed_registration,
            self.fixed_dataset_from_registration_matrix,
        )

    def save(
        self: ValisTransformBundle,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Serialize transform metadata and displacement arrays."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        forward_path = output_dir / "forward_displacement_field.npz"
        backward_path = output_dir / "backward_displacement_field.npz"
        if self.forward_displacement is not None:
            _save_field(self.forward_displacement, forward_path)
        if self.backward_displacement is not None:
            _save_field(self.backward_displacement, backward_path)

        metadata_path = output_dir / "transform_chain.json"
        metadata = self.to_metadata(
            forward_path=Path(forward_path.name) if forward_path.exists() else None,
            backward_path=Path(backward_path.name) if backward_path.exists() else None,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        outputs = {"transform_chain": metadata_path}
        if forward_path.exists():
            outputs["forward_displacement"] = forward_path
        if backward_path.exists():
            outputs["backward_displacement"] = backward_path
        return outputs

    def to_metadata(
        self: ValisTransformBundle,
        *,
        forward_path: Path | None = None,
        backward_path: Path | None = None,
    ) -> dict[str, Any]:
        """Return JSON-compatible coordinate-frame and matrix metadata."""
        return {
            "version": 1,
            "matrix_convention": "forward homogeneous xy column vectors",
            "coordinate_chain": [
                "moving_dataset_physical_um",
                "moving_original_dapi_xy_pixels",
                "moving_registration_xy_pixels",
                "moving_preoriented_xy_pixels",
                "fixed_registration_xy_pixels",
                "fixed_original_dapi_xy_pixels",
                "fixed_dataset_physical_um",
            ],
            "selected_mode": self.selected_mode,
            "moving_dataset_to_image": np.asarray(
                self.moving_dataset_to_image
            ).tolist(),
            "moving_image_to_registration": np.asarray(
                self.moving_image_to_registration
            ).tolist(),
            "pre_matrix": np.asarray(self.pre_matrix).tolist(),
            "global_matrix": np.asarray(self.global_matrix).tolist(),
            "fixed_image_to_registration": np.asarray(
                self.fixed_image_to_registration
            ).tolist(),
            "fixed_dataset_to_image": np.asarray(self.fixed_dataset_to_image).tolist(),
            "global_dataset_matrix": self.global_dataset_matrix.tolist(),
            "forward_displacement_path": (
                None if forward_path is None else str(forward_path)
            ),
            "backward_displacement_path": (
                None if backward_path is None else str(backward_path)
            ),
        }

    @classmethod
    def load(
        cls: type[ValisTransformBundle],
        metadata_path: Path,
    ) -> ValisTransformBundle:
        """Reload a transform bundle serialized by :meth:`save`."""
        metadata_path = Path(metadata_path)
        payload = json.loads(metadata_path.read_text())
        forward_path = payload.get("forward_displacement_path")
        backward_path = payload.get("backward_displacement_path")
        return cls(
            moving_dataset_to_image=np.asarray(
                payload["moving_dataset_to_image"],
                dtype=np.float64,
            ),
            moving_image_to_registration=np.asarray(
                payload["moving_image_to_registration"],
                dtype=np.float64,
            ),
            pre_matrix=np.asarray(payload["pre_matrix"], dtype=np.float64),
            global_matrix=np.asarray(payload["global_matrix"], dtype=np.float64),
            fixed_image_to_registration=np.asarray(
                payload["fixed_image_to_registration"],
                dtype=np.float64,
            ),
            fixed_dataset_to_image=np.asarray(
                payload["fixed_dataset_to_image"],
                dtype=np.float64,
            ),
            selected_mode=str(payload["selected_mode"]),
            forward_displacement=(
                None
                if forward_path is None
                else _load_field(_resolve_field_path(metadata_path, forward_path))
            ),
            backward_displacement=(
                None
                if backward_path is None
                else _load_field(_resolve_field_path(metadata_path, backward_path))
            ),
        )


def _save_field(field: DisplacementField, path: Path) -> None:
    np.savez_compressed(
        path,
        x_coordinates=np.asarray(field.x_coordinates, dtype=np.float32),
        y_coordinates=np.asarray(field.y_coordinates, dtype=np.float32),
        displacement_xy=np.asarray(field.displacement_xy, dtype=np.float32),
    )


def _load_field(path: Path) -> DisplacementField:
    with np.load(path) as arrays:
        return DisplacementField(
            x_coordinates=np.asarray(arrays["x_coordinates"], dtype=np.float64),
            y_coordinates=np.asarray(arrays["y_coordinates"], dtype=np.float64),
            displacement_xy=np.asarray(
                arrays["displacement_xy"],
                dtype=np.float64,
            ),
        )


def _resolve_field_path(metadata_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    candidate = metadata_path.parent / path
    if candidate.exists():
        return candidate
    return path
