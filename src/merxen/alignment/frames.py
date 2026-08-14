"""Resolve DAPI pixels, physical coordinates, and registration-image frames."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage.transform import resize
from spatialdata.transformations import get_transformation

from merxen.alignment.dapi import (
    create_dapi_tissue_mask,
    create_registration_validity_mask,
    dapi_edge_artifact_metrics,
    derive_acquired_support_mask,
    mask_outline,
    process_dapi_image,
)
from merxen.alignment.tissue import (
    AlignmentTissueAnnotation,
    rasterize_alignment_tissue_annotation,
)
from merxen.config import (
    AlignmentImageConfig,
    DAPIProcessingConfig,
    ValisAlignmentConfig,
)
from merxen.io.image_source import (
    MERSCOPE_ZPROJ_IMAGE_NAME,
    image_to_cyx,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DapiFrame:
    """Original DAPI data and its dataset-physical to pixel transform."""

    platform: str
    image_key: str
    image: Any
    original_shape_rc: tuple[int, int]
    dataset_to_image_matrix: np.ndarray
    pixel_size_xy_um: tuple[float, float]
    channel_name: str
    coordinate_metadata_source: str
    coordinate_metadata_trusted: bool


@dataclass(frozen=True)
class RegistrationFrame:
    """One DAPI image represented on the shared padded registration canvas."""

    platform: str
    image_key: str
    original_shape_rc: tuple[int, int]
    registration_shape_rc: tuple[int, int]
    dataset_to_image_matrix: np.ndarray
    original_to_registration_matrix: np.ndarray
    registration_pixel_size_um: float
    processed_image: np.ndarray
    tissue_mask: np.ndarray
    support_mask: np.ndarray
    valid_mask: np.ndarray
    edge_artifact_metrics: dict[str, float]
    coordinate_metadata_source: str
    coordinate_metadata_trusted: bool
    tissue_mask_unclipped: np.ndarray | None = None
    tissue_annotation_metadata: dict[str, Any] | None = None

    @property
    def dataset_to_registration_matrix(self: RegistrationFrame) -> np.ndarray:
        """Return dataset-physical to registration-pixel affine."""
        return np.asarray(
            self.original_to_registration_matrix @ self.dataset_to_image_matrix,
            dtype=np.float64,
        )

    @property
    def tissue_scoring_mask(self: RegistrationFrame) -> np.ndarray:
        """Return annotated tissue restricted to reliable registration pixels."""
        return np.asarray(
            (
                (np.asarray(self.tissue_mask) > 0) & (np.asarray(self.valid_mask) > 0)
            ).astype(np.uint8)
            * 255,
            dtype=np.uint8,
        )


def resolve_dapi_frame(
    sdata_obj: Any,
    *,
    platform: str,
    config: AlignmentImageConfig,
) -> DapiFrame:
    """Resolve DAPI pixels and the dataset-physical to image-pixel affine."""
    normalized_platform = str(platform).upper()
    if config.image_path is not None:
        import tifffile

        image_path = Path(config.image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"{normalized_platform} DAPI image not found: {image_path}"
            )
        raw = tifffile.imread(image_path)
        if raw.ndim != 2:
            raise ValueError(
                f"External {normalized_platform} DAPI image must be single-channel; "
                f"got shape {raw.shape}"
            )
        dapi = raw
        image_key = str(image_path)
        image_element = None
    else:
        image_key = _choose_image_key(
            sdata_obj.images,
            platform=normalized_platform,
            configured_key=config.image_key,
        )
        image_element = sdata_obj.images[image_key]
        dapi = _select_dapi_from_element(
            image_element,
            requested_channel=config.dapi_channel,
        )

    original_shape_rc = (int(dapi.shape[-2]), int(dapi.shape[-1]))
    (
        dataset_to_image,
        coordinate_metadata_source,
        coordinate_metadata_trusted,
    ) = _resolve_dataset_to_image_matrix(
        sdata_obj,
        image_element=image_element,
        configured_matrix=config.dataset_to_image_matrix,
        configured_pixel_size_um=config.pixel_size_um,
        platform=normalized_platform,
    )
    pixel_size_xy = _pixel_size_from_dataset_to_image(dataset_to_image)
    if config.pixel_size_um is not None:
        configured_pixel_size = float(config.pixel_size_um)
        if not np.allclose(
            pixel_size_xy,
            [configured_pixel_size, configured_pixel_size],
            rtol=0.02,
            atol=1e-6,
        ):
            raise ValueError(
                f"{normalized_platform} pixel_size_um={configured_pixel_size} "
                "conflicts with dataset_to_image_matrix, which implies "
                f"{pixel_size_xy} µm/px"
            )
    return DapiFrame(
        platform=normalized_platform,
        image_key=image_key,
        image=dapi,
        original_shape_rc=original_shape_rc,
        dataset_to_image_matrix=dataset_to_image,
        pixel_size_xy_um=pixel_size_xy,
        channel_name=str(config.dapi_channel),
        coordinate_metadata_source=coordinate_metadata_source,
        coordinate_metadata_trusted=coordinate_metadata_trusted,
    )


def prepare_registration_frames(
    fixed: DapiFrame,
    moving: DapiFrame,
    *,
    config: ValisAlignmentConfig,
    output_dir: Path,
    fixed_tissue_annotation: AlignmentTissueAnnotation | None = None,
    moving_tissue_annotation: AlignmentTissueAnnotation | None = None,
) -> tuple[RegistrationFrame, RegistrationFrame]:
    """Resample both DAPI images to one isotropic, padded registration canvas."""
    target_pixel_size = _shared_registration_pixel_size(fixed, moving, config)
    fixed_shape = _resampled_shape(fixed, target_pixel_size)
    moving_shape = _resampled_shape(moving, target_pixel_size)
    max_diag = max(
        float(np.hypot(*fixed_shape)),
        float(np.hypot(*moving_shape)),
    )
    canvas_side = int(
        np.ceil(max_diag * (1.0 + 2.0 * float(config.canvas_padding_fraction)))
    )
    canvas_shape = (max(canvas_side, 1), max(canvas_side, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_reg = _prepare_one_frame(
        fixed,
        target_shape_rc=fixed_shape,
        canvas_shape_rc=canvas_shape,
        target_pixel_size_um=target_pixel_size,
        processing=config.preprocessing,
        tissue_annotation=fixed_tissue_annotation,
    )
    moving_reg = _prepare_one_frame(
        moving,
        target_shape_rc=moving_shape,
        canvas_shape_rc=canvas_shape,
        target_pixel_size_um=target_pixel_size,
        processing=config.preprocessing,
        tissue_annotation=moving_tissue_annotation,
    )
    _save_frame_qc(fixed_reg, output_dir)
    _save_frame_qc(moving_reg, output_dir)
    return fixed_reg, moving_reg


def _choose_image_key(
    images: Any,
    *,
    platform: str,
    configured_key: str | None,
) -> str:
    keys = [str(key) for key in images]
    if configured_key is not None:
        if configured_key not in images:
            raise KeyError(
                f"{platform} alignment image {configured_key!r} not found; "
                f"available images: {keys}"
            )
        return configured_key

    preferred = (
        [MERSCOPE_ZPROJ_IMAGE_NAME]
        if platform == "MERSCOPE"
        else ["morphology_focus", "morphology_mip"]
    )
    for key in preferred:
        if key in images:
            return key
    dapi_keys = [key for key in keys if "dapi" in key.casefold()]
    if len(dapi_keys) == 1:
        return dapi_keys[0]
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"Could not choose an unambiguous {platform} DAPI image. "
        f"Set image_key explicitly; available images: {keys}"
    )


def _select_dapi_from_element(
    image_element: Any,
    *,
    requested_channel: str,
) -> Any:
    cyx = image_to_cyx(image_element)
    channel_values = [
        str(value) for value in np.asarray(cyx.coords["c"].values).tolist()
    ]
    requested = str(requested_channel).strip().casefold()
    matching = [
        idx
        for idx, name in enumerate(channel_values)
        if name.strip().casefold() == requested
    ]
    if len(matching) != 1:
        if int(cyx.sizes["c"]) == 1 and channel_values == ["c0"]:
            logger.warning(
                "Using the only unnamed image channel as DAPI for %s",
                requested_channel,
            )
            matching = [0]
        else:
            raise ValueError(
                f"Expected exactly one {requested_channel!r} channel; "
                f"available channels are {channel_values}"
            )
    return cyx.isel(c=matching[0])


def _resolve_dataset_to_image_matrix(
    sdata_obj: Any,
    *,
    image_element: Any | None,
    configured_matrix: list[list[float]] | None,
    configured_pixel_size_um: float | None,
    platform: str,
) -> tuple[np.ndarray, str, bool]:
    if configured_matrix is not None:
        matrix = np.asarray(configured_matrix, dtype=np.float64)
        _validate_affine_matrix(matrix, label=f"{platform} dataset_to_image_matrix")
        return matrix, "configured dataset_to_image_matrix", True

    # An explicit physical pixel size is an authoritative override. In
    # particular, it must supersede an identity transform left on a rewritten
    # SpatialData image, because identity would otherwise be misinterpreted as
    # 1 µm/px.
    if configured_pixel_size_um is not None:
        pixel_size = float(configured_pixel_size_um)
        return (
            np.array(
                [
                    [1.0 / pixel_size, 0.0, 0.0],
                    [0.0, 1.0 / pixel_size, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "configured pixel_size_um",
            True,
        )

    if image_element is not None:
        image_to_global = _element_affine_to_global(image_element)
        dataset_key, dataset_element, trusted = _choose_dataset_element(
            sdata_obj,
            platform=platform,
        )
        if dataset_element is not None:
            dataset_to_global = _element_affine_to_global(dataset_element)
            matrix = np.linalg.inv(image_to_global) @ dataset_to_global
            _validate_affine_matrix(
                matrix,
                label=f"inferred {platform} dataset-to-image transform",
            )
            logger.info(
                "%s DAPI physical frame inferred from SpatialData element %r",
                platform,
                dataset_key,
            )
            return (
                np.asarray(matrix, dtype=np.float64),
                f"SpatialData element {dataset_key!r}",
                trusted,
            )
    raise ValueError(
        f"Cannot reliably determine the {platform} dataset-physical to DAPI-pixel "
        "transform. Provide dataset_to_image_matrix and pixel_size_um explicitly."
    )


def _choose_dataset_element(
    sdata_obj: Any,
    *,
    platform: str,
) -> tuple[str | None, Any | None, bool]:
    """Choose an element whose native xy values are physical dataset units.

    Latest MerXen stores intentionally contain derived points and polygons in
    micron coordinates with identity transforms, while the DAPI image remains
    in native pixels. The original vendor segmentation retains the required
    micron-to-image transform and is registered in ``merxen_schema``. Selecting
    the first points element therefore silently yields the wrong 1 µm/px scale.
    """
    shapes = getattr(sdata_obj, "shapes", {})
    attrs = getattr(sdata_obj, "attrs", {})
    schema = attrs.get("merxen_schema", {}) if isinstance(attrs, dict) else {}
    segmentations = schema.get("segmentations", {}) if isinstance(schema, dict) else {}
    original = (
        segmentations.get("original", {}) if isinstance(segmentations, dict) else {}
    )
    schema_shape_key = original.get("shape") if isinstance(original, dict) else None
    preferred_shape_keys = [
        schema_shape_key,
        (
            "merscope_cell_boundaries"
            if str(platform).upper() == "MERSCOPE"
            else "xenium_cell_boundaries"
        ),
    ]
    for key in preferred_shape_keys:
        if key is None or key not in shapes:
            continue
        element = shapes[key]
        _element_affine_to_global(element)
        return str(key), element, True

    # Compatibility fallback for pre-schema stores: prefer a non-identity
    # transform, since an arbitrary identity-derived element cannot connect
    # physical coordinates to a native-pixel image.
    identity_candidate: tuple[str, Any] | None = None
    for mapping_name in ("points", "shapes", "labels"):
        mapping = getattr(sdata_obj, mapping_name, {})
        for key, element in mapping.items():
            try:
                matrix = _element_affine_to_global(element)
            except (KeyError, ValueError):
                continue
            candidate = (f"{mapping_name}.{key}", element)
            if not np.allclose(matrix, np.eye(3), atol=1e-8):
                return candidate[0], candidate[1], False
            if identity_candidate is None:
                identity_candidate = candidate
    if identity_candidate is not None:
        return identity_candidate[0], identity_candidate[1], False
    return None, None, False


def _element_affine_to_global(element: Any) -> np.ndarray:
    transformation = get_transformation(element, to_coordinate_system="global")
    matrix = np.asarray(
        transformation.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        dtype=np.float64,
    )
    _validate_affine_matrix(matrix, label="SpatialData global transform")
    return matrix


def _validate_affine_matrix(matrix: np.ndarray, *, label: str) -> None:
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} must be 3x3, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    if abs(float(np.linalg.det(matrix[:2, :2]))) < 1e-12:
        raise ValueError(f"{label} is singular")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} is not a 2D homogeneous affine")


def _pixel_size_from_dataset_to_image(matrix: np.ndarray) -> tuple[float, float]:
    image_to_dataset = np.linalg.inv(matrix)
    linear = image_to_dataset[:2, :2]
    pixel_size_x = float(np.linalg.norm(linear[:, 0]))
    pixel_size_y = float(np.linalg.norm(linear[:, 1]))
    if (
        not np.isfinite(pixel_size_x)
        or not np.isfinite(pixel_size_y)
        or pixel_size_x <= 0
        or pixel_size_y <= 0
    ):
        raise ValueError("Could not derive positive DAPI pixel sizes from transform")
    return pixel_size_x, pixel_size_y


def _shared_registration_pixel_size(
    fixed: DapiFrame,
    moving: DapiFrame,
    config: ValisAlignmentConfig,
) -> float:
    if config.registration_pixel_size_um is not None:
        requested = float(config.registration_pixel_size_um)
    else:
        requested = max(*fixed.pixel_size_xy_um, *moving.pixel_size_xy_um)

    physical_diagonals = []
    for frame in (fixed, moving):
        height, width = frame.original_shape_rc
        physical_diagonals.append(
            float(
                np.hypot(
                    height * frame.pixel_size_xy_um[1],
                    width * frame.pixel_size_xy_um[0],
                )
            )
        )
    size_limited = max(physical_diagonals) / float(
        config.registration_source_max_dim_px
    )
    return max(requested, size_limited)


def _resampled_shape(
    frame: DapiFrame,
    target_pixel_size_um: float,
) -> tuple[int, int]:
    height, width = frame.original_shape_rc
    target_height = max(
        1,
        int(round(height * frame.pixel_size_xy_um[1] / target_pixel_size_um)),
    )
    target_width = max(
        1,
        int(round(width * frame.pixel_size_xy_um[0] / target_pixel_size_um)),
    )
    return target_height, target_width


def _prepare_one_frame(
    frame: DapiFrame,
    *,
    target_shape_rc: tuple[int, int],
    canvas_shape_rc: tuple[int, int],
    target_pixel_size_um: float,
    processing: DAPIProcessingConfig,
    tissue_annotation: AlignmentTissueAnnotation | None = None,
) -> RegistrationFrame:
    source = _materialize_downsampled(frame.image, target_shape_rc)
    support_mask = derive_acquired_support_mask(
        source,
        platform=frame.platform,
    )
    processed = process_dapi_image(
        source,
        pixel_size_um=target_pixel_size_um,
        config=processing,
        acquired_support_mask=support_mask,
    )
    valid_mask = create_registration_validity_mask(
        target_shape_rc,
        pixel_size_um=target_pixel_size_um,
        edge_exclusion_um=float(processing.edge_exclusion_um),
        acquired_support_mask=support_mask,
    )
    edge_metrics = dapi_edge_artifact_metrics(
        processed,
        acquired_support_mask=support_mask,
    )
    pad_y = (canvas_shape_rc[0] - target_shape_rc[0]) // 2
    pad_x = (canvas_shape_rc[1] - target_shape_rc[1]) // 2
    canvas_image = np.zeros(canvas_shape_rc, dtype=np.uint8)
    canvas_mask = np.zeros(canvas_shape_rc, dtype=np.uint8)
    canvas_support = np.zeros(canvas_shape_rc, dtype=np.uint8)
    canvas_valid = np.zeros(canvas_shape_rc, dtype=np.uint8)
    canvas_image[
        pad_y : pad_y + target_shape_rc[0],
        pad_x : pad_x + target_shape_rc[1],
    ] = processed
    canvas_support[
        pad_y : pad_y + target_shape_rc[0],
        pad_x : pad_x + target_shape_rc[1],
    ] = support_mask
    canvas_valid[
        pad_y : pad_y + target_shape_rc[0],
        pad_x : pad_x + target_shape_rc[1],
    ] = valid_mask

    original_height, original_width = frame.original_shape_rc
    scale_x = target_shape_rc[1] / float(original_width)
    scale_y = target_shape_rc[0] / float(original_height)
    original_to_registration = np.array(
        [
            [scale_x, 0.0, float(pad_x)],
            [0.0, scale_y, float(pad_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tissue_mask_unclipped: np.ndarray | None = None
    tissue_annotation_metadata: dict[str, Any] | None = None
    if tissue_annotation is None:
        # Compatibility path for source-level diagnostics. The production VALIS
        # dispatcher always supplies required manual annotations.
        tissue_mask = create_dapi_tissue_mask(
            processed,
            pixel_size_um=target_pixel_size_um,
            config=processing,
        )
        tissue_mask = np.where(valid_mask > 0, tissue_mask, 0).astype(
            np.uint8,
            copy=False,
        )
        canvas_mask[
            pad_y : pad_y + target_shape_rc[0],
            pad_x : pad_x + target_shape_rc[1],
        ] = tissue_mask
        tissue_annotation_metadata = {"source": "automatic_dapi_compatibility"}
    else:
        rasterized = rasterize_alignment_tissue_annotation(
            tissue_annotation,
            dataset_to_registration_matrix=(
                original_to_registration @ frame.dataset_to_image_matrix
            ),
            shape_rc=canvas_shape_rc,
            acquired_support_mask=canvas_support,
            registration_pixel_size_um=target_pixel_size_um,
        )
        canvas_mask = rasterized.tissue
        tissue_mask_unclipped = rasterized.unclipped
        tissue_annotation_metadata = dict(rasterized.metadata)
        tissue_annotation_metadata["fraction_outside_eroded_validity"] = float(
            np.count_nonzero(
                (np.asarray(canvas_mask) > 0) & (np.asarray(canvas_valid) == 0)
            )
            / max(1, np.count_nonzero(canvas_mask))
        )
    if not np.any(canvas_mask):
        raise ValueError(f"{frame.platform} tissue mask contains no foreground")
    return RegistrationFrame(
        platform=frame.platform,
        image_key=frame.image_key,
        original_shape_rc=frame.original_shape_rc,
        registration_shape_rc=canvas_shape_rc,
        dataset_to_image_matrix=frame.dataset_to_image_matrix,
        original_to_registration_matrix=original_to_registration,
        registration_pixel_size_um=float(target_pixel_size_um),
        processed_image=canvas_image,
        tissue_mask=canvas_mask,
        support_mask=canvas_support,
        valid_mask=canvas_valid,
        edge_artifact_metrics=edge_metrics,
        coordinate_metadata_source=frame.coordinate_metadata_source,
        coordinate_metadata_trusted=frame.coordinate_metadata_trusted,
        tissue_mask_unclipped=tissue_mask_unclipped,
        tissue_annotation_metadata=tissue_annotation_metadata,
    )


def _materialize_downsampled(
    image: Any,
    target_shape_rc: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = int(image.shape[-2]), int(image.shape[-1])
    target_height, target_width = target_shape_rc
    step_y = max(1, source_height // max(target_height * 2, 1))
    step_x = max(1, source_width // max(target_width * 2, 1))

    if hasattr(image, "isel"):
        sampled = image.isel(
            y=slice(None, None, step_y),
            x=slice(None, None, step_x),
        )
        data = sampled.data
    else:
        data = np.asarray(image)[::step_y, ::step_x]
    if hasattr(data, "compute"):
        data = data.compute()
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D selected DAPI channel, got shape {arr.shape}")
    if arr.shape != target_shape_rc:
        arr = resize(
            arr,
            target_shape_rc,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
    return np.asarray(arr, dtype=np.float32)


def _save_frame_qc(frame: RegistrationFrame, output_dir: Path) -> None:
    import tifffile

    prefix = frame.platform.lower()
    tifffile.imwrite(
        output_dir / f"{prefix}_dapi_processed.tif",
        frame.processed_image,
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / f"{prefix}_tissue_mask.tif",
        frame.tissue_mask,
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / f"{prefix}_tissue_mask_outline.tif",
        mask_outline(frame.tissue_mask),
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / f"{prefix}_acquired_support_mask.tif",
        frame.support_mask,
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / f"{prefix}_registration_valid_mask.tif",
        frame.valid_mask,
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / f"{prefix}_tissue_scoring_mask.tif",
        frame.tissue_scoring_mask,
        photometric="minisblack",
    )
    if frame.tissue_mask_unclipped is not None:
        tifffile.imwrite(
            output_dir / f"{prefix}_tissue_mask_before_support_clip.tif",
            frame.tissue_mask_unclipped,
            photometric="minisblack",
        )
    if frame.tissue_annotation_metadata is not None:
        (output_dir / f"{prefix}_tissue_mask_metadata.json").write_text(
            json.dumps(frame.tissue_annotation_metadata, indent=2)
        )
    if frame.tissue_annotation_metadata is not None and (
        frame.tissue_annotation_metadata.get("source") == "manual_annotation"
    ):
        _save_tissue_annotation_overlay(
            frame,
            output_dir / f"{prefix}_tissue_annotation_overlay.png",
        )
    (output_dir / f"{prefix}_dapi_edge_metrics.json").write_text(
        json.dumps(frame.edge_artifact_metrics, indent=2)
    )


def _save_tissue_annotation_overlay(frame: RegistrationFrame, path: Path) -> None:
    """Write processed DAPI with anatomical and reliable-domain outlines."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(frame.processed_image, cmap="gray")
    axis.contour(frame.tissue_mask > 0, levels=[0.5], colors=["lime"], linewidths=0.8)
    axis.contour(
        frame.tissue_scoring_mask > 0,
        levels=[0.5],
        colors=["cyan"],
        linewidths=0.5,
    )
    axis.set_title(
        f"{frame.platform} tissue annotation: anatomy=green, image scoring=cyan"
    )
    axis.axis("off")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
