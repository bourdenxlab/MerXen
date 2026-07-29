"""DAPI-only preprocessing and tissue masking for VALIS registration."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology

from merxen.alignment.valis_compat import apply_valis_numpy_compatibility
from merxen.config import DAPIProcessingConfig

apply_valis_numpy_compatibility()

try:  # VALIS is optional outside the dedicated alignment environment.
    from valis.preprocessing import ImageProcesser as _ValisImageProcesser
except ImportError:  # pragma: no cover - exercised in the base test environment

    class _ValisImageProcesser:  # type: ignore[no-redef]
        """Small import-safe stand-in for VALIS' processor base class."""

        def __init__(
            self: _ValisImageProcesser,
            image: Any,
            src_f: str = "",
            level: int = 0,
            series: int = 0,
            reader: Any = None,
        ) -> None:
            del reader
            self.image = image
            self.src_f = src_f
            self.level = level
            self.series = series


def select_dapi_channel(
    image: Any,
    *,
    channel_names: list[str] | None = None,
    dapi_channel: str = "DAPI",
) -> np.ndarray:
    """Select DAPI explicitly and return a two-dimensional array.

    Args:
        image: Image in ``(y, x)``, ``(c, y, x)``, or ``(y, x, c)`` order.
        channel_names: Channel labels in image order when the image is multichannel.
        dapi_channel: Requested channel label, matched case-insensitively.

    Returns:
        Two-dimensional DAPI image.

    Raises:
        ValueError: If a multichannel image has no unambiguous DAPI channel.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"DAPI image must be 2D or 3D, got shape {arr.shape}")

    if channel_names is not None:
        normalized = [str(name).strip().casefold() for name in channel_names]
        requested = str(dapi_channel).strip().casefold()
        matches = [idx for idx, name in enumerate(normalized) if name == requested]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {dapi_channel!r} channel; "
                f"available channels are {channel_names}"
            )
        channel_idx = matches[0]
        if arr.shape[0] == len(channel_names):
            return np.asarray(arr[channel_idx])
        if arr.shape[-1] == len(channel_names):
            return np.asarray(arr[..., channel_idx])
        raise ValueError(
            "Channel labels do not match either image channel axis: "
            f"shape={arr.shape}, channels={channel_names}"
        )

    singleton_axes = [axis for axis, size in enumerate(arr.shape) if size == 1]
    if len(singleton_axes) == 1:
        return np.squeeze(arr, axis=singleton_axes[0])
    raise ValueError(
        "A multichannel registration image requires named channels so DAPI can "
        "be selected explicitly"
    )


def process_dapi_image(
    image: Any,
    *,
    pixel_size_um: float,
    config: DAPIProcessingConfig,
    acquired_support_mask: Any | None = None,
) -> np.ndarray:
    """Convert DAPI into a smoothed, contrast-normalized uint8 image.

    ``acquired_support_mask`` describes pixels that were physically imaged. It
    is deliberately separate from the tissue mask: acquisition padding is not
    biological background and must not contribute to either Gaussian
    background estimation or registration features.
    """
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DAPI image, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    support = _coerce_support_mask(acquired_support_mask, shape_rc=arr.shape)

    background_sigma_px = max(
        0.5,
        float(config.background_sigma_um) / float(pixel_size_um),
    )
    boundary_mode = str(config.background_boundary_mode)
    background = _support_normalized_gaussian(
        arr,
        support,
        sigma=background_sigma_px,
        mode=boundary_mode,
    )
    arr = np.maximum(arr - background, 0.0)
    arr[~support] = 0.0

    finite = arr[support & np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("DAPI image contains no finite intensities")
    low, high = np.percentile(
        finite,
        [float(config.lower_percentile), float(config.upper_percentile)],
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("DAPI image has no usable robust intensity range")
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)

    if config.compression == "log1p":
        arr = np.log1p(9.0 * arr) / np.log(10.0)
    else:
        arr = np.arcsinh(5.0 * arr) / np.arcsinh(5.0)

    arr = exposure.equalize_adapthist(
        arr,
        clip_limit=float(config.clahe_clip_limit),
    ).astype(np.float32, copy=False)
    arr[~support] = 0.0
    smoothing_sigma_px = float(config.smoothing_sigma_um) / float(pixel_size_um)
    if smoothing_sigma_px > 0:
        arr = _support_normalized_gaussian(
            arr,
            support,
            sigma=smoothing_sigma_px,
            mode=boundary_mode,
        )
        arr[~support] = 0.0
    processed = np.asarray(
        exposure.rescale_intensity(
            arr,
            in_range="image",
            out_range=np.uint8,
        ),
        dtype=np.uint8,
    )
    taper_width_px = int(np.ceil(float(config.edge_taper_um) / float(pixel_size_um)))
    taper = cosine_support_taper(support, width_px=taper_width_px)
    processed = np.rint(processed.astype(np.float32) * taper).astype(np.uint8)
    processed[~support] = 0
    return processed


def derive_acquired_support_mask(
    image: Any,
    *,
    platform: str | None,
) -> np.ndarray:
    """Infer the imaged footprint conservatively as a uint8 0/255 mask.

    Vizgen MERSCOPE mosaics use exact (or numerically negligible) zeros outside
    acquired FOVs. Small holes inside that footprint can be legitimate dark
    pixels, so they are closed and filled. Other platforms retain the complete
    rectangular support because a zero-valued Xenium pixel is valid background,
    not evidence that it was unacquired.
    """
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DAPI image, got shape {arr.shape}")
    shape_rc = (int(arr.shape[0]), int(arr.shape[1]))
    if str(platform or "").strip().upper() != "MERSCOPE":
        return np.full(shape_rc, 255, dtype=np.uint8)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(shape_rc, 255, dtype=np.uint8)
    magnitude = np.abs(np.asarray(arr, dtype=np.float64))
    scale = float(np.nanpercentile(magnitude[finite], 99.9))
    threshold = max(np.finfo(np.float32).eps * max(scale, 1.0) * 8.0, 0.0)
    observed = finite & (magnitude > threshold)
    observed_fraction = float(np.mean(observed))
    # A sparse signal image does not provide enough evidence to distinguish an
    # acquisition footprint from ordinary background. Full support is safer.
    if observed_fraction < 0.05:
        return np.full(shape_rc, 255, dtype=np.uint8)

    support = ndi.binary_closing(
        observed,
        structure=morphology.disk(1),
    )
    maximum_hole_area = max(16, int(np.ceil(0.00002 * support.size)))
    support = _fill_holes_smaller_than(support, maximum_hole_area)
    minimum_component_area = max(16, int(np.ceil(0.0001 * support.size)))
    support = _retain_components_at_least(support, minimum_component_area)
    support_fraction = float(np.mean(support))
    if support_fraction < 0.05:
        return np.full(shape_rc, 255, dtype=np.uint8)
    if support_fraction >= 0.995:
        support = np.ones(shape_rc, dtype=bool)
    return np.asarray(support.astype(np.uint8) * 255, dtype=np.uint8)


def cosine_support_taper(
    support_mask: Any,
    *,
    width_px: int,
) -> np.ndarray:
    """Return a cosine taper measured inward from an arbitrary support edge."""
    support = np.asarray(support_mask) > 0
    if support.ndim != 2:
        raise ValueError(f"Support mask must be 2D, got shape {support.shape}")
    if int(width_px) <= 0:
        return support.astype(np.float32)
    distance = _distance_inside_support(support)
    normalized = np.clip(
        (distance - 1.0) / float(max(1, int(width_px))),
        0.0,
        1.0,
    )
    taper = np.sin(0.5 * np.pi * normalized) ** 2
    taper[~support] = 0.0
    return np.asarray(taper, dtype=np.float32)


def cosine_edge_taper(
    shape_rc: tuple[int, int],
    *,
    width_px: int,
) -> np.ndarray:
    """Return a smooth image-footprint taper that is zero at the outer edge."""
    height, width = (int(shape_rc[0]), int(shape_rc[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Image shape must be positive, got {shape_rc}")
    return cosine_support_taper(
        np.ones((height, width), dtype=bool),
        width_px=int(width_px),
    )


def create_registration_validity_mask(
    shape_rc: tuple[int, int],
    *,
    pixel_size_um: float,
    edge_exclusion_um: float,
    acquired_support_mask: Any | None = None,
) -> np.ndarray:
    """Mark pixels sufficiently far inside the acquired image footprint."""
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    height, width = (int(shape_rc[0]), int(shape_rc[1]))
    exclusion_px = int(np.ceil(float(edge_exclusion_um) / float(pixel_size_um)))
    support = _coerce_support_mask(
        acquired_support_mask,
        shape_rc=(height, width),
    )
    if exclusion_px <= 0:
        valid = support
    else:
        valid = support & (_distance_inside_support(support) > exclusion_px)
    return np.asarray(valid.astype(np.uint8) * 255, dtype=np.uint8)


def dapi_edge_artifact_metrics(
    image: Any,
    *,
    band_width_px: int = 8,
    acquired_support_mask: Any | None = None,
) -> dict[str, float]:
    """Compare robust intensity at rectangular and acquired-support edges."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DAPI image, got shape {arr.shape}")
    width = min(
        max(1, int(band_width_px)),
        max(1, (min(arr.shape) - 1) // 4),
    )
    rows = np.minimum(np.arange(arr.shape[0]), np.arange(arr.shape[0])[::-1])
    cols = np.minimum(np.arange(arr.shape[1]), np.arange(arr.shape[1])[::-1])
    distance = np.minimum(rows[:, None], cols[None, :])
    edge_values = arr[distance < width]
    interior_values = arr[(distance >= 2 * width) & (distance < 3 * width)]
    edge_p95 = float(np.percentile(edge_values, 95)) if edge_values.size else 0.0
    interior_p95 = (
        float(np.percentile(interior_values, 95)) if interior_values.size else 0.0
    )
    denominator = max(interior_p95, 1.0)
    metrics = {
        "band_width_px": float(width),
        "edge_p95": edge_p95,
        "interior_p95": interior_p95,
        "edge_to_interior_p95_ratio": float(edge_p95 / denominator),
    }
    support = _coerce_support_mask(acquired_support_mask, shape_rc=arr.shape)
    distance = _distance_inside_support(support)
    support_edge_values = arr[support & (distance <= width)]
    support_inner_values = arr[
        support & (distance > 2 * width) & (distance <= 3 * width)
    ]
    support_edge_p95 = (
        float(np.percentile(support_edge_values, 95))
        if support_edge_values.size
        else 0.0
    )
    support_inner_p95 = (
        float(np.percentile(support_inner_values, 95))
        if support_inner_values.size
        else 0.0
    )
    metrics.update(
        {
            "support_fraction": float(np.mean(support)),
            "support_boundary_p95": support_edge_p95,
            "support_inner_p95": support_inner_p95,
            "support_boundary_to_inner_p95_ratio": float(
                support_edge_p95 / max(support_inner_p95, 1.0)
            ),
        }
    )
    return metrics


def _coerce_support_mask(
    support_mask: Any | None,
    *,
    shape_rc: tuple[int, int],
) -> np.ndarray:
    """Return a validated boolean support mask, defaulting to full support."""
    expected_shape = (int(shape_rc[0]), int(shape_rc[1]))
    if support_mask is None:
        return np.ones(expected_shape, dtype=bool)
    support = np.asarray(support_mask) > 0
    if support.shape != expected_shape:
        raise ValueError(
            "Acquired-support mask must match the DAPI image: "
            f"mask={support.shape}, image={expected_shape}"
        )
    if not np.any(support):
        raise ValueError("Acquired-support mask contains no acquired pixels")
    return np.asarray(support, dtype=bool)


def _support_normalized_gaussian(
    image: np.ndarray,
    support: np.ndarray,
    *,
    sigma: float,
    mode: str,
) -> np.ndarray:
    """Gaussian-filter without treating unacquired pixels as zero signal."""
    weights = np.asarray(support, dtype=np.float32)
    numerator = ndi.gaussian_filter(
        np.asarray(image, dtype=np.float32) * weights,
        sigma=float(sigma),
        mode=mode,
    )
    denominator = ndi.gaussian_filter(
        weights,
        sigma=float(sigma),
        mode=mode,
    )
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > np.finfo(np.float32).eps,
    )
    result[~support] = 0.0
    return result


def _distance_inside_support(support: np.ndarray) -> np.ndarray:
    """Measure distance to support boundary, including the array boundary."""
    padded = np.pad(np.asarray(support, dtype=bool), 1, mode="constant")
    return np.asarray(
        ndi.distance_transform_edt(padded)[1:-1, 1:-1],
        dtype=np.float32,
    )


def create_dapi_tissue_mask(
    image: Any,
    *,
    pixel_size_um: float,
    config: DAPIProcessingConfig,
) -> np.ndarray:
    """Create a multi-fragment tissue mask from a processed DAPI image."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D DAPI image, got shape {arr.shape}")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")

    downsample = int(config.mask_downsample)
    small = arr[::downsample, ::downsample]
    small_pixel_size = float(pixel_size_um) * downsample
    sigma_px = max(
        0.5,
        float(config.mask_smoothing_sigma_um) / small_pixel_size,
    )
    density = ndi.gaussian_filter(
        small,
        sigma=sigma_px,
        mode=str(config.background_boundary_mode),
    )
    positive = density[np.isfinite(density)]
    if positive.size == 0 or float(np.nanmax(positive)) <= float(np.nanmin(positive)):
        raise ValueError("DAPI image has no intensity variation for tissue masking")

    threshold = filters.threshold_otsu(positive)
    mask = density > threshold
    closing_radius = int(
        np.ceil(float(config.mask_closing_radius_um) / small_pixel_size)
    )
    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))

    min_area_px = max(
        1,
        int(np.ceil(float(config.mask_min_area_um2) / (small_pixel_size**2))),
    )
    hole_area_px = max(
        1,
        int(np.ceil(float(config.mask_hole_area_um2) / (small_pixel_size**2))),
    )
    mask = _retain_components_at_least(mask, min_area_px)
    mask = _fill_holes_smaller_than(mask, hole_area_px)
    dilation_radius = int(np.ceil(float(config.mask_dilation_um) / small_pixel_size))
    if dilation_radius > 0:
        mask = morphology.dilation(mask, morphology.disk(dilation_radius))

    if not np.any(mask):
        raise ValueError("DAPI tissue masking removed all foreground")
    upsampled = ndi.zoom(
        mask.astype(np.uint8),
        zoom=(arr.shape[0] / mask.shape[0], arr.shape[1] / mask.shape[1]),
        order=0,
        prefilter=False,
    )
    upsampled = upsampled[: arr.shape[0], : arr.shape[1]]
    if upsampled.shape != arr.shape:
        padded = np.zeros(arr.shape, dtype=np.uint8)
        padded[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
        upsampled = padded
    return np.asarray((upsampled > 0).astype(np.uint8) * 255, dtype=np.uint8)


def mask_outline(mask: Any) -> np.ndarray:
    """Return a one-pixel uint8 outline for a binary tissue mask."""
    binary = np.asarray(mask) > 0
    eroded = morphology.erosion(binary, morphology.disk(1))
    return np.asarray((binary & ~eroded).astype(np.uint8) * 255, dtype=np.uint8)


def _retain_components_at_least(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    labels, count = ndi.label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(minimum_area)
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def _fill_holes_smaller_than(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    holes = ndi.binary_fill_holes(binary) & ~binary
    labels, count = ndi.label(holes)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    fill = sizes < int(maximum_area)
    fill[0] = False
    return np.asarray(binary | fill[labels], dtype=bool)


class DapiImageProcesser(_ValisImageProcesser):
    """VALIS ``ImageProcesser`` specialized for smoothed DAPI morphology."""

    pixel_size_um: float = 1.0
    processing_config: DAPIProcessingConfig = DAPIProcessingConfig()
    input_is_preprocessed: bool = False

    def process_image(
        self: DapiImageProcesser,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the configured single-channel uint8 registration image."""
        del args
        pixel_size_um = float(kwargs.pop("pixel_size_um", self.pixel_size_um))
        input_is_preprocessed = bool(
            kwargs.pop("input_is_preprocessed", self.input_is_preprocessed)
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown DAPI processor arguments: {unknown}")
        dapi = select_dapi_channel(self.image)
        if input_is_preprocessed:
            return np.clip(dapi, 0, 255).astype(np.uint8, copy=False)
        return process_dapi_image(
            dapi,
            pixel_size_um=pixel_size_um,
            config=self.processing_config,
        )

    def create_mask(self: DapiImageProcesser) -> np.ndarray:
        """Create a multi-component DAPI tissue mask for VALIS."""
        dapi = select_dapi_channel(self.image)
        processed = (
            np.clip(dapi, 0, 255).astype(np.uint8, copy=False)
            if self.input_is_preprocessed
            else process_dapi_image(
                dapi,
                pixel_size_um=float(self.pixel_size_um),
                config=self.processing_config,
            )
        )
        return create_dapi_tissue_mask(
            processed,
            pixel_size_um=float(self.pixel_size_um),
            config=self.processing_config,
        )


def configured_dapi_processor_class(
    *,
    pixel_size_um: float,
    config: DAPIProcessingConfig,
    input_is_preprocessed: bool,
) -> type[DapiImageProcesser]:
    """Create a VALIS processor class carrying run-specific physical settings."""

    class ConfiguredDapiImageProcesser(DapiImageProcesser):
        pass

    ConfiguredDapiImageProcesser.__name__ = "ConfiguredDapiImageProcesser"
    ConfiguredDapiImageProcesser.pixel_size_um = float(pixel_size_um)
    ConfiguredDapiImageProcesser.processing_config = config.model_copy(deep=True)
    ConfiguredDapiImageProcesser.input_is_preprocessed = bool(input_is_preprocessed)
    return ConfiguredDapiImageProcesser


def tissue_fragment_count(mask: Any) -> int:
    """Return the number of disconnected foreground fragments."""
    return int(measure.label(np.asarray(mask) > 0, connectivity=2).max())
