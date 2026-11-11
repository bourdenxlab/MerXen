"""Read-only export of native, production-context Cellpose annotation pilots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tarfile
from dataclasses import asdict, dataclass
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import spatialdata as sd
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label
from spatialdata.transformations import get_transformation

from merxen.io.bead_removal.provenance import (
    check_segmentation_sidecar,
    read_image_preprocessing_attr,
)
from merxen.io.image_source import (
    build_image_source,
    fetch_tile,
    image_to_cyx,
    prepare_cellpose_input,
)
from merxen.platforms import find_um_per_px, platform_spec
from merxen.segmentation.cellpose import iter_core_tiles
from merxen.segmentation.hitl.regions import RoiContext, read_viewer_regions
from merxen.segmentation.model_identity import (
    read_model_identity,
    require_cellpose_runtime,
)

GUIDE_VERSION = "pilot-v2"


class ChannelNormalization(TypedDict):
    """Percentile bounds and the actual batch/row/column sampling strides."""

    low: float
    high: float
    scale: float
    bounds_dtype: str
    stride: list[int]


class TileNormalization(TypedDict):
    """Recorded normalization of the complete enclosing production tile."""

    batch_axis: bool
    percentile: list[int]
    channels: list[ChannelNormalization]


class ViewerExportSettings(TypedDict):
    """Portable descriptor for the viewer's local correction-package exporter."""

    schema_version: int
    model_channels: list[str]
    input_normalization: str
    transcript_density: dict[str, Any] | None
    baseline_stats_model_identity: dict[str, Any] | None
    guide_version: str
    training_normalize: bool
    source_metadata_sha256: NotRequired[str]


class CropBox(BaseModel):
    """A native-pixel crop, with legacy size-only squares still supported."""

    y0: int = Field(ge=0)
    x0: int = Field(ge=0)
    size: int = Field(default=768, ge=256, le=1024)
    height: int | None = Field(default=None, ge=256)
    width: int | None = Field(default=None, ge=256)
    roi_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")

    @property
    def height_px(self) -> int:
        """Return height in pixels, falling back to the legacy square size."""
        return self.height if self.height is not None else self.size

    @property
    def width_px(self) -> int:
        """Return width in pixels, falling back to the legacy square size."""
        return self.width if self.width is not None else self.size


class HitlExportConfig(BaseModel):
    """Configuration of an unstratified annotation pilot, not a test split.

    The source and prefill labels are read only. ``baseline_stats_path`` pins
    the original full-image Cellpose grid, even when memory probing selected a
    smaller tile. Explicit boxes bypass tissue sampling but retain core margins.
    """

    model_config = ConfigDict(extra="forbid")

    source_zarr: Path
    baseline_stats_path: Path
    output_dir: Path
    protected_roots: list[Path] = Field(min_length=1)
    platform: Literal["ATERA", "XENIUM"] = "ATERA"
    dataset_alias: str = Field(default="ATERA_PILOT", pattern=r"^[A-Z][A-Z0-9_]*$")
    mask_key: str = "cellpose_cell_labels"
    baseline_mask_path: Path | None = None
    channels: list[str] = Field(default_factory=lambda: ["DAPI", "18S"])
    view_channels: list[str] | None = None
    n_crops: int = Field(default=6, ge=1, le=100)
    crop_size_px: int = Field(default=768, ge=256, le=1024)
    margin_px: int = Field(default=656, ge=0)
    min_tissue_fraction: float = Field(default=0.9, gt=0, le=1)
    seed: int = 0
    boxes: list[CropBox] | None = None
    roi_geojson_path: Path | None = None
    transcript_density_path: Path | None = None
    input_normalization: Literal["joint", "per_channel"] | None = None

    @model_validator(mode="after")
    def _validate_channels(self) -> HitlExportConfig:
        for channels in (self.channels, self.view_channels):
            if channels is not None and (
                not 1 <= len(channels) <= 3
                or len(set(channels)) != len(channels)
                or any(not name.strip() for name in channels)
            ):
                raise ValueError("Select one to three unique, nonempty channels")
        if self.boxes is not None and not self.boxes:
            raise ValueError("Explicit boxes cannot be empty")
        if self.boxes is not None and self.roi_geojson_path is not None:
            raise ValueError("Choose either boxes or roi_geojson_path")
        return self


@dataclass(frozen=True)
class HitlExportResult:
    """Locations of a completed annotation pilot and its portable archive."""

    output_dir: Path
    package_path: Path
    crop_count: int


def normalize_training_tile(
    model_input: np.ndarray,
) -> tuple[np.ndarray, TileNormalization]:
    """Apply Cellpose's default normalization to a whole prepared halo tile.

    Args:
        model_input: Exact production uint8 YXC input, after joint p2/p98.

    Returns:
        Float32 YXC route-A input and the per-channel percentile/stride record.
        Constant channels are zero, including the padded third channel.
    """
    from cellpose import models, transforms

    # eval adds a batch axis before normalize_img. Its percentile subsampling
    # depends on that shape, so normalizing a crop or an unbatched tile differs.
    batched = model_input[np.newaxis, ...].astype(np.float32)
    records: list[ChannelNormalization] = []
    for channel in range(batched.shape[-1]):
        plane = batched[..., channel]
        stride = (
            [max(1, plane.shape[0] // 50)]
            + [max(1, size // 224) for size in plane.shape[1:]]
            if plane.size > 224**3
            else [1] * plane.ndim
        )
        sampled = plane[tuple(slice(None, None, step) for step in stride)]
        # Scalar calls preserve float32 in supported NumPy builds; a joint
        # percentile call can promote the bounds and change rounded inputs.
        low = np.percentile(sampled, 1.0)
        high = np.percentile(sampled, 99.0)
        records.append(
            {
                "low": float(low),
                "high": float(high),
                "scale": float(high - low),
                "bounds_dtype": str(np.asarray(low).dtype),
                "stride": stride,
            }
        )
    normalized = transforms.normalize_img(
        batched, **{**models.normalize_default, "normalize": True}
    )
    # Check the recorded contract independently, including constant channels.
    reconstructed = np.zeros_like(normalized)
    for channel, record in enumerate(records):
        if record["scale"] > 1e-3:
            plane = model_input[..., channel].astype(np.float32)
            bounds_type = np.dtype(record["bounds_dtype"]).type
            plane -= bounds_type(record["low"])
            plane /= bounds_type(record["scale"])
            reconstructed[..., channel] = plane
    np.testing.assert_array_equal(normalized, reconstructed)
    return normalized[0], {
        "batch_axis": True,
        "percentile": [1, 99],
        "channels": records,
    }


def split_crop_labels(mask: np.ndarray) -> np.ndarray:
    """Give each eight-connected component of each prefill label its own ID.

    Args:
        mask: Integer label crop; zero is background.

    Returns:
        Dense uint32 labels, preserving boundaries between touching cells.
    """
    if mask.ndim != 2 or mask.dtype.kind not in "iu" or np.any(mask < 0):
        raise ValueError("Prefill must be a nonnegative, two-dimensional integer mask")
    return np.asarray(label(mask, background=0, connectivity=2), dtype=np.uint32)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _native_pixel_size(config: HitlExportConfig, sdata: Any) -> float:
    try:
        value = find_um_per_px(config.source_zarr, config.platform)
    except FileNotFoundError:
        # Review exports retain the selected vendor metadata, including any
        # build override, even when they omit the corresponding spec sidecars.
        value = (
            float(
                sdata.attrs.get("merxen_atera", {})
                .get("experiment", {})
                .get("pixel_size", 0)
            )
            if config.platform == "ATERA"
            else 0
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Source has no valid native pixel size; no host/platform fallback is used"
        )
    return value


def _make_roi_context(
    config: HitlExportConfig, height: int, width: int, stats: dict, pixel_size: float
) -> RoiContext:
    if (int(stats["height"]), int(stats["width"])) != (height, width):
        raise ValueError("Baseline tile grid dimensions differ from the native image")
    metadata = next(
        (
            config.source_zarr / name
            for name in ("zarr.json", ".zattrs")
            if (config.source_zarr / name).is_file()
        ),
        None,
    )
    if metadata is None:
        raise ValueError("Source lacks root metadata for binding viewer selections")
    return RoiContext(
        platform=config.platform,
        image_key=platform_spec(config.platform).native_image_key,
        dataset_alias=config.dataset_alias,
        height=height,
        width=width,
        pixel_size_um=pixel_size,
        tile_size=int(stats["tile_size"]),
        overlap_px=int(stats["stitch_overlap_px"]),
        margin_px=config.margin_px,
        crop_size_px=config.crop_size_px,
        source_metadata_sha256=_sha256(metadata),
        baseline_stats_sha256=_sha256(config.baseline_stats_path),
    )


def prepare_viewer_roi_context(
    config: HitlExportConfig, output_path: Path
) -> RoiContext:
    """Write portable native-frame context for selecting crops in the viewer.

    Args:
        config: Original source, baseline tile grid and intended crop size.
        output_path: New JSON file outside the source and protected outputs.

    Returns:
        Context consumed by the lightweight napari ROI helper.
    """
    resolved = output_path.resolve()
    if any(
        resolved.is_relative_to(root.resolve())
        for root in [config.source_zarr, *config.protected_roots]
    ):
        raise ValueError("ROI context must be outside source/protected roots")
    if output_path.exists():
        raise ValueError("ROI context already exists; choose a new filename")
    sdata = sd.read_zarr(config.source_zarr, selection=("images",))
    key = platform_spec(config.platform).native_image_key
    if key not in sdata.images:
        raise ValueError("Source lacks its platform's native image")
    image = image_to_cyx(sdata.images[key])
    stats = json.loads(config.baseline_stats_path.read_text())
    context = _make_roi_context(
        config,
        int(image.sizes["y"]),
        int(image.sizes["x"]),
        stats,
        _native_pixel_size(config, sdata),
    )
    export_settings = viewer_export_settings(config)
    export_settings["source_metadata_sha256"] = context.source_metadata_sha256
    companion = output_path.parent / "hitl_export_settings.json"
    if companion.exists() and json.loads(companion.read_text()) != export_settings:
        raise ValueError("Crop export settings already exist; use a fresh setup folder")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        stream.write(json.dumps(asdict(context), indent=2) + "\n")
    if not companion.exists():
        with companion.open("x") as stream:
            stream.write(json.dumps(export_settings, indent=2) + "\n")
    return context


def viewer_export_settings(config: HitlExportConfig) -> ViewerExportSettings:
    """Build portable channel/normalization settings for local GUI packages.

    Args:
        config: Original source, baseline statistics and optional density input.

    Returns:
        The companion descriptor consumed by the viewer's annotation manager.
    """
    stats = json.loads(config.baseline_stats_path.read_text())
    identity = stats.get("model_identity") or {}
    density = (identity.get("inference_context") or {}).get("transcript_density")
    path = config.transcript_density_path
    from merxen.segmentation.density_storage import has_embedded_density

    if (
        "channels" in config.model_fields_set
        and "transcript_density" not in config.channels
    ):
        path, density = None, None
    elif path is None and has_embedded_density(config.source_zarr):
        path = config.source_zarr
    if path is None and density:
        path = Path(density["raster_path"])
    if path is not None:
        from merxen.segmentation.transcript_density import density_input_identity

        verified = density_input_identity(path)
        assert verified is not None
        density = verified.model_dump(mode="json")
    channels = config.channels
    if density is not None and "channels" not in config.model_fields_set:
        channels = ["DAPI", "18S", "transcript_density"]
    normalization = config.input_normalization or (
        "per_channel"
        if "transcript_density" in channels
        else (identity.get("cellpose_config") or {}).get("input_normalization", "joint")
    )
    return {
        "schema_version": 1,
        "model_channels": channels,
        "input_normalization": normalization,
        "transcript_density": density,
        "baseline_stats_model_identity": stats.get("model_identity"),
        "guide_version": GUIDE_VERSION,
        "training_normalize": False,
    }


def _require_fresh_output(config: HitlExportConfig) -> None:
    output = config.output_dir.resolve()
    source = config.source_zarr.resolve()
    roots = [source, *[root.resolve() for root in config.protected_roots]]
    if any(
        output.is_relative_to(root) or root.is_relative_to(output) for root in roots
    ):
        raise ValueError(
            "Annotation output must be separate from source/protected roots"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            "Annotation output must be new or empty; edits are never replaced"
        )


def _thumbnail(image: Any, channel: str) -> np.ndarray:
    if hasattr(image, "children"):
        levels = [image_to_cyx(node.ds["image"]) for node in image.children.values()]
        array = min(levels, key=lambda value: value.sizes["y"] * value.sizes["x"])
    else:
        array = image_to_cyx(image)
    array = array.sel(c=channel)
    step = max(1, math.ceil(max(array.shape) / 1600))
    return np.asarray(
        array.data[::step, ::step].compute()
        if hasattr(array.data, "compute")
        else array.data[::step, ::step]
    )


def _eligible_boxes(
    tiles: list[dict[str, int]], size: int, margin: int
) -> list[tuple[CropBox, int]]:
    result = []
    for index, tile in enumerate(tiles):
        for y0 in range(
            tile["core_y0"] + margin, tile["core_y1"] - margin - size + 1, size
        ):
            for x0 in range(
                tile["core_x0"] + margin, tile["core_x1"] - margin - size + 1, size
            ):
                result.append((CropBox(y0=y0, x0=x0, size=size), index))
    return result


def _choose_boxes(
    config: HitlExportConfig,
    tiles: list[dict[str, int]],
    thumbnail: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[CropBox, int]]:
    if config.boxes is not None:
        result: list[tuple[CropBox, int]] = []
        for box in config.boxes:
            owners = [
                i
                for i, tile in enumerate(tiles)
                if box.y0 >= tile["core_y0"] + config.margin_px
                and box.x0 >= tile["core_x0"] + config.margin_px
                and box.y0 + box.height_px <= tile["core_y1"] - config.margin_px
                and box.x0 + box.width_px <= tile["core_x1"] - config.margin_px
            ]
            if len(owners) != 1:
                raise ValueError(
                    "Each crop must lie inside one tile core with the requested margin"
                )
            if any(
                box.y0 < other.y0 + other.height_px
                and box.y0 + box.height_px > other.y0
                and box.x0 < other.x0 + other.width_px
                and box.x0 + box.width_px > other.x0
                for other, _ in result
            ):
                raise ValueError("Explicit annotation crops overlap")
            result.append((box, owners[0]))
        return result
    if not np.isfinite(thumbnail).all() or np.ptp(thumbnail) == 0:
        raise ValueError("Native thumbnail contains no usable tissue contrast")
    # Log contrast elevates scanner/background noise almost to tissue levels.
    # Smooth linear DAPI instead, then fill enclosed low-DAPI tissue regions.
    signal = ndi.gaussian_filter(np.maximum(thumbnail.astype(np.float32), 0), sigma=2)
    foreground = ndi.binary_fill_holes(
        ndi.binary_closing(signal > threshold_otsu(signal), iterations=2)
    )
    candidates = []
    for box, index in _eligible_boxes(tiles, config.crop_size_px, config.margin_px):
        y0, x0 = (
            int(box.y0 * foreground.shape[0] / shape[0]),
            int(box.x0 * foreground.shape[1] / shape[1]),
        )
        y1 = math.ceil((box.y0 + box.size) * foreground.shape[0] / shape[0])
        x1 = math.ceil((box.x0 + box.size) * foreground.shape[1] / shape[1])
        if foreground[y0:y1, x0:x1].mean() >= config.min_tissue_fraction:
            candidates.append((box, index))
    if len(candidates) < config.n_crops:
        raise ValueError(
            "Too few tissue-positive windows within tile cores; "
            "inspect size/margin/tissue settings"
        )
    rng = np.random.default_rng(config.seed)
    chosen = [candidates.pop(int(rng.integers(len(candidates))))]
    while len(chosen) < config.n_crops:
        distances = [
            min(
                (box.y0 - other.y0) ** 2 + (box.x0 - other.x0) ** 2
                for other, _ in chosen
            )
            for box, _ in candidates
        ]
        chosen.append(candidates.pop(int(np.argmax(distances))))
    return chosen


def _save_overview(
    path: Path,
    thumbnail: np.ndarray,
    boxes: list[tuple[CropBox, int]],
    shape: tuple[int, int],
) -> None:
    figure = Figure(figsize=(8, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.imshow(
        np.log1p(np.maximum(thumbnail, 0)),
        cmap="gray",
        extent=(0, shape[1], shape[0], 0),
    )
    for index, (box, _) in enumerate(boxes, 1):
        axes.add_patch(
            Rectangle(
                (box.x0, box.y0),
                box.width_px,
                box.height_px,
                fill=False,
                edgecolor="cyan",
            )
        )
        axes.text(box.x0, box.y0, str(index), color="cyan")
    axes.set(
        xlabel="Native x (pixels)",
        ylabel="Native y (pixels)",
        title="Annotation pilot crop locations",
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _save_crop_preview(path: Path, view: np.ndarray, outlines: np.ndarray) -> None:
    display = np.zeros((*view.shape[1:], 3), dtype=np.float32)
    for index, target in enumerate((2, 1, 0)[: len(view)]):
        low, high = np.percentile(view[index], (2, 99))
        display[..., target] = np.clip(
            (view[index].astype(np.float32) - low) / max(high - low, 1e-8), 0, 1
        )
    overlay = display.copy()
    overlay[outlines > 0] = [1, 0.8, 0]
    figure = Figure(figsize=(10, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, image, title in zip(
        axes, (display, overlay), ("Native stains", "Current prefill"), strict=True
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=120, bbox_inches="tight")


def export_annotation_pilot(config: HitlExportConfig) -> HitlExportResult:
    """Export a portable correction package without modifying the input store.

    Args:
        config: Source, prefill, production tile grid and pilot sampling settings.

    Returns:
        Completed folder, archive and number of crops. No training is performed.

    Raises:
        ValueError: For unsafe output, mismatched grids/channels or invalid crops.
    """
    _require_fresh_output(config)
    runtime = require_cellpose_runtime()
    spec = platform_spec(config.platform)
    # Avoid loading any transcript partitions or whole-transcriptome tables.
    sdata = sd.read_zarr(config.source_zarr, selection=("images", "labels"))
    if spec.native_image_key not in sdata.images:
        raise ValueError("Source lacks its platform's native image")
    image = sdata.images[spec.native_image_key]
    native = image_to_cyx(image)
    stats = json.loads(config.baseline_stats_path.read_text())
    context = (stats.get("model_identity") or {}).get("inference_context") or {}
    density_identity = context.get("transcript_density")
    density_path = config.transcript_density_path
    from merxen.segmentation.density_storage import has_embedded_density

    if (
        "channels" in config.model_fields_set
        and "transcript_density" not in config.channels
    ):
        density_path, density_identity = None, None
    elif density_path is None and has_embedded_density(config.source_zarr):
        density_path = config.source_zarr
    if density_path is None and density_identity:
        density_path = Path(density_identity["raster_path"])
    if density_path is not None:
        if config.platform != "ATERA":
            raise ValueError("Transcript-density HITL input is Atera-only")
        from merxen.segmentation.transcript_density import (
            density_input_identity,
            image_with_density,
        )

        native = image_with_density(sdata, config.source_zarr, density_path)
        actual = density_input_identity(density_path)
        assert actual is not None
        if (
            config.transcript_density_path is None
            and density_identity is not None
            and actual.fingerprint != density_identity["fingerprint"]
        ):
            raise ValueError(
                "The inferred density raster differs from the saved mask input"
            )
        density_identity = actual.model_dump(mode="json")
        if "channels" not in config.model_fields_set:
            config = config.model_copy(
                update={"channels": ["DAPI", "18S", "transcript_density"]}
            )
    normalization = config.input_normalization or (
        "per_channel"
        if "transcript_density" in config.channels
        else ((stats.get("model_identity") or {}).get("cellpose_config") or {}).get(
            "input_normalization", "joint"
        )
    )
    if "transcript_density" in config.channels and normalization != "per_channel":
        raise ValueError("Density training inputs require per-channel normalization")
    available = [str(value) for value in native.coords["c"].values]
    view_channels = config.view_channels or list(config.channels)
    boundary = "ATP1A1/CD45/E-Cadherin"
    if (
        config.view_channels is None
        and boundary in available
        and boundary not in view_channels
        and len(view_channels) < 3
    ):
        view_channels.append(boundary)
    if any(channel not in available for channel in config.channels + view_channels):
        raise ValueError("Native image is missing requested model/view channels")
    height, width = int(native.sizes["y"]), int(native.sizes["x"])
    if (int(stats["height"]), int(stats["width"])) != (height, width):
        raise ValueError("Baseline tile grid dimensions differ from the native image")
    tile_size, overlap = int(stats["tile_size"]), int(stats["stitch_overlap_px"])
    tiles = iter_core_tiles(height, width, tile_size, overlap)
    source_preprocessing = read_image_preprocessing_attr(config.source_zarr)
    image_affine = get_transformation(
        image, to_coordinate_system="global"
    ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    if config.baseline_mask_path is not None:
        check_segmentation_sidecar(
            config.baseline_mask_path.resolve(),
            source_preprocessing,
            dataset_name=config.dataset_alias,
        )
        masks = np.load(config.baseline_mask_path, mmap_mode="r", allow_pickle=False)
        baseline_identity = read_model_identity(config.baseline_mask_path).model_dump(
            mode="json"
        )
    else:
        if config.mask_key not in sdata.labels:
            raise ValueError("Source lacks the selected prefill label layer")
        mask_element = sdata.labels[config.mask_key]
        label_affine = get_transformation(
            mask_element, to_coordinate_system="global"
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        if not np.allclose(image_affine, label_affine):
            raise ValueError(
                "Prefill labels and native image use different coordinate grids"
            )
        masks = image_to_cyx(mask_element).isel(c=0).data
        baseline_identity = {
            "status": "unverified",
            "evidence": (
                "Prefill read from a label layer; "
                "original checkpoint identity not inferred"
            ),
        }
    if masks.shape != (height, width):
        raise ValueError("Prefill mask shape differs from the native image")
    pixel_size = _native_pixel_size(config, sdata)
    roi_context = _make_roi_context(config, height, width, stats, pixel_size)
    selection_config = config
    if config.roi_geojson_path is not None:
        boxes = [
            CropBox(
                y0=box.y0,
                x0=box.x0,
                height=box.height_px,
                width=box.width_px,
                roi_id=box.roi_id,
            )
            for box in read_viewer_regions(config.roi_geojson_path, roi_context)
        ]
        selection_config = config.model_copy(update={"boxes": boxes})
    thumbnail = _thumbnail(image, "DAPI" if "DAPI" in available else config.channels[0])
    selected = _choose_boxes(selection_config, tiles, thumbnail, (height, width))
    source = build_image_source(
        native, requested_channels=config.channels, as_float32=True
    )
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "model_input", "context"):
        (output / name).mkdir()
    _save_overview(
        output / "context" / "overview.png", thumbnail, selected, (height, width)
    )
    guide = resource_files("merxen").joinpath("assets/hitl/annotation-guide.md")
    (output / "annotation-guide.md").write_text(guide.read_text())
    (output / "roi_context.json").write_text(
        json.dumps(asdict(roi_context), indent=2) + "\n"
    )
    if config.roi_geojson_path is not None:
        (output / "selected_regions.geojson").write_bytes(
            config.roi_geojson_path.read_bytes()
        )
    rows: list[dict[str, Any]] = []
    # Group by the global tile so normalization is done once per distinct halo.
    for tile_index in sorted({tile_index for _, tile_index in selected}):
        tile = tiles[tile_index]
        halo = fetch_tile(
            source, tile["tile_y0"], tile["tile_y1"], tile["tile_x0"], tile["tile_x1"]
        )
        p2, p98 = np.percentile(
            halo, (2, 98), axis=(0, 1) if normalization == "per_channel" else None
        )
        _, model_input, _ = prepare_cellpose_input(halo, normalization=normalization)
        normalized, normalization_record = normalize_training_tile(model_input)
        del halo, model_input
        for crop_index, (box, owner) in enumerate(selected, 1):
            if owner != tile_index:
                continue
            crop_id = f"{config.dataset_alias}_pilot_{crop_index:03d}"
            y0, y1, x0, x1 = (
                box.y0,
                box.y0 + box.height_px,
                box.x0,
                box.x0 + box.width_px,
            )
            local_y, local_x = y0 - tile["tile_y0"], x0 - tile["tile_x0"]
            raw = np.asarray(
                native.data[:, y0:y1, x0:x1].compute()
                if hasattr(native.data, "compute")
                else native.data[:, y0:y1, x0:x1]
            )
            labels = split_crop_labels(
                np.asarray(
                    masks[y0:y1, x0:x1].compute()
                    if hasattr(masks, "compute")
                    else masks[y0:y1, x0:x1]
                )
            )
            from cellpose.utils import masks_to_outlines

            outlines = labels * masks_to_outlines(labels)
            view = raw[[available.index(channel) for channel in view_channels]]
            _save_crop_preview(
                output / "context" / f"{crop_id}_preview.png", view, outlines
            )
            tifffile.imwrite(
                output / f"{crop_id}.tif",
                view.astype(np.float32),
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )
            np.save(
                output / f"{crop_id}_seg.npy",
                np.array(
                    {
                        "masks": labels,
                        "outlines": outlines,
                        "filename": f"{crop_id}.tif",
                        "ismanual": np.zeros(int(labels.max()), dtype=bool),
                    },
                    dtype=object,
                ),
            )
            np.save(output / "raw" / f"{crop_id}.npy", raw)
            np.save(
                output / "model_input" / f"{crop_id}.npy",
                np.moveaxis(
                    normalized[
                        local_y : local_y + box.height_px,
                        local_x : local_x + box.width_px,
                    ],
                    -1,
                    0,
                ),
            )
            rows.append(
                {
                    "crop_id": crop_id,
                    "roi_id": box.roi_id,
                    "split": "pilot_train",
                    "stratum": "unstratified_pilot",
                    "bbox_y0_y1_x0_x1": [y0, y1, x0, x1],
                    "bbox_um_y0_y1_x0_x1": [
                        value * pixel_size for value in (y0, y1, x0, x1)
                    ],
                    "tile_index": owner,
                    "tile": tile,
                    "normalization": {
                        "route": "A",
                        "input_normalization": normalization,
                        (
                            "per_channel_p2_p98"
                            if normalization == "per_channel"
                            else "joint_p2_p98"
                        ): [np.asarray(p2).tolist(), np.asarray(p98).tolist()],
                        **normalization_record,
                    },
                    "prefill_cells": int(labels.max()),
                }
            )
        del normalized
    rows.sort(key=lambda row: row["crop_id"])
    manifest = {
        "schema_version": 1,
        "exporter_source_sha256": _sha256(Path(__file__)),
        "guide_version": GUIDE_VERSION,
        "platform": config.platform,
        "dataset_alias": config.dataset_alias,
        "image_key": spec.native_image_key,
        "source_zarr": str(config.source_zarr.resolve()),
        "image_to_global_affine": image_affine.tolist(),
        "image_preprocessing": source_preprocessing,
        "baseline_stats_model_identity": stats.get("model_identity"),
        "source_mask": str(config.baseline_mask_path.resolve())
        if config.baseline_mask_path
        else config.mask_key,
        "baseline_model_identity": baseline_identity,
        "baseline_stats_sha256": _sha256(config.baseline_stats_path),
        "pixel_size_um": pixel_size,
        "raw_channels": available,
        "model_channels": config.channels,
        "input_normalization": normalization,
        "transcript_density": density_identity,
        "view_channels": view_channels,
        "cellpose_version": runtime[0],
        "torch_version": runtime[1],
        "sampling": {
            "method": "viewer_rois"
            if config.roi_geojson_path
            else "explicit_boxes"
            if config.boxes
            else "tissue_spread_pilot",
            "roi_geojson_sha256": _sha256(config.roi_geojson_path)
            if config.roi_geojson_path
            else None,
            "seed": config.seed,
            "min_tissue_fraction": config.min_tissue_fraction,
            "uses_baseline_errors": False,
            "tissue_mask": "smoothed_linear_DAPI_otsu_closed_filled",
            "thumbnail_smoothing_sigma": 2,
        },
        "training_normalize": False,
        "evaluation_eligible": False,
        "crops": rows,
    }
    for row in rows:
        crop_id = row["crop_id"]
        row["files_sha256"] = {
            str(path.relative_to(output)): _sha256(path)
            for path in (
                output / f"{crop_id}.tif",
                output / f"{crop_id}_seg.npy",
                output / "raw" / f"{crop_id}.npy",
                output / "model_input" / f"{crop_id}.npy",
            )
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "review.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["crop_id", "status", "reason", "minutes", "notes"])
        writer.writerows([[row["crop_id"], "pending", "", "", ""] for row in rows])
    files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    package = output / "annotation-pilot.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        for path in [*files, output / "SHA256SUMS"]:
            archive.add(path, arcname=f"annotation-pilot/{path.relative_to(output)}")
    return HitlExportResult(output, package, len(rows))
