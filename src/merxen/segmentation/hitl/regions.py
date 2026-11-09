"""Portable native-pixel ROI exchange between a viewer and the HITL exporter.

Only NumPy and the standard library are imported, so this module also runs in
an existing napari environment without the pipeline's scientific dependencies.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merxen.platforms import platform_spec


@dataclass(frozen=True)
class RoiContext:
    """Native image and production-grid identity carried by a viewer ROI file."""

    platform: str
    image_key: str
    dataset_alias: str
    height: int
    width: int
    pixel_size_um: float
    tile_size: int
    overlap_px: int
    margin_px: int
    crop_size_px: int
    source_metadata_sha256: str
    baseline_stats_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.platform not in ("ATERA", "XENIUM"):
            raise ValueError("Unsupported HITL ROI context")
        if self.image_key != platform_spec(self.platform).native_image_key:
            raise ValueError("ROI selection requires the native morphology image")
        if not math.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError("ROI context requires a positive native pixel size")
        if min(self.height, self.width, self.tile_size) <= 0 or (
            self.overlap_px < 0
            or self.tile_size <= 2 * self.overlap_px
            or self.margin_px < 0
            or not 256 <= self.crop_size_px <= 1024
        ):
            raise ValueError("Invalid native ROI grid or crop size")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (self.source_metadata_sha256, self.baseline_stats_sha256)
        ):
            raise ValueError("ROI context requires source and baseline-grid checksums")

    @classmethod
    def read(cls, path: Path) -> RoiContext:
        """Read a context file produced by cellpose-hitl-prepare-rois.

        Args:
            path: Portable context JSON; source data need not exist locally.

        Returns:
            Validated native image/grid identity.
        """
        return cls(**json.loads(path.read_text()))

    def placement_windows(self) -> list[tuple[int, int, int, int]]:
        """Return legal crop-origin intervals as (y_min, y_max, x_min, x_max)."""
        core = self.tile_size - 2 * self.overlap_px
        windows = []
        for y in range(0, self.height, core):
            for x in range(0, self.width, core):
                bounds = (
                    y + self.margin_px,
                    min(y + core, self.height) - self.margin_px - self.crop_size_px,
                    x + self.margin_px,
                    min(x + core, self.width) - self.margin_px - self.crop_size_px,
                )
                if bounds[1] >= bounds[0] and bounds[3] >= bounds[2]:
                    windows.append(bounds)
        return windows


@dataclass(frozen=True)
class ViewerRegion:
    """One user-selected square crop in native level-zero pixel coordinates."""

    roi_id: str
    y0: int
    x0: int
    size: int
    height: int | None = None
    width: int | None = None

    @property
    def height_px(self) -> int:
        """Return the crop height, retaining legacy square selections."""
        return self.height if self.height is not None else self.size

    @property
    def width_px(self) -> int:
        """Return the crop width, retaining legacy square selections."""
        return self.width if self.width is not None else self.size


def _parse_regions(payload: dict[str, Any], expected: RoiContext) -> list[ViewerRegion]:
    if (
        payload.get("type") != "FeatureCollection"
        or payload.get("coordinate_system") != "native_pixels"
    ):
        raise ValueError("Use the HITL viewer helper to export native-pixel GeoJSON")
    if RoiContext(**payload.get("merxen_hitl_context", {})) != expected:
        raise ValueError(
            "Viewer ROIs refer to a different source, tile grid or crop context"
        )
    regions_version = payload.get("merxen_hitl_regions_version", 1)
    if regions_version not in (1, 2):
        raise ValueError("Unsupported HITL region geometry version")
    regions: list[ViewerRegion] = []
    seen = set()
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        if (
            geometry.get("type") != "Polygon"
            or len(geometry.get("coordinates", [])) != 1
        ):
            raise ValueError("Each HITL region must be one square without holes")
        ring = np.asarray(geometry["coordinates"][0], dtype=float)
        if (
            ring.shape != (5, 2)
            or not np.isfinite(ring).all()
            or not np.array_equal(ring[0], ring[-1])
        ):
            raise ValueError("Each ROI needs four finite corners and a closing vertex")
        rounded = np.rint(ring).astype(np.int64)
        if np.any(abs(ring - rounded) > 1e-6):
            raise ValueError("ROI coordinates must lie on native integer pixel edges")
        x0, y0 = rounded.min(axis=0)
        x1, y1 = rounded.max(axis=0)
        height, width = int(y1 - y0), int(x1 - x0)
        if min(height, width) < 256:
            raise ValueError(
                "HITL crops must be at least 256 native pixels on each axis"
            )
        if regions_version == 1 and (width != height or width > 1024):
            raise ValueError(
                "Legacy HITL files require square crops of 256–1024 pixels"
            )
        if x0 < 0 or y0 < 0 or x1 > expected.width or y1 > expected.height:
            raise ValueError("HITL regions must lie inside the native image")
        corners = {(x0, y0), (x1, y0), (x1, y1), (x0, y1)}
        edges = np.diff(rounded, axis=0)
        if set(map(tuple, rounded[:-1])) != corners or not np.all(
            np.count_nonzero(edges, axis=1) == 1
        ):
            raise ValueError(
                "HITL crop rectangles cannot be rotated or self-intersecting"
            )
        roi_id = str(
            feature.get("properties", {}).get("roi_id", f"roi_{len(regions) + 1:03d}")
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", roi_id) or roi_id in seen:
            raise ValueError("ROI IDs must be unique safe identifiers")
        seen.add(roi_id)
        core = expected.tile_size - 2 * expected.overlap_px
        cy, cx = (int(y0) // core) * core, (int(x0) // core) * core
        if (
            y0 < cy + expected.margin_px
            or x0 < cx + expected.margin_px
            or y1 > min(cy + core, expected.height) - expected.margin_px
            or x1 > min(cx + core, expected.width) - expected.margin_px
        ):
            raise ValueError("Move the crop completely inside a valid tile-core guide")
        if any(
            y0 < other.y0 + other.height_px
            and y1 > other.y0
            and x0 < other.x0 + other.width_px
            and x1 > other.x0
            for other in regions
        ):
            raise ValueError("Viewer-selected HITL crops cannot overlap")
        regions.append(
            ViewerRegion(
                roi_id,
                int(y0),
                int(x0),
                width if width == height else expected.crop_size_px,
                height=height,
                width=width,
            )
        )
    if not regions:
        raise ValueError("Select at least one HITL crop")
    return regions


def read_viewer_regions(path: Path, expected: RoiContext) -> list[ViewerRegion]:
    """Read viewer-selected crops and verify their source/grid and geometry.

    Args:
        path: GeoJSON saved by the dedicated HITL ROI helper.
        expected: Context of the source now being exported.

    Returns:
        Native-pixel squares in the user's export order.
    """
    return _parse_regions(json.loads(path.read_text()), expected)


def write_viewer_regions(
    path: Path, context: RoiContext, rectangles_yx: Sequence[np.ndarray]
) -> None:
    """Validate and save a napari Shapes layer without overwriting ROI files.

    Args:
        path: New GeoJSON path, outside the input store.
        context: Portable native image and tile-grid identity.
        rectangles_yx: Napari rectangle vertices in native pixel (y, x) order.
    """
    features = []
    for index, rectangle in enumerate(rectangles_yx, 1):
        points = np.asarray(rectangle, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Select two-dimensional rectangles with four corners")
        points = np.rint(points).astype(np.int64)
        xy = points[:, ::-1].tolist()
        features.append(
            {
                "type": "Feature",
                "properties": {"roi_id": f"roi_{index:03d}"},
                "geometry": {"type": "Polygon", "coordinates": [[*xy, xy[0]]]},
            }
        )
    payload = {
        "type": "FeatureCollection",
        "coordinate_system": "native_pixels",
        "merxen_hitl_regions_version": 2,
        "merxen_hitl_context": asdict(context),
        "features": features,
    }
    _parse_regions(payload, context)
    # Exclusive creation keeps earlier selections available for comparison.
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
