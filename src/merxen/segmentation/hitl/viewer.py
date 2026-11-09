"""Optional HITL crop-location dock for an already-running napari viewer.

No GUI dependency is imported until the helper is installed. Cell outlines are
corrected later in Cellpose; this dock selects locations only and never writes
to the displayed SpatialData store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merxen.segmentation.hitl.regions import (
    RoiContext,
    read_viewer_regions,
    write_viewer_regions,
)


@dataclass(frozen=True)
class RoiPicker:
    """Dock and independent crop/guide layers installed in the user's viewer."""

    widget: Any
    roi_layer: Any
    guide_layer: Any


def validate_native_viewer_image(image: Any, context: RoiContext) -> None:
    """Refuse aligned, shifted or pixel-unit displays before placing micron ROIs.

    Args:
        image: The native napari Image layer chosen in the dock.
        context: Native level-zero geometry prepared from the pipeline store.

    Raises:
        ValueError: When the selected layer is not the standalone native frame.
    """
    name = str(image.name)
    if (
        context.image_key not in name
        or context.platform not in name.upper()
        or "aligned" in name.lower()
    ):
        raise ValueError(
            f"Choose the native {context.image_key} image, not an aligned image"
        )
    if image.ndim != 2:
        raise ValueError("Use a two-dimensional native image layer")
    data = image.data[0] if image.multiscale else image.data
    shape = data.shape[-3:-1] if image.rgb else data.shape[-2:]
    factors = np.asarray([context.height, context.width], float) / np.asarray(
        shape, float
    )
    expected_steps = factors * context.pixel_size_um
    origin = np.asarray(image.data_to_world((0, 0))[-2:], float)
    step_y = np.asarray(image.data_to_world((1, 0))[-2:], float) - origin
    step_x = np.asarray(image.data_to_world((0, 1))[-2:], float) - origin
    matrix = np.column_stack([step_y, step_x])
    # Pyramid levels can round their extent by one displayed pixel. Bound the
    # scale tolerance by that extent rather than allowing a fixed slide fraction.
    has_native_steps = np.all(
        abs(np.diag(matrix) - expected_steps)
        <= expected_steps / np.asarray(shape) + 1e-8
    ) and np.allclose(matrix - np.diag(np.diag(matrix)), 0, atol=1e-8)
    if not has_native_steps or np.any(abs(origin) > expected_steps):
        raise ValueError(
            "The image is not in native dataset microns. Open the standalone "
            "native store; aligned, flipped, shifted and pixel-unit views are refused."
        )


def add_roi_picker(
    viewer: Any,
    context_path: str | Path,
    *,
    regions_path: str | Path | None = None,
) -> RoiPicker:
    """Add a crop-location dock to the MerXen napari viewer.

    Args:
        viewer: Existing napari viewer with the standalone native image loaded.
        context_path: JSON from ``merxen cellpose-hitl-prepare-rois``.
        regions_path: Optional earlier HITL GeoJSON to reopen for editing.

    Returns:
        Dock and independent Shapes layers; the source store stays untouched.

    Raises:
        RuntimeError: If called outside an environment providing napari/Qt.
        ValueError: If no legal crop window exists in the pinned tile grid.
    """
    try:
        from napari.layers import Image
        from qtpy.QtWidgets import (
            QComboBox,
            QFileDialog,
            QLabel,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise RuntimeError(
            "Run add_roi_picker in the napari viewer's Python console"
        ) from error

    context = RoiContext.read(Path(context_path))
    windows = context.placement_windows()
    if not windows:
        raise ValueError("The pinned tile grid has no room for this crop size/margin")
    roi_name = "HITL crop regions"
    guide_name = "HITL valid crop areas"
    if roi_name in viewer.layers or guide_name in viewer.layers:
        raise ValueError(
            "A HITL picker is already open; preserve its regions before replacing it"
        )
    rectangles = []
    if regions_path is not None:
        for box in read_viewer_regions(Path(regions_path), context):
            rectangles.append(_rectangle(box.y0, box.x0, box.size))
    roi_layer = viewer.add_shapes(
        rectangles,
        name=roi_name,
        ndim=2,
        shape_type="rectangle",
        scale=(context.pixel_size_um, context.pixel_size_um),
        face_color="transparent",
        edge_color="yellow",
        edge_width=3,
    )
    guides = [
        np.array(
            [
                [y0, x0],
                [y0, x1 + context.crop_size_px],
                [y1 + context.crop_size_px, x1 + context.crop_size_px],
                [y1 + context.crop_size_px, x0],
            ],
            float,
        )
        for y0, y1, x0, x1 in windows
    ]
    guide_layer = viewer.add_shapes(
        guides,
        name=guide_name,
        ndim=2,
        shape_type="rectangle",
        scale=(context.pixel_size_um, context.pixel_size_um),
        face_color="transparent",
        edge_color="cyan",
        edge_width=2,
        opacity=0.4,
    )
    guide_layer.editable = False
    widget = QWidget()
    layout = QVBoxLayout(widget)
    description = QLabel(
        "Choose crop locations here; correct cell outlines in Cellpose later."
    )
    description.setWordWrap(True)
    layout.addWidget(description)
    layout.addWidget(QLabel("Native morphology image"))
    images = QComboBox()
    names = [str(layer.name) for layer in viewer.layers if isinstance(layer, Image)]
    images.addItems(names)
    matching = [
        name
        for name in names
        if context.image_key in name and "DAPI" in name and "aligned" not in name
    ]
    if matching:
        images.setCurrentText(matching[0])
    layout.addWidget(images)
    status = QLabel(
        f"Crop width: {context.crop_size_px} native pixels. "
        "Keep rectangles inside cyan guides."
    )
    status.setWordWrap(True)
    layout.addWidget(status)

    def require_frame() -> None:
        if not images.currentText() or images.currentText() not in viewer.layers:
            raise ValueError("Load and select the native morphology image first")
        validate_native_viewer_image(viewer.layers[images.currentText()], context)

    def add_crop() -> None:
        try:
            require_frame()
            centre = (
                np.asarray(viewer.camera.center[-2:], float) / context.pixel_size_um
            )
            wanted = centre - context.crop_size_px / 2
            choices = [
                np.array([np.clip(wanted[0], y0, y1), np.clip(wanted[1], x0, x1)])
                for y0, y1, x0, x1 in windows
            ]
            selected = min(
                choices, key=lambda point: float(np.square(point - wanted).sum())
            )
            y0, x0 = np.rint(selected).astype(int)
            roi_layer.add(
                _rectangle(int(y0), int(x0), context.crop_size_px),
                shape_type="rectangle",
            )
            roi_layer.mode = "select"
            viewer.layers.selection.active = roi_layer
            status.setText(
                "Crop added at the nearest valid position. "
                "Drag to adjust; keep it inside one cyan guide."
            )
        except ValueError as error:
            QMessageBox.warning(widget, "HITL crop regions", str(error))

    def save_regions() -> None:
        try:
            require_frame()
            if not len(roi_layer.data):
                raise ValueError("Select at least one HITL crop before saving")
            path, _ = QFileDialog.getSaveFileName(
                widget,
                "Save HITL crop regions",
                "hitl-regions.geojson",
                "GeoJSON (*.geojson)",
            )
            if not path:
                return
            write_viewer_regions(Path(path), context, roi_layer.data)
            status.setText(
                "Regions saved. Export pre-filled crops with this file, "
                "then open them in Cellpose."
            )
        except (ValueError, OSError) as error:
            QMessageBox.warning(widget, "HITL crop regions", str(error))

    add_button = QPushButton("Add crop at view centre")
    add_button.clicked.connect(add_crop)
    layout.addWidget(add_button)
    save_button = QPushButton("Save crop regions…")
    save_button.clicked.connect(save_regions)
    layout.addWidget(save_button)
    viewer.window.add_dock_widget(widget, name="HITL crop selection", area="right")
    viewer.layers.selection.active = roi_layer
    return RoiPicker(widget, roi_layer, guide_layer)


def _rectangle(y0: int, x0: int, size: int) -> np.ndarray:
    return np.asarray(
        [[y0, x0], [y0, x0 + size], [y0 + size, x0 + size], [y0 + size, x0]], float
    )
