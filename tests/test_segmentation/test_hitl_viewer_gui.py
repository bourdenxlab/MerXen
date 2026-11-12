"""Optional Qt/napari-layer smoke check without creating an OpenGL canvas."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merxen.segmentation.hitl.regions import RoiContext, read_viewer_regions
from merxen.segmentation.hitl.viewer import add_roi_picker


@pytest.mark.skipif(
    importlib.util.find_spec("napari") is None
    or importlib.util.find_spec("qtpy") is None,
    reason="optional viewer/Qt environment unavailable",
)
def test_dock_add_and_save_uses_real_napari_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("qtpy.QtWidgets")
    from napari.components.layerlist import LayerList
    from napari.layers import Image, Shapes

    app = widgets.QApplication.instance() or widgets.QApplication([])
    context = RoiContext(
        platform="ATERA",
        image_key="morphology_focus",
        dataset_alias="ATERA_PILOT",
        height=2048,
        width=2048,
        pixel_size_um=0.21249222,
        tile_size=1024,
        overlap_px=64,
        margin_px=32,
        crop_size_px=256,
        source_metadata_sha256="1" * 64,
        baseline_stats_sha256="2" * 64,
    )
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(asdict(context)))
    layers = LayerList(
        [
            Image(
                np.zeros((256, 256)),
                name="ATERA morphology_focus DAPI",
                scale=(context.pixel_size_um * 8,) * 2,
            )
        ]
    )

    def add_shapes(data: list, **kwargs: object) -> Shapes:
        layer = Shapes(data, **kwargs)
        layers.append(layer)
        return layer

    viewer = SimpleNamespace(
        layers=layers,
        add_shapes=add_shapes,
        camera=SimpleNamespace(
            center=(0, 300 * context.pixel_size_um, 400 * context.pixel_size_um)
        ),
        window=SimpleNamespace(add_dock_widget=lambda *a, **k: None),
    )
    picker = add_roi_picker(viewer, context_path)
    errors = []
    monkeypatch.setattr(widgets.QMessageBox, "warning", lambda *a: errors.append(a[-1]))
    buttons = {
        button.text(): button
        for button in picker.widget.findChildren(widgets.QPushButton)
    }
    buttons["Add crop at view centre"].click()
    app.processEvents()
    assert not errors
    assert len(picker.roi_layer.data) == 1
    # Dragging in a micron viewer usually leaves subpixel native coordinates.
    picker.roi_layer.data = [picker.roi_layer.data[0] + [0.75, 0.6]]
    path = tmp_path / "regions.geojson"
    monkeypatch.setattr(
        widgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "")
    )
    buttons["Save crop regions…"].click()
    app.processEvents()
    assert not errors
    regions = read_viewer_regions(path, context)
    assert [(box.y0, box.x0, box.size) for box in regions] == [(173, 273, 256)]
    picker.widget.close()
