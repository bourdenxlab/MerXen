"""Characterisation of the three Xenium pixel-size parsers.

Enrichment (``enrich._find_xenium_pixel_size``), segmentation
(``pipeline._load_xenium_transform_matrix``) and the sanity plots
(``sanity_plots._find_xenium_microns_per_pixel``) each read the Xenium pixel
size from sidecar files, with different candidates, keys and failure modes. The
table below records what each returns for the same layouts today, so that a
later merge into one platform-registry parser (AT-6) can be measured against
it. ``None`` means the parser found nothing (enrichment and segmentation raise
``FileNotFoundError``; the sanity plots return ``None``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from merxen.config import SegmentationConfig
from merxen.enrichment.enrich import _find_xenium_pixel_size, _TransformInputs
from merxen.segmentation.pipeline import _load_xenium_transform_matrix
from merxen.visualization.sanity_plots import _find_xenium_microns_per_pixel


def _write_json(path: Path, data: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _layout(tmp_path: Path, name: str) -> tuple[Path, Path | None]:
    """Create one sidecar layout; return the store and the transform path."""
    store = tmp_path / "run" / "store"
    store.mkdir(parents=True)
    transform: Path | None = None
    if name == "experiment_xenium":
        _write_json(store / "experiment.xenium", {"pixel_size": 0.2125})
    elif name == "specs_microns_per_pixel":
        _write_json(store / "specs.json", {"microns_per_pixel": 0.25})
    elif name == "nested_specs":
        _write_json(store / "specs" / "specs.json", {"pixel_size": 0.3})
    elif name == "experiment_xenium_in_parent":
        _write_json(store.parent / "experiment.xenium", {"pixel_size": 0.2125})
    elif name == "transform_csv":
        transform = tmp_path / "micron_to_pixel.csv"
        np.savetxt(transform, np.diag([4.0, 4.0, 1.0]))
    elif name != "nothing":
        raise AssertionError(name)
    return store, transform


def _enrichment(store: Path, transform: Path | None) -> float | None:
    try:
        return _find_xenium_pixel_size(
            _TransformInputs(
                platform="XENIUM", transform_path=transform, original_data_path=store
            )
        )
    except FileNotFoundError:
        return None


def _segmentation(
    store: Path, transform: Path | None, output_dir: Path
) -> np.ndarray | None:
    config = SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "fixture",
                "platform": "XENIUM",
                "data_path": str(store),
                "channels": ["DAPI", "18S"],
                "output_dir": str(output_dir),
                "transform_path": str(transform) if transform else None,
            }
        }
    )
    try:
        return _load_xenium_transform_matrix(config)
    except FileNotFoundError:
        return None


@pytest.mark.parametrize(
    ("layout", "enrichment", "segmentation", "sanity"),
    [
        ("experiment_xenium", 0.2125, 0.2125, 0.2125),
        # Segmentation reads "pixel_size" only.
        ("specs_microns_per_pixel", 0.25, None, 0.25),
        ("nested_specs", 0.3, 0.3, 0.3),
        # Only the sanity plots look beside and above the store.
        ("experiment_xenium_in_parent", None, None, 0.2125),
        # Enrichment inverts a micron->pixel matrix; segmentation returns it
        # whole; the sanity plots never see a transform path.
        ("transform_csv", 0.25, 0.25, None),
        ("nothing", None, None, None),
    ],
)
def test_xenium_pixel_size_parsers_today(
    tmp_path: Path,
    layout: str,
    enrichment: float | None,
    segmentation: float | None,
    sanity: float | None,
) -> None:
    store, transform = _layout(tmp_path, layout)

    assert _enrichment(store, transform) == pytest.approx(enrichment)

    matrix = _segmentation(store, transform, tmp_path / "segment_out")
    if segmentation is None:
        assert matrix is None
    else:
        assert matrix is not None
        expected = np.diag([1 / segmentation, 1 / segmentation, 1.0])
        np.testing.assert_allclose(matrix, expected)

    assert _find_xenium_microns_per_pixel(store) == pytest.approx(sanity)


def test_segmentation_prefers_the_xenium_spec_path(tmp_path: Path) -> None:
    """``xenium_spec_path`` comes before every other segmentation candidate."""
    store, _ = _layout(tmp_path, "experiment_xenium")
    spec = tmp_path / "spec" / "experiment.xenium"
    _write_json(spec, {"pixel_size": 0.5})
    config = SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "fixture",
                "platform": "XENIUM",
                "data_path": str(store),
                "channels": ["DAPI", "18S"],
                "output_dir": str(tmp_path / "segment_out"),
                "xenium_spec_path": str(spec),
            }
        }
    )

    np.testing.assert_allclose(
        _load_xenium_transform_matrix(config), np.diag([2.0, 2.0, 1.0])
    )
