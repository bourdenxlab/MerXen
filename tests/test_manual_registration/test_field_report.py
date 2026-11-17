"""The landmark-check figure on a numpy phantom (no images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merxen.manual_registration.field_fit import FieldFitReport, build_landmark_field
from merxen.manual_registration.field_report import write_field_figure
from merxen.manual_registration.image_evidence import RoleImages
from merxen.manual_registration.schema import CropRect, StoreAttrs
from merxen.manual_registration.simulate import FieldPhantom, make_field_phantom

from .test_field_fit import PIXEL_UM, _inputs, _suggestions


@pytest.fixture(scope="module")
def phantom() -> FieldPhantom:
    return make_field_phantom(seed=5, n_clicks=30, n_gross=1, n_medium=1)


def _report(phantom: FieldPhantom) -> tuple[FieldFitReport, StoreAttrs]:
    similarity, store, _labelling, settings = _inputs(phantom, disabled=(4,))
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops={},
        image_check=False,
        windows=False,
        run_id="X_p",
    )
    return report, store


def test_the_figure_is_written_as_png_and_pdf(
    phantom: FieldPhantom, tmp_path: Path
) -> None:
    report, store = _report(phantom)
    crop = CropRect(x0=0, y0=0, x1=11000, y1=8000)
    overview = RoleImages(
        images={"dapi": np.random.default_rng(0).random((40, 55)).astype(np.float32)},
        to_level0=np.array([[200.0, 0.0, 99.5], [0.0, 200.0, 99.5], [0.0, 0.0, 1.0]]),
        pixel_size_um=400.0,
        level0_pixel_size_um=PIXEL_UM,
    )
    path = write_field_figure(
        tmp_path / "deep" / "check.png",
        report=report,
        store=store,
        crops={"reference": crop, "moving": crop},
        suggestions=_suggestions(report),
        overview=overview,
    )
    assert path.exists() and path.stat().st_size > 10_000
    assert path.with_suffix(".pdf").exists()


def test_the_figure_works_without_an_overview(
    phantom: FieldPhantom, tmp_path: Path
) -> None:
    report, store = _report(phantom)
    crop = CropRect(x0=0, y0=0, x1=11000, y1=8000)
    path = write_field_figure(
        tmp_path / "check.png",
        report=report,
        store=store,
        crops={"reference": crop, "moving": crop},
        suggestions=[],
    )
    assert path.exists()


def test_unknown_figure_kinds_are_refused(
    phantom: FieldPhantom, tmp_path: Path
) -> None:
    report, store = _report(phantom)
    crop = CropRect(x0=0, y0=0, x1=11000, y1=8000)
    with pytest.raises(ValueError, match="unknown figure kind"):
        write_field_figure(
            tmp_path / "x.png",
            report=report,
            store=store,
            crops={"reference": crop, "moving": crop},
            suggestions=[],
            kind="register",  # type: ignore[arg-type]
        )
