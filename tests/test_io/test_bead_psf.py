"""Tests of the radial PSF templates and the PSF library."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from merxen.io.bead_removal.params import DEFAULT_PARAMS, FLUX_TARGET
from merxen.io.bead_removal.psf import (
    PsfLibrary,
    Template,
    derive_high_templates,
    kern2d,
    ring_prominence,
    to_jsonable,
)

from . import bead_synth as bs


def _ring_template(radius: float = 25.0, r_sup: float = 60.0) -> Template:
    nodes = np.arange(int(np.ceil(r_sup / 0.5)) + 1) * 0.5
    vals = 0.2 * np.exp(-(nodes**2) / 18.0) + np.exp(-((nodes - radius) ** 2) / 8.0)
    return Template(0.5, r_sup, vals)


def test_template_interpolates_and_is_zero_beyond_support() -> None:
    tmpl = Template(1.0, 4.0, np.array([4.0, 3.0, 2.0, 1.0, 0.0]))
    assert tmpl(np.array([0.0, 0.5, 2.0]))[1] == pytest.approx(3.5)
    assert tmpl(np.array([10.0]))[0] == 0.0
    assert Template(0.5, 10.0).vals.shape == (21,)


def test_dilation_conserves_flux() -> None:
    tmpl = _ring_template()
    base = tmpl.flux(1e9)
    for scale in (1.0, 1.4, 2.0):
        # Radial integral of the dilated profile on a fine grid.
        radii = np.arange(0, 400, 0.05)
        dilated = tmpl(radii, scale)
        flux = float(np.sum(dilated * 2 * np.pi * radii) * 0.05)
        assert flux == pytest.approx(base, rel=2e-3)


def test_kernel_is_radially_symmetric() -> None:
    tmpl = _ring_template(radius=12.0, r_sup=30.0)
    kernel, half = kern2d(tmpl)
    assert half == 30 and kernel.shape == (61, 61) and kernel.dtype == np.float32
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(kernel, kernel[::-1])
    assert kernel[half, half] == pytest.approx(float(tmpl.vals[0]))
    dilated, half2 = kern2d(tmpl, 1.5)
    assert half2 == 45 and dilated.shape == (91, 91)


def test_ring_prominence_finds_the_ring() -> None:
    prominence, radius = ring_prominence(_ring_template(25.0), 20.0, 40.0)
    assert radius == pytest.approx(25.0, abs=0.5)
    assert prominence > 0.5
    flat = Template(0.5, 60.0, np.exp(-((np.arange(121) * 0.5) ** 2) / 50.0))
    flat_prominence, _ = ring_prominence(flat, 20.0, 40.0)
    assert flat_prominence <= 0.0 + 1e-9


def test_derived_planes_follow_the_defocus_law_and_flux_targets() -> None:
    lib = bs.library_from_generator()
    base = lib.tmpls[:4]
    derived = derive_high_templates(base)
    assert len(derived) == len(DEFAULT_PARAMS.r_sup)
    # z0..z3 are passed through unchanged.
    for k in range(4):
        assert derived[k] is base[k]
    ref_flux = base[3].flux(1e9)
    for k in range(4, 8):
        expected = FLUX_TARGET[k] / FLUX_TARGET[3] * ref_flux
        assert derived[k].flux(1e9) == pytest.approx(expected, rel=1e-6)
        assert derived[k].r_sup == DEFAULT_PARAMS.r_sup[k]
    # Ring radius grows with the plane (7 + 22.8 z law).
    ring = [t.nodes[int(np.argmax(t.nodes * t.vals))] for t in derived[3:]]
    assert ring == sorted(ring)

    absolute = derive_high_templates(base, flux_targets=True)
    flux0 = base[0].flux(1e9)
    for k in range(4, 8):
        assert absolute[k].flux(1e9) == pytest.approx(FLUX_TARGET[k] * flux0, rel=1e-6)
    for k in (2, 3):
        assert absolute[k].flux(1e9) == pytest.approx(FLUX_TARGET[k] * flux0, rel=1e-6)


def test_library_roundtrip_and_summary(tmp_path: Path) -> None:
    lib = bs.library_from_generator()
    lib.info = {"tiles": np.array([1, 2]), "ring": np.float32(24.5)}
    path = tmp_path / "psf.npz"
    lib.save(path)
    back = PsfLibrary.load(path)
    assert len(back.tmpls) == 8
    for a, b in zip(back.tmpls, lib.tmpls, strict=True):
        assert a.ds == b.ds and a.r_sup == b.r_sup
        assert np.array_equal(a.vals, b.vals)
    rows = lib.summary()
    assert [r["plane"] for r in rows] == list(range(8))
    assert rows[0]["flux_rel"] == pytest.approx(1.0)
    for row in rows[1:4]:
        assert row["ring_radius_px"] == pytest.approx(row["law_radius_px"], abs=3.0)
    described = lib.to_dict()
    # Numpy scalars and arrays in the info must not break the JSON dump.
    assert json.loads(json.dumps(described))["info"] == {"tiles": [1, 2], "ring": 24.5}


def test_to_jsonable_converts_nested_numpy() -> None:
    value = {1: (np.int64(2), [np.float64(0.5)]), "a": np.arange(2)}
    out = to_jsonable(value)
    assert out == {"1": [2, [0.5]], "a": [0, 1]}
    assert type(out["1"][0]) is int
