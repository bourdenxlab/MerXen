"""Radial PSF templates and the per-dataset PSF library.

A template is the response of one z plane to a unit-amplitude z0 point source,
as a radial profile on nodes ``r_n = n * ds``. The library holds one template per
plane z0..z7 (z4..z7 are derived from z3 by self-similar scaling because those
planes are too faint to stack).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .params import DEFAULT_PARAMS, FLUX_TARGET, PsfSubtractParams

logger = logging.getLogger(__name__)


class Template:
    """Radial profile on nodes ``r_n = n * ds``, zero beyond ``r_sup``.

    Args:
        ds: Node spacing in px.
        r_sup: Support radius in px.
        vals: Response per unit z0 amplitude at the nodes (zeros if omitted).

    Attributes:
        ds: Node spacing in px.
        n: Number of nodes.
        r_sup: Support radius in px.
        vals: Profile values at the nodes.
        nodes: Node radii in px.
    """

    def __init__(self, ds: float, r_sup: float, vals: np.ndarray | None = None) -> None:
        self.ds = ds
        self.n = int(np.ceil(r_sup / ds)) + 1
        self.r_sup = r_sup
        self.vals = np.zeros(self.n, np.float64) if vals is None else vals
        self.nodes = np.arange(self.n) * ds

    def __call__(self, r: np.ndarray, s: float = 1.0) -> np.ndarray:
        """Evaluate the profile at radii ``r``, dilated by the factor ``s``.

        The dilation conserves flux, so the profile is divided by ``s**2``.
        """
        profile: np.ndarray
        if s == 1.0:
            profile = np.interp(r, self.nodes, self.vals, right=0.0)
        else:
            profile = np.interp(r / s, self.nodes, self.vals, right=0.0) / (s * s)
        return profile

    def flux(self, radius: float) -> float:
        """Integrate the profile inside ``radius`` (px)."""
        inside = self.nodes <= radius
        return float(
            np.sum(self.vals[inside] * 2 * np.pi * self.nodes[inside]) * self.ds
        )


def kern2d(tmpl: Template, s: float = 1.0) -> tuple[np.ndarray, int]:
    """Render a template, dilated by ``s``, as a 2-D kernel.

    Args:
        tmpl: Radial template.
        s: Dilation factor.

    Returns:
        The float32 kernel of shape (2 * half + 1, 2 * half + 1) and its half
        size ``half`` in px.
    """
    half = int(np.ceil(tmpl.r_sup * s))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    kernel = tmpl(np.hypot(yy, xx).astype(np.float32), s).astype(np.float32)
    return kernel, half


def ring_prominence(tmpl: Template, lo: float, hi: float) -> tuple[float, float]:
    """Measure the ring prominence of a radial template.

    The prominence is (peak in [lo, hi] - minimum just inside it) / template
    maximum.

    Args:
        tmpl: Radial template.
        lo: Inner radius of the search range in px.
        hi: Outer radius of the search range in px.

    Returns:
        (prominence, radius of the peak in px).
    """
    r, v = tmpl.nodes, tmpl.vals
    in_range = (r >= lo) & (r <= hi)
    peak = int(np.argmax(np.where(in_range, v, -1e18)))
    inside = (r < r[peak]) & (r >= max(lo - 12.0, 4.0))
    minimum = v[inside].min() if inside.any() else v[peak]
    prominence = (v[peak] - minimum) / max(np.abs(v).max(), 1e-12)
    return float(prominence), float(r[peak])


def _defocus_radius(plane: int) -> float:
    """Defocus ring radius (px) of a plane: the 7 + 22.8 z law."""
    return 7.0 + 22.8 * plane


def derive_high_templates(
    tmpls: list[Template],
    params: PsfSubtractParams = DEFAULT_PARAMS,
    k_ref: int = 3,
    flux_targets: bool = False,
) -> list[Template]:
    """Derive the templates of the planes above ``k_ref`` by self-similar scaling.

    The defocus radius follows ``7 + 22.8 z`` px, so the reference template is
    dilated to the radius of each higher plane.

    Args:
        tmpls: Templates for planes 0..k_ref (more are ignored).
        params: Fixed parameters (support radii, node spacing).
        k_ref: Last stacked plane.
        flux_targets: Set the plane flux to ``FLUX_TARGET`` (absolute) instead of
            relative to the ``k_ref`` plane flux.

    Returns:
        Templates for planes 0..len(params.r_sup) - 1.
    """
    out = list(tmpls[: k_ref + 1])
    flux0 = tmpls[0].flux(1e9)
    if flux_targets:
        for k in range(2, k_ref + 1):
            flux = out[k].flux(1e9)
            if flux > 0:
                out[k] = Template(
                    params.ds,
                    out[k].r_sup,
                    out[k].vals * (FLUX_TARGET[k] * flux0 / flux),
                )
    ref = out[k_ref]

    for k in range(k_ref + 1, len(params.r_sup)):
        lam = _defocus_radius(k) / _defocus_radius(k_ref)
        n = int(np.ceil(params.r_sup[k] / params.ds)) + 1
        nodes = np.arange(n) * params.ds
        vals = np.interp(nodes / lam, ref.nodes, ref.vals, right=0.0) / lam**2
        taper = np.clip((params.r_sup[k] - nodes) / (params.r_sup[k] * 0.12), 0, 1)
        vals = vals * (0.5 - 0.5 * np.cos(np.pi * taper))
        tmpl = Template(params.ds, params.r_sup[k], vals)
        flux = tmpl.flux(1e9)
        if flux > 0:
            if flux_targets:
                target = FLUX_TARGET[k] * flux0
            else:
                target = FLUX_TARGET[k] / FLUX_TARGET[k_ref] * ref.flux(1e9)
            tmpl.vals = tmpl.vals * (target / flux)
        out.append(tmpl)
    return out


@dataclass
class PsfLibrary:
    """Per-dataset PSF library: one radial template per plane z0..z7.

    Attributes:
        tmpls: Templates, index = plane.
        info: Free-form provenance (number of tiles, per-tile info).
    """

    tmpls: list[Template]
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write the templates to an ``.npz`` file."""
        arrays: dict[str, np.ndarray] = {
            f"vals{k}": t.vals for k, t in enumerate(self.tmpls)
        }
        arrays["ds"] = np.array(self.tmpls[0].ds)
        arrays["r_sup"] = np.array([t.r_sup for t in self.tmpls])
        np.savez(path, **arrays)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> PsfLibrary:
        """Read a library written by :meth:`save`."""
        data = np.load(path)
        n_planes = len(data["r_sup"])
        return cls(
            [
                Template(
                    float(data["ds"]),
                    float(data["r_sup"][k]),
                    data[f"vals{k}"].astype(np.float64),
                )
                for k in range(n_planes)
            ]
        )

    def summary(self) -> list[dict[str, float]]:
        """Summarise every plane's template.

        Returns:
            Per plane: ring radius (argmax of r x template), centre value, flux
            relative to z0, and the 7 + 22.8 z law radius.
        """
        flux0 = self.tmpls[0].flux(1e9)
        rows = []
        for k, tmpl in enumerate(self.tmpls):
            i = int(np.argmax(tmpl.nodes * tmpl.vals))
            rows.append(
                {
                    "plane": k,
                    "ring_radius_px": float(tmpl.nodes[i]),
                    "centre": float(tmpl.vals[0]),
                    "flux_rel": (
                        float(tmpl.flux(1e9) / flux0) if flux0 else float("nan")
                    ),
                    "law_radius_px": _defocus_radius(k),
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """Describe the library as JSON-able data (the arrays go to the .npz)."""
        return {
            "ds": float(self.tmpls[0].ds),
            "r_sup": [float(t.r_sup) for t in self.tmpls],
            "summary": self.summary(),
            "info": to_jsonable(self.info),
        }


def to_jsonable(obj: object) -> Any:
    """Convert nested containers and numpy scalars/arrays to JSON-able values.

    Args:
        obj: Any nesting of dicts, lists, tuples, numpy arrays and numpy scalars.

    Returns:
        The same structure built from plain dicts, lists, ints and floats.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    return obj
