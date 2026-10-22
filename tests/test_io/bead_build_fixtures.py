"""Helpers for the tests that run bead removal on a synthetic MERSCOPE folder.

They write synthetic planes as ``mosaic_<stain>_z<N>.tif`` files, wrap the
stand-in calibration of ``bead_synth`` in a real, saveable ``BeadCalibration``
and open the store the driver writes into. No real data or paths are used.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from merxen.io.bead_removal.calibrate import BeadCalibration

from . import bead_synth as bs


def write_mosaic(
    folder: Path,
    stack: dict[str, Any],
    n_planes: int,
    stains: Iterable[str] = ("PolyT", "DAPI"),
) -> None:
    """Write the planes 0..n_planes-1 of the given stains as mosaic TIFFs.

    Args:
        folder: Output folder (the MERSCOPE ``images`` folder).
        stack: Synthetic stacks from ``bead_synth.make_stack``.
        n_planes: Number of planes to write per stain.
        stains: Stains to write.
    """
    import tifffile

    folder.mkdir(parents=True, exist_ok=True)
    for stain in stains:
        for z in range(n_planes):
            plane = np.clip(stack[stain][z], 1, 65535).astype(np.uint16)
            tifffile.imwrite(folder / f"mosaic_{stain}_z{z}.tif", plane)


def raw_max(stack: dict[str, Any], stain: str, z_layers: Iterable[int]) -> np.ndarray:
    """Return the plain max projection of the written (uint16) planes."""
    planes = [np.clip(stack[stain][z], 1, 65535).astype(np.uint16) for z in z_layers]
    return np.stack(planes).max(axis=0)


def real_calibration() -> BeadCalibration:
    """Return a saveable ``BeadCalibration`` of the synthetic stand-in PSF."""
    syn = bs.make_calibration()
    return BeadCalibration(
        version=syn.version,
        params_sha256=syn.params_sha256,
        corrected_stains=list(syn.corrected_stains),
        reference_stain=syn.reference_stain,
        library=syn.library,
        thresholds=syn.thresholds,
        noise=dict(syn.noise),
        dog_noise_z0=syn.dog_noise_z0,
        a0_sel=syn.a0_sel,
        a0_median=syn.a0_median,
        ell_default=syn.ell_default,
        rho=dict(syn.rho),
        decision={"status": "synthetic"},
    )


def open_projection_store(
    path: Path, *, n_channels: int, shape: tuple[int, int], chunk_px: int
) -> Any:
    """Create the uint16 zarr array [channel, y, x] the driver writes into."""
    import zarr

    chunks = (1, min(chunk_px, shape[0]), min(chunk_px, shape[1]))
    return zarr.open_array(
        str(path), mode="w", shape=(n_channels, *shape), chunks=chunks, dtype="uint16"
    )
