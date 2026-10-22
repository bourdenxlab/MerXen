"""Memory-mapped access to the stitched MERSCOPE mosaic planes.

The planes are the ``mosaic_<stain>_z<N>.tif`` files of the MERSCOPE images
folder. A whole plane is never loaded: windows are read through
``tifffile.memmap``, and pixels outside the mosaic are zero (the "invalid" value
used everywhere in the method).

The caller should call :meth:`MosaicPlaneReader.close` after every chunk of work:
the pages of a memory map stay resident and inflate the process RSS by many GB on
a large mosaic otherwise.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Same stain and plane syntax as the builder's channel discovery
# (``_get_channel_names`` in ``merxen.io.builders.merscope``); a test keeps the
# two in step.
_FILE_RE = re.compile(r"^mosaic_(?P<stain>[\w|-]+[0-9]?)_z(?P<z>[0-9]+)\.tif$")


def list_mosaic_planes(images_dir: str | Path) -> dict[str, list[int]]:
    """List the mosaic planes found in a MERSCOPE images folder.

    Args:
        images_dir: Folder with the ``mosaic_<stain>_z<N>.tif`` files.

    Returns:
        Stain name -> sorted z planes found for that stain.
    """
    found: dict[str, list[int]] = {}
    for path in sorted(Path(images_dir).iterdir()):
        match = _FILE_RE.match(path.name)
        if match:
            found.setdefault(match["stain"], []).append(int(match["z"]))
    return {stain: sorted(planes) for stain, planes in found.items()}


class MosaicPlaneReader:
    """Lazy reader of mosaic planes; one memmap per (stain, plane), opened on use.

    Args:
        images_dir: Folder with the ``mosaic_<stain>_z<N>.tif`` files.
        stains: Stains that will be read.
        planes: Planes that will be read (all must exist for every stain).

    Raises:
        FileNotFoundError: A requested file does not exist.
        ValueError: A requested plane is not a 2-D image.
    """

    def __init__(
        self,
        images_dir: str | Path,
        stains: Sequence[str],
        planes: Sequence[int],
    ) -> None:
        self.images_dir = Path(images_dir)
        self.stains = list(stains)
        self.planes = sorted(int(z) for z in planes)
        self._maps: dict[tuple[str, int], np.ndarray] = {}
        for stain in self.stains:
            for z in self.planes:
                if not self.path(stain, z).exists():
                    raise FileNotFoundError(
                        f"missing mosaic plane {self.path(stain, z).name} "
                        f"in {self.images_dir}"
                    )
        first = self._map(self.stains[0], self.planes[0])
        self._shape = (int(first.shape[0]), int(first.shape[1]))

    def path(self, stain: str, z: int) -> Path:
        """Return the file of one plane."""
        return self.images_dir / f"mosaic_{stain}_z{int(z)}.tif"

    def _map(self, stain: str, z: int) -> np.ndarray:
        key = (stain, int(z))
        if key not in self._maps:
            # tifffile is a transitive dependency and only needed once images
            # are actually read.
            import tifffile

            mapped = tifffile.memmap(self.path(stain, z), mode="r")
            if mapped.ndim != 2:
                raise ValueError(
                    f"{self.path(stain, z).name}: expected a 2-D plane, "
                    f"got shape {mapped.shape}"
                )
            self._maps[key] = mapped
        return self._maps[key]

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the mosaic in px."""
        return self._shape

    def read(
        self, stain: str, z: int, y0: int, y1: int, x0: int, x1: int
    ) -> np.ndarray:
        """Read a window of one plane.

        Args:
            stain: Stain name.
            z: Plane index.
            y0: First row (may be negative).
            y1: End row (exclusive; may exceed the mosaic).
            x0: First column (may be negative).
            x1: End column (exclusive; may exceed the mosaic).

        Returns:
            float32 window of shape (y1 - y0, x1 - x0), zero outside the mosaic.
        """
        mapped = self._map(stain, z)
        height, width = mapped.shape
        out = np.zeros((y1 - y0, x1 - x0), np.float32)
        ya, yb = max(0, y0), min(height, y1)
        xa, xb = max(0, x0), min(width, x1)
        if yb > ya and xb > xa:
            out[ya - y0 : yb - y0, xa - x0 : xb - x0] = mapped[ya:yb, xa:xb]
        return out

    def read_stack(
        self,
        stain: str,
        planes: Sequence[int],
        y0: int,
        y1: int,
        x0: int,
        x1: int,
    ) -> dict[int, np.ndarray]:
        """Read the same window of several planes as ``{z: float32 plane}``."""
        return {int(z): self.read(stain, z, y0, y1, x0, x1) for z in planes}

    def read_rows(self, stain: str, z: int, step: int) -> np.ndarray:
        """Read every ``step``-th row of the full plane.

        Used for the overview, so only a fraction of the plane is touched.

        Returns:
            float32 array of shape (ceil(height / step), width).
        """
        mapped = self._map(stain, z)
        return np.asarray(mapped[::step, :], np.float32)

    def overview(self, stain: str, z: int, factor: int = 16) -> np.ndarray:
        """Build a 1/factor overview.

        Takes every ``factor``-th row, then the mean of ``factor`` neighbouring
        columns.
        """
        rows = self.read_rows(stain, z, factor)
        width = (rows.shape[1] // factor) * factor
        blocks = rows[:, :width].reshape(rows.shape[0], width // factor, factor)
        overview: np.ndarray = blocks.mean(axis=2)
        return overview

    def close(self) -> None:
        """Drop the memmaps (releases the file handles and resident pages)."""
        self._maps.clear()

    def __enter__(self) -> MosaicPlaneReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
