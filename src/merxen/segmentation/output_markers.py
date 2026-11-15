"""Markers that tell a resumed segmentation step its outputs were left incomplete.

Each segmentation step writes into persistent folders that outlive a Nextflow task
(``<outdir>/<pair>/<platform>/segmentation`` and ``latest``). A task that is killed,
or that fails with an exception, leaves whatever it had written there, and a
resumed task would otherwise reuse it because the files exist. Each step therefore
writes a hidden marker (:func:`merxen.path_utils.mark_incomplete`) before it starts
changing its outputs and removes it after its last write; a step that finds its
marker discards the outputs and computes them again.

The marker names start with a dot, so globs such as the HITL seed's
``cellpose_nuclei*`` never copy them.
"""

from __future__ import annotations

from pathlib import Path

from merxen.path_utils import incomplete_marker_path

CELL_MASK_MARKER_NAME = "cellpose_cell_masks"
NUCLEI_MASK_MARKER_NAME = "cellpose_nuclei_masks"


def cell_mask_marker_path(mask_path: Path) -> Path:
    """Marker for the Cellpose cell step (masks, probabilities, ProSeg CSV)."""
    return incomplete_marker_path(Path(mask_path), name=CELL_MASK_MARKER_NAME)


def nuclei_mask_marker_path(mask_path: Path) -> Path:
    """Marker for the DAPI-only Cellpose nuclei step."""
    return incomplete_marker_path(Path(mask_path), name=NUCLEI_MASK_MARKER_NAME)


def store_marker_path(store_path: Path) -> Path:
    """Marker beside a SpatialData store that a step builds or changes in place."""
    return incomplete_marker_path(Path(store_path))


def leftover_temporaries(path: Path) -> list[Path]:
    """Hidden temporary files an interrupted writer of ``path`` can leave behind.

    Covers the mask filter's ``.<name>.tmp.<pid>.npy``, the atomic writers'
    ``.<name>.tmp-<pid>``, the ProSeg CSV's ``.<name>.partial``, ProSeg's
    ``.<name>.proseg-shifted`` staging store and the store upgrade's
    ``.<name>.merxen-upgrade-*`` / ``.<name>.merxen-backup-*``.
    """
    path = Path(path)
    if not path.parent.is_dir():
        return []
    prefixes = (
        f".{path.name}.tmp",
        f".{path.name}.partial",
        f".{path.name}.proseg-shifted",
        f".{path.name}.merxen-upgrade-",
        f".{path.name}.merxen-backup-",
    )
    return sorted(
        candidate
        for candidate in path.parent.iterdir()
        if candidate.name.startswith(prefixes)
    )
