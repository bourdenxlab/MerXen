"""CSV samplesheet parsing and validation for multi-pair pipeline runs."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SamplePair:
    """A paired MERSCOPE + Xenium dataset for processing.

    Attributes:
        pair_id: Unique identifier for this sample pair.
        merscope_zarr_path: Path to the MERSCOPE input zarr.
        merscope_image_prefix: Prefix for matching z-plane image keys.
        merscope_z_range: Tuple of (z_start, z_end) inclusive.
        merscope_transform_path: Path to the micron-to-mosaic transform CSV.
        merscope_channels: Channel names for MERSCOPE (e.g. ['DAPI', 'PolyT']).
        xenium_dir: Path to the Xenium output directory.
        xenium_channels: Channel names for Xenium (e.g. ['DAPI', '18S']).
        xenium_min_qv: Minimum quality value for Xenium transcript filtering.
        merscope_voxel_layers: ProSeg voxel layers for MERSCOPE.
        xenium_voxel_layers: ProSeg voxel layers for Xenium.
    """

    pair_id: str
    merscope_zarr_path: Path
    merscope_image_prefix: str
    merscope_z_range: tuple[int, int]
    merscope_transform_path: Path
    merscope_channels: list[str] = field(default_factory=lambda: ["DAPI", "PolyT"])
    xenium_dir: Path = Path(".")
    xenium_channels: list[str] = field(default_factory=lambda: ["DAPI", "18S"])
    xenium_min_qv: float = 20.0
    merscope_voxel_layers: int = 7
    xenium_voxel_layers: int = 2


def parse_samplesheet(csv_path: Path) -> list[SamplePair]:
    """Parse a CSV samplesheet into a list of SamplePair objects.

    Expected columns:
        pair_id, merscope_zarr_path, merscope_image_prefix, merscope_z_range,
        merscope_transform_path, merscope_channels, xenium_dir, xenium_channels,
        xenium_min_qv, merscope_voxel_layers, xenium_voxel_layers

    Args:
        csv_path: Path to the samplesheet CSV.

    Returns:
        List of validated SamplePair dataclass instances.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Samplesheet not found: {csv_path}")

    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"pair_id", "merscope_zarr_path", "xenium_dir"}
        if reader.fieldnames is None:
            raise ValueError(f"Empty samplesheet: {csv_path}")
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Samplesheet missing required columns: {missing}. "
                f"Found: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            z_range_str = row.get("merscope_z_range", "0-6")
            z_parts = z_range_str.split("-")
            z_range = (int(z_parts[0]), int(z_parts[1]))

            pair = SamplePair(
                pair_id=row["pair_id"],
                merscope_zarr_path=Path(row["merscope_zarr_path"]),
                merscope_image_prefix=row.get("merscope_image_prefix", ""),
                merscope_z_range=z_range,
                merscope_transform_path=Path(row.get("merscope_transform_path", "")),
                merscope_channels=_parse_list(
                    row.get("merscope_channels", "DAPI,PolyT")
                ),
                xenium_dir=Path(row["xenium_dir"]),
                xenium_channels=_parse_list(row.get("xenium_channels", "DAPI,18S")),
                xenium_min_qv=float(row.get("xenium_min_qv", "20.0")),
                merscope_voxel_layers=int(row.get("merscope_voxel_layers", "7")),
                xenium_voxel_layers=int(row.get("xenium_voxel_layers", "2")),
            )
            pairs.append(pair)
            logger.info("Parsed sample pair %d: %s", row_num - 1, pair.pair_id)

    return pairs


def validate_samplesheet(pairs: list[SamplePair]) -> None:
    """Validate that all paths in a samplesheet exist.

    Args:
        pairs: List of SamplePair instances to validate.

    Raises:
        FileNotFoundError: If any required path does not exist.
    """
    errors = []
    for pair in pairs:
        if not pair.merscope_zarr_path.exists():
            errors.append(
                f"[{pair.pair_id}] MERSCOPE zarr not found: {pair.merscope_zarr_path}"
            )
        if not pair.xenium_dir.exists():
            errors.append(f"[{pair.pair_id}] Xenium dir not found: {pair.xenium_dir}")
        if (
            pair.merscope_transform_path
            and str(pair.merscope_transform_path) != "."
            and not pair.merscope_transform_path.exists()
        ):
            errors.append(
                f"[{pair.pair_id}] MERSCOPE transform not found: "
                f"{pair.merscope_transform_path}"
            )
    if errors:
        raise FileNotFoundError("Samplesheet validation failed:\n" + "\n".join(errors))


def _parse_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [v.strip() for v in value.split(",") if v.strip()]
