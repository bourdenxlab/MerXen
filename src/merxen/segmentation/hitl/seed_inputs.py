"""Copy processed inputs into an independent HITL inference run."""

from __future__ import annotations

import errno
import hashlib
import json
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from merxen.segmentation.hitl.import_annotations import artifact_sha256


class HitlSeedConfig(BaseModel):
    """Explicit source/destination paths for a fresh custom-model run."""

    model_config = ConfigDict(extra="forbid")

    source_zarr: Path
    destination_zarr: Path
    nuclei_source_dir: Path | None = None
    nuclei_destination_dir: Path | None = None
    protected_output_roots: list[Path] = Field(default_factory=list)


@dataclass(frozen=True)
class SeededHitlInputs:
    """Completed independent source store and its source identity."""

    source_zarr: Path
    fingerprint: str
    reused: bool


def _tree_identity(path: Path) -> str:
    files = [
        (str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def _copy_independent(source: str, destination: str) -> str:
    # Reflinks share storage extents but have independent inodes and copy on
    # write. Hard links would let an inference run change its published source.
    cloned = False
    if sys.platform == "linux":
        import fcntl

        try:
            with open(source, "rb") as original, open(destination, "wb") as copied:
                fcntl.ioctl(copied.fileno(), 0x40049409, original.fileno())  # FICLONE
            cloned = True
        except OSError as error:
            if error.errno not in (
                errno.EXDEV,
                errno.EOPNOTSUPP,
                errno.EINVAL,
                errno.ENOTTY,
            ):
                raise
    if not cloned:
        shutil.copyfile(source, destination)
    shutil.copystat(source, destination)
    Path(destination).chmod(stat.S_IMODE(Path(source).stat().st_mode) | stat.S_IWUSR)
    return destination


def seed_hitl_inputs(config: HitlSeedConfig) -> SeededHitlInputs:
    """Create an independent source copy while preserving original output data.

    Args:
        config: Processed source, fresh run destination and protected roots.

    Returns:
        The independent source path and its recorded source fingerprint.

    Raises:
        ValueError: If a destination is protected, overlapping or already unrelated.
    """
    source, destination = (
        config.source_zarr.resolve(),
        config.destination_zarr.resolve(),
    )
    if not source.is_dir():
        raise ValueError("The processed source SpatialData store does not exist.")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("HITL source and destination stores must be separate.")
    destinations = [destination]
    if config.nuclei_destination_dir is not None:
        destinations.append(config.nuclei_destination_dir.resolve())
    if any(
        path.is_relative_to(root.resolve())
        for path in destinations
        for root in config.protected_output_roots
    ):
        raise ValueError("HITL input seeding targets a protected baseline root.")
    fingerprint = _tree_identity(source)
    marker = destination.parent / "hitl_seed.json"
    reused = destination.exists()
    if reused:
        if (
            not marker.is_file()
            or json.loads(marker.read_text()).get("source_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Destination source already exists; choose a fresh HITL outdir."
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hitl-source-", dir=destination.parent
        ) as temporary:
            copied = Path(temporary) / "source.zarr"
            shutil.copytree(
                source, copied, copy_function=_copy_independent, symlinks=False
            )
            for directory in [
                copied,
                *[path for path in copied.rglob("*") if path.is_dir()],
            ]:
                directory.chmod(
                    stat.S_IMODE(directory.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR
                )
            copied.rename(destination)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_fingerprint": fingerprint,
                    "source_zarr": str(source),
                },
                indent=2,
            )
            + "\n"
        )
    if config.nuclei_source_dir is not None:
        if config.nuclei_destination_dir is None:
            raise ValueError("Nuclei reuse requires its destination directory.")
        paths = sorted(config.nuclei_source_dir.glob("cellpose_nuclei*"))
        if not any(path.suffix == ".npy" for path in paths):
            raise ValueError(
                "No completed stock nuclei mask was found in the source run."
            )
        target = config.nuclei_destination_dir
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if not path.is_file():
                continue
            output = target / path.name
            if output.exists():
                if artifact_sha256(output) != artifact_sha256(path):
                    raise ValueError(
                        "Nuclei destination contains different existing outputs."
                    )
            else:
                _copy_independent(str(path), str(output))
    return SeededHitlInputs(destination, fingerprint, reused)
