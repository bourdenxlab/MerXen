#!/usr/bin/env python
"""Turn the KI hippocampus Xenium bundles into MerXen-ready Xenium folders.

Each Zenodo bundle (see ``fetch_ki_hippocampus.py``) is the complete XOA 1.4
output of one section. ``spatialdata_io.xenium`` reads that layout as it is, so
no file is converted; this script only selects, verifies and links.

Subcommand:

``prepare`` (per section)
    1. Extract only the members MerXen needs from the tarball in ``raw/`` (or,
       with ``--stream``, from the Zenodo URL in ``raw/metadata/manifest.tsv``,
       with no tarball on disk) into ``extracted/<section>/``. The bundle's md5
       is computed on the fly and must equal the Zenodo value; the members are
       moved into place only then. ``--with-mip``, ``--with-zstack`` and
       ``--with-analysis`` keep optional members the pipeline does not read.
    2. Check the extracted folder against the reader contract (experiment
       metadata, file set, parquet columns, ``cells.zarr.zip`` arrays, the
       cell-id agreement of the h5, ``cells.parquet`` and boundary files that
       the reader asserts, the morphology image's channels and pixel size).
       It does not compare ``seg_mask_value`` with the full-resolution masks
       (that needs the whole mask in memory); the build checks it.
    3. Hard-link (symlink across file systems) the files into
       ``xenium/<pair_id>/``, the ``xenium_dir`` of the samplesheet, and write
       ``PUBLIC_DATA_PROVENANCE.json`` there.

    The raw tarball is only read, unless ``--delete-tarball`` asks for it to be
    removed after a verified extraction.

The ``morphology_focus.ome.tif`` of XOA below 2.0 holds one channel (DAPI) that
spatialdata-io names ``0``; MerXen's Xenium build names such a single unnamed
channel ``DAPI`` itself, so the staged folder runs as it is.

``prepare`` has ``--dry-run``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, BinaryIO, Final

from merxen import local_config

logger = logging.getLogger(__name__)


def _load_fetch_module() -> ModuleType:
    """Import the sibling fetch script (shared paths, manifest, labels)."""
    key = "_ki_hippocampus_fetch_ki_hippocampus"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).resolve().with_name("fetch_ki_hippocampus.py")
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch_module()

PAIR_ID_PREFIX: Final = "PUBXHPC_"
PROVENANCE_NAME: Final = "PUBLIC_DATA_PROVENANCE.json"
CHUNK_BYTES: Final = 8 * 1024 * 1024

#: Files ``spatialdata_io.xenium`` reads for a pre-2.0 bundle with MerXen's
#: fixed build options (labels, boundaries, transcripts, morphology_focus).
REQUIRED_MEMBERS: Final[tuple[str, ...]] = (
    "experiment.xenium",
    "cell_feature_matrix.h5",
    "cells.parquet",
    "cells.zarr.zip",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "transcripts.parquet",
    "morphology_focus.ome.tif",
)
#: Small files no stage reads but worth keeping (panel, run metrics).
KEPT_MEMBERS: Final[tuple[str, ...]] = ("gene_panel.json", "metrics_summary.csv")
#: Optional groups: exact names, or folder prefixes ending in ``/``.
EXTRA_MEMBERS: Final[dict[str, tuple[str, ...]]] = {
    "mip": ("morphology_mip.ome.tif",),
    "zstack": ("morphology.ome.tif",),
    "analysis": ("analysis_summary.html", "analysis.zarr.zip", "analysis/"),
}

PARQUET_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "cells.parquet": ("cell_id", "x_centroid", "y_centroid"),
    "cell_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),
    "nucleus_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),
    "transcripts.parquet": (
        "x_location",
        "y_location",
        "z_location",
        "feature_name",
        "cell_id",
    ),
}
_VERSION = re.compile(r"^[xX]enium-(\d+)\.(\d+)\.(\d+)")
#: Matches merxen.io.builders.xenium.MAX_XENIUM_PANEL_GENES.
MAX_XENIUM_PANEL_GENES: Final = 10_000


class PrepareError(RuntimeError):
    """A bundle could not be extracted, verified or staged."""


@dataclass
class ValidationReport:
    """What the contract check found in one extracted section.

    Attributes:
        region_name: ``region_name`` from ``experiment.xenium``.
        analysis_sw_version: XOA version string.
        parsed_version: ``(major, minor, patch)`` as spatialdata-io parses it.
        pixel_size_um: Micrometres per morphology pixel.
        panel_targets: Predesigned plus custom panel targets.
        morphology_channels: OME channel names of ``morphology_focus.ome.tif``.
        morphology_size_c: OME ``SizeC`` (1 expected).
        morphology_physical_size_um: OME ``PhysicalSizeX``, when given.
        has_qv: Whether ``transcripts.parquet`` has a ``qv`` column.
        n_cells: Cells in ``cells.parquet`` (equal to the h5 barcodes).
        reader_channel_name: Channel name spatialdata-io gives the image
            (MerXen's build then names a single unnamed channel ``DAPI``).
        warnings: Non-fatal findings.
    """

    region_name: str = ""
    analysis_sw_version: str = ""
    parsed_version: tuple[int, int, int] | None = None
    pixel_size_um: float = 0.0
    panel_targets: int = 0
    morphology_channels: list[str] = field(default_factory=list)
    morphology_size_c: int | None = None
    morphology_physical_size_um: float | None = None
    has_qv: bool = False
    n_cells: int = 0
    reader_channel_name: str = ""
    warnings: list[str] = field(default_factory=list)


def pair_id_for(section: str) -> str:
    """Return the samplesheet ``pair_id`` of a section label."""
    return f"{PAIR_ID_PREFIX}{section}"


def xenium_dir(public_root: Path, section: str) -> Path:
    """Return the MerXen-ready Xenium folder of a section (``xenium_dir``)."""
    return Path(fetch.dataset_root(public_root)) / "xenium" / pair_id_for(section)


def spatialdata_path(public_root: Path, section: str) -> Path:
    """Return the cached source zarr of a section.

    ``make_samplesheet_ki_hippocampus.py --cache-source-zarr`` writes it into
    ``xenium_spatialdata_path``; by default that column is blank.
    """
    pair_id = pair_id_for(section)
    root = Path(fetch.dataset_root(public_root))
    return root / "spatialdata" / f"{pair_id}_xenium.zarr"


def wanted_member(relative: str, extras: Sequence[str]) -> bool:
    """Whether a bundle member (path below the top folder) is extracted.

    Args:
        relative: Member path relative to the bundle's top-level folder.
        extras: Names of the optional groups to keep.

    Returns:
        True for required, kept and selected optional members.
    """
    if relative in REQUIRED_MEMBERS or relative in KEPT_MEMBERS:
        return True
    for group in extras:
        for pattern in EXTRA_MEMBERS[group]:
            if pattern.endswith("/") and relative.startswith(pattern):
                return True
            if relative == pattern:
                return True
    return False


def _relative_member_path(name: str) -> tuple[str, str] | None:
    """Split ``top/rest`` into ``(top, rest)``; refuse unsafe names.

    Returns:
        ``None`` for the top-level folder itself.

    Raises:
        PrepareError: For absolute names or ``..`` components.
    """
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise PrepareError(f"unsafe member name in bundle: {name!r}")
    parts = [part for part in path.parts if part not in ("", ".")]
    if len(parts) < 2:
        return None
    return parts[0], "/".join(parts[1:])


class HashingReader:
    """File-like wrapper that md5-hashes and counts every byte read.

    Args:
        source: Binary stream to read.
    """

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.md5(usedforsecurity=False)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        """Read and hash up to ``size`` bytes."""
        data = self._source.read(size)
        self._digest.update(data)
        self.bytes_read += len(data)
        return data

    def drain(self) -> None:
        """Read (and hash) whatever the tar reader left unread."""
        for chunk in iter(lambda: self.read(CHUNK_BYTES), b""):
            del chunk

    def hexdigest(self) -> str:
        """Return the md5 of everything read so far."""
        return self._digest.hexdigest()


def extract_members(
    stream: BinaryIO,
    destination: Path,
    extras: Sequence[str] = (),
) -> dict[str, int]:
    """Stream a ``.tar.gz`` bundle and write the wanted members.

    The whole archive is read: member order differs between bundles, so no
    member can be assumed to come last. Only regular files are written;
    links and device entries are refused.

    Args:
        stream: The compressed bundle, read sequentially.
        destination: Folder receiving the members (top folder stripped).
        extras: Optional member groups to keep (keys of ``EXTRA_MEMBERS``).

    Returns:
        Member path -> size for every file written.

    Raises:
        PrepareError: On unsafe members or a bundle with several top folders.
    """
    written: dict[str, int] = {}
    top_folder: str | None = None
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:  # type: ignore[call-overload]
        for member in archive:
            split = _relative_member_path(member.name)
            if split is None:
                continue
            top, relative = split
            if top_folder is None:
                top_folder = top
            elif top != top_folder:
                raise PrepareError(
                    f"bundle has several top folders: {top_folder}, {top}"
                )
            if not wanted_member(relative, extras):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise PrepareError(f"refusing non-regular member {member.name!r}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise PrepareError(f"cannot read member {member.name!r}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, CHUNK_BYTES)
            written[relative] = member.size
            logger.info("  extracted %s (%s B)", relative, f"{member.size:,}")
    return written


def _open_source(
    item: Any, public_root: Path, *, stream_from_url: bool
) -> tuple[BinaryIO, str]:
    if stream_from_url:
        request = urllib.request.Request(
            item.url, headers={"User-Agent": fetch.USER_AGENT}
        )
        response = urllib.request.urlopen(request, timeout=300)  # noqa: S310
        return response, item.url
    tarball = fetch.raw_dir(public_root) / item.file_name
    if not tarball.is_file():
        raise PrepareError(
            f"{item.section}: tarball not downloaded (run fetch_ki_hippocampus.py "
            f"--download --sections {item.section}, or pass --stream)"
        )
    if tarball.stat().st_size != item.size_bytes:
        raise PrepareError(
            f"{tarball.name}: size differs from Zenodo; re-run the fetch"
        )
    return tarball.open("rb"), tarball.name


def extract_section(
    item: Any,
    public_root: Path,
    *,
    extras: Sequence[str] = (),
    stream_from_url: bool = False,
) -> Path:
    """Extract one section's wanted members, verifying the bundle md5.

    Members land in ``extracted/<section>.partial`` and the folder is renamed to
    ``extracted/<section>`` only after the md5 and the required set check out.

    Args:
        item: Manifest entry (``fetch.RemoteFile``).
        public_root: Public-data root.
        extras: Optional member groups.
        stream_from_url: Read from the Zenodo URL instead of the local tarball.

    Returns:
        The extracted folder.

    Raises:
        PrepareError: On a checksum or size mismatch or a missing member.
    """
    final = Path(fetch.extracted_dir(public_root, item.section))
    partial = final.with_name(final.name + ".partial")
    source, label = _open_source(item, public_root, stream_from_url=stream_from_url)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    logger.info("%s: extracting from %s", item.section, label)
    reader = HashingReader(source)
    try:
        with source:
            written = extract_members(reader, partial, extras)  # type: ignore[arg-type]
            reader.drain()
    except (tarfile.TarError, OSError, EOFError) as error:
        shutil.rmtree(partial, ignore_errors=True)
        raise PrepareError(f"{item.section}: extraction failed: {error}") from error
    md5 = reader.hexdigest()
    problems = []
    if reader.bytes_read != item.size_bytes:
        problems.append(f"read {reader.bytes_read} bytes, expected {item.size_bytes}")
    if md5 != item.md5:
        problems.append(f"md5 {md5}, expected {item.md5}")
    missing = [name for name in REQUIRED_MEMBERS if name not in written]
    if missing:
        problems.append(f"bundle lacks {missing}")
    if problems:
        shutil.rmtree(partial, ignore_errors=True)
        raise PrepareError(f"{item.section}: " + "; ".join(problems))
    record = {
        "section": item.section,
        "record_id": item.record_id,
        "source_file": item.file_name,
        "source_md5": md5,
        "md5_verified": True,
        "source_bytes": reader.bytes_read,
        "extras": list(extras),
        "members": written,
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (partial / fetch.EXTRACTION_RECORD).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    logger.info("%s: md5 verified, %d members kept", item.section, len(written))
    return final


def extraction_is_complete(public_root: Path, item: Any, extras: Sequence[str]) -> bool:
    """Whether a verified extraction of ``item`` with ``extras`` is on disk.

    Args:
        public_root: Public-data root.
        item: Manifest entry (``fetch.RemoteFile``).
        extras: Optional member groups that must be present.

    Returns:
        True when the record matches and every member has its recorded size.
    """
    if not fetch.is_extracted(public_root, item):
        return False
    folder = fetch.extracted_dir(public_root, item.section)
    record = json.loads((folder / fetch.EXTRACTION_RECORD).read_text(encoding="utf-8"))
    if not set(extras) <= set(record.get("extras", [])):
        return False
    members = record.get("members", {})
    return all(
        (folder / name).is_file() and (folder / name).stat().st_size == size
        for name, size in members.items()
    )


def parse_xenium_version(text: str) -> tuple[int, int, int] | None:
    """Parse an XOA version as spatialdata-io does (``None`` when unparsed)."""
    match = _VERSION.match(text)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _ome_channels(path: Path) -> tuple[int | None, list[str], float | None]:
    """Read ``SizeC``, channel names and ``PhysicalSizeX`` from the OME-XML.

    Returns ``(None, [], None)`` when the file has no parseable OME-XML.
    """
    import xml.etree.ElementTree as ElementTree

    import tifffile

    description = tifffile.tiffcomment(path)
    if not description:
        return None, [], None
    if isinstance(description, bytes):
        description = description.decode("utf-8", errors="replace")
    try:
        root = ElementTree.fromstring(description)
    except ElementTree.ParseError:
        return None, [], None
    pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
    if pixels is None:
        return None, [], None
    size_c = int(pixels.get("SizeC", "0")) or None
    physical = pixels.get("PhysicalSizeX")
    names = [
        node.get("Name", "")
        for node in pixels.iter()
        if node.tag.endswith("}Channel") or node.tag == "Channel"
    ]
    return size_c, names, float(physical) if physical else None


def _zip_top_arrays(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    found: set[str] = set()
    for name in names:
        parts = name.split("/")
        for depth in (1, 2, 3):
            if len(parts) > depth:
                found.add("/".join(parts[:depth]))
    return found


def _string_column(path: Path, column: str) -> list[str]:
    """Read one parquet column as ``str`` values (bytes decoded as UTF-8)."""
    import pyarrow.parquet as pq

    values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _h5_barcodes(path: Path) -> list[str]:
    """Return the cell barcodes of a 10x (v3 layout) ``cell_feature_matrix.h5``.

    Raises:
        PrepareError: If the file is not a v3 10x matrix.
    """
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if "matrix" not in handle or "barcodes" not in handle["matrix"]:
                raise PrepareError(
                    f"{path.name}: no matrix/barcodes; spatialdata-io reads only "
                    "the v3 10x layout"
                )
            barcodes = handle["matrix"]["barcodes"][:]
    except OSError as error:
        raise PrepareError(f"{path.name}: not a readable HDF5 file: {error}") from error
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in barcodes
    ]


def check_cell_tables(folder: Path, section: str) -> int:
    """Check the cell-id agreements the pre-2.0 reader asserts.

    ``spatialdata_io.xenium`` asserts that ``cells.parquet`` lists the
    ``cell_feature_matrix.h5`` barcodes in the same order and, below XOA 2.0,
    that each boundaries file, grouped by ``cell_id`` (sorted), gives exactly
    the table's cell ids in the table's order. Checking it here costs a few
    small column reads and fails before staging rather than in the build.

    Args:
        folder: Extracted section folder.
        section: Section label, for messages.

    Returns:
        The number of cells.

    Raises:
        PrepareError: When the build would fail one of those assertions.
    """
    cell_ids = _string_column(folder / "cells.parquet", "cell_id")
    barcodes = _h5_barcodes(folder / "cell_feature_matrix.h5")
    if cell_ids != barcodes:
        raise PrepareError(
            f"{section}: cells.parquet cell_id differs from the "
            "cell_feature_matrix.h5 barcodes (the reader asserts they are equal)"
        )
    for name in ("cell_boundaries.parquet", "nucleus_boundaries.parquet"):
        grouped = sorted(set(_string_column(folder / name, "cell_id")))
        if grouped != cell_ids:
            raise PrepareError(
                f"{section}: the cell ids of {name}, sorted, differ from "
                "cells.parquet (the pre-2.0 reader asserts they are equal)"
            )
    return len(cell_ids)


def validate_section(folder: Path, section: str) -> ValidationReport:
    """Check an extracted section against the MerXen Xenium intake contract.

    Args:
        folder: Folder holding the bundle members.
        section: Expected section label (``KI_HC_S<n>``).

    Returns:
        The findings.

    Raises:
        PrepareError: When the build would fail or the bundle is the wrong one.
    """
    import pyarrow.parquet as pq

    report = ValidationReport()
    missing = [name for name in REQUIRED_MEMBERS if not (folder / name).is_file()]
    if missing:
        raise PrepareError(f"{section}: missing {missing}")
    experiment = json.loads((folder / "experiment.xenium").read_text(encoding="utf-8"))
    for key in ("pixel_size", "analysis_sw_version"):
        if key not in experiment:
            raise PrepareError(f"{section}: experiment.xenium has no {key}")
    report.region_name = str(experiment.get("region_name", ""))
    report.analysis_sw_version = str(experiment["analysis_sw_version"])
    report.parsed_version = parse_xenium_version(report.analysis_sw_version)
    report.pixel_size_um = float(experiment["pixel_size"])
    report.panel_targets = int(
        experiment.get("panel_num_targets_predesigned", 0)
    ) + int(experiment.get("panel_num_targets_custom", 0))
    if report.analysis_sw_version.startswith("atera-") or (
        report.panel_targets > MAX_XENIUM_PANEL_GENES
    ):
        raise PrepareError(f"{section}: an Atera bundle, not a Xenium one")
    region_number = re.search(r"_(\d+)$", report.region_name)
    expected_number = int(section[len(fetch.SECTION_LABEL_PREFIX) :])
    if region_number is None or int(region_number.group(1)) != expected_number:
        raise PrepareError(
            f"{section}: experiment.xenium region_name does not end in "
            f"_{expected_number}; the bundle belongs to another section"
        )
    if report.parsed_version is None or report.parsed_version >= (2, 0, 0):
        raise PrepareError(
            f"{section}: XOA version {report.analysis_sw_version!r} is not the "
            "pre-2.0 layout this script stages"
        )

    for name, columns in PARQUET_COLUMNS.items():
        present = set(pq.read_schema(folder / name).names)
        absent = [column for column in columns if column not in present]
        if absent:
            raise PrepareError(f"{section}: {name} lacks columns {absent}")
        if name == "transcripts.parquet":
            report.has_qv = "qv" in present
    if not report.has_qv:
        report.warnings.append(
            "transcripts.parquet has no qv; xenium_min_qv is ignored"
        )

    arrays = _zip_top_arrays(folder / "cells.zarr.zip")
    needed = ["masks/0", "masks/1", "cell_id"]
    if report.parsed_version >= (1, 3, 0):
        needed.append("seg_mask_value")
    absent_arrays = [name for name in needed if name not in arrays]
    if absent_arrays:
        raise PrepareError(f"{section}: cells.zarr.zip lacks {absent_arrays}")
    report.n_cells = check_cell_tables(folder, section)
    expected_cells = experiment.get("num_cells")
    if isinstance(expected_cells, int) and expected_cells != report.n_cells:
        report.warnings.append(
            f"experiment.xenium num_cells {expected_cells} differs from "
            f"{report.n_cells} cells in cells.parquet"
        )

    size_c, names, physical = _ome_channels(folder / "morphology_focus.ome.tif")
    report.morphology_size_c = size_c
    report.morphology_channels = names
    report.morphology_physical_size_um = physical
    if size_c not in (None, 1):
        raise PrepareError(
            f"{section}: morphology_focus.ome.tif has {size_c} channels; one "
            "(DAPI) is expected for XOA 1.x"
        )
    if names and names[0].upper() != "DAPI":
        report.warnings.append(f"morphology channel is named {names[0]!r}, not DAPI")
    if physical is not None and abs(physical - report.pixel_size_um) > 1e-4:
        report.warnings.append(
            f"OME PhysicalSizeX {physical} differs from pixel_size "
            f"{report.pixel_size_um}"
        )
    # spatialdata-io gives a pre-2.0 morphology_focus no channel names.
    report.reader_channel_name = "0"
    return report


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def link_file(source: Path, target: Path, *, force: bool = False) -> str:
    """Hard-link ``source`` to ``target``; fall back to a relative symlink.

    Args:
        source: Existing file.
        target: Link to create.
        force: Replace a different existing ``target``.

    Returns:
        ``"exists"``, ``"hardlink"`` or ``"symlink"``.

    Raises:
        PrepareError: If ``target`` exists, differs and ``force`` is off.
    """
    if target.exists() or target.is_symlink():
        if _same_file(source, target):
            return "exists"
        if not force:
            raise PrepareError(f"{target} exists and is not {source.name}; use --force")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        # Another file system: a relative symlink keeps the tree relocatable.
        target.symlink_to(os.path.relpath(source, target.parent))
        return "symlink"


def stage_section(
    folder: Path,
    target: Path,
    section: str,
    report: ValidationReport,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Link the extracted files into the MerXen Xenium folder.

    Args:
        folder: Extracted section folder.
        target: ``xenium/<pair_id>`` folder.
        section: Section label.
        report: Validation findings, recorded in the provenance file.
        force: Replace differing existing links.

    Returns:
        File -> link mode.
    """
    record = json.loads((folder / fetch.EXTRACTION_RECORD).read_text(encoding="utf-8"))
    # Links an earlier run staged may be replaced: they point at an earlier
    # extraction of the same verified bundle (e.g. before --with-mip).
    previous: set[str] = set()
    try:
        earlier = json.loads((target / PROVENANCE_NAME).read_text(encoding="utf-8"))
        previous = set(earlier.get("links", {}))
    except (OSError, ValueError, AttributeError):
        previous = set()
    modes: dict[str, str] = {}
    for name in sorted(record.get("members", {})):
        modes[name] = link_file(
            folder / name, target / name, force=force or name in previous
        )
    record_id = record.get("record_id")
    provenance = {
        "dataset": "KI human hippocampus Xenium (Dumitru et al., Science 2025)",
        "doi": "10.1126/science.adu9575",
        "zenodo_record": record_id,
        "zenodo_doi": f"10.5281/zenodo.{record_id}" if record_id else None,
        "license": "CC BY 4.0",
        "section": section,
        "pair_id": pair_id_for(section),
        "source_file": record.get("source_file"),
        "source_md5": record.get("source_md5"),
        "md5_verified": record.get("md5_verified"),
        "converted": (
            "nothing; the vendor files (experiment.xenium included) are linked "
            "unchanged, none is synthesised"
        ),
        "links": modes,
        "validation": asdict(report),
        "channel_note": (
            "spatialdata-io names the single pre-2.0 morphology channel '0'; "
            "MerXen's Xenium build names it DAPI"
        ),
        "staged_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return modes


# --------------------------------------------------------------------------
# CLI


def _manifest_entries(public_root: Path, sections: Sequence[str] | None) -> list[Any]:
    manifest = fetch.metadata_dir(public_root) / fetch.MANIFEST_NAME
    if not manifest.is_file():
        raise PrepareError(
            f"{manifest} is missing: run fetch_ki_hippocampus.py --metadata-only first"
        )
    entries = fetch.read_manifest(manifest)
    return list(fetch.select_sections(entries, sections))


def kept_bytes_estimate(extras: Sequence[str]) -> int:
    """Planning estimate of the bytes one section's extraction keeps.

    Args:
        extras: Optional member groups requested.

    Returns:
        Bytes: the required and kept members plus the selected extras.
    """
    estimate = int(fetch.KEPT_BYTES_PER_SECTION_ESTIMATE)
    if "mip" in extras:
        estimate += int(fetch.MIP_BYTES_PER_SECTION_ESTIMATE)
    if "zstack" in extras:
        estimate += int(fetch.ZSTACK_BYTES_PER_SECTION_ESTIMATE)
    return estimate


def _dry_run_prepare(item: Any, public_root: Path, args: argparse.Namespace) -> None:
    folder = fetch.extracted_dir(public_root, item.section)
    tarball = fetch.raw_dir(public_root) / item.file_name
    target = xenium_dir(public_root, item.section)
    if extraction_is_complete(public_root, item, args.extras):
        record = json.loads(
            (folder / fetch.EXTRACTION_RECORD).read_text(encoding="utf-8")
        )
        kept = sum(int(size) for size in record.get("members", {}).values())
        logger.info(
            "%s: already extracted (%s kept)", item.section, fetch.format_bytes(kept)
        )
    else:
        estimate = kept_bytes_estimate(args.extras)
        source = "Zenodo stream" if args.stream else str(tarball)
        present = "present" if tarball.is_file() else "NOT downloaded"
        logger.info(
            "%s: would extract %s + %s%s from %s (tarball %s); about %s",
            item.section,
            ", ".join(REQUIRED_MEMBERS),
            ", ".join(KEPT_MEMBERS),
            f" + extras {args.extras}" if args.extras else "",
            source,
            present,
            fetch.format_bytes(estimate),
        )
        if args.delete_tarball:
            logger.info("%s: would then delete %s", item.section, tarball.name)
    logger.info(
        "%s: would hard-link the files into %s (0 extra bytes on one file system)",
        item.section,
        target,
    )


def _run_prepare(args: argparse.Namespace) -> int:
    entries = _manifest_entries(args.public_root, args.sections)
    if args.dry_run:
        for item in entries:
            _dry_run_prepare(item, args.public_root, args)
        return 0
    for item in entries:
        folder = fetch.extracted_dir(args.public_root, item.section)
        if extraction_is_complete(args.public_root, item, args.extras):
            logger.info("%s: extraction already complete", item.section)
        else:
            # Both routes write the kept members next to the raw folder (the
            # tarball route on top of the tarball already there).
            need = kept_bytes_estimate(args.extras)
            space = fetch.check_space(folder, need, int(args.margin_gb * 1e9))
            if not space.is_enough:
                raise PrepareError(
                    f"{item.section}: {fetch.format_bytes(space.free_bytes)} free "
                    f"at {space.probe_path}; need about {fetch.format_bytes(need)} "
                    f"plus a {args.margin_gb:g} GB margin"
                )
            folder = extract_section(
                item, args.public_root, extras=args.extras, stream_from_url=args.stream
            )
        report = validate_section(folder, item.section)
        for warning in report.warnings:
            logger.warning("%s: %s", item.section, warning)
        target = xenium_dir(args.public_root, item.section)
        modes = stage_section(folder, target, item.section, report, force=args.force)
        logger.info(
            "%s: staged %d files in %s (%s); XOA %s, pixel %.4f um, channel %s",
            item.section,
            len(modes),
            target,
            ", ".join(sorted(set(modes.values()))),
            report.analysis_sw_version,
            report.pixel_size_um,
            report.morphology_channels or "unnamed",
        )
        if args.delete_tarball:
            tarball = fetch.raw_dir(args.public_root) / item.file_name
            for path in (tarball, fetch.sidecar_path(tarball)):
                if path.is_file():
                    path.unlink()
                    logger.info("%s: deleted %s", item.section, path.name)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Public-data root holding xenium_v1_human_hippocampus_ff_ki/",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Extract, check and stage sections.")
    prepare.add_argument("--sections", nargs="+", default=None)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument(
        "--stream",
        action="store_true",
        help="Stream the bundle from Zenodo instead of reading raw/<tarball>.",
    )
    prepare.add_argument(
        "--delete-tarball",
        action="store_true",
        help="Delete the tarball after a verified extraction and staging.",
    )
    prepare.add_argument("--with-mip", action="store_true", help="Keep morphology_mip.")
    prepare.add_argument(
        "--with-zstack", action="store_true", help="Keep the 3-D morphology.ome.tif."
    )
    prepare.add_argument(
        "--with-analysis", action="store_true", help="Keep the XOA analysis outputs."
    )
    prepare.add_argument(
        "--force", action="store_true", help="Replace differing links."
    )
    prepare.add_argument("--margin-gb", type=float, default=10.0)

    args = parser.parse_args(argv)
    args.extras = [
        group
        for group, flag in (
            ("mip", args.with_mip),
            ("zstack", args.with_zstack),
            ("analysis", args.with_analysis),
        )
        if flag
    ]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prepare CLI.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    try:
        return _run_prepare(args)
    except (PrepareError, fetch.FetchError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
