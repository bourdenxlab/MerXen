#!/usr/bin/env python
"""Plan or run the download of the public KI human hippocampus Xenium dataset.

The dataset (Dumitru et al., Science 2025, doi 10.1126/science.adu9575) is six
sequential sections of one donor, published as two Zenodo records under CC BY
4.0 (10.5281/zenodo.14879552 and 10.5281/zenodo.14879626). Each Zenodo file is
one complete Xenium Onboard Analysis (XOA 1.4) output bundle, a ``.tar.gz`` of
about 12-15 GB, for one section.

The file list, sizes and md5 checksums are read from the Zenodo REST API at run
time; no file name is written into this script. Each file name has the form
``output-<instrument>__<slide>__<region>__<date>__<time>.tar.gz``, and the
section number is the trailing number of ``<region>``. Sections get the public
labels ``KI_HC_S1`` .. ``KI_HC_S6``.

Modes:

- default (dry run): list every file, its size and checksum, the per-section and
  total bytes, the destination and its free space. Nothing is written.
- ``--metadata-only``: save the Zenodo record JSONs and ``manifest.tsv`` (the
  input of ``prepare_ki_hippocampus.py``) under ``raw/metadata/``. Zenodo offers
  no smaller per-section file than the bundle itself.
- ``--download``: download the selected tarballs into ``raw/`` with HTTP Range
  resume, then verify each against its Zenodo md5. A verified file gets a
  ``<name>.md5`` sidecar, so a rerun skips it. The download is refused when the
  free space is below the bytes still needed plus ``--margin-gb``.

The data go to ``$MERXEN_XENIUM_PUBLIC_ROOT/xenium_v1_human_hippocampus_ff_ki/``
(``--public-root`` overrides). To keep the peak disk use near one tarball, run
one section at a time and let the prepare step delete each tarball::

    for s in S1 S2 S3 S4 S5 S6; do
        python scripts/public_data/xenium_hippocampus/fetch_ki_hippocampus.py \\
            --download --sections "$s"
        python scripts/public_data/xenium_hippocampus/prepare_ki_hippocampus.py \\
            prepare --sections "$s" --delete-tarball
    done

``prepare_ki_hippocampus.py prepare --stream`` instead streams each bundle
straight from Zenodo through the extraction, with no tarball on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import IO, Any, BinaryIO, Final

from merxen import local_config

logger = logging.getLogger(__name__)

DATASET_KEY: Final = "ki_hippocampus"
DATASET_DIR: Final = "xenium_v1_human_hippocampus_ff_ki"
ZENODO_RECORDS: Final[tuple[str, ...]] = ("14879552", "14879626")
ZENODO_API: Final = "https://zenodo.org/api/records/{record_id}"
SECTION_LABEL_PREFIX: Final = "KI_HC_S"
MANIFEST_NAME: Final = "manifest.tsv"
#: Written by prepare_ki_hippocampus.py once a bundle is extracted and its md5
#: checked; a section with this record needs no tarball any more.
EXTRACTION_RECORD: Final = "extraction.json"
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "section",
    "record_id",
    "file_name",
    "size_bytes",
    "md5",
    "url",
)
CHUNK_BYTES: Final = 8 * 1024 * 1024
GIB: Final = 1024**3
USER_AGENT: Final = "merxen-public-data-fetch/1.0"

#: Planning estimates from streamed inventories of these bundles (bytes per
#: section): the members MerXen needs plus gene_panel.json and
#: metrics_summary.csv; the optional extras. The prepare step reports the
#: real sizes once a bundle is extracted.
KEPT_BYTES_PER_SECTION_ESTIMATE: Final = 1_885_000_000
MIP_BYTES_PER_SECTION_ESTIMATE: Final = 1_200_000_000
ZSTACK_BYTES_PER_SECTION_ESTIMATE: Final = 7_930_000_000

_BUNDLE_NAME = re.compile(
    r"^output-(?P<instrument>[^_]+(?:_[^_]+)*)__(?P<slide>[^_]+(?:_[^_]+)*)"
    r"__(?P<region>.+?)__(?P<date>\d{8})__(?P<time>\d{6})\.tar\.gz$"
)
_REGION_SECTION = re.compile(r"_(?P<number>\d+)$")


class FetchError(RuntimeError):
    """A listing, space check, download or verification failed."""


@dataclass(frozen=True)
class RemoteFile:
    """One bundle offered by a Zenodo record.

    Attributes:
        section: Public section label, e.g. ``KI_HC_S1``.
        record_id: Zenodo record id.
        file_name: File name as Zenodo lists it.
        size_bytes: Size in bytes.
        md5: Lower-case hex md5 from the Zenodo API.
        url: Download URL (the API ``content`` link).
    """

    section: str
    record_id: str
    file_name: str
    size_bytes: int
    md5: str
    url: str


@dataclass(frozen=True)
class SpaceCheck:
    """Free space at a destination against what a run still needs.

    Attributes:
        probe_path: Existing directory the free space was measured at.
        free_bytes: Free bytes there.
        needed_bytes: Bytes still to be written.
        margin_bytes: Safety margin required on top.
    """

    probe_path: Path
    free_bytes: int
    needed_bytes: int
    margin_bytes: int

    @property
    def is_enough(self) -> bool:
        """Whether the free space covers the need plus the margin."""
        return self.free_bytes >= self.needed_bytes + self.margin_bytes


def dataset_root(public_root: Path) -> Path:
    """Return the dataset folder under the public-data root.

    Args:
        public_root: Value of ``$MERXEN_XENIUM_PUBLIC_ROOT``.

    Returns:
        ``<public_root>/xenium_v1_human_hippocampus_ff_ki``.
    """
    return Path(public_root) / DATASET_DIR


def raw_dir(public_root: Path) -> Path:
    """Return the folder that receives the downloaded tarballs."""
    return dataset_root(public_root) / "raw"


def metadata_dir(public_root: Path) -> Path:
    """Return the folder holding the record JSONs and the manifest."""
    return raw_dir(public_root) / "metadata"


def extracted_dir(public_root: Path, section: str) -> Path:
    """Return the folder holding the extracted members of one section."""
    return dataset_root(public_root) / "extracted" / section


def is_extracted(public_root: Path, item: RemoteFile) -> bool:
    """Whether the section of ``item`` was extracted from a verified bundle.

    Args:
        public_root: Public-data root.
        item: The bundle.

    Returns:
        True when the extraction record names this bundle's md5 as verified.
    """
    record_path = extracted_dir(public_root, item.section) / EXTRACTION_RECORD
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and record.get("source_md5") == item.md5
        and record.get("md5_verified") is True
    )


def section_label_from_file_name(file_name: str) -> str:
    """Derive the public section label from a bundle file name.

    Args:
        file_name: ``output-<instrument>__<slide>__<region>__<date>__<time>.tar.gz``.

    Returns:
        ``KI_HC_S<n>``, where ``n`` is the trailing number of the region name.

    Raises:
        FetchError: If the name does not have the expected form.
    """
    match = _BUNDLE_NAME.match(file_name)
    if match is None:
        raise FetchError(f"unexpected bundle file name: {file_name!r}")
    section = _REGION_SECTION.search(match.group("region"))
    if section is None:
        raise FetchError(f"no section number in the region of {file_name!r}")
    return f"{SECTION_LABEL_PREFIX}{int(section.group('number'))}"


def parse_record_files(record: dict[str, Any]) -> list[RemoteFile]:
    """Turn a Zenodo record API response into bundle descriptions.

    Args:
        record: Parsed JSON of ``GET /api/records/<id>``.

    Returns:
        One entry per ``.tar.gz`` file, in section order.

    Raises:
        FetchError: If a file lacks a size, an md5 checksum or a link.
    """
    record_id = str(record.get("id", record.get("recid", "")))
    files: list[RemoteFile] = []
    for entry in record.get("files", []):
        name = str(entry.get("key", ""))
        if not name.endswith(".tar.gz"):
            logger.warning("record %s: ignoring non-bundle file %s", record_id, name)
            continue
        checksum = str(entry.get("checksum", ""))
        algorithm, _, digest = checksum.partition(":")
        if algorithm != "md5" or not re.fullmatch(r"[0-9a-f]{32}", digest):
            raise FetchError(f"{name}: no md5 checksum in the record ({checksum!r})")
        url = str(entry.get("links", {}).get("self", ""))
        size = entry.get("size")
        if not url or not isinstance(size, int) or size <= 0:
            raise FetchError(f"{name}: the record gives no size or download link")
        files.append(
            RemoteFile(
                section=section_label_from_file_name(name),
                record_id=record_id,
                file_name=name,
                size_bytes=size,
                md5=digest,
                url=url,
            )
        )
    return sort_by_section(files)


def sort_by_section(files: Iterable[RemoteFile]) -> list[RemoteFile]:
    """Sort bundles by section number."""
    return sorted(
        files, key=lambda item: int(item.section[len(SECTION_LABEL_PREFIX) :])
    )


def _http_get_json(url: str, timeout: float = 60.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise FetchError(f"{url}: expected a JSON object")
    return payload


def fetch_records(
    record_ids: Sequence[str] = ZENODO_RECORDS,
    get_json: Callable[[str], dict[str, Any]] = _http_get_json,
) -> dict[str, dict[str, Any]]:
    """Fetch the Zenodo record JSONs.

    Args:
        record_ids: Record ids to fetch.
        get_json: Callable returning the parsed JSON of a URL (tests inject one).

    Returns:
        Record id -> record JSON.
    """
    return {rid: get_json(ZENODO_API.format(record_id=rid)) for rid in record_ids}


def list_remote_files(records: dict[str, dict[str, Any]]) -> list[RemoteFile]:
    """List every bundle of the given records, checking section labels are unique.

    Args:
        records: Record id -> record JSON.

    Returns:
        All bundles in section order.

    Raises:
        FetchError: If two bundles map to one section label.
    """
    files = [item for record in records.values() for item in parse_record_files(record)]
    seen: dict[str, str] = {}
    for item in files:
        if item.section in seen:
            raise FetchError(
                f"{item.file_name} and {seen[item.section]} both map to {item.section}"
            )
        seen[item.section] = item.file_name
    return sort_by_section(files)


def normalise_section(value: str) -> str:
    """Accept ``S3``, ``3`` or ``KI_HC_S3`` and return ``KI_HC_S3``."""
    token = value.strip().upper()
    if token.startswith(SECTION_LABEL_PREFIX):
        token = token[len(SECTION_LABEL_PREFIX) :]
    token = token.removeprefix("S")
    if not token.isdigit():
        raise FetchError(f"invalid section {value!r}: use e.g. S3 or KI_HC_S3")
    return f"{SECTION_LABEL_PREFIX}{int(token)}"


def select_sections(
    files: Sequence[RemoteFile], sections: Sequence[str] | None
) -> list[RemoteFile]:
    """Keep the requested sections (all when ``sections`` is empty).

    Raises:
        FetchError: If a requested section is not offered.
    """
    if not sections:
        return list(files)
    wanted = [normalise_section(value) for value in sections]
    offered = {item.section for item in files}
    missing = [label for label in wanted if label not in offered]
    if missing:
        raise FetchError(f"sections not offered: {missing}; offered: {sorted(offered)}")
    return [item for item in files if item.section in wanted]


def write_manifest(path: Path, files: Sequence[RemoteFile]) -> None:
    """Write the bundle list as a TSV the prepare step reads.

    Args:
        path: Destination ``manifest.tsv``.
        files: Bundles to list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for item in files:
            writer.writerow(
                [
                    item.section,
                    item.record_id,
                    item.file_name,
                    item.size_bytes,
                    item.md5,
                    item.url,
                ]
            )
    tmp.replace(path)


def read_manifest(path: Path) -> list[RemoteFile]:
    """Read a manifest written by :func:`write_manifest`."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return sort_by_section(
        RemoteFile(
            section=row["section"],
            record_id=row["record_id"],
            file_name=row["file_name"],
            size_bytes=int(row["size_bytes"]),
            md5=row["md5"],
            url=row["url"],
        )
        for row in rows
    )


def md5_of_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Return the hex md5 of a file, read in chunks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(path: Path) -> Path:
    """Return the ``<file>.md5`` marker written after a successful check."""
    return path.with_name(path.name + ".md5")


def is_verified(path: Path, item: RemoteFile) -> bool:
    """Whether ``path`` is complete and carries a matching md5 sidecar."""
    marker = sidecar_path(path)
    if not path.is_file() or not marker.is_file():
        return False
    if path.stat().st_size != item.size_bytes:
        return False
    recorded = marker.read_text(encoding="utf-8").split()
    return bool(recorded) and recorded[0] == item.md5


def verify_file(path: Path, item: RemoteFile) -> None:
    """Check size and md5 of a downloaded bundle and write the sidecar.

    Args:
        path: The downloaded file.
        item: Its expected size and md5.

    Raises:
        FetchError: On a size or checksum mismatch.
    """
    size = path.stat().st_size
    if size != item.size_bytes:
        raise FetchError(f"{path.name}: {size} bytes, expected {item.size_bytes}")
    actual = md5_of_file(path)
    if actual != item.md5:
        raise FetchError(f"{path.name}: md5 {actual}, expected {item.md5}")
    sidecar_path(path).write_text(f"{actual}  {path.name}\n", encoding="utf-8")


def bytes_present(path: Path) -> int:
    """Bytes already on disk for a bundle (finished file or ``.part``)."""
    if path.is_file():
        return path.stat().st_size
    part = path.with_name(path.name + ".part")
    return part.stat().st_size if part.is_file() else 0


def remaining_bytes(destination: Path, files: Sequence[RemoteFile]) -> int:
    """Bytes still to download for ``files`` into ``destination``."""
    total = 0
    for item in files:
        path = destination / item.file_name
        if is_verified(path, item):
            continue
        total += max(item.size_bytes - bytes_present(path), 0)
    return total


def nearest_existing(path: Path) -> Path:
    """Return ``path`` or its closest existing ancestor."""
    probe = Path(path).absolute()
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return probe


def check_space(destination: Path, needed_bytes: int, margin_bytes: int) -> SpaceCheck:
    """Measure the free space where ``destination`` is (or will be).

    Args:
        destination: Target folder; it need not exist yet.
        needed_bytes: Bytes still to write.
        margin_bytes: Extra free space to keep.

    Returns:
        The measurement.
    """
    probe = nearest_existing(destination)
    return SpaceCheck(
        probe_path=probe,
        free_bytes=shutil.disk_usage(probe).free,
        needed_bytes=needed_bytes,
        margin_bytes=margin_bytes,
    )


def _open_range(url: str, start: int, timeout: float) -> Any:
    headers = {"User-Agent": USER_AGENT}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def response_header(response: Any, name: str) -> str | None:
    """Return one header of an HTTP response (``None`` when absent)."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return None if value is None else str(value)


def range_start(response: Any) -> int | None:
    """Return the first byte a ``206`` response covers (``Content-Range``).

    Returns:
        The start offset, or ``None`` when the header is absent or unparsable.
    """
    value = response_header(response, "Content-Range")
    match = re.match(r"bytes\s+(\d+)-\d+/(?:\d+|\*)", value or "")
    return int(match.group(1)) if match else None


def _copy_stream(
    source: BinaryIO, target: IO[bytes], progress: Callable[[int], None]
) -> None:
    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
        target.write(chunk)
        progress(len(chunk))


def download_file(
    item: RemoteFile,
    destination: Path,
    *,
    retries: int = 20,
    timeout: float = 120.0,
    retry_wait_seconds: float = 15.0,
) -> Path:
    """Download one bundle with Range resume, then verify it.

    The bytes go to ``<name>.part``; a dropped connection resumes from its
    length. The file is renamed to its final name only after its size matches,
    and the md5 sidecar is written only after the checksum matches.

    Args:
        item: Bundle to download.
        destination: Folder receiving the file.
        retries: Connection attempts before giving up.
        timeout: Socket timeout in seconds.
        retry_wait_seconds: Pause between attempts.

    Returns:
        Path of the verified file.

    Raises:
        FetchError: If the download keeps failing or the checksum differs.
    """
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / item.file_name
    if is_verified(final, item):
        logger.info("%s: already verified, skipped", item.file_name)
        return final
    if final.is_file():
        logger.info("%s: complete but unverified; checking md5", item.file_name)
        verify_file(final, item)
        return final
    part = final.with_name(final.name + ".part")
    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if start > item.size_bytes:
            logger.warning(
                "%s: partial file longer than expected; restarting", part.name
            )
            part.unlink()
            start = 0
        if start == item.size_bytes:
            break
        try:
            with _open_range(item.url, start, timeout) as response:
                status = getattr(response, "status", 200)
                resumed = (
                    start > 0
                    and status == 206
                    and range_start(response) in (start, None)
                )
                mode = "ab" if resumed else "wb"
                if start > 0 and not resumed:
                    logger.warning(
                        "%s: server did not resume at byte %d; restarting",
                        item.file_name,
                        start,
                    )
                if not resumed and status == 206:
                    # A partial body from another offset cannot start or
                    # extend the file: drop the part and ask for all of it.
                    logger.warning(
                        "%s: partial response %r; restarting from byte 0",
                        item.file_name,
                        response_header(response, "Content-Range"),
                    )
                    part.unlink(missing_ok=True)
                    continue
                written = start if mode == "ab" else 0
                last_report = time.monotonic()

                def progress(n_bytes: int) -> None:
                    nonlocal written, last_report
                    written += n_bytes
                    if time.monotonic() - last_report > 30:
                        last_report = time.monotonic()
                        logger.info(
                            "%s: %.1f / %.1f GiB",
                            item.file_name,
                            written / GIB,
                            item.size_bytes / GIB,
                        )

                with part.open(mode) as handle:
                    _copy_stream(response, handle, progress)
        except (urllib.error.URLError, HTTPException, OSError, TimeoutError) as error:
            logger.warning(
                "%s: attempt %d/%d failed: %s", item.file_name, attempt, retries, error
            )
            time.sleep(retry_wait_seconds)
            continue
        if part.stat().st_size == item.size_bytes:
            break
    else:
        raise FetchError(
            f"{item.file_name}: download incomplete after {retries} attempts"
        )
    if part.stat().st_size != item.size_bytes:
        raise FetchError(
            f"{item.file_name}: download incomplete after {retries} attempts"
        )
    part.replace(final)
    verify_file(final, item)
    logger.info("%s: downloaded and md5-verified", item.file_name)
    return final


def format_bytes(n_bytes: int) -> str:
    """Render a byte count as GB (10^9) and GiB."""
    return f"{n_bytes:,} B ({n_bytes / 1e9:.2f} GB, {n_bytes / GIB:.2f} GiB)"


def report_plan(
    files: Sequence[RemoteFile],
    destination: Path,
    space: SpaceCheck,
) -> list[str]:
    """Build the human-readable plan of a run.

    Args:
        files: Selected bundles.
        destination: Where the tarballs go.
        space: Free-space measurement for the remaining bytes.

    Returns:
        Report lines.
    """
    lines = [f"Destination: {destination}", "Files:"]
    for item in files:
        state = "verified" if is_verified(destination / item.file_name, item) else ""
        present = bytes_present(destination / item.file_name)
        if not state and present:
            state = f"partial {present:,} B"
        lines.append(
            f"  {item.section}  record {item.record_id}  {item.file_name}  "
            f"{item.size_bytes:,} B  md5 {item.md5}  {state}".rstrip()
        )
    total = sum(item.size_bytes for item in files)
    n_sections = len(files)
    largest = max((item.size_bytes for item in files), default=0)
    kept = KEPT_BYTES_PER_SECTION_ESTIMATE * n_sections
    lines += [
        f"Sections: {n_sections}",
        f"Total download: {format_bytes(total)}",
        f"Still to download: {format_bytes(space.needed_bytes)}",
        f"Kept after extraction (estimate): {format_bytes(kept)}",
        "Optional extras (estimate): morphology_mip "
        f"{format_bytes(MIP_BYTES_PER_SECTION_ESTIMATE * n_sections)}; "
        "3-D morphology z-stack "
        f"{format_bytes(ZSTACK_BYTES_PER_SECTION_ESTIMATE * n_sections)}",
        "Peak disk by route (estimate):",
        f"  one section at a time, tarball deleted after extraction: "
        f"{format_bytes(kept + largest)}",
        f"  all tarballs kept, then extracted: {format_bytes(kept + total)}",
        f"  streamed extraction (prepare --stream): {format_bytes(kept)}",
        f"Free space at {space.probe_path}: {format_bytes(space.free_bytes)} "
        f"(margin {format_bytes(space.margin_bytes)}): "
        + ("enough" if space.is_enough else "NOT ENOUGH for this download"),
    ]
    return lines


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Public-data root; the dataset goes to xenium_v1_human_hippocampus_ff_ki/",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download", action="store_true", help="Download and verify the tarballs."
    )
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Save only the record JSONs and manifest.tsv.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=None,
        help="Sections to act on (S1..S6, 1..6 or KI_HC_S1..); default all.",
    )
    parser.add_argument(
        "--margin-gb",
        type=float,
        default=10.0,
        help="Free space to keep beyond the download, in GB (default 10).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fetch CLI.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    destination = raw_dir(args.public_root)
    try:
        records = fetch_records()
        files = select_sections(list_remote_files(records), args.sections)
    except (FetchError, OSError, HTTPException, ValueError) as error:
        # OSError covers URLError, socket timeouts and connection resets.
        logger.error("listing failed: %s", error)
        return 2
    extracted = [item for item in files if is_extracted(args.public_root, item)]
    for item in extracted:
        logger.info("%s: already extracted from a verified bundle", item.section)
    pending = [item for item in files if item not in extracted]
    margin = int(args.margin_gb * 1e9)
    space = check_space(destination, remaining_bytes(destination, pending), margin)
    for line in report_plan(files, destination, space):
        logger.info("%s", line)

    if args.metadata_only or args.download:
        meta = metadata_dir(args.public_root)
        meta.mkdir(parents=True, exist_ok=True)
        for record_id, record in records.items():
            (meta / f"zenodo_{record_id}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
        # The manifest always lists every section, so a later subset run
        # does not shrink it.
        write_manifest(meta / MANIFEST_NAME, list_remote_files(records))
        logger.info("Wrote record metadata and %s to %s", MANIFEST_NAME, meta)
    if not args.download:
        if not args.metadata_only:
            logger.info("Dry run: nothing downloaded. Pass --download to fetch.")
        return 0
    if not space.is_enough:
        logger.error(
            "Refusing to download: %s free, %s needed plus %s margin.",
            format_bytes(space.free_bytes),
            format_bytes(space.needed_bytes),
            format_bytes(space.margin_bytes),
        )
        return 3
    try:
        for item in pending:
            download_file(item, destination)
    except FetchError as error:
        logger.error("%s", error)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
