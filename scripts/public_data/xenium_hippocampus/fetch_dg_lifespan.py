"""List and download the public human dentate-gyrus lifespan Xenium dataset.

The dataset is GEO series GSE317381 (Xenium Prime 5K prototype panel, one
dentate-gyrus section per donor, each deposited as its own GSM with six
per-region XOA output files). The optional processed R objects of the paper
live in Zenodo record 17872293 (``--include-processed``); MerXen does not need
them.

Nothing is hard-coded about the files: the GSM accessions, titles and file
names come from the GEO sample SOFT text at run time, the exact sizes from the
series ``filelist.txt`` (HEAD ``Content-Length`` when a name is missing there),
and the Zenodo names, sizes and MD5 sums from the Zenodo REST API.

The default is a dry run. It prints every file with its size, the per-section
and total bytes, the gzip-trailer estimate of the decompressed size, the
destination ``$MERXEN_XENIUM_PUBLIC_ROOT/<DATASET_DIR>/raw`` and the free space
there. ``--download`` fetches with ``curl --continue-at -``
(resumable), verifies every file (size for GEO, which publishes no checksums;
size and MD5 for Zenodo) and refuses to start when the free space is below the
bytes still to fetch plus ``--margin-gb``. Complete files are skipped, so the
command can be re-run after an interruption. ``--metadata-only`` fetches only
the small files (at most ``--small-limit-mb`` each). Every non-dry run writes
``raw/listing.json``, which ``prepare_dg_lifespan.py`` and
``make_samplesheet_dg_lifespan.py`` read.

Usage::

    python scripts/public_data/xenium_hippocampus/fetch_dg_lifespan.py
    python scripts/public_data/xenium_hippocampus/fetch_dg_lifespan.py --metadata-only
    python scripts/public_data/xenium_hippocampus/fetch_dg_lifespan.py --download \
        --sections teen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

from merxen import local_config

logger = logging.getLogger("fetch_dg_lifespan")

DATASET_KEY: Final = "dg_lifespan"
DATASET_DIR: Final = "xenium_prime_5k_proto_human_dentate_gyrus_ff"
PAIR_ID_PREFIX: Final = "PUBXDG_"
GEO_SERIES: Final = "GSE317381"
ZENODO_RECORD: Final = "17872293"
GEO_SOFT_URL: Final = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    f"?acc={GEO_SERIES}&targ=gsm&form=text&view=brief"
)
GEO_FILELIST_URL: Final = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    f"{GEO_SERIES[:-3]}nnn/{GEO_SERIES}/suppl/filelist.txt"
)
ZENODO_API_URL: Final = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
USER_AGENT: Final = "merxen-public-data-fetch/1.0"
LISTING_FILE: Final = "listing.json"
PROCESSED_DIR: Final = f"zenodo_{ZENODO_RECORD}"
#: Suffix of each per-section GEO file (after ``<GSM>_<title>_``) and the role
#: the prepare step gives it. These are the generic Xenium output names.
ROLE_SUFFIXES: Final[dict[str, str]] = {
    "transcripts.parquet.gz": "transcripts",
    "morphology.ome.tif.gz": "morphology",
    "cell_feature_matrix.h5": "cell_feature_matrix",
    "cells.parquet.gz": "cells",
    "cell_boundaries.parquet.gz": "cell_boundaries",
    "nucleus_boundaries.parquet.gz": "nucleus_boundaries",
}
REQUIRED_ROLES: Final = frozenset(ROLE_SUFFIXES.values())
GZIP_ISIZE_MODULUS: Final = 2**32
TIMEOUT_S: Final = 60
GIB: Final = 1024**3


class FetchError(RuntimeError):
    """Raised when the listing is inconsistent or a download cannot proceed."""


@dataclass
class RemoteFile:
    """One downloadable file.

    Attributes:
        name: File name on the server (and on disk).
        url: HTTPS download URL.
        size: Exact size in bytes.
        role: Prepare role (see :data:`ROLE_SUFFIXES`), or None for a file
            MerXen does not read.
        md5: Published MD5 hex digest, or None (GEO publishes none).
        isize: Gzip trailer ``ISIZE`` (decompressed size mod 2**32), or None.
    """

    name: str
    url: str
    size: int
    role: str | None = None
    md5: str | None = None
    isize: int | None = None

    @property
    def decompressed_estimate(self) -> int | None:
        """Decompressed size from the gzip trailer, or None if unknown."""
        if self.isize is None:
            return None
        return estimate_decompressed_size(self.size, self.isize)


@dataclass
class Section:
    """One tissue section (one GEO sample).

    Attributes:
        accession: GSM accession.
        title: GEO sample title.
        label: Short section label used in the pair id.
        pair_id: MerXen pair id of the section.
        files: The section's files.
    """

    accession: str
    title: str
    label: str
    pair_id: str
    files: list[RemoteFile] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Sum of the section's file sizes."""
        return sum(item.size for item in self.files)


@dataclass
class Listing:
    """Everything the dataset offers.

    Attributes:
        sections: GEO sections, in GEO order.
        processed: Optional Zenodo files (empty unless requested).
    """

    sections: list[Section]
    processed: list[RemoteFile] = field(default_factory=list)


@dataclass
class PlannedFile:
    """A file selected for download with its local state.

    Attributes:
        remote: The remote file.
        path: Destination path.
        local_bytes: Bytes already on disk (complete file or ``.part``).
        is_complete: Whether the destination already holds the whole file.
    """

    remote: RemoteFile
    path: Path
    local_bytes: int
    is_complete: bool

    @property
    def remaining_bytes(self) -> int:
        """Bytes still to download."""
        return 0 if self.is_complete else max(self.remote.size - self.local_bytes, 0)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_gsm_soft(text: str) -> list[tuple[str, str, list[str]]]:
    """Parse GEO sample SOFT text into samples.

    Args:
        text: SOFT text with ``^SAMPLE``, ``!Sample_title`` and
            ``!Sample_supplementary_file*`` lines.

    Returns:
        ``(accession, title, supplementary URLs)`` per sample, in file order.
    """
    samples: list[tuple[str, str, list[str]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        key, _, value = line.partition(" = ")
        value = value.strip()
        if key == "^SAMPLE":
            samples.append((value, "", []))
        elif not samples:
            continue
        elif key == "!Sample_title":
            accession, _, urls = samples[-1]
            samples[-1] = (accession, value, urls)
        elif key.startswith("!Sample_supplementary_file") and value.upper() != "NONE":
            samples[-1][2].append(value)
    return samples


def parse_filelist(text: str) -> dict[str, int]:
    """Parse a GEO ``filelist.txt`` into file sizes.

    Args:
        text: Tab-separated ``Archive/File, Name, Time, Size, Type`` rows.

    Returns:
        Size in bytes by file name (``File`` rows only, not the archive).
    """
    sizes: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4 or parts[0] != "File":
            continue
        sizes[parts[1]] = int(parts[3])
    return sizes


def https_url(url: str) -> str:
    """Return the HTTPS form of an NCBI FTP URL (other URLs unchanged)."""
    return re.sub(r"^ftp://", "https://", url.strip())


def file_role(name: str) -> str | None:
    """Return the prepare role of a GEO file name, or None if unrecognised."""
    for suffix, role in ROLE_SUFFIXES.items():
        if name.endswith("_" + suffix) or name == suffix:
            return role
    return None


def section_label(title: str, accession: str) -> str:
    """Return the short label of a section from its GEO title.

    The label is the trailing alphabetic word of the title (the deposited
    titles end in the donor age group), else the accession.

    Args:
        title: GEO sample title.
        accession: GSM accession (fallback).

    Returns:
        An upper-case alphanumeric label.
    """
    match = re.search(r"([A-Za-z]+)\s*$", title)
    label = match.group(1) if match else accession
    return re.sub(r"[^A-Za-z0-9]", "", label).upper() or accession.upper()


def build_sections(
    soft_text: str,
    filelist_text: str,
    head_size: Callable[[str], int] | None = None,
) -> list[Section]:
    """Combine the SOFT text and the file list into sections.

    Args:
        soft_text: GEO sample SOFT text.
        filelist_text: GEO series ``filelist.txt``.
        head_size: Called with a URL for a file missing from the file list;
            None makes that an error.

    Returns:
        The sections, with unique labels and pair ids.

    Raises:
        FetchError: If a size is unknown, a required role is missing or two
            sections would share a label.
    """
    sizes = parse_filelist(filelist_text)
    sections: list[Section] = []
    for accession, title, urls in parse_gsm_soft(soft_text):
        files: list[RemoteFile] = []
        for url in urls:
            https = https_url(url)
            name = https.rsplit("/", 1)[-1]
            size = sizes.get(name)
            if size is None:
                if head_size is None:
                    raise FetchError(f"{name}: not in filelist.txt and no HEAD size")
                size = head_size(https)
            files.append(
                RemoteFile(name=name, url=https, size=size, role=file_role(name))
            )
        roles = {item.role for item in files}
        missing = sorted(REQUIRED_ROLES - roles)
        if missing:
            raise FetchError(f"{accession}: no file for roles {missing}")
        label = section_label(title, accession)
        sections.append(
            Section(
                accession=accession,
                title=title,
                label=label,
                pair_id=PAIR_ID_PREFIX + label,
                files=files,
            )
        )
    labels = [section.label for section in sections]
    if len(set(labels)) != len(labels):
        for section in sections:
            section.label = section.accession.upper()
            section.pair_id = PAIR_ID_PREFIX + section.label
    if not sections:
        raise FetchError("the GEO SOFT text lists no samples")
    return sections


def parse_zenodo_record(record: dict[str, object]) -> list[RemoteFile]:
    """Return the files of a Zenodo REST API record.

    Args:
        record: Parsed JSON of ``/api/records/<id>``.

    Returns:
        One file per entry, with its MD5 when the checksum is ``md5:...``.

    Raises:
        FetchError: If the record has no file list.
    """
    entries = record.get("files")
    if not isinstance(entries, list):
        raise FetchError("Zenodo record has no files list")
    files: list[RemoteFile] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry["key"])
        links = entry.get("links") or {}
        url = str(links.get("self") or f"{ZENODO_API_URL}/files/{name}/content")
        checksum = str(entry.get("checksum") or "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        files.append(RemoteFile(name=name, url=url, size=int(entry["size"]), md5=md5))
    return files


def estimate_decompressed_size(compressed_size: int, isize: int) -> int:
    """Resolve a gzip ``ISIZE`` (size mod 2**32) to a full size estimate.

    Picks the smallest wrap count giving at least 99 % of the compressed size:
    already-compressed payloads (JPEG 2000 tiles, ZSTD pages) barely shrink
    under gzip, so the true size is never far below the compressed one.

    Args:
        compressed_size: Size of the ``.gz`` file.
        isize: Its last four bytes as a little-endian integer.

    Returns:
        The estimated decompressed size in bytes.
    """
    floor = int(compressed_size * 0.99)
    wraps = max(0, -(-(floor - isize) // GZIP_ISIZE_MODULUS))
    return isize + wraps * GZIP_ISIZE_MODULUS


# --------------------------------------------------------------------------- #
# Network (listing only; downloads go through curl)
# --------------------------------------------------------------------------- #


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None):  # noqa: ANN202
    request = urllib.request.Request(url, method=method)
    request.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)  # noqa: S310


def fetch_text(url: str) -> str:
    """Return the body of a small text resource."""
    with _request(url) as response:
        body: bytes = response.read()
    return body.decode("utf-8", errors="replace")


def head_size(url: str) -> int:
    """Return the ``Content-Length`` of a URL from a HEAD request."""
    with _request(url, method="HEAD") as response:
        length = response.headers.get("Content-Length")
    if length is None:
        raise FetchError(f"{url}: no Content-Length")
    return int(length)


def fetch_isize(url: str) -> int:
    """Return the gzip ``ISIZE`` trailer of a remote ``.gz`` (4-byte range)."""
    with _request(url, headers={"Range": "bytes=-4"}) as response:
        tail = response.read()
    if len(tail) != 4:
        raise FetchError(f"{url}: range request returned {len(tail)} bytes")
    return int.from_bytes(tail, "little")


def fetch_listing(*, include_processed: bool, with_isize: bool) -> Listing:
    """Query GEO (and optionally Zenodo) for the dataset's files.

    Args:
        include_processed: Also list the Zenodo processed-data record.
        with_isize: Read the gzip trailer of every ``.gz`` file.

    Returns:
        The listing.
    """
    sections = build_sections(
        fetch_text(GEO_SOFT_URL), fetch_text(GEO_FILELIST_URL), head_size
    )
    processed: list[RemoteFile] = []
    if include_processed:
        processed = parse_zenodo_record(json.loads(fetch_text(ZENODO_API_URL)))
    if with_isize:
        gz_files = [
            item
            for section in sections
            for item in section.files
            if item.name.endswith(".gz")
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for item, isize in zip(
                gz_files, pool.map(lambda f: fetch_isize(f.url), gz_files), strict=True
            ):
                item.isize = isize
    return Listing(sections=sections, processed=processed)


# --------------------------------------------------------------------------- #
# Selection, planning, verification
# --------------------------------------------------------------------------- #


def select_sections(
    sections: Sequence[Section], wanted: Iterable[str]
) -> list[Section]:
    """Return the sections named by accession, label or pair id.

    Args:
        sections: All sections.
        wanted: Names (case-insensitive); empty selects every section.

    Returns:
        The selected sections in listing order.

    Raises:
        FetchError: If a name matches no section.
    """
    keys = [name.strip().upper() for name in wanted if name.strip()]
    if not keys:
        return list(sections)
    chosen: list[Section] = []
    for key in keys:
        matches = [
            s
            for s in sections
            if key in {s.accession.upper(), s.label.upper(), s.pair_id.upper()}
        ]
        if not matches:
            known = ", ".join(f"{s.label} ({s.accession})" for s in sections)
            raise FetchError(f"unknown section {key!r}; known: {known}")
        chosen.extend(m for m in matches if m not in chosen)
    return [s for s in sections if s in chosen]


def dataset_dir(public_root: Path) -> Path:
    """Return ``<public root>/xenium_prime_5k_proto_human_dentate_gyrus_ff``."""
    return public_root / DATASET_DIR


def plan_files(
    listing: Listing,
    raw_dir: Path,
    *,
    sections: Sequence[Section],
    small_limit: int | None = None,
) -> list[PlannedFile]:
    """Return the files to fetch with their local state.

    Args:
        listing: The listing (its processed files are always included).
        raw_dir: ``.../dg_lifespan/raw``.
        sections: Selected sections.
        small_limit: When set, keep only files of at most this many bytes.

    Returns:
        One entry per file: ``raw/<GSM>/<name>`` or ``raw/zenodo_<id>/<name>``.
    """
    pairs: list[tuple[RemoteFile, Path]] = [
        (item, raw_dir / section.accession / item.name)
        for section in sections
        for item in section.files
    ]
    pairs += [(item, raw_dir / PROCESSED_DIR / item.name) for item in listing.processed]
    planned: list[PlannedFile] = []
    for remote, path in pairs:
        if small_limit is not None and remote.size > small_limit:
            continue
        part = partial_path(path)
        if path.is_file():
            local = path.stat().st_size
            is_complete = local == remote.size
        elif part.is_file():
            local = part.stat().st_size
            is_complete = False
        else:
            local, is_complete = 0, False
        planned.append(PlannedFile(remote, path, local, is_complete))
    return planned


def partial_path(path: Path) -> Path:
    """Return the in-progress download path of ``path``."""
    return path.with_name(path.name + ".part")


def md5_stamp_path(path: Path) -> Path:
    """Return the stamp file that records a verified MD5."""
    return path.with_name(path.name + ".md5.ok")


def file_md5(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()  # noqa: S324 - matching the published checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, remote: RemoteFile) -> None:
    """Check a downloaded file's size and, when published, its MD5.

    A verified MD5 is recorded in a ``.md5.ok`` stamp so a re-run does not
    hash the file again.

    Args:
        path: Local file.
        remote: The file it should equal.

    Raises:
        FetchError: On a size or MD5 mismatch.
    """
    size = path.stat().st_size
    if size != remote.size:
        raise FetchError(f"{path.name}: {size} bytes on disk, expected {remote.size}")
    if remote.md5 is None:
        return
    stamp = md5_stamp_path(path)
    if stamp.is_file() and stamp.read_text().strip() == remote.md5:
        return
    actual = file_md5(path)
    if actual != remote.md5:
        raise FetchError(f"{path.name}: MD5 {actual}, expected {remote.md5}")
    stamp.write_text(remote.md5 + "\n")


def free_bytes(path: Path) -> int:
    """Return the free bytes on the file system that holds (or would hold) path."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return shutil.disk_usage(probe).free


def curl_command(url: str, part: Path) -> list[str]:
    """Return the resumable curl command for one file."""
    return [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "15",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--user-agent",
        USER_AGENT,
        "--output",
        str(part),
        url,
    ]


def download_file(planned: PlannedFile, runner: Callable[[list[str]], None]) -> None:
    """Download one file resumably, verify it and move it into place.

    Args:
        planned: The file and its destination.
        runner: Runs a command and raises on failure (``subprocess`` in
            production, a fake in tests).

    Raises:
        FetchError: If the result fails verification.
    """
    path = planned.path
    if path.is_file() and path.stat().st_size == planned.remote.size:
        verify_file(path, planned.remote)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = partial_path(path)
    if path.is_file():
        # A wrong-size complete file is resumed as a partial download.
        path.replace(part)
    if part.is_file() and part.stat().st_size > planned.remote.size:
        part.unlink()
    if not (part.is_file() and part.stat().st_size == planned.remote.size):
        # A complete .part (interrupted before the rename) is not re-requested:
        # a range starting at the end of the file fails with HTTP 416.
        logger.info(
            "Downloading %s (%s)",
            planned.remote.name,
            human_bytes(planned.remote.size),
        )
        runner(curl_command(planned.remote.url, part))
    try:
        verify_file(part, planned.remote)
    except FetchError:
        # A corrupt .part would otherwise be resumed (and fail) forever.
        part.unlink(missing_ok=True)
        raise
    stamp = md5_stamp_path(part)
    part.replace(path)
    if stamp.is_file():
        stamp.replace(md5_stamp_path(path))
    logger.info("Verified %s", planned.remote.name)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)  # noqa: S603


# --------------------------------------------------------------------------- #
# Listing file and report
# --------------------------------------------------------------------------- #


def listing_to_dict(listing: Listing) -> dict[str, object]:
    """Return the JSON form of a listing (written to ``raw/listing.json``)."""
    return {
        "dataset": DATASET_KEY,
        "geo_series": GEO_SERIES,
        "zenodo_record": ZENODO_RECORD,
        "sections": [asdict(section) for section in listing.sections],
        "processed": [asdict(item) for item in listing.processed],
    }


def listing_from_dict(data: dict[str, object]) -> Listing:
    """Rebuild a listing from :func:`listing_to_dict` output."""
    sections_raw = data.get("sections")
    processed_raw = data.get("processed") or []
    if not isinstance(sections_raw, list) or not isinstance(processed_raw, list):
        raise FetchError("listing.json has no sections list")
    sections = [
        Section(
            accession=s["accession"],
            title=s["title"],
            label=s["label"],
            pair_id=s["pair_id"],
            files=[RemoteFile(**f) for f in s["files"]],
        )
        for s in sections_raw
    ]
    return Listing(
        sections=sections, processed=[RemoteFile(**f) for f in processed_raw]
    )


def human_bytes(value: float) -> str:
    """Format a byte count with binary units (``12.3 GiB``)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def report(
    listing: Listing,
    sections: Sequence[Section],
    planned: Sequence[PlannedFile],
    raw_dir: Path,
    free: int,
    margin: int,
) -> int:
    """Log the dry-run report and return the bytes still to download."""
    by_path = {p.path: p for p in planned}
    total = 0
    decompressed_total = 0
    zstack_peak = 0
    for section in sections:
        logger.info(
            "\n%s  %s  title=%r  pair_id=%s",
            section.accession,
            section.label,
            section.title,
            section.pair_id,
        )
        for item in section.files:
            path = raw_dir / section.accession / item.name
            state = by_path.get(path)
            status = "-"
            if state is not None:
                status = (
                    "complete"
                    if state.is_complete
                    else (
                        f"partial {human_bytes(state.local_bytes)}"
                        if state.local_bytes
                        else "missing"
                    )
                )
            out = item.decompressed_estimate
            if item.role == "morphology" and out is not None:
                zstack_peak = max(zstack_peak, out)
            elif out is not None:
                decompressed_total += out
            elif item.role is not None and item.name.endswith(".gz"):
                # No trailer read (--no-isize): the compressed size is a floor.
                decompressed_total += item.size
            # Files that are not gzipped (the h5) are hard-linked: no new bytes.
            logger.info(
                "  %-22s %16s B  %10s  decompressed~%-10s  %-9s %s",
                item.role or "-",
                f"{item.size:,}",
                human_bytes(item.size),
                human_bytes(out) if out is not None else "n/a",
                status,
                item.name,
            )
        logger.info(
            "  section total %s B (%s)",
            f"{section.total_bytes:,}",
            human_bytes(section.total_bytes),
        )
        total += section.total_bytes
    if listing.processed:
        logger.info("\nZenodo record %s (optional processed R objects)", ZENODO_RECORD)
        for item in listing.processed:
            logger.info(
                "  %16s B  %10s  md5=%s  %s",
                f"{item.size:,}",
                human_bytes(item.size),
                item.md5,
                item.name,
            )
            total += item.size
    remaining = sum(p.remaining_bytes for p in planned)
    selected = sum(p.remote.size for p in planned)
    logger.info("\nAll listed files: %s B (%s)", f"{total:,}", human_bytes(total))
    logger.info(
        "Selected for this run: %s B (%s); still to fetch: %s",
        f"{selected:,}",
        human_bytes(selected),
        human_bytes(remaining),
    )
    logger.info("Destination: %s", raw_dir)
    logger.info(
        "Free space there: %s (margin required: %s)",
        human_bytes(free),
        human_bytes(margin),
    )
    if decompressed_total:
        logger.info(
            "Prepare step (prepare_dg_lifespan.py): about %s of decompressed "
            "parquet kept beside the downloads (the h5 is hard-linked), plus a "
            "transient decompressed z-stack of up to %s per section, a 2-D "
            "focus image and a cells.zarr.zip per section (see its --dry-run).",
            human_bytes(decompressed_total),
            human_bytes(zstack_peak),
        )
    if free < remaining + margin:
        logger.warning(
            "NOT ENOUGH SPACE: %s free < %s to fetch + %s margin",
            human_bytes(free),
            human_bytes(remaining),
            human_bytes(margin),
        )
    return remaining


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Root of the public validation datasets.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download", action="store_true", help="Download (default: dry run)."
    )
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only files of at most --small-limit-mb each.",
    )
    parser.add_argument(
        "--sections",
        default="",
        help="Comma-separated GSM accessions, labels or pair ids (default: all).",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help=f"Also list/fetch Zenodo record {ZENODO_RECORD} (processed R objects).",
    )
    parser.add_argument("--small-limit-mb", type=float, default=250.0)
    parser.add_argument(
        "--margin-gb", type=float, default=20.0, help="Free-space margin in GiB."
    )
    parser.add_argument("--jobs", type=int, default=3, help="Parallel downloads.")
    parser.add_argument(
        "--no-isize",
        action="store_true",
        help="Skip the 4-byte range requests that estimate decompressed sizes.",
    )
    parser.add_argument(
        "--write-listing",
        type=Path,
        default=None,
        help="Also write the listing JSON to this path (allowed in a dry run).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the listing, the dry-run report and optionally the download.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    raw_dir = dataset_dir(args.public_root) / "raw"
    listing = fetch_listing(
        include_processed=args.include_processed, with_isize=not args.no_isize
    )
    sections = select_sections(listing.sections, args.sections.split(","))
    small_limit = int(args.small_limit_mb * 1024**2) if args.metadata_only else None
    planned = plan_files(listing, raw_dir, sections=sections, small_limit=small_limit)
    free = free_bytes(raw_dir)
    margin = int(args.margin_gb * GIB)
    remaining = report(listing, sections, planned, raw_dir, free, margin)
    if args.write_listing is not None:
        args.write_listing.parent.mkdir(parents=True, exist_ok=True)
        args.write_listing.write_text(
            json.dumps(listing_to_dict(listing), indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Listing written to %s", args.write_listing)
    if not (args.download or args.metadata_only):
        logger.info("\nDry run: nothing downloaded. Use --metadata-only or --download.")
        return 0
    if free < remaining + margin:
        logger.error("Refusing to download: not enough free space (see above).")
        return 2
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / LISTING_FILE).write_text(
        json.dumps(listing_to_dict(listing), indent=2) + "\n", encoding="utf-8"
    )
    todo = [p for p in planned if not p.is_complete or p.remote.md5]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(download_file, p, _run): p for p in todo}
        for future, planned_file in futures.items():
            try:
                future.result()
            except (FetchError, subprocess.CalledProcessError, OSError) as error:
                failures.append(f"{planned_file.remote.name}: {error}")
    for failure in failures:
        logger.error("FAILED %s", failure)
    logger.info(
        "%d of %d files complete and verified.",
        len(planned) - len(failures),
        len(planned),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FetchError, urllib.error.URLError) as error:
        logger.error("%s", error)
        sys.exit(1)
