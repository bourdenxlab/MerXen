"""Public vendor panel gene lists for ``annotation-panel-simulate`` (plan §8.8).

The simulation design aid (``merxen annotation-panel-simulate``) predicts what
a panel can resolve before any data exist. Its first inputs are the public
10x Genomics panel definitions the plan names (M3b exit): the Xenium Prime 5K
Human and Mouse Pan Tissue and Pathways panels and the Xenium v1 Human Brain
(266 genes) and Mouse Brain (248 genes) panels. They are small gene tables
(gene symbol, Ensembl ID, codewords, annotations), not datasets: no public
validation data are downloaded (OD-E1 / OD-E9, simulation only).

Each list is pinned by URL, size, sha256 and gene count, checked on the first
download (M3b, 2026-09-27). ``fetch_public_panel`` downloads it (or reuses a
verified copy), refuses a file whose sha256 changed (a vendor update is a new
panel family, never a silent swap) and writes a normalised gene table
(``gene_symbol``, ``gene_id``) that ``annotation-panel-simulate`` and
``annotation-panel --panel-genes-path`` read, plus a manifest with the URL,
size, sha256 and retrieval time. The lists are written outside the repository
(no data in the repo).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

TENX_PANEL_BASE_URL: Final = (
    "https://cdn.10xgenomics.com/raw/upload/software-support/Xenium-panels"
)
# The 10x CDN answers scripted clients only with a browser-like agent.
USER_AGENT: Final = "Mozilla/5.0 (merxen annotation-panel-fetch)"
DOWNLOAD_TIMEOUT_S: Final = 120
MANIFEST_SUFFIX: Final = ".manifest.json"
GENE_LIST_SUFFIX: Final = ".gene_list.csv"
RAW_DIR: Final = "raw"


class PublicPanelError(RuntimeError):
    """A public panel list could not be fetched or failed its pin."""


@dataclass(frozen=True)
class PublicPanelList:
    """A pinned public panel gene list.

    Attributes:
        key: Registry key (``annotation-panel-fetch --panel``).
        title: Vendor panel name.
        species: ``human`` or ``mouse``.
        platform: Platform the panel serves (``XENIUM``).
        url: Download URL.
        file_name: File name of the download.
        size: Size in bytes at pinning.
        sha256: sha256 at pinning.
        n_genes: Genes (rows) at pinning.
        symbol_column: Column of the gene symbols.
        id_column: Column of the Ensembl gene IDs.
    """

    key: str
    title: str
    species: str
    platform: str
    url: str
    file_name: str
    size: int
    sha256: str
    n_genes: int
    symbol_column: str
    id_column: str


# Pinned 2026-09-27 (M3b stage D): URLs from the 10x "Pre-designed Xenium
# Prime 5K" and "Pre-designed Xenium v1" panel pages; size and sha256
# measured on the first download.
PUBLIC_PANEL_LISTS: Final[dict[str, PublicPanelList]] = {
    item.key: item
    for item in (
        PublicPanelList(
            key="xenium_prime_5k_human",
            title="Xenium Prime 5K Human Pan Tissue and Pathways Panel",
            species="human",
            platform="XENIUM",
            url=f"{TENX_PANEL_BASE_URL}/5K_panel_files/"
            "XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv",
            file_name="XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv",
            size=1_091_372,
            sha256="833ddb3008f2eb5e1b39053a988869c74db82cbaa8124d932965459f3169c58a",
            n_genes=5001,
            symbol_column="gene_name",
            id_column="gene_id",
        ),
        PublicPanelList(
            key="xenium_prime_5k_mouse",
            title="Xenium Prime 5K Mouse Pan Tissue and Pathways Panel",
            species="mouse",
            platform="XENIUM",
            url=f"{TENX_PANEL_BASE_URL}/5K_panel_files/"
            "XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv",
            file_name="XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv",
            size=831_051,
            sha256="68e79a502fb41169aed2399cddf8ef62023aa0b068d6f59aafe74be2a0a26105",
            n_genes=5006,
            symbol_column="gene_name",
            id_column="gene_id",
        ),
        PublicPanelList(
            key="xenium_human_brain_v1",
            title="Xenium Human Brain Gene Expression Panel (v1)",
            species="human",
            platform="XENIUM",
            url=f"{TENX_PANEL_BASE_URL}/hBrain_panel_files/"
            "Xenium_hBrain_v1_metadata.csv",
            file_name="Xenium_hBrain_v1_metadata.csv",
            size=10_586,
            sha256="014acee5a1f94ac2fe08ac6f830cc89ebe3109a8c89ca5d66b076ee139ebfc77",
            n_genes=266,
            symbol_column="Genes",
            id_column="Ensembl_ID",
        ),
        PublicPanelList(
            key="xenium_mouse_brain_v1",
            title="Xenium Mouse Brain Gene Expression Panel (v1)",
            species="mouse",
            platform="XENIUM",
            url=f"{TENX_PANEL_BASE_URL}/mBrain_panel_files/"
            "Xenium_mBrain_v1_metadata.csv",
            file_name="Xenium_mBrain_v1_metadata.csv",
            size=11_085,
            sha256="7b4e12f769011da17cee091a2394ec01f6babaf0499fd36dbbba3e4751fe2de0",
            n_genes=248,
            symbol_column="Genes",
            id_column="Ensembl_ID",
        ),
    )
}

Downloader = Callable[[str], bytes]


def download_url(url: str) -> bytes:
    """Download a URL (``PublicPanelList.url``) and return its bytes.

    Args:
        url: The URL.

    Returns:
        The response body.

    Raises:
        PublicPanelError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
            data: bytes = response.read()
    except OSError as error:
        raise PublicPanelError(f"download of {url} failed: {error}") from error
    return data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_panel_table(data: bytes, item: PublicPanelList) -> list[tuple[str, str]]:
    """Return the (symbol, Ensembl ID) rows of a pinned vendor table.

    Args:
        data: The downloaded file.
        item: Its registry entry.

    Returns:
        One ``(symbol, gene_id)`` per row, in file order.

    Raises:
        PublicPanelError: If a column is missing, a row is empty or repeated,
            or the gene count differs from the pin.
    """
    # utf-8-sig drops the byte-order mark some vendor tables start with.
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    fields = [str(name).strip() for name in reader.fieldnames or []]
    for column in (item.symbol_column, item.id_column):
        if column not in fields:
            raise PublicPanelError(
                f"{item.file_name} has no column {column!r}: {fields}"
            )
    rows: list[tuple[str, str]] = []
    for record in reader:
        cleaned = {
            str(key).strip(): (value or "").strip() for key, value in record.items()
        }
        symbol, gene_id = cleaned[item.symbol_column], cleaned[item.id_column]
        if not symbol or not gene_id:
            raise PublicPanelError(f"{item.file_name} has an empty row: {record}")
        rows.append((symbol, gene_id))
    ids = [gene_id for _symbol, gene_id in rows]
    if len(set(ids)) != len(ids):
        raise PublicPanelError(f"{item.file_name} repeats gene IDs")
    if len(rows) != item.n_genes:
        raise PublicPanelError(
            f"{item.file_name} has {len(rows)} genes, pinned {item.n_genes}"
        )
    return rows


@dataclass(frozen=True)
class FetchedPanel:
    """A fetched public panel list.

    Attributes:
        item: Its registry entry.
        raw_path: The verified vendor file.
        gene_list_path: The normalised gene table (``gene_symbol``,
            ``gene_id``).
        manifest_path: URL, size, sha256, gene count and retrieval time.
        downloaded: Whether this call downloaded it (else a verified copy).
    """

    item: PublicPanelList
    raw_path: Path
    gene_list_path: Path
    manifest_path: Path
    downloaded: bool


def _previous_retrieval(manifest_path: Path, sha256: str) -> dict[str, Any] | None:
    """The retrieval record of an earlier manifest of the same pinned file."""
    if not manifest_path.is_file():
        return None
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(previous, dict) or previous.get("sha256") != sha256:
        return None
    if "first_retrieved_at" not in previous:
        return None
    return {
        "downloaded": bool(previous.get("downloaded")),
        "first_retrieved_at": previous.get("first_retrieved_at"),
    }


def fetch_public_panel(
    key: str,
    out_dir: Path | str,
    *,
    downloader: Downloader = download_url,
) -> FetchedPanel:
    """Fetch one pinned public panel list and write its normalised gene table.

    A verified copy under ``<out_dir>/raw`` is reused; otherwise the file is
    downloaded and must match the pinned size and sha256. The manifest
    records the first retrieval (``downloaded``, ``first_retrieved_at``),
    kept from an earlier manifest when the copy is reused, and the latest
    check (``last_verified_at``).

    Args:
        key: ``PUBLIC_PANEL_LISTS`` key.
        out_dir: Output directory (outside the repository).
        downloader: ``url -> bytes`` (tests pass a fake).

    Returns:
        The fetched panel.

    Raises:
        PublicPanelError: If the key is unknown or the file fails its pin.
    """
    item = PUBLIC_PANEL_LISTS.get(key)
    if item is None:
        raise PublicPanelError(
            f"unknown public panel {key!r}; known: {sorted(PUBLIC_PANEL_LISTS)}"
        )
    root = Path(out_dir)
    raw_path = root / RAW_DIR / item.file_name
    data: bytes | None = None
    downloaded = False
    if raw_path.is_file():
        cached = raw_path.read_bytes()
        if _sha256(cached) == item.sha256:
            data = cached
        else:
            logger.warning("%s does not match its pin; downloading again", raw_path)
    if data is None:
        data = downloader(item.url)
        downloaded = True
        digest = _sha256(data)
        if digest != item.sha256 or len(data) != item.size:
            raise PublicPanelError(
                f"{item.url}: sha256 {digest} / {len(data)} bytes, pinned "
                f"{item.sha256} / {item.size} bytes (the vendor file changed; a "
                "new version is a new panel family and needs a new pin)"
            )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(data)
    rows = parse_panel_table(data, item)
    gene_list_path = root / f"{item.key}{GENE_LIST_SUFFIX}"
    with gene_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene_symbol", "gene_id"])
        writer.writerows(rows)
    manifest_path = root / f"{item.key}{MANIFEST_SUFFIX}"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    previous = _previous_retrieval(manifest_path, item.sha256)
    if downloaded:
        retrieval = {"downloaded": True, "first_retrieved_at": now}
    else:
        retrieval = previous or {"downloaded": False, "first_retrieved_at": None}
    manifest = {
        "key": item.key,
        "title": item.title,
        "species": item.species,
        "platform": item.platform,
        "url": item.url,
        "file": str(raw_path),
        "size": len(data),
        "sha256": item.sha256,
        "n_genes": len(rows),
        "gene_list": str(gene_list_path),
        "gene_list_sha256": _sha256(gene_list_path.read_bytes()),
        **retrieval,
        "last_verified_at": now,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "%s: %d genes (%s, sha256 %s)",
        item.key,
        len(rows),
        "downloaded" if downloaded else "verified copy",
        item.sha256[:16],
    )
    return FetchedPanel(
        item=item,
        raw_path=raw_path,
        gene_list_path=gene_list_path,
        manifest_path=manifest_path,
        downloaded=downloaded,
    )
