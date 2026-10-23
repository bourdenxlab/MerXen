"""Acquire and verify the official Allen inputs used to build pciSeq profiles."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)
ATLAS_RELEASE = "20260711"
MANIFEST_SHA256 = "67a6af49af3e304e10c2550d2e00bdc6fed71c8e7db8f8b0687017db0817355f"
_BASE_URL = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
_BLOCK_SIZE = 8 * 1024 * 1024
_TAXONOMY_KEYS = (
    "cluster",
    "cluster_annotation_term",
    "cluster_annotation_term_set",
    "cluster_to_cluster_annotation_membership",
)
_SOURCES = {
    "wmb": ("WMB-10Xv3", "WMB-10X", "WMB-taxonomy", 13),
    "whb": ("WHB-10Xv3", "WHB-10Xv3", "WHB-taxonomy", 2),
    "seaad": (
        "SEA-AD-Multiregion-10X",
        "SEA-AD-Multiregion-10X",
        "SEA-AD-Multiregion-taxonomy",
        10,
    ),
}


@dataclass(frozen=True)
class AtlasAssets:
    """Verified expression and metadata paths for one pinned Allen source."""

    source: str
    root: Path
    release: str
    manifest_path: Path
    expression_paths: tuple[Path, ...]
    metadata_paths: dict[str, Path]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _Asset:
    relative_path: str
    url: str
    size: int
    md5: str


def _cache_roots(root: Path) -> tuple[Path, ...]:
    return root / "pciseq_atlas", root / "abc_atlas", root


@contextmanager
def _lock(root: Path, name: str) -> Iterator[None]:
    directory = root / "pciseq_atlas" / ".asset_locks"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(name.encode()).hexdigest()
    with (directory / f"{digest}.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(root: Path, auto_download: bool) -> tuple[Path, dict[str, Any]]:
    relative = f"releases/{ATLAS_RELEASE}/manifest.json"
    for directory in _cache_roots(root):
        path = directory / relative
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        if hashlib.sha256(data).hexdigest() == MANIFEST_SHA256:
            return path, json.loads(data)
        logger.warning(
            "Ignoring a manifest that differs from the pinned release: %s", path
        )
    if not auto_download:
        raise FileNotFoundError(
            f"Verified Allen manifest {ATLAS_RELEASE} is missing; enable auto_download"
        )
    with _lock(root, relative):
        destination = root / "pciseq_atlas" / relative
        if destination.exists():
            data = destination.read_bytes()
            if hashlib.sha256(data).hexdigest() == MANIFEST_SHA256:
                return destination, json.loads(data)
        with urllib.request.urlopen(_BASE_URL + relative, timeout=30) as response:
            data = response.read(8 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Downloaded Allen manifest fails its pinned SHA256")
        manifest = json.loads(data)
        if manifest.get("version") != ATLAS_RELEASE:
            raise ValueError("Downloaded Allen manifest has an unexpected version")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        return destination, manifest


def _asset(value: dict[str, Any]) -> _Asset:
    relative = value.get("relative_path", "")
    if not isinstance(relative, str):
        raise ValueError("Allen manifest contains an unsafe relative path")
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or str(path) != relative
    ):
        raise ValueError("Allen manifest contains an unsafe relative path")
    url = value.get("url", "")
    if url != _BASE_URL + relative:
        raise ValueError("Allen asset URL does not match its official bucket path")
    size, digest = value.get("size"), value.get("file_hash", "")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{32}", digest) is None
    ):
        raise ValueError("Allen manifest asset lacks a valid size or MD5")
    return _Asset(relative, url, size, digest)


def _select_assets(
    manifest: dict[str, Any], source: str
) -> tuple[list[_Asset], dict[str, _Asset]]:
    expression_set, metadata_set, taxonomy_set, expected_count = _SOURCES[source]
    listing = manifest["file_listing"]
    expression = [
        _asset(value["raw"]["files"]["h5ad"])
        for _, value in sorted(listing[expression_set]["expression_matrices"].items())
    ]
    if len(expression) != expected_count:
        raise ValueError(
            f"Allen {source} manifest has an incomplete raw expression set"
        )
    keys = ["cell_metadata", "gene"]
    if source == "wmb":
        keys.append("region_of_interest_metadata")
    elif source == "whb":
        keys += [
            "donor",
            "region_of_interest_structure_map",
            "anatomical_division_structure_map",
        ]
    else:
        keys += ["donor", "library", "disease", "value_sets"]
    metadata = {
        key: _asset(listing[metadata_set]["metadata"][key]["files"]["csv"])
        for key in keys
    }
    for key in _TAXONOMY_KEYS:
        metadata[key] = _asset(listing[taxonomy_set]["metadata"][key]["files"]["csv"])
    if source == "seaad":
        key = "cell_to_cluster_membership"
        metadata[key] = _asset(listing[taxonomy_set]["metadata"][key]["files"]["csv"])
    return expression, metadata


def _structure(path: Path, suffix: str) -> dict[str, Any]:
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            if not header or len(set(header)) != len(header):
                raise ValueError("Allen metadata has an empty or duplicate CSV header")
            rows = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("Allen metadata has a malformed CSV record")
                rows += 1
        if rows == 0:
            raise ValueError("Allen metadata CSV contains no records")
        return {"format": "csv", "rows": rows, "columns": header}
    with h5py.File(path, "r") as store:
        obs, var = store["obs"], store["var"]
        n_obs = len(obs[obs.attrs.get("_index", "_index")])
        n_vars = len(var[var.attrs.get("_index", "_index")])
        counts = store["X"]
        shape = tuple(int(value) for value in counts.attrs["shape"])
        if (
            shape != (n_obs, n_vars)
            or counts.attrs.get("encoding-type") != "csr_matrix"
        ):
            raise ValueError("Allen raw H5AD has inconsistent dimensions or encoding")
        pointers, indices, values = counts["indptr"], counts["indices"], counts["data"]
        if (
            len(pointers) != n_obs + 1
            or len(indices) != len(values)
            or int(pointers[0]) != 0
            or int(pointers[-1]) != len(values)
        ):
            raise ValueError("Allen raw H5AD has inconsistent sparse counts")
        for offset in {0, max(0, len(values) - 1024)}:
            data, columns = (
                values[offset : offset + 1024],
                indices[offset : offset + 1024],
            )
            if (
                not np.isfinite(data).all()
                or (data < 0).any()
                or (data != np.floor(data)).any()
                or (columns < 0).any()
                or (columns >= n_vars).any()
            ):
                raise ValueError("Allen raw H5AD contains invalid raw counts")
        return {"format": "h5ad", "n_obs": n_obs, "n_vars": n_vars}


def _fingerprint(path: Path) -> dict[str, int]:
    info = path.stat()
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "inode": info.st_ino,
        "device": info.st_dev,
    }


def _verification_path(root: Path, path: Path) -> Path:
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return root / "pciseq_atlas" / ".asset_verification" / f"{key}.json"


def _save_verification(
    root: Path,
    path: Path,
    asset: _Asset,
    provenance: dict[str, Any],
    verify_checksums: bool,
) -> None:
    cache = _verification_path(root, path)
    try:
        _atomic_json(
            cache,
            {
                "fingerprint": _fingerprint(path),
                "official_md5": asset.md5,
                "manifest_sha256": MANIFEST_SHA256,
                "md5_verified": verify_checksums,
                "provenance": provenance,
            },
        )
    except PermissionError:
        logger.debug("Verification cache is read-only: %s", cache)


def _cached_provenance(
    previous: Any,
    path: Path,
    asset: _Asset,
    fingerprint: dict[str, int],
    verify_checksums: bool,
) -> dict[str, Any] | None:
    if (
        not isinstance(previous, dict)
        or previous.get("fingerprint") != fingerprint
        or previous.get("official_md5") != asset.md5
        or previous.get("manifest_sha256") != MANIFEST_SHA256
        or not isinstance(previous.get("md5_verified"), bool)
        or (verify_checksums and previous["md5_verified"] is not True)
    ):
        return None
    provenance = previous.get("provenance")
    if not isinstance(provenance, dict):
        return None
    sha256, md5 = provenance.get("sha256"), provenance.get("md5")
    structure = provenance.get("structure")
    if (
        provenance.get("path") != str(path)
        or provenance.get("size") != fingerprint["size"]
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(md5, str)
        or re.fullmatch(r"[0-9a-f]{32}", md5) is None
        or provenance.get("official_md5") != asset.md5
        or provenance.get("md5_verified") is not previous["md5_verified"]
        or (previous["md5_verified"] and md5 != asset.md5)
        or provenance.get("url") != asset.url
        or not isinstance(structure, dict)
        or structure.get("format") != Path(asset.relative_path).suffix.removeprefix(".")
    ):
        return None
    if structure["format"] == "csv":
        rows, columns = structure.get("rows"), structure.get("columns")
        if (
            type(rows) is not int
            or rows < 1
            or not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) for column in columns)
            or len(set(columns)) != len(columns)
        ):
            return None
    elif any(
        type(structure.get(key)) is not int or structure[key] < 0
        for key in ("n_obs", "n_vars")
    ):
        return None
    return dict(provenance)


def _verify(
    root: Path, path: Path, asset: _Asset, verify_checksums: bool
) -> dict[str, Any]:
    fingerprint = _fingerprint(path)
    if fingerprint["size"] != asset.size:
        raise ValueError(f"Allen asset has the wrong size: {path}")
    cache = _verification_path(root, path)
    try:
        previous = json.loads(cache.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        previous = {}
    # A damaged optional sidecar must trigger verification of the original asset.
    cached = _cached_provenance(previous, path, asset, fingerprint, verify_checksums)
    if cached is not None:
        return cached
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_SIZE):
            sha.update(block)
            md5.update(block)
    if verify_checksums and md5.hexdigest() != asset.md5:
        raise ValueError(f"Allen asset fails the official MD5: {path}")
    structure = _structure(path, Path(asset.relative_path).suffix)
    if _fingerprint(path) != fingerprint:
        raise ValueError(f"Allen asset changed while being verified: {path}")
    provenance = {
        "path": str(path),
        "size": fingerprint["size"],
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "official_md5": asset.md5,
        "md5_verified": verify_checksums,
        "url": asset.url,
        "structure": structure,
    }
    _save_verification(root, path, asset, provenance, verify_checksums)
    return provenance


def _download(root: Path, asset: _Asset, verify_checksums: bool) -> Path:
    destination = root / "pciseq_atlas" / asset.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset.size:
            partial.unlink()
            offset = 0
        if offset == asset.size:
            break
        if shutil.disk_usage(destination.parent).free < asset.size - offset:
            raise OSError("Insufficient space for the requested Allen asset")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(asset.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", 200)
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(
                            "Allen download returned an invalid byte range"
                        )
                elif status == 200:
                    offset = 0
                else:
                    raise ValueError(f"Unexpected Allen download HTTP status: {status}")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(_BLOCK_SIZE):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size < asset.size:
                raise OSError("Allen download ended before its declared size")
            break
        except (OSError, urllib.error.URLError, http.client.IncompleteRead):
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))
    try:
        provenance = _verify(root, partial, asset, verify_checksums)
    except (OSError, ValueError, KeyError):
        partial.unlink(missing_ok=True)
        raise
    # Preserve a damaged writable-cache file before publishing its replacement.
    if destination.exists():
        try:
            _verify(root, destination, asset, verify_checksums)
        except (OSError, ValueError, KeyError):
            pass
        else:
            partial.unlink()
            return destination
        quarantined = destination.with_name(
            destination.name + ".corrupt-" + uuid.uuid4().hex
        )
        os.replace(destination, quarantined)
    os.replace(partial, destination)
    provenance["path"] = str(destination)
    _save_verification(root, destination, asset, provenance, verify_checksums)
    return destination


def _ensure_file(
    root: Path, asset: _Asset, auto_download: bool, verify_checksums: bool
) -> tuple[Path, dict[str, Any]]:
    errors: list[str] = []
    for directory in _cache_roots(root):
        candidate = directory / asset.relative_path
        if not candidate.is_file():
            continue
        try:
            return candidate, _verify(root, candidate, asset, verify_checksums)
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
            logger.warning("Ignoring an unusable Allen cache asset: %s", error)
    if not auto_download:
        message = f"Allen asset missing or corrupt: {asset.relative_path}"
        if errors:
            raise ValueError(message + "; " + "; ".join(errors))
        raise FileNotFoundError(message + "; enable auto_download")
    with _lock(root, asset.relative_path):
        # Another process may have completed this file while this one waited.
        destination = root / "pciseq_atlas" / asset.relative_path
        if destination.is_file():
            try:
                return destination, _verify(root, destination, asset, verify_checksums)
            except (OSError, ValueError, KeyError):
                logger.warning(
                    "Replacing damaged writable Allen cache: %s", destination
                )
        destination = _download(root, asset, verify_checksums)
        return destination, _verify(root, destination, asset, verify_checksums)


def ensure_atlas_assets(
    root: Path,
    source: str,
    *,
    auto_download: bool = True,
    verify_checksums: bool = True,
) -> AtlasAssets:
    """Locate or download verified raw matrices and metadata for one atlas.

    Args:
        root: Parent reference directory containing atlas caches.
        source: ``wmb``, ``whb``, or ``seaad``.
        auto_download: Fetch missing or corrupt assets from Allen's public bucket.
        verify_checksums: Compare full-file MD5 with the pinned official manifest.

    Returns:
        Exact source paths and content fingerprints for reference construction.

    Raises:
        FileNotFoundError: Required data is missing with downloads disabled.
        ValueError: A source, manifest, checksum, or structural contract is invalid.
    """
    root, source = Path(root), source.lower()
    if source not in _SOURCES:
        raise ValueError("Allen reference source must be wmb, whb, or seaad")
    manifest_path, manifest = _load_manifest(root, auto_download)
    expression, metadata = _select_assets(manifest, source)
    files: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for asset in [*metadata.values(), *expression]:
        if asset.relative_path in paths:
            continue
        path, provenance = _ensure_file(root, asset, auto_download, verify_checksums)
        paths[asset.relative_path] = path
        files[asset.relative_path] = provenance
    metadata_paths = {
        key: paths[asset.relative_path] for key, asset in metadata.items()
    }
    for key in ("donor", "library", "disease"):
        if key in metadata_paths:
            metadata_paths[f"{key}_metadata"] = metadata_paths[key]
    source_fingerprint = hashlib.sha256(
        json.dumps(
            {key: value["sha256"] for key, value in sorted(files.items())},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return AtlasAssets(
        source=source,
        root=root,
        release=ATLAS_RELEASE,
        manifest_path=manifest_path,
        expression_paths=tuple(paths[asset.relative_path] for asset in expression),
        metadata_paths=metadata_paths,
        provenance={
            "source": source,
            "release": ATLAS_RELEASE,
            "manifest_url": _BASE_URL + f"releases/{ATLAS_RELEASE}/manifest.json",
            "manifest_sha256": MANIFEST_SHA256,
            "source_fingerprint": source_fingerprint,
            "files": files,
        },
    )
