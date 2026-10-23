"""Small source-backed fixtures for atlas cache integrity and acquisition."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from merxen.analysis import allen_reference_assets as assets


class _Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, offset: int = 0) -> None:
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {offset}-{offset + len(data) - 1}/*"}


@pytest.fixture
def atlas_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    """Populate all three complete release contracts with tiny valid files."""
    counts_path = tmp_path / "tiny.h5ad"
    with h5py.File(counts_path, "w") as store:
        for name, labels in [
            ("obs", ["cell_a", "cell_b"]),
            ("var", ["gene_a", "gene_b"]),
        ]:
            group = store.create_group(name)
            group.attrs["_index"] = "_index"
            group.create_dataset("_index", data=np.asarray(labels, dtype="S"))
        matrix = store.create_group("X")
        matrix.attrs["encoding-type"] = "csr_matrix"
        matrix.attrs["shape"] = [2, 2]
        matrix.create_dataset("indptr", data=[0, 2, 3])
        matrix.create_dataset("indices", data=[0, 1, 0])
        matrix.create_dataset("data", data=[1, 2, 3])
    counts = counts_path.read_bytes()
    root = tmp_path / "references"
    payloads: dict[str, bytes] = {}
    listing: dict[str, Any] = {}

    def entry(relative: str, data: bytes) -> dict[str, Any]:
        payloads[assets._BASE_URL + relative] = data
        path = root / "abc_atlas" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {
            "relative_path": relative,
            "url": assets._BASE_URL + relative,
            "size": len(data),
            "file_hash": hashlib.md5(data).hexdigest(),
        }

    taxonomy_keys = [
        "cluster",
        "cluster_annotation_term",
        "cluster_annotation_term_set",
        "cluster_to_cluster_annotation_membership",
    ]
    for source, (expression, metadata, taxonomy, count) in assets._SOURCES.items():
        metadata_keys = ["cell_metadata", "gene"]
        if source == "wmb":
            metadata_keys.append("region_of_interest_metadata")
        elif source == "whb":
            metadata_keys += [
                "donor",
                "region_of_interest_structure_map",
                "anatomical_division_structure_map",
            ]
        else:
            metadata_keys += ["donor", "library", "disease", "value_sets"]
        for dataset, keys in [
            (metadata, metadata_keys),
            (
                taxonomy,
                taxonomy_keys
                + (["cell_to_cluster_membership"] if source == "seaad" else []),
            ),
        ]:
            listing.setdefault(dataset, {})["metadata"] = {
                key: {
                    "files": {
                        "csv": entry(
                            f"metadata/{dataset}/20260711/{key}.csv",
                            b"label,value\na,1\n",
                        )
                    }
                }
                for key in keys
            }
        listing.setdefault(expression, {})["expression_matrices"] = {
            f"matrix_{index}": {
                "raw": {
                    "files": {
                        "h5ad": entry(
                            f"expression_matrices/{expression}/20260711/matrix_{index}-raw.h5ad",
                            counts,
                        )
                    }
                }
            }
            for index in range(count)
        }
    manifest = json.dumps(
        {"version": assets.ATLAS_RELEASE, "file_listing": listing}
    ).encode()
    manifest_path = (
        root / "abc_atlas" / "releases" / assets.ATLAS_RELEASE / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest)
    monkeypatch.setattr(assets, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest())
    return root, payloads


def _raw_entry(root: Path) -> dict[str, Any]:
    path = root / "abc_atlas" / "releases" / assets.ATLAS_RELEASE / "manifest.json"
    manifest = json.loads(path.read_text())
    return manifest["file_listing"]["WHB-10Xv3"]["expression_matrices"]["matrix_0"][
        "raw"
    ]["files"]["h5ad"]


def _fake_downloads(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    *,
    ignore_range: bool = False,
) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []

    def urlopen(request: Any, timeout: int = 30) -> _Response:
        url = request if isinstance(request, str) else request.full_url
        requested_range = (
            None if isinstance(request, str) else request.get_header("Range")
        )
        calls.append((url, requested_range))
        data = payloads[url]
        if requested_range and not ignore_range:
            offset = int(requested_range.removeprefix("bytes=").removesuffix("-"))
            return _Response(data[offset:], 206, offset)
        return _Response(data)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize(("source", "count"), [("wmb", 13), ("whb", 2), ("seaad", 10)])
def test_ensure_uses_complete_raw_source_and_records_content_hashes(
    atlas_cache: tuple[Path, dict[str, bytes]], source: str, count: int
) -> None:
    """All shards and join metadata belong to the exact pinned release."""
    root, _ = atlas_cache
    result = assets.ensure_atlas_assets(root, source, auto_download=False)
    assert isinstance(result, assets.AtlasAssets)
    assert result.root == root
    assert result.release == "20260711"
    assert len(result.expression_paths) == count
    assert "cell_metadata" in result.metadata_paths
    assert "cluster_annotation_term" in result.metadata_paths
    assert result.provenance["manifest_sha256"] == assets.MANIFEST_SHA256
    assert all(record["md5_verified"] for record in result.provenance["files"].values())
    assert all(
        len(record["sha256"]) == 64 for record in result.provenance["files"].values()
    )
    if source == "seaad":
        assert "cell_to_cluster_membership" in result.metadata_paths
        assert (
            result.metadata_paths["library"]
            == result.metadata_paths["library_metadata"]
        )


def test_corrupt_preferred_cache_falls_back_to_valid_existing_file(
    atlas_cache: tuple[Path, dict[str, bytes]],
) -> None:
    """A damaged new cache must not hide a healthy read-only legacy cache."""
    root, _ = atlas_cache
    entry = _raw_entry(root)
    bad = root / "pciseq_atlas" / entry["relative_path"]
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"x" * entry["size"])
    result = assets.ensure_atlas_assets(root, "whb", auto_download=False)
    assert root / "abc_atlas" / entry["relative_path"] in result.expression_paths
    assert bad.read_bytes() == b"x" * entry["size"]


def test_corrupt_cache_is_replaced_from_official_data_and_preserved(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a checksum-verified replacement can take the published filename."""
    root, payloads = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    destination = root / "pciseq_atlas" / entry["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    corrupt = b"x" * entry["size"]
    destination.write_bytes(corrupt)
    calls = _fake_downloads(monkeypatch, payloads)
    result = assets.ensure_atlas_assets(root, "whb")
    assert destination in result.expression_paths
    assert destination.read_bytes() == payloads[entry["url"]]
    assert (
        list(destination.parent.glob(destination.name + ".corrupt-*"))[0].read_bytes()
        == corrupt
    )
    assert calls == [(entry["url"], None)]


@pytest.mark.parametrize("ignore_range", [False, True])
def test_partial_download_resumes_or_restarts_when_range_is_ignored(
    atlas_cache: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    ignore_range: bool,
) -> None:
    """Interrupted acquisition is safe with both range-aware and plain servers."""
    root, payloads = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    destination = root / "pciseq_atlas" / entry["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(payloads[entry["url"]][:64])
    calls = _fake_downloads(monkeypatch, payloads, ignore_range=ignore_range)
    assets.ensure_atlas_assets(root, "whb")
    assert calls == [(entry["url"], "bytes=64-")]
    assert destination.read_bytes() == payloads[entry["url"]]
    assert not partial.exists()


def test_hash_mismatch_never_publishes_download(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HTTP success cannot turn corrupt bytes into a trusted cache asset."""
    root, payloads = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    payloads[entry["url"]] = b"x" * entry["size"]
    _fake_downloads(monkeypatch, payloads)
    with pytest.raises(ValueError, match="MD5"):
        assets.ensure_atlas_assets(root, "whb")
    destination = root / "pciseq_atlas" / entry["relative_path"]
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_short_download_eof_resumes_the_remaining_bytes(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An early EOF without a socket exception still keeps resumable bytes."""
    root, payloads = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    calls: list[str | None] = []
    payload = payloads[entry["url"]]

    def urlopen(request: Any, timeout: int = 30) -> _Response:
        requested_range = request.get_header("Range")
        calls.append(requested_range)
        if len(calls) == 1:
            return _Response(payload[:64])
        return _Response(payload[64:], 206, 64)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(assets.time, "sleep", lambda seconds: None)
    result = assets.ensure_atlas_assets(root, "whb")
    destination = root / "pciseq_atlas" / entry["relative_path"]
    assert calls == [None, "bytes=64-"]
    assert destination in result.expression_paths
    assert destination.read_bytes() == payload


def test_repeated_short_eof_keeps_partial_for_a_future_run(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausting bounded retries must preserve valid incomplete bytes."""
    root, payloads = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    calls: list[str | None] = []
    payload = payloads[entry["url"]]

    def urlopen(request: Any, timeout: int = 30) -> _Response:
        requested_range = request.get_header("Range")
        calls.append(requested_range)
        if len(calls) == 1:
            return _Response(payload[:64])
        return _Response(b"", 206, 64)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(assets.time, "sleep", lambda seconds: None)
    with pytest.raises(OSError, match="declared size"):
        assets.ensure_atlas_assets(root, "whb")
    destination = root / "pciseq_atlas" / entry["relative_path"]
    assert not destination.exists()
    assert (
        destination.with_name(destination.name + ".partial").read_bytes()
        == payload[:64]
    )
    assert calls == [None, "bytes=64-", "bytes=64-"]


def test_read_only_cache_does_not_require_writable_verification_sidecars(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing valid data remains usable when caching verification is denied."""
    root, _ = atlas_cache

    def deny_sidecar(path: Path, value: dict[str, Any]) -> None:
        raise PermissionError("read-only verification cache")

    monkeypatch.setattr(assets, "_atomic_json", deny_sidecar)
    result = assets.ensure_atlas_assets(root, "whb", auto_download=False)
    assert len(result.expression_paths) == 2
    assert not (root / "pciseq_atlas" / ".asset_locks").exists()


def test_unchanged_files_reuse_verified_fingerprints(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated runs reuse size/mtime/inode fingerprints and recorded hashes."""
    root, _ = atlas_cache
    first = assets.ensure_atlas_assets(root, "whb", auto_download=False)

    def reject_second_parse(path: Path, suffix: str) -> dict[str, Any]:
        pytest.fail("unchanged verified asset was parsed again")

    monkeypatch.setattr(assets, "_structure", reject_second_parse)
    second = assets.ensure_atlas_assets(root, "whb", auto_download=False)
    assert first.provenance == second.provenance


@pytest.mark.parametrize(
    "damage",
    [
        "non_utf8",
        "invalid_json",
        "list",
        "null",
        "missing_provenance",
        "non_mapping_provenance",
        "missing_hash",
        "non_boolean_verified",
        "inconsistent_md5",
        "incomplete_structure",
        "invalid_csv_columns",
    ],
)
def test_damaged_verification_sidecar_rechecks_healthy_asset_without_downloads(
    atlas_cache: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """Optional cached verification cannot hide or misdescribe healthy source data."""
    root, _ = atlas_cache
    first = assets.ensure_atlas_assets(root, "whb", auto_download=False)
    path = (
        first.metadata_paths["gene"]
        if damage == "invalid_csv_columns"
        else first.expression_paths[0]
    )
    sidecar = assets._verification_path(root, path)
    original = json.loads(sidecar.read_text())
    damaged: Any = json.loads(sidecar.read_text())
    if damage == "non_utf8":
        sidecar.write_bytes(b"\xff")
    elif damage == "invalid_json":
        sidecar.write_text("{")
    else:
        if damage == "list":
            damaged = []
        elif damage == "null":
            damaged = None
        elif damage == "missing_provenance":
            damaged.pop("provenance")
        elif damage == "non_mapping_provenance":
            damaged["provenance"] = []
        elif damage == "missing_hash":
            damaged["provenance"].pop("sha256")
        elif damage == "non_boolean_verified":
            damaged["md5_verified"] = damaged["provenance"]["md5_verified"] = "true"
        elif damage == "inconsistent_md5":
            damaged["provenance"]["md5"] = "0" * 32
        elif damage == "incomplete_structure":
            damaged["provenance"]["structure"] = {"format": "h5ad"}
        elif damage == "invalid_csv_columns":
            damaged["provenance"]["structure"]["columns"] = [None]
        sidecar.write_text(json.dumps(damaged))
    parsed: list[Path] = []
    original_structure = assets._structure

    def record_parse(candidate: Path, suffix: str) -> dict[str, Any]:
        parsed.append(candidate)
        return original_structure(candidate, suffix)

    monkeypatch.setattr(assets, "_structure", record_parse)
    second = assets.ensure_atlas_assets(root, "whb", auto_download=False)
    assert parsed == [path]
    assert second.provenance == first.provenance
    assert json.loads(sidecar.read_text()) == original


def test_missing_file_with_downloads_disabled_names_the_asset(
    atlas_cache: tuple[Path, dict[str, bytes]],
) -> None:
    """An incomplete cache gives an actionable error without network access."""
    root, _ = atlas_cache
    entry = _raw_entry(root)
    (root / "abc_atlas" / entry["relative_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="matrix_0-raw.h5ad"):
        assets.ensure_atlas_assets(root, "whb", auto_download=False)


def test_manifest_path_cannot_escape_cache(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an explicitly supplied manifest cannot write outside its cache."""
    root, _ = atlas_cache
    path = root / "abc_atlas" / "releases" / assets.ATLAS_RELEASE / "manifest.json"
    manifest = json.loads(path.read_text())
    value = manifest["file_listing"]["WHB-10Xv3"]["expression_matrices"]["matrix_0"][
        "raw"
    ]["files"]["h5ad"]
    value["relative_path"] = "../escaped.h5ad"
    data = json.dumps(manifest).encode()
    path.write_bytes(data)
    monkeypatch.setattr(assets, "MANIFEST_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="unsafe relative path"):
        assets.ensure_atlas_assets(root, "whb", auto_download=False)


def test_officially_hashed_but_malformed_csv_is_rejected(
    atlas_cache: tuple[Path, dict[str, bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural parsing checks all rows rather than accepting a valid header."""
    root, _ = atlas_cache
    manifest_path = (
        root / "abc_atlas" / "releases" / assets.ATLAS_RELEASE / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["file_listing"]["WHB-10Xv3"]["metadata"]["gene"]["files"]["csv"]
    malformed = b"gene,value\na,1\nb\n"
    (root / "abc_atlas" / entry["relative_path"]).write_bytes(malformed)
    entry.update(size=len(malformed), file_hash=hashlib.md5(malformed).hexdigest())
    data = json.dumps(manifest).encode()
    manifest_path.write_bytes(data)
    monkeypatch.setattr(assets, "MANIFEST_SHA256", hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="malformed CSV"):
        assets.ensure_atlas_assets(root, "whb", auto_download=False)


def test_unknown_source_is_rejected_before_manifest_access(tmp_path: Path) -> None:
    """Only the three explicitly supported reference sources may be requested."""
    with pytest.raises(ValueError, match="wmb, whb, or seaad"):
        assets.ensure_atlas_assets(tmp_path, "unknown")
