"""Tests for the KI hippocampus Xenium download, prepare and samplesheet shims.

Everything runs on tiny synthetic bundles in ``tmp_path``; no network.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "public_data"
    / "xenium_hippocampus"
)
TOP = "output-XETG00000__0000001__D1_2__20000101__000000"
BUNDLE = f"{TOP}.tar.gz"


def _load(name: str) -> ModuleType:
    # The same keys the scripts use to load each other, so one instance each.
    key = f"_ki_hippocampus_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch() -> ModuleType:
    return _load("fetch_ki_hippocampus")


@pytest.fixture(scope="module")
def prepare() -> ModuleType:
    return _load("prepare_ki_hippocampus")


@pytest.fixture(scope="module")
def samplesheet() -> ModuleType:
    return _load("make_samplesheet_ki_hippocampus")


def _record(record_id: str, entries: list[tuple[str, int, str]]) -> dict[str, Any]:
    return {
        "id": int(record_id),
        "files": [
            {
                "key": name,
                "size": size,
                "checksum": f"md5:{md5}",
                "links": {
                    "self": f"https://zenodo.org/api/records/{record_id}/files/{name}/content"
                },
            }
            for name, size, md5 in entries
        ],
    }


CANNED = {
    "1001": _record(
        "1001",
        [
            ("output-XETG00000__0000002__D1_3__20000101__000000.tar.gz", 30, "c" * 32),
            ("output-XETG00000__0000001__D1_1__20000101__000000.tar.gz", 10, "a" * 32),
            ("README.txt", 5, "d" * 32),
        ],
    ),
    "1002": _record(
        "1002",
        [("output-XETG00001__0000003__D1_10__20000101__000001.tar.gz", 20, "b" * 32)],
    ),
}


# --------------------------------------------------------------------------
# fetch: listing


def test_list_remote_files_parses_canned_records_in_section_order(
    fetch: ModuleType,
) -> None:
    files = fetch.list_remote_files(CANNED)

    assert [item.section for item in files] == ["KI_HC_S1", "KI_HC_S3", "KI_HC_S10"]
    assert files[0].record_id == "1001"
    assert files[0].md5 == "a" * 32
    assert files[0].url.endswith("/content")
    assert sum(item.size_bytes for item in files) == 60


def test_fetch_records_uses_the_injected_getter(fetch: ModuleType) -> None:
    calls: list[str] = []

    def fake_get(url: str) -> dict[str, Any]:
        calls.append(url)
        return CANNED[url.rsplit("/", 1)[-1]]

    records = fetch.fetch_records(("1001", "1002"), get_json=fake_get)

    assert calls == [
        "https://zenodo.org/api/records/1001",
        "https://zenodo.org/api/records/1002",
    ]
    assert set(records) == {"1001", "1002"}


def test_parse_record_files_rejects_a_bundle_without_md5(fetch: ModuleType) -> None:
    record = _record("7", [(BUNDLE, 1, "e" * 32)])
    record["files"][0]["checksum"] = "sha256:abc"

    with pytest.raises(fetch.FetchError, match="no md5"):
        fetch.parse_record_files(record)


def test_list_remote_files_rejects_two_bundles_for_one_section(
    fetch: ModuleType,
) -> None:
    records = {
        "1": _record("1", [(BUNDLE, 1, "a" * 32)]),
        "2": _record("2", [(BUNDLE.replace("XETG00000", "XETG00009"), 1, "b" * 32)]),
    }

    with pytest.raises(fetch.FetchError, match="both map to KI_HC_S2"):
        fetch.list_remote_files(records)


@pytest.mark.parametrize(
    "name", ["bundle.tar.gz", "output-A__B__nosection__20000101__000000.tar.gz"]
)
def test_section_label_rejects_unexpected_names(fetch: ModuleType, name: str) -> None:
    with pytest.raises(fetch.FetchError):
        fetch.section_label_from_file_name(name)


def test_select_sections_accepts_short_forms_and_rejects_unknown(
    fetch: ModuleType,
) -> None:
    files = fetch.list_remote_files(CANNED)

    chosen = fetch.select_sections(files, ["S3", "1"])
    assert [item.section for item in chosen] == ["KI_HC_S1", "KI_HC_S3"]
    with pytest.raises(fetch.FetchError, match="not offered"):
        fetch.select_sections(files, ["KI_HC_S2"])


def test_manifest_round_trip(fetch: ModuleType, tmp_path: Path) -> None:
    files = fetch.list_remote_files(CANNED)
    path = tmp_path / "manifest.tsv"

    fetch.write_manifest(path, files)

    assert fetch.read_manifest(path) == files


# --------------------------------------------------------------------------
# fetch: verification, resume and the space guard


def _remote(fetch: ModuleType, payload: bytes, name: str = BUNDLE) -> Any:
    return fetch.RemoteFile(
        section="KI_HC_S2",
        record_id="1",
        file_name=name,
        size_bytes=len(payload),
        md5=hashlib.md5(payload).hexdigest(),
        url="https://example.invalid/bundle",
    )


def test_verify_file_writes_sidecar_and_rejects_bad_checksum(
    fetch: ModuleType, tmp_path: Path
) -> None:
    payload = b"bundle bytes"
    item = _remote(fetch, payload)
    path = tmp_path / item.file_name
    path.write_bytes(payload)

    fetch.verify_file(path, item)
    assert fetch.is_verified(path, item)

    path.write_bytes(b"bundle bytez")
    fetch.sidecar_path(path).unlink()
    with pytest.raises(fetch.FetchError, match="md5"):
        fetch.verify_file(path, item)
    path.write_bytes(b"short")
    with pytest.raises(fetch.FetchError, match="bytes, expected"):
        fetch.verify_file(path, item)


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


def test_download_file_resumes_a_partial_file_with_a_range_request(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = bytes(range(256)) * 40
    item = _remote(fetch, payload)
    (tmp_path / f"{item.file_name}.part").write_bytes(payload[:1000])
    starts: list[int] = []

    def fake_open(url: str, start: int, timeout: float) -> _FakeResponse:
        starts.append(start)
        return _FakeResponse(payload[start:], 206 if start else 200)

    monkeypatch.setattr(fetch, "_open_range", fake_open)

    path = fetch.download_file(item, tmp_path, retries=2, retry_wait_seconds=0)

    assert starts == [1000]
    assert path.read_bytes() == payload
    assert fetch.is_verified(path, item)
    # A second call is a no-op: the file is complete and verified.
    monkeypatch.setattr(fetch, "_open_range", lambda *a: pytest.fail("refetched"))
    fetch.download_file(item, tmp_path, retries=1, retry_wait_seconds=0)


def test_download_file_restarts_when_the_server_ignores_the_range(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"0123456789" * 50
    item = _remote(fetch, payload)
    (tmp_path / f"{item.file_name}.part").write_bytes(b"garbage!!")
    monkeypatch.setattr(
        fetch, "_open_range", lambda url, start, timeout: _FakeResponse(payload, 200)
    )

    path = fetch.download_file(item, tmp_path, retries=1, retry_wait_seconds=0)

    assert path.read_bytes() == payload


class _RangeResponse(_FakeResponse):
    def __init__(self, data: bytes, status: int, content_range: str) -> None:
        super().__init__(data, status)
        self.headers = {"Content-Range": content_range}


def test_download_file_restarts_when_the_range_starts_elsewhere(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefghij" * 30
    item = _remote(fetch, payload)
    (tmp_path / f"{item.file_name}.part").write_bytes(payload[:100])
    starts: list[int] = []

    def fake_open(url: str, start: int, timeout: float) -> _FakeResponse:
        starts.append(start)
        if start:
            # A 206 for byte 50 onwards, not the 100 asked for.
            total = len(payload)
            return _RangeResponse(payload[50:], 206, f"bytes 50-{total - 1}/{total}")
        return _FakeResponse(payload, 200)

    monkeypatch.setattr(fetch, "_open_range", fake_open)

    path = fetch.download_file(item, tmp_path, retries=3, retry_wait_seconds=0)

    assert starts == [100, 0]
    assert path.read_bytes() == payload
    assert fetch.is_verified(path, item)


def test_main_refuses_download_without_space(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "fetch_records", lambda: CANNED)
    monkeypatch.setattr(
        fetch.shutil, "disk_usage", lambda path: type("U", (), {"free": 50})()
    )
    monkeypatch.setattr(
        fetch, "download_file", lambda *a, **k: pytest.fail("downloaded")
    )

    code = fetch.main(
        ["--public-root", str(tmp_path), "--download", "--margin-gb", "0"]
    )

    assert code == 3
    meta = fetch.metadata_dir(tmp_path)
    assert (meta / "manifest.tsv").is_file()
    assert not list(fetch.raw_dir(tmp_path).glob("*.tar.gz*"))


def test_main_dry_run_writes_nothing(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "fetch_records", lambda: CANNED)

    assert fetch.main(["--public-root", str(tmp_path / "root")]) == 0
    assert not (tmp_path / "root").exists()


# --------------------------------------------------------------------------
# prepare: a miniature XOA 1.4 bundle


PIXEL_SIZE = 0.2125
CELL_IDS = ["aaaaaaab-1", "aaaaaaac-1"]


def _boundaries(bounds: list[tuple[int, int]]) -> pd.DataFrame:
    """Square polygons (closed rings, micrometres) for the two cells."""
    rows = []
    for cell_id, (low, high) in zip(CELL_IDS, bounds, strict=True):
        for x, y in [(low, low), (high, low), (high, high), (low, high), (low, low)]:
            rows.append((cell_id, x * PIXEL_SIZE, y * PIXEL_SIZE))
    return pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"])


def _write_bundle_folder(
    folder: Path, *, region: str = "D1_2", channels: int = 1
) -> None:
    """Write a miniature but readable XOA 1.4 output folder.

    Every file spatialdata-io reads has the vendor layout: a v3 10x h5, a zarr
    v2 ``cells.zarr.zip`` with ``seg_mask_value`` matching the masks, sorted
    cell ids shared by the table and both boundary files, micrometre
    coordinates and an OME-TIFF focus image.
    """
    import h5py
    import tifffile
    import zarr

    folder.mkdir(parents=True)
    experiment = {
        "major_version": 2,
        "minor_version": 0,
        "region_name": region,
        "pixel_size": PIXEL_SIZE,
        "analysis_sw_version": "xenium-1.4.0.6",
        "panel_num_targets_predesigned": 266,
        "panel_num_targets_custom": 50,
        "num_cells": 2,
    }
    (folder / "experiment.xenium").write_text(json.dumps(experiment))
    cell_mask = np.zeros((64, 64), np.uint32)
    nucleus_mask = np.zeros_like(cell_mask)
    cell_mask[4:20, 4:20] = 1
    cell_mask[30:50, 30:50] = 2
    nucleus_mask[8:16, 8:16] = 1
    nucleus_mask[35:45, 35:45] = 2
    store = zarr.storage.ZipStore(folder / "cells.zarr.zip", mode="w")
    root = zarr.group(store=store, zarr_format=2)
    masks = root.create_group("masks")
    masks.create_array("0", data=nucleus_mask)
    masks.create_array("1", data=cell_mask)
    # Prefix 1 and 2 decode to aaaaaaab / aaaaaaac; dataset suffix 1.
    root.create_array("cell_id", data=np.array([[1, 1], [2, 1]], dtype=np.uint32))
    root.create_array("seg_mask_value", data=np.array([1, 2], dtype=np.uint32))
    store.close()
    centroids = np.array([12.0, 40.0]) * PIXEL_SIZE
    pd.DataFrame(
        {
            "cell_id": CELL_IDS,
            "x_centroid": centroids,
            "y_centroid": centroids,
            "transcript_counts": [2, 1],
            "control_probe_counts": [0, 0],
            "control_codeword_counts": [0, 0],
            "total_counts": [2, 1],
            "cell_area": [11.6, 18.1],
            "nucleus_area": [2.9, 4.5],
        }
    ).to_parquet(folder / "cells.parquet")
    _boundaries([(4, 20), (30, 50)]).to_parquet(folder / "cell_boundaries.parquet")
    _boundaries([(8, 16), (35, 45)]).to_parquet(folder / "nucleus_boundaries.parquet")
    pd.DataFrame(
        {
            "transcript_id": np.arange(1, 6, dtype=np.uint64) + 2**48,
            "cell_id": [
                CELL_IDS[0],
                CELL_IDS[0],
                CELL_IDS[1],
                "UNASSIGNED",
                "UNASSIGNED",
            ],
            "overlaps_nucleus": np.array([1, 0, 1, 0, 0], dtype=np.uint8),
            "feature_name": [
                b"GFAP",
                b"CALB2",
                b"GFAP",
                b"NegControlProbe_00042",
                b"BLANK_0006",
            ],
            "x_location": np.array([12, 14, 40, 60, 61], dtype=float) * PIXEL_SIZE,
            "y_location": np.array([12, 14, 40, 60, 61], dtype=float) * PIXEL_SIZE,
            "z_location": [15.0, 16.0, 17.0, 18.0, 19.0],
            "qv": [40.0, 15.0, 30.0, 40.0, 40.0],
            "fov_name": ["A1"] * 5,
            "nucleus_distance": [0.0, 1.0, 0.0, 5.0, 6.0],
        }
    ).to_parquet(folder / "transcripts.parquet")
    with h5py.File(folder / "cell_feature_matrix.h5", "w") as handle:
        matrix = handle.create_group("matrix")
        matrix["barcodes"] = np.array(CELL_IDS, dtype="S")
        # features x cells (CSC): GFAP, CALB2, a control probe.
        matrix["data"] = np.array([1, 1, 1], dtype=np.int32)
        matrix["indices"] = np.array([0, 1, 0], dtype=np.int64)
        matrix["indptr"] = np.array([0, 2, 3], dtype=np.int64)
        matrix["shape"] = np.array([3, 2], dtype=np.int32)
        features = matrix.create_group("features")
        features["id"] = np.array([b"G1", b"G2", b"NegControlProbe_00042"])
        features["name"] = np.array([b"GFAP", b"CALB2", b"NegControlProbe_00042"])
        features["feature_type"] = np.array(
            [b"Gene Expression", b"Gene Expression", b"Negative Control Probe"]
        )
        features["genome"] = np.array([b"Unknown"] * 3)
    image = (np.arange(64 * 64, dtype=np.uint16).reshape(1, 64, 64)) % 4096
    tifffile.imwrite(
        folder / "morphology_focus.ome.tif",
        np.repeat(image, channels, axis=0),
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["DAPI", "Other"][:channels]},
            "PhysicalSizeX": PIXEL_SIZE,
            "PhysicalSizeY": PIXEL_SIZE,
        },
    )
    (folder / "morphology.ome.tif").write_bytes(b"z-stack placeholder")
    (folder / "morphology_mip.ome.tif").write_bytes(b"mip placeholder")
    (folder / "gene_panel.json").write_text("{}")
    (folder / "metrics_summary.csv").write_text("region_name\nD1_2\n")
    (folder / "analysis").mkdir()
    (folder / "analysis" / "clusters.csv").write_text("x\n")


def _make_tarball(source_folder: Path, tarball: Path, top: str = TOP) -> None:
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(source_folder, arcname=top)


def _plenty_of_space(fetch: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the disk-space guard pass whatever the test machine has free."""
    monkeypatch.setattr(
        fetch.shutil, "disk_usage", lambda path: type("U", (), {"free": 10**15})()
    )


@pytest.fixture()
def staged_root(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Any]:
    """A public root with one downloaded bundle (section 2) and a manifest."""
    _plenty_of_space(fetch, monkeypatch)
    root = tmp_path / "public"
    _write_bundle_folder(tmp_path / "bundle_src")
    raw = fetch.raw_dir(root)
    raw.mkdir(parents=True)
    _make_tarball(tmp_path / "bundle_src", raw / BUNDLE)
    item = _remote(fetch, (raw / BUNDLE).read_bytes())
    fetch.write_manifest(fetch.metadata_dir(root) / fetch.MANIFEST_NAME, [item])
    return root, item


def test_prepare_extracts_needed_members_and_links_them(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, item = staged_root

    assert (
        prepare.main(["--public-root", str(root), "prepare", "--sections", "S2"]) == 0
    )

    extracted = fetch.extracted_dir(root, "KI_HC_S2")
    assert not (extracted / "morphology.ome.tif").exists()
    assert not (extracted / "analysis").exists()
    target = prepare.xenium_dir(root, "KI_HC_S2")
    assert target.name == "PUBXHPC_KI_HC_S2"
    for name in prepare.REQUIRED_MEMBERS + prepare.KEPT_MEMBERS:
        assert (target / name).is_file()
        assert (target / name).stat().st_ino == (extracted / name).stat().st_ino
    provenance = json.loads((target / prepare.PROVENANCE_NAME).read_text())
    assert provenance["md5_verified"] is True
    assert provenance["validation"]["morphology_channels"] == ["DAPI"]
    assert provenance["validation"]["parsed_version"] == [1, 4, 0]
    assert provenance["validation"]["reader_channel_name"] == "0"
    assert fetch.is_extracted(root, item)
    # The raw download is untouched and a rerun is a no-op.
    assert (fetch.raw_dir(root) / BUNDLE).is_file()
    assert (
        prepare.main(["--public-root", str(root), "prepare", "--sections", "S2"]) == 0
    )


def test_prepare_dry_run_changes_nothing(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, _ = staged_root

    code = prepare.main(
        ["--public-root", str(root), "prepare", "--dry-run", "--delete-tarball"]
    )

    assert code == 0
    assert not (fetch.dataset_root(root) / "extracted").exists()
    assert not (fetch.dataset_root(root) / "xenium").exists()
    assert (fetch.raw_dir(root) / BUNDLE).is_file()


def test_prepare_with_extras_and_delete_tarball(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, _ = staged_root

    code = prepare.main(
        [
            "--public-root",
            str(root),
            "prepare",
            "--with-zstack",
            "--with-analysis",
            "--delete-tarball",
        ]
    )

    assert code == 0
    target = prepare.xenium_dir(root, "KI_HC_S2")
    assert (target / "morphology.ome.tif").is_file()
    assert (target / "analysis" / "clusters.csv").is_file()
    assert not (fetch.raw_dir(root) / BUNDLE).exists()


def test_prepare_stream_reads_the_bundle_from_its_url(
    prepare: ModuleType,
    fetch: ModuleType,
    staged_root: tuple[Path, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, item = staged_root
    moved = tmp_path / "elsewhere.tar.gz"
    (fetch.raw_dir(root) / BUNDLE).rename(moved)
    urls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> Any:
        urls.append(request.full_url)
        return moved.open("rb")

    monkeypatch.setattr(prepare.urllib.request, "urlopen", fake_urlopen)

    assert prepare.main(["--public-root", str(root), "prepare", "--stream"]) == 0
    assert urls == [item.url]
    assert fetch.is_extracted(root, item)
    assert (prepare.xenium_dir(root, "KI_HC_S2") / "transcripts.parquet").is_file()


def test_prepare_rejects_a_checksum_mismatch(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, item = staged_root
    wrong = fetch.RemoteFile(**{**item.__dict__, "md5": "0" * 32})
    fetch.write_manifest(fetch.metadata_dir(root) / fetch.MANIFEST_NAME, [wrong])

    assert prepare.main(["--public-root", str(root), "prepare"]) == 2
    assert not fetch.extracted_dir(root, "KI_HC_S2").exists()
    assert not prepare.xenium_dir(root, "KI_HC_S2").exists()


def test_prepare_rejects_a_bundle_of_another_section(
    prepare: ModuleType, fetch: ModuleType, tmp_path: Path
) -> None:
    _write_bundle_folder(tmp_path / "src", region="D1_5")

    with pytest.raises(prepare.PrepareError, match="another section"):
        prepare.validate_section(tmp_path / "src", "KI_HC_S2")


def test_validate_rejects_a_multichannel_morphology(
    prepare: ModuleType, tmp_path: Path
) -> None:
    _write_bundle_folder(tmp_path / "src", channels=2)

    with pytest.raises(prepare.PrepareError, match="2 channels"):
        prepare.validate_section(tmp_path / "src", "KI_HC_S2")


def test_extract_members_refuses_path_traversal(
    prepare: ModuleType, tmp_path: Path
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"{TOP}/../experiment.xenium")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    buffer.seek(0)

    with pytest.raises(prepare.PrepareError, match="unsafe"):
        prepare.extract_members(buffer, tmp_path / "out")


def test_prepare_refuses_without_space_for_the_kept_members(
    prepare: ModuleType,
    fetch: ModuleType,
    staged_root: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = staged_root
    monkeypatch.setattr(
        fetch.shutil, "disk_usage", lambda path: type("U", (), {"free": 50})()
    )

    assert prepare.main(["--public-root", str(root), "prepare"]) == 2
    assert not fetch.extracted_dir(root, "KI_HC_S2").exists()
    assert (fetch.raw_dir(root) / BUNDLE).is_file()


def test_prepare_restages_after_a_reextraction_with_extras(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, _ = staged_root
    args = ["--public-root", str(root), "prepare"]
    assert prepare.main(args) == 0

    # Asking for an extra re-extracts the bundle (new inodes); the links the
    # first run staged are replaced without --force.
    assert prepare.main([*args, "--with-mip"]) == 0

    extracted = fetch.extracted_dir(root, "KI_HC_S2")
    target = prepare.xenium_dir(root, "KI_HC_S2")
    for name in (*prepare.REQUIRED_MEMBERS, "morphology_mip.ome.tif"):
        assert (target / name).stat().st_ino == (extracted / name).stat().st_ino


def test_prepare_does_not_replace_a_foreign_file(
    prepare: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    root, _ = staged_root
    target = prepare.xenium_dir(root, "KI_HC_S2")
    target.mkdir(parents=True)
    (target / "cells.parquet").write_text("not ours")

    assert prepare.main(["--public-root", str(root), "prepare"]) == 2
    assert (target / "cells.parquet").read_text() == "not ours"


def test_validate_rejects_cell_tables_the_reader_would_assert_on(
    prepare: ModuleType, tmp_path: Path
) -> None:
    folder = tmp_path / "src"
    _write_bundle_folder(folder)
    report = prepare.validate_section(folder, "KI_HC_S2")
    assert report.n_cells == 2
    assert report.has_qv
    assert report.reader_channel_name == "0"

    cells = pd.read_parquet(folder / "cells.parquet")
    cells.iloc[::-1].to_parquet(folder / "cells.parquet")
    with pytest.raises(prepare.PrepareError, match="barcodes"):
        prepare.validate_section(folder, "KI_HC_S2")

    cells.to_parquet(folder / "cells.parquet")
    boundaries = pd.read_parquet(folder / "nucleus_boundaries.parquet")
    boundaries[boundaries["cell_id"] == CELL_IDS[0]].to_parquet(
        folder / "nucleus_boundaries.parquet"
    )
    with pytest.raises(prepare.PrepareError, match="nucleus_boundaries"):
        prepare.validate_section(folder, "KI_HC_S2")


@pytest.mark.integration
def test_staged_section_builds_with_the_merxen_xenium_builder(
    prepare: ModuleType, fetch: ModuleType, staged_root: tuple[Path, Any]
) -> None:
    """The staged folder goes through MerXen's Xenium build as it is.

    The built image offers the ``DAPI`` channel the nuclei pass asks for
    (spatialdata-io names the single pre-2.0 channel ``0``; the build names it),
    and the transcripts keep their ``qv`` and the micrometre to pixel scale of
    ``experiment.xenium``.
    """
    sd = pytest.importorskip("spatialdata")
    pytest.importorskip("spatialdata_io")
    from spatialdata.models import get_channel_names
    from spatialdata.transformations import get_transformation

    from merxen.config import XeniumBuildConfig
    from merxen.io.builders.xenium import write_xenium_spatialdata
    from merxen.io.image_source import build_image_source

    root, _ = staged_root
    assert prepare.main(["--public-root", str(root), "prepare"]) == 0
    zarr_path = prepare.spatialdata_path(root, "KI_HC_S2")

    write_xenium_spatialdata(
        input_path=prepare.xenium_dir(root, "KI_HC_S2"),
        output_path=zarr_path,
        build_config=XeniumBuildConfig(),
    )

    built = sd.read_zarr(zarr_path)
    assert set(built.labels) == {"cell_labels", "nucleus_labels"}
    assert set(built.shapes) == {"cell_boundaries", "nucleus_boundaries"}
    assert list(built.tables["table"].var_names) == ["GFAP", "CALB2"]
    points = built.points["transcripts"]
    assert "qv" in points.columns
    scale = get_transformation(points).scale
    assert scale[0] == pytest.approx(1 / PIXEL_SIZE)
    assert (zarr_path / "experiment.xenium").is_file()
    image = built.images["morphology_focus"]
    assert [str(name) for name in get_channel_names(image)] == ["DAPI"]
    source = build_image_source(image, requested_channels=["DAPI"])
    assert source["channels"] == ["DAPI"]


# --------------------------------------------------------------------------
# samplesheet


def test_samplesheet_rows_from_env_root(
    samplesheet: ModuleType,
    fetch: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from merxen import local_config

    root = tmp_path / "public"
    fetch.write_manifest(
        fetch.metadata_dir(root) / fetch.MANIFEST_NAME, fetch.list_remote_files(CANNED)
    )
    monkeypatch.setenv("MERXEN_XENIUM_PUBLIC_ROOT", str(root))
    local_config.clear_cache()
    out = tmp_path / "sheets"

    try:
        assert samplesheet.main(["--output-dir", str(out)]) == 0
    finally:
        local_config.clear_cache()

    assert sorted(path.name for path in out.iterdir()) == [samplesheet.SAMPLESHEET_NAME]
    with (out / samplesheet.SAMPLESHEET_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["pair_id"] for row in rows] == [
        "PUBXHPC_KI_HC_S1",
        "PUBXHPC_KI_HC_S3",
        "PUBXHPC_KI_HC_S10",
    ]
    first = rows[0]
    assert first["analysis_mode"] == "xenium"
    assert first["segmentation_source"] == "cellpose_proseg"
    assert first["xenium_channels"] == "DAPI"
    assert first["stop_stage"] == "mender"
    assert first["spatial_gene_analysis_transcript_analysis_enabled"] == "false"
    assert first["clustering_squidpy_mode"] == "legacy"
    assert first["mecr_enabled"] == "false"
    assert (
        Path(first["xenium_dir"])
        == root / "xenium_v1_human_hippocampus_ff_ki/xenium/PUBXHPC_KI_HC_S1"
    )
    assert first["start_stage"] == "build_spatialdata"
    # Blank by default: a filled column is read as the latest (enriched) zarr
    # when a run restarts after enrich.
    assert {row["xenium_spatialdata_path"] for row in rows} == {""}


def test_samplesheet_map_first_leaves_region_blank(
    samplesheet: ModuleType, tmp_path: Path
) -> None:
    options = samplesheet.RowOptions(clustering_mode="map_first", mecr_enabled=True)

    row = samplesheet.build_row(tmp_path, "KI_HC_S4", options)

    assert row["anatomical_region"] == ""
    assert row["mecr_enabled"] == "true"
    assert row["xenium_spatialdata_path"] == ""


def test_samplesheet_cache_source_zarr_fills_the_source_path(
    samplesheet: ModuleType, prepare: ModuleType, tmp_path: Path
) -> None:
    options = samplesheet.RowOptions(cache_source_zarr=True)

    row = samplesheet.build_row(tmp_path, "KI_HC_S1", options)

    assert row["xenium_spatialdata_path"] == str(
        prepare.spatialdata_path(tmp_path, "KI_HC_S1")
    )
    assert row["xenium_spatialdata_path"].endswith("PUBXHPC_KI_HC_S1_xenium.zarr")
    assert row["stop_stage"] == "mender"


def test_staged_folder_discovery_skips_non_section_folders(
    samplesheet: ModuleType, fetch: ModuleType, prepare: ModuleType, tmp_path: Path
) -> None:
    staged = fetch.dataset_root(tmp_path) / "xenium"
    for name in ["KI_HC_S2", "KI_HC_S10", "backup", "KI_HC_S02", "KI_HC_S3"]:
        (staged / f"{prepare.PAIR_ID_PREFIX}{name}").mkdir(parents=True)
    (staged / f"{prepare.PAIR_ID_PREFIX}KI_HC_S3").rmdir()
    (staged / f"{prepare.PAIR_ID_PREFIX}KI_HC_S3").write_text("not a folder")

    labels = samplesheet.discover_sections(tmp_path, None)

    assert labels == ["KI_HC_S2", "KI_HC_S10"]


def test_invalid_sections_raise_a_samplesheet_error(
    samplesheet: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(samplesheet.SamplesheetError, match="invalid section"):
        samplesheet.discover_sections(tmp_path, ["backup"])


def test_samplesheet_refuses_a_path_git_would_track(
    samplesheet: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(samplesheet, "is_git_ignored", lambda path: False)

    with pytest.raises(samplesheet.SamplesheetError, match="not gitignored"):
        samplesheet.write_samplesheet(tmp_path / "sheet.csv", [])


def test_default_samplesheet_location_is_gitignored(samplesheet: ModuleType) -> None:
    target = samplesheet.REPO_ROOT / "workflows" / samplesheet.SAMPLESHEET_NAME

    assert samplesheet.is_git_ignored(target) in (True, None)
