"""Tests for the dentate-gyrus lifespan download/prepare/samplesheet shims.

Everything runs on tiny synthetic fixtures; no network.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "public_data"
    / "xenium_hippocampus"
)
PIXEL_SIZE = 0.2125


def _load(name: str) -> ModuleType:
    key = f"_public_data_{name}"
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
    return _load("fetch_dg_lifespan")


@pytest.fixture(scope="module")
def prepare() -> ModuleType:
    return _load("prepare_dg_lifespan")


@pytest.fixture(scope="module")
def samplesheet() -> ModuleType:
    return _load("make_samplesheet_dg_lifespan")


ROLES = (
    "cell_boundaries.parquet.gz",
    "cell_feature_matrix.h5",
    "cells.parquet.gz",
    "morphology.ome.tif.gz",
    "nucleus_boundaries.parquet.gz",
    "transcripts.parquet.gz",
)


def _soft(samples: list[tuple[str, str]]) -> str:
    lines = []
    for accession, title in samples:
        lines += [f"^SAMPLE = {accession}", f"!Sample_title = {title}"]
        lines += [
            "!Sample_supplementary_file = ftp://ftp.example.org/geo/samples/"
            f"{accession}/suppl/{accession}_{title}_{suffix}"
            for suffix in ROLES
        ]
    return "\n".join(lines) + "\n"


def _filelist(samples: list[tuple[str, str]], skip: str | None = None) -> str:
    rows = [
        "#Archive/File\tName\tTime\tSize\tType",
        "Archive\tX_RAW.tar\t01/01/2026\t999\tTAR",
    ]
    for index, (accession, title) in enumerate(samples):
        for offset, suffix in enumerate(ROLES):
            name = f"{accession}_{title}_{suffix}"
            if name != skip:
                rows.append(
                    f"File\t{name}\t01/01/2026\t{1000 * (index + 1) + offset}\tX"
                )
    return "\n".join(rows) + "\n"


SAMPLES = [
    ("GSM0000001", "SLIDEA_DONORA-adult"),
    ("GSM0000002", "SLIDEA_DONORB-infant"),
]


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def test_build_sections_parses_soft_and_filelist(fetch: ModuleType) -> None:
    sections = fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES))
    assert [s.accession for s in sections] == ["GSM0000001", "GSM0000002"]
    assert [s.pair_id for s in sections] == ["PUBXDG_ADULT", "PUBXDG_INFANT"]
    first = sections[0]
    assert {f.role for f in first.files} == set(fetch.REQUIRED_ROLES)
    assert all(f.url.startswith("https://") for f in first.files)
    assert first.total_bytes == sum(1000 + k for k in range(len(ROLES)))


def test_build_sections_uses_head_for_files_missing_from_filelist(
    fetch: ModuleType,
) -> None:
    skipped = f"GSM0000001_SLIDEA_DONORA-adult_{ROLES[0]}"
    seen: list[str] = []

    def head(url: str) -> int:
        seen.append(url)
        return 42

    sections = fetch.build_sections(
        _soft(SAMPLES), _filelist(SAMPLES, skip=skipped), head
    )
    assert seen == [f"https://ftp.example.org/geo/samples/GSM0000001/suppl/{skipped}"]
    assert sections[0].files[0].size == 42
    with pytest.raises(fetch.FetchError):
        fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES, skip=skipped))


def test_build_sections_requires_every_role(fetch: ModuleType) -> None:
    soft = "\n".join(
        line for line in _soft(SAMPLES[:1]).splitlines() if "transcripts" not in line
    )
    with pytest.raises(fetch.FetchError, match="transcripts"):
        fetch.build_sections(soft, _filelist(SAMPLES))


def test_duplicate_labels_fall_back_to_accessions(fetch: ModuleType) -> None:
    samples = [("GSM0000001", "A-adult"), ("GSM0000002", "B-adult")]
    sections = fetch.build_sections(_soft(samples), _filelist(samples))
    assert [s.pair_id for s in sections] == ["PUBXDG_GSM0000001", "PUBXDG_GSM0000002"]


def test_parse_zenodo_record_reads_md5(fetch: ModuleType) -> None:
    record = {
        "files": [
            {
                "key": "processed.tar.gz",
                "size": 10,
                "checksum": "md5:abc",
                "links": {"self": "https://zenodo.example/content"},
            }
        ]
    }
    (item,) = fetch.parse_zenodo_record(record)
    assert (item.name, item.size, item.md5, item.url) == (
        "processed.tar.gz",
        10,
        "abc",
        "https://zenodo.example/content",
    )


def test_estimate_decompressed_size_resolves_wraps(fetch: ModuleType) -> None:
    payload = b"x" * 5000
    compressed = gzip.compress(payload)
    isize = int.from_bytes(compressed[-4:], "little")
    assert fetch.estimate_decompressed_size(len(compressed), isize) == 5000
    big = 6 * 2**32 + 123
    assert fetch.estimate_decompressed_size(big - 1000, big % 2**32) == big


def test_select_sections_by_label_accession_or_pair_id(fetch: ModuleType) -> None:
    sections = fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES))
    assert [s.label for s in fetch.select_sections(sections, ["infant"])] == ["INFANT"]
    assert len(fetch.select_sections(sections, ["gsm0000001", "PUBXDG_ADULT"])) == 1
    assert len(fetch.select_sections(sections, [""])) == 2
    with pytest.raises(fetch.FetchError):
        fetch.select_sections(sections, ["teen"])


def _remote(fetch: ModuleType, data: bytes, md5: str | None = None) -> Any:
    return fetch.RemoteFile(
        name="f.bin", url="https://x/f.bin", size=len(data), md5=md5
    )


def test_verify_file_checks_size_and_md5(fetch: ModuleType, tmp_path: Path) -> None:
    data = b"hello world"
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    fetch.verify_file(path, _remote(fetch, data))
    good = hashlib.md5(data).hexdigest()  # noqa: S324
    fetch.verify_file(path, _remote(fetch, data, md5=good))
    assert fetch.md5_stamp_path(path).read_text().strip() == good
    with pytest.raises(fetch.FetchError, match="MD5"):
        fetch.md5_stamp_path(path).unlink()
        fetch.verify_file(path, _remote(fetch, data, md5="0" * 32))
    with pytest.raises(fetch.FetchError, match="bytes"):
        fetch.verify_file(path, _remote(fetch, data + b"!"))


def test_plan_files_reports_local_state(fetch: ModuleType, tmp_path: Path) -> None:
    sections = fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES))
    listing = fetch.Listing(sections=sections)
    first = sections[0]
    complete, partial = first.files[0], first.files[1]
    (tmp_path / first.accession).mkdir()
    (tmp_path / first.accession / complete.name).write_bytes(b"x" * complete.size)
    fetch.partial_path(tmp_path / first.accession / partial.name).write_bytes(b"x" * 10)
    planned = fetch.plan_files(listing, tmp_path, sections=sections)
    by_name = {p.remote.name: p for p in planned}
    assert by_name[complete.name].is_complete
    assert by_name[complete.name].remaining_bytes == 0
    assert by_name[partial.name].remaining_bytes == partial.size - 10
    assert len(planned) == 2 * len(ROLES)
    small = fetch.plan_files(listing, tmp_path, sections=sections, small_limit=1002)
    assert {p.remote.size for p in small} <= {1000, 1001, 1002}


def test_download_file_resumes_verifies_and_renames(
    fetch: ModuleType, tmp_path: Path
) -> None:
    data = b"0123456789" * 10
    remote = fetch.RemoteFile(
        name="f.bin",
        url="https://x/f.bin",
        size=len(data),
        md5=hashlib.md5(data).hexdigest(),  # noqa: S324
    )
    path = tmp_path / "sec" / "f.bin"
    path.parent.mkdir()
    fetch.partial_path(path).write_bytes(data[:30])
    commands: list[list[str]] = []

    def fake_curl(command: list[str]) -> None:
        commands.append(command)
        assert "--continue-at" in command
        out = Path(command[command.index("--output") + 1])
        existing = out.stat().st_size
        with out.open("ab") as handle:
            handle.write(data[existing:])

    planned = fetch.PlannedFile(remote, path, 30, False)
    fetch.download_file(planned, fake_curl)
    assert path.read_bytes() == data
    assert not fetch.partial_path(path).exists()
    assert fetch.md5_stamp_path(path).is_file()
    fetch.download_file(planned, fake_curl)
    assert len(commands) == 1


def test_download_file_does_not_rerequest_a_complete_part(
    fetch: ModuleType, tmp_path: Path
) -> None:
    data = b"abcdef" * 7
    remote = fetch.RemoteFile(name="f.bin", url="https://x/f.bin", size=len(data))
    path = tmp_path / "f.bin"
    fetch.partial_path(path).write_bytes(data)

    def no_curl(command: list[str]) -> None:
        raise AssertionError(f"unexpected download: {command}")

    fetch.download_file(fetch.PlannedFile(remote, path, len(data), False), no_curl)
    assert path.read_bytes() == data


def test_download_file_drops_a_corrupt_part(fetch: ModuleType, tmp_path: Path) -> None:
    data = b"0123456789"
    remote = fetch.RemoteFile(
        name="f.bin", url="https://x/f.bin", size=len(data), md5="0" * 32
    )
    path = tmp_path / "f.bin"
    fetch.partial_path(path).write_bytes(data)
    with pytest.raises(fetch.FetchError, match="MD5"):
        fetch.download_file(
            fetch.PlannedFile(remote, path, len(data), False), lambda _c: None
        )
    assert not fetch.partial_path(path).exists()
    assert not path.exists()


@pytest.fixture
def offline_fetch(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[list[str]]]:
    """``fetch.main`` wired to canned listings and a recording fake curl."""
    sections = fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        fetch, "fetch_listing", lambda **_kwargs: fetch.Listing(sections=sections)
    )
    monkeypatch.setattr(fetch, "_run", calls.append)
    monkeypatch.setenv("MERXEN_XENIUM_PUBLIC_ROOT", str(tmp_path / "public"))
    return tmp_path / "public", calls


def test_fetch_dry_run_downloads_and_writes_nothing(
    fetch: ModuleType, offline_fetch: tuple[Path, list[list[str]]]
) -> None:
    root, calls = offline_fetch
    assert fetch.main([]) == 0
    assert calls == []
    assert not root.exists()


def test_fetch_refuses_to_download_without_space(
    fetch: ModuleType,
    offline_fetch: tuple[Path, list[list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, calls = offline_fetch
    monkeypatch.setattr(fetch, "free_bytes", lambda _path: 10)
    assert fetch.main(["--download", "--margin-gb", "0"]) == 2
    assert calls == []
    assert not root.exists()


def test_listing_round_trips_through_json(fetch: ModuleType) -> None:
    sections = fetch.build_sections(_soft(SAMPLES), _filelist(SAMPLES))
    listing = fetch.Listing(sections=sections)
    data = json.loads(json.dumps(fetch.listing_to_dict(listing)))
    again = fetch.listing_from_dict(data)
    assert again.sections[1].pair_id == "PUBXDG_INFANT"
    assert again.sections[0].files[0] == sections[0].files[0]


# --------------------------------------------------------------------------- #
# prepare: miniature bundle
# --------------------------------------------------------------------------- #

HEIGHT, WIDTH = 72, 88
FEATURES = pd.DataFrame(
    {
        "id": [
            "ENSG01",
            "ENSG02",
            "NegControlProbe_00001",
            "UnassignedCodeword_0001",
            "Intergenic_Region_1",
        ],
        "name": [
            "GENEA",
            "GENEB",
            "NegControlProbe_00001",
            "UnassignedCodeword_0001",
            "Intergenic_Region_1",
        ],
        "feature_type": [
            "Gene Expression",
            "Gene Expression",
            "Negative Control Probe",
            "Unassigned Codeword",
            "Genomic Control",
        ],
    }
)


def _cell_ids(n: int) -> list[str]:
    from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32

    return list(
        cell_id_str_from_prefix_suffix_uint32(np.arange(1, n + 1), np.ones(n, int))
    )


def _write_h5(path: Path, barcodes: list[str], features: pd.DataFrame) -> None:
    n_cells, n_features = len(barcodes), len(features)
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 3, (n_cells, n_features))
    indptr, indices, data = [0], [], []
    for row in dense:
        nz = np.flatnonzero(row)
        indices += nz.tolist()
        data += row[nz].tolist()
        indptr.append(len(indices))
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix["barcodes"] = np.array(barcodes, dtype="S")
        matrix["data"] = np.array(data, dtype=np.int32)
        matrix["indices"] = np.array(indices, dtype=np.int64)
        matrix["indptr"] = np.array(indptr, dtype=np.int64)
        matrix["shape"] = np.array([n_features, n_cells], dtype=np.int32)
        group = matrix.create_group("features")
        for key in ("id", "name", "feature_type"):
            group[key] = np.array(features[key].tolist(), dtype="S")
        group["genome"] = np.array(["Unknown"] * n_features, dtype="S")


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
        (cx - half, cy - half),
    ]


def _write_gz_parquet(frame: pd.DataFrame, path: Path) -> None:
    raw = path.with_suffix("")
    frame.to_parquet(raw, index=False)
    path.write_bytes(gzip.compress(raw.read_bytes()))
    raw.unlink()


def _zstack(n_planes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 50, (n_planes, HEIGHT, WIDTH)).astype(np.uint16)
    stack[1, 10:20, 10:20] = 1000  # sharpest structure on plane 1
    stack[2, 40:60, 40:70] = 300
    return stack


@pytest.fixture
def bundle(tmp_path: Path, prepare: ModuleType) -> tuple[Path, Any]:
    """A miniature raw download of one section plus its listing.json."""
    root = tmp_path / "public"
    raw = root / "xenium_prime_5k_proto_human_dentate_gyrus_ff" / "raw"
    accession, title = "GSM0000003", "SLIDEB_DONORC-teen"
    section_dir = raw / accession
    section_dir.mkdir(parents=True)
    ids = _cell_ids(4)
    centres_um = [(3.0, 3.0), (10.0, 4.0), (5.0, 11.0), (14.0, 12.0)]
    cells = pd.DataFrame(
        {
            "cell_id": ids,
            "x_centroid": [c[0] for c in centres_um],
            "y_centroid": [c[1] for c in centres_um],
            "transcript_counts": [3, 2, 1, 0],
            "cell_area": [9.0, 9.0, 9.0, 0.01],
            "nucleus_area": [2.0, 2.0, 2.0, 0.0],
        }
    )
    # label_id as XOA >= 2.0 writes it: cell row + 1 for cells, nucleus
    # polygon row + 1 for nuclei.
    boundaries, nuclei = [], []
    for row, (cell_id, (cx, cy)) in enumerate(zip(ids, centres_um, strict=True)):
        # The last cell's polygon lies outside the image, so the prepare step
        # must stamp it at its centroid.
        bx, by = (-5.0, -5.0) if cell_id == ids[-1] else (cx, cy)
        boundaries += [(cell_id, x, y, row + 1) for x, y in _square(bx, by, 1.5)]
        nuclei += [(cell_id, x, y, row + 1) for x, y in _square(cx - 0.5, cy, 0.4)]
    nuclei += [(ids[0], x, y, 5) for x, y in _square(3.6, 3.0, 0.3)]  # binucleate
    columns = ["cell_id", "vertex_x", "vertex_y", "label_id"]
    transcripts = pd.DataFrame(
        {
            "transcript_id": np.arange(5, dtype=np.int64),
            "cell_id": [ids[0], ids[0], ids[1], "UNASSIGNED", ids[2]],
            "overlaps_nucleus": np.array([1, 0, 0, 0, 1], dtype=np.int32),
            "feature_name": [
                "GENEA",
                "GENEB",
                "GENEA",
                "NegControlProbe_00001",
                "GENEB",
            ],
            "x_location": np.array([3.0, 3.5, 10.0, 1.0, 5.0], dtype=np.float32),
            "y_location": np.array([3.0, 3.2, 4.0, 1.0, 11.0], dtype=np.float32),
            "z_location": np.array([1.0, 2.0, 1.5, 1.0, 2.0], dtype=np.float32),
            "qv": np.array([30.0, 40.0, 25.0, 10.0, 35.0], dtype=np.float32),
        }
    )
    prefix = f"{accession}_{title}_"
    _write_gz_parquet(cells, section_dir / f"{prefix}cells.parquet.gz")
    _write_gz_parquet(
        pd.DataFrame(boundaries, columns=columns),
        section_dir / f"{prefix}cell_boundaries.parquet.gz",
    )
    _write_gz_parquet(
        pd.DataFrame(nuclei, columns=columns),
        section_dir / f"{prefix}nucleus_boundaries.parquet.gz",
    )
    _write_gz_parquet(transcripts, section_dir / f"{prefix}transcripts.parquet.gz")
    _write_h5(section_dir / f"{prefix}cell_feature_matrix.h5", ids, FEATURES)
    tif = section_dir / "zstack.ome.tif"
    tifffile.imwrite(
        tif,
        _zstack(),
        bigtiff=True,
        tile=(16, 16),
        compression="zlib",
        resolution=(1 / PIXEL_SIZE, 1 / PIXEL_SIZE),
        resolutionunit="MICROMETER",
        metadata={"axes": "ZYX", "Channel": {"Name": ["DAPI"]}},
    )
    (section_dir / f"{prefix}morphology.ome.tif.gz").write_bytes(
        gzip.compress(tif.read_bytes())
    )
    tif.unlink()
    files = []
    for path in sorted(section_dir.iterdir()):
        suffix = path.name[len(prefix) :]
        files.append(
            {
                "name": path.name,
                "url": "https://example.org/" + path.name,
                "size": path.stat().st_size,
                "role": _load("fetch_dg_lifespan").ROLE_SUFFIXES[suffix],
                "md5": None,
                "isize": None,
            }
        )
    listing = {
        "sections": [
            {
                "accession": accession,
                "title": title,
                "label": "TEEN",
                "pair_id": "PUBXDG_TEEN",
                "files": files,
            }
        ]
    }
    (raw / "listing.json").write_text(json.dumps(listing))
    (inputs,) = prepare.load_sections(raw)
    return root, inputs


def _scale0(element: Any) -> Any:
    """Return the full-resolution DataArray of a (multiscale) raster element."""
    if hasattr(element, "children") and "scale0" in element.children:
        node = element["scale0"]
        return next(iter(node.ds.data_vars.values()))
    return element


def _options(prepare: ModuleType, **overrides: Any) -> Any:
    defaults: dict[str, Any] = {"tile_size": 32, "mask_chunk": 32, "workers": 2}
    defaults.update(overrides)
    return prepare.Options(**defaults)


def test_project_zstack_mip_and_focus(prepare: ModuleType, tmp_path: Path) -> None:
    stack = _zstack()
    source = tmp_path / "z.ome.tif"
    tifffile.imwrite(
        source,
        stack,
        tile=(16, 16),
        resolution=(1 / PIXEL_SIZE, 1 / PIXEL_SIZE),
        resolutionunit="MICROMETER",
        metadata={"axes": "ZYX"},
    )
    mip = prepare.project_zstack(
        source, tmp_path / "mip.ome.tif", tile_size=32, workers=2
    )
    assert (mip.height, mip.width, mip.planes_used) == (HEIGHT, WIDTH, 3)
    assert mip.pixel_size_um == pytest.approx(PIXEL_SIZE)
    np.testing.assert_array_equal(
        tifffile.imread(tmp_path / "mip.ome.tif"), stack.max(axis=0)
    )
    with tifffile.TiffFile(tmp_path / "mip.ome.tif") as tiff:
        assert len(tiff.pages.first.subifds or []) == prepare.pyramid_level_count(
            HEIGHT, WIDTH, 32
        )
        assert 'Name="DAPI"' in (tiff.ome_metadata or "")
        assert prepare.tiff_pixel_size_um(tiff) == pytest.approx(PIXEL_SIZE)
    prepare.project_zstack(
        source, tmp_path / "focus.ome.tif", method="focus", tile_size=32
    )
    focus = tifffile.imread(tmp_path / "focus.ome.tif")
    # Each 32 px tile comes from a single plane.
    np.testing.assert_array_equal(focus[0:32, 0:32], stack[1, 0:32, 0:32])


def test_control_name_problems_flags_unmatched_controls(prepare: ModuleType) -> None:
    assert prepare.control_name_problems(FEATURES) == []
    odd = pd.concat(
        [
            FEATURES,
            pd.DataFrame(
                {
                    "id": ["X"],
                    "name": ["MysteryControl"],
                    "feature_type": ["Genomic Control"],
                }
            ),
        ]
    )
    assert prepare.control_name_problems(odd) == [
        "MysteryControl (Genomic Control) classified as gene"
    ]


def test_dry_run_plan_reports_sizes_without_writing(
    prepare: ModuleType, bundle: tuple[Path, Any]
) -> None:
    root, inputs = bundle
    out_dir = root / prepare.DATASET_DIR / "xenium" / "PUBXDG_TEEN"
    actions = prepare.plan_section(inputs, out_dir, _options(prepare))
    assert not out_dir.exists()
    by_output = {a.output.name: a for a in actions}
    transcripts = by_output["transcripts.parquet"]
    assert transcripts.estimated_bytes and transcripts.estimated_bytes > 0
    assert not any(a.is_done for a in actions)
    assert by_output["morphology_focus_0000.ome.tif"].estimated_bytes is not None


def test_prepare_section_builds_a_folder_the_xenium_reader_accepts(
    prepare: ModuleType, bundle: tuple[Path, Any]
) -> None:
    from spatialdata_io import xenium as xenium_reader

    from merxen.config import XeniumBuildConfig

    root, inputs = bundle
    raw_before = {p: p.stat().st_mtime_ns for p in inputs.files.values()}
    out_dir = root / prepare.DATASET_DIR / "xenium" / "PUBXDG_TEEN"
    record = prepare.prepare_section(inputs, out_dir, _options(prepare))
    assert {p: p.stat().st_mtime_ns for p in inputs.files.values()} == raw_before

    expected = {
        "cell_feature_matrix.h5",
        "cells.parquet",
        "cell_boundaries.parquet",
        "nucleus_boundaries.parquet",
        "transcripts.parquet",
        "cells.zarr.zip",
        "experiment.xenium",
        "gene_panel.json",
        "merxen_provenance.json",
        "morphology_focus",
    }
    assert {p.name for p in out_dir.iterdir()} == expected
    assert (out_dir / "cell_feature_matrix.h5").stat().st_ino == inputs.files[
        "cell_feature_matrix"
    ].stat().st_ino
    experiment = json.loads((out_dir / "experiment.xenium").read_text())
    assert experiment["pixel_size"] == pytest.approx(PIXEL_SIZE)
    assert experiment["panel_num_targets_predesigned"] == 2
    # The predesigned/custom split is not deposited, so it is not invented.
    assert "panel_num_targets_custom" not in experiment
    sources = experiment["merxen_value_sources"]
    assert set(sources) >= {"analysis_sw_version", "pixel_size", "num_cells"}
    assert "XResolution" in sources["pixel_size"]
    stats = record["cells_zarr"]
    assert stats["cell_labels_stamped"] == 1
    assert stats["cell_label_id_mismatches"] == 0
    assert stats["nucleus_label_id_mismatches"] == 0
    assert record["image"]["source_channel_name"] == "DAPI"
    import zarr

    store = zarr.storage.ZipStore(out_dir / "cells.zarr.zip", read_only=True)
    group: Any = zarr.open_group(store, mode="r")
    # Nucleus mask values are nucleus-polygon rows + 1; their owners are in
    # polygon_sets/0/cell_index (XOA >= 2.0).
    owners = np.asarray(group["polygon_sets/0/cell_index"][...])
    assert owners.tolist() == [0, 1, 2, 3, 0]
    nucleus_mask = np.asarray(group["masks/0"][...])
    assert set(np.unique(nucleus_mask)) == {0, 1, 2, 3, 4, 5}
    second = (int(3.0 / PIXEL_SIZE), int(3.6 / PIXEL_SIZE))
    assert nucleus_mask[second] == 5
    store.close()

    config = XeniumBuildConfig()
    sdata = xenium_reader(
        out_dir,
        cells_boundaries=config.cells_boundaries,
        nucleus_boundaries=config.nucleus_boundaries,
        cells_as_circles=config.cells_as_circles,
        cells_labels=config.cells_labels,
        nucleus_labels=config.nucleus_labels,
        transcripts=config.transcripts,
        morphology_mip=config.morphology_mip,
        morphology_focus=config.morphology_focus,
        aligned_images=config.aligned_images,
        cells_table=config.cells_table,
    )
    from spatialdata.models import get_channel_names

    image = sdata.images["morphology_focus"]
    assert list(get_channel_names(image)) == ["DAPI"]
    assert _scale0(image).shape[-2:] == (HEIGHT, WIDTH)
    table = sdata.tables["table"]
    assert table.n_obs == 4
    assert table.n_vars == 2
    assert table.obs["cell_labels"].tolist() == [1, 2, 3, 4]
    assert table.obs["nucleus_count"].tolist() == [2, 1, 1, 1]
    labels = np.asarray(_scale0(sdata.labels["cell_labels"]))
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    centre = (int(3.0 / PIXEL_SIZE), int(3.0 / PIXEL_SIZE))
    assert labels[centre] == 1
    assert len(sdata.points["transcripts"].compute()) == 5

    # Re-running skips finished outputs and keeps their provenance.
    again = prepare.prepare_section(inputs, out_dir, _options(prepare))
    assert again["steps_this_run"] == []
    assert again["image"] == json.loads(json.dumps(record["image"]))
    assert again["cells_zarr"] == record["cells_zarr"]
    assert len(again["steps_history"]) == 2
    again_experiment = json.loads((out_dir / "experiment.xenium").read_text())
    assert (
        again_experiment["merxen_value_sources"]["pixel_size"]
        == (sources["pixel_size"])
    )


def test_prepare_refuses_when_space_is_short(
    prepare: ModuleType, bundle: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, inputs = bundle
    out_dir = root / "out"
    monkeypatch.setattr(prepare, "free_bytes", lambda _path: 1000)
    with pytest.raises(prepare.PrepareError, match="not enough space"):
        prepare.prepare_section(inputs, out_dir, _options(prepare))
    assert not out_dir.exists()


def test_projection_names_the_channel_dapi(prepare: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "z.ome.tif"
    tifffile.imwrite(
        source,
        _zstack(),
        resolution=(1 / PIXEL_SIZE, 1 / PIXEL_SIZE),
        resolutionunit="MICROMETER",
        metadata={"axes": "ZYX", "Channel": {"Name": ["nuclear stain"]}},
    )
    info = prepare.project_zstack(source, tmp_path / "out.ome.tif", tile_size=32)
    assert (info.channel_name, info.source_channel_name) == ("DAPI", "nuclear stain")
    with tifffile.TiffFile(tmp_path / "out.ome.tif") as tiff:
        assert 'Name="DAPI"' in (tiff.ome_metadata or "")


def test_prepare_refuses_incomplete_downloads(
    prepare: ModuleType, bundle: tuple[Path, Any]
) -> None:
    root, inputs = bundle
    inputs.sizes["transcripts"] += 1
    with pytest.raises(prepare.PrepareError, match="incomplete download"):
        prepare.prepare_section(inputs, root / "out", _options(prepare))


def test_gene_panel_lists_genes_and_negative_probes(prepare: ModuleType) -> None:
    panel = prepare.synthesize_gene_panel(FEATURES)
    descriptors = [t["type"]["descriptor"] for t in panel["payload"]["targets"]]
    assert descriptors == ["gene", "gene", "negative_control"]


# --------------------------------------------------------------------------- #
# samplesheet
# --------------------------------------------------------------------------- #


def test_samplesheet_rows_from_listing_and_prepared_folders(
    samplesheet: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.io.samplesheet import parse_samplesheet, validate_samplesheet

    root = tmp_path / "public"
    dataset = root / "xenium_prime_5k_proto_human_dentate_gyrus_ff"
    (dataset / "raw").mkdir(parents=True)
    listing = {
        "sections": [
            {"accession": "GSM0000001", "pair_id": "PUBXDG_ADULT", "label": "ADULT"},
            {"accession": "GSM0000002", "pair_id": "PUBXDG_INFANT", "label": "INFANT"},
        ]
    }
    (dataset / "raw" / "listing.json").write_text(json.dumps(listing))
    prepared = dataset / "xenium" / "PUBXDG_ADULT"
    prepared.mkdir(parents=True)
    (prepared / "merxen_provenance.json").write_text(
        json.dumps({"pair_id": "PUBXDG_ADULT", "image": {"channel_name": "DAPI"}})
    )
    monkeypatch.setenv("MERXEN_XENIUM_PUBLIC_ROOT", str(root))
    output = tmp_path / "sheet.csv"
    assert samplesheet.main(["--output", str(output)]) == 0
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["pair_id"] for r in rows] == ["PUBXDG_ADULT", "PUBXDG_INFANT"]
    first = rows[0]
    assert first["analysis_mode"] == "xenium"
    assert first["segmentation_source"] == "cellpose_proseg"
    assert first["xenium_channels"] == "DAPI"
    assert first["stop_stage"] == "mender"
    assert first["clustering_squidpy_mode"] == "legacy"
    assert first["mecr_enabled"] == "false"
    assert first["xenium_dir"] == str(prepared)
    # Blank by default: a filled column doubles as the latest zarr on restarts.
    assert first["xenium_spatialdata_path"] == ""
    pairs = parse_samplesheet(output)
    assert pairs[0].xenium_channels == ["DAPI"]
    assert pairs[0].xenium_spatialdata_path is None
    validate_samplesheet(pairs[:1], analysis_mode="xenium")
    assert (
        samplesheet.main(
            ["--output", str(output), "--cache-source-zarr", "--sections", "adult"]
        )
        == 0
    )
    with output.open() as handle:
        (cached,) = list(csv.DictReader(handle))
    assert cached["xenium_spatialdata_path"] == str(
        dataset / "spatialdata" / "PUBXDG_ADULT_xenium_source.zarr"
    )
    assert samplesheet.main(["--output", str(output), "--require-prepared"]) == 1
    assert (
        samplesheet.main(
            [
                "--output",
                str(output),
                "--annotation",
                "map_first_frontal",
                "--sections",
                "PUBXDG_ADULT",
            ]
        )
        == 0
    )
    with output.open() as handle:
        (row,) = list(csv.DictReader(handle))
    assert (row["clustering_squidpy_mode"], row["anatomical_region"]) == (
        "map_first",
        "",
    )


def test_samplesheet_requires_some_section(
    samplesheet: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(samplesheet.SamplesheetError):
        samplesheet.discover_sections(tmp_path)


def test_samplesheet_target_is_gitignored(samplesheet: ModuleType) -> None:
    import subprocess

    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", str(samplesheet.DEFAULT_OUTPUT)],  # noqa: S607
        cwd=samplesheet.REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0


def test_merxen_build_spatialdata_accepts_the_prepared_folder(
    prepare: ModuleType, bundle: tuple[Path, Any], tmp_path: Path
) -> None:
    import spatialdata as sd

    from merxen.config import XeniumBuildConfig
    from merxen.io.builders.xenium import write_xenium_spatialdata

    root, inputs = bundle
    out_dir = root / prepare.DATASET_DIR / "xenium" / "PUBXDG_TEEN"
    prepare.prepare_section(inputs, out_dir, _options(prepare))
    zarr_path = write_xenium_spatialdata(
        input_path=out_dir,
        output_path=tmp_path / "source_spatialdata.zarr",
        build_config=XeniumBuildConfig(),
    )
    built = sd.read_zarr(zarr_path)
    assert set(built.labels) == {"cell_labels", "nucleus_labels"}
    assert set(built.images) == {"morphology_focus"}
    assert built.tables["table"].n_obs == 4
    sidecar = json.loads((zarr_path / "experiment.xenium").read_text())
    assert sidecar["pixel_size"] == pytest.approx(PIXEL_SIZE)


# --------------------------------------------------------------------------
# binary -> UTF-8 cast of Polars-written parquet columns


def _binary_transcripts(path: Path, cell_ids: list[bytes]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(cell_ids)
    table = pa.table(
        {
            "transcript_id": pa.array(range(n), pa.uint64()),
            "cell_id": pa.array(cell_ids, pa.large_binary()),
            "feature_name": pa.array([b"ABAT"] * n, pa.large_binary()),
            "qv": pa.array([30.0] * n, pa.float32()),
        }
    )
    pq.write_table(table, path, row_group_size=2, compression="zstd")


def test_binary_columns_are_cast_to_utf8_in_place(
    prepare: ModuleType, tmp_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "transcripts.parquet"
    ids = [b"UNASSIGNED", b"aaab-1", b"aaac-1", b"UNASSIGNED", b"x"]
    _binary_transcripts(path, ids)

    assert prepare.cast_binary_columns_to_utf8(path) == ["cell_id", "feature_name"]

    source = pq.ParquetFile(path)
    assert source.schema_arrow.field("cell_id").type == pa.large_string()
    assert source.schema_arrow.field("feature_name").type == pa.large_string()
    assert source.schema_arrow.field("qv").type == pa.float32()
    assert source.metadata.num_row_groups == 3
    assert source.metadata.row_group(0).column(0).compression == "ZSTD"
    table = source.read()
    assert table.column("cell_id").to_pylist() == [
        "UNASSIGNED",
        "aaab-1",
        "aaac-1",
        "UNASSIGNED",
        "x",
    ]
    assert table.column("transcript_id").to_pylist() == [0, 1, 2, 3, 4]
    assert not [p for p in prepare.parquet_column_problems(path) if "cell_id" in p]
    # A second call finds nothing to cast and leaves the file alone.
    before = path.stat().st_mtime_ns
    assert prepare.cast_binary_columns_to_utf8(path) == []
    assert path.stat().st_mtime_ns == before


def test_invalid_utf8_fails_and_keeps_the_original(
    prepare: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "transcripts.parquet"
    _binary_transcripts(path, [b"ok-1", b"\xff\xfe"])
    original = path.read_bytes()

    with pytest.raises(prepare.PrepareError, match="not valid UTF-8"):
        prepare.cast_binary_columns_to_utf8(path)

    assert path.read_bytes() == original
    assert not (tmp_path / "transcripts.parquet.partial").exists()
