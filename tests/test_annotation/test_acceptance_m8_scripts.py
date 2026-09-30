"""Tests of the M8 acceptance scripts: referee, legacy comparison, scoring."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"
SET_A = "ffc6dd2d96e2a6cce6f37fe95754ba6e057aa412da7df2e0688ad921395739c6"
SEAAD = "dc3c500f2ed8e5f43ec2900a43ef5214957eea095694b2d9e6d8886868f5c240"


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def referee() -> ModuleType:
    return _load("marker_referee")


@pytest.fixture(scope="module")
def legacy() -> ModuleType:
    return _load("compare_legacy")


@pytest.fixture(scope="module")
def acceptance() -> ModuleType:
    return _load("run_acceptance")


@pytest.fixture(scope="module")
def criteria() -> ModuleType:
    return _load("resolve_criteria")


# ---------------------------------------------------------------------------
# marker_referee.py

SYMBOLS = ["SLC17A7", "GAD1", "AQP4", "GJA1", "MOBP", "MOG", "OTHER"]


def _referee_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ct_broad_status": [
                "confident",
                "confident",
                "confident",
                "unresolved",
                "confident",
                "confident",
            ],
            "ct_broad_name": [
                "Neurons",
                "Astrocytes",
                "Oligodendrocytes",
                "Astrocytes",
                "Neurons",
                "Mixed/Unknown",
            ],
        },
        index=pd.Index([f"c{i}" for i in range(6)], name="cell_id"),
    )


def _referee_counts() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.array(
            [
                [5, 0, 0, 0, 0, 0, 1],  # neuron markers: new Neurons wins
                [0, 0, 0, 0, 4, 0, 1],  # oligo markers: new Astrocytes loses
                [0, 0, 0, 0, 3, 3, 0],  # oligo: agree with legacy (no dispute)
                [0, 0, 3, 0, 0, 0, 0],  # not confident: no new label
                [1, 0, 1, 0, 0, 0, 0],  # tie Neurons vs Astrocytes
                [0, 0, 0, 0, 2, 0, 0],  # outside the classes: no dispute
            ],
            dtype=np.float64,
        )
    )


LEGACY_BROAD = np.array(
    [
        "Astrocytes",
        "Oligodendrocytes",
        "Oligodendrocytes",
        "Neurons",
        "Astrocytes",
        "Oligodendrocytes",
    ],
    dtype=object,
)


def _referee_sample(referee: ModuleType) -> Any:
    return referee.RefereeSample(
        pair="P7513",
        segmentation="proseg_hybrid",
        sample_id="P7513_MERSCOPE",
        platform="MERSCOPE",
        labels=_referee_labels(),
        counts=_referee_counts(),
        symbols=SYMBOLS,
        legacy_broad=LEGACY_BROAD,
    )


def test_marker_referee_scores_exactly_as_resolve_criteria(
    referee: ModuleType, criteria: ModuleType
) -> None:
    sample = _referee_sample(referee)
    scores, _ = referee.marker_class_scores(sample.counts, sample.symbols)
    mine = referee.referee_row(sample, scores)
    resolved = criteria.ResolvedSample(
        pair=sample.pair,
        segmentation=sample.segmentation,
        sample_id=sample.sample_id,
        platform=sample.platform,
        labels=sample.labels.assign(mmc_whb_supercluster_name="Astrocyte"),
        summary={},
        sea=pd.DataFrame({"broad": [None] * 6}, index=sample.labels.index),
        counts=sample.counts,
        lognorm=sample.counts,
        symbols=sample.symbols,
        legacy_broad=sample.legacy_broad,
    )
    theirs = next(
        row
        for row in criteria.referee_rows(resolved)
        if row["first"] == referee.NEW and row["second"] == referee.LEGACY
    )
    for column in referee.CHECKED + ("dispute_share", "top_disputes"):
        assert mine[column] == theirs[column], column
    # 3 disputes: c0 new wins, c1 legacy wins, c4 tie.
    assert mine["n_disputes"] == 3
    assert mine["first_wins"] == pytest.approx(1 / 3)
    assert mine["second_wins"] == pytest.approx(1 / 3)
    assert mine["undecided"] == pytest.approx(1 / 3)
    table = criteria.referee_table_rows([mine])
    assert table[0]["criterion"] == "H10" and table[0]["passes"] is False


def test_new_labels_are_only_the_confident_broad_names(referee: ModuleType) -> None:
    labels = referee.new_labels(_referee_labels())
    assert list(labels) == [
        "Neurons",
        "Astrocytes",
        "Oligodendrocytes",
        None,
        "Neurons",
        "Mixed/Unknown",
    ]


def test_dispute_rows_count_the_wins_per_class_pair(referee: ModuleType) -> None:
    sample = _referee_sample(referee)
    scores, _ = referee.marker_class_scores(sample.counts, sample.symbols)
    rows = {
        (row["new"], row["legacy"]): row for row in referee.dispute_rows(sample, scores)
    }
    assert set(rows) == {
        ("Neurons", "Astrocytes"),
        ("Astrocytes", "Oligodendrocytes"),
    }
    neuron = rows[("Neurons", "Astrocytes")]
    assert neuron["n"] == 2
    assert neuron["new_wins"] == pytest.approx(0.5)
    assert neuron["undecided"] == pytest.approx(0.5)
    assert rows[("Astrocytes", "Oligodendrocytes")]["legacy_wins"] == 1.0
    assert sum(row["share_of_disputes"] for row in rows.values()) == pytest.approx(1.0)


def test_check_against_lists_every_difference(referee: ModuleType) -> None:
    sample = _referee_sample(referee)
    scores, _ = referee.marker_class_scores(sample.counts, sample.symbols)
    row = referee.referee_row(sample, scores)
    same = pd.DataFrame([row])
    assert referee.check_against([row], same) == []
    other = same.copy()
    other.loc[0, "first_wins"] = 0.5
    problems = referee.check_against([row], other)
    assert len(problems) == 1 and "first_wins" in problems[0]
    extra = pd.concat([same, same.assign(sample_id="P7513_XENIUM")])
    assert any(
        "only in the criteria" in item for item in referee.check_against([row], extra)
    )
    assert referee.check_against([row], same.assign(first="WHB argmax")) == [
        "('proseg_hybrid', 'P7513_MERSCOPE'): not in the criteria referee.csv"
    ]


def _write_labels(path: Path, labels: pd.DataFrame, in_table: list[bool]) -> None:
    frame = labels.reset_index()
    frame["in_table"] = in_table
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_load_referee_sample_aligns_the_h5ad_to_the_table_cells(
    referee: ModuleType, tmp_path: Path
) -> None:
    labels = _referee_labels()
    resolve_dir = tmp_path / "resolve" / "P7513" / "proseg_hybrid"
    _write_labels(
        resolve_dir / "merscope" / "P7513_MERSCOPE_celltype_labels.parquet",
        labels,
        [True, True, True, True, True, False],
    )
    counts = _referee_counts().toarray()[::-1]
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(
            {"broad_class": pd.Categorical(LEGACY_BROAD[::-1])},
            index=[f"c{i}" for i in range(5, -1, -1)],
        ),
        var=pd.DataFrame(index=SYMBOLS),
    )
    adata.layers["counts"] = sparse.csr_matrix(counts)
    h5ad = (
        tmp_path
        / "results/P7513/proseg_hybrid/clustering_squidpy"
        / "clustering_squidpy_out/merscope"
    )
    h5ad.mkdir(parents=True)
    adata.write_h5ad(h5ad / "P7513_MERSCOPE_clustered.h5ad")
    summary = {
        "samples": {
            "P7513_MERSCOPE": {
                "labels": "merscope/P7513_MERSCOPE_celltype_labels.parquet"
            }
        }
    }
    sample = referee.load_referee_sample(
        resolve_dir, tmp_path / "results", summary, "P7513", "proseg_hybrid", "MERSCOPE"
    )
    assert list(sample.labels.index) == ["c0", "c1", "c2", "c3", "c4"]
    np.testing.assert_array_equal(
        sample.counts.toarray(), _referee_counts().toarray()[:5]
    )
    assert list(sample.legacy_broad) == list(LEGACY_BROAD[:5])


def test_marker_referee_help_runs(referee: ModuleType) -> None:
    with pytest.raises(SystemExit) as excinfo:
        referee.main(["--help"])
    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# compare_legacy.py


def _tree(root: Path) -> Path:
    results = root / "results"
    store = results / "P7513" / "merscope" / "latest" / "latest_spatialdata.zarr"
    for table in (
        "table_MOSAIK_proseg_hybrid",
        "table_MOSAIK_proseg_hybrid_clustering_squidpy",
    ):
        (store / "tables" / table / "obs").mkdir(parents=True)
        (store / "tables" / table / "zarr.json").write_text(f'{{"t": "{table}"}}')
        (store / "tables" / table / "obs" / "c0").write_bytes(b"abcd")
    (store / "zarr.json").write_text('{"root": 1}')
    (store / "points").mkdir()
    (store / "points" / "p0").write_bytes(b"points")
    clustered = results / "P7513" / "proseg_hybrid" / "clustering_squidpy" / "out"
    clustered.mkdir(parents=True)
    (clustered / "P7513_MERSCOPE_clustered.h5ad").write_bytes(b"h5ad")
    mender = results / "P7513" / "proseg_hybrid" / "mender" / "mender_out"
    mender.mkdir(parents=True)
    (mender / "m.h5ad").write_bytes(b"mender")
    other = results / "P7513" / "proseg_hybrid" / "visualization"
    other.mkdir(parents=True)
    (other / "plot.png").write_bytes(b"png")
    os.symlink("plot.png", other / "link.png")
    return results


def test_needs_hash_selects_tables_legacy_outputs_and_depth(legacy: ModuleType) -> None:
    zarr = "P7513/merscope/latest/latest_spatialdata.zarr"
    assert legacy.needs_hash(f"{zarr}/tables/t/obs/c0")
    assert legacy.needs_hash(f"{zarr}/zarr.json")
    assert not legacy.needs_hash(f"{zarr}/points/p0")
    assert legacy.needs_hash("P7513/reseg/clustering_squidpy/out/x.h5ad")
    assert legacy.needs_hash("P7513/proseg_hybrid/mender_mapfirst/mender_out/x")
    assert legacy.needs_hash("P7513/proseg_hybrid/annotation_map/annotation_map_out/m")
    assert legacy.needs_hash("P7513/xenium/compute_cortical_depth/out/q.json")
    assert legacy.needs_hash("acceptance/2026-10-01/runs/x")
    assert not legacy.needs_hash("P7513/proseg_hybrid/visualization/plot.png")
    assert not legacy.needs_hash("P7513/xenium/segmentation/s.parquet")


def test_manifest_records_entries_hashes_and_links(
    legacy: ModuleType, tmp_path: Path
) -> None:
    results = _tree(tmp_path)
    out = tmp_path / "m.tsv"
    counts = legacy.write_manifest(results, ["P7513", "acceptance"], out, workers=1)
    rows = legacy.load_manifest(out)
    assert counts["entries"] == len(rows)
    table_file = (
        "P7513/merscope/latest/latest_spatialdata.zarr/tables/"
        "table_MOSAIK_proseg_hybrid/obs/c0"
    )
    assert rows[table_file]["sha256"] == legacy.file_sha256(str(results / table_file))
    assert (
        rows["P7513/merscope/latest/latest_spatialdata.zarr/points/p0"]["sha256"] == "-"
    )
    link = rows["P7513/proseg_hybrid/visualization/link.png"]
    assert link["type"] == "l" and link["link_target"] == "plot.png"
    assert rows["P7513"]["type"] == "d"
    assert "acceptance" not in rows  # an absent root adds nothing


def _after_run(results: Path) -> None:
    """What the acceptance run may do: a new _m8 table, the root, new publish dirs."""
    store = results / "P7513" / "merscope" / "latest" / "latest_spatialdata.zarr"
    new = store / "tables" / "table_MOSAIK_proseg_hybrid_clustering_squidpy_m8"
    (new / "obs").mkdir(parents=True)
    (new / "obs" / "c0").write_bytes(b"new")
    (store / "zarr.json").write_text('{"root": 2, "consolidated": true}')
    runs = results / "acceptance" / "2026-10-01" / "runs" / "P7513"
    runs.mkdir(parents=True)
    (runs / "summary.json").write_text("{}")
    (results / "P7513" / "merscope" / "latest" / "lock").write_text("")


def _manifests(legacy: ModuleType, tmp_path: Path, change: Any = None) -> tuple:
    results = _tree(tmp_path)
    before = tmp_path / "before.tsv"
    after = tmp_path / "after.tsv"
    legacy.write_manifest(results, ["P7513", "acceptance"], before, workers=1)
    _after_run(results)
    if change is not None:
        change(results)
    legacy.write_manifest(results, ["P7513", "acceptance"], after, workers=1)
    return legacy.load_manifest(before), legacy.load_manifest(after)


def test_untouched_passes_when_the_run_only_adds(
    legacy: ModuleType, tmp_path: Path
) -> None:
    before, after = _manifests(legacy, tmp_path)
    groups, summary = legacy.check_untouched(before, after)
    assert summary["passes"] is True
    assert summary["n_legacy_tables"] == 2
    assert summary["n_legacy_clustering"] == 1 and summary["n_legacy_mender"] == 1
    assert summary["failing_groups"] == []
    assert any(
        key.endswith("table_MOSAIK_proseg_hybrid_clustering_squidpy_m8")
        for key in summary["new_groups"]["new_table"]
    )
    # the root re-consolidation is reported beside H13, never counted against it
    changed = {item.kind for item in groups if not item.untouched}
    assert "root_metadata" in changed and not changed & set(legacy.H13_GROUPS)


def _same_size_rewrite(path: Path, content: bytes) -> None:
    stat = path.stat()
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        (
            lambda r: _same_size_rewrite(
                r
                / "P7513/merscope/latest/latest_spatialdata.zarr/tables"
                / "table_MOSAIK_proseg_hybrid_clustering_squidpy/obs/c0",
                b"abce",
            ),
            "tables/table_MOSAIK_proseg_hybrid_clustering_squidpy",
        ),
        (
            lambda r: (r / "P7513/proseg_hybrid/mender/mender_out/m.h5ad").unlink(),
            "P7513/proseg_hybrid/mender",
        ),
        (
            lambda r: (
                r / "P7513/proseg_hybrid/clustering_squidpy/out/extra.h5ad"
            ).write_bytes(b"x"),
            "P7513/proseg_hybrid/clustering_squidpy",
        ),
    ],
)
def test_untouched_fails_on_any_legacy_change(
    legacy: ModuleType, tmp_path: Path, change: Any, failing: str
) -> None:
    before, after = _manifests(legacy, tmp_path, change)
    _, summary = legacy.check_untouched(before, after)
    assert summary["passes"] is False
    assert any(key.endswith(failing) for key in summary["failing_groups"])


def test_untouched_needs_the_sha256_of_legacy_files(
    legacy: ModuleType, tmp_path: Path
) -> None:
    before, after = _manifests(legacy, tmp_path)
    key = "P7513/proseg_hybrid/clustering_squidpy/out/P7513_MERSCOPE_clustered.h5ad"
    before[key] = {**before[key], "sha256": "-"}
    after[key] = {**after[key], "sha256": "-"}
    _, summary = legacy.check_untouched(before, after)
    assert summary["passes"] is False and summary["unhashed_files"] == [key]


def test_untouched_fails_without_any_legacy_table(legacy: ModuleType) -> None:
    rows = {
        "P7513/proseg_hybrid/visualization/plot.png": {
            "type": "f",
            "path": "P7513/proseg_hybrid/visualization/plot.png",
            "size": "3",
            "mtime_ns": "1",
            "sha256": "-",
            "link_target": "",
        }
    }
    _, summary = legacy.check_untouched(rows, rows)
    assert summary["passes"] is False and summary["n_legacy_tables"] == 0


def test_a_new_table_pattern_never_hides_a_legacy_table(legacy: ModuleType) -> None:
    import re

    pattern = re.compile(legacy.DEFAULT_NEW_TABLE_PATTERN)
    zarr = "P1/merscope/latest/latest_spatialdata.zarr/tables"
    for name, kind in (
        ("table_MOSAIK_proseg_hybrid_clustering_squidpy_m8", "new_table"),
        ("table_MOSAIK_proseg_hybrid_clustering_squidpy_m8_state", "new_table"),
        (".table_original_clustering_squidpy_m8.merxen-backup-1a2b", "new_table"),
        ("table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst", "legacy_table"),
        ("table_MOSAIK_proseg_hybrid_clustering_squidpy", "legacy_table"),
        ("table_original", "legacy_table"),
        ("zarr.json", "tables_metadata"),
    ):
        assert legacy.classify_path(f"{zarr}/{name}/x", pattern)[0] == kind, name


def _h5ad(path: Path, ids: list[str], broad: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(
        X=sparse.csr_matrix((len(ids), 1)),
        obs=pd.DataFrame({"broad_class": pd.Categorical(broad)}, index=ids),
    ).write_h5ad(path)


def test_compare_dataset_counts_disputes_crosswalk_and_ids(
    legacy: ModuleType, tmp_path: Path
) -> None:
    _h5ad(
        tmp_path / "legacy.h5ad",
        ["a", "b", "c", "d"],
        ["Neurons", "Astrocytes", "Microglia", "Neurons"],
    )
    _h5ad(
        tmp_path / "mapfirst.h5ad",
        ["a", "b", "c", "e"],
        ["Neurons", "Oligodendrocyte lineage", "Mixed/Unknown", "Neurons"],
    )
    labels = pd.DataFrame(
        {
            "ct_broad_status": ["confident", "confident", "unresolved", "confident"],
            "ct_broad_name": ["Neurons", "Oligodendrocytes", "Microglia", "Neurons"],
        },
        index=pd.Index(["a", "b", "c", "e"], name="cell_id"),
    )
    _write_labels(tmp_path / "labels.parquet", labels, [True, True, True, True])
    out = legacy.compare_dataset(
        tmp_path / "legacy.h5ad",
        tmp_path / "mapfirst.h5ad",
        tmp_path / "labels.parquet",
        {
            "pair": "P7513",
            "segmentation": "proseg_hybrid",
            "sample_id": "P7513_MERSCOPE",
        },
    )
    h10 = out["h10_inputs"][0]
    assert h10["n_common"] == 3 and h10["n_new_only"] == 1 and h10["n_legacy_only"] == 1
    assert h10["n_both_in_classes"] == 2 and h10["n_disputes"] == 1
    assert h10["agreement"] == pytest.approx(0.5)
    crosswalk = {
        (row["legacy_broad_class"], row["mapfirst_broad_class"]): row["n"]
        for row in out["crosswalk"]
    }
    assert crosswalk == {
        ("Neurons", "Neurons"): 1,
        ("Astrocytes", "Oligodendrocyte lineage"): 1,
        ("Microglia", "Mixed/Unknown"): 1,
    }
    ids = out["id_sets"][0]
    assert ids["mapfirst_equals_label_table"] and not ids["legacy_equals_mapfirst"]
    shares = {(row["source"], row["class"]): row["share"] for row in out["composition"]}
    assert shares[("new_confident", "not labelled")] == pytest.approx(0.25)


def test_compare_legacy_help_runs(legacy: ModuleType) -> None:
    with pytest.raises(SystemExit) as excinfo:
        legacy.main(["--help"])
    assert excinfo.value.code == 0


# ---------------------------------------------------------------------------
# run_acceptance.py: parsing, P5, H14


def test_trace_values_parse(acceptance: ModuleType) -> None:
    assert acceptance.parse_duration("1h 2m 3s") == pytest.approx(3723.0)
    assert acceptance.parse_duration("57.7s") == pytest.approx(57.7)
    assert acceptance.parse_duration("300ms") == pytest.approx(0.3)
    assert acceptance.parse_duration("1d 1h") == pytest.approx(90000.0)
    assert np.isnan(acceptance.parse_duration("-"))
    assert np.isnan(acceptance.parse_duration("12 minutes"))
    assert acceptance.parse_memory_gb("8.7 GB") == pytest.approx(8.7)
    assert acceptance.parse_memory_gb("512 MB") == pytest.approx(0.5)
    assert acceptance.parse_memory_gb("1 TB") == pytest.approx(1024.0)
    assert acceptance.parse_memory_gb("0") == 0.0
    assert np.isnan(acceptance.parse_memory_gb("-"))


TRACE_COLUMNS = [
    "name",
    "process",
    "tag",
    "status",
    "exit",
    "cpus",
    "realtime",
    "peak_rss",
    "workdir",
]


def _trace(path: Path, rows: list[list[str]]) -> Path:
    pd.DataFrame(rows, columns=TRACE_COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def _task(
    process: str,
    tag: str,
    realtime: str,
    rss: str = "1 GB",
    status: str = "COMPLETED",
    workdir: str = "",
) -> list[str]:
    return [
        f"{process} ({tag})",
        f"A:B:{process}",
        tag,
        status,
        "0",
        "4",
        realtime,
        rss,
        workdir,
    ]


def test_load_trace_keeps_a_computed_record_over_a_cached_one(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    first = _trace(tmp_path / "a.tsv", [_task("ANNOTATE_PANEL", "P7513:reseg", "10s")])
    second = _trace(
        tmp_path / "b.tsv",
        [
            _task("ANNOTATE_PANEL", "P7513:reseg", "0ms", status="CACHED"),
            _task("CLUSTERING_SQUIDPY_ANNOTATE_MAP", "P7513:reseg", "1m"),
        ],
    )
    trace = acceptance.load_trace([first, second])
    panel = trace[trace["process"] == "ANNOTATE_PANEL"].iloc[0]
    assert panel["realtime_s"] == 10.0 and panel["trace"] == "a.tsv"
    assert set(trace["process"]) == {
        "ANNOTATE_PANEL",
        "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
    }


def _h14_trace(
    acceptance: ModuleType,
    tmp_path: Path,
    map_rss: str = "10 GB",
    segs: tuple = ("proseg_hybrid", "reseg", "proseg_mask", "original_seg"),
) -> pd.DataFrame:
    rows = []
    for pair in ("P7513", "P7113"):
        for seg in segs:
            tag = f"{pair}:{seg}"
            rows.append(_task("ANNOTATE_PANEL", tag, "1m"))
            rows.append(_task("CLUSTERING_SQUIDPY_ANNOTATE_MAP", tag, "30m", map_rss))
            rows.append(_task("CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE", tag, "2m"))
            rows.append(_task("CLUSTERING_SQUIDPY_COMPUTE_CPU", tag, "40m"))
    return acceptance.load_trace([_trace(tmp_path / "t.tsv", rows)])


def test_h14_sums_the_annotation_wall_over_all_segmentations(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    trace = _h14_trace(acceptance, tmp_path)
    rows = {
        (row.criterion, row.pair): row
        for row in acceptance.h14_rows(
            trace, ["P7513", "P7113"], acceptance.SEGMENTATIONS, []
        )
    }
    wall = rows[("H14/annotation_wall", "P7513")]
    assert wall.value == pytest.approx(4 * 33 / 60)  # COMPUTE_CPU is not annotation
    assert wall.passes is True and wall.scored is True
    assert rows[("H14/annotation_wall", "P7113")].scored is False
    assert rows[("H14/map_peak_rss", "P7513")].value == pytest.approx(10.0)


def test_h14_fails_above_the_limits_and_needs_every_segmentation(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    trace = _h14_trace(acceptance, tmp_path, map_rss="17 GB")
    rows = acceptance.h14_rows(trace, ["P7513"], acceptance.SEGMENTATIONS, [])
    peak = next(row for row in rows if row.criterion == "H14/map_peak_rss")
    assert peak.passes is False and peak.verdict == "FAIL-OUTSIDE"
    short = _h14_trace(acceptance, tmp_path, segs=("proseg_hybrid", "reseg"))
    wall = next(
        row
        for row in acceptance.h14_rows(short, ["P7513"], acceptance.SEGMENTATIONS, [])
        if row.criterion == "H14/annotation_wall"
    )
    assert wall.passes is None and wall.verdict == "NOT_AVAILABLE"
    assert "missing original_seg,proseg_mask" in wall.note
    long = acceptance.load_trace(
        [
            _trace(
                tmp_path / "long.tsv",
                [
                    _task("CLUSTERING_SQUIDPY_ANNOTATE_MAP", f"P7513:{seg}", "40m")
                    for seg in acceptance.SEGMENTATIONS
                ],
            )
        ]
    )
    wall = next(
        row
        for row in acceptance.h14_rows(long, ["P7513"], acceptance.SEGMENTATIONS, [])
        if row.criterion == "H14/annotation_wall"
    )
    assert wall.value == pytest.approx(160 / 60) and wall.passes is False


def test_h14_prep_rows_use_each_bundles_build_time_and_check_gpus(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    hidden = tmp_path / "w1"
    hidden.mkdir()
    # Nextflow's .command.run exports the config env with single quotes
    (hidden / ".command.run").write_text("    export CUDA_VISIBLE_DEVICES=''\n")
    exposed = tmp_path / "w2"
    exposed.mkdir()
    (exposed / ".command.run").write_text("nothing\n")
    rows = [_task("ANNOTATE_REFERENCE_PREP", "whb:1", "12s", workdir=str(hidden))]
    trace = acceptance.load_trace([_trace(tmp_path / "t.tsv", rows)])
    bundles = [
        {"reference_id": "whb", "build_hash": SET_A, "build_wall_time_s": 137.5},
        {"reference_id": "slow", "build_hash": "b" * 64, "build_wall_time_s": 3 * 3600},
    ]
    quoted = tmp_path / "w3"
    quoted.mkdir()
    (quoted / ".command.run").write_text('export CUDA_VISIBLE_DEVICES=""\n')
    rows.append(_task("ANNOTATION_REPORT", "P7513:reseg", "30s", workdir=str(quoted)))
    trace = acceptance.load_trace([_trace(tmp_path / "t1.tsv", rows)])
    out = acceptance.h14_rows(trace, [], acceptance.SEGMENTATIONS, bundles)
    prep = [row for row in out if row.criterion == "H14/prep_wall"]
    assert [row.passes for row in prep] == [True, False]
    gpu = next(row for row in out if row.criterion == "H14/no_gpu")
    assert gpu.passes is True and "CUDA hidden in 2 of 2" in gpu.note
    rows.append(
        _task("CLUSTERING_SQUIDPY_COMPUTE", "P7513:reseg", "1m", workdir=str(exposed))
    )
    trace = acceptance.load_trace([_trace(tmp_path / "t2.tsv", rows)])
    gpu = next(
        row
        for row in acceptance.h14_rows(trace, [], acceptance.SEGMENTATIONS, [])
        if row.criterion == "H14/no_gpu"
    )
    assert gpu.passes is False and "CLUSTERING_SQUIDPY_COMPUTE" in gpu.note


def _bundle(root: Path, build_hash: str, created: str, workers: int = 8) -> Path:
    directory = root / "store" / build_hash
    directory.mkdir(parents=True)
    (directory / "bundle.json").write_text(
        json.dumps({"created_at": created, "wall_time_s": 100.0})
    )
    (directory / "resolvability_summary.json").write_text(
        json.dumps(
            {"mapping_runs": [{"n_processors": workers}, {"n_processors": workers}]}
        )
    )
    ref = root / "refs" / build_hash[:8] / "bundle_ref.json"
    ref.parent.mkdir(parents=True)
    ref.write_text(
        json.dumps(
            {"reference_id": "whb", "build_hash": build_hash, "path": str(directory)}
        )
    )
    return ref


RUN_START = "2026-10-01T08:00:00+00:00"


def test_p5_accepts_reused_d1_bundles(acceptance: ModuleType, tmp_path: Path) -> None:
    refs = [
        _bundle(tmp_path, SET_A, "2026-09-30T10:52:48+00:00"),
        _bundle(tmp_path, SEAAD, "2026-10-01T09:00:00+00:00", workers=8),
    ]
    check = acceptance.check_prep_bundles(refs, acceptance._parse_time(RUN_START))
    assert check.ok, check.reasons
    assert [b["created_before_run"] for b in check.bundles] == [True, False]


@pytest.mark.parametrize(
    ("build_hash", "created", "workers", "logged", "message"),
    [
        ("a" * 64, "2026-09-30T10:00:00+00:00", 8, False, "is not a D1 bundle"),
        (SET_A, "2026-10-01T09:00:00+00:00", 6, False, "records [6] workers"),
    ],
)
def test_p5_refuses_other_bundles_and_other_worker_counts(
    acceptance: ModuleType,
    tmp_path: Path,
    build_hash: str,
    created: str,
    workers: int,
    logged: bool,
    message: str,
) -> None:
    refs = [_bundle(tmp_path, build_hash, created, workers)]
    check = acceptance.check_prep_bundles(refs, acceptance._parse_time(RUN_START))
    assert not check.ok and message in check.reasons[0]


def test_p5_accepts_a_logged_reuse_and_checks_every_map_run(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    refs = [_bundle(tmp_path, SET_A, "2026-10-01T09:00:00+00:00", workers=6)]
    start = acceptance._parse_time(RUN_START)
    reuse = acceptance.prep_reuse_from_logs(
        [f"annotation-reference-prep: reused whb_frontal_supc_clus {SET_A[:16]} -> /x"]
    )
    assert reuse == {SET_A[:16]: True}
    check = acceptance.check_prep_bundles(refs, start, reuse_logged={SET_A: True})
    assert check.ok
    manifest = tmp_path / "map_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": {
                    "P7513_MERSCOPE": {
                        "runs": {
                            "whb": {"build_hash": SET_A},
                            "seaad": {"build_hash": SEAAD},
                        }
                    }
                }
            }
        )
    )
    check = acceptance.check_prep_bundles(
        refs,
        start,
        reuse_logged={SET_A: True},
        map_manifests={("P7513", "reseg"): manifest},
    )
    assert not check.ok and "seaad" in check.reasons[0]
    assert acceptance.check_prep_bundles([], start).ok is False


def test_prep_logs_of_a_build_are_not_a_reuse(acceptance: ModuleType) -> None:
    reuse = acceptance.prep_reuse_from_logs(
        [f"annotation-reference-prep: built whb {SET_A[:16]} -> /x"]
    )
    assert reuse == {SET_A[:16]: False}


def test_p5_checks_the_second_policys_map_runs(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    main = runs / acceptance.MAP_MANIFEST_PATH.format(pair="P7513", seg="proseg_hybrid")
    state = runs / acceptance.STATE_MAP_MANIFEST_PATH.format(pair="P7513")
    for path, build_hash in ((main, SET_A), (state, "b" * 64)):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "samples": {
                        "P7513_MERSCOPE": {"runs": {"whb": {"build_hash": build_hash}}}
                    }
                }
            )
        )
    found = acceptance.map_manifests(runs, ["P7513"], ["proseg_hybrid", "reseg"])
    assert found == {
        ("P7513", "proseg_hybrid"): main,
        ("P7513", "proseg_hybrid/mender_state_policy"): state,
    }
    refs = [_bundle(tmp_path, SET_A, "2026-09-30T10:52:48+00:00")]
    check = acceptance.check_prep_bundles(
        refs, acceptance._parse_time(RUN_START), map_manifests=found
    )
    assert not check.ok
    assert len(check.reasons) == 1 and "mender_state_policy" in check.reasons[0]


# ---------------------------------------------------------------------------
# run_acceptance.py: criteria rows with the §18 scopes and the report cross-check


def _criteria_table() -> pd.DataFrame:
    def row(
        criterion: str,
        seg: str,
        dataset: str,
        value: float | None,
        threshold: float | None,
        comparator: str,
        passes: bool | None,
        scored: bool,
        note: str = "",
    ) -> dict:
        return {
            "criterion": criterion,
            "pair": dataset.split("_")[0],
            "segmentation": seg,
            "dataset": dataset,
            "value": value,
            "comparator": comparator,
            "threshold": threshold,
            "passes": passes,
            "scored": scored,
            "note": note,
        }

    return pd.DataFrame(
        [
            row("H1", "proseg_hybrid", "P7513", 0.129, 0.17, "<=", True, True),
            row(
                "H2",
                "proseg_hybrid",
                "P5011_MERSCOPE",
                0.01018,
                0.01,
                "<=",
                False,
                True,
            ),
            row("H2", "proseg_hybrid", "P7513_MERSCOPE", 0.02, 0.01, "<=", False, True),
            row("H17/H2", "reseg", "P1212_MERSCOPE", 0.01036, 0.01, "<=", False, True),
            row(
                "H8/warning",
                "proseg_hybrid",
                "P1212_MERSCOPE",
                0.1482,
                None,
                "==",
                False,
                True,
            ),
            row("H8", "proseg_hybrid", "P1212_MERSCOPE", None, None, "==", True, True),
            row("H4", "proseg_hybrid", "P1212_MERSCOPE", 4.0, 6.0, ">=", False, True),
            row(
                "H4[m4_resolve_heldout]",
                "proseg_hybrid",
                "P1212_MERSCOPE",
                5.0,
                6.0,
                ">=",
                False,
                True,  # run_acceptance reports it whatever the input says (D6)
            ),
            row(
                "H5",
                "proseg_hybrid",
                "P7513_MERSCOPE",
                0.0,
                0.02,
                "<=",
                True,
                True,
                "confident COP supercluster share",
            ),
            row(
                "H5",
                "proseg_hybrid",
                "P7513_MERSCOPE",
                0.0101,
                0.10,
                "<=",
                True,
                True,
                "confident broad OPC share; COP-derived share 0.5",
            ),
            row(
                "H10",
                "proseg_hybrid",
                "P7513_MERSCOPE",
                0.991,
                0.70,
                ">=",
                True,
                True,
                "48854 disputes",
            ),
            row("H7", "proseg_hybrid", "P7113_XENIUM", 0.30, 0.39, ">=", False, False),
        ]
    )


def _samples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segmentation": "proseg_hybrid",
                "sample_id": "P1212_MERSCOPE",
                "gate_level": "broad_only",
                "gate_warning": True,
            },
            {
                "segmentation": "proseg_hybrid",
                "sample_id": "P5011_MERSCOPE",
                "gate_level": "broad_only",
                "gate_warning": True,
            },
        ]
    )


def _h2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segmentation": "proseg_hybrid",
                "sample_id": "P5011_MERSCOPE",
                "node": "all",
                "n_broad_confident": 0,
                "n_supercluster_confident": 0,
            },
            {
                "segmentation": "reseg",
                "sample_id": "P1212_MERSCOPE",
                "node": "all",
                "n_broad_confident": 2,
                "n_supercluster_confident": 0,
            },
        ]
    )


def _enrichment(ceiling_gap: float = 0.001) -> pd.DataFrame:
    rows = []
    for cls, passes in (
        ("Neurons", True),
        ("Astrocytes", True),
        ("Oligodendrocyte precursors", True),
        ("Fibroblasts", True),
        ("Oligodendrocytes", False),
        ("Microglia", False),
        ("Vascular cells", False),
    ):
        rows.append(
            {
                "label_set": "m4_resolve_heldout_whb_only",
                "pair": "P1212",
                "platform": "MERSCOPE",
                "broad_class": cls,
                "fold": 8.0,
                "auroc": 0.8 if passes else 0.64,
                "auroc_ceiling": 0.9 if passes else 0.64 + ceiling_gap,
                "passes": passes,
            }
        )
    return pd.DataFrame(rows)


def _report(**values: Any) -> list[dict]:
    """acceptance_metrics.json records of one pair x segmentation."""
    records = []
    for (criterion, name, sample, extra), value in values.get("records", {}).items():
        records.append(
            {
                "criterion": criterion,
                "name": name,
                "sample_id": sample,
                **dict(extra),
                "value": value,
                "status": "measured",
            }
        )
    return records


def _scored(
    acceptance: ModuleType,
    reports: dict | None = None,
    enrichment: pd.DataFrame | None = None,
    referee: pd.DataFrame | None = None,
) -> dict:
    rows = acceptance.criteria_rows(
        _criteria_table(),
        _samples(),
        _h2(),
        _enrichment() if enrichment is None else enrichment,
        reports or {},
        referee,
    )
    return {
        (row.criterion, row.segmentation, row.dataset, row.note[:12]): row
        for row in rows
    }


def _get(rows: dict, criterion: str, dataset: str, seg: str = "proseg_hybrid") -> Any:
    return next(
        row
        for key, row in rows.items()
        if key[0] == criterion and key[2] == dataset and key[1] == seg
    )


def test_criteria_rows_apply_the_exceptions_only_inside_their_scope(
    acceptance: ModuleType,
) -> None:
    rows = _scored(acceptance)
    assert _get(rows, "H1", "P7513").verdict == "PASS"
    assert _get(rows, "H2", "P5011_MERSCOPE").verdict == "EXCEPTION (D5)"
    assert _get(rows, "H2", "P7513_MERSCOPE").verdict == "FAIL-OUTSIDE"
    # D5 text: no flagged cell with a confident broad label; 2 here
    reseg = _get(rows, "H17/H2", "P1212_MERSCOPE", "reseg")
    assert (
        reseg.verdict == "EXCEPTION-RECHECK (D5)"
        and "confident broad" in reseg.scope_check
    )
    assert _get(rows, "H8/warning", "P1212_MERSCOPE").verdict == "EXCEPTION (D3)"
    assert _get(rows, "H4", "P1212_MERSCOPE").verdict == "EXCEPTION (D7)"
    reported = _get(rows, "H4[m4_resolve_heldout]", "P1212_MERSCOPE")
    assert reported.scored is False and reported.verdict == "INFO fail"
    assert _get(rows, "H7", "P7113_XENIUM").verdict == "INFO fail"
    moved = _scored(acceptance, enrichment=_enrichment(ceiling_gap=0.05))
    assert _get(moved, "H4", "P1212_MERSCOPE").verdict == "EXCEPTION-RECHECK (D7)"
    assert all(
        not acceptance.back_to_user(row)
        or row.verdict.startswith(("FAIL-OUTSIDE", "EXCEPTION-RECHECK"))
        for row in rows.values()
    )


def test_criteria_rows_cross_check_the_report(acceptance: ModuleType) -> None:
    good = {
        ("P7513", "proseg_hybrid"): [
            {
                "criterion": "H1",
                "name": "broad_jsd",
                "region": "whole_section",
                "kind": "set_a:soft",
                "value": 0.129000,
            },
            {
                "criterion": "H1",
                "name": "broad_jsd",
                "region": "shared_mask",
                "kind": "set_a:soft",
                "value": 0.5,
            },
            {
                "criterion": "H2",
                "name": "flag_implausible_share",
                "sample_id": "P7513_MERSCOPE",
                "value": 0.02,
            },
            {
                "criterion": "H5",
                "name": "confident_cop_supercluster_share",
                "sample_id": "P7513_MERSCOPE",
                "value": 0.0,
            },
            {
                "criterion": "H5",
                "name": "confident_broad_opc_share",
                "sample_id": "P7513_MERSCOPE",
                "value": 0.0101,
            },
        ],
        ("P1212", "proseg_hybrid"): [
            {
                "criterion": "H8",
                "name": "gate_level",
                "sample_id": "P1212_MERSCOPE",
                "value": "broad_only",
            },
            {
                "criterion": "H8",
                "name": "gate_warning",
                "sample_id": "P1212_MERSCOPE",
                "value": True,
            },
            {
                "criterion": "H8",
                "name": "gate_broad_coverage_segmented",
                "sample_id": "P1212_MERSCOPE",
                "value": 0.148200,
            },
        ],
    }
    rows = _scored(acceptance, reports=good)
    assert _get(rows, "H1", "P7513").crosscheck == "ok"
    assert _get(rows, "H2", "P7513_MERSCOPE").crosscheck == "ok"
    assert [row.crosscheck for key, row in rows.items() if key[0] == "H5"] == [
        "ok",
        "ok",
    ]
    assert _get(rows, "H8", "P1212_MERSCOPE").crosscheck == "ok"
    assert _get(rows, "H8/warning", "P1212_MERSCOPE").crosscheck == "ok"
    bad = {key: [dict(record) for record in value] for key, value in good.items()}
    bad[("P7513", "proseg_hybrid")][0]["value"] = 0.1291
    bad[("P1212", "proseg_hybrid")][1]["value"] = False
    rows = _scored(acceptance, reports=bad)
    h1 = _get(rows, "H1", "P7513")
    assert h1.crosscheck.startswith("CROSSCHECK-MISMATCH") and acceptance.back_to_user(
        h1
    )
    assert h1.verdict == "PASS"
    assert _get(rows, "H8", "P1212_MERSCOPE").crosscheck.startswith(
        "CROSSCHECK-MISMATCH"
    )
    missing = {("P7513", "proseg_hybrid"): good[("P7513", "proseg_hybrid")][1:]}
    rows = _scored(acceptance, reports=missing)
    assert "has no H1" in _get(rows, "H1", "P7513").crosscheck


def test_crosscheck_allows_the_reports_six_digit_rounding(
    acceptance: ModuleType,
) -> None:
    # report_model._clean stores float(f"{x:.6g}"): up to 5e-6 relative.
    values = 10 ** np.random.default_rng(0).uniform(-4, 0, 20_000)
    assert all(acceptance._close(v, float(f"{v:.6g}")) for v in values)
    assert acceptance._close(0.0101767267, 0.0101767)  # stage A3 H2 P5011_MERSCOPE
    assert acceptance._close(0.1771234567, 0.177123)
    # a difference in the first five significant digits is a mismatch
    assert not acceptance._close(0.0101767267, 0.0101769)
    assert not acceptance._close(0.1482, 0.14822)


def test_unrounded_criteria_values_match_the_rounded_report(
    acceptance: ModuleType,
) -> None:
    table = _criteria_table()
    at = (table["criterion"] == "H2") & (table["dataset"] == "P5011_MERSCOPE")
    table.loc[at, "value"] = 0.0101767267
    record = {
        "criterion": "H2",
        "name": "flag_implausible_share",
        "sample_id": "P5011_MERSCOPE",
        "value": 0.0101767,
    }
    reports = {("P5011", "proseg_hybrid"): [record]}

    def h2() -> Any:
        rows = acceptance.criteria_rows(
            table, _samples(), _h2(), _enrichment(), reports, None
        )
        return next(
            row
            for row in rows
            if row.criterion == "H2" and row.dataset == "P5011_MERSCOPE"
        )

    row = h2()
    assert row.crosscheck == "ok"
    assert row.verdict == "EXCEPTION (D5)" and not acceptance.back_to_user(row)
    record["value"] = 0.0101777
    row = h2()
    assert row.crosscheck.startswith("CROSSCHECK-MISMATCH")
    assert acceptance.back_to_user_reason(row) == "cross-check mismatch"


def test_first_measured_failures_go_back_even_when_not_scored(
    acceptance: ModuleType,
) -> None:
    def row(criterion: str, passes: bool | None, scored: bool = False) -> Any:
        return acceptance.finalize(
            acceptance.Row(
                criterion, "P7113", "all", "P7113", 1.0, "==", 2.0, passes, scored, "x"
            )
        )

    for criterion in (
        "H6/broad",
        "H6/pooled",
        "H12",
        "H13/id_sets",
        "H13/depth_violins",
        "H14/annotation_wall",
        "H14/map_peak_rss",
        "H15/seed1",
    ):
        failed = row(criterion, False)
        assert failed.verdict == "INFO fail"
        assert acceptance.back_to_user(failed), criterion
        assert "first-measured" in acceptance.back_to_user_reason(failed)
        assert not acceptance.back_to_user(row(criterion, True))
        assert not acceptance.back_to_user(row(criterion, None))
    held_out = row("H2", False)
    assert held_out.verdict == "INFO fail" and not acceptance.back_to_user(held_out)
    assert acceptance.back_to_user_reason(row("H13/id_sets", False, True)) == (
        "verdict FAIL-OUTSIDE"
    )


def test_criteria_rows_read_only_the_new_vs_legacy_referee_rows(
    acceptance: ModuleType,
) -> None:
    rows = pd.DataFrame(
        [
            {
                "segmentation": "proseg_hybrid",
                "sample_id": "P7513_MERSCOPE",
                "first": first,
                "second": second,
                "first_wins": wins,
            }
            for first, second, wins in (
                ("new confident (RESOLVE)", "legacy broad_class", 0.991),
                ("WHB argmax", "SEA-AD argmax", 0.5),
                ("new confident (RESOLVE)", "SEA-AD argmax", 0.6),
            )
        ]
    )
    row = _get(_scored(acceptance, referee=rows), "H10", "P7513_MERSCOPE")
    assert row.crosscheck.startswith("ok")


def test_criteria_rows_compare_h10_with_marker_referee(acceptance: ModuleType) -> None:
    same = pd.DataFrame(
        [
            {
                "segmentation": "proseg_hybrid",
                "sample_id": "P7513_MERSCOPE",
                "first_wins": 0.991,
            }
        ]
    )
    assert _get(
        _scored(acceptance, referee=same), "H10", "P7513_MERSCOPE"
    ).crosscheck.startswith("ok")
    other = same.assign(first_wins=0.95)
    row = _get(_scored(acceptance, referee=other), "H10", "P7513_MERSCOPE")
    assert row.crosscheck.startswith("CROSSCHECK-MISMATCH") and acceptance.back_to_user(
        row
    )
    absent = same.assign(sample_id="P7513_XENIUM")
    assert (
        "not in marker_referee"
        in _get(_scored(acceptance, referee=absent), "H10", "P7513_MERSCOPE").crosscheck
    )


# ---------------------------------------------------------------------------
# run_acceptance.py: H12 and H13


def _h12_records(xenium_tiles: bool = True) -> list[dict]:
    records = []
    for platform, tiles in (("MERSCOPE", True), ("XENIUM", xenium_tiles)):
        sid = f"P7513_{platform}"
        records += [
            {
                "criterion": "H12",
                "name": "depth_ordering_passes",
                "platform": platform,
                "sample_id": sid,
                "kind": None,
                "value": True,
                "status": "measured",
            },
            {
                "criterion": "H12",
                "name": "depth_ordering_passes",
                "platform": platform,
                "sample_id": sid,
                "kind": "square_tile_500um",
                "value": tiles,
                "status": "measured",
            },
            {
                "criterion": "H12",
                "name": "oligodendrocyte_wm_minus_gm_share",
                "platform": platform,
                "sample_id": sid,
                "kind": "outside_ribbon_proxy",
                "value": 0.5,
                "ci_low": 0.45,
                "ci_high": 0.55,
                "status": "measured",
            },
        ]
    records += [
        {
            "criterion": "H12",
            "name": "depth_ci_method",
            "value": "tangential_block_500um",
            "status": "measured",
        },
        {
            "criterion": "H12",
            "name": "depth_ordering_replicated",
            "kind": "square_tile_500um",
            "value": xenium_tiles,
            "status": "measured",
        },
    ]
    return records


def test_h12_scores_the_square_tiles_and_reports_the_blocks_beside(
    acceptance: ModuleType,
) -> None:
    rows, scores = acceptance.h12_rows(
        {("P7513", "proseg_hybrid"): _h12_records()}, ["P7513", "P1212"]
    )
    by_pair = {row.pair: row for row in rows}
    assert by_pair["P7513"].verdict == "PASS"
    assert by_pair["P1212"].verdict == "NOT_AVAILABLE"
    assert scores[0]["ci_scored"] == "square_tile_500um"
    rows, _ = acceptance.h12_rows(
        {("P7513", "proseg_hybrid"): _h12_records(xenium_tiles=False)}, ["P7513"]
    )
    assert rows[0].verdict == "FAIL-OUTSIDE" and rows[0].passes is False
    assert "beside (tangential blocks, not scored)" in rows[0].note
    rows, _ = acceptance.h12_rows(
        {("P7513", "reseg"): _h12_records(xenium_tiles=False)}, ["P7513"]
    )
    reseg = next(row for row in rows if row.segmentation == "reseg")
    assert reseg.scored is False and reseg.verdict == "INFO fail"


def _report_with_mender(
    path: Path,
    policy: str,
    pair: str = "P7513",
    status: str = "complete",
    ids: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        {
            "criterion": "H13",
            "name": "table_ids_equal_clustered",
            "sample_id": f"{pair}_{platform}",
            "value": ids,
        }
        for platform in ("MERSCOPE", "XENIUM")
    ]
    samples = {
        f"{pair}_{platform}": {
            "mender": {"status": status, "unassigned_state_policy": policy}
        }
        for platform in ("MERSCOPE", "XENIUM")
    }
    path.write_text(
        json.dumps({"metrics": metrics, "provenance": {"samples": samples}})
    )


def _h13_tree(
    acceptance: ModuleType, runs: Path, *, state: bool = True, violins: bool = True
) -> dict:
    reports = {}
    for seg in acceptance.SEGMENTATIONS:
        path = runs / acceptance.REPORT_PATH.format(pair="P7513", seg=seg)
        _report_with_mender(path, "exclude_from_features")
        reports[("P7513", seg)] = json.loads(path.read_text())["metrics"]
        for platform in ("merscope", "xenium"):
            depth = runs / acceptance.DEPTH_DIR.format(
                pair="P7513", seg=seg, platform=platform
            )
            depth.mkdir(parents=True, exist_ok=True)
            if violins or seg != "reseg":
                (depth / "violin_broad.png").write_bytes(b"")
    if state:
        _report_with_mender(
            runs / acceptance.STATE_REPORT_PATH.format(pair="P7513"), "state"
        )
    return reports


def test_h13_parts_pass_on_a_complete_run(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    reports = _h13_tree(acceptance, tmp_path)
    rows = {
        row.criterion: row
        for row in acceptance.h13_rows(
            tmp_path, reports, {"passes": True, "failing_groups": []}, ["P7513"]
        )
    }
    assert rows["H13/mender"].verdict == "PASS"
    assert rows["H13/depth_violins"].verdict == "PASS"
    assert rows["H13/id_sets"].verdict == "PASS"
    assert rows["H13/legacy_untouched"].verdict == "PASS"
    assert rows["H13/viewer"].verdict == "NOT_AVAILABLE"


def test_h13_parts_need_both_policies_violins_and_untouched_tables(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    reports = _h13_tree(acceptance, tmp_path, state=False, violins=False)
    rows = {
        row.criterion: row
        for row in acceptance.h13_rows(
            tmp_path, reports, {"passes": False, "failing_groups": ["x"]}, ["P7513"]
        )
    }
    assert rows["H13/mender"].verdict == "NOT_AVAILABLE"
    assert rows["H13/depth_violins"].verdict == "FAIL-OUTSIDE"
    assert "reseg/merscope" in rows["H13/depth_violins"].note
    assert rows["H13/legacy_untouched"].verdict == "FAIL-OUTSIDE"
    wrong = tmp_path / "wrong"
    reports = _h13_tree(acceptance, wrong)
    _report_with_mender(
        wrong / acceptance.STATE_REPORT_PATH.format(pair="P7513"),
        "exclude_from_features",
    )
    rows = {
        row.criterion: row
        for row in acceptance.h13_rows(wrong, reports, None, ["P7513"])
    }
    assert (
        rows["H13/mender"].verdict == "FAIL-OUTSIDE"
    )  # the second policy must be "state"
    assert rows["H13/legacy_untouched"].verdict == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# run_acceptance.py: H6


def _e2_cells(p_broad: float = 0.9, wrong_astro: int = 0) -> pd.DataFrame:
    rows = []
    for depth in (10, 15, 30, 60, 120):
        for cls, sup, n in (
            ("Exc", "Upper-layer intratelencephalic", 100),
            ("Astro", "Astrocyte", 60),
            ("Immune", "Microglia", 20),
        ):
            for index in range(n):
                call, call_sup = cls, sup
                if cls == "Exc" and index < wrong_astro:
                    call, call_sup = "Astro", "Astrocyte"
                rows.append(
                    {
                        "scenario": "contam",
                        "D": str(depth),
                        "true_broad": cls,
                        "true_super": sup,
                        "ho_broad": call,
                        "ho_super": call_sup,
                        "ho_broad_p": p_broad,
                        "ho_super_p": 0.8,
                    }
                )
    frame = pd.DataFrame(rows)
    other = frame.head(5).assign(scenario="clean", ho_broad="Immune")
    return pd.concat([frame, other], ignore_index=True)


def test_soft_mass_adds_the_assigned_node_and_runner_ups(
    acceptance: ModuleType,
) -> None:
    labels = pd.DataFrame(
        {
            "in_table": [True, True, False],
            "mmc_whb_supercluster_name": ["Astrocyte", "Microglia", "Astrocyte"],
            "mmc_whb_supercluster_bp": [0.7, 0.9, 1.0],
            "mmc_whb_supercluster_runner_up_1_name": ["Microglia", None, "Astrocyte"],
            "mmc_whb_supercluster_runner_up_1_bp": [0.2, np.nan, 0.0],
        }
    )
    mass = acceptance.soft_supercluster_mass(labels)
    assert mass.to_dict() == pytest.approx({"Astrocyte": 0.7, "Microglia": 1.1})


def test_h6_reweights_the_simulation_to_each_dataset(acceptance: ModuleType) -> None:
    cells = _e2_cells(wrong_astro=6)[lambda f: f["scenario"] == "contam"]
    mapping = acceptance.truth_map(cells)
    assert mapping["Astrocyte"] == "Astro"
    # 6 of 100 Exc cells called Astro: unweighted Astro precision 60 / 66
    unweighted = acceptance.h6_precision(cells, None)
    astro = next(
        r
        for r in unweighted
        if r["level"] == "broad" and r["class"] == "Astro" and r["depth"] == 15
    )
    assert astro["precision"] == pytest.approx(60 / 66) and astro["passes"] is True
    # Immune has 20 test cells: never a class row
    assert not any(r["class"] == "Immune" for r in unweighted)
    pooled = next(
        r
        for r in unweighted
        if r["level"] == "broad" and r["class"] == "pooled" and r["depth"] == 15
    )
    # the Exc and Astro calls together: 154 correct of 160 (Immune calls left out)
    assert pooled["precision"] == pytest.approx(154 / 160)
    assert pooled["n_confident_calls"] == 160
    # a dataset with few astrocytes: the Exc errors weigh more
    few_astro = acceptance.dataset_composition(
        pd.Series(
            {"Upper-layer intratelencephalic": 0.9, "Astrocyte": 0.1, "Splatter": 5.0}
        ),
        mapping,
    )
    assert few_astro["broad"].to_dict() == pytest.approx({"Exc": 0.9, "Astro": 0.1})
    weighted = acceptance.h6_precision(cells, few_astro)
    astro = next(
        r
        for r in weighted
        if r["level"] == "broad" and r["class"] == "Astro" and r["depth"] == 15
    )
    w_astro, w_exc = 0.1 / (60 / 180), 0.9 / (100 / 180)
    assert astro["precision"] == pytest.approx(
        60 * w_astro / (60 * w_astro + 6 * w_exc)
    )
    assert astro["passes"] is False
    assert {r["depth"] for r in weighted if r["level"] == "supercluster"} == {
        30,
        60,
        120,
    }
    assert {r["depth"] for r in weighted if r["level"] == "broad"} == {15, 30, 60, 120}


def test_h6_uses_the_v1_raw_thresholds_and_the_confident_broad_chain(
    acceptance: ModuleType,
) -> None:
    at = _e2_cells(p_broad=0.73 - 5e-7)[lambda f: f["scenario"] == "contam"]
    rows = acceptance.h6_precision(at, None)
    assert all(r["n_confident_calls"] > 0 for r in rows)
    below = _e2_cells(p_broad=0.72)[lambda f: f["scenario"] == "contam"]
    rows = acceptance.h6_precision(below, None)
    assert all(r["n_confident_calls"] == 0 and r["evaluable"] is False for r in rows)
    assert all(r["level"] in ("broad", "supercluster") for r in rows)


def test_h6_rows_fail_a_dataset_outside_the_targets(acceptance: ModuleType) -> None:
    cells = _e2_cells(wrong_astro=6)
    masses = {
        "P7513_MERSCOPE": pd.Series(
            {"Upper-layer intratelencephalic": 0.5, "Astrocyte": 0.5}
        ),
        "P1212_XENIUM": pd.Series(
            {"Upper-layer intratelencephalic": 0.9, "Astrocyte": 0.1}
        ),
    }
    rows, detail = acceptance.h6_rows(cells, masses)
    by = {row.dataset: row for row in rows if row.criterion == "H6"}
    assert by["P7513_MERSCOPE"].verdict == "PASS"
    info = {row.dataset: row for row in rows if row.criterion == "H6/pooled"}
    assert info["P1212_XENIUM"].scored is False
    assert info["P1212_XENIUM"].verdict in ("INFO pass", "INFO fail")
    assert (
        by["P1212_XENIUM"].verdict == "FAIL-OUTSIDE"
        and "broad Astro D15" in by["P1212_XENIUM"].note
    )
    assert set(detail["dataset"]) == {"unweighted", "P7513_MERSCOPE", "P1212_XENIUM"}


# ---------------------------------------------------------------------------
# run_acceptance.py: H15


def _labels_table(
    status: list[str], names: list[str], ids: list[str] | None = None
) -> pd.DataFrame:
    ids = ids or [f"c{i}" for i in range(len(status))]
    return pd.DataFrame(
        {
            "cell_id": ids,
            "in_table": True,
            "ct_broad_status": status,
            "ct_broad_name": names,
        }
    )


def test_seed_change_counts_status_and_name_changes(acceptance: ModuleType) -> None:
    seed0 = _labels_table(
        ["confident"] * 4 + ["unresolved"],
        ["Neurons", "Neurons", "Astrocytes", "Microglia", "Neurons"],
    )
    seed1 = _labels_table(
        ["confident", "unresolved", "confident", "confident", "confident"],
        ["Neurons", "Neurons", "Microglia", "Microglia", "Neurons"],
    )
    change = acceptance.seed_change(seed0, seed1)
    # c1 loses its label, c2 changes name, c4 gains one: 3 changes over 4 labels
    assert change["n_changed"] == 3 and change["n_confident_seed0"] == 4
    assert change["share_changed"] == pytest.approx(0.75)
    assert acceptance.seed_change(seed0, seed0)["n_changed"] == 0


def test_h15_compare_finds_identical_and_differing_tables(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    table = _labels_table(["confident"] * 3, ["Neurons", "Astrocytes", "Microglia"])
    for run in ("pipeline", "rerun", "seed1"):
        for platform in ("merscope", "xenium"):
            directory = tmp_path / run / "resolve" / platform
            directory.mkdir(parents=True)
            frame = table.copy()
            if run == "seed1" and platform == "xenium":
                frame.loc[0, "ct_broad_name"] = "Microglia"
            frame.to_parquet(
                directory / f"P7513_{platform.upper()}_celltype_labels.parquet"
            )
            mapped = tmp_path / run / "map" / platform
            mapped.mkdir(parents=True)
            frame[["cell_id"]].assign(bp=0.5).to_parquet(
                mapped / f"P7513_{platform.upper()}_mmc_whb.parquet"
            )
    record = acceptance.h15_compare(
        "P7513",
        tmp_path / "pipeline/map",
        tmp_path / "pipeline/resolve",
        tmp_path / "rerun/map",
        tmp_path / "rerun/resolve",
        tmp_path / "seed1/resolve",
    )
    assert record["identical"] is True and len(record["identical_files"]) == 4
    assert (
        record["seed1"]["n_changed"] == 1 and record["seed1"]["n_confident_seed0"] == 6
    )
    rows = {
        row.criterion: row
        for row in acceptance.h15_rows({"P7513": record})
        if row.pair == "P7513"
    }
    assert rows["H15/identical_rerun"].verdict == "PASS"
    assert rows["H15/seed1"].verdict == "FAIL-OUTSIDE"  # 1 / 6 > 1%
    changed = table.assign(ct_broad_name=["Neurons", "Neurons", "Microglia"])
    changed.to_parquet(
        tmp_path / "rerun/resolve/merscope/P7513_MERSCOPE_celltype_labels.parquet"
    )
    record = acceptance.h15_compare(
        "P7513",
        tmp_path / "pipeline/map",
        tmp_path / "pipeline/resolve",
        tmp_path / "rerun/map",
        tmp_path / "rerun/resolve",
        None,
    )
    assert record["identical"] is False and "seed1" not in record
    rows = acceptance.h15_rows({"P7513": record})
    assert [row.verdict for row in rows if row.pair == "P7513"] == [
        "FAIL-OUTSIDE",
        "NOT_AVAILABLE",
    ]
    assert all(row.verdict == "NOT_AVAILABLE" for row in rows if row.pair == "P1212")


def test_h15_seed_threshold_is_one_percent(acceptance: ModuleType) -> None:
    record = {
        "identical": True,
        "identical_files": [{"file": "a", "equal": True}],
        "seed1": {"share_changed": 0.01, "n_changed": 1, "n_confident_seed0": 100},
    }
    rows = {
        row.criterion: row
        for row in acceptance.h15_rows({"P7513": record})
        if row.pair == "P7513"
    }
    assert rows["H15/seed1"].verdict == "PASS"
    record["seed1"]["share_changed"] = 0.0101
    rows = {
        row.criterion: row
        for row in acceptance.h15_rows({"P7513": record})
        if row.pair == "P7513"
    }
    assert rows["H15/seed1"].verdict == "FAIL-OUTSIDE"


def test_h15_failed_rerun_is_not_measured_not_a_failure(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    table = _labels_table(["confident"] * 2, ["Neurons", "Astrocytes"])
    for platform in ("merscope", "xenium"):
        for part in ("resolve", "map"):
            directory = tmp_path / "pipeline" / part / platform
            directory.mkdir(parents=True)
            table.to_parquet(
                directory / f"P7513_{platform.upper()}_celltype_labels.parquet"
            )
    (tmp_path / "rerun/map").mkdir(parents=True)  # the task failed: no output
    record = acceptance.h15_compare(
        "P7513",
        tmp_path / "pipeline/map",
        tmp_path / "pipeline/resolve",
        tmp_path / "rerun/map",
        tmp_path / "rerun/resolve",
        tmp_path / "seed1/resolve",
        notes=["rerun MAP exit 1"],
    )
    assert record["identical"] is None and "identical_files" not in record
    assert "re-run MAP" in record["rerun_problem"]
    assert "holds no parquet" in record["rerun_problem"]
    assert "re-run RESOLVE" in record["rerun_problem"]
    assert record["seed1"] is None and "labels missing" in record["seed1_problem"]
    assert record["notes"] == ["rerun MAP exit 1"]
    rows = acceptance.h15_rows({"P7513": record})
    rows = [row for row in rows if row.pair == "P7513"]
    assert [row.verdict for row in rows] == ["NOT_AVAILABLE", "NOT_AVAILABLE"]
    assert all(acceptance.back_to_user(row) for row in rows)
    assert "re-run MAP" in rows[0].note and "labels missing" in rows[1].note
    out = tmp_path / "h15_P7513.json"
    assert (
        acceptance.main(
            [
                "h15",
                "--pair",
                "P7513",
                "--pipeline-map",
                str(tmp_path / "pipeline/map"),
                "--pipeline-resolve",
                str(tmp_path / "pipeline/resolve"),
                "--note",
                "rerun MAP exit 1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    written = json.loads(out.read_text())
    assert written["notes"] == ["rerun MAP exit 1"] and "identical" not in written


# ---------------------------------------------------------------------------
# run_acceptance.py: H18 (D4) and the end-to-end score


def _draw_spread(
    directory: Path,
    *,
    opc_missing: str = "15,120",
    raised_extra: bool = False,
    workers: int = 8,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "draw_spread_run.json").write_text(
        json.dumps(
            {
                "bundle_build_hash": SET_A,
                "draws": ["c0e0"],
                "scored_draw_check": {"reproduces": True},
                "n_processors": workers,
                "self_map_n_processors": 8,
            }
        )
    )
    base = {
        "draw": "c0e0",
        "d_max": 250.0,
        "floor": 10.0,
        "floor_family": "human_set_a",
        "required": True,
    }
    pd.DataFrame(
        [
            {
                **base,
                "dataset": "PREP[MERSCOPE floors]",
                "level": "broad",
                "class": "Exc",
                "expected": "15",
                "emitted": "15",
                "missing": "",
                "passes": True,
            },
            {
                **base,
                "dataset": "P7513_MERSCOPE",
                "level": "broad",
                "class": "OPC",
                "expected": "15,30,60,120",
                "emitted": "30,60",
                "missing": opc_missing,
                "passes": False,
            },
            {
                **base,
                "dataset": "P7513_MERSCOPE",
                "level": "broad",
                "class": "Astro",
                "expected": "15",
                "emitted": "",
                "missing": "15",
                "passes": False,
            },
            {
                **base,
                "dataset": "P7513_MERSCOPE",
                "level": "broad",
                "class": "Exc",
                "expected": "15",
                "emitted": "15",
                "missing": "",
                "passes": True,
                "required": False,
            },
        ]
    ).to_csv(directory / "draw_spread_h18.csv", index=False)
    raised = [
        {
            "draw": "c0e0",
            "dataset": "PREP[MERSCOPE floors]",
            "level": "broad",
            "class": "OPC",
            "depth": 15,
        },
        {
            "draw": "c0e0",
            "dataset": "P7513_MERSCOPE",
            "level": "broad",
            "class": "OPC",
            "depth": 15,
        },
    ]
    if raised_extra:
        raised.append(
            {
                "draw": "c0e0",
                "dataset": "P7513_MERSCOPE",
                "level": "broad",
                "class": "Exc",
                "depth": 60,
            }
        )
    pd.DataFrame(raised).to_csv(directory / "draw_spread_would_raise.csv", index=False)
    pd.DataFrame(
        [{"dataset": "P7513_MERSCOPE", "metric": "H7", "scored_value": 0.6}]
    ).to_csv(directory / "draw_spread.csv", index=False)


REFERENCE = pd.DataFrame(
    [{"dataset": "P7513_MERSCOPE", "level": "broad", "class": "OPC", "depth": 15}]
)


def test_h18_rows_check_d4s_scope(acceptance: ModuleType, tmp_path: Path) -> None:
    _draw_spread(tmp_path / "ok")
    rows, problems = acceptance.h18_rows(tmp_path / "ok", REFERENCE)
    by = {(row.criterion, row.dataset): row for row in rows}
    assert problems == []
    assert by[("H18/broad", "P7513_MERSCOPE OPC")].verdict == "EXCEPTION (D4)"
    assert by[("H18/broad", "P7513_MERSCOPE Astro")].verdict == "FAIL-OUTSIDE"
    assert by[("H18/broad", "PREP[MERSCOPE floors] Exc")].verdict == "PASS"
    assert ("H18/broad", "P7513_MERSCOPE Exc") not in by  # not required
    assert by[("H18/would_raise", "PREP[MERSCOPE floors]")].verdict == "EXCEPTION (D4)"
    assert by[("H18/would_raise", "P7513_MERSCOPE")].verdict == "EXCEPTION (D4)"
    assert by[("H18/draw_check", "draw_spread_run.json")].verdict == "PASS"
    _draw_spread(tmp_path / "moved", opc_missing="15,30", raised_extra=True, workers=6)
    rows, problems = acceptance.h18_rows(tmp_path / "moved", REFERENCE)
    by = {(row.criterion, row.dataset): row for row in rows}
    assert by[("H18/broad", "P7513_MERSCOPE OPC")].verdict == "EXCEPTION-RECHECK (D4)"
    assert by[("H18/would_raise", "P7513_MERSCOPE")].verdict == "EXCEPTION-RECHECK (D4)"
    assert by[("H18/draw_check", "draw_spread_run.json")].verdict == "FAIL-OUTSIDE"
    assert problems and "workers 6" in problems[0]


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (None, "records no scored_draw_check"),
        ({"skipped": True}, "skipped the scored-draw check"),
        ({"reproduces": False, "calls_changed": 0.033}, "does not reproduce"),
    ],
)
def test_h18_draw_check_needs_the_scored_draws_control(
    acceptance: ModuleType, tmp_path: Path, check: dict | None, message: str
) -> None:
    _draw_spread(tmp_path)
    path = tmp_path / "draw_spread_run.json"
    run = json.loads(path.read_text())
    if check is None:
        run.pop("scored_draw_check")
    else:
        run["scored_draw_check"] = check
    path.write_text(json.dumps(run))
    rows, problems = acceptance.h18_rows(tmp_path, REFERENCE)
    assert len(problems) == 1 and message in problems[0]
    draw = next(row for row in rows if row.criterion == "H18/draw_check")
    assert draw.verdict == "FAIL-OUTSIDE" and acceptance.back_to_user(draw)


def _score_inputs(
    acceptance: ModuleType, root: Path, *, bundle_hash: str = SET_A
) -> list[str]:
    runs = root / "runs"
    criteria = root / "criteria"
    criteria.mkdir(parents=True)
    _criteria_table().to_csv(criteria / "criteria_table.csv", index=False)
    _samples().to_csv(criteria / "criteria_samples.csv", index=False)
    _h2().to_csv(criteria / "h2_breakdown.csv", index=False)
    _enrichment().to_csv(criteria / "heldout_enrichment_m4.csv", index=False)
    _h13_tree(acceptance, runs)
    report = runs / acceptance.REPORT_PATH.format(pair="P7513", seg="proseg_hybrid")
    payload = json.loads(report.read_text())
    payload["metrics"] += [
        {
            "criterion": "H1",
            "name": "broad_jsd",
            "region": "whole_section",
            "kind": "set_a:soft",
            "value": 0.129,
        },
        {
            "criterion": "H2",
            "name": "flag_implausible_share",
            "sample_id": "P7513_MERSCOPE",
            "value": 0.02,
        },
        {
            "criterion": "H5",
            "name": "confident_cop_supercluster_share",
            "sample_id": "P7513_MERSCOPE",
            "value": 0.0,
        },
        {
            "criterion": "H5",
            "name": "confident_broad_opc_share",
            "sample_id": "P7513_MERSCOPE",
            "value": 0.0101,
        },
    ]
    report.write_text(json.dumps(payload))
    labels = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "in_table": [True, True],
            "ct_broad_status": ["confident", "confident"],
            "ct_broad_name": ["Neurons", "Astrocytes"],
            "mmc_whb_supercluster_name": [
                "Upper-layer intratelencephalic",
                "Astrocyte",
            ],
            "mmc_whb_supercluster_bp": [0.9, 0.9],
        }
    )
    for platform in ("merscope", "xenium"):
        path = (
            runs
            / "P7513/proseg_hybrid/annotation_resolve/annotation_resolve_out"
            / platform
        )
        path.mkdir(parents=True)
        labels.to_parquet(path / f"P7513_{platform.upper()}_celltype_labels.parquet")
    ref = _bundle(root, bundle_hash, "2026-09-30T10:52:48+00:00")
    rows = [
        _task("ANNOTATE_PANEL", f"P7513:{seg}", "1m")
        for seg in acceptance.SEGMENTATIONS
    ] + [_task("ANNOTATE_REFERENCE_PREP", "whb:1", "12s")]
    trace = _trace(root / "trace.tsv", rows)
    legacy_dir = root / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "legacy_untouched.json").write_text(
        json.dumps({"passes": True, "failing_groups": [], "n_legacy_tables": 3})
    )
    _draw_spread(root / "draw")
    reference = root / "d4.csv"
    REFERENCE.to_csv(reference, index=False)
    cells = root / "cells.csv"
    _e2_cells().to_csv(cells, index=False)
    return [
        "score",
        "--runs-root",
        str(runs),
        "--criteria",
        str(criteria),
        "--legacy",
        str(legacy_dir),
        "--draw-spread",
        str(root / "draw"),
        "--d4-reference",
        str(reference),
        "--trace",
        str(trace),
        "--prep-refs",
        str(ref.parent.parent),
        "--run-start",
        RUN_START,
        "--e2-cells",
        str(cells),
        "--datasets",
        "P7513",
        "--date",
        "2026-10-01",
        "--out",
        str(root / "out"),
    ]


def test_score_writes_the_summary_and_sends_failures_back(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    argv = _score_inputs(acceptance, tmp_path)
    assert acceptance.main(argv) == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["p5_check"]["ok"] is True
    assert summary["protocol"]["h12_ci_scored"] == "square_tile_500um"
    assert summary["protocol"]["h4_scored_set"] == "m4_resolve_heldout_whb_only"
    verdicts = summary["verdict_counts"]
    assert verdicts["EXCEPTION (D5)"] == 1 and verdicts["EXCEPTION (D3)"] == 1
    back = {
        (row["criterion"], row["dataset"])
        for row in summary["rows"]
        if row["back_to_user"]
    }
    assert ("H2", "P7513_MERSCOPE") in back  # a failure no decision names
    assert ("H15/seed1", "P7513") in back  # not measured: first measurement
    assert ("H13/viewer", "viewer") in back
    assert ("H1", "P7513") not in back
    assert summary["all_scored_rows_pass_or_inside_an_exception"] is False
    assert summary["nothing_back_to_user"] is False
    assert all(
        row["back_to_user_reason"] for row in summary["rows"] if row["back_to_user"]
    )
    assert (tmp_path / "out/summary.html").read_text().startswith("<!doctype html>")
    for name in (
        "criteria_scored.csv",
        "h6_precision.csv",
        "draw_spread.csv",
        "p5_bundles.csv",
    ):
        assert (tmp_path / "out/tables" / name).is_file(), name
    assert any(path.endswith("criteria_table.csv") for path in summary["inputs"])


def test_summary_flags_separate_scored_rows_from_the_rest(
    acceptance: ModuleType,
) -> None:
    unscored = {"back_to_user": True, "scored": False}
    scored = {"back_to_user": True, "scored": True}
    fine = {"back_to_user": False, "scored": True}
    flags = acceptance.summary_flags([fine, unscored], [])
    assert flags == {
        "n_back_to_user": 1,
        "all_scored_rows_pass_or_inside_an_exception": True,
        "nothing_back_to_user": False,
    }
    flags = acceptance.summary_flags([fine], ["no draw spread"])
    assert flags["all_scored_rows_pass_or_inside_an_exception"] is True
    assert flags["nothing_back_to_user"] is False
    flags = acceptance.summary_flags([fine, scored], [])
    assert flags["all_scored_rows_pass_or_inside_an_exception"] is False
    assert acceptance.summary_flags([fine], [])["nothing_back_to_user"] is True


def test_score_scores_nothing_when_the_run_did_not_use_the_d1_bundles(
    acceptance: ModuleType, tmp_path: Path
) -> None:
    argv = _score_inputs(acceptance, tmp_path, bundle_hash="c" * 64)
    assert acceptance.main(argv) == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["p5_check"]["ok"] is False
    assert set(summary["verdict_counts"]) == {"UNSCORED (P5)"}
    assert all(row["back_to_user"] for row in summary["rows"])


def test_run_acceptance_help_runs(acceptance: ModuleType) -> None:
    for argv in (["--help"], ["score", "--help"], ["h15", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            acceptance.main(argv)
        assert excinfo.value.code == 0
