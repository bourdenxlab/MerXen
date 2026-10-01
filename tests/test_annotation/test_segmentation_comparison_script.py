"""Tests of scripts/acceptance/segmentation_comparison.py (M8b; pre-reg §19)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import shapely
from scipy import sparse

from merxen.annotation import composition as co
from merxen.annotation import segmentation_compare as sc
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"


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
def m8b() -> ModuleType:
    return _load("segmentation_comparison")


# ---------------------------------------------------------------------------
# The gene lists (fixed before the first run)


def test_gene_lists_are_sourced_and_fixed(m8b: ModuleType) -> None:
    assert set(m8b.GENE_LISTS) == {"nuclear_retained", "process_localised"}
    for entries in m8b.GENE_LISTS.values():
        symbols = [entry.symbol for entry in entries]
        assert len(symbols) == len(set(symbols))
        assert all("doi:10." in entry.source for entry in entries)
        assert all(entry.evidence for entry in entries)
    process = {entry.symbol for entry in m8b.PROCESS_LOCALISED}
    assert {"MAPT", "MOBP", "AQP4", "GJA1", "PTPRZ1"} <= process
    table = m8b.gene_list_table(["MAPT", "AQP4", "SOX9"])
    on_panel = set(table.loc[table["on_panel"], "symbol"])
    assert on_panel == {"MAPT", "AQP4"}


# ---------------------------------------------------------------------------
# hybrid_matched: thinning the prepared H5AD


def _prepared(path: Path) -> tuple[pd.Series, pd.Series]:
    counts = np.array(
        [
            [20, 10, 0, 3],  # thinned to 12
            [4, 4, 0, 0],  # hybrid 8 < 10: not a table cell
            [5, 5, 2, 1],  # hybrid 12 <= reseg 30: kept
            [30, 0, 5, 0],  # reseg 4: thinned below the table threshold
            [9, 9, 9, 0],  # no reseg object: target 0
        ],
        dtype=np.int64,
    )
    var = pd.DataFrame(index=["GENE1", "GENE2", "GENE3", "Blank-1"])
    obs = pd.DataFrame(
        {
            "total_counts": counts.sum(axis=1),
            "log1p_total_counts": np.log1p(counts.sum(axis=1)),
            "n_genes_by_counts": (counts > 0).sum(axis=1).astype(np.int32),
            "log1p_n_genes_by_counts": np.log1p((counts > 0).sum(axis=1)),
            "control_counts": counts[:, 3].astype(float),
            "pct_control_counts": 0.0,
            "cell_area": np.arange(5, dtype=float),
        },
        index=pd.Index(["1", "2", "3", "4", "5"], name="obs_id"),
    )
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.obsm["spatial"] = np.arange(10, dtype=float).reshape(5, 2)
    adata.write_h5ad(path)
    reseg = pd.Series([12, 9, 30, 4], index=["1", "2", "3", "4"])
    hybrid = pd.Series(counts[:, :3].sum(axis=1), index=["1", "2", "3", "4", "5"])
    return reseg, hybrid


def test_thin_prepared_h5ad_thins_genes_keeps_controls(
    m8b: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "in" / "S_MERSCOPE_prepared.h5ad"
    source.parent.mkdir()
    reseg, hybrid = _prepared(source)
    record = m8b.thin_prepared_h5ad(
        source,
        tmp_path / "out" / "S_MERSCOPE_prepared.h5ad",
        platform="MERSCOPE",
        reseg_totals=reseg,
        hybrid_totals=hybrid,
    )
    before = ad.read_h5ad(source)
    after = ad.read_h5ad(tmp_path / "out" / "S_MERSCOPE_prepared.h5ad")
    x_before = before.X.toarray()
    x_after = after.X.toarray()
    np.testing.assert_array_equal(x_after[:, :3].sum(axis=1), [12, 8, 12, 4, 0])
    np.testing.assert_array_equal(x_after[:, 3], x_before[:, 3])
    assert (x_after <= x_before).all()
    np.testing.assert_array_equal(x_after[[1, 2]], x_before[[1, 2]])
    np.testing.assert_array_equal(after.obs["total_counts"], x_after.sum(axis=1))
    np.testing.assert_array_equal(
        after.obs["n_genes_by_counts"], (x_after > 0).sum(axis=1)
    )
    np.testing.assert_array_equal(after.obs["cell_area"], before.obs["cell_area"])
    np.testing.assert_array_equal(after.obsm["spatial"], before.obsm["spatial"])
    assert list(after.obs_names) == list(before.obs_names)
    assert record["n_thinned"] == 3
    assert record["n_without_reseg_object"] == 1
    assert record["n_table_hybrid"] == 4
    assert record["n_table_matched"] == 2
    assert record["n_control_features"] == 1
    assert not (tmp_path / "out" / "S_MERSCOPE_prepared.h5ad.partial").exists()


def test_thin_prepared_h5ad_is_deterministic(m8b: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "S_MERSCOPE_prepared.h5ad"
    reseg, hybrid = _prepared(source)
    outputs = []
    for name in ("a", "b"):
        destination = tmp_path / name / "x.h5ad"
        m8b.thin_prepared_h5ad(
            source,
            destination,
            platform="MERSCOPE",
            reseg_totals=reseg,
            hybrid_totals=hybrid,
        )
        outputs.append(ad.read_h5ad(destination).X.toarray())
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_thin_prepared_h5ad_refuses_a_gene_set_mismatch(
    m8b: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "S_MERSCOPE_prepared.h5ad"
    reseg, hybrid = _prepared(source)
    with pytest.raises(SystemExit, match="gene totals"):
        m8b.thin_prepared_h5ad(
            source,
            tmp_path / "o.h5ad",
            platform="MERSCOPE",
            reseg_totals=reseg,
            hybrid_totals=hybrid + 1,
        )


def test_merge_columns_places_blocks(m8b: ModuleType) -> None:
    left = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
    right = sparse.csr_matrix(np.array([[9], [8]]))
    merged = m8b.merge_columns((2, 3), (np.array([0, 2]), left), (np.array([1]), right))
    np.testing.assert_array_equal(merged.toarray(), [[1, 9, 2], [3, 8, 4]])
    with pytest.raises(ValueError):
        m8b.merge_columns((2, 3), (np.array([0, 1]), left))


# ---------------------------------------------------------------------------
# Re-running B1's tasks


def _trace(path: Path, rows: list[dict[str, str]]) -> Path:
    columns = ["task_id", "process", "tag", "status", "workdir"]
    lines = ["\t".join(columns)]
    lines += ["\t".join(row.get(column, "") for column in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_find_task_takes_the_last_completed(m8b: ModuleType, tmp_path: Path) -> None:
    old, new = tmp_path / "w1", tmp_path / "w2"
    old.mkdir()
    new.mkdir()
    process = "A:B:CLUSTERING_SQUIDPY_ANNOTATE_MAP"
    first = _trace(
        tmp_path / "t1.tsv",
        [
            {
                "process": process,
                "tag": "P:proseg_hybrid",
                "status": "COMPLETED",
                "workdir": str(old),
            }
        ],
    )
    second = _trace(
        tmp_path / "t2.tsv",
        [
            {
                "process": process,
                "tag": "P:proseg_hybrid",
                "status": "COMPLETED",
                "workdir": str(new),
            },
            {
                "process": process,
                "tag": "P:proseg_hybrid",
                "status": "FAILED",
                "workdir": str(old),
            },
            {
                "process": process,
                "tag": "P:reseg",
                "status": "COMPLETED",
                "workdir": str(old),
            },
        ],
    )
    assert (
        m8b.find_task(
            [first, second], "CLUSTERING_SQUIDPY_ANNOTATE_MAP", "P:proseg_hybrid"
        )
        == new
    )
    with pytest.raises(SystemExit):
        m8b.find_task([first], "ANNOTATION_REPORT", "P:proseg_hybrid")


def test_stage_task_replaces_only_the_named_inputs(
    m8b: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "work"
    (source / "map_inputs" / "bundle_refs").mkdir(parents=True)
    targets = {name: tmp_path / name for name in ("prep", "panel", "ref1", "new_prep")}
    for target in targets.values():
        target.mkdir()
    (source / "map_inputs" / "clustering_prepare_out").symlink_to(targets["prep"])
    (source / "map_inputs" / "annotation_panel_out").symlink_to(targets["panel"])
    (source / "map_inputs" / "bundle_refs" / "bundle_ref_1.json").symlink_to(
        targets["ref1"]
    )
    (source / ".command.sh").write_text("echo")
    out = tmp_path / "task"
    out.mkdir()
    staged = m8b.stage_task(
        source, out, {"map_inputs/clustering_prepare_out": targets["new_prep"]}
    )
    links = {item["path"]: item["target"] for item in staged}
    assert links["map_inputs/clustering_prepare_out"] == str(targets["new_prep"])
    assert links["map_inputs/annotation_panel_out"] == str(targets["panel"])
    assert (
        out / "map_inputs" / "bundle_refs" / "bundle_ref_1.json"
    ).resolve() == targets["ref1"]
    assert not (out / ".command.sh").exists()
    with pytest.raises(SystemExit):
        m8b.stage_task(
            source, tmp_path / "task2", {"map_inputs/missing": targets["prep"]}
        )


def test_rewrite_pythonpath(m8b: ModuleType) -> None:
    script = (
        "set -e\n"
        'export PYTHONPATH="/old/export/workflows/../src:${PYTHONPATH:-}"\n'
        "merxen annotate --x\n"
    )
    updated, old, new = m8b.rewrite_pythonpath(script, Path("/m8b/code/head"))
    assert old == "/old/export/workflows/../src"
    assert new == "/m8b/code/head/src"
    assert 'export PYTHONPATH="/m8b/code/head/src:${PYTHONPATH:-}"' in updated
    assert updated.replace(new, old) == script
    with pytest.raises(SystemExit):
        m8b.rewrite_pythonpath("echo no export", Path("/x"))


def test_task_replacements(m8b: ModuleType, tmp_path: Path) -> None:
    locations = m8b.Locations(
        tmp_path / "runs", tmp_path / "results", tmp_path / "b", tmp_path / "out"
    )
    assert set(m8b.task_replacements(locations, "P7513", "MAP")) == {
        "map_inputs/clustering_prepare_out"
    }
    resolve = m8b.task_replacements(locations, "P7513", "RESOLVE")
    assert resolve["resolve_inputs/annotation_map_out"] == locations.map_dir(
        "hybrid_matched", "P7513"
    )
    report = m8b.task_replacements(locations, "P7513", "REPORT")
    assert set(report) == {
        "report_inputs/annotation_resolve_out",
        "report_inputs/annotation_map_out",
    }
    with pytest.raises(SystemExit):
        m8b.task_replacements(locations, "P7513", "PANEL")


def test_locations_point_hybrid_matched_at_its_own_outputs(
    m8b: ModuleType, tmp_path: Path
) -> None:
    locations = m8b.Locations(
        tmp_path / "runs", tmp_path / "results", tmp_path / "b", tmp_path / "out"
    )
    assert "arms/hybrid_matched/P1212" in str(
        locations.resolve_dir("hybrid_matched", "P1212")
    )
    assert locations.resolve_dir("hybrid", "P1212") == (
        tmp_path / "runs/P1212/proseg_hybrid/annotation_resolve/annotation_resolve_out"
    )
    assert locations.panel_dir("hybrid_matched", "P1212") == locations.panel_dir(
        "hybrid", "P1212"
    )
    assert (
        locations.clustered_path("reseg", "P1212", "XENIUM").name
        == "P1212_XENIUM_clustered.h5ad"
    )


def test_farm_qc_summary_swaps_the_segmentation_rows(m8b: ModuleType) -> None:
    frame = pd.DataFrame(
        {
            "sample": ["P", "P", "P"],
            "seg": ["proseg_hybrid", "reseg", "original_seg"],
            "platform": ["merscope"] * 3,
            "n_cells": [10, 20, 30],
        }
    )
    swapped = m8b.farm_qc_summary(frame, "reseg")
    row = swapped[swapped["seg"] == "proseg_hybrid"]
    assert row["n_cells"].tolist() == [20]
    assert m8b.farm_qc_summary(frame, "proseg_hybrid").equals(frame)


# ---------------------------------------------------------------------------
# Analysis 1


def _records(value: float, arm_extra: float = 0.0) -> list[dict[str, Any]]:
    return [
        {
            "criterion": "H7",
            "name": "confident_coverage_table",
            "kind": None,
            "level": "broad",
            "region": None,
            "scope": "dataset",
            "platform": "MERSCOPE",
            "group": None,
            "value": value,
            "ci_low": None,
            "ci_high": None,
            "n": 100,
            "status": "measured",
            "definition": "cov",
        },
        {
            "criterion": "H8",
            "name": "gate_level",
            "kind": None,
            "level": None,
            "region": None,
            "scope": "dataset",
            "platform": "MERSCOPE",
            "group": None,
            "value": "full",
            "status": "measured",
        },
        {
            "criterion": "H12",
            "name": "median_depth",
            "kind": None,
            "level": None,
            "region": None,
            "scope": "dataset",
            "platform": "MERSCOPE",
            "group": None,
            "value": 0.1 + arm_extra,
            "status": "measured",
        },
        {
            "criterion": "H12",
            "name": "median_depth",
            "kind": None,
            "level": None,
            "region": None,
            "scope": "dataset",
            "platform": "MERSCOPE",
            "group": None,
            "value": 0.2,
            "status": "measured",
        },
    ]


def test_records_wide_lays_arms_side_by_side(m8b: ModuleType) -> None:
    wide = m8b.records_wide(
        {
            "reseg": _records(0.7, 0.05),
            "hybrid": _records(0.6),
            "hybrid_matched": _records(0.65),
            "proseg_mask": [],
        },
        "P7513",
    )
    assert len(wide) == 4
    coverage = wide[wide["name"] == "confident_coverage_table"].iloc[0]
    assert coverage["reseg_minus_hybrid"] == pytest.approx(0.1)
    assert coverage["reseg_minus_hybrid_matched"] == pytest.approx(0.05)
    assert coverage["role"] == "development"
    gate = wide[wide["name"] == "gate_level"].iloc[0]
    assert gate["value_reseg"] == "full" and pd.isna(gate["reseg_minus_hybrid"])
    depth = wide[wide["name"] == "median_depth"].sort_values("occurrence")
    assert depth["occurrence"].tolist() == [0, 1]
    assert depth["reseg_minus_hybrid"].tolist() == pytest.approx([0.05, 0.0])
    assert "value_proseg_mask" in wide


def test_criteria_wide(m8b: ModuleType) -> None:
    def frame(value: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "criterion": "H7",
                    "pair": "P5011",
                    "dataset": "P5011_MERSCOPE",
                    "value": value,
                    "comparator": ">=",
                    "threshold": 0.5,
                    "passes": True,
                    "note": "",
                },
                {
                    "criterion": "H8",
                    "pair": "P5011",
                    "dataset": "P5011_MERSCOPE",
                    "value": "full",
                    "comparator": "==",
                    "threshold": "full",
                    "passes": True,
                    "note": "",
                },
            ]
        )

    wide = m8b.criteria_wide({"reseg": frame(0.4), "hybrid": frame(0.6)})
    h7 = wide[wide["criterion"] == "H7"].iloc[0]
    assert h7["reseg_minus_hybrid"] == pytest.approx(-0.2)
    assert h7["role"] == "held_out"
    assert pd.isna(wide[wide["criterion"] == "H8"].iloc[0]["reseg_minus_hybrid"])
    assert math.isnan(h7.get("value_hybrid_matched", math.nan))


def test_depth_bin_coverage(m8b: ModuleType) -> None:
    labels = pd.DataFrame(
        {
            "depth_bin": [10, 10, 15, 250],
            "ct_broad_status": [
                "confident",
                "low_confidence",
                "confident",
                "not_resolvable",
            ],
            "ct_lineage_status": ["confident"] * 4,
        }
    )
    frame = m8b.depth_bin_coverage(labels, pair="P1212", platform="XENIUM", arm="reseg")
    broad = frame[frame["level"] == "broad"].set_index("depth_bin")
    assert broad.loc[10, "coverage"] == pytest.approx(0.5)
    assert broad.loc[10, "share_low_confidence"] == pytest.approx(0.5)
    assert broad.loc[250, "share_not_resolvable"] == pytest.approx(1.0)
    assert broad.loc["all", "n_table"] == 4 and broad.loc[
        "all", "coverage"
    ] == pytest.approx(0.5)
    assert math.isnan(broad.loc[30, "coverage"])
    assert set(frame["level"]) == {"broad", "lineage"}


def _section(rng: np.random.Generator, n: int, platform: str) -> co.SectionComposition:
    soft = rng.dirichlet(np.ones(8), n)
    names = rng.choice(list(HUMAN_BROAD_CLASSES), n)
    return co.section_composition(
        f"S_{platform}",
        platform,
        soft,
        total_counts=rng.integers(10, 100, n),
        argmax_broad=names,
        confident_broad=names,
        confident=rng.random(n) > 0.3,
        xy=rng.random((n, 2)) * 3000,
        aligned_frame=True,
    )


def test_paired_jsd_rows_identical_arms_have_zero_difference(m8b: ModuleType) -> None:
    rng = np.random.default_rng(0)
    first = {
        "MERSCOPE": _section(rng, 300, "MERSCOPE"),
        "XENIUM": _section(rng, 250, "XENIUM"),
    }
    rows = m8b.paired_jsd_rows(
        first, first, mask=None, names=("reseg", "hybrid"), pair="P7513"
    )
    assert {row["kind"] for row in rows} == set(m8b.JSD_KINDS)
    for row in rows:
        assert row["region"] == "whole_section"
        assert row["difference"] == pytest.approx(0.0)
        assert row["difference_ci_low"] == pytest.approx(0.0)
        assert row["difference_ci_high"] == pytest.approx(0.0)
        point = co.jensen_shannon_distance(
            first["MERSCOPE"].matrices[row["kind"]].sum(0),
            first["XENIUM"].matrices[row["kind"]].sum(0),
        )
        assert row["jsd_first"] == pytest.approx(float(point))


def test_paired_jsd_rows_ci_covers_a_real_difference(m8b: ModuleType) -> None:
    rng = np.random.default_rng(1)
    first = {
        "MERSCOPE": _section(rng, 400, "MERSCOPE"),
        "XENIUM": _section(rng, 400, "XENIUM"),
    }
    second = {"MERSCOPE": _section(rng, 400, "MERSCOPE"), "XENIUM": first["XENIUM"]}
    rows = m8b.paired_jsd_rows(
        first, second, mask=None, names=("a", "b"), pair="P1212", kinds=("soft",)
    )
    row = rows[0]
    assert row["difference_ci_low"] <= row["difference"] <= row["difference_ci_high"]
    assert row["difference"] == pytest.approx(row["jsd_first"] - row["jsd_second"])


# ---------------------------------------------------------------------------
# Analysis 2: the transcript tallies


def _transcripts(m8b: ModuleType) -> tuple[Any, dict[str, Any]]:
    """Seven transcripts of two genes around two nuclei and three cells.

    Nucleus 0 (0-2 x 0-2) lies in cell 10 (0-3 x 0-3); nucleus 1 (5-6 x 5-6)
    lies in no cell polygon. Cells 10 and 11 are table cells of both
    segmentations, cell 12 is a reseg non-table cell.
    """
    none = m8b.NO_CELL
    x = np.array([1.0, 1.5, 2.5, 5.5, 8.0, 1.0, 20.0])
    y = np.array([1.0, 1.5, 2.5, 5.5, 8.0, 0.5, 20.0])
    inputs = m8b.TranscriptInputs(
        x=x,
        y=y,
        gene=np.array([0, 0, 0, 1, 1, 1, 0]),
        genes=["AAA", "BBB"],
        assignment=np.array([10, 12, 10, 0, 11, 11, 0], dtype=np.uint64),
        valid_reseg=np.array([1, 1, 1, 0, 1, 1, 0], bool),
        background=np.array([0, 0, 0, 1, 0, 0, 1], bool),
        hybrid_assignment=np.array([10, 10, 10, 0, 11, 10, 0], dtype=np.uint64),
        valid_hybrid=np.array([1, 1, 1, 0, 1, 1, 0], bool),
        hybrid_background=np.array([0, 0, 0, 1, 0, 0, 1], bool),
        record={},
    )
    nuclei = [shapely.box(0, 0, 2, 2), shapely.box(5, 5, 6, 6)]
    cells = [shapely.box(0, 0, 3, 3), shapely.box(7, 7, 9, 9)]
    holders = m8b.holder_ids(sc.nucleus_holders(nuclei, cells, [10, 11]))
    assert holders.tolist() == [10, int(none)]
    nucleus_index = sc.points_in_polygons(x, y, nuclei)
    assert nucleus_index.tolist() == [0, 0, -1, 1, -1, 0, -1]
    return inputs, {
        "nucleus_index": nucleus_index,
        "holders": holders,
        "reseg_table": np.array([10, 11], dtype=np.uint64),
        "hybrid_table": np.array([10, 11], dtype=np.uint64),
    }


def test_transcript_tally_frames_overall(m8b: ModuleType) -> None:
    inputs, extra = _transcripts(m8b)
    frame = m8b.transcript_tally_frames(
        inputs,
        groupings={
            "broad": ({10: "Astrocytes", 11: "Neurons"}, ["Astrocytes", "Neurons"])
        },
        **extra,
    )
    overall = frame[frame["level"] == "all"].set_index("gene")
    # AAA: transcripts 0, 1, 2, 6; reseg assigns 0 and 2 to table cell 10
    # (1 goes to non-table cell 12); hybrid assigns 0, 1, 2.
    assert overall.loc["AAA", "n_decoded"] == 4
    assert overall.loc["AAA", "n_assigned_reseg"] == 2
    assert overall.loc["AAA", "n_assigned_hybrid"] == 3
    assert overall.loc["AAA", "n_in_nucleus"] == 2
    assert overall.loc["AAA", "n_in_nucleus_unassigned_reseg"] == 1
    assert overall.loc["AAA", "n_in_nucleus_nontable_reseg"] == 1
    assert overall.loc["AAA", "n_in_nucleus_background_reseg"] == 0
    assert overall.loc["AAA", "n_assigned_reseg_same_cell"] == 2
    # BBB: 3 (background, in the holder-less nucleus 1), 4 (both to cell 11),
    # 5 (in nucleus 0: reseg cell 11, hybrid cell 10 -> not the same cell)
    assert overall.loc["BBB", "n_decoded"] == 3
    assert overall.loc["BBB", "n_assigned_reseg"] == 2
    assert overall.loc["BBB", "n_assigned_hybrid"] == 2
    assert overall.loc["BBB", "n_in_nucleus"] == 2
    assert overall.loc["BBB", "n_in_nucleus_unassigned_reseg"] == 1
    assert overall.loc["BBB", "n_in_nucleus_background_reseg"] == 1
    assert overall.loc["BBB", "n_assigned_reseg_same_cell"] == 1


def test_transcript_tally_frames_groups_by_the_holding_cell(m8b: ModuleType) -> None:
    inputs, extra = _transcripts(m8b)
    frame = m8b.transcript_tally_frames(
        inputs,
        groupings={
            "broad": ({10: "Astrocytes", 11: "Neurons"}, ["Astrocytes", "Neurons"])
        },
        **extra,
    )
    broad = frame[frame["level"] == "broad"].set_index(["group", "gene"])
    # Astrocytes = cell 10: AAA 0, 1 (nucleus 0), 2 (assigned); BBB 5 (nucleus 0)
    assert broad.loc[("Astrocytes", "AAA"), "n_decoded"] == 3
    assert broad.loc[("Astrocytes", "BBB"), "n_decoded"] == 1
    assert broad.loc[("Astrocytes", "BBB"), "n_assigned_reseg"] == 1
    assert broad.loc[("Astrocytes", "BBB"), "n_assigned_reseg_same_cell"] == 0
    # Neurons = cell 11: BBB 4 only; transcript 3's nucleus has no holder
    assert broad.loc[("Neurons", "BBB"), "n_decoded"] == 1
    assert broad.loc[("Neurons", "AAA"), "n_decoded"] == 0
    assert frame[frame["level"] == "broad"]["n_decoded"].sum() == 5


def test_assigned_to_and_label_codes(m8b: ModuleType) -> None:
    ids = np.array([5, 7, 9, 5], dtype=np.uint64)
    table = np.array([5, 9], dtype=np.uint64)
    assigned = m8b.assigned_to(
        ids, np.array([1, 1, 1, 0], bool), np.array([0, 0, 1, 0], bool), table
    )
    # 7 is not a table cell, 9 is background, the last is unset
    assert assigned.tolist() == [True, False, False, False]
    empty = np.array([], np.uint64)
    assert (
        m8b.assigned_to(ids, np.ones(4, bool), np.zeros(4, bool), empty).tolist()
        == [False] * 4
    )
    codes = m8b.label_codes(
        np.array([9, 5, 3], dtype=np.uint64), {5: "B", 9: "A"}, ["A", "B"]
    )
    assert codes.tolist() == [0, 1, -1]


def test_same_cell_needs_a_proseg_hybrid_table_cell(m8b: ModuleType) -> None:
    """A reseg table cell that is no proseg_hybrid table cell is never 'same'."""
    inputs = m8b.TranscriptInputs(
        x=np.array([50.0, 60.0]),
        y=np.array([50.0, 60.0]),
        gene=np.array([0, 0]),
        genes=["AAA"],
        assignment=np.array([12, 10], dtype=np.uint64),
        valid_reseg=np.array([True, True]),
        background=np.array([False, False]),
        hybrid_assignment=np.array([12, 10], dtype=np.uint64),
        valid_hybrid=np.array([True, True]),
        hybrid_background=np.array([False, False]),
        record={},
    )
    frame = m8b.transcript_tally_frames(
        inputs,
        nucleus_index=np.array([-1, -1]),
        holders=np.array([], dtype=np.uint64),
        reseg_table=np.array([10, 12], dtype=np.uint64),
        hybrid_table=np.array([10], dtype=np.uint64),
        groupings={},
    )
    row = frame[frame["level"] == "all"].iloc[0]
    assert row["n_assigned_reseg"] == 2
    assert row["n_assigned_hybrid"] == 1
    assert row["n_assigned_reseg_same_cell"] == 1


def test_confident_labels_and_glia(m8b: ModuleType) -> None:
    labels = pd.DataFrame(
        {
            "cell_id": ["1", "2", "3"],
            "in_table": [True, True, False],
            "ct_broad_status": ["confident", "low_confidence", "confident"],
            "ct_broad_name": ["Fibroblasts", "Neurons", "Neurons"],
        }
    )
    assert m8b.confident_labels(labels, "broad") == {1: "Fibroblasts"}
    assert m8b.glia_or_neuron("Neurons") == "neurons"
    assert m8b.glia_or_neuron("Fibroblasts") == "glia"
    assert m8b.glia_or_neuron("Microglia") == "glia"


# ---------------------------------------------------------------------------
# Analysis 2 and 4: covariates, tests, expression


def _gene_rows(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    genes = [f"G{i}" for i in range(n - 3)] + ["MAPT", "MOBP", "AQP4"]
    return pd.DataFrame(
        {
            "gene": genes,
            "gene_id": [f"ENSG{i}" for i in range(n)],
            "n_assigned_hybrid": rng.integers(0, 1000, n),
            "L": rng.random(n),
            "N_bg": rng.random(n),
        }
    )


def test_gene_covariates(m8b: ModuleType) -> None:
    genes = _gene_rows()
    genes.loc[0, "n_assigned_hybrid"] = 0
    lengths = pd.DataFrame(
        {"gene_id": ["ENSG1", "ENSG2"], "transcript_length": [1000, 5000]}
    )
    probes = pd.DataFrame({"gene_name": ["G1", "MAPT"], "probe_count": [8.0, 7.0]})
    out = m8b.gene_covariates(genes, lengths=lengths, probes=probes, n_hybrid_table=100)
    assert math.isnan(out.loc[0, "expression_level"])
    assert out.loc[1, "expression_level"] == pytest.approx(
        math.log10(genes.loc[1, "n_assigned_hybrid"] / 100)
    )
    assert out.loc[1, "gene_length"] == 1000 and math.isnan(out.loc[3, "gene_length"])
    assert out.loc[1, "probe_count"] == 8.0
    assert out["process_localised"].sum() == 3
    assert out["nuclear_retained"].sum() == 0
    none = m8b.gene_covariates(genes, lengths=lengths, probes=None, n_hybrid_table=100)
    assert none["probe_count"].isna().all()


def test_gene_tests_rows(m8b: ModuleType) -> None:
    genes = m8b.gene_covariates(
        _gene_rows(),
        lengths=pd.DataFrame(
            {
                "gene_id": [f"ENSG{i}" for i in range(30)],
                "transcript_length": np.arange(30) + 1,
            }
        ),
        probes=None,
        n_hybrid_table=10,
    )
    glia = pd.DataFrame(
        {"L_glia": np.linspace(0, 1, 10), "L_neurons": np.linspace(0, 1, 10) - 0.1}
    )
    rows = m8b.gene_tests(
        genes,
        pair="P7513",
        platform="MERSCOPE",
        glia_neurons=glia,
        probe_note="not_available: x",
    )
    frame = pd.DataFrame(rows)
    spearman = frame[frame["test"] == "spearman"]
    assert len(spearman) == 6
    assert (
        spearman.loc[spearman["covariate"] == "probe_count", "note"].tolist()
        == ["not_available: x"] * 2
    )
    assert spearman.loc[spearman["covariate"] == "probe_count", "p_value"].isna().all()
    mann = frame[frame["test"] == "mann_whitney"]
    assert len(mann) == 4
    nuclear = mann[mann["covariate"] == "member:nuclear_retained"]
    assert nuclear["p_value"].isna().all() and (nuclear["n_on_panel"] == 0).all()
    glia_row = frame[frame["covariate"] == "glia_minus_neurons"].iloc[0]
    assert glia_row["median_difference"] == pytest.approx(0.1)


def test_platform_tests(m8b: ModuleType) -> None:
    genes = pd.DataFrame(
        {
            "pair": ["P"] * 8,
            "platform": ["MERSCOPE"] * 4 + ["XENIUM"] * 4,
            "gene": ["a", "b", "c", "d"] * 2,
            "L": [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.35],
        }
    )
    rows = m8b.platform_tests(genes, "P")
    assert rows[0]["median_difference"] == pytest.approx(0.4)
    assert rows[0]["effect"] == pytest.approx(1.0)
    assert m8b.platform_tests(genes[genes["platform"] == "XENIUM"], "P") == []


def _arm(m8b: ModuleType, ids: list[str], names: list[Any], counts: np.ndarray) -> Any:
    labels = pd.DataFrame(
        {
            "cell_id": ids,
            "in_table": True,
            "ct_broad_status": ["confident" if n else "low_confidence" for n in names],
            "ct_broad_name": names,
            "contamination_score": np.linspace(0, 0.1, len(ids)),
        },
        index=pd.Index(ids, name="cell_id"),
    )
    return m8b.ArmData(
        labels=labels,
        counts=sparse.csr_matrix(counts),
        genes=["SLC17A7", "AQP4", "MOBP"],
        gene_ids=["E1", "E2", "E3"],
        xy=np.zeros((len(ids), 2)),
        aligned=True,
    )


def _reference() -> pd.DataFrame:
    values = np.full((len(HUMAN_BROAD_CLASSES), 3), 1 / 3)
    return pd.DataFrame(
        values, index=list(HUMAN_BROAD_CLASSES), columns=["E1", "E2", "E3"]
    )


def test_expression_frames_uses_cells_confident_in_both(m8b: ModuleType) -> None:
    reseg = _arm(
        m8b,
        ["1", "2", "3"],
        ["Neurons", "Astrocytes", "Neurons"],
        np.array([[4, 0, 0], [0, 4, 0], [2, 2, 0]]),
    )
    hybrid = _arm(
        m8b,
        ["1", "2", "3"],
        ["Neurons", "Astrocytes", "Astrocytes"],
        np.array([[8, 1, 1], [1, 8, 1], [3, 3, 3]]),
    )
    frames = m8b.expression_frames(
        reseg, hybrid, reference=_reference(), pair="P7513", platform="MERSCOPE"
    )
    expression = frames["expression"].set_index(["class", "gene"])
    # Neurons: cell 1 only (cell 3 differs between arms)
    assert expression.loc[("Neurons", "SLC17A7"), "n_cells_reseg"] == 1
    assert expression.loc[("Neurons", "SLC17A7"), "p_reseg"] == pytest.approx(5 / 7)
    assert expression.loc[("Neurons", "SLC17A7"), "p_hybrid"] == pytest.approx(9 / 13)
    assert expression.loc[
        ("Neurons", "SLC17A7"), "log2_reseg_over_hybrid"
    ] == pytest.approx(math.log2((5 / 7) / (9 / 13)))
    assert expression.loc[("Neurons", "SLC17A7"), "factor_reseg"] == pytest.approx(
        math.log2((5 / 7) * 3)
    )
    assert math.isnan(expression.loc[("Microglia", "AQP4"), "p_reseg"])
    contrasts = frames["marker_contrasts"]
    row = contrasts[
        (contrasts["marker"] == "SLC17A7") & (contrasts["other_class"] == "Astrocytes")
    ].iloc[0]
    assert row["contrast_reseg"] == pytest.approx(math.log2((5 / 7) / (1 / 7)))
    assert row["contrast_reseg_minus_hybrid"] == pytest.approx(
        row["contrast_reseg"] - row["contrast_hybrid"]
    )
    summary = frames["expression_summary"].set_index("class")
    assert summary.loc["Neurons", "n_cells_reseg"] == 1


def test_expression_frames_combined_arms(m8b: ModuleType) -> None:
    reseg = _arm(
        m8b, ["1", "2"], ["Neurons", "Astrocytes"], np.array([[4, 0, 0], [0, 4, 0]])
    )
    hybrid = _arm(m8b, ["1", "2"], ["Neurons", None], np.array([[8, 1, 1], [1, 8, 1]]))
    combined = np.array(["Neurons", "Astrocytes"], dtype=object)
    arms = {
        "reseg": (reseg, m8b.confident_names(reseg.labels, "broad")),
        "hybrid": (hybrid, m8b.confident_names(hybrid.labels, "broad")),
        m8b.COMBINED: (hybrid, combined),
    }
    frames = m8b.expression_frames(
        reseg,
        hybrid,
        reference=_reference(),
        pair="P5011",
        platform="XENIUM",
        arms=arms,
    )
    expression = frames["expression"].set_index(["class", "gene"])
    assert expression.loc[("Astrocytes", "AQP4"), "n_cells_combined"] == 1
    assert expression.loc[("Astrocytes", "AQP4"), "n_cells_hybrid"] == 0
    assert expression.loc[("Astrocytes", "AQP4"), "p_combined"] == pytest.approx(9 / 13)
    assert (
        "log2_combined_over_reseg" in expression
        and "log2_combined_over_hybrid" in expression
    )


def test_combined_coverage(m8b: ModuleType) -> None:
    hybrid = _arm(
        m8b, ["1", "2", "3"], ["Neurons", None, "Astrocytes"], np.ones((3, 3))
    )
    reseg = pd.DataFrame(
        {
            "cell_id": ["1", "2", "3", "4"],
            "in_table": [True, True, False, True],
            "ct_broad_status": ["confident", "confident", "low_counts", "confident"],
            "ct_broad_name": ["Neurons", "Microglia", None, "Neurons"],
        }
    )
    combined, rows = m8b.combined_coverage(
        hybrid, reseg, pair="P1212", platform="MERSCOPE"
    )
    assert combined["ct_broad_name"].tolist() == ["Neurons", "Microglia", None]
    row = rows[0]
    assert row["level"] == "broad"
    assert row["n_hybrid_table"] == 3
    assert row["share_without_reseg_table_cell"] == pytest.approx(1 / 3)
    assert row["share_confident_combined"] == pytest.approx(2 / 3)
    assert row["share_confident_hybrid"] == pytest.approx(2 / 3)
    assert row["share_confident_reseg_of_reseg_table"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Analysis 3


def test_analysis3_sample_bins_and_differences(m8b: ModuleType) -> None:
    ids = ["1", "2", "3", "4"]
    counts_r = np.array([[5, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])
    counts_h = np.array([[5, 1, 0], [5, 1, 1], [1, 5, 0], [0, 0, 5]])
    reseg = _arm(
        m8b, ids, ["Neurons", "Neurons", "Astrocytes", "Oligodendrocytes"], counts_r
    )
    hybrid = _arm(m8b, ids, ["Neurons", "Neurons", "Astrocytes", None], counts_h)
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [12.0, 0.0], [1000.0, 1000.0]])
    negatives = pd.DataFrame(
        {
            "broad_class": ["Neurons", "Neurons", "Astrocytes"],
            "gene_id": ["E2", "E3", "E1"],
            "negative": [True, True, True],
        }
    )
    out = m8b.analysis3_sample(
        reseg,
        hybrid,
        xy_reseg=xy,
        xy_hybrid=xy,
        negatives=negatives,
        pair="P7513",
        platform="XENIUM",
    )
    summary = out["summary"][0]
    assert summary["n_both_confident"] == 3 and summary["n_same_label"] == 3
    bins = pd.DataFrame(out["bins"])
    neurons = bins[
        (bins["cells"] == "both_confident")
        & (bins["class"] == "Neurons")
        & (bins["metric"] == "negative_detection_rate")
    ].set_index("distance_bin")
    # cell 1 at 12 µm from astrocyte 3 (bin 10-15); cell 2 at 88 µm (> 50)
    assert neurons.loc["10-15", "n_reseg"] == 1 and neurons.loc[">50", "n_hybrid"] == 1
    assert neurons.loc["10-15", "mean_reseg"] == pytest.approx(0.0)
    assert neurons.loc["10-15", "mean_hybrid"] == pytest.approx(0.5)
    assert neurons.loc[">50", "hybrid_minus_reseg"] == pytest.approx(1.0)
    assert neurons.loc["all", "hybrid_minus_reseg"] == pytest.approx(0.75)
    genes = pd.DataFrame(out["genes"])
    aqp4 = genes[
        (genes["cells"] == "both_confident")
        & (genes["gene"] == "AQP4")
        & (genes["class"] == "Neurons")
    ].iloc[0]
    assert aqp4["detection_reseg"] == 0.0 and aqp4["detection_hybrid"] == 1.0
    gene_bins = pd.DataFrame(out["gene_bins"])
    assert set(gene_bins["distance_bin"]) <= set(sc.DISTANCE_BIN_LABELS)


# ---------------------------------------------------------------------------
# Outputs


def test_write_analysis_and_render(m8b: ModuleType, tmp_path: Path) -> None:
    directory = tmp_path / "segmentation_comparison"
    depth = pd.DataFrame(
        {
            "pair": ["P7513"] * 6,
            "role": ["development"] * 6,
            "platform": ["MERSCOPE"] * 6,
            "arm": ["reseg", "hybrid", "hybrid_matched"] * 2,
            "level": ["broad"] * 6,
            "depth_bin": ["10", "10", "10", "all", "all", "all"],
            "coverage": [0.2, 0.3, 0.25, 0.6, 0.7, 0.65],
            "n_table": [10] * 6,
            "n_confident": [2, 3, 2, 6, 7, 6],
        }
    )
    path = m8b.write_analysis(
        directory,
        1,
        {
            "depth_bins": depth,
            "h12": pd.DataFrame(
                [
                    {
                        "pair": "P7513",
                        "role": "development",
                        "arm": "reseg",
                        "verdict": "PASS",
                    }
                ]
            ),
        },
    )
    combined = pd.read_csv(path)
    assert set(combined["table"]) == {"depth_bins", "h12"}
    assert (directory / "tables" / "analysis1_depth_bins.csv").is_file()
    figures = m8b.render_figures(directory)
    assert [record.stem for record in figures] == [
        "a1_broad_coverage_depth_development"
    ]
    assert (
        figures[0].png.is_file()
        and figures[0].pdf.is_file()
        and figures[0].csv.is_file()
    )
    page = m8b.render_page(directory, figures).read_text()
    assert "http://" not in page and "https://" not in page and "<script" not in page
    assert "analysis1.csv" in page and "Held-out donors" in page
    assert "figures/a1_broad_coverage_depth_development.png" in page


def test_series_colours_are_fixed_per_arm(m8b: ModuleType) -> None:
    colours = m8b.series_colours(["reseg", "hybrid", "P1 X", "P2 Y"])
    assert colours["hybrid"] == m8b.ARM_COLOURS["hybrid"]
    assert colours["reseg"] == m8b.ARM_COLOURS["reseg"]
    assert colours["P1 X"] == m8b.SERIES_COLOURS[0]
    assert colours["P2 Y"] == m8b.SERIES_COLOURS[1]


def test_write_json_is_nan_safe(m8b: ModuleType, tmp_path: Path) -> None:
    path = m8b.write_json(
        {"a": math.nan, "b": [np.float64(1.5), np.int64(2)]}, tmp_path / "x.json"
    )
    assert json.loads(path.read_text()) == {"a": None, "b": [1.5, 2]}


def test_source_tree_differences(m8b: ModuleType, tmp_path: Path) -> None:
    first, second = tmp_path / "a" / "src", tmp_path / "b" / "src"
    for root in (first, second):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "same.py").write_text("x = 1\n")
    (first / "pkg" / "changed.py").write_text("y = 1\n")
    (second / "pkg" / "changed.py").write_text("y = 2\n")
    (second / "pkg" / "new.py").write_text("z = 1\n")
    (second / "pkg" / "__pycache__").mkdir()
    (second / "pkg" / "__pycache__" / "junk.py").write_text("")
    differences = m8b.source_tree_differences(
        tmp_path / "a" / "x" / ".." / "src", second
    )
    assert differences == [
        {"file": "pkg/changed.py", "state": "differs"},
        {"file": "pkg/new.py", "state": "only_second"},
    ]


def test_acceptance_commands_per_arm(m8b: ModuleType, tmp_path: Path) -> None:
    locations = m8b.Locations(
        tmp_path / "runs", tmp_path / "results", tmp_path / "b", tmp_path / "out"
    )
    args = m8b.build_parser().parse_args(
        [
            "acceptance",
            "--runs-root",
            "r",
            "--results-root",
            "R",
            "--stage-b",
            "b",
            "--out",
            str(tmp_path / "out"),
            "--export",
            "/e",
            "--arm",
            "reseg",
            "--store",
            "/store",
            "--gene-id-fallback-csv",
            "/fb.h5ad",
            "--pairs",
            "P7513,P5011",
        ]
    )
    names = [name for name, _ in m8b.acceptance_commands(locations, "reseg", args)]
    assert names == ["heldout_genes", "resolve_criteria", "marker_referee"]
    criteria = dict(m8b.acceptance_commands(locations, "reseg", args))[
        "resolve_criteria"
    ]
    assert "--skip-h4" not in criteria
    assert criteria[criteria.index("--pairs") + 1] == "P7513,P5011"
    assert criteria[criteria.index("--segmentations") + 1] == "proseg_hybrid"
    matched = dict(m8b.acceptance_commands(locations, "hybrid_matched", args))
    assert set(matched) == {"resolve_criteria", "marker_referee"}
    assert "--skip-h4" in matched["resolve_criteria"]
    with pytest.raises(SystemExit):
        m8b._pairs(type("A", (), {"pairs": "P9999"})())
