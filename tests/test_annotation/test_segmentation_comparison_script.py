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
    colours = m8b.series_colours(
        ["reseg", "hybrid", "P7513 MERSCOPE", "P7513 XENIUM", "P1212 XENIUM", "L"]
    )
    assert colours["hybrid"] == m8b.ARM_COLOURS["hybrid"]
    assert colours["reseg"] == m8b.ARM_COLOURS["reseg"]
    # a dataset takes its pair's colour, never an arm's
    assert colours["P7513 MERSCOPE"] == colours["P7513 XENIUM"] == m8b.OTHER_PALETTE[0]
    assert colours["P1212 XENIUM"] == m8b.OTHER_PALETTE[1]
    assert colours["L"] == m8b.OTHER_PALETTE[2]
    assert not set(m8b.OTHER_PALETTE) & set(m8b.ARM_PALETTE)
    assert (
        m8b.series_colours(["reseg - hybrid_matched"])["reseg - hybrid_matched"]
        == (m8b.ARM_COLOURS["hybrid_matched"])
    )


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


def test_render_draws_every_figure_from_its_tables(
    m8b: ModuleType, tmp_path: Path
) -> None:
    directory = tmp_path / "segmentation_comparison"
    datasets = [("P7513", "MERSCOPE"), ("P7513", "XENIUM"), ("P5011", "XENIUM")]

    def role(pair: str) -> str:
        return "development" if pair == "P7513" else "held_out"

    genes = pd.DataFrame(
        [
            {
                "pair": p,
                "role": role(p),
                "platform": plat,
                "gene": g,
                "L": 0.3,
                "N_bg": 0.4,
            }
            for p, plat in datasets
            for g in ("A", "B", "C")
        ]
    )
    by_type = pd.DataFrame(
        [
            {
                "pair": p,
                "role": role(p),
                "platform": plat,
                "level": "broad",
                "group": c,
                "gene": g,
                "L": 0.5,
                "N_bg": 0.2,
                "n_assigned_reseg": 5,
                "n_assigned_hybrid": 10,
                "n_in_nucleus": 8,
                "n_in_nucleus_unassigned_reseg": 2,
            }
            for p, plat in datasets
            for c in ("Neurons", "Astrocytes")
            for g in ("A", "B")
        ]
    )
    bins = pd.DataFrame(
        [
            {
                "pair": p,
                "role": role(p),
                "platform": plat,
                "cells": "both_confident",
                "class": c,
                "metric": m,
                "distance_bin": b,
                "hybrid_minus_reseg": 0.01,
                "ci_low": 0.0,
                "ci_high": 0.02,
            }
            for p, plat in datasets
            for c in ("Neurons", "Astrocytes")
            for m in ("contamination_score", "negative_detection_rate")
            for b in ("0-10", ">50", "all")
        ]
    )
    jsd = pd.DataFrame(
        [
            {
                "pair": "P7513",
                "role": "development",
                "kind": k,
                "region": "whole_section",
                "first": "reseg",
                "second": s,
                "difference": -0.01,
                "difference_ci_low": -0.02,
                "difference_ci_high": 0.0,
            }
            for k in ("soft", "confident")
            for s in ("hybrid", "hybrid_matched", "resolve_summary")
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "pair": p,
                "role": role(p),
                "platform": plat,
                "level": "broad",
                "share_confident_hybrid": 0.6,
                "share_confident_combined": 0.7,
            }
            for p, plat in datasets
        ]
    )
    m8b.write_analysis(directory, 1, {"jsd_paired": jsd})
    m8b.write_analysis(directory, 2, {"genes": genes, "genes_by_type": by_type})
    m8b.write_analysis(directory, 3, {"bins": bins})
    m8b.write_analysis(directory, 4, {"coverage": coverage})
    figures = m8b.render_figures(directory)
    stems = {record.stem for record in figures}
    assert stems == {
        "a1_jsd_paired",
        "a2_loss_per_dataset",
        "a2_loss_by_class_pooled_development",
        "a2_loss_by_class_pooled_held_out",
        "a2_loss_by_class_median_development",
        "a2_loss_by_class_median_held_out",
        "a3_contamination_score_development",
        "a3_contamination_score_held_out",
        "a3_negative_detection_rate_development",
        "a3_negative_detection_rate_held_out",
        "a4_coverage",
    }
    pooled = pd.read_csv(
        directory / "figures" / "a2_loss_by_class_pooled_development.csv"
    )
    assert set(pooled["metric"]) == {"L", "N_bg"}
    assert pooled.loc[pooled["metric"] == "L", "value"].tolist() == pytest.approx(
        [0.5] * 4
    )
    assert pooled.loc[pooled["metric"] == "N_bg", "value"].tolist() == pytest.approx(
        [0.25] * 4
    )
    page = m8b.render_page(directory, figures).read_text()
    assert page.count("<figure>") == len(figures)


# ---------------------------------------------------------------------------
# Post hoc (after the 2026-10-01 review; not pre-registered)


def _labels(ids: list[str], status: list[str], implausible: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ids,
            "in_table": True,
            "total_counts": np.arange(10, 10 + len(ids), dtype=np.float64),
            "flag_implausible": implausible,
            "ct_broad_status": status,
        },
        index=pd.Index(ids, name="cell_id"),
    )


def test_label_metric(m8b: ModuleType) -> None:
    labels = _labels(["1", "2"], ["confident", "low_counts"], [True, False])
    assert m8b.label_metric(labels, "coverage_broad") == pytest.approx(0.5)
    assert m8b.label_metric(labels, "H2_implausible_share") == pytest.approx(0.5)
    assert math.isnan(m8b.label_metric(labels, "coverage_lineage"))
    assert math.isnan(m8b.label_metric(labels.iloc[:0], "coverage_broad"))
    with pytest.raises(ValueError):
        m8b.label_metric(labels, "H7")


def test_selection_depth_rows_split_the_headline(m8b: ModuleType) -> None:
    confident, low = "confident", "low_counts"
    # R = {1, 2, 3, 9}; H = {1, 2, 3, 4, 5}; S = {1, 2, 3}
    reseg = _labels(["1", "2", "3", "9"], [confident] * 3 + [low], [False] * 4)
    hybrid = _labels(
        ["1", "2", "3", "4", "5"], [confident, confident, low, low, low], [False] * 5
    )
    matched = _labels(["1", "2", "3"], [confident, low, low], [False] * 3)
    rows = m8b.selection_depth_rows(
        reseg, hybrid, matched, pair="P7513", platform="MERSCOPE"
    )
    row = next(r for r in rows if r["metric"] == "coverage_broad")
    assert row["n_shared"] == 3 and row["n_hybrid_only"] == 2
    assert row["n_reseg_only"] == 1 and row["matched_table_is_shared"]
    assert row["reseg_all"] == pytest.approx(0.75)
    assert row["hybrid_all"] == pytest.approx(0.4)
    assert row["reseg_shared"] == pytest.approx(1.0)
    assert row["hybrid_shared"] == pytest.approx(2 / 3)
    assert row["matched_shared"] == pytest.approx(1 / 3)
    assert row["hybrid_only"] == pytest.approx(0.0)
    assert row["headline_reseg_minus_hybrid"] == pytest.approx(0.35)
    assert row["part_cell_selection"] == pytest.approx((2 / 3 - 0.4) + (0.75 - 1.0))
    assert row["part_depth"] == pytest.approx(1 / 3 - 2 / 3)
    assert row["part_same_cells_same_depth"] == pytest.approx(1 - 1 / 3)
    parts = (
        row["part_cell_selection"]
        + row["part_depth"]
        + row["part_same_cells_same_depth"]
    )
    assert parts == pytest.approx(row["headline_reseg_minus_hybrid"])
    assert row["hybrid_matched_closes_share_of_headline"] == pytest.approx(
        (1 / 3 - 0.4) / 0.35
    )
    assert {r["metric"] for r in rows} == set(m8b.SELECTION_METRICS)


def test_ci_side_and_jsd_reading(m8b: ModuleType) -> None:
    assert m8b.ci_side(-0.2, -0.1) == "below 0"
    assert m8b.ci_side(0.1, 0.2) == "above 0"
    assert m8b.ci_side(-0.1, 0.1) == "spans 0"
    assert m8b.ci_side(math.nan, 0.1) == "no CI"
    jsd = pd.DataFrame(
        [
            {
                "pair": "P7113",
                "kind": "soft",
                "region": "whole_section",
                "first": "reseg",
                "second": second,
                "jsd_first": 0.2,
                "jsd_second": jsd_second,
                "difference": 0.2 - jsd_second,
                "difference_ci_low": low,
                "difference_ci_high": high,
            }
            for second, jsd_second, low, high in (
                ("hybrid", 0.1, 0.05, 0.15),
                ("hybrid_matched", 0.25, -0.08, -0.02),
                ("resolve_summary", 0.2, math.nan, math.nan),
            )
        ]
    )
    frame = m8b.jsd_reading(jsd)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["role"] == "held_out" and row["jsd_reseg"] == pytest.approx(0.2)
    assert row["reseg_minus_hybrid_ci_side"] == "above 0"
    assert row["reseg_minus_hybrid_matched_ci_side"] == "below 0"
    assert row["reseg_minus_hybrid_matched"] == pytest.approx(-0.05)


def _bin_rows(pair: str, cls: str, near: float, far: float) -> list[dict[str, Any]]:
    role = "development" if pair == "P7513" else "held_out"
    rows = []
    for label, value in (("0-10", near), ("20-30", 0.0), (">50", far), ("all", 0.5)):
        rows.append(
            {
                "pair": pair,
                "role": role,
                "platform": "XENIUM",
                "cells": "both_confident",
                "class": cls,
                "metric": "contamination_score",
                "distance_bin": label,
                "mean_reseg": 0.1,
                "mean_hybrid": 0.1 + value,
                "hybrid_minus_reseg": value,
                "ci_low": value - 0.01,
                "ci_high": value + 0.01,
            }
        )
    return rows


def test_spillover_summary_counts_bins_apart_from_classes(m8b: ModuleType) -> None:
    bins = pd.DataFrame(
        _bin_rows("P7513", "Neurons", 0.3, 0.1)
        + _bin_rows("P7513", "Astrocytes", -0.2, 0.0)
        + _bin_rows("P5011", "Neurons", 0.2, 0.2)
    )
    frame = m8b.spillover_summary(bins, analysis="a", neighbours="n")
    every = frame[frame["scope"] == "all datasets"].iloc[0]
    # 3 rows x 3 distance bins: above 0 = 0.3, 0.1, 0.2, 0.2; below 0 = -0.2
    assert every["n_bin_rows"] == 9
    assert every["n_bin_ci_above_0"] == 4 and every["n_bin_ci_below_0"] == 1
    assert every["n_class_rows"] == 3 and every["n_class_ci_above_0"] == 3
    assert every["n_near_far_rows"] == 3
    # near - far: 0.2, -0.2, 0.0
    assert every["median_near_minus_far"] == pytest.approx(0.0)
    assert every["share_near_minus_far_positive"] == pytest.approx(1 / 3)
    development = frame[frame["scope"] == "development"].iloc[0]
    assert development["n_class_rows"] == 2
    assert development["analysis"] == "a" and development["neighbours"] == "n"


def test_leaking_genes_ranks_by_median(m8b: ModuleType) -> None:
    genes = pd.DataFrame(
        [
            {
                "pair": pair,
                "platform": "XENIUM",
                "cells": cells,
                "class": cls,
                "gene": gene,
                "hybrid_minus_reseg": value,
                "detection_hybrid": value,
                "detection_reseg": 0.0,
                "ci_low": value - 0.01,
                "ci_high": value + 0.01,
            }
            for pair, cells, cls, gene, value in (
                ("P7513", "both_confident", "Neurons", "CD68", 0.30),
                ("P1212", "both_confident", "Neurons", "CD68", 0.00),
                ("P7113", "both_confident", "Neurons", "CD68", 0.01),
                ("P7513", "both_confident", "Neurons", "ERMN", 0.05),
                ("P1212", "both_confident", "Astrocytes", "ERMN", 0.04),
                ("P7513", "same_label", "Neurons", "ERMN", 0.90),
            )
        ]
    )
    frame = m8b.leaking_genes(genes)
    assert frame["gene"].tolist() == ["ERMN", "CD68"]
    assert frame["rank_by_median"].tolist() == [1, 2]
    cd68 = frame.set_index("gene").loc["CD68"]
    assert cd68["max_difference"] == pytest.approx(0.30)
    assert cd68["max_row"] == "P7513 XENIUM Neurons" and cd68["n_rows"] == 3
    assert frame.set_index("gene").loc["ERMN", "classes_where_negative"] == (
        "Astrocytes; Neurons"
    )


def test_noise_stratum(m8b: ModuleType) -> None:
    assert m8b.noise_stratum(2.0) == "<=2x"
    assert m8b.noise_stratum(2.01) == "2-10x"
    assert m8b.noise_stratum(10.0) == ">=10x"
    assert m8b.noise_stratum(math.nan) is None


def _noise_gene_rows(
    platform: str, decoded: list[int], loss: list[float]
) -> pd.DataFrame:
    n = len(decoded)
    hybrid = np.array(decoded) // 2
    reseg = np.round(hybrid * (1 - np.array(loss))).astype(int)
    return pd.DataFrame(
        {
            "pair": "P1212",
            "role": "development",
            "platform": platform,
            "gene": [f"G{i}" for i in range(n)],
            "n_decoded": decoded,
            "n_assigned_reseg": reseg,
            "n_assigned_hybrid": hybrid,
            "n_in_nucleus": np.array(decoded) // 4,
            "n_in_nucleus_unassigned_reseg": np.array(decoded) // 8,
            "L": loss,
            "N_bg": 0.5,
            "expression_level": np.log10(np.array(decoded, dtype=float)),
        }
    )


def test_noise_floor_tables(m8b: ModuleType) -> None:
    decoded = [100, 150, 300, 600, 1000, 5000, 8000]
    loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    merscope = _noise_gene_rows("MERSCOPE", decoded, loss)
    xenium = _noise_gene_rows("XENIUM", decoded, [0.2] * 7)
    genes = pd.concat([merscope, xenium], ignore_index=True)
    controls = sc.loss_shares(
        pd.DataFrame(
            {
                "gene": ["Blank-1", "Blank-2", "Blank-3"],
                "n_decoded": [80, 100, 120],
                "n_assigned_reseg": [2, 2, 2],
                "n_assigned_hybrid": [40, 50, 60],
                "n_assigned_reseg_same_cell": [2, 2, 2],
                "n_in_nucleus": [40, 40, 40],
                "n_in_nucleus_unassigned_reseg": [38, 39, 40],
                "n_in_nucleus_background_reseg": [38, 39, 40],
                "n_in_nucleus_nontable_reseg": [0, 0, 0],
            }
        )
    )
    tables = m8b.noise_floor_tables(
        genes,
        {
            ("P1212", "MERSCOPE"): (controls, {"note": ""}),
            ("P1212", "XENIUM"): (pd.DataFrame(), {"note": "none here"}),
        },
    )
    floor = tables["noise_floor"].set_index("platform")
    assert floor.loc["XENIUM", "note"] == "none here"
    row = floor.loc["MERSCOPE"]
    assert row["median_control_decoded"] == pytest.approx(100.0)
    assert row["control_pooled_A_hybrid"] == pytest.approx(150 / 300)
    assert row["control_pooled_L"] == pytest.approx(1 - (6 / 300) / (150 / 300))
    # mean blank assigned (hybrid 50) x 7 genes over the genes' hybrid total
    hybrid_total = float(merscope["n_assigned_hybrid"].sum())
    assert row["noise_share_of_table_counts_hybrid"] == pytest.approx(
        50 * 7 / hybrid_total
    )
    strata = tables["noise_strata"].set_index("stratum")
    # ratios 1, 1.5, 3, 6, 10, 50, 80
    assert strata.loc["<=2x", "n_genes"] == 2
    assert strata.loc["2-10x", "n_genes"] == 2
    assert strata.loc[">=10x", "n_genes"] == 3
    assert strata.loc["all genes", "n_genes"] == 7
    assert strata.loc[">=10x", "median_L"] == pytest.approx(0.4)
    assert strata.loc[">=10x", "median_L_minus_other_platform"] == pytest.approx(0.2)
    assert strata.loc[">=10x", "other_platform"] == "XENIUM"
    assert strata.loc["<=2x", "share_of_gene_transcripts"] == pytest.approx(
        250 / sum(decoded)
    )
    assert strata.loc["control features", "n_genes"] == 3
    assert set(tables["noise_strata"]["platform"]) == {"MERSCOPE"}


def test_common_bins_sample_uses_one_distance_for_both_arms(m8b: ModuleType) -> None:
    ids = ["1", "2", "3", "4"]
    counts_r = np.array([[5, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])
    counts_h = np.array([[5, 1, 0], [5, 1, 1], [1, 5, 0], [0, 0, 5]])
    names = ["Neurons", "Neurons", "Astrocytes", "Oligodendrocytes"]
    reseg = _arm(m8b, ids, names, counts_r)
    hybrid = _arm(m8b, ids, names, counts_h)
    xy_hybrid = np.array([[0.0, 0.0], [100.0, 0.0], [12.0, 0.0], [1000.0, 1000.0]])
    # reseg's own centroids would put cell 1 far from the astrocyte
    xy_reseg = xy_hybrid + np.array([[-60.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0, 0]])
    negatives = pd.DataFrame(
        {
            "broad_class": ["Neurons", "Neurons", "Astrocytes"],
            "gene_id": ["E2", "E3", "E1"],
            "negative": [True, True, True],
        }
    )
    out = m8b.common_bins_sample(
        reseg,
        hybrid,
        xy_reseg=xy_reseg,
        xy_hybrid=xy_hybrid,
        negatives=negatives,
        pair="P7513",
        platform="XENIUM",
    )
    bins = pd.DataFrame(out["bins"])
    assert (bins["n_reseg"] == bins["n_hybrid"]).all()
    neurons = bins[
        (bins["class"] == "Neurons") & (bins["metric"] == "negative_detection_rate")
    ].set_index("distance_bin")
    # cell 1: 12 µm from astrocyte 3 on the proseg_hybrid centroids (10-15)
    assert neurons.loc["10-15", "n_reseg"] == 1
    assert neurons.loc["10-15", "mean_hybrid"] == pytest.approx(0.5)
    assert neurons.loc[">50", "hybrid_minus_reseg"] == pytest.approx(1.0)
    assert out["cells"][0]["n_same_label"] == 4


def test_load_transcripts_keeps_genes_or_controls(
    m8b: ModuleType, tmp_path: Path
) -> None:
    store = tmp_path / "store.zarr"
    part = store / "points" / "transcripts" / "points.parquet"
    part.mkdir(parents=True)
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "gene": pd.Categorical(["GAD1", "Blank-1", "GAD1", "Blank-2"]),
            "qv": np.float32([30, 30, 30, 30]),
            "assignment": pd.array([1, None, 2, 3], dtype="UInt32"),
            "background": [False, True, False, False],
            "hybrid_assignment": pd.array([1, 1, None, 3], dtype="UInt32"),
            "hybrid_background": [False, False, True, False],
        }
    ).to_parquet(part / "part.0.parquet", index=False)
    genes = m8b.load_transcripts(store, "MERSCOPE", None)
    controls = m8b.load_transcripts(store, "MERSCOPE", None, features="controls")
    assert genes.genes == ["GAD1"] and len(genes.x) == 2
    assert controls.genes == ["Blank-1", "Blank-2"] and len(controls.x) == 2
    assert genes.record["n_control_removed"] == controls.record["n_control_removed"]
    assert controls.record["features"] == "controls"
    assert controls.valid_reseg.tolist() == [False, True]
    with pytest.raises(ValueError):
        m8b.load_transcripts(store, "MERSCOPE", None, features="all")


def test_control_tallies_marks_stores_without_controls(
    m8b: ModuleType, tmp_path: Path
) -> None:
    locations = m8b.Locations(
        tmp_path / "runs", tmp_path / "results", tmp_path / "b", tmp_path / "out"
    )
    xenium = locations.transcripts_cache("P7513", "XENIUM")
    m8b.write_json({"filter": {"n_control_removed": 0}}, xenium / "tally_record.json")
    merscope = locations.transcripts_cache("P7513", "MERSCOPE")
    m8b.write_json({"filter": {"n_control_removed": 5}}, merscope / "tally_record.json")
    cache = locations.controls_cache("P7113", "MERSCOPE")
    cache.mkdir(parents=True)
    tally = pd.DataFrame(
        {"level": ["all"], "group": ["all"], "gene": ["Blank-1"]}
        | {field: [4] for field in sc.TALLY_FIELDS}
    )
    tally.to_parquet(cache / "tally.parquet", index=False)
    m8b.write_json({"note": ""}, cache / "tally_record.json")
    out = m8b.control_tallies(locations, ["P7513", "P7113"])
    assert set(out) == {("P7513", "XENIUM"), ("P7113", "MERSCOPE")}
    assert out[("P7513", "XENIUM")][1]["note"] == m8b.NO_CONTROLS
    assert out[("P7113", "MERSCOPE")][0]["A_hybrid"].tolist() == [1.0]


def test_matched_count_sources_and_run_args(m8b: ModuleType) -> None:
    frame = m8b.matched_count_sources()
    by_item = dict(zip(frame["item"], frame["depth_matched"], strict=True))
    assert any(
        item.startswith("H9") and value.startswith("no")
        for item, value in by_item.items()
    )
    assert any(
        item.startswith("MAP") and value == "yes" for item, value in by_item.items()
    )
    args = m8b.build_parser().parse_args(
        [
            "posthoc",
            "--runs-root",
            "r",
            "--results-root",
            "s",
            "--stage-b",
            "b",
            "--out",
            "o",
            "--store",
            "x",
        ]
    )
    assert args.handler is m8b.command_posthoc
    recorded = m8b._run_args(args)
    assert "handler" not in recorded and recorded["store"] == "x"


def test_render_page_adds_the_posthoc_section(m8b: ModuleType, tmp_path: Path) -> None:
    directory = tmp_path / "segmentation_comparison"
    selection = pd.DataFrame(
        [
            {
                "pair": "P7513",
                "role": "development",
                "platform": "MERSCOPE",
                "metric": "coverage_broad",
                "reseg_shared": 0.9,
                "hybrid_shared": 0.8,
                "matched_shared": 0.7,
                "hybrid_only": 0.3,
            }
        ]
    )
    m8b.write_tables(
        directory,
        "posthoc",
        {
            "selection_depth": selection,
            "matched_count_sources": m8b.matched_count_sources(),
        },
    )
    (directory / m8b.MEMO_NAME).write_text("memo\n")
    figures = m8b.render_figures(directory)
    assert [record.stem for record in figures] == ["p_same_cell_coverage_development"]
    page = m8b.render_page(directory, figures).read_text()
    assert "Post-hoc additions after the review" in page
    assert 'href="posthoc.csv"' in page and f'href="{m8b.MEMO_NAME}"' in page
    assert "figures/p_same_cell_coverage_development.png" in page
