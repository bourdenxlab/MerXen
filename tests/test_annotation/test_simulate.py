"""Tests for ``annotation-panel-simulate`` (plan §8.8; M3b stage D)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation import simulate
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.simulate import (
    REFERENCE_SOURCE_PARAMS,
    GatePUnavailableError,
    ReferenceBuild,
    compare_calls,
    depth_profile_weights,
    emission_summary,
    panel_regime,
    predicted_levels,
    prefilter_verdict,
    reference_sources,
    register_gate_p_hook,
    require_gate_p_hook,
    resolvable_share,
    run_panel_simulation,
)
from merxen.annotation.store import BuildContext, BundleBuilder, ReferenceStore
from merxen.cli import main as cli_main

REPO = Path(__file__).resolve().parents[2]


def cells_frame(
    calls: list[tuple[str, str, str, float, bool, str]], recipe: str = "R1_contam_HO"
) -> pd.DataFrame:
    """Cells-table rows: (level, sim_id, call, bp, correct, truth_parent)."""
    return pd.DataFrame(
        [
            {
                "recipe": recipe,
                "seed": 0,
                "level": level,
                "sim_id": sim_id,
                "call": call,
                "bp": bp,
                "correct": correct,
                "truth_parent": parent,
            }
            for level, sim_id, call, bp, correct, parent in calls
        ]
    )


def paired_cells(
    n: int, *, agree: int, level: str = "broad", parent: str = "Astro"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n confident unfiltered calls; the first ``agree`` agree prefiltered."""
    unfiltered = cells_frame(
        [(level, f"{parent}{index}", "A", 0.95, True, parent) for index in range(n)]
    )
    prefiltered = cells_frame(
        [
            (
                level,
                f"{parent}{index}",
                "A" if index < agree else "B",
                0.9,
                index < agree,
                parent,
            )
            for index in range(n)
        ]
    )
    return prefiltered, unfiltered


def test_compare_calls_judges_classes_with_enough_confident_calls() -> None:
    pre_a, unf_a = paired_cells(100, agree=97)
    pre_b, unf_b = paired_cells(100, agree=90, parent="Oligo")
    pre_c, unf_c = paired_cells(20, agree=10, parent="COP")
    pre_d, unf_d = paired_cells(100, agree=50, level="cluster", parent="Astro")
    comparison = compare_calls(
        pd.concat([pre_a, pre_b, pre_c, pre_d]),
        pd.concat([unf_a, unf_b, unf_c, unf_d]),
        reference_id="wmb_panel",
        emitted_levels=["broad"],
        recipe="R1_contam_HO",
    )
    by = comparison.set_index(["level", "class"])
    assert by.loc[("broad", "Astro"), "status"] == "pass"
    assert by.loc[("broad", "Astro"), "agreement_unfiltered_confident"] == 0.97
    assert by.loc[("broad", "Oligo"), "status"] == "fail"
    assert by.loc[("broad", "COP"), "status"] == "not_evaluable"
    assert by.loc[("broad", "Astro"), "precision_prefiltered"] == 0.97
    # A level the panel does not emit is reported, not judged.
    assert not bool(by.loc[("cluster", "Astro"), "emitted_level"])
    verdict = prefilter_verdict(
        comparison,
        prefiltered_min_markers=7,
        unfiltered_min_markers=9,
        prefiltered_weak_parents=[],
        unfiltered_weak_parents=[],
    )
    assert not verdict["passes"] and verdict["n_fail"] == 1
    assert verdict["failing"][0]["class"] == "Oligo"
    assert verdict["n_not_evaluable"] == 1 and verdict["n_pass"] == 1


def test_compare_calls_uses_the_decision_recipe_at_seed_0() -> None:
    prefiltered, unfiltered = paired_cells(60, agree=60)
    other = cells_frame(
        [("broad", f"Astro{index}", "B", 0.99, False, "Astro") for index in range(60)],
        recipe="clean",
    )
    comparison = compare_calls(
        pd.concat([prefiltered, other]),
        unfiltered,
        reference_id="x",
        emitted_levels=["broad"],
        recipe="R1_contam_HO",
    )
    assert comparison["status"].tolist() == ["pass"]


def test_prefilter_verdict_fails_only_parents_the_prefilter_made_weak() -> None:
    prefiltered, unfiltered = paired_cells(60, agree=60)
    comparison = compare_calls(
        prefiltered,
        unfiltered,
        reference_id="x",
        emitted_levels=["broad"],
        recipe="R1_contam_HO",
    )
    # A parent the panel already leaves weak is a panel property.
    panel_weak = prefilter_verdict(
        comparison,
        prefiltered_min_markers=1,
        unfiltered_min_markers=1,
        prefiltered_weak_parents=["CLAS/a"],
        unfiltered_weak_parents=["CLAS/a"],
    )
    assert panel_weak["passes"] and not panel_weak["no_parent_below_minimum"]
    made_weak = prefilter_verdict(
        comparison,
        prefiltered_min_markers=3,
        unfiltered_min_markers=12,
        prefiltered_weak_parents=["CLAS/b"],
        unfiltered_weak_parents=[],
    )
    assert not made_weak["passes"] and made_weak["agreement_passes"]
    assert made_weak["parents_weak_only_with_prefilter"] == ["CLAS/b"]


def test_regime_follows_the_validation_basis() -> None:
    assert panel_regime("validated", "real_data") == "validated"
    # Simulation-validated families keep the provisional rule (§8.2).
    assert panel_regime("validated", "simulation") == "provisional"
    assert panel_regime("provisional", None) == "provisional"
    assert panel_regime("broad_only", None) == "provisional"


def decisions() -> pd.DataFrame:
    rows = []
    for regime in ("validated", "provisional"):
        for depth, status in ((10, "not_resolvable"), (20, "emitted"), (50, "emitted")):
            rows.append(
                {
                    "kind": "decision",
                    "regime": regime,
                    "level": "class",
                    "class": "30 Astro-Epen",
                    "depth": depth,
                    "status": status,
                    "reason": None if status == "emitted" else "coverage_below_minimum",
                    "extrapolated": depth == 50,
                    "pooled": depth == 50,
                }
            )
    return pd.DataFrame(rows)


def test_predicted_levels_and_their_summary() -> None:
    predicted = predicted_levels(decisions(), "wmb_panel", "provisional")
    assert predicted["regime"].unique().tolist() == ["provisional"]
    assert list(predicted.columns) == list(simulate.PREDICTED_COLUMNS)
    summary = emission_summary(predicted)
    entry = summary["class"]["classes"]["30 Astro-Epen"]
    assert entry["emitted_depths"] == [20, 50] and entry["first_depth"] == 20
    assert entry["n_extrapolated"] == 1
    assert summary["class"]["n_classes_emitted"] == 1
    assert predicted_levels(pd.DataFrame(), "x", "provisional").empty


def test_depth_profile_shares_and_resolvable_share() -> None:
    shares = depth_profile_weights([5, 15, 25, 60, 70], [10, 20, 50])
    assert shares == {0: 0.2, 10: 0.2, 20: 0.2, 50: 0.4}
    predicted = predicted_levels(decisions(), "wmb_panel", "provisional")
    share = resolvable_share(predicted, shares)
    assert share == {"class": {"30 Astro-Epen": pytest.approx(0.6)}}


def test_reference_sources_match_the_nextflow_source_params() -> None:
    groovy = (REPO / "workflows" / "lib" / "AnnotationReferences.groovy").read_text()
    block = groovy[
        groovy.index("SOURCE_PARAMS = [") : groovy.index("].asImmutable()\n\n")
    ]
    parsed: dict[str, dict[str, str]] = {}
    current = None
    for line in block.splitlines():
        header = re.match(r"\s*(\w+): \[$", line)
        entry = re.match(r'\s*(\w+): "(\w+)",', line)
        if header:
            current = header.group(1)
            parsed[current] = {}
        elif entry and current is not None:
            parsed[current][entry.group(1)] = entry.group(2)
    assert parsed == REFERENCE_SOURCE_PARAMS
    sources = reference_sources(
        "wmb_panel",
        {"annotation_wmb_h5ad_dir": Path("/w"), "annotation_whb_h5ad_dir": Path("/h")},
    )
    assert sources == {"wmb_h5ad_dir": Path("/w")}


def test_gate_p_is_refused_until_m13_registers_it() -> None:
    register_gate_p_hook(None)
    with pytest.raises(GatePUnavailableError, match="M13"):
        require_gate_p_hook()
    calls: list[Any] = []

    def hook(request: Any) -> dict[str, Any]:
        calls.append(request)
        return {"np": "ok"}

    register_gate_p_hook(hook)
    try:
        assert require_gate_p_hook() is hook
    finally:
        register_gate_p_hook(None)


def test_cli_refuses_gate_p_before_any_compute(tmp_path: Path) -> None:
    register_gate_p_hook(None)
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("gene_symbol,gene_id\nGfap,ENSMUSG00000020932\n")
    result = CliRunner().invoke(
        cli_main,
        [
            "annotation-panel-simulate",
            "--gene-list",
            str(gene_list),
            "--species",
            "mouse",
            "--store",
            str(tmp_path / "store"),
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--out-dir",
            str(tmp_path / "out"),
            "--gate-p",
        ],
    )
    assert result.exit_code != 0
    assert "M13" in result.output
    assert not (tmp_path / "out").exists() and not (tmp_path / "store").exists()


def panel_builder() -> BundleBuilder:
    """A builder without a self-map that records panel coverage."""

    def build(context: BuildContext) -> Mapping[str, Any]:
        (context.work_dir / "query_markers.json").write_text('{"None": {}}')
        assert context.panel is not None
        return {
            "levels": ["CCN202210140_SUPC"],
            "markers": {"n_candidate_genes": context.panel.n_genes, "prefilter": None},
            "panel_coverage": {
                "n_panel_genes": context.panel.n_genes,
                "n_query_genes_used": context.panel.n_genes,
                "root_markers": 40,
                "root_children": 3,
                "root_children_separated": 3,
            },
            "timings_s": {"reference_markers": 1.0},
        }

    return BundleBuilder(name="sim_builder", build=build, taxonomy_id="CCN202210140")


def test_simulation_writes_its_report_for_a_bundle_without_a_self_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [f"ENSG{index:011d}" for index in range(1, 61)]
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text(
        "gene_symbol,gene_id\n"
        + "\n".join(f"G{index},{gene}" for index, gene in enumerate(ids))
        + "\n"
    )
    config = AnnotationConfig(species="human", resolvability={"enabled": False})
    spec = AnnotationReferenceSpec(
        reference_id="whb_frontal_supc_clus", species="human", role="primary"
    )
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    report = run_panel_simulation(
        gene_list=gene_list,
        species="human",
        name="sixty",
        config=config,
        store=store,
        builds=lambda panel: [ReferenceBuild(spec=spec, builder=panel_builder())],
        out_dir=tmp_path / "out",
        scratch_dir=tmp_path / "sim_scratch",
        platform="XENIUM",
        expected_depth=40,
    )
    assert report["status"] == "done"
    assert report["panel"]["n_genes"] == 60 and not report["panel"]["large_panel"]
    record = report["references"]["whb_frontal_supc_clus"]
    assert record["status"] == "built" and record["regime"] == "provisional"
    assert record["prefilter_comparison"]["status"] == "not_run"
    assert record["resources"]["disk"]["bundle_gb"] >= 0.0
    peak = report["resources"]["process_tree_peak"]
    assert peak["peak_tree_rss_gb"] > 0
    out = tmp_path / "out"
    for name in (
        simulate.REPORT_JSON,
        simulate.REPORT_TXT,
        simulate.PREDICTED_CSV,
        simulate.PREFILTER_CSV,
        simulate.RESOURCES_CSV,
    ):
        assert (out / name).is_file()
    assert json.loads((out / simulate.REPORT_JSON).read_text())["name"] == "sixty"
    text = (out / simulate.REPORT_TXT).read_text()
    assert "whb_frontal_supc_clus: built" in text
    resources = pd.read_csv(out / simulate.RESOURCES_CSV)
    assert "reference_markers" in resources["step"].tolist()
    # A second run reuses the bundle.
    again = run_panel_simulation(
        gene_list=gene_list,
        species="human",
        name="sixty",
        config=config,
        store=store,
        builds=lambda panel: [ReferenceBuild(spec=spec, builder=panel_builder())],
        out_dir=tmp_path / "out2",
        scratch_dir=tmp_path / "sim_scratch",
    )
    assert again["references"]["whb_frontal_supc_clus"]["reused"] is True
    assert np.isfinite(again["resources"]["wall_s"])


def test_judged_levels_leave_out_report_only_fine_levels() -> None:
    frame = pd.DataFrame(
        {
            "level": ["class", "subclass", "supertype", "nt"],
            "status": ["emitted", "emitted", "emitted", "not_resolvable"],
        }
    )
    assert simulate.judged_levels(frame, fine_levels={"supertype"}) == [
        "class",
        "subclass",
    ]
    assert simulate.judged_levels(
        frame, fine_levels={"supertype"}, allow_fine_levels=True
    ) == ["class", "subclass", "supertype"]
    assert simulate.fine_levels_of(Path("/nonexistent")) == set()
