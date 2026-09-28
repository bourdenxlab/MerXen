"""Tests for the RESOLVE criteria and marker pseudo-label acceptance scripts."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

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
def pseudo() -> ModuleType:
    return _load("marker_pseudo_labels")


@pytest.fixture(scope="module")
def criteria() -> ModuleType:
    return _load("resolve_criteria")


# ---------------------------------------------------------------------------
# marker_pseudo_labels


def test_p1212_pseudo_labels_scale_by_positive_mean_and_need_60_percent(
    pseudo: ModuleType,
) -> None:
    genes = ["AQP4", "GJA1", "SLC17A7", "GAD1", "DCN", "FLT1", "OTHER"]
    counts = np.array(
        [
            [4, 2, 0, 0, 0, 0, 5],  # astrocyte: 2 units at the markers' means
            [0, 0, 2, 2, 0, 0, 0],  # Exc / Inh split 50:50 -> unassigned
            [0, 0, 0, 0, 1, 0, 0],  # 1 unit < 1.5
            [0, 0, 0, 0, 2, 2, 0],  # Fibro 1.33 units + Vasc 0.5 -> unassigned
            [0, 0, 0, 0, 0, 6, 0],  # Vasc, 1.5 units -> Vascular in 6 classes
            [2, 2, 1, 0, 0, 0, 0],  # Astro 1.67 units, 71% of the scores
        ]
    )
    result = pseudo.p1212_pseudo_labels(sparse.csr_matrix(counts), genes)
    assert list(result["pseudo8"]) == [
        "Astro",
        "Unassigned",
        "Unassigned",
        "Unassigned",
        "Vasc",
        "Astro",
    ]
    assert list(result["pseudo6"]) == [
        "Astro",
        "Unassigned",
        "Unassigned",
        "Unassigned",
        "Vascular",
        "Astro",
    ]


def _p7513_fixture() -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    rng = np.random.default_rng(0)
    genes = ["AQP4", "GJA1", "MOBP", "MOG", "SLC17A7", "CUX2", "BG1", "BG2"]
    marker_columns = {"Astro": (0, 1), "Oligo": (2, 3), "Exc": (4, 5)}
    rows = []
    for columns in marker_columns.values():
        for _ in range(60):
            row = rng.poisson(0.3, len(genes)).astype(float)
            row[list(columns)] += rng.poisson(8, 2)
            rows.append(row)
    rows.append(np.zeros(len(genes)))  # no counts: unresolved
    mixed = np.zeros(len(genes))
    mixed[[0, 1, 2, 3]] = 40  # astro + oligo doublet
    rows.append(mixed)
    counts = sparse.csr_matrix(np.vstack(rows))
    totals = np.asarray(counts.sum(1)).ravel()
    target = float(np.median(totals[totals > 0]))
    scale = np.divide(target, totals, out=np.zeros_like(totals), where=totals > 0)
    lognorm = sparse.csr_matrix(sparse.diags(scale) @ counts)
    lognorm.data = np.log1p(lognorm.data)
    return counts, lognorm, genes


def test_p7513_pseudo_labels_seed_profiles_unresolved_and_mixed(
    pseudo: ModuleType,
) -> None:
    counts, lognorm, genes = _p7513_fixture()
    result = pseudo.p7513_pseudo_labels(counts, lognorm, genes)
    labels = result["pseudo"].to_numpy()
    assert (labels[:60] == "Astro").mean() > 0.95
    assert (labels[60:120] == "Oligo").mean() > 0.95
    assert (labels[120:180] == "Exc").mean() > 0.95
    assert labels[180] == "Unresolved"
    assert labels[181] == "Mixed"
    assert result["pseudo_broad"].iloc[130] == "Neuron"
    six = pseudo.resolved_six(result, "p7513_c")
    assert six[180] is None and six[181] is None


def test_resolved_six_rejects_an_unknown_method(pseudo: ModuleType) -> None:
    with pytest.raises(ValueError, match="unknown pseudo-label method"):
        pseudo.resolved_six(pd.DataFrame({"pseudo6": ["Astro"]}), "e1")


def test_h9_agreement_scores_confident_resolved_cells_on_six_classes(
    pseudo: ModuleType,
) -> None:
    names = np.array(
        [
            "Fibroblasts",  # -> Vascular, pseudo Vascular: agrees
            "Vascular cells",  # agrees
            "Microglia",  # pseudo Immune_other: resolved, disagrees
            "Neurons",  # not confident: not scored
            "Astrocytes",  # unresolved pseudo-label: not scored
            "Oligodendrocytes",  # disagrees (pseudo OPC)
        ],
        dtype=object,
    )
    confident = np.array([True, True, True, False, True, True])
    six = np.array(
        ["Vascular", "Vascular", "Immune_other", "Neuron", None, "OPC"], dtype=object
    )
    result = pseudo.h9_agreement(names, confident, six, "p7513_c")
    assert result.n_confident == 5
    assert result.n_scored == 4
    assert result.agreement == pytest.approx(0.5)
    assert result.by_class["Vascular"] == {"agreement": 1.0, "n": 2.0}
    assert result.by_class["Micro"]["agreement"] == 0.0
    row = result.to_row()
    assert row["agree_Vascular"] == 1.0 and row["n_Oligo"] == 1.0
    empty = pseudo.h9_agreement(names, np.zeros(6, bool), six, "p1212_4")
    assert empty.n_scored == 0 and math.isnan(empty.agreement)


# ---------------------------------------------------------------------------
# resolve_criteria


@pytest.mark.parametrize(
    ("criterion", "pair", "segmentation", "expected"),
    [
        ("H9", "P7513", "proseg_hybrid", True),
        ("H9", "P7113", "proseg_hybrid", False),
        ("H7", "P5011", "proseg_hybrid", True),
        ("H10", "P5011", "proseg_hybrid", False),
        ("H4", "P7113", "proseg_hybrid", True),
        ("H4", "P5011", "proseg_hybrid", False),
        ("H17/H1", "P1212", "reseg", True),
        ("H17/H9", "P7113", "reseg", False),
        ("H7", "P7513", "reseg", False),
        ("H17/H2", "P7513", "proseg_hybrid", False),
    ],
)
def test_scored_follows_the_human_flip_rule(
    criteria: ModuleType, criterion: str, pair: str, segmentation: str, expected: bool
) -> None:
    assert criteria.scored(criterion, pair, segmentation) is expected


def test_criterion_row_compares_with_the_threshold(criteria: ModuleType) -> None:
    low = criteria.criterion_row(
        "H2", "P7513", "proseg_hybrid", "S", 0.0102, 0.01, "<="
    )
    assert low["passes"] is False and low["scored"] is True
    edge = criteria.criterion_row("H7", "P7513", "proseg_hybrid", "S", 0.62, 0.62, ">=")
    assert edge["passes"] is True
    missing = criteria.criterion_row(
        "H9", "P7513", "proseg_hybrid", "S", math.nan, 0.9, ">="
    )
    assert missing["passes"] is None
    label_set = criteria.criterion_row(
        "H4[m4_resolve_heldout]", "P1212", "proseg_hybrid", "S", 5.0, 6.0, ">="
    )
    assert label_set["passes"] is False and label_set["scored"] is True


def _labels(n: int = 8) -> pd.DataFrame:
    cop = "Committed oligodendrocyte precursor"
    frame = pd.DataFrame(
        {
            "total_counts": [5, 25, 40, 80, 120, 30, 22, 60][:n],
            "mmc_whb_supercluster_name": [
                cop,
                cop,
                "Oligodendrocyte precursor",
                "Astrocyte",
                "Miscellaneous",
                "Astrocyte",
                "Upper-layer intratelencephalic",
                "Astrocyte",
            ][:n],
            "ct_broad_name": [
                None,
                "Oligodendrocyte precursors",
                "Oligodendrocyte precursors",
                "Astrocytes",
                None,
                "Astrocytes",
                "Neurons",
                "Astrocytes",
            ][:n],
            "ct_broad_status": [
                "low_counts",
                "confident",
                "confident",
                "confident",
                "implausible",
                "low_confidence",
                "confident",
                "confident",
            ][:n],
            "ct_lineage_status": ["confident"] * n,
            "ct_supercluster_name": [
                None,
                cop,
                None,
                "Astrocyte",
                None,
                None,
                None,
                None,
            ][:n],
            "ct_supercluster_status": [
                "low_counts",
                "confident",
                "low_confidence",
                "confident",
                "implausible",
                "parent_unresolved",
                "not_attempted_gate",
                "below_floor",
            ][:n],
            "flag_implausible": [False, False, False, False, True, False, False, False][
                :n
            ],
            "flag_cop_suppressed": [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ][:n],
            "mmc_seaad_subclass_name": [
                "OPC",
                "OPC",
                "OPC",
                "Astrocyte",
                "L2/3 IT",
                "Astrocyte",
                "L2/3 IT",
                "Microglia-PVM",
            ][:n],
            "mmc_whb_supercluster_runner_up_1_name": [
                "Oligodendrocyte",
                "Oligodendrocyte",
                cop,
                "Oligodendrocyte",
                "Upper-layer intratelencephalic",
                "Oligodendrocyte",
                "Deep-layer intratelencephalic",
                "Microglia",
            ][:n],
        },
        index=pd.Index([f"c{i}" for i in range(n)], name="cell_id"),
    )
    return frame


def test_cop_control_counts_confident_cop_and_opc(criteria: ModuleType) -> None:
    shares = criteria.cop_control(_labels())
    assert shares["cop_argmax_share"] == pytest.approx(2 / 8)
    assert shares["cop_confident_supercluster_share"] == pytest.approx(1 / 8)
    assert shares["opc_confident_broad_share"] == pytest.approx(2 / 8)
    assert shares["cop_derived_share_of_confident_opc"] == pytest.approx(0.5)
    assert shares["cop_suppressed_share"] == pytest.approx(1 / 8)


def _summary(level: str = "full", warning: bool = False) -> dict:
    return {
        "trust": {"state": "validated"},
        "resolution": {
            "degraded_mode": {"name": "whb_sea"},
            "gate": {
                "level": level,
                "warning": warning,
                "frac_ge30": 0.5,
                "level_reasons": [],
                "warning_reasons": [],
                "broad_coverage_segmented": 0.4,
            },
            "resolvability_extrapolated_share": 0.1,
        },
    }


def _resolved_sample(criteria: ModuleType, sample_id: str, **summary: object) -> object:
    labels = _labels()
    pair, platform = sample_id.split("_")
    sea = pd.DataFrame(
        {
            "broad": [
                "Oligodendrocyte precursors",
                "Oligodendrocyte precursors",
                "Oligodendrocyte precursors",
                "Astrocytes",
                "Neurons",
                "Astrocytes",
                "Neurons",
                "Microglia",
            ]
        },
        index=labels.index,
    )
    counts = sparse.csr_matrix(np.ones((len(labels), 3)))
    return criteria.ResolvedSample(
        pair=pair,
        segmentation="proseg_hybrid",
        sample_id=sample_id,
        platform=platform,
        labels=labels,
        summary=_summary(**summary),
        sea=sea,
        counts=counts,
        lognorm=counts,
        symbols=["AQP4", "GJA1", "SLC17A7"],
        legacy_broad=np.array(["Neurons"] * len(labels), dtype=object),
    )


def test_sample_criteria_and_table_rows(criteria: ModuleType) -> None:
    sample = _resolved_sample(criteria, "P7513_MERSCOPE")
    six = np.array(
        ["Astro", "OPC", "OPC", "Astro", None, "Astro", "Neuron", "Micro"], dtype=object
    )
    row, detail = criteria.sample_criteria(sample, {"p7513_c": six, "p1212_4": six})
    assert row["H2_implausible_share"] == pytest.approx(1 / 8)
    assert row["H7_broad_coverage_table"] == pytest.approx(5 / 8)
    # >= 20 counts: cells 1-7; WHB argmax broad vs SEA-AD broad.
    assert row["H3_n_ge20"] == 7
    # Confident broad cells 1, 2, 3, 6, 7 all agree with the pseudo-labels
    # except cell 7 (Astrocytes vs Micro).
    assert row["H9_p7513_c"] == pytest.approx(4 / 5)
    assert {item["method"] for item in detail} == {"p7513_c", "p1212_4"}
    table = criteria.table_rows_for_sample(row)
    by_criterion = {item["criterion"]: item for item in table}
    assert by_criterion["H2"]["passes"] is False
    assert by_criterion["H7"]["threshold"] == 0.62
    assert by_criterion["H7"]["passes"] is True
    assert by_criterion["H8"]["passes"] is True
    assert by_criterion["H9"]["passes"] is False
    assert "H17/H2" not in by_criterion


def test_h8_rows_expect_broad_only_and_note_an_extra_warning(
    criteria: ModuleType,
) -> None:
    for sample_id, level, warning, passes in (
        ("P5011_MERSCOPE", "broad_only", True, True),
        ("P1212_MERSCOPE", "broad_only", True, True),
        ("P1212_MERSCOPE", "full", False, False),
        ("P7113_XENIUM", "broad_only", False, False),
    ):
        sample = _resolved_sample(criteria, sample_id, level=level, warning=warning)
        row, _ = criteria.sample_criteria(sample, None)
        h8 = next(
            item
            for item in criteria.table_rows_for_sample(row)
            if item["criterion"] == "H8"
        )
        assert h8["passes"] is passes
        if sample_id == "P1212_MERSCOPE" and warning:
            assert "expectation is False" in h8["note"]


def test_h8_warning_rows_compare_the_flag_with_its_expectation(
    criteria: ModuleType,
) -> None:
    for sample_id, warning, segmentation, passes in (
        ("P5011_MERSCOPE", True, "proseg_hybrid", True),
        ("P1212_MERSCOPE", True, "proseg_hybrid", False),
        ("P1212_MERSCOPE", False, "proseg_hybrid", True),
        ("P7513_XENIUM", True, "proseg_hybrid", False),
        ("P1212_MERSCOPE", True, "reseg", True),
    ):
        sample = _resolved_sample(
            criteria, sample_id, level="broad_only", warning=warning
        )
        sample = dataclasses.replace(sample, segmentation=segmentation)
        row, _ = criteria.sample_criteria(sample, None)
        rows = {item["criterion"]: item for item in criteria.table_rows_for_sample(row)}
        warning_row = rows["H8/warning"]
        assert warning_row["passes"] is passes
        assert warning_row["value"] == pytest.approx(0.4)
        assert warning_row["scored"] is (
            segmentation == "proseg_hybrid" and rows["H8"]["scored"]
        )
    assert criteria.scored("H8/warning", "P1212", "proseg_hybrid") is True
    assert criteria.scored("H8/warning", "P1212", "reseg") is False


def test_h2_breakdown_splits_the_implausible_calls_by_node(
    criteria: ModuleType,
) -> None:
    sample = _resolved_sample(criteria, "P7513_MERSCOPE")
    labels = sample.labels.copy()
    labels.loc["c6", "flag_implausible"] = True
    labels.loc["c6", "mmc_whb_supercluster_name"] = "Amygdala excitatory"
    labels.loc["c7", "ct_lineage_status"] = "low_confidence"
    labels.loc["c7", "flag_implausible"] = True
    labels.loc["c7", "mmc_whb_supercluster_name"] = "Amygdala excitatory"
    rows = {
        row["node"]: row
        for row in criteria.h2_breakdown(dataclasses.replace(sample, labels=labels))
    }
    assert rows["all"]["n"] == 3
    assert rows["all"]["share_table"] == pytest.approx(3 / 8)
    amygdala = rows["Amygdala excitatory"]
    assert amygdala["kind"] == "region_implausible"
    assert amygdala["n"] == 2
    assert amygdala["lineage_confident_share"] == pytest.approx(0.5)
    assert amygdala["median_counts"] == pytest.approx(41.0)
    assert amygdala["h2_without_node"] == pytest.approx(1 / 8)
    assert json.loads(amygdala["seaad_subclass_top"]) == {
        "L2/3 IT": 0.5,
        "Microglia-PVM": 0.5,
    }
    assert rows["Miscellaneous"]["kind"] == "sink"
    assert math.isnan(rows["all"]["h2_without_node"])


def test_auroc_ceiling_bounds_the_held_out_auroc(criteria: ModuleType) -> None:
    from merxen.annotation.shadow import auroc

    assert criteria.auroc_ceiling(0.1, 0.02) == pytest.approx(0.1 + 0.5 * 0.9 * 0.98)
    assert criteria.auroc_ceiling(1.0, 0.0) == pytest.approx(1.0)
    assert math.isnan(criteria.auroc_ceiling(math.nan, 0.1))
    # Perfect labels, sparse detection: the AUROC reaches the ceiling, no more.
    rng = np.random.default_rng(0)
    assigned = np.r_[np.ones(400, bool), np.zeros(4000, bool)]
    detected = np.r_[rng.random(400) < 0.1, rng.random(4000) < 0.02]
    score = np.where(detected, 0.01 + rng.random(len(detected)) * 0.01, 0.0)
    score[assigned & detected] += 1.0
    ceiling = criteria.auroc_ceiling(
        float(detected[assigned].mean()), float(detected[~assigned].mean())
    )
    assert auroc(score, assigned) == pytest.approx(ceiling)
    assert ceiling < 0.70


def test_jsd_rows_add_the_h17_margin_on_reseg(criteria: ModuleType) -> None:
    summary = {
        "pair": {
            "jsd": [
                {
                    "kind": "soft",
                    "region": "whole_section",
                    "jsd": 0.18,
                    "ci_low": 0.17,
                    "ci_high": 0.19,
                    "resampling": "joint",
                },
                {"kind": "argmax", "region": "whole_section", "jsd": 0.2},
            ],
            "mask_note": "applied",
        }
    }
    rows = criteria.jsd_rows(summary, "P7513", "reseg")
    assert rows[0]["threshold"] == pytest.approx(0.19)
    assert rows[1]["threshold"] is None
    table = criteria.jsd_table_rows(rows)
    assert len(table) == 1
    assert table[0]["criterion"] == "H17/H1" and table[0]["passes"] is True
    proseg = criteria.jsd_table_rows(
        criteria.jsd_rows(summary, "P7513", "proseg_hybrid")
    )
    assert proseg[0]["criterion"] == "H1" and proseg[0]["passes"] is False


def test_h16_rows_check_that_strata_above_15_percent_are_marked(
    criteria: ModuleType,
) -> None:
    strata = [
        {"flag": "diffuse", "class": "A", "rate": 0.20, "informative_h16": False},
        {"flag": "diffuse", "class": "B", "rate": 0.10, "informative_h16": True},
        {"flag": "contaminated", "class": "C", "rate": None, "informative_h16": False},
    ]
    summary = {"samples": {"S": {"flags": {"strata": strata}}}}
    rows = criteria.h16_rows(summary, "P7513", "proseg_hybrid")
    assert [row["h16_marked_ok"] for row in rows] == [True, True, True]
    assert [row["above_h16"] for row in rows] == [True, False, False]
    table = criteria.h16_table_rows(pd.DataFrame(rows))
    assert table[0]["passes"] is True
    strata.append({"flag": "ood", "class": "D", "rate": 0.5, "informative_h16": True})
    bad = criteria.h16_table_rows(
        pd.DataFrame(criteria.h16_rows(summary, "P7513", "proseg_hybrid"))
    )
    assert bad[0]["passes"] is False and "1 of them not marked" in bad[0]["note"]


def test_referee_table_rows_take_the_new_vs_legacy_comparison(
    criteria: ModuleType,
) -> None:
    rows = [
        {
            "pair": "P7513",
            "segmentation": "proseg_hybrid",
            "sample_id": "P7513_MERSCOPE",
            "first": "new confident (RESOLVE)",
            "second": "legacy broad_class",
            "first_wins": 0.98,
            "n_disputes": 10,
        },
        {
            "pair": "P7513",
            "segmentation": "proseg_hybrid",
            "sample_id": "P7513_MERSCOPE",
            "first": "WHB argmax",
            "second": "SEA-AD argmax",
            "first_wins": 0.4,
            "n_disputes": 10,
        },
    ]
    table = criteria.referee_table_rows(rows)
    assert len(table) == 1 and table[0]["passes"] is True


@dataclasses.dataclass(frozen=True)
class _FakeRun:
    record: SimpleNamespace
    tidy: object


def test_patched_runs_swap_the_primary_tidy_and_drop_sea_when_whb_only(
    criteria: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = {
        "whb": _FakeRun(
            SimpleNamespace(role="primary", purposes=["annotation"], build_hash="b1"),
            "production",
        ),
        "setc": _FakeRun(
            SimpleNamespace(
                role="primary", purposes=["xplat_sensitivity"], build_hash="b2"
            ),
            "setc",
        ),
        "sea": _FakeRun(
            SimpleNamespace(role="secondary", purposes=["annotation"], build_hash="b3"),
            "sea",
        ),
    }
    monkeypatch.setattr(criteria.pipeline, "load_resolve_runs", lambda *a, **k: runs)
    monkeypatch.setattr(
        criteria, "read_tidy_parquet", lambda path: ("heldout", {"build_hash": "b1"})
    )
    sample = SimpleNamespace(sample_id="S")
    both = criteria._patched_runs({"S": Path("x")}, whb_only=False)("map", sample)
    assert both["whb"].tidy == "heldout"
    assert both["setc"].tidy == "setc" and both["sea"].tidy == "sea"
    alone = criteria._patched_runs({"S": Path("x")}, whb_only=True)("map", sample)
    assert set(alone) == {"whb", "setc"}
    monkeypatch.setattr(
        criteria, "read_tidy_parquet", lambda path: ("heldout", {"build_hash": "zz"})
    )
    with pytest.raises(ValueError, match="another bundle"):
        criteria._patched_runs({"S": Path("x")}, whb_only=False)("map", sample)


def test_h4_summary_counts_classes_passing(criteria: ModuleType) -> None:
    enrichment = pd.DataFrame(
        {
            "pair": ["P7513"] * 7 + ["P5011"] * 7 + ["P7513"] * 7,
            "platform": ["MERSCOPE"] * 21,
            "label_set": ["m4_resolve_heldout"] * 14
            + ["diag_m4_resolve_heldout_ge30"] * 7,
            "broad_class": [f"c{i}" for i in range(7)] * 3,
            "fold": [5.0] * 21,
            "auroc": [0.8] * 6 + [0.6] + [0.6] * 7 + [0.8] * 7,
            "passes": [True] * 6 + [False] + [False] * 7 + [True] * 7,
            # c6 of P7513 fails at its ceiling (0.1 + 0.5 * 0.9 * 1.0 = 0.55).
            "detection_assigned": [0.9] * 6 + [0.1] + [0.9] * 7 + [0.9] * 7,
            "detection_other": [0.1] * 6 + [0.0] + [0.1] * 7 + [0.1] * 7,
        }
    )
    summary = {
        (row["pair"], row["label_set"]): row for row in criteria.h4_summary(enrichment)
    }
    main = summary[("P7513", "m4_resolve_heldout")]
    assert main["n_pass"] == 6 and main["h4_pass"] is True
    assert main["diagnostic"] is False
    assert main["failing_at_auroc_ceiling"] == "c6"
    assert summary[("P5011", "m4_resolve_heldout")]["h4_scored_pair"] is False
    assert summary[("P5011", "m4_resolve_heldout")]["failing_at_auroc_ceiling"] == ""
    assert summary[("P7513", "diag_m4_resolve_heldout_ge30")]["diagnostic"] is True
    rows = criteria.h4_table_rows(summary.values())
    assert {row["criterion"] for row in rows} == {"H4[m4_resolve_heldout]"}
    assert len(rows) == 2
    assert "c6 (fold 5.0, AUROC 0.60)" in main["failing"]


def test_scripts_help_runs(criteria: ModuleType, pseudo: ModuleType) -> None:
    with pytest.raises(SystemExit) as raised:
        criteria.main(["--help"])
    assert raised.value.code == 0
    assert pseudo.main() == 0


def test_main_requires_the_h4_inputs(criteria: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        criteria.main(
            [
                "--resolve-root",
                str(tmp_path),
                "--runs-root",
                str(tmp_path),
                "--results-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
                "--gene-id-fallback-csv",
                str(tmp_path / "fb.h5ad"),
            ]
        )
