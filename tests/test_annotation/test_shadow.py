"""Tests for the M3 shadow evaluation helpers (plan §5.2-§5.5, §12 M3)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial.distance import jensenshannon

from merxen.annotation.config import AnnotationGate
from merxen.annotation.shadow import (
    COMPOSITION_COLUMNS,
    E1_REFEREE_MARKERS,
    GLIAL_COLUMNS,
    OPC,
    SECOND_VOTES,
    UNALLOCATED,
    FloorLookup,
    HumanRuleInputs,
    argmax_broad_names,
    auroc,
    beta_binomial_upper_tail,
    block_bootstrap_jsd,
    broad_class_index,
    class_profiles,
    composition_shares,
    contamination_flags,
    dataset_gate,
    depth_grid,
    distinct_gene_quantiles,
    evaluate_human_rules,
    expected_genes_quantile,
    fit_beta_binomial,
    foreign_marker_fraction,
    heldout_enrichment,
    jensen_shannon_distance,
    label_agreement,
    marker_class_scores,
    marker_referee,
    mouse_confidence,
    negative_counts,
    negative_gene_mask,
    occupied_area_mm2,
    one_hot_broad_matrix,
    paired_block_bootstrap_jsd_difference,
    profile_matrix,
    realised_rates,
    reference_pseudobulk_log2_factors,
    rescale_counts,
    rule_inputs_from_provisional,
    seaad_broad_calls,
    seaad_soft_broad_matrix,
    select_heldout_markers,
    soft_broad_matrix,
    soft_matrix_from_provisional,
    tile_codes,
    tile_sums,
    whb_broad_of,
    whb_labels_from_tidy,
)
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, UNASSIGNED_LABEL

UNALLOCATED_INDEX = len(HUMAN_BROAD_CLASSES)


# --------------------------------------------------------------------------
# Composition and JSD


def test_whb_broad_of_sends_sinks_to_unassigned() -> None:
    broad_of = whb_broad_of()
    assert broad_of["Astrocyte"] == "Astrocytes"
    assert broad_of["Upper-layer intratelencephalic"] == "Neurons"
    assert broad_of["Committed oligodendrocyte precursor"] == (
        "Oligodendrocyte precursors"
    )
    assert broad_of["Splatter"] == UNASSIGNED_LABEL
    assert broad_of["Miscellaneous"] == UNASSIGNED_LABEL


def test_broad_class_index_maps_unknown_and_missing_to_unallocated() -> None:
    index = broad_class_index(
        ["Astrocyte", "Splatter", None, float("nan"), "not a node"], whb_broad_of()
    )
    assert index[0] == HUMAN_BROAD_CLASSES.index("Astrocytes")
    assert list(index[1:]) == [UNALLOCATED_INDEX] * 4


def test_soft_broad_matrix_aggregates_runner_ups_and_keeps_residual() -> None:
    names = np.array(
        [
            ["Astrocyte", "Oligodendrocyte", "Splatter"],
            ["Upper-layer intratelencephalic", "CGE interneuron", None],
            [None, None, None],
        ],
        dtype=object,
    )
    probabilities = np.array(
        [[0.6, 0.2, 0.1], [0.5, 0.3, np.nan], [np.nan, np.nan, np.nan]]
    )
    matrix = soft_broad_matrix(names, probabilities, broad_of=whb_broad_of())
    assert matrix.shape == (3, len(COMPOSITION_COLUMNS))
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    astro = HUMAN_BROAD_CLASSES.index("Astrocytes")
    oligo = HUMAN_BROAD_CLASSES.index("Oligodendrocytes")
    neurons = HUMAN_BROAD_CLASSES.index("Neurons")
    assert matrix[0, astro] == pytest.approx(0.6)
    assert matrix[0, oligo] == pytest.approx(0.2)
    # Splatter (a sink) and the residual both go to unallocated.
    assert matrix[0, UNALLOCATED_INDEX] == pytest.approx(0.2)
    assert matrix[1, neurons] == pytest.approx(0.8)
    assert matrix[2, UNALLOCATED_INDEX] == pytest.approx(1.0)


def test_soft_broad_matrix_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="one shape"):
        soft_broad_matrix(
            np.array([["Astrocyte"]], dtype=object),
            np.array([[0.5, 0.5]]),
            broad_of=whb_broad_of(),
        )


def test_soft_broad_matrix_renormalises_mass_above_one() -> None:
    names = np.array([["Astrocyte", "Microglia"]], dtype=object)
    matrix = soft_broad_matrix(names, np.array([[0.9, 0.3]]), broad_of=whb_broad_of())
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, UNALLOCATED_INDEX] == pytest.approx(0.0)


def test_one_hot_broad_matrix_excludes_cells_outside_include() -> None:
    matrix = one_hot_broad_matrix(
        ["Neurons", "Mixed/Unknown", "Astrocytes"],
        include=np.array([True, True, False]),
    )
    assert matrix[0, HUMAN_BROAD_CLASSES.index("Neurons")] == 1.0
    assert matrix[1, UNALLOCATED_INDEX] == 1.0
    assert matrix[2].sum() == 0.0


def test_jensen_shannon_distance_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        p = rng.random(len(COMPOSITION_COLUMNS))
        q = rng.random(len(COMPOSITION_COLUMNS))
        expected = jensenshannon(p[:7] / p[:7].sum(), q[:7] / q[:7].sum(), base=2)
        assert jensen_shannon_distance(p, q) == pytest.approx(expected)


def test_jensen_shannon_distance_bounds_and_empty() -> None:
    same = np.array([1, 2, 3, 0, 0, 0, 0, 5.0])
    assert jensen_shannon_distance(same, same * 3) == pytest.approx(0.0)
    disjoint_a = np.array([1.0, 0, 0, 0, 0, 0, 0])
    disjoint_b = np.array([0, 1.0, 0, 0, 0, 0, 0])
    assert jensen_shannon_distance(disjoint_a, disjoint_b) == pytest.approx(1.0)
    assert math.isnan(jensen_shannon_distance(np.zeros(7), disjoint_a))


def test_jensen_shannon_distance_is_vectorised_over_rows() -> None:
    first = np.array([[1.0, 1, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0]])
    second = np.array([[1.0, 1, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0]])
    np.testing.assert_allclose(jensen_shannon_distance(first, second), [0.0, 1.0])


def test_composition_shares_include_unallocated() -> None:
    matrix = one_hot_broad_matrix(["Neurons", "Neurons", "Mixed/Unknown", "Astrocytes"])
    shares = composition_shares(matrix)
    assert shares["Neurons"] == pytest.approx(0.5)
    assert shares[UNALLOCATED] == pytest.approx(0.25)
    assert math.isnan(composition_shares(np.zeros((2, 8)))["Neurons"])


# --------------------------------------------------------------------------
# Tiles and the block bootstrap


def test_tile_codes_group_cells_into_square_tiles() -> None:
    xy = np.array([[10.0, 10.0], [499.0, 20.0], [501.0, 20.0], [np.nan, 1.0]])
    codes = tile_codes(xy, 500.0)
    assert codes[0] == codes[1]
    assert codes[2] != codes[0]
    assert codes[3] == -1
    with pytest.raises(ValueError, match="positive"):
        tile_codes(xy, 0)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        tile_codes(np.zeros((3, 3)))


def test_tile_sums_add_rows_per_tile_and_drop_unplaced_cells() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [9.0, 9.0]])
    sums = tile_sums(matrix, np.array([0, 0, 1, -1]))
    np.testing.assert_allclose(sums, [[1.0, 2.0], [3.0, 0.0]])


def _section(rng: np.random.Generator, n_tiles: int, weights: np.ndarray) -> np.ndarray:
    return rng.multinomial(200, weights, size=n_tiles).astype(float)


def test_block_bootstrap_jsd_is_reproducible_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    first = _section(rng, 40, np.array([0.4, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0]))
    second = _section(rng, 30, np.array([0.3, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05, 0]))
    result = block_bootstrap_jsd(first, second, n_reps=200, seed=1)
    again = block_bootstrap_jsd(first, second, n_reps=200, seed=1)
    np.testing.assert_array_equal(result.replicates, again.replicates)
    assert result.n_reps == 200
    assert (result.n_tiles_a, result.n_tiles_b) == (40, 30)
    assert result.ci_low <= result.jsd <= result.ci_high
    assert result.jsd == pytest.approx(
        jensen_shannon_distance(first.sum(axis=0), second.sum(axis=0))
    )


def test_block_bootstrap_jsd_needs_tiles_in_both_sections() -> None:
    with pytest.raises(ValueError, match="non-empty tile"):
        block_bootstrap_jsd(np.zeros((3, 8)), np.ones((3, 8)))


# --------------------------------------------------------------------------
# SEA-AD broad calls


def _level(
    names: list[str | None], bp: list[float], runners: list[list[tuple[str, float]]]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"name": names, "bp": bp},
        index=pd.Index([f"c{i}" for i in range(len(names))], name="cell_id"),
    )
    for rank in range(1, 6):
        frame[f"runner_up_{rank}_name"] = [
            row[rank - 1][0] if len(row) >= rank else None for row in runners
        ]
        frame[f"runner_up_{rank}_probability"] = [
            row[rank - 1][1] if len(row) >= rank else np.nan for row in runners
        ]
    return frame


def test_seaad_broad_calls_aggregate_runner_ups_and_split_vlmc() -> None:
    subclass = _level(
        [
            "Astrocyte",
            "VLMC & Perivascular",
            "VLMC & Perivascular",
            "Ependymal",
            "L2/3 IT",
        ],
        [0.6, 0.7, 0.8, 0.9, 0.5],
        [
            [("Oligodendrocyte", 0.3)],
            [("Endothelial", 0.2)],
            [("Endothelial", 0.1)],
            [],
            [("L4 IT", 0.3), ("Astrocyte", 0.1)],
        ],
    )
    supertype = _level(
        ["Astro_2", "VLMC_1", "Pericyte_1", "Ependymal_1", "L2/3 IT_1"],
        [0.9, 0.6, 0.5, 1.0, 0.4],
        [[], [("Pericyte_1", 0.3)], [("SMC_1", 0.4), ("VLMC_2", 0.1)], [], []],
    ).iloc[::-1]
    calls = seaad_broad_calls(subclass, supertype)
    assert list(calls["broad"]) == [
        "Astrocytes",
        "Fibroblasts",
        "Vascular cells",
        UNASSIGNED_LABEL,
        "Neurons",
    ]
    raw = calls["broad_raw"].to_numpy()
    assert raw[0] == pytest.approx(0.6)
    # VLMC & Perivascular: subclass mass (0.7; Endothelial is vascular, not
    # fibroblast) times the fibroblast supertype mass (VLMC_1 0.6).
    assert raw[1] == pytest.approx(0.7 * 0.6)
    # Vascular: (0.8 + 0.1 Endothelial) x (Pericyte_1 0.5 + SMC_1 0.4).
    assert raw[2] == pytest.approx(0.9 * 0.9)
    assert raw[3] == 0.0
    assert raw[4] == pytest.approx(0.8)


def test_seaad_broad_calls_without_supertypes_leave_vlmc_unassigned() -> None:
    subclass = _level(["VLMC & Perivascular"], [0.9], [[]])
    calls = seaad_broad_calls(subclass)
    assert calls["broad"].iloc[0] == UNASSIGNED_LABEL
    assert calls["broad_raw"].iloc[0] == 0.0


# --------------------------------------------------------------------------
# Floors, rules and the gate


def test_packaged_floors_follow_the_table() -> None:
    floors = FloorLookup.packaged()
    names = np.array(
        [
            "MGE interneuron",
            "Committed oligodendrocyte precursor",
            "Splatter",
            "Astrocyte",
        ],
        dtype=object,
    )
    np.testing.assert_array_equal(
        floors.per_cell("broad", "XENIUM", names), [30, 10, math.inf, 10]
    )
    np.testing.assert_array_equal(
        floors.per_cell("supercluster", "MERSCOPE", names), [30, 120, math.inf, 10]
    )
    with pytest.raises(KeyError):
        floors.per_cell("broad", "CosMx", names)
    with pytest.raises(ValueError, match="no rows"):
        FloorLookup.packaged("unknown_family")


def _inputs(rows: list[dict[str, object]]) -> HumanRuleInputs:
    frame = pd.DataFrame(rows)

    def column(name: str) -> np.ndarray:
        return frame[name].to_numpy()

    return HumanRuleInputs(
        total_counts=column("counts").astype(float),
        whb_supercluster=column("super"),
        whb_supercluster_bp=column("super_bp").astype(float),
        whb_lineage=column("lineage"),
        whb_lineage_raw=column("raw").astype(float),
        whb_broad=column("broad"),
        whb_broad_raw=column("raw").astype(float),
        sea_broad=column("sea"),
        sea_broad_raw=column("sea_raw").astype(float),
        ll_broad=column("ll"),
    )


def _cell(
    counts: int,
    supercluster: str,
    lineage: str,
    broad: str,
    *,
    raw: float = 0.95,
    super_bp: float = 0.9,
    sea: str | None = None,
    sea_raw: float = 0.9,
    ll: str | None = None,
) -> dict[str, object]:
    return {
        "counts": counts,
        "super": supercluster,
        "super_bp": super_bp,
        "lineage": lineage,
        "broad": broad,
        "raw": raw,
        "sea": sea if sea is not None else broad,
        "sea_raw": sea_raw,
        "ll": ll,
    }


ASTRO = ("Astrocyte", "Astrocytes", "Astrocytes")
COP = (
    "Committed oligodendrocyte precursor",
    "Oligodendrocyte lineage",
    "Oligodendrocyte precursors",
)
INH = ("MGE interneuron", "Neurons", "Neurons")


def test_rules_second_vote_variants() -> None:
    rows = [
        _cell(100, *ASTRO, ll="Astro"),  # agreement everywhere
        _cell(40, *ASTRO, sea="Microglia", ll="Astro"),  # SEA disagrees below 60
        _cell(
            100, *ASTRO, sea="Microglia", sea_raw=0.9, ll="Immune"
        ),  # confident disagreement
        _cell(
            100, *ASTRO, sea="Microglia", sea_raw=0.3, ll="Immune"
        ),  # unconfident disagreement
        _cell(40, *INH, sea="Neurons", ll="Exc"),  # LL lenient: Exc agrees with neurons
    ]
    inputs = _inputs(rows)
    expected = {
        "none": [True, True, True, True, True],
        "seaad_from60": [True, True, False, True, True],
        "seaad": [True, False, False, True, True],
        "ll_below60": [True, True, True, True, True],
        "seaad_ll_below60": [True, True, False, True, True],
        "seaad_or_ll": [True, True, False, True, True],
    }
    assert set(expected) == set(SECOND_VOTES)
    for vote, flags in expected.items():
        result = evaluate_human_rules(inputs, platform="MERSCOPE", second_vote=vote)  # type: ignore[arg-type]
        assert list(result.broad_confident) == flags, vote


def test_rules_ll_vote_blocks_disagreement_below_60() -> None:
    inputs = _inputs([_cell(40, *ASTRO, ll="Oligo"), _cell(40, *ASTRO, ll=None)])
    result = evaluate_human_rules(inputs, platform="MERSCOPE", second_vote="ll_below60")
    assert list(result.broad_confident) == [False, False]


def test_rules_cop_rule_and_platform_floors() -> None:
    rows = [
        # Below 120 counts; SEA agrees but not confidently.
        _cell(50, *COP, sea="Oligodendrocyte precursors", sea_raw=0.5),
        _cell(150, *COP, super_bp=0.8),  # passes the COP supercluster rule
        _cell(
            50, *COP, super_bp=0.5, sea="Oligodendrocyte precursors", ll="OPC"
        ),  # SEA OPC
        _cell(20, *INH),  # Inh broad floor: Xenium 30, MERSCOPE 10
    ]
    inputs = _inputs(rows)
    merscope = evaluate_human_rules(inputs, platform="MERSCOPE")
    xenium = evaluate_human_rules(inputs, platform="XENIUM")
    assert list(merscope.cop_suppressed) == [True, False, False, False]
    assert list(merscope.lineage_confident) == [True, True, True, True]
    assert list(merscope.broad_confident) == [False, True, True, True]
    assert merscope.broad_name[1] == "Oligodendrocyte precursors"
    assert merscope.broad_name[0] is None
    assert list(xenium.broad_confident) == [False, True, True, False]
    # Without SEA-AD's roles the confident SEA OPC call no longer rescues COP.
    ll_only = evaluate_human_rules(
        inputs, platform="MERSCOPE", second_vote="ll_below60"
    )
    assert ll_only.lineage_confident[2]
    assert ll_only.cop_suppressed[2]
    assert not ll_only.broad_confident[2]


def test_rules_implausible_nodes_keep_lineage_only_with_agreement() -> None:
    rows = [
        _cell(100, "Hippocampal CA1-3", "Neurons", "Neurons", sea="Neurons"),
        _cell(100, "Hippocampal CA1-3", "Neurons", "Neurons", sea="Astrocytes"),
        _cell(100, "Splatter", "Neurons", UNASSIGNED_LABEL, sea="Neurons"),
    ]
    result = evaluate_human_rules(_inputs(rows), platform="MERSCOPE")
    assert list(result.implausible) == [True, True, True]
    assert list(result.lineage_confident) == [True, False, True]
    assert not result.broad_confident.any()
    none_vote = evaluate_human_rules(
        _inputs(rows), platform="MERSCOPE", second_vote="none"
    )
    assert not none_vote.lineage_confident.any()


def test_rules_thresholds_and_hard_floor() -> None:
    rows = [
        _cell(100, *ASTRO, raw=0.72),
        _cell(100, *ASTRO, raw=0.73),
        _cell(9, *ASTRO),
        _cell(100, *ASTRO, super_bp=0.68),
    ]
    result = evaluate_human_rules(_inputs(rows), platform="MERSCOPE")
    assert list(result.broad_confident) == [False, True, False, True]
    assert list(result.supercluster_confident) == [False, True, False, False]
    assert list(result.final_level()) == ["none", "supercluster", "none", "broad"]


def test_rules_need_the_second_method() -> None:
    inputs = _inputs([_cell(100, *ASTRO)])
    bare = HumanRuleInputs(
        total_counts=inputs.total_counts,
        whb_supercluster=inputs.whb_supercluster,
        whb_supercluster_bp=inputs.whb_supercluster_bp,
        whb_lineage=inputs.whb_lineage,
        whb_lineage_raw=inputs.whb_lineage_raw,
        whb_broad=inputs.whb_broad,
        whb_broad_raw=inputs.whb_broad_raw,
    )
    assert len(bare) == 1
    with pytest.raises(ValueError, match="SEA-AD"):
        evaluate_human_rules(bare, platform="MERSCOPE", second_vote="seaad")
    with pytest.raises(ValueError, match="likelihood"):
        evaluate_human_rules(bare, platform="MERSCOPE", second_vote="ll_below60")
    with pytest.raises(ValueError, match="second_vote"):
        evaluate_human_rules(bare, platform="MERSCOPE", second_vote="both")  # type: ignore[arg-type]
    assert evaluate_human_rules(
        bare, platform="MERSCOPE", second_vote="none"
    ).broad_confident[0]


def test_rules_gate_blocks_superclusters_on_shallow_data() -> None:
    rows = [_cell(20, *ASTRO)] * 8 + [_cell(100, *ASTRO)] * 2
    result = evaluate_human_rules(_inputs(rows), platform="MERSCOPE")
    assert result.gate.level == "broad_only"
    assert result.broad_confident.all()
    assert not result.supercluster_confident.any()


def test_dataset_gate_truth_table() -> None:
    counts = np.array([10, 20, 30, 40])
    confident = np.array([True, True, False, False])
    full = dataset_gate(counts, confident, n_segmented=8)
    assert (full.level, full.warning) == ("full", False)
    assert full.frac_ge30 == 0.5
    assert full.broad_coverage_segmented == 0.25
    broad_only = dataset_gate(np.array([10, 20, 20, 40]), confident)
    assert broad_only.level == "broad_only"
    assert math.isnan(broad_only.broad_coverage_segmented)
    failed = dataset_gate(np.array([10, 20, 30, 40, 50]), np.eye(5, dtype=bool)[0])
    assert failed.level == "failed"
    warned = dataset_gate(counts, confident, n_segmented=40)
    assert warned.warning and warned.level == "full"
    assert any("segmented" in reason for reason in warned.reasons)
    strict = dataset_gate(counts, confident, gate=AnnotationGate(min_frac_ge30=0.8))
    assert strict.level == "broad_only"


# --------------------------------------------------------------------------
# Agreement and referee


def test_label_agreement_counts_out_of_class_labels_as_e1() -> None:
    first = ["Neurons", "Astrocytes", "Mixed/Unknown", "Splatter", None]
    second = ["Neurons", "Microglia", "Mixed/Unknown", None, "Neurons"]
    share, n_cells = label_agreement(first, second)
    assert (share, n_cells) == (pytest.approx(3 / 5), 5)
    strict, _ = label_agreement(first, second, strict=True)
    assert strict == pytest.approx(1 / 5)
    masked, n_masked = label_agreement(first, second, np.array([1, 1, 0, 0, 0], bool))
    assert (masked, n_masked) == (0.5, 2)
    assert math.isnan(label_agreement([], [])[0])


def test_marker_referee_scores_disputes_with_canonical_markers() -> None:
    symbols = ["AQP4", "GJA1", "CX3CR1", "SLC17A7", "OTHER"]
    counts = sparse.csr_matrix(
        np.array(
            [
                [5, 3, 0, 0, 2],  # astrocyte markers
                [0, 0, 4, 0, 1],  # microglia markers
                [0, 0, 0, 6, 0],  # neuron, labels agree
                [1, 0, 1, 0, 0],  # tie
            ]
        )
    )
    scores, present = marker_class_scores(counts, symbols)
    assert present["Astrocytes"] == ("AQP4", "GJA1")
    assert present["Fibroblasts"] == ()
    astro = list(E1_REFEREE_MARKERS).index("Astrocytes")
    assert scores[0, astro] == pytest.approx(0.8)
    result = marker_referee(
        scores,
        ["Astrocytes", "Astrocytes", "Neurons", "Astrocytes"],
        ["Microglia", "Microglia", "Neurons", "Microglia"],
    )
    assert result.n_cells == 4
    assert result.n_disputes == 3
    assert result.first_wins == pytest.approx(1 / 3)
    assert result.second_wins == pytest.approx(1 / 3)
    assert result.undecided == pytest.approx(1 / 3)
    assert result.consensus_top_marker_ok == 1.0
    assert result.top_disputes == {"Astrocytes | Microglia": 3}


def test_marker_referee_ignores_labels_outside_the_classes() -> None:
    scores = np.zeros((2, len(E1_REFEREE_MARKERS)))
    result = marker_referee(scores, ["Mixed/Unknown", None], ["Neurons", "Neurons"])
    assert result.n_disputes == 0
    assert math.isnan(result.first_wins)
    assert math.isnan(result.consensus_top_marker_ok)


# --------------------------------------------------------------------------
# Inputs from provisional tables


def _provisional() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "total_counts": [100, 40, 15],
            "ct_lineage_name": ["Astrocytes", "Neurons", "Neurons"],
            "ct_lineage_raw": [0.9, 0.8, 0.3],
            "ct_broad_name": ["Astrocytes", "Neurons", UNASSIGNED_LABEL],
            "ct_broad_raw": [0.9, 0.8, 0.3],
            "mmc_whb_supercluster_name": ["Astrocyte", "CGE interneuron", "Splatter"],
            "mmc_whb_supercluster_bp": [0.9, 0.5, 0.3],
            "mmc_whb_supercluster_runner_up_1_name": [
                None,
                "MGE interneuron",
                "Astrocyte",
            ],
            "mmc_whb_supercluster_runner_up_1_bp": [np.nan, 0.3, 0.2],
        },
        index=pd.Index(["a", "b", "c"], name="cell_id"),
    )
    return frame


def test_rule_inputs_from_provisional_align_second_methods_by_cell() -> None:
    labels = _provisional()
    sea = pd.DataFrame(
        {"broad": ["Neurons", "Astrocytes"], "broad_raw": [0.9, 0.95]},
        index=["b", "a"],
    )
    ll = pd.Series(["Astro", "Inh"], index=["a", "b"])
    inputs = rule_inputs_from_provisional(labels, sea, ll)
    assert list(inputs.sea_broad) == ["Astrocytes", "Neurons", None]
    np.testing.assert_allclose(inputs.sea_broad_raw[:2], [0.95, 0.9])
    assert list(inputs.ll_broad) == ["Astro", "Inh", None]
    assert list(inputs.whb_supercluster) == ["Astrocyte", "CGE interneuron", "Splatter"]
    result = evaluate_human_rules(inputs, platform="MERSCOPE")
    assert list(result.broad_confident) == [True, True, False]
    bare = rule_inputs_from_provisional(labels)
    assert bare.sea_broad is None and bare.ll_broad is None


def test_soft_and_argmax_from_provisional() -> None:
    labels = _provisional()
    soft = soft_matrix_from_provisional(labels)
    neurons = HUMAN_BROAD_CLASSES.index("Neurons")
    astro = HUMAN_BROAD_CLASSES.index("Astrocytes")
    assert soft[1, neurons] == pytest.approx(0.8)
    assert soft[2, astro] == pytest.approx(0.2)
    assert soft[2, UNALLOCATED_INDEX] == pytest.approx(0.8)
    assert list(argmax_broad_names(labels)) == [
        "Astrocytes",
        "Neurons",
        UNASSIGNED_LABEL,
    ]


# --------------------------------------------------------------------------
# M3 stage C2: LL vote, SEA-AD soft composition, paired JSD, held-out genes,
# X1 factors, flags and E8 helpers


def test_rules_seaad_or_ll_vote_with_the_seven_class_scheme() -> None:
    rows = [
        _cell(40, *ASTRO, sea="Microglia", ll="Astrocytes"),  # LL rescues below 60
        _cell(40, *ASTRO, sea="Microglia", ll="Microglia"),  # neither agrees
        _cell(40, *ASTRO, sea="Astrocytes", ll=None),  # SEA agrees
        _cell(40, *INH, sea="Astrocytes", ll="Neurons"),  # LL agrees at 7 classes
    ]
    base = _inputs(rows)
    inputs = HumanRuleInputs(
        **{**base.__dict__, "ll_scheme": "broad7"}  # type: ignore[arg-type]
    )
    v11 = evaluate_human_rules(inputs, platform="MERSCOPE", second_vote="seaad_or_ll")
    assert list(v11.broad_confident) == [True, False, True, True]
    v1 = evaluate_human_rules(inputs, platform="MERSCOPE", second_vote="seaad")
    assert list(v1.broad_confident) == [False, False, True, False]
    ll_only = evaluate_human_rules(
        inputs, platform="MERSCOPE", second_vote="ll_below60"
    )
    assert list(ll_only.broad_confident) == [True, False, False, True]
    with pytest.raises(ValueError, match="likelihood"):
        evaluate_human_rules(
            HumanRuleInputs(**{**base.__dict__, "ll_broad": None}),  # type: ignore[arg-type]
            platform="MERSCOPE",
            second_vote="seaad_or_ll",
        )


def test_rule_inputs_from_provisional_carry_the_ll_scheme() -> None:
    labels = _provisional()
    inputs = rule_inputs_from_provisional(
        labels, ll_broad=pd.Series(["Astrocytes"], index=["a"]), ll_scheme="broad7"
    )
    assert inputs.ll_scheme == "broad7"


def test_jensen_shannon_distance_on_a_class_subset() -> None:
    p = np.array([0.5, 0.2, 0.3, 0, 0, 0, 0, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0, 0.0])
    assert jensen_shannon_distance(p, q, columns=[1, 2]) == pytest.approx(0.0)
    assert jensen_shannon_distance(p, q) > 0
    assert math.isnan(jensen_shannon_distance(p, q, columns=[3, 4]))


def test_seaad_soft_broad_matrix_splits_vlmc_by_supertype() -> None:
    subclass = _level(
        ["Astrocyte", "VLMC & Perivascular", "L2/3 IT"],
        [0.6, 0.8, 0.5],
        [
            [("Oligodendrocyte", 0.3)],
            [("Endothelial", 0.1)],
            [("VLMC & Perivascular", 0.2), ("Astrocyte", 0.1)],
        ],
    )
    supertype = _level(
        ["Astro_2", "Pericyte_1", "L2/3 IT_1"],
        [0.9, 0.75, 0.4],
        [[], [("VLMC_1", 0.25)], []],
    )
    matrix = seaad_soft_broad_matrix(subclass, supertype)
    column = {name: index for index, name in enumerate(COMPOSITION_COLUMNS)}
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix[0, column["Astrocytes"]] == pytest.approx(0.6)
    assert matrix[0, column["Oligodendrocytes"]] == pytest.approx(0.3)
    # The assigned VLMC & Perivascular mass splits 0.75 / 0.25 by supertype.
    assert matrix[1, column["Vascular cells"]] == pytest.approx(0.8 * 0.75 + 0.1)
    assert matrix[1, column["Fibroblasts"]] == pytest.approx(0.8 * 0.25)
    # As a runner-up it is unallocated.
    assert matrix[2, column["Neurons"]] == pytest.approx(0.5)
    assert matrix[2, column[UNALLOCATED]] == pytest.approx(0.4)
    no_supertype = seaad_soft_broad_matrix(subclass)
    assert no_supertype[1, column[UNALLOCATED]] == pytest.approx(0.9)


def test_paired_bootstrap_difference_is_zero_for_identical_labellings() -> None:
    rng = np.random.default_rng(0)
    tiles_a = rng.random((12, 8))
    tiles_b = rng.random((10, 8))
    same = paired_block_bootstrap_jsd_difference(
        tiles_a, tiles_b, tiles_a, tiles_b, n_reps=50
    )
    assert same.difference == 0.0
    assert same.ci_low == same.ci_high == 0.0
    shifted = tiles_b.copy()
    shifted[:, 0] += 5.0
    worse = paired_block_bootstrap_jsd_difference(
        tiles_a, shifted, tiles_a, tiles_b, n_reps=50
    )
    assert worse.difference > 0
    assert worse.ci_low > 0
    assert worse.share_positive == 1.0
    assert worse.first_ci[0] <= worse.first_jsd <= worse.first_ci[1]
    glia = paired_block_bootstrap_jsd_difference(
        tiles_a, shifted, tiles_a, tiles_b, columns=GLIAL_COLUMNS, n_reps=20
    )
    assert glia.difference == pytest.approx(0.0)
    with pytest.raises(ValueError, match="same tiles"):
        paired_block_bootstrap_jsd_difference(tiles_a, tiles_b, tiles_a[:3], tiles_b)
    with pytest.raises(ValueError, match="non-empty"):
        paired_block_bootstrap_jsd_difference(
            np.zeros((2, 8)), tiles_b, np.zeros((2, 8)), tiles_b
        )


def test_block_bootstrap_jsd_on_a_class_subset() -> None:
    tiles = np.ones((4, 8))
    result = block_bootstrap_jsd(tiles, tiles * 2, columns=[0, 1], n_reps=10)
    assert result.jsd == pytest.approx(0.0)


HELDOUT_TABLE = pd.DataFrame(
    {
        "marker_class": ["Exc", "Inh", "Inh", "Astro", "Astro", "Immune", "Immune"]
        + ["Immune", "Oligo", "Oligo", "Exc"],
        "gene_symbol": ["SLC17A7", "GAD1", "GAD2", "AQP4", "GJA1", "CSF1R"]
        + ["P2RY12", "CX3CR1", "MOBP", "PLP1", "SLC17A6"],
        "avoid_platforms": ["", "", "MERSCOPE", "", "", "", "MERSCOPE", "", "", ""]
        + [""],
        "rank": [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2],
    }
)


def test_select_heldout_markers_by_rank_panel_and_platform() -> None:
    panel = ["SLC17A7", "GAD1", "GAD2", "AQP4", "GJA1", "P2RY12", "CX3CR1"]
    panel += ["MOBP", "SLC17A6"]
    merscope = select_heldout_markers(HELDOUT_TABLE, panel, "MERSCOPE")
    assert merscope.markers["Neurons"] == ("SLC17A7", "GAD1", "SLC17A6")
    assert merscope.avoided["Neurons"] == ("GAD2",)
    assert "Microglia" in merscope.skipped  # only CX3CR1 on MERSCOPE
    assert merscope.not_on_panel["Microglia"] == ("CSF1R",)
    assert "Oligodendrocytes" in merscope.skipped  # PLP1 missing
    xenium = select_heldout_markers(HELDOUT_TABLE, panel, "XENIUM", max_markers=2)
    assert xenium.markers["Neurons"] == ("SLC17A7", "GAD1")
    assert xenium.markers["Microglia"] == ("P2RY12", "CX3CR1")
    assert xenium.all_markers[:2] == ("SLC17A7", "GAD1")
    assert "Fibroblasts" in xenium.skipped


def test_auroc_matches_the_rank_definition() -> None:
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5
    assert auroc(np.array([0.3, 0.1, 0.2]), np.array([1, 0, 1])) == pytest.approx(1.0)
    assert auroc(np.array([0.3, 0.1, 0.2]), np.array([0, 1, 0])) == pytest.approx(0.0)
    assert math.isnan(auroc(np.array([1.0]), np.array([1])))


def test_heldout_enrichment_scores_assigned_vs_other_cells() -> None:
    counts = pd.DataFrame(
        {"AQP4": [5, 4, 0, 0, 1, 0], "GJA1": [3, 0, 0, 1, 0, 0]},
    )
    totals = np.array([20, 20, 20, 20, 20, 20])
    labels = ["Astrocytes", "Astrocytes", "Neurons", "Neurons", "Neurons", None]
    [result] = heldout_enrichment(
        counts, totals, labels, {"Astrocytes": ("AQP4", "GJA1")}
    )
    assert result.n_assigned == 2 and result.n_other == 3
    assert result.rate_assigned == pytest.approx(12 / 40)
    assert result.rate_other == pytest.approx(2 / 60)
    assert result.fold == pytest.approx(9.0)
    assert result.detection_assigned == 1.0
    assert result.detection_other == pytest.approx(2 / 3)
    assert result.auroc == pytest.approx(1.0)
    assert result.passes
    [confident] = heldout_enrichment(
        counts,
        totals,
        labels,
        {"Astrocytes": ("AQP4",)},
        include=np.array([True, False, True, True, True, True]),
        min_fold=100.0,
    )
    assert confident.n_assigned == 1 and not confident.passes


PROFILE_GENES = ["G1", "G2", "G3"]


def _profile_table() -> pd.DataFrame:
    rows = []
    for node, name, n_cells, values in (
        ("N1", "Astrocyte", 10, [80.0, 10.0, 10.0]),
        ("N2", "Oligodendrocyte", 30, [10.0, 80.0, 10.0]),
        ("N3", "Microglia", 5, [10.0, 10.0, 80.0]),
    ):
        for gene, value in zip(PROFILE_GENES, values, strict=True):
            rows.append(
                {
                    "level": "SUPC",
                    "node": node,
                    "node_name": name,
                    "n_cells": n_cells,
                    "gene_id": gene,
                    "mean_cpm": value,
                }
            )
    return pd.DataFrame(rows)


def test_profile_matrix_renormalises_over_the_query_genes() -> None:
    matrix = profile_matrix(_profile_table(), "SUPC", ["G1", "G2", "G_NEW"])
    assert list(matrix.index) == ["Astrocyte", "Microglia", "Oligodendrocyte"]
    assert matrix.loc["Astrocyte"].tolist() == pytest.approx([80 / 90, 10 / 90, 0.0])
    with pytest.raises(ValueError, match="no rows"):
        profile_matrix(_profile_table(), "CLUS", PROFILE_GENES)


def test_reference_pseudobulk_factors_recover_a_platform_effect() -> None:
    rng = np.random.default_rng(3)
    genes = [f"G{index}" for index in range(40)]
    profiles = pd.DataFrame(
        rng.dirichlet(np.full(40, 2.0), size=3),
        index=["Astrocyte", "Microglia", "Oligodendrocyte"],
        columns=genes,
    )
    efficiency = np.ones(40)
    efficiency[5] = 4.0
    rows, labels = [], []
    for name in profiles.index:
        expected = profiles.loc[name].to_numpy() * efficiency
        rows.append(rng.multinomial(2000, expected / expected.sum(), size=100))
        labels += [name] * 100
    counts = sparse.csr_matrix(np.vstack(rows))
    capped, uncapped = reference_pseudobulk_log2_factors(
        counts, np.array(labels, dtype=object), profiles, cap_log2=1.0
    )
    assert np.median(uncapped) == pytest.approx(0.0)
    assert uncapped[5] == pytest.approx(2.0, abs=0.25)
    assert np.delete(np.abs(uncapped), 5).max() < 0.3
    assert capped[5] == pytest.approx(1.0)
    include = np.zeros(len(labels), dtype=bool)
    with pytest.raises(ValueError, match="no labelled cell"):
        reference_pseudobulk_log2_factors(counts, labels, profiles, include=include)
    with pytest.raises(ValueError, match="one column per"):
        reference_pseudobulk_log2_factors(counts[:, :2], labels, profiles)


def test_rescale_counts_divides_by_the_factor_and_rounds() -> None:
    counts = sparse.csr_matrix(np.array([[4, 1, 0], [2, 3, 5]]))
    scaled = rescale_counts(counts, np.array([1.0, 0.0, -1.0]), scale=10.0)
    assert scaled.dtype == np.int32
    assert scaled.toarray().tolist() == [[20, 10, 0], [10, 30, 100]]
    with pytest.raises(ValueError, match="one log2 factor"):
        rescale_counts(counts, np.zeros(2))


def test_negative_counts_use_the_assigned_class_genes() -> None:
    negatives = pd.DataFrame(
        {
            "broad_class": ["Astrocytes", "Astrocytes", "Neurons", "Neurons"],
            "gene_id": ["G2", "G3", "G1", "G_OTHER"],
            "negative": [True, False, True, True],
        }
    )
    mask = negative_gene_mask(negatives, PROFILE_GENES, ["Neurons", "Astrocytes"])
    assert mask.tolist() == [[True, False, False], [False, True, False]]
    counts = sparse.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    values = negative_counts(
        counts, ["Astrocytes", "Neurons", None], mask, ["Neurons", "Astrocytes"]
    )
    assert values[:2].tolist() == [2.0, 4.0]
    assert math.isnan(values[2])


def test_fit_beta_binomial_recovers_the_null_rate() -> None:
    rng = np.random.default_rng(4)
    trials = rng.integers(50, 300, size=2000)
    rates = rng.beta(2.0, 98.0, size=trials.size)
    successes = rng.binomial(trials, rates)
    fit = fit_beta_binomial(successes, trials)
    assert fit.alpha / (fit.alpha + fit.beta) == pytest.approx(0.02, abs=0.003)
    assert fit.n_cells == 2000
    tail = beta_binomial_upper_tail(
        np.array([0, 3, 30]), np.array([100, 100, 100]), fit
    )
    assert tail[0] == pytest.approx(1.0)
    assert tail[0] > tail[1] > tail[2]
    with pytest.raises(ValueError, match="needs cells"):
        fit_beta_binomial(np.array([]), np.array([]))


def test_contamination_flags_find_cells_above_the_deep_null() -> None:
    rng = np.random.default_rng(5)
    n = 400
    totals = rng.integers(40, 400, size=n).astype(float)
    negative = rng.binomial(totals.astype(int), 0.01).astype(float)
    negative[:10] = totals[:10] * 0.3  # heavily contaminated
    labels = np.array(["Neurons"] * (n - 20) + ["Microglia"] * 20, dtype=object)
    confident = np.ones(n, dtype=bool)
    flags = contamination_flags(
        negative, totals, labels, confident, ["Neurons", "Microglia"]
    )
    assert flags.flag[:10].all()
    assert flags.flag[10 : n - 20].mean() < 0.05
    assert flags.fits["Neurons"] is not None
    assert flags.fits["Microglia"] is None  # 5 null cells < 30
    assert np.isnan(flags.p_value[n - 20 :]).all()
    assert not flags.flag[n - 20 :].any()
    assert flags.score[0] == pytest.approx(0.3)


def test_class_profiles_weight_nodes_by_their_cells() -> None:
    table = _profile_table()
    profiles = class_profiles(
        table,
        "SUPC",
        PROFILE_GENES,
        {"Astrocyte": "Glia", "Oligodendrocyte": "Glia", "Microglia": "Immune"},
        ["Glia", "Immune", "Absent"],
    )
    assert set(profiles) == {"Glia", "Immune"}
    expected = (np.array([80, 10, 10]) * 10 + np.array([10, 80, 10]) * 30) / 40
    assert profiles["Glia"] == pytest.approx(expected / expected.sum())


def test_depth_grid_and_distinct_gene_quantiles() -> None:
    grid = depth_grid(1000, n_points=10)
    assert grid[0] == 1 and grid[-1] == 1000
    assert (np.diff(grid) > 0).all()
    uniform = np.full(50, 1 / 50)
    q = distinct_gene_quantiles(uniform, np.array([1, 5, 2000]), n_simulations=50)
    assert q[0] == 1.0
    assert q[1] <= 5.0
    assert q[2] == 50.0


def test_expected_genes_quantile_interpolates_per_class() -> None:
    profiles = {"A": np.full(20, 1 / 20), "B": np.array([0.9] + [0.1 / 19] * 19)}
    depth = np.array([10.0, 10.0, 10.0])
    values = expected_genes_quantile(
        depth, ["A", "B", "C"], profiles, n_simulations=100
    )
    assert values[0] > values[1]
    assert math.isnan(values[2])
    assert len(expected_genes_quantile(np.array([]), [], profiles)) == 0


def test_realised_rates_mark_uninformative_strata() -> None:
    frame = realised_rates(
        np.array([True, False, True, True, False]),
        ["A", "A", "B", "B", None],
        ["A", "B", "C"],
        max_informative=0.6,
    )
    rows = frame.set_index("class")
    assert rows.loc["A", "rate"] == 0.5 and rows.loc["A", "informative"]
    assert rows.loc["B", "rate"] == 1.0 and not rows.loc["B", "informative"]
    assert rows.loc["C", "n_cells"] == 0 and not rows.loc["C", "informative"]


def test_occupied_area_counts_bins_with_enough_cells() -> None:
    xy = np.array([[10, 10], [20, 20], [30, 30], [150, 10], [np.nan, 1.0]])
    assert occupied_area_mm2(xy, bin_um=100.0, min_cells=3) == pytest.approx(0.01)
    assert occupied_area_mm2(xy, bin_um=100.0, min_cells=1) == pytest.approx(0.02)
    assert occupied_area_mm2(np.zeros((0, 2))) == 0.0
    with pytest.raises(ValueError, match="bin_um"):
        occupied_area_mm2(xy, bin_um=0)


def test_foreign_marker_fraction_excludes_the_own_class() -> None:
    scores = np.array([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1]])
    values = foreign_marker_fraction(scores, ["A", None], classes=("A", "B", "C"))
    assert values[0] == pytest.approx(0.1)
    assert math.isnan(values[1])


def test_mouse_confidence_applies_the_d_m4_thresholds() -> None:
    labels = pd.DataFrame(
        {
            "total_counts": [100, 30, 15, 100],
            "mmc_wmb_class_bp": [0.95, 0.95, 0.99, 0.8],
            "mmc_wmb_subclass_bp": [0.85, 0.9, 0.9, 0.9],
        }
    )
    result = mouse_confidence(labels)
    assert result.class_confident.tolist() == [True, True, False, False]
    assert result.subclass_confident.tolist() == [True, False, False, False]


def test_whb_labels_from_tidy_aggregate_like_the_provisional_table() -> None:
    vocab = pd.DataFrame(
        {
            "level": ["SUPC"] * 3,
            "node": ["S_AST", "S_OLI", "S_OPC"],
            "node_name": ["Astrocyte", "Oligodendrocyte", "OPC"],
            "broad_class": ["Astrocytes", "Oligodendrocytes", OPC],
            "lineage": ["Astrocytes", "Oligodendrocyte lineage"] * 1
            + ["Oligodendrocyte lineage"],
        }
    )
    tidy = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "level": ["SUPC", "SUPC"],
            "level_name": ["supercluster", "supercluster"],
            "assignment": ["S_OLI", "S_AST"],
            "name": ["Oligodendrocyte", "Astrocyte"],
            "bp": [0.5, 0.9],
            "runner_up_1_assignment": ["S_OPC", None],
            "runner_up_1_name": ["OPC", None],
            "runner_up_1_probability": [0.4, np.nan],
        }
    )
    for rank in range(2, 6):
        for field in ("assignment", "name"):
            tidy[f"runner_up_{rank}_{field}"] = None
        tidy[f"runner_up_{rank}_probability"] = np.nan
    labels = whb_labels_from_tidy(
        tidy, vocab, pd.Series([50.0, 70.0], index=["a", "b"]), level="SUPC"
    )
    assert labels.loc["a", "ct_lineage_name"] == "Oligodendrocyte lineage"
    assert labels.loc["a", "ct_lineage_raw"] == pytest.approx(0.9)
    assert labels.loc["a", "ct_broad_raw"] == pytest.approx(0.5)
    assert labels.loc["b", "total_counts"] == 70.0
    assert labels.loc["a", "mmc_whb_supercluster_runner_up_1_name"] == "OPC"
    soft = soft_matrix_from_provisional(labels)
    assert soft.shape == (2, len(COMPOSITION_COLUMNS))
