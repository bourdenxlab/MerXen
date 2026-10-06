"""Tests for the human marker-consistency referee (plan §8.8, §8.6; M13 C13)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import real_qc as qc
from merxen.annotation.config import (
    AnnotationConfig,
    AnnotationFlagsConfig,
    AnnotationRealQcConfig,
)
from merxen.annotation.human_referee import (
    COMPARATOR_CLASS,
    COMPARATOR_NODE,
    HUMAN_IMMEDIATE_EARLY_GENES,
    HumanRefereeProfiles,
    HumanRefereeSettings,
    RefereeMarkers,
    derive_referee_markers,
    human_marker_referee,
    load_referee_profiles,
)
from merxen.annotation.mouse_flags import IMMEDIATE_EARLY_GENES
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

from .test_real_qc_checks import new_panel_signals

MakeTrust = Callable[..., Any]
LEVEL = "CCN202210140_SUPC"
FLAGS = AnnotationFlagsConfig()
# WHB superclusters of the frontal bundle: (broad class, n_cells). The sink
# Splatter and the small Committed oligodendrocyte precursor (COP) node are
# what separates the two comparators.
NODES: dict[str, tuple[str | None, int]] = {
    "Upper-layer intratelencephalic": ("Neurons", 400),
    "MGE interneuron": ("Neurons", 200),
    "Astrocyte": ("Astrocytes", 150),
    "Oligodendrocyte": ("Oligodendrocytes", 300),
    "Oligodendrocyte precursor": ("Oligodendrocyte precursors", 100),
    "Committed oligodendrocyte precursor": ("Oligodendrocyte precursors", 5),
    "Microglia": ("Microglia", 80),
    "Vascular": ("Vascular cells", 30),
    "Fibroblast": ("Fibroblasts", 10),
    "Splatter": (None, 20),
}
PREFIX = {
    "Neurons": "NEU",
    "Astrocytes": "AST",
    "Oligodendrocytes": "OLI",
    "Oligodendrocyte precursors": "OPC",
    "Microglia": "MIC",
    "Vascular cells": "VAS",
    "Fibroblasts": "FIB",
}
CLASS_GENES = {
    cls: [f"{prefix}{index}" for index in range(1, 4)] for cls, prefix in PREFIX.items()
}
# FOS is neuron-specific in the fixture: an immediate-early gene, never a
# marker. ACTB is shared by every node.
SYMBOLS = [gene for genes in CLASS_GENES.values() for gene in genes] + ["FOS", "ACTB"]
GENE_IDS = [f"ENSG{index:011d}" for index in range(len(SYMBOLS))]


def _node_matrix() -> np.ndarray:
    position = {symbol: index for index, symbol in enumerate(SYMBOLS)}
    matrix = np.full((len(NODES), len(SYMBOLS)), 1e-5)
    for row, (name, (cls, _)) in enumerate(NODES.items()):
        genes = list(CLASS_GENES[cls]) if cls is not None else []
        if cls == "Neurons":
            genes.append("FOS")
        if name == "Committed oligodendrocyte precursor":
            genes += CLASS_GENES["Oligodendrocytes"]
        if name == "Splatter":
            genes += CLASS_GENES["Neurons"]
        matrix[row, [position[gene] for gene in genes]] = 0.3
        matrix[row, position["ACTB"]] = 0.1
    normalised: np.ndarray = matrix / matrix.sum(axis=1, keepdims=True)
    return normalised


def profiles_table(*, levels: Sequence[str] = (LEVEL,)) -> pd.DataFrame:
    """A bundle's ``profiles.parquet`` (long) for the fixture nodes."""
    matrix = _node_matrix()
    frames = []
    for level in levels:
        for row, (name, (_, n_cells)) in enumerate(NODES.items()):
            frames.append(
                pd.DataFrame(
                    {
                        "level": level,
                        "node_name": name,
                        "n_cells": n_cells,
                        "gene_id": GENE_IDS,
                        "expected_fraction": matrix[row],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def fixture_profiles() -> HumanRefereeProfiles:
    return HumanRefereeProfiles.from_table(
        profiles_table(), level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )


def settings(**change: Any) -> HumanRefereeSettings:
    base = HumanRefereeSettings.from_config(AnnotationRealQcConfig())
    return dataclasses.replace(base, **change)


def _sets(markers: RefereeMarkers) -> dict[str, tuple[str, ...]]:
    return {group: genes.symbols for group, genes in markers.markers.items()}


def simulated_cells(
    n_per_class: int = 50, depth: int = 200, seed: int = 0
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Multinomial table cells drawn from one node per broad class."""
    rng = np.random.default_rng(seed)
    matrix = _node_matrix()
    names = list(NODES)
    rows = [
        names.index(name)
        for name in (
            "Upper-layer intratelencephalic",
            "Astrocyte",
            "Oligodendrocyte",
            "Oligodendrocyte precursor",
            "Microglia",
            "Vascular",
            "Fibroblast",
        )
    ]
    counts = np.vstack(
        [rng.multinomial(depth, matrix[row], size=n_per_class) for row in rows]
    )
    truth = np.repeat(np.array(HUMAN_BROAD_CLASSES, dtype=object), n_per_class)
    return sparse.csr_matrix(counts), truth


# --------------------------------------------------------------------------
# Profiles


def test_profiles_group_whb_superclusters_into_the_human_broad_classes() -> None:
    profiles = fixture_profiles()
    assert profiles.node_names == tuple(sorted(NODES))
    classes = dict(zip(profiles.node_names, profiles.node_classes, strict=True))
    assert classes["Committed oligodendrocyte precursor"] == (
        "Oligodendrocyte precursors"
    )
    assert classes["Splatter"] is None  # a sink belongs to no broad class
    cells = dict(zip(profiles.node_names, profiles.node_cells.tolist(), strict=True))
    assert cells["Upper-layer intratelencephalic"] == 400
    assert profiles.node_profiles.shape == (len(NODES), len(GENE_IDS))
    assert profiles.present_classes() == HUMAN_BROAD_CLASSES


def test_profiles_fill_absent_genes_with_zero_and_refuse_bad_inputs() -> None:
    extra = [*GENE_IDS, "ENSG99999999999"]
    profiles = HumanRefereeProfiles.from_table(
        profiles_table(), level=LEVEL, gene_ids=extra, symbols=[*SYMBOLS, "NEW"]
    )
    assert np.all(profiles.node_profiles[:, -1] == 0.0)
    with pytest.raises(ValueError, match="one entry per gene"):
        HumanRefereeProfiles.from_table(
            profiles_table(), level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS[:-1]
        )
    with pytest.raises(ValueError, match="no CCN202210140_CLUS rows"):
        HumanRefereeProfiles.from_table(
            profiles_table(),
            level="CCN202210140_CLUS",
            gene_ids=GENE_IDS,
            symbols=SYMBOLS,
        )
    with pytest.raises(ValueError, match="lacks columns"):
        HumanRefereeProfiles.from_table(
            profiles_table().drop(columns="expected_fraction"),
            level=LEVEL,
            gene_ids=GENE_IDS,
            symbols=SYMBOLS,
        )


def test_profiles_load_from_a_bundle_directory(tmp_path: Path) -> None:
    profiles_table(levels=(LEVEL, "CCN202210140_CLUS")).to_parquet(
        tmp_path / "profiles.parquet"
    )
    loaded = load_referee_profiles(
        tmp_path, level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.node_profiles, fixture_profiles().node_profiles)
    missing = load_referee_profiles(
        tmp_path / "absent", level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    assert missing is None
    wrong_level = load_referee_profiles(
        tmp_path, level="CCN202210140_SUBC", gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    assert wrong_level is None


# --------------------------------------------------------------------------
# Marker derivation (§8.6 specificity rule)


def test_human_immediate_early_genes_are_the_e3_list_in_human_case() -> None:
    assert {symbol.upper() for symbol in IMMEDIATE_EARLY_GENES} == (
        HUMAN_IMMEDIATE_EARLY_GENES
    )
    assert "FOS" in HUMAN_IMMEDIATE_EARLY_GENES
    for comparator in (COMPARATOR_NODE, COMPARATOR_CLASS):
        markers = derive_referee_markers(
            fixture_profiles(), FLAGS, settings(comparator=comparator)
        )
        assert all("FOS" not in genes.symbols for genes in markers.markers.values())


def test_node_comparator_is_the_g2_rule_on_whb_superclusters() -> None:
    """G2 as ported: the group's mean node profile against every other node.

    The sink Splatter carries the neuronal genes and the COP node the
    oligodendrocyte genes, so the literal rule leaves Neurons and
    Oligodendrocytes without markers.
    """
    markers = derive_referee_markers(
        fixture_profiles(), FLAGS, settings(comparator=COMPARATOR_NODE)
    )
    assert markers.comparator == COMPARATOR_NODE
    assert _sets(markers) == {
        cls: tuple(CLASS_GENES[cls])
        for cls in HUMAN_BROAD_CLASSES
        if cls not in {"Neurons", "Oligodendrocytes"}
    }
    assert markers.left_out == ("Neurons", "Oligodendrocytes")
    assert markers.absent == ()
    assert markers.source == "derived"


def test_node_comparator_uses_the_unweighted_mean_of_the_group_nodes() -> None:
    """A gene of only one node of a two-node group fails the group mean."""
    table = profiles_table()
    gene = GENE_IDS[SYMBOLS.index("NEU1")]
    mge = (table["node_name"] == "MGE interneuron") & (table["gene_id"] == gene)
    table.loc[mge, "expected_fraction"] = 1e-5
    splatter = table["node_name"] == "Splatter"
    table = table[~splatter]
    profiles = HumanRefereeProfiles.from_table(
        table, level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    markers = derive_referee_markers(
        profiles, FLAGS, settings(comparator=COMPARATOR_NODE, min_group_markers=1)
    )
    # NEU1 is still ~0.115 in the group mean (half of its ~0.23 in the IT
    # node) against ~1e-5 outside.
    assert set(markers.markers["Neurons"].symbols) == {"NEU1", "NEU2", "NEU3"}
    astro = (table["node_name"] == "Astrocyte") & (table["gene_id"] == gene)
    table.loc[astro, "expected_fraction"] = 0.01  # ~0.115 / 0.01 < 20
    profiles = HumanRefereeProfiles.from_table(
        table, level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    markers = derive_referee_markers(
        profiles, FLAGS, settings(comparator=COMPARATOR_NODE, min_group_markers=1)
    )
    assert set(markers.markers["Neurons"].symbols) == {"NEU2", "NEU3"}


def test_class_comparator_uses_broad_class_profiles() -> None:
    """Cell-weighted class profiles against the other classes' profiles.

    COP (5 of 105 OPC-class cells) dilutes into the OPC profile, and the
    sink Splatter belongs to no class, so every class keeps its markers.
    """
    markers = derive_referee_markers(
        fixture_profiles(), FLAGS, settings(comparator=COMPARATOR_CLASS)
    )
    assert markers.comparator == COMPARATOR_CLASS
    assert _sets(markers) == {
        cls: tuple(CLASS_GENES[cls]) for cls in HUMAN_BROAD_CLASSES
    }
    assert markers.left_out == ()
    profiles = fixture_profiles()
    opc = profiles.class_profile("Oligodendrocyte precursors")
    rows = [
        profiles.node_names.index(name)
        for name in ("Oligodendrocyte precursor", "Committed oligodendrocyte precursor")
    ]
    expected = (
        profiles.node_profiles[rows[0]] * 100 + profiles.node_profiles[rows[1]] * 5
    ) / 105
    np.testing.assert_allclose(opc, expected)


def test_class_comparator_refuses_profiles_without_cell_counts() -> None:
    profiles = dataclasses.replace(fixture_profiles(), node_cells=np.zeros(len(NODES)))
    with pytest.raises(ValueError, match="n_cells"):
        derive_referee_markers(profiles, FLAGS, settings(comparator=COMPARATOR_CLASS))


def test_groups_with_too_few_markers_and_absent_classes_are_recorded() -> None:
    table = profiles_table()
    table = table[~table["node_name"].isin(["Fibroblast"])]
    profiles = HumanRefereeProfiles.from_table(
        table, level=LEVEL, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    markers = derive_referee_markers(
        profiles, FLAGS, settings(comparator=COMPARATOR_CLASS, min_group_markers=4)
    )
    assert markers.markers == {}
    assert markers.left_out == tuple(
        cls for cls in HUMAN_BROAD_CLASSES if cls != "Fibroblasts"
    )
    assert markers.absent == ("Fibroblasts",)
    record = markers.to_json()
    assert record["absent"] == ["Fibroblasts"]
    assert record["min_group_markers"] == 4


def test_marker_sets_round_trip_through_a_frozen_table() -> None:
    """D27 (a): the derived sets are written and fixed before the run."""
    markers = derive_referee_markers(
        fixture_profiles(), FLAGS, settings(comparator=COMPARATOR_CLASS)
    )
    frame = markers.to_frame()
    assert list(frame.columns) == ["group", "rank", "gene_id", "symbol", "ratio"]
    assert len(frame) == 21
    restored = RefereeMarkers.from_frame(frame, gene_ids=GENE_IDS, symbols=SYMBOLS)
    assert _sets(restored) == _sets(markers)
    assert restored.source == "supplied"
    assert restored.fingerprint == markers.fingerprint
    assert len(markers.fingerprint) == 64
    # The fingerprint is the marker sets: dropping one gene changes it.
    fewer = RefereeMarkers.from_frame(
        frame[frame["symbol"] != "AST1"], gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    assert fewer.fingerprint != markers.fingerprint
    # The order of the rows is irrelevant.
    shuffled = RefereeMarkers.from_frame(
        frame.sample(frac=1.0, random_state=0), gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    assert shuffled.fingerprint == markers.fingerprint
    # The fingerprint is the sets, not the ranks inside them.
    forward = RefereeMarkers.from_symbols(
        {"Astrocytes": ["AST1", "AST2", "AST3"], "Microglia": ["MIC1", "MIC2", "MIC3"]},
        gene_ids=GENE_IDS,
        symbols=SYMBOLS,
    )
    backward = RefereeMarkers.from_symbols(
        {"Astrocytes": ["AST3", "AST2", "AST1"], "Microglia": ["MIC2", "MIC1", "MIC3"]},
        gene_ids=GENE_IDS,
        symbols=SYMBOLS,
    )
    assert forward.symbol_sets() != backward.symbol_sets()
    assert forward.fingerprint == backward.fingerprint
    with pytest.raises(ValueError, match="not among the query genes"):
        RefereeMarkers.from_frame(frame, gene_ids=GENE_IDS[1:], symbols=SYMBOLS[1:])


def test_hand_curated_marker_sets_keep_only_panel_genes() -> None:
    """D27 (b): hand-curated alternates, scored by the same rule (reported)."""
    alternates = RefereeMarkers.from_symbols(
        {
            "Astrocytes": ["AST1", "AST2", "SLC4A4"],
            "Oligodendrocyte precursors": ["OPC1", "OPC2", "OPC3"],
            "Fibroblasts": ["FIB1"],
        },
        gene_ids=GENE_IDS,
        symbols=SYMBOLS,
        min_group_markers=2,
    )
    assert _sets(alternates) == {
        "Astrocytes": ("AST1", "AST2"),
        "Oligodendrocyte precursors": ("OPC1", "OPC2", "OPC3"),
    }
    assert alternates.left_out == ("Fibroblasts",)
    assert alternates.missing_symbols == {"Astrocytes": ("SLC4A4",)}
    assert alternates.source == "supplied"
    with pytest.raises(ValueError, match="not a human broad class"):
        RefereeMarkers.from_symbols(
            {"Exc": ["NEU1"]}, gene_ids=GENE_IDS, symbols=SYMBOLS
        )


# --------------------------------------------------------------------------
# Scoring on confident ct_broad


def test_referee_scores_confident_broad_calls_against_pseudo_labels() -> None:
    counts, truth = simulated_cells()
    calls = truth.copy()
    # 35 astrocytes called Neurons: consistency (350 - 35) / 350.
    astro = np.flatnonzero(truth == "Astrocytes")
    calls[astro[:35]] = "Neurons"
    confident = np.ones(len(calls), dtype=bool)
    # Unconfident cells and cells without a broad name are not scored, even
    # though they are pseudo-labelled; a confident cell without marker counts
    # is not pseudo-labelled.
    empty = sparse.csr_matrix((1, counts.shape[1]), dtype=counts.dtype)
    counts = sparse.vstack([counts, counts[:20], empty]).tocsr()
    calls = np.concatenate(
        [calls, np.array(["Astrocytes"] * 10 + [None] * 10 + ["Neurons"])]
    )
    confident = np.concatenate(
        [confident, np.zeros(10, bool), np.ones(10, bool), np.ones(1, bool)]
    )
    referee = human_marker_referee(
        counts,
        fixture_profiles(),
        calls,
        confident,
        flags_config=FLAGS,
        settings=settings(comparator=COMPARATOR_CLASS, min_pseudo_confident=50),
    )
    assert referee.reason is None
    assert referee.n_cells == 371
    assert referee.n_pseudo_confident == 370
    assert referee.n_confident == 351
    assert referee.n_scored == 350
    assert referee.consistency == pytest.approx(315 / 350)
    assert referee.recall["Astrocytes"] == pytest.approx(15 / 50)
    assert referee.recall["Neurons"] == pytest.approx(1.0)
    assert referee.precision["Neurons"] == pytest.approx(50 / 85)
    assert referee.confusion["Astrocytes"] == {"Astrocytes": 15, "Neurons": 35}
    signal = referee.signal()
    assert isinstance(signal, qc.MarkerConsistencySignal)
    assert signal.consistency == pytest.approx(0.9)
    assert (signal.n_marker_groups, signal.n_pseudo_labelled) == (7, 350)
    assert signal.details["comparator"] == COMPARATOR_CLASS
    assert signal.details["fingerprint"] == referee.markers.fingerprint
    assert json.loads(json.dumps(referee.to_json())) == referee.to_json()


def test_referee_accepts_dense_counts_and_is_deterministic() -> None:
    counts, truth = simulated_cells(seed=3)
    kwargs: dict[str, Any] = {
        "flags_config": FLAGS,
        "settings": settings(comparator=COMPARATOR_CLASS, min_pseudo_confident=50),
    }
    confident = np.ones(len(truth), dtype=bool)
    first = human_marker_referee(counts, fixture_profiles(), truth, confident, **kwargs)
    second = human_marker_referee(
        counts.toarray(), fixture_profiles(), list(truth), confident, **kwargs
    )
    assert first.to_json() == second.to_json()
    assert first.consistency == pytest.approx(1.0)


def test_frozen_markers_replace_the_derivation() -> None:
    counts, truth = simulated_cells()
    confident = np.ones(len(truth), dtype=bool)
    config = settings(comparator=COMPARATOR_CLASS, min_pseudo_confident=50)
    derived = derive_referee_markers(fixture_profiles(), FLAGS, config)
    frozen = RefereeMarkers.from_frame(
        derived.to_frame(), gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    supplied = human_marker_referee(
        counts,
        None,  # no profiles needed when the sets are supplied
        truth,
        confident,
        flags_config=FLAGS,
        settings=config,
        markers=frozen,
    )
    computed = human_marker_referee(
        counts,
        fixture_profiles(),
        truth,
        confident,
        flags_config=FLAGS,
        settings=config,
    )
    assert supplied.consistency == computed.consistency
    assert supplied.markers is not None and supplied.markers.source == "supplied"
    assert supplied.markers.fingerprint == computed.markers.fingerprint


def test_referee_is_not_evaluable_with_fewer_than_two_marker_groups() -> None:
    counts, truth = simulated_cells()
    confident = np.ones(len(truth), dtype=bool)
    referee = human_marker_referee(
        counts,
        fixture_profiles(),
        truth,
        confident,
        flags_config=FLAGS,
        settings=settings(min_group_markers=4, min_pseudo_confident=50),
    )
    assert referee.consistency is None
    assert referee.reason == "0 marker group(s) with >= 4 markers (2 needed)"
    one = RefereeMarkers.from_symbols(
        {"Microglia": ["MIC1", "MIC2", "MIC3"]}, gene_ids=GENE_IDS, symbols=SYMBOLS
    )
    single = human_marker_referee(
        counts,
        None,
        truth,
        confident,
        flags_config=FLAGS,
        settings=settings(min_pseudo_confident=50),
        markers=one,
    )
    assert single.consistency is None and "1 marker group(s)" in str(single.reason)
    outcome = qc.marker_consistency_outcome(single.signal(), warn=0.75, broad_only=0.70)
    assert outcome.state == "not_evaluable"
    assert outcome.outcome == "not_evaluable"


def test_referee_is_not_evaluable_with_too_few_pseudo_confident_cells() -> None:
    counts, truth = simulated_cells()
    confident = np.ones(len(truth), dtype=bool)
    referee = human_marker_referee(
        counts,
        fixture_profiles(),
        truth,
        confident,
        flags_config=FLAGS,
        settings=settings(comparator=COMPARATOR_CLASS),  # 200 needed
        markers=None,
    )
    assert referee.consistency is not None  # 350 scored cells
    few = human_marker_referee(
        counts[:100],
        fixture_profiles(),
        truth[:100],
        confident[:100],
        flags_config=FLAGS,
        settings=settings(comparator=COMPARATOR_CLASS),
    )
    assert few.consistency is None
    assert few.reason == "100 scored pseudo-confident cells < 200"
    assert few.recall  # the numbers are still reported


def test_referee_is_not_evaluable_without_counts_or_profiles() -> None:
    _, truth = simulated_cells()
    confident = np.ones(len(truth), dtype=bool)
    kwargs: dict[str, Any] = {"flags_config": FLAGS, "settings": settings()}
    no_counts = human_marker_referee(
        None, fixture_profiles(), truth, confident, **kwargs
    )
    assert no_counts.consistency is None
    assert no_counts.reason == "no query counts"
    counts, _ = simulated_cells()
    no_profiles = human_marker_referee(counts, None, truth, confident, **kwargs)
    assert no_profiles.consistency is None
    assert no_profiles.reason == "no profiles and no supplied marker sets"
    assert no_profiles.signal().reason == no_profiles.reason


def test_referee_refuses_misaligned_inputs() -> None:
    counts, truth = simulated_cells()
    confident = np.ones(len(truth), dtype=bool)
    kwargs: dict[str, Any] = {"flags_config": FLAGS, "settings": settings()}
    with pytest.raises(ValueError, match="one entry per table cell"):
        human_marker_referee(
            counts, fixture_profiles(), truth[:-1], confident, **kwargs
        )
    with pytest.raises(ValueError, match="one entry per table cell"):
        human_marker_referee(
            counts, fixture_profiles(), truth, confident[:-1], **kwargs
        )
    with pytest.raises(ValueError, match="query genes"):
        human_marker_referee(
            counts[:, :-1], fixture_profiles(), truth, confident, **kwargs
        )


# --------------------------------------------------------------------------
# Settings and the outcome (D18)


def test_settings_follow_the_real_qc_config() -> None:
    config = AnnotationRealQcConfig()
    loaded = HumanRefereeSettings.from_config(config)
    assert loaded == HumanRefereeSettings(
        min_group_markers=3,
        min_marker_units=1.5,
        min_marker_share=0.6,
        min_pseudo_confident=200,
        comparator=COMPARATOR_NODE,
    )
    changed = HumanRefereeSettings.from_config(
        AnnotationRealQcConfig(marker_referee_comparator="class")
    )
    assert changed.comparator == COMPARATOR_CLASS
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(marker_referee_comparator="cluster")
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(marker_referee_min_marker_share=1.5)
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(marker_referee_min_group_markers=0)
    with pytest.raises(ValueError, match="comparator"):
        dataclasses.replace(loaded, comparator="cluster")
    with pytest.raises(ValueError, match="min_marker_share"):
        dataclasses.replace(loaded, min_marker_share=0.0)


@pytest.mark.parametrize(
    ("n_miscalled", "expected"),
    [(35, "pass"), (90, "warn"), (110, "fail")],
)
def test_the_referee_drives_the_d18_outcome_in_real_data_qc(
    make_trust: MakeTrust, n_miscalled: int, expected: str
) -> None:
    """315 / 350 = .90 passes, 260 / 350 = .743 warns, 240 / 350 = .686 caps."""
    counts, truth = simulated_cells()
    calls = truth.copy()
    wrong = np.flatnonzero(truth != "Neurons")[:n_miscalled]
    calls[wrong] = "Neurons"
    referee = human_marker_referee(
        counts,
        fixture_profiles(),
        calls,
        np.ones(len(calls), dtype=bool),
        flags_config=FLAGS,
        settings=settings(comparator=COMPARATOR_CLASS, min_pseudo_confident=50),
    )
    result = qc.real_data_qc(
        new_panel_signals(marker_consistency=referee.signal()),
        make_trust("provisional"),
        AnnotationConfig(species="human"),
    )
    record = result.provenance()
    assert record.outcomes["marker_consistency"] == expected
    lowered = result.effects.gate_cap
    assert lowered == ("broad_only" if expected == "fail" else None)
    outcome = next(
        item for item in result.outcomes if item.check == "marker_consistency"
    )
    assert outcome.details["fingerprint"] == referee.markers.fingerprint
    assert outcome.details["consistency"] == pytest.approx((350 - n_miscalled) / 350)
