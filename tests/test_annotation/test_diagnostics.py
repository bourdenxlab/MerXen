"""Tests for panel diagnostics, validated families and trust states.

``merxen.annotation.diagnostics`` (plan §8.2, §4.7): the packaged validated
tables, family inheritance from them, the trust-state truth table, the
effects of each state and the rev3 rule that promotion by simulation never
changes what is emitted.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.config import AnnotationConfig
from merxen.annotation.diagnostics import (
    VALIDATED_GENES_COLUMNS,
    VALIDATED_LEVELS_COLUMNS,
    VALIDATED_PANEL_GENES_FILE,
    VALIDATED_PANEL_LEVELS_FILE,
    VALIDATED_PANELS_COLUMNS,
    VALIDATED_PANELS_FILE,
    CoverageDiagnostics,
    GeneIdDiagnostics,
    PanelGeneList,
    ResolvabilityTrust,
    TrustDecision,
    TrustRules,
    ValidatedPanelRecord,
    ValidatedPanelsError,
    ValidatedPanelTable,
    family_validation_preview,
    level_rank,
    load_validated_panels,
    panel_diagnostics,
    panel_provenance,
    read_validated_panels,
    trust_for_panel,
    trust_state,
    validated_share_by_class,
)
from merxen.annotation.panel import (
    PANEL_GENES_FILE,
    AnnotationPanel,
    PanelFamily,
    compute_panel_hash,
    load_annotation_panel,
    load_curated_setc_families,
    panel_family,
    panel_from_gene_list,
)
from merxen.annotation.provenance import AnnotationProvenance
from merxen.annotation.schema import CellStatus

# First 16 hex digits of each seeded panel hash (the loader re-hashes the
# gene lists, so the full digests are checked there).
SEEDED_HASH_PREFIXES = {
    "human_set_a_296": "dcab8c6d18959395",
    "human_set_a_297": "6e5fd5fb86ef0c3e",
    "human_set_c_264": "08b83ea5cdd761cb",
    "human_set_c_265": "77bd80ed0fa7603f",
    "mouse_ag7_500": "f4c291864ba1c911",
    "mouse_vzg2_815": "2a224a8ef15c904d",
}


def ids(n: int, offset: int = 0) -> list[str]:
    return [f"ENSG{index:011d}" for index in range(offset, offset + n)]


# --------------------------------------------------------------------------
# Table fixtures


def panel_row(
    panel_id: str,
    gene_ids: Sequence[str],
    *,
    family_id: str = "test_family",
    species: str = "human",
    platforms: str = "XENIUM",
    max_level: str = "supercluster",
    basis: str = "real_data",
) -> dict[str, str]:
    return {
        "panel_id": panel_id,
        "family_id": family_id,
        "panel_hash": compute_panel_hash(sorted(gene_ids)),
        "panel_role": "sample_panel",
        "species": species,
        "platforms": platforms,
        "n_genes": str(len(gene_ids)),
        "validated_max_level": max_level,
        "validation_basis": basis,
        "root_marker_source": "test",
        "evidence": "test",
        "date": "2026-09-27",
        "approving_pr": "#0",
        "note": "",
    }


def level_row(
    family_id: str,
    panel_hash: str,
    level: str,
    cls: str,
    *,
    status: str = "validated",
    min_depth: int | None = 30,
    max_depth: int | None = 120,
    in_class_set: bool = True,
) -> dict[str, str]:
    return {
        "family_id": family_id,
        "panel_hash": panel_hash,
        "level": level,
        "class": cls,
        "in_class_set": "true" if in_class_set else "false",
        "status": status,
        "validated_min_depth": "" if min_depth is None else str(min_depth),
        "tested_max_depth": "" if max_depth is None else str(max_depth),
        "evidence": "test",
    }


def write_tables(
    directory: Path,
    panels: Sequence[Mapping[str, str]],
    *,
    levels: Sequence[Mapping[str, str]] = (),
    genes: Mapping[str, tuple[Sequence[str], Sequence[str]]] | None = None,
) -> Path:
    """Write the three tables; ``genes`` maps panel_id to (ids, root markers)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        (VALIDATED_PANELS_FILE, VALIDATED_PANELS_COLUMNS, panels),
        (VALIDATED_PANEL_LEVELS_FILE, VALIDATED_LEVELS_COLUMNS, levels),
    ):
        with (directory / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
    with (directory / VALIDATED_PANEL_GENES_FILE).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(VALIDATED_GENES_COLUMNS)
        for panel_id, (gene_ids, roots) in (genes or {}).items():
            for gene_id in sorted(gene_ids):
                writer.writerow(
                    [
                        panel_id,
                        gene_id,
                        gene_id,
                        "true" if gene_id in roots else "false",
                    ]
                )
    return directory


def simulation_table(tmp_path: Path) -> tuple[ValidatedPanelTable, list[str]]:
    """A simulation-validated human family validated at broad for two classes."""
    gene_ids = ids(100)
    row = panel_row(
        "sim_panel",
        gene_ids,
        family_id="sim_family",
        max_level="broad",
        basis="simulation",
    )
    levels = [
        level_row("sim_family", row["panel_hash"], "broad", "Exc", min_depth=15),
        level_row("sim_family", row["panel_hash"], "broad", "Astro", min_depth=60),
        level_row(
            "sim_family",
            row["panel_hash"],
            "supercluster",
            "Exc",
            status="failed:NP3",
            min_depth=None,
            max_depth=None,
        ),
        level_row(
            "sim_family",
            row["panel_hash"],
            "broad",
            "Vascular",
            status="not_evaluable",
            min_depth=None,
            max_depth=None,
            in_class_set=False,
        ),
    ]
    directory = write_tables(
        tmp_path / "sim", [row], levels=levels, genes={"sim_panel": (gene_ids, ())}
    )
    return read_validated_panels(directory), gene_ids


# --------------------------------------------------------------------------
# The packaged tables


def test_packaged_tables_list_the_seeded_real_data_families() -> None:
    table = load_validated_panels()
    assert table.source == "packaged"
    by_id = {record.panel_id: record for record in table.records}
    assert {
        key: record.panel_hash[:16] for key, record in by_id.items()
    } == SEEDED_HASH_PREFIXES
    assert {record.validation_basis for record in table.records} == {"real_data"}
    assert table.family_ids() == ["human_set_a", "mouse_ag7", "mouse_vzg2"]
    assert by_id["human_set_a_297"].platforms == ("MERSCOPE", "XENIUM")
    assert by_id["mouse_vzg2_815"].platforms == ("MERSCOPE",)
    assert {r.validated_max_level for r in table.family_records("human_set_a")} == {
        "supercluster"
    }
    assert by_id["mouse_ag7_500"].validated_max_level == "subclass"
    # No simulation family yet: the levels table is header-only (§4.7).
    assert table.levels == ()
    assert set(table.sha256) == {
        VALIDATED_PANELS_FILE,
        VALIDATED_PANEL_LEVELS_FILE,
        VALIDATED_PANEL_GENES_FILE,
    }


def test_packaged_gene_lists_hash_to_their_rows_and_carry_root_markers() -> None:
    table = load_validated_panels()
    counts = {
        panel_id: (len(genes.ensembl_ids), len(genes.root_markers))
        for panel_id, genes in table.genes.items()
    }
    assert counts == {
        "human_set_a_296": (296, 230),
        "human_set_a_297": (297, 230),
        "human_set_c_264": (264, 212),
        "human_set_c_265": (265, 212),
        "mouse_ag7_500": (500, 491),
        "mouse_vzg2_815": (815, 806),
    }
    genes_297 = set(table.genes["human_set_a_297"].ensembl_ids)
    genes_296 = set(table.genes["human_set_a_296"].ensembl_ids)
    assert genes_297 - genes_296 == {"ENSG00000188486"}  # H2AX = Xenium H2AFX


def test_packaged_set_a_rows_match_the_curated_set_c_family() -> None:
    table = load_validated_panels()
    (family,) = load_curated_setc_families("human")
    set_a = {r.panel_hash for r in table.records if r.panel_role == "set_a"}
    assert set_a == set(family.set_a_panel_hashes)
    for set_a_id, set_c_id in (
        ("human_set_a_296", "human_set_c_264"),
        ("human_set_a_297", "human_set_c_265"),
    ):
        expected = set(table.genes[set_a_id].ensembl_ids) - set(family.excluded_ids)
        assert set(table.genes[set_c_id].ensembl_ids) == expected


def test_packaged_families_are_what_the_set_a_and_mouse_panels_inherit() -> None:
    table = load_validated_panels()
    genes = table.genes["human_set_a_297"]
    pair = ["MERSCOPE", "XENIUM"]
    known = table.known_families("human")
    listed = panel_family(
        genes.ensembl_ids, species="human", platforms=pair, known_families=known
    )
    assert (listed.family_id, listed.basis) == ("human_set_a", "listed")
    assert listed.matched_platforms == pair
    # One platform of the pair run on its own (per_platform) keeps the
    # family validated on both platforms (OD-E7 asks for the same platform).
    alone = panel_family(
        genes.ensembl_ids, species="human", platforms=["XENIUM"], known_families=known
    )
    assert (alone.family_id, alone.basis) == ("human_set_a", "listed")
    assert alone.matched_platforms == ["XENIUM"]
    # The review case: set a plus one gene (NRP1-like) on Xenium alone
    # inherits set a (P5011 Xenium, per_platform).
    extra = sorted({*genes.ensembl_ids, "ENSG00000099250"})
    plus_one = panel_family(
        extra, species="human", platforms=["XENIUM"], known_families=known
    )
    assert (plus_one.family_id, plus_one.basis) == ("human_set_a", "inherited")
    assert plus_one.jaccard == pytest.approx(297 / 298, abs=1e-6)
    assert plus_one.matched_platforms == ["XENIUM"]
    non_root = sorted(set(genes.ensembl_ids) - genes.root_markers)
    near = sorted(set(genes.ensembl_ids) - set(non_root[:5]))
    inherited = panel_family(
        near, species="human", platforms=pair, known_families=known
    )
    assert (inherited.family_id, inherited.basis) == ("human_set_a", "inherited")
    missing_root = sorted(set(genes.ensembl_ids) - {sorted(genes.root_markers)[0]})
    assert (
        panel_family(
            missing_root, species="human", platforms=pair, known_families=known
        ).basis
        == "own"
    )
    vzg2 = table.genes["mouse_vzg2_815"].ensembl_ids
    assert (
        panel_family(
            vzg2,
            species="mouse",
            platforms=["MERSCOPE"],
            known_families=table.known_families("mouse"),
        ).family_id
        == "mouse_vzg2"
    )
    # A Xenium panel resembling VZG2 (a MERSCOPE-only family) never inherits.
    xenium_vzg2 = panel_family(
        vzg2,
        species="mouse",
        platforms=["XENIUM"],
        known_families=table.known_families("mouse"),
    )
    assert xenium_vzg2.basis == "own"
    assert xenium_vzg2.matched_platforms == []


def test_load_validated_panels_from_a_path(tmp_path: Path) -> None:
    gene_ids = ids(60)
    directory = write_tables(tmp_path / "t", [panel_row("custom", gene_ids)])
    by_dir = load_validated_panels(directory)
    by_file = load_validated_panels(directory / VALIDATED_PANELS_FILE)
    assert by_dir.records == by_file.records
    assert by_dir.genes == {}  # no gene list: exact-hash matching only
    (known,) = by_dir.known_families()
    assert known.ensembl_ids == frozenset()
    assert (
        panel_family(
            gene_ids, species="human", platforms=["XENIUM"], known_families=[known]
        ).basis
        == "listed"
    )
    with pytest.raises(ValidatedPanelsError, match="must be named"):
        load_validated_panels(tmp_path / "other.csv")
    with pytest.raises(FileNotFoundError):
        load_validated_panels(tmp_path / "missing")


# --------------------------------------------------------------------------
# Table validation


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("duplicate_hash", "duplicated panel_hash"),
        ("family_disagrees", "rows disagree on validated_max_level"),
        ("levels_for_real_data", "simulation rows only"),
        ("simulation_without_levels", "has no validated_panel_levels.csv rows"),
        ("bad_status", "failed:NP<k>"),
        ("validated_without_depths", "needs validated_min_depth"),
        ("depths_reversed", "tested_max_depth < validated_min_depth"),
        ("gene_hash", "does not hash to panel_hash"),
        ("gene_count", "listed genes"),
        ("cp_class_not_validated", "gate-P rule"),
        ("simulation_below_broad", "validated_max_level >= broad"),
        ("fine_level", "up to the leaf"),
        ("bad_platform", "platforms must be"),
        ("unknown_level_family", "unknown family"),
    ],
)
def test_malformed_tables_are_refused(tmp_path: Path, case: str, match: str) -> None:
    gene_ids = ids(60)
    base = panel_row("p1", gene_ids)
    panels: list[dict[str, str]] = [base]
    levels: list[dict[str, str]] = []
    genes: dict[str, tuple[Sequence[str], Sequence[str]]] = {}
    sim = panel_row(
        "p1", gene_ids, basis="simulation", max_level="broad", family_id="sim"
    )
    if case == "duplicate_hash":
        panels.append({**base, "panel_id": "p2"})
    elif case == "family_disagrees":
        panels.append({**panel_row("p2", ids(61)), "validated_max_level": "broad"})
    elif case == "levels_for_real_data":
        levels = [level_row("test_family", base["panel_hash"], "broad", "Exc")]
    elif case == "simulation_without_levels":
        panels = [sim]
    elif case == "bad_status":
        panels = [sim]
        levels = [level_row("sim", sim["panel_hash"], "broad", "Exc", status="ok")]
    elif case == "validated_without_depths":
        panels = [sim]
        levels = [level_row("sim", sim["panel_hash"], "broad", "Exc", min_depth=None)]
    elif case == "depths_reversed":
        panels = [sim]
        levels = [
            level_row(
                "sim", sim["panel_hash"], "broad", "Exc", min_depth=60, max_depth=30
            )
        ]
    elif case == "gene_hash":
        genes = {"p1": (ids(60, offset=1), ())}
    elif case == "gene_count":
        panels = [{**base, "n_genes": "61"}]
        genes = {"p1": (gene_ids, ())}
    elif case == "cp_class_not_validated":
        panels = [sim]
        levels = [
            level_row("sim", sim["panel_hash"], "broad", "Exc"),
            level_row(
                "sim",
                sim["panel_hash"],
                "broad",
                "Astro",
                status="failed:NP4",
                min_depth=None,
                max_depth=None,
            ),
        ]
    elif case == "simulation_below_broad":
        panels = [{**sim, "validated_max_level": "lineage"}]
    elif case == "fine_level":
        panels = [{**base, "validated_max_level": "cluster"}]
    elif case == "bad_platform":
        panels = [{**base, "platforms": "COSMX"}]
    elif case == "unknown_level_family":
        panels = [sim]
        levels = [
            level_row("sim", sim["panel_hash"], "broad", "Exc"),
            level_row("nobody", sim["panel_hash"], "broad", "Exc"),
        ]
    write_tables(tmp_path, panels, levels=levels, genes=genes)
    with pytest.raises(ValidatedPanelsError, match=match):
        read_validated_panels(tmp_path)


def test_programmatic_tables_are_checked_too() -> None:
    gene_ids = ids(60)
    record = ValidatedPanelRecord.model_validate(
        {**panel_row("p1", gene_ids), "platforms": ("XENIUM",), "n_genes": 60}
    )
    genes = PanelGeneList(
        panel_id="p1",
        ensembl_ids=tuple(gene_ids),
        symbols={},
        root_markers=frozenset({"ENSG99999999999"}),
    )
    with pytest.raises(ValidatedPanelsError, match="root markers outside"):
        ValidatedPanelTable(records=(record,), genes={"p1": genes})
    with pytest.raises(ValidatedPanelsError, match="unknown panel_id"):
        ValidatedPanelTable(records=(), genes={"p1": genes})


def test_level_ranks() -> None:
    assert level_rank("human", "lineage") == 0
    assert level_rank("human", "supercluster") == level_rank("human", "seaad_subclass")
    assert level_rank("human", "cluster") == 4
    assert level_rank("mouse", "subclass") == 3
    assert level_rank("mouse", "supercluster") is None


# --------------------------------------------------------------------------
# Diagnostics


def bundle_manifest(
    reference_id: str = "whb_frontal_supc_clus",
    *,
    panel_hash: str | None = None,
    root_markers: int = 230,
    n_query_genes_used: int = 297,
    trust: Mapping[str, Any] | None = None,
    resolvability: bool = True,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "panel_coverage": {
            "absent_from_reference": ["ENSG00000000009"],
            "markers_per_parent_median": 77.0,
            "markers_per_parent_min": 4,
            "n_collapsed_parents": 1,
            "n_marker_genes": 247,
            "n_panel_genes": 297,
            "n_query_genes_used": n_query_genes_used,
            "n_weak_parents": 1,
            "root_child_min_markers": 10,
            "root_children": 17,
            "root_children_separated": 15,
            "root_markers": root_markers,
            "share_in_reference": 1.0,
        },
        "markers": {
            "weak_parent_markers": 5,
            "weak_parents": ["SUPC/465"],
            "collapsed": [
                {
                    "key": "SUPC/9",
                    "level": "SUPC",
                    "hidden_leaves": ["CLUS/1", "CLUS/2"],
                }
            ],
        },
    }
    if resolvability:
        output["resolvability"] = {
            "trust": dict(trust or {"state": None, "reasons": [], "leaf_classes": []})
        }
    return {
        "reference_id": reference_id,
        "build_hash": "b" * 64,
        "panel": {"panel_hash": panel_hash, "n_genes": 297},
        "builder_output": output,
    }


def test_coverage_and_resolvability_come_from_the_bundle_manifest() -> None:
    manifest = bundle_manifest(
        trust={
            "state": "broad_only",
            "reasons": ["leaf_unresolvable: x"],
            "leaf_share_by_depth": {"10": 0.25},
            "leaf_classes": ["Exc"],
        }
    )
    coverage = CoverageDiagnostics.from_bundle_manifest(manifest)
    assert coverage.root_markers == 230 and coverage.root_children_separated == 15
    assert coverage.absent_from_reference == ("ENSG00000000009",)
    assert coverage.weak_parents == ("SUPC/465",)
    assert coverage.n_hidden_leaves == 2
    markers = coverage.marker_provenance(lookup_sha256="e" * 64)
    assert (markers.root_markers, markers.n_collapsed_parents) == (230, 1)
    trust = ResolvabilityTrust.from_bundle_manifest(manifest)
    assert trust is not None and trust.state == "broad_only"
    assert trust.leaf_share_by_depth == {"10": 0.25}
    assert (
        ResolvabilityTrust.from_bundle_manifest(bundle_manifest(resolvability=False))
        is None
    )
    with pytest.raises(ValueError, match="panel_coverage"):
        CoverageDiagnostics.from_bundle_manifest({"reference_id": "wmb_region_share"})


def report_entry(status: str = "ok", reasons: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "sample_id": "S_X",
        "platform": "XENIUM",
        "status": status,
        "refusal_reasons": list(reasons),
        "n_features_in": 320,
        "controls_removed": {
            "feature_type:negative_control_probe": [
                "NegControlProbe_1",
                "NegControlProbe_2",
            ],
            "name:blank": ["BLANK_0001"],
        },
        "n_non_control": 317,
        "n_genes": 300,
        "gene_id_resolution": {"native": 298, "fallback_table": 2},
        "gene_id_resolution_share": 300 / 317,
        "unresolved": {"MAPT4R": "not_in_fallback_table"},
        "merged_duplicates": {},
        "species_check": {"status": "pass"},
    }


def annotation_panel(
    gene_ids: Sequence[str], family: PanelFamily | None
) -> AnnotationPanel:
    ordered = sorted(gene_ids)
    return AnnotationPanel(
        name="xenium",
        kind="platform",
        species="human",
        platforms=["XENIUM"],
        sample_ids=["S_X"],
        panel_mode="per_platform",
        panel_hash=compute_panel_hash(ordered),
        n_genes=len(ordered),
        ensembl_ids=ordered,
        symbols=ordered,
        panel_family=family,
    )


def test_panel_diagnostics_gathers_report_and_bundles() -> None:
    gene_ids = ids(297)
    panel = annotation_panel(gene_ids, None)
    report = {
        "declared_panels": {
            "S_X": report_entry(),
            "S_M": {**report_entry(), "sample_id": "S_M", "platform": "MERSCOPE"},
        }
    }
    diagnostics = panel_diagnostics(
        panel,
        panel_report=report,
        bundles=[bundle_manifest(panel_hash=panel.panel_hash)],
    )
    assert [item.name for item in diagnostics.gene_ids] == ["S_X"]
    (gene_id_diag,) = diagnostics.gene_ids
    assert gene_id_diag.controls_removed == {
        "feature_type:negative_control_probe": 2,
        "name:blank": 1,
    }
    assert gene_id_diag.n_controls_removed == 3
    assert gene_id_diag.species_check == "pass"
    assert diagnostics.n_unmapped() == 1
    assert diagnostics.resolved_by_source() == {"fallback_table": 2, "native": 298}
    assert set(diagnostics.coverage) == {"whb_frontal_supc_clus"}
    with pytest.raises(ValueError, match="was built on panel"):
        panel_diagnostics(panel, bundles=[bundle_manifest(panel_hash="c" * 64)])


def test_gene_id_diagnostics_from_a_declared_panel() -> None:
    from merxen.annotation.panel import PanelSource, declared_panel, raw_panel_from_var

    names = [*ids(55), "NegControlProbe_00001"]
    var = pd.DataFrame(
        {
            "gene_ids": names,
            "feature_types": ["Gene Expression"] * 55 + ["Negative Control Probe"],
        },
        index=names,
    )
    declared = declared_panel(
        raw_panel_from_var(var, source=PanelSource(kind="h5ad_var")),
        species="human",
        platform="XENIUM",
        sample_id="S_X",
    )
    item = GeneIdDiagnostics.from_declared(declared)
    assert (item.name, item.status, item.n_genes) == ("S_X", "ok", 55)
    assert item.n_controls_removed == 1
    assert item.resolved_by_source == {"native": 55}


# --------------------------------------------------------------------------
# The trust-state truth table


@pytest.fixture
def real_table(tmp_path: Path) -> tuple[ValidatedPanelTable, list[str]]:
    gene_ids = ids(100)
    row = panel_row("real_panel", gene_ids, family_id="real_family")
    table = read_validated_panels(
        write_tables(tmp_path / "real", [row], genes={"real_panel": (gene_ids, ids(5))})
    )
    return table, gene_ids


def family_of(table: ValidatedPanelTable, gene_ids: Sequence[str]) -> PanelFamily:
    return panel_family(
        gene_ids,
        species="human",
        platforms=["XENIUM"],
        known_families=table.known_families(),
    )


def decide(
    table: ValidatedPanelTable,
    gene_ids: Sequence[str],
    *,
    role: str = "primary",
    gene_status: str = "ok",
    mapped: int = 100,
    root: int = 50,
    resolvability: str | None = "none",
    rules: TrustRules | None = None,
    coverage: bool = True,
) -> TrustDecision:
    family = family_of(table, gene_ids)
    constraint = (
        None
        if resolvability is None
        else ResolvabilityTrust(
            state=None if resolvability == "none" else resolvability,  # type: ignore[arg-type]
            reasons=() if resolvability == "none" else (f"{resolvability}: planted",),
        )
    )
    return trust_state(
        reference_id="whb_frontal_supc_clus" if role == "primary" else "seaad_mr_panel",
        role=role,  # type: ignore[arg-type]
        species="human",
        panel_hash=compute_panel_hash(sorted(gene_ids)),
        n_panel_genes=len(gene_ids),
        family=family,
        validated=table,
        rules=rules,
        gene_ids=[
            GeneIdDiagnostics.from_report_entry(
                "S_X",
                report_entry(
                    gene_status,
                    ["species_mismatch"] if gene_status == "refused" else [],
                ),
            )
        ],
        coverage=(
            CoverageDiagnostics.from_bundle_manifest(
                bundle_manifest(root_markers=root, n_query_genes_used=mapped)
            )
            if coverage
            else None
        ),
        resolvability=constraint,
    )


# (listed family?, gene-ID status, mapped genes, root markers, resolvability)
TRUTH_TABLE = [
    # refusals win over everything, listed or not
    (True, "refused", 100, 50, "none", "refused", ["gene_ids:species_mismatch"]),
    (False, "refused", 100, 50, "none", "refused", ["gene_ids:species_mismatch"]),
    (True, "ok", 49, 50, "none", "refused", ["min_mapped_genes"]),
    (False, "ok", 100, 9, "none", "refused", ["root_markers"]),
    (True, "ok", 100, 50, "refused", "refused", ["resolvability:broad_unresolvable"]),
    (
        False,
        "refused",
        40,
        3,
        "refused",
        "refused",
        [
            "gene_ids:species_mismatch",
            "min_mapped_genes",
            "root_markers",
            "resolvability:broad_unresolvable",
        ],
    ),
    # broad-only: the leaf test, and the fail-safe without resolvability
    (
        False,
        "ok",
        100,
        50,
        "broad_only",
        "broad_only",
        ["resolvability:leaf_unresolvable"],
    ),
    (
        True,
        "ok",
        100,
        50,
        "broad_only",
        "broad_only",
        ["resolvability:leaf_unresolvable"],
    ),
    (False, "ok", 100, 50, None, "broad_only", ["resolvability_not_run"]),
    # the family decides between validated and provisional
    (True, "ok", 100, 50, "none", "validated", ["family_validated"]),
    (True, "ok", 100, 50, None, "validated", ["family_validated"]),
    (False, "ok", 100, 50, "none", "provisional", ["family_not_validated"]),
    # boundaries: exactly at the minimums passes
    (False, "ok", 50, 10, "none", "provisional", ["family_not_validated"]),
]


@pytest.mark.parametrize(
    ("listed", "gene_status", "mapped", "root", "resolvability", "state", "codes"),
    TRUTH_TABLE,
)
def test_trust_state_truth_table(
    real_table: tuple[ValidatedPanelTable, list[str]],
    listed: bool,
    gene_status: str,
    mapped: int,
    root: int,
    resolvability: str | None,
    state: str,
    codes: list[str],
) -> None:
    table, gene_ids = real_table
    panel_ids = gene_ids if listed else ids(100, offset=500)
    decision = decide(
        table,
        panel_ids,
        gene_status=gene_status,
        mapped=mapped,
        root=root,
        resolvability=resolvability,
    )
    assert decision.state == state
    assert decision.reason_codes == codes
    assert (decision.validation_basis is not None) == (state == "validated")
    assert decision.family_basis == ("listed" if listed else "own")
    if listed and state in ("refused", "broad_only"):
        # A listed family that fails the automatic checks is an H18
        # regression: its verdict is kept as a note.
        assert "family_validated" in [note.code for note in decision.notes]


def test_every_trust_combination_has_exactly_one_state(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    order = ["refused", "broad_only", "provisional", "validated"]
    for listed, gene_status, mapped, root, resolvability in itertools.product(
        (True, False),
        ("ok", "refused"),
        (49, 50),
        (9, 10),
        ("none", "refused", "broad_only", None),
    ):
        decision = decide(
            table,
            gene_ids if listed else ids(100, offset=500),
            gene_status=gene_status,
            mapped=mapped,
            root=root,
            resolvability=resolvability,
        )
        refused = (
            gene_status == "refused"
            or mapped < 50
            or root < 10
            or resolvability == "refused"
        )
        if refused:
            expected = "refused"
        elif resolvability == "broad_only" or (resolvability is None and not listed):
            expected = "broad_only"
        else:
            expected = "validated" if listed else "provisional"
        assert decision.state == expected
        assert decision.state in order


def test_other_platform_or_species_never_inherits_validated(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    family = panel_family(
        gene_ids,
        species="human",
        platforms=["MERSCOPE"],
        known_families=table.known_families(),
    )
    assert family.basis == "own"
    decision = trust_state(
        reference_id="whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=compute_panel_hash(sorted(gene_ids)),
        n_panel_genes=100,
        family=family,
        validated=table,
        coverage=CoverageDiagnostics.from_bundle_manifest(
            bundle_manifest(n_query_genes_used=100)
        ),
        resolvability=ResolvabilityTrust(),
    )
    assert decision.state == "provisional"
    # A family id that happens to be listed but was not earned (own basis) or
    # belongs to another species gives no trust either.
    borrowed = PanelFamily(
        family_id="real_family", basis="own", reference_panel_hash="a" * 64, jaccard=1.0
    )
    preview = family_validation_preview(borrowed, "human", table)
    assert preview["listed"] is False and preview["expected_trust"] == "provisional"
    listed = family.model_copy(update={"family_id": "real_family", "basis": "listed"})
    assert family_validation_preview(listed, "mouse", table)["listed"] is False
    assert (
        family_validation_preview(listed, "human", table)["expected_trust"]
        == "validated"
    )


def test_a_subset_panel_keeps_its_family_trust(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    subset = PanelFamily(
        family_id="real_family",
        basis="subset",
        reference_panel_hash=compute_panel_hash(sorted(gene_ids)),
        jaccard=0.97,
    )
    decision = trust_state(
        reference_id="whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=compute_panel_hash(sorted(gene_ids[:97])),
        n_panel_genes=97,
        family=subset,
        validated=table,
        coverage=CoverageDiagnostics.from_bundle_manifest(
            bundle_manifest(n_query_genes_used=97)
        ),
        resolvability=ResolvabilityTrust(),
    )
    assert (decision.state, decision.validation_basis) == ("validated", "real_data")


def test_missing_coverage_or_resolvability_is_recorded(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    before_prep = decide(table, gene_ids, coverage=False)
    assert before_prep.state == "validated" and before_prep.complete is False
    assert "coverage_unavailable" in [note.code for note in before_prep.notes]
    # Before PREP an unlisted panel previews as provisional: the resolvability
    # fail-safe needs a bundle that lacks its self-map.
    preview = decide(table, ids(100, offset=500), coverage=False, resolvability=None)
    assert preview.state == "provisional" and not preview.complete
    assert [note.code for note in preview.notes] == ["coverage_unavailable"]
    refused_early = decide(
        table, ids(100, offset=500), coverage=False, gene_status="refused"
    )
    assert refused_early.state == "refused"
    no_selfmap = decide(table, gene_ids, resolvability=None)
    assert no_selfmap.state == "validated"
    assert "resolvability_not_run" in [note.code for note in no_selfmap.notes]
    disabled = decide(
        table,
        ids(100, offset=500),
        resolvability=None,
        rules=TrustRules(resolvability_enabled=False),
    )
    assert disabled.state == "broad_only"
    assert "disabled" in disabled.reasons[0].detail
    # Roles without a self-map are not held to the resolvability fail-safe.
    sensitivity = trust_state(
        reference_id="ll",
        role="likelihood",
        species="human",
        panel_hash="a" * 64,
        n_panel_genes=100,
        family=None,
        validated=table,
        coverage=CoverageDiagnostics.from_bundle_manifest(
            bundle_manifest(n_query_genes_used=100)
        ),
    )
    assert sensitivity.state == "provisional" and sensitivity.complete


def test_trust_rules_follow_the_config() -> None:
    config = AnnotationConfig(species="human")
    rules = TrustRules.from_config(config)
    assert (rules.min_mapped_genes, rules.min_root_markers) == (50, 10)
    assert rules.resolvability_enabled is True


# --------------------------------------------------------------------------
# Effects


def test_effects_of_each_state(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    other = ids(100, offset=500)
    refused = decide(table, other, root=3)
    assert refused.map_skipped and refused.exclude_hard
    assert (refused.gate_level_cap, refused.gate_reason) == ("failed", "panel_refused")
    assert refused.level_status_override("broad") == CellStatus.NOT_ATTEMPTED_GATE
    assert refused.banner and not refused.gate_warning
    secondary = decide(table, other, root=3, role="secondary")
    assert secondary.degraded_mode == "single_method"
    assert secondary.gate_level_cap is None and not secondary.exclude_hard
    broad_only = decide(table, other, resolvability="broad_only")
    assert not broad_only.map_skipped
    assert (broad_only.gate_level_cap, broad_only.gate_reason) == (
        "broad_only",
        "panel_broad_only",
    )
    assert broad_only.subcluster_status == "not_resolvable_panel"
    assert (
        broad_only.level_status_override("supercluster")
        == CellStatus.NOT_ATTEMPTED_GATE
    )
    assert broad_only.level_status_override("cluster") == CellStatus.NOT_ATTEMPTED_GATE
    for level in ("lineage", "broad", "nt"):
        assert broad_only.level_status_override(level) is None
    provisional = decide(table, other)
    assert provisional.gate_level_cap is None  # warns, never lowers the level
    assert provisional.banner and provisional.gate_warning
    validated = decide(table, gene_ids)
    assert not validated.banner and not validated.gate_warning
    assert validated.effects() == {
        "map_skipped": False,
        "gate_level_cap": None,
        "gate_reason": None,
        "degraded_mode": None,
        "exclude_hard": False,
        "subcluster_status": None,
        "banner": False,
        "gate_warning": False,
    }
    record = json.loads(json.dumps(validated.to_json()))
    assert record["state"] == "validated" and record["effects"]["banner"] is False


def test_regimes_thresholds_and_floors(
    real_table: tuple[ValidatedPanelTable, list[str]], tmp_path: Path
) -> None:
    table, gene_ids = real_table
    real = decide(table, gene_ids)
    for level in ("lineage", "broad", "nt", "supercluster", "seaad_subclass"):
        assert real.emission_regime(level) == "validated"
        assert real.threshold_source(level) == "validated_default"
        assert real.floor_source(level) == "packaged"
        assert not real.applies_provisional_margins(level)
    # Levels above validated_max_level are treated as provisional.
    assert real.emission_regime("cluster") == "provisional"
    assert real.floor_source("cluster") == "unknown_panel"
    provisional = decide(table, ids(100, offset=500))
    assert provisional.emission_regime("broad") == "provisional"
    assert provisional.threshold_source("broad") == "resolvability_local"
    assert provisional.floor_source("broad") == "unknown_panel"
    assert provisional.floor_warning("broad")
    sim_table, sim_ids = simulation_table(tmp_path)
    simulation = decide_sim(sim_table, sim_ids)
    assert (
        simulation.state == "validated" and simulation.validation_basis == "simulation"
    )
    assert simulation.emission_regime("broad") == "provisional"
    assert simulation.floor_source("broad") == "simulation_validated"
    assert not simulation.floor_warning("broad")
    assert simulation.applies_provisional_margins("broad")


def decide_sim(
    table: ValidatedPanelTable, gene_ids: Sequence[str], **kwargs: Any
) -> TrustDecision:
    family = panel_family(
        gene_ids,
        species="human",
        platforms=["XENIUM"],
        known_families=table.known_families(),
    )
    return trust_state(
        reference_id="whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=compute_panel_hash(sorted(gene_ids)),
        n_panel_genes=len(gene_ids),
        family=family,
        validated=table,
        coverage=CoverageDiagnostics.from_bundle_manifest(
            bundle_manifest(n_query_genes_used=len(gene_ids))
        ),
        resolvability=ResolvabilityTrust(**kwargs),
    )


# --------------------------------------------------------------------------
# Validated region and the promotion property


def test_simulation_family_validates_per_level_class_and_depth(tmp_path: Path) -> None:
    table, gene_ids = simulation_table(tmp_path)
    decision = decide_sim(table, gene_ids)
    assert decision.is_validated("broad", "Exc", 15)
    assert not decision.is_validated("broad", "Exc", 14)
    assert decision.is_validated("broad", "Astro", 60)
    assert not decision.is_validated("broad", "Astro", 30)
    # A failed or unevaluable class stays provisional without blocking others.
    assert not decision.is_validated("supercluster", "Exc", 500)
    assert not decision.is_validated("broad", "Vascular", 500)
    assert not decision.is_validated("broad", None, 500)
    assert decision.is_extrapolated("broad", "Exc", 250)
    assert not decision.is_extrapolated("broad", "Exc", 120)
    classes = np.array(
        ["Exc", "Exc", "Astro", "Astro", "Vascular", "Exc"], dtype=object
    )
    depths = np.array([10, 30, 30, 90, 400, 200])
    confident = np.array([True, True, True, True, True, False])
    mask = decision.validated_mask("broad", classes, depths, confident)
    assert mask.tolist() == [False, True, False, True, False, False]
    overall, per_class = validated_share_by_class(
        decision, "broad", classes, depths, confident
    )
    assert overall == pytest.approx(0.4)
    assert per_class == {"astro": 0.5, "exc": 0.5, "vascular": 0.0}
    warn, reasons = decision.unvalidated_share_warning(
        {"broad": overall, "supercluster": None}
    )
    assert warn and reasons == ["unvalidated_share:broad"]
    assert decision.unvalidated_share_warning({"broad": 0.9}) == (False, [])


def test_real_data_family_validates_every_confident_label_up_to_its_max(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    decision = decide(table, gene_ids)
    confident = np.array([True, False, True])
    mask = decision.validated_mask(
        "supercluster", ["Exc", "Exc", None], [5, 500, 500], confident
    )
    assert mask.tolist() == [True, False, True]
    assert not decision.validated_mask(
        "cluster", ["Exc"] * 3, [500] * 3, confident
    ).any()
    assert decision.unvalidated_share_warning({"broad": 0.0}) == (False, [])
    provisional = decide(table, ids(100, offset=500))
    assert not provisional.validated_mask("broad", ["Exc"], [500], [True]).any()


def test_promotion_by_simulation_never_changes_what_is_emitted(tmp_path: Path) -> None:
    """Provisional -> validated (simulation) changes only validation marks."""
    sim_table, gene_ids = simulation_table(tmp_path)
    empty = ValidatedPanelTable(records=())
    rng = np.random.default_rng(0)
    levels = ["lineage", "broad", "nt", "supercluster", "seaad_subclass", "cluster"]
    for constraint in ({}, {"state": "broad_only", "reasons": ("x",)}):
        before = decide_sim(empty, gene_ids, **constraint)
        after = decide_sim(sim_table, gene_ids, **constraint)
        if constraint:
            # Resolvability's broad-only verdict wins over any validation.
            assert before.state == after.state == "broad_only"
        else:
            assert (before.state, after.state) == ("provisional", "validated")
        for level in levels:
            assert before.emission_regime(level) == after.emission_regime(level)
            assert before.threshold_source(level) == after.threshold_source(level)
            assert before.applies_provisional_margins(level) == (
                after.applies_provisional_margins(level)
            )
            assert before.level_status_override(level) == after.level_status_override(
                level
            )
        for name in (
            "map_skipped",
            "gate_level_cap",
            "degraded_mode",
            "exclude_hard",
            "subcluster_status",
        ):
            assert getattr(before, name) == getattr(after, name)
        classes = rng.choice(["Exc", "Astro", "Vascular", "Inh"], size=200)
        depths = rng.integers(10, 400, size=200)
        confident = rng.random(200) < 0.7
        before_mask = before.validated_mask("broad", classes, depths, confident)
        after_mask = after.validated_mask("broad", classes, depths, confident)
        assert not before_mask.any()
        assert not (after_mask & ~confident).any()
        if not constraint:
            assert after_mask.any()
            assert before.banner and not after.banner
            assert before.gate_warning and not after.gate_warning


# --------------------------------------------------------------------------
# Provenance and the panel report


def test_panel_provenance_round_trips(
    real_table: tuple[ValidatedPanelTable, list[str]],
) -> None:
    table, gene_ids = real_table
    family = family_of(table, gene_ids)
    panel = annotation_panel(gene_ids, family)
    diagnostics = panel_diagnostics(
        panel,
        panel_report={"declared_panels": {"S_X": report_entry()}},
        bundles=[bundle_manifest(panel_hash=panel.panel_hash, n_query_genes_used=100)],
    )
    decision = trust_for_panel(
        diagnostics,
        reference_id="whb_frontal_supc_clus",
        role="primary",
        validated=table,
    )
    assert decision.state == "validated" and decision.family_basis == "listed"
    provenance = panel_provenance(
        decision,
        diagnostics,
        panel_mode="per_platform",
        validated_share={"broad": 1.0},
        n_missing_panel_genes=0,
    )
    assert provenance.banner is False and provenance.trust_reasons == [
        "family_validated"
    ]
    assert provenance.validated_panels_sha256 == table.sha256[VALIDATED_PANELS_FILE]
    assert provenance.controls_removed == {
        "feature_type_negative_control_probe": 2,
        "name_blank": 1,
    }
    assert provenance.gene_id_resolution == {"fallback_table": 2, "native": 298}
    full = AnnotationProvenance(species="human", panel=provenance)
    assert AnnotationProvenance.from_uns_json(full.to_uns_json()) == full
    refused = decide(table, ids(100, offset=500), root=3)
    refused_provenance = panel_provenance(refused, diagnostics)
    assert refused_provenance.panel_trust == "refused" and refused_provenance.banner


def test_the_panel_report_records_the_family_validation(tmp_path: Path) -> None:
    gene_ids = ids(60)
    path = tmp_path / "genes.csv"
    pd.DataFrame({"ensembl_id": gene_ids}).to_csv(path, index=False)
    unlisted = panel_from_gene_list(
        path, "human", output_dir=tmp_path / "a", platform="XENIUM"
    )
    entry = unlisted.report["annotation_panels"]["gene_list"]["family_validation"]
    assert entry["listed"] is False and entry["expected_trust"] == "provisional"
    assert unlisted.report["validated_panels"]["source"] == "packaged"
    directory = write_tables(tmp_path / "tables", [panel_row("custom", gene_ids)])
    config = AnnotationConfig(species="human")
    config = config.model_copy(
        update={
            "panel": config.panel.model_copy(
                update={"validated_panels_path": directory / VALIDATED_PANELS_FILE}
            )
        }
    )
    listed = panel_from_gene_list(
        path, "human", output_dir=tmp_path / "b", platform="XENIUM", config=config
    )
    entry = listed.report["annotation_panels"]["gene_list"]["family_validation"]
    assert entry["listed"] is True and entry["family_id"] == "test_family"
    assert entry["validation_basis"] == "real_data"
    panel = load_annotation_panel(tmp_path / "b" / PANEL_GENES_FILE)
    assert panel.panel_family is not None and panel.panel_family.basis == "listed"
    assert listed.report["validated_panels"]["families"]["test_family"][
        "panel_ids"
    ] == ["custom"]
