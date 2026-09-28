"""Tests for the gene-ID resolver and the exact-case species test (plan §8.4)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.config import AnnotationPanelConfig
from merxen.annotation.gene_ids import (
    FeatureInput,
    GeneIdSources,
    GeneTable,
    ResolutionRules,
    clean_native_value,
    gene_id_sources,
    load_alias_table,
    load_gene_table,
    load_overrides,
    native_kind,
    resolve_gene_ids,
    species_check,
    strip_version,
    summing_matrix,
)

# Symbols of the current panels (the first 40 of each resolved panel):
# ag7 (MERSCOPE mouse, 500 genes), VZG2 (MERSCOPE mouse, 815 genes) and the
# P7513 human set a (MERSCOPE / Xenium, 297 genes).
AG7_SYMBOLS = [
    "Ngfr", "Ccnd2", "Th", "Lhx2", "Sox9", "Oprm1", "Dbh", "Nfix", "Npas1",
    "Klf4", "Rab3b", "Crhr2", "Ddr1", "Calb2", "Grm3", "Etv1", "Sst", "Chn2",
    "Crabp2", "Bcan", "Ndrg1", "Prlr", "Slc1a6", "Igf1r", "Kit", "Agrp",
    "Pvalb", "Trh", "Tph2", "Sulf2", "Tcap", "Rbfox1", "Bmp7", "Cabp7",
    "Slc47a1", "Prox1", "Ebf3", "Gabra1", "Vipr2", "Pax5",
]  # fmt: skip
VZG2_SYMBOLS = [
    "Ngfr", "Ccnd2", "Th", "Lhx2", "Glra1", "Sox9", "Hoxb6", "Serpinf1",
    "Oprm1", "Car4", "Bcl11a", "Dbh", "Col6a1", "Col18a1", "Nkx2-1", "Irx2",
    "Col1a1", "Nfix", "Grik3", "Gria3", "Npas1", "Ppp1r17", "Cd36", "Rab3b",
    "St8sia6", "Crhr2", "Ddr1", "Fosb", "Calb2", "Man1a", "Grm3", "Etv1",
    "Pax2", "Sst", "Chn2", "Aqp1", "Bcan", "Cd44", "Prlr", "AW551984",
]  # fmt: skip
P7513_MERSCOPE_SYMBOLS = [
    "LAMP2", "TAC1", "PAX6", "MGST1", "TENM1", "HHATL", "CD4", "DCN", "CLDN11",
    "RNASET2", "ANK1", "VCAN", "CDH1", "MYO16", "CAPG", "ROS1", "FSTL4",
    "NNAT", "NPFFR2", "ATG5", "ATP2C2", "CALCRL", "ERBB3", "SPI1", "FGFR2",
    "TP53BP1", "FGFR3", "TFE3", "ABCC9", "TRPC5", "TRHDE", "TSG101",
    "STXBP2", "FBLN1", "ITGA8", "PSEN1", "HSP90AA1", "PTPRC", "C1orf162",
    "H2AX",
]  # fmt: skip


def _mouse_case(symbol: str) -> str:
    """The MGI spelling of a gene symbol (as in WMB gene.csv) for the fixtures."""
    if symbol.isupper() and symbol.startswith("AW"):
        return symbol
    return symbol[:1].upper() + symbol[1:].lower()


def _both_tables() -> dict[str, GeneTable]:
    """Local WHB / WMB-like tables holding every fixture symbol in its own case."""
    universe = sorted(
        {s.upper() for s in AG7_SYMBOLS + VZG2_SYMBOLS + P7513_MERSCOPE_SYMBOLS}
    )
    human = [
        ("C1orf162" if symbol == "C1ORF162" else symbol, f"ENSG{index:011d}")
        for index, symbol in enumerate(universe)
    ]
    mouse = [
        (
            "AW551984" if symbol == "AW551984" else _mouse_case(symbol),
            f"ENSMUSG{index:011d}",
        )
        for index, symbol in enumerate(universe)
    ]
    return {
        "human": GeneTable.from_pairs("human", human, path="whb_gene.csv"),
        "mouse": GeneTable.from_pairs("mouse", mouse, path="wmb_gene.csv"),
    }


def _symbols_only(symbols: list[str]) -> list[FeatureInput]:
    return [FeatureInput(name=symbol, symbol=symbol) for symbol in symbols]


def _sources(species: str, **kwargs: object) -> GeneIdSources:
    return GeneIdSources(
        species=species,  # type: ignore[arg-type]
        gene_tables=kwargs.pop("gene_tables", _both_tables()),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Version suffixes and native IDs


def test_version_suffixes_are_stripped() -> None:
    assert strip_version("ENSG00000131095.12") == "ENSG00000131095"
    assert strip_version(" ENSMUSG00000020932.3 ") == "ENSMUSG00000020932"
    assert strip_version("GFAP") == "GFAP"
    assert clean_native_value("ENSG00000131095.7") == "ENSG00000131095"
    # MERSCOPE codebooks write -1 for features without a transcript ID.
    assert clean_native_value("-1") == ""
    assert clean_native_value(float("nan")) == ""


def test_native_ids_with_a_version_resolve_natively() -> None:
    features = [
        FeatureInput(name="TAC1", symbol="TAC1", native_value="ENSG00000000033.4"),
        FeatureInput(name="PAX6", symbol="PAX6", native_value="ENSG00000000031.12"),
    ]
    table = GeneTable.from_pairs(
        "human", [("TAC1", "ENSG00000000033"), ("PAX6", "ENSG00000000031")]
    )

    result = resolve_gene_ids(
        features, "human", GeneIdSources(species="human", gene_tables={"human": table})
    )

    assert result.status == "ok"
    assert result.ids_by_name() == {
        "TAC1": "ENSG00000000033",
        "PAX6": "ENSG00000000031",
    }
    assert result.source_counts() == {"native": 2}
    assert result.native_prefix_share == 1.0


def test_native_kinds() -> None:
    assert native_kind("", "human") == "none"
    assert native_kind("ENSG00000131095", "human") == "run"
    assert native_kind("ENSMUSG00000020932", "human") == "other_species"
    assert native_kind("ENST00000262410", "human") == "non_gene"
    assert native_kind("ENST00000262410_MAPT3R_exon9", "human") == "non_gene"
    assert native_kind("Gfap", "mouse") == "non_id"
    # Ensembl gene IDs in another case are still the run species' IDs.
    assert native_kind("ensg00000141510", "human") == "run"
    assert native_kind("Ensg00000141510", "human") == "run"
    assert native_kind("ensmusg00000020932", "human") == "other_species"


def test_lower_case_native_ids_resolve_natively() -> None:
    assert clean_native_value("ensg00000141510.3") == "ENSG00000141510"
    assert clean_native_value("Ensg00000141510") == "ENSG00000141510"
    assert clean_native_value("Gfap") == "Gfap"
    table = GeneTable.from_pairs("human", [("TP53", "ENSG00000141510")])
    result = resolve_gene_ids(
        [FeatureInput("TP53", "TP53", native_value="ensg00000141510")],
        "human",
        GeneIdSources(species="human", gene_tables={"human": table}),
    )
    assert result.status == "ok"
    assert result.ids_by_name() == {"TP53": "ENSG00000141510"}
    assert result.features[0].source == "native"
    assert result.other_species_share == 0.0


def test_release_drift_takes_the_reference_id() -> None:
    table = GeneTable.from_pairs("human", [("H2AX", "ENSG00000188486")])
    features = [FeatureInput("H2AFX", "H2AX", native_value="ENSG00000999999")]

    result = resolve_gene_ids(
        features, "human", GeneIdSources(species="human", gene_tables={"human": table})
    )

    (feature,) = result.features
    assert feature.gene_id == "ENSG00000188486"
    assert feature.source == "symbol_fallback"
    assert result.release_drift == {"ENSG00000999999": "ENSG00000188486"}


# --------------------------------------------------------------------------
# The exact-case species test


@pytest.mark.parametrize("symbols", [AG7_SYMBOLS, VZG2_SYMBOLS], ids=["ag7", "vzg2"])
def test_mouse_panels_under_species_human_are_refused(symbols: list[str]) -> None:
    result = resolve_gene_ids(_symbols_only(symbols), "human", _sources("human"))

    # The case-insensitive fallback resolves them to human IDs ...
    assert result.resolution_share == 1.0
    # ... but the exact-case test refuses the panel.
    assert result.status == "refused"
    assert "species_mismatch" in result.refusal_reasons
    check = result.species_check
    assert check.status == "species_mismatch"
    assert check.exact_matches["mouse"] > check.exact_matches["human"]
    assert check.exact_case_ratio is not None and check.exact_case_ratio < 0.5


def test_a_human_merscope_panel_under_species_mouse_is_refused() -> None:
    result = resolve_gene_ids(
        _symbols_only(P7513_MERSCOPE_SYMBOLS), "mouse", _sources("mouse")
    )

    assert result.status == "refused"
    assert result.refusal_reasons[0] == "species_mismatch"
    assert result.species_check.exact_matches["human"] == len(P7513_MERSCOPE_SYMBOLS)


def test_the_ratio_rule_alone_refuses_without_the_other_table() -> None:
    human_only = {"human": _both_tables()["human"]}

    result = resolve_gene_ids(
        _symbols_only(AG7_SYMBOLS), "human", _sources("human", gene_tables=human_only)
    )

    assert result.refusal_reasons == ["species_mismatch"]
    check = result.species_check
    assert check.exact_matches == {"human": 0}
    assert check.casefold_matches == {"human": len(AG7_SYMBOLS)}
    assert check.exact_case_ratio == 0.0
    assert check.tables["mouse"] is None


def test_the_other_table_rule_alone_refuses_when_the_ratio_passes() -> None:
    # Mouse symbols under species = human: the human table holds only 6 of
    # the 40, all in exact case (ratio 1.0, the ratio rule passes), while
    # the mouse table holds all 40 in exact case.
    few = AG7_SYMBOLS[:6]
    human = GeneTable.from_pairs(
        "human",
        [(symbol, f"ENSG{index:011d}") for index, symbol in enumerate(few)],
        path="whb_gene.csv",
    )
    tables = {"human": human, "mouse": _both_tables()["mouse"]}

    result = resolve_gene_ids(
        _symbols_only(AG7_SYMBOLS),
        "human",
        _sources("human", gene_tables=tables),
        rules=ResolutionRules(min_gene_id_resolution=0.1),
    )

    assert result.refusal_reasons == ["species_mismatch"]
    check = result.species_check
    assert check.exact_case_ratio == 1.0
    assert check.exact_matches == {"human": 6, "mouse": len(AG7_SYMBOLS)}
    assert check.reasons == [
        f"{len(AG7_SYMBOLS)} symbols match the mouse gene table in exact case "
        "vs 6 for human"
    ]
    assert check.tables["mouse"] == "wmb_gene.csv"


@pytest.mark.parametrize(
    ("symbols", "species"),
    [
        (AG7_SYMBOLS, "mouse"),
        (VZG2_SYMBOLS, "mouse"),
        (P7513_MERSCOPE_SYMBOLS, "human"),
    ],
    ids=["ag7", "vzg2", "p7513"],
)
def test_panels_of_the_run_species_pass(symbols: list[str], species: str) -> None:
    result = resolve_gene_ids(_symbols_only(symbols), species, _sources(species))  # type: ignore[arg-type]

    assert result.status == "ok", result.refusal_details
    assert result.species_check.status == "pass"
    assert result.species_check.exact_case_ratio == 1.0
    assert result.source_counts() == {"fallback_table": len(symbols)}


def test_species_check_is_not_evaluable_without_a_table() -> None:
    check = species_check(["GFAP"], "human", {})
    assert check.status == "not_evaluable"
    assert check.tables == {"human": None, "mouse": None}


# --------------------------------------------------------------------------
# Resolution order, duplicates, aliases, overrides, unmapped


def test_resolution_order_and_sources(tmp_path: Path) -> None:
    table = GeneTable.from_pairs(
        "human",
        [
            ("GENE0", "ENSG00000000010"),
            ("GENE1", "ENSG00000000011"),
            ("GENE2", "ENSG00000000012"),
            ("NEWNAME", "ENSG00000000013"),
        ],
    )
    aliases = tmp_path / "hgnc.tsv"
    pd.DataFrame(
        {
            "symbol": ["NEWNAME"],
            "alias_symbol": ["OLDNAME"],
            "prev_symbol": [""],
            "ensembl_gene_id": ["ENSG00000000013"],
        }
    ).to_csv(aliases, sep="\t", index=False)
    features = [
        FeatureInput("GENE0", "GENE0", native_value="ENSG00000000010.2"),
        FeatureInput("GENE1", "GENE1"),
        FeatureInput("gene2", "gene2"),
        FeatureInput("OLDNAME", "OLDNAME"),
        FeatureInput("H2AX", "H2AX"),
    ]
    sources = GeneIdSources(
        species="human",
        gene_tables={"human": table},
        pair_lookup={"GENE1": "ENSG00000000011"},
        aliases=load_alias_table(aliases, "human"),
        overrides=load_overrides("human"),
    )

    result = resolve_gene_ids(features, "human", sources)

    assert {f.name: f.source for f in result.features} == {
        "GENE0": "native",
        "GENE1": "pair_lookup",
        "gene2": "fallback_table",  # case-insensitive, unique candidate
        "OLDNAME": "alias",
        "H2AX": "override",  # the packaged M0e override
    }
    assert result.ids_by_name()["H2AX"] == "ENSG00000188486"
    assert result.status == "ok"


def test_exact_case_symbol_wins_over_a_case_variant() -> None:
    table = GeneTable.from_pairs(
        "mouse", [("Abc1", "ENSMUSG00000000001"), ("ABC1", "ENSMUSG00000000002")]
    )
    sources = GeneIdSources(species="mouse", gene_tables={"mouse": table})

    result = resolve_gene_ids(_symbols_only(["ABC1", "abc1"]), "mouse", sources)

    assert result.ids_by_name()["ABC1"] == "ENSMUSG00000000002"
    # No exact match and two case-insensitive candidates: ambiguous.
    assert result.unmapped() == {"abc1": "ambiguous_in_fallback_table"}


def test_duplicates_are_merged_and_summed() -> None:
    table = GeneTable.from_pairs("human", [("H2AX", "ENSG00000188486")])
    features = [
        FeatureInput("H2AX", "H2AX"),
        FeatureInput("H2AFX", "H2AFX", native_value="ENSG00000188486"),
    ]

    result = resolve_gene_ids(
        features, "human", GeneIdSources(species="human", gene_tables={"human": table})
    )

    assert result.merged_duplicates() == {"ENSG00000188486": ["H2AX", "H2AFX"]}
    assert result.gene_ids == ["ENSG00000188486"]
    ids = [result.ids_by_name()[f.name] for f in features] + [""]
    counts = sparse.csr_matrix(np.array([[2, 3, 9], [1, 0, 4]], dtype=np.float32))
    summed = counts @ summing_matrix(ids, ["ENSG00000188486"])
    np.testing.assert_array_equal(summed.toarray(), [[5], [1]])


def test_single_target_aliases_only(tmp_path: Path) -> None:
    table = GeneTable.from_pairs(
        "mouse",
        [
            ("Snap25", "ENSMUSG00000027273"),
            ("Gfap", "ENSMUSG00000020932"),
            ("Aqp4", "ENSMUSG00000024411"),
        ],
    )
    aliases = tmp_path / "mgi.tsv"
    pd.DataFrame(
        {
            "Marker Symbol": ["Snap25", "Gfap", "Aqp4"],
            "Marker Synonyms (pipe-separated)": [
                "Snap|SNAP-25",
                "Gfa|Shared",
                "Shared",
            ],
        }
    ).to_csv(aliases, sep="\t", index=False)
    sources = GeneIdSources(
        species="mouse",
        gene_tables={"mouse": table},
        aliases=load_alias_table(aliases, "mouse"),
    )

    result = resolve_gene_ids(_symbols_only(["SNAP-25", "Shared"]), "mouse", sources)

    assert result.ids_by_name() == {"SNAP-25": "ENSMUSG00000027273"}
    assert result.unmapped() == {"Shared": "ambiguous_alias"}
    assert result.features[0].detail == "alias of Snap25"


def test_a_current_symbol_resolves_through_its_previous_symbol(
    tmp_path: Path,
) -> None:
    # The gene table predates the H2AFX -> H2AX rename (the M0e case the
    # override CSV patched); the HGNC row carries the current symbol's ID.
    hgnc = tmp_path / "hgnc.tsv"
    pd.DataFrame(
        {
            "symbol": ["H2AX", "OTHER"],
            "alias_symbol": ["", ""],
            "prev_symbol": ["H2AFX", "OLDOTHER"],
            "ensembl_gene_id": ["ENSG00000188486", ""],
        }
    ).to_csv(hgnc, sep="\t", index=False)
    aliases = load_alias_table(hgnc, "human")
    old_table = GeneTable.from_pairs(
        "human", [("H2AFX", "ENSG00000188486"), ("OLDOTHER", "ENSG00000000077")]
    )
    sources = GeneIdSources(
        species="human", gene_tables={"human": old_table}, aliases=aliases
    )
    result = resolve_gene_ids(_symbols_only(["H2AX", "OTHER"]), "human", sources)
    assert result.ids_by_name() == {
        "H2AX": "ENSG00000188486",
        # No ID on its row: the gene table's ID of its previous symbol.
        "OTHER": "ENSG00000000077",
    }
    assert {f.name: f.source for f in result.features} == {
        "H2AX": "alias",
        "OTHER": "alias",
    }
    assert result.features[0].detail == "approved symbol of H2AFX"
    # The row's own ID resolves without any gene table, too.
    alone = GeneIdSources(species="human", aliases=aliases)
    assert aliases.resolve("H2AX", None) == (
        "ENSG00000188486",
        "approved symbol of H2AFX",
        "",
    )
    assert resolve_gene_ids(_symbols_only(["H2AX"]), "human", alone).ids_by_name() == {
        "H2AX": "ENSG00000188486"
    }
    # The single-target rule holds: a previous symbol that the gene table
    # gives another ID makes the current symbol ambiguous.
    clash = GeneTable.from_pairs("human", [("H2AFX", "ENSG00000000099")])
    ambiguous = GeneIdSources(
        species="human", gene_tables={"human": clash}, aliases=aliases
    )
    assert resolve_gene_ids(_symbols_only(["H2AX"]), "human", ambiguous).unmapped() == {
        "H2AX": "ambiguous_alias"
    }


def test_overrides_need_a_reason_and_the_run_species(tmp_path: Path) -> None:
    good = tmp_path / "overrides.csv"
    good.write_text("symbol,ensembl_id,reason\nH2AX,ENSG00000000077,local fix\n")
    rows = load_overrides("human", good)
    # A configured row replaces the packaged one of the same symbol.
    assert rows["H2AX"].ensembl_id == "ENSG00000000077"
    assert rows["LIF"].reason.startswith("M0e")
    no_reason = tmp_path / "no_reason.csv"
    no_reason.write_text("symbol,ensembl_id,reason\nGFP,ENSG00000000001,\n")
    with pytest.raises(ValueError, match="no reason"):
        load_overrides("human", no_reason)
    other = tmp_path / "other.csv"
    other.write_text("symbol,ensembl_id,reason\nGfap,ENSMUSG00000020932,x\n")
    with pytest.raises(ValueError, match="not a human Ensembl gene ID"):
        load_overrides("human", other)
    assert load_overrides("mouse")["H2ax"].ensembl_id == "ENSMUSG00000049932"


def test_unmapped_features_are_listed_with_their_reason() -> None:
    table = GeneTable.from_pairs(
        "mouse", [(f"Gene{i}", f"ENSMUSG{i:011d}") for i in range(40)]
    )
    symbols = [f"Gene{i}" for i in range(40)] + ["GFP", "tdTomato", "Snap25a"]

    result = resolve_gene_ids(
        _symbols_only(symbols),
        "mouse",
        GeneIdSources(species="mouse", gene_tables={"mouse": table}),
    )

    assert result.unmapped() == {
        "GFP": "not_in_fallback_table",
        "tdTomato": "not_in_fallback_table",
        "Snap25a": "not_in_fallback_table",
    }
    # 40 of 43 resolve (0.93 < 0.95): refused, with the unmapped listed.
    assert result.refusal_reasons == ["gene_id_resolution"]
    lenient = resolve_gene_ids(
        _symbols_only(symbols),
        "mouse",
        GeneIdSources(species="mouse", gene_tables={"mouse": table}),
        rules=ResolutionRules(min_gene_id_resolution=0.9),
    )
    assert lenient.status == "ok"
    assert set(lenient.summary()["unmapped"]) == {"GFP", "tdTomato", "Snap25a"}


# --------------------------------------------------------------------------
# Refusals of the ID column


def test_transcript_ids_of_a_codebook_do_not_count_as_ids() -> None:
    features = [
        FeatureInput(symbol, symbol, native_value=f"ENST{index:011d}")
        for index, symbol in enumerate(P7513_MERSCOPE_SYMBOLS)
    ]

    result = resolve_gene_ids(features, "human", _sources("human"))

    assert result.status == "ok"
    assert result.n_native_values == 0
    assert result.native_prefix_share is None
    assert result.source_counts() == {"fallback_table": len(features)}


# --------------------------------------------------------------------------
# Local gene tables and config


def test_gene_tables_from_gene_csv_and_h5ad_var(tmp_path: Path) -> None:
    csv = tmp_path / "gene.csv"
    pd.DataFrame(
        {
            "gene_identifier": ["ENSMUSG00000020932", "ENSMUSG00000024411"],
            "gene_symbol": ["Gfap", "Aqp4"],
        }
    ).to_csv(csv, index=False)
    h5ad = tmp_path / "reference.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix((1, 2), dtype=np.float32),
        var=pd.DataFrame(
            {"gene_symbol": ["GFAP", "AQP4"]},
            index=["ENSG00000131095.5", "ENSG00000171885"],
        ),
    ).write_h5ad(h5ad)

    mouse = load_gene_table(csv, "mouse")
    human = load_gene_table(h5ad, "human")

    assert mouse is not None and mouse.unique_id("Gfap") == ("ENSMUSG00000020932", "")
    assert human is not None and human.unique_id("GFAP") == ("ENSG00000131095", "")
    # A table of the other species holds no ID of the run species.
    assert load_gene_table(csv, "human") is None

    config = AnnotationPanelConfig(
        gene_id_fallback_csv=h5ad, gene_tables={"mouse": csv}
    )
    sources = gene_id_sources(config, "human")
    assert sources.run_table is not None and sources.run_table.path == str(h5ad)
    assert set(sources.gene_tables) == {"human", "mouse"}
    described = sources.describe()
    assert described["gene_tables"]["mouse"]["n_symbols"] == 2
    json.dumps(described)
    # gene_tables[species] wins over the M0e fallback table.
    explicit = gene_id_sources(
        AnnotationPanelConfig(
            gene_id_fallback_csv=csv, gene_tables={"human": h5ad, "mouse": csv}
        ),
        "human",
    )
    assert explicit.run_table is not None and explicit.run_table.path == str(h5ad)


def test_resolution_table_sha256_ignores_feature_order() -> None:
    sources = _sources("human")
    forward = resolve_gene_ids(_symbols_only(P7513_MERSCOPE_SYMBOLS), "human", sources)
    backward = resolve_gene_ids(
        _symbols_only(P7513_MERSCOPE_SYMBOLS[::-1]), "human", sources
    )
    assert forward.table_sha256() == backward.table_sha256()
    assert forward.summary()["resolution_table_sha256"] == forward.table_sha256()
