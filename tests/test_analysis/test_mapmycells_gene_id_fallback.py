"""Tests for the local gene-ID fallback used when building MapMyCells queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merxen.analysis.mapmycells import (
    load_gene_id_fallback_table,
    prepare_mapmycells_query,
    run_mapmycells,
    summarize_gene_id_resolution,
)
from merxen.config import MapMyCellsConfig, MapMyCellsSampleConfig

# Human MERSCOPE panel genes whose Ensembl IDs are blank in the P7513 var.
MERSCOPE_MISSING_IDS = {
    "LIF": "ENSG00000128342",
    "LIFR": "ENSG00000113594",
    "SQSTM1": "ENSG00000161011",
    "H2AX": "ENSG00000188486",
}


@pytest.fixture
def whb_gene_csv(tmp_path: Path) -> Path:
    """Write a small gene table in the Allen WHB ``gene.csv`` layout."""
    rows = [
        ("ENSG00000152661", "GJA1", "protein_coding"),
        ("ENSG00000142192", "APP", "protein_coding"),
        *(
            (gene_id, symbol, "protein_coding")
            for symbol, gene_id in MERSCOPE_MISSING_IDS.items()
        ),
        # Two IDs for one symbol: must not be guessed.
        ("ENSG00000000001", "DUPSYM", "lncRNA"),
        ("ENSG00000000002", "DUPSYM", "lncRNA"),
        # Mouse row: must never reach a human query.
        ("ENSMUSG00000034394", "MOUSEONLY", "protein_coding"),
    ]
    path = tmp_path / "gene.csv"
    pd.DataFrame(rows, columns=["gene_identifier", "gene_symbol", "biotype"]).to_csv(
        path, index=False
    )
    return path


def _write_clustered(path: Path, var: pd.DataFrame) -> Path:
    n_genes = len(var)
    adata = ad.AnnData(
        X=np.ones((2, n_genes), dtype=np.float32),
        obs=pd.DataFrame(index=["cell1", "cell2"]),
        var=var,
    )
    adata.layers["counts"] = np.ones((2, n_genes), dtype=np.int64)
    adata.obsm["X_umap"] = np.zeros((2, 2), dtype=np.float32)
    adata.obsm["spatial"] = np.zeros((2, 2), dtype=np.float32)
    adata.write_h5ad(path)
    return path


def _merscope_var(extra_symbols: tuple[str, ...] = ()) -> pd.DataFrame:
    symbols = ["GJA1", "APP", *MERSCOPE_MISSING_IDS, *extra_symbols]
    ids = ["ENSG00000152661", "ENSG00000142192"] + [""] * (len(symbols) - 2)
    return pd.DataFrame({"gene": symbols, "ensembl_id": ids}, index=symbols)


def test_prepare_merscope_query_fills_missing_ids_from_fallback_csv(
    tmp_path: Path,
    whb_gene_csv: Path,
) -> None:
    """LIF, LIFR, SQSTM1 and H2AX should get IDs from the local gene table."""
    input_h5ad = _write_clustered(tmp_path / "merscope.h5ad", _merscope_var())
    report_path = tmp_path / "query_gene_ids.json"

    output_h5ad = prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query.h5ad",
        gene_id_column="ensembl_id",
        gene_id_fallback=load_gene_id_fallback_table(whb_gene_csv),
        gene_id_report_path=report_path,
    )

    out = ad.read_h5ad(output_h5ad)
    assert list(out.var_names) == [
        "ENSG00000152661",
        "ENSG00000142192",
        *MERSCOPE_MISSING_IDS.values(),
    ]
    report = json.loads(report_path.read_text())
    assert report["resolved_by_fallback"] == MERSCOPE_MISSING_IDS
    assert report["resolved_by_reference_lookup"] == {}
    assert report["unresolved"] == {}
    assert report["n_input_gene_ids"] == 2
    assert report["n_resolved"] == report["n_features"] == 6
    assert report["gene_id_fallback"]["path"] == str(whb_gene_csv)
    assert report["gene_id_fallback"]["n_species_rows"] == 8


def test_prepare_query_reports_genes_the_fallback_cannot_resolve(
    tmp_path: Path,
    whb_gene_csv: Path,
) -> None:
    """Absent, ambiguous, other-species and duplicate-ID genes stay unresolved."""
    var = _merscope_var(extra_symbols=("NOTINTABLE", "DUPSYM", "MOUSEONLY", "gja1"))
    input_h5ad = _write_clustered(tmp_path / "merscope.h5ad", var)
    report_path = tmp_path / "query_gene_ids.json"

    output_h5ad = prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query.h5ad",
        gene_id_column="ensembl_id",
        gene_id_fallback=load_gene_id_fallback_table(whb_gene_csv),
        gene_id_report_path=report_path,
    )

    out = ad.read_h5ad(output_h5ad)
    assert list(out.var_names[-4:]) == ["NOTINTABLE", "DUPSYM", "MOUSEONLY", "gja1"]
    report = json.loads(report_path.read_text())
    assert report["unresolved"] == {
        "NOTINTABLE": "not_in_fallback_table",
        "DUPSYM": "ambiguous_in_fallback_table",
        "MOUSEONLY": "not_in_fallback_table",
        "gja1": "fallback_id_already_in_query",
    }
    assert report["n_unresolved"] == 4
    assert set(report["resolved_by_fallback"]) == set(MERSCOPE_MISSING_IDS)


def test_prepare_query_without_fallback_reports_unresolved_genes(
    tmp_path: Path,
) -> None:
    """Missing IDs should be reported even when no fallback table is set."""
    input_h5ad = _write_clustered(tmp_path / "merscope.h5ad", _merscope_var())
    report_path = tmp_path / "query_gene_ids.json"

    prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query.h5ad",
        gene_id_column="ensembl_id",
        gene_id_lookup={"LIF": "ENSG00000128342"},
        gene_id_report_path=report_path,
    )

    report = json.loads(report_path.read_text())
    assert report["gene_id_fallback"] is None
    assert report["resolved_by_reference_lookup"] == {"LIF": "ENSG00000128342"}
    assert report["resolved_by_fallback"] == {}
    assert report["unresolved"] == {
        "LIFR": "no_fallback_table",
        "SQSTM1": "no_fallback_table",
        "H2AX": "no_fallback_table",
    }


def test_prepare_query_leaves_symbols_for_gene_mapper_untouched(
    tmp_path: Path,
    whb_gene_csv: Path,
) -> None:
    """Symbols deferred to the gene-mapping database must not be mixed with IDs."""
    symbols = ["GJA1", "LIF"]
    input_h5ad = _write_clustered(
        tmp_path / "symbols_only.h5ad", pd.DataFrame(index=symbols)
    )
    report_path = tmp_path / "query_gene_ids.json"

    output_h5ad = prepare_mapmycells_query(
        input_h5ad,
        tmp_path / "query.h5ad",
        gene_id_column="ensembl_id",
        allow_gene_symbol_fallback=True,
        gene_id_fallback=load_gene_id_fallback_table(whb_gene_csv),
        gene_id_report_path=report_path,
    )

    assert list(ad.read_h5ad(output_h5ad).var_names) == symbols
    assert json.loads(report_path.read_text())["deferred_to_gene_mapping_db"] is True


def test_load_gene_id_fallback_table_keeps_only_query_species_ids(
    whb_gene_csv: Path,
) -> None:
    """A human table must not supply IDs to a mouse query, and vice versa."""
    human = load_gene_id_fallback_table(whb_gene_csv, query_species="human")
    mouse = load_gene_id_fallback_table(whb_gene_csv, query_species="mouse")

    assert human.candidate_ids("H2AX") == ("ENSG00000188486",)
    assert human.candidate_ids("h2ax") == ("ENSG00000188486",)
    assert human.candidate_ids("MOUSEONLY") == ()
    assert human.candidate_ids("DUPSYM") == ("ENSG00000000001", "ENSG00000000002")
    assert mouse.n_species_rows == 1
    assert mouse.candidate_ids("H2AX") == ()
    assert mouse.candidate_ids("MOUSEONLY") == ("ENSMUSG00000034394",)


def test_load_gene_id_fallback_table_reads_reference_h5ad_var(tmp_path: Path) -> None:
    """An Allen raw matrix ``var`` (ID index, gene_symbol) is a valid table."""
    reference = ad.AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["cell1"]),
        var=pd.DataFrame(
            {"gene_symbol": ["H2AX", "LIF"]},
            index=pd.Index(["ENSG00000188486.4", "ENSG00000128342"]),
        ),
    )
    path = tmp_path / "WHB-10Xv3-Nonneurons-raw.h5ad"
    reference.write_h5ad(path)

    table = load_gene_id_fallback_table(path)

    assert table.n_rows == table.n_species_rows == 2
    assert table.candidate_ids("H2AX") == ("ENSG00000188486",)
    assert table.candidate_ids("LIF") == ("ENSG00000128342",)


def test_load_gene_id_fallback_table_rejects_missing_or_unusable_tables(
    tmp_path: Path,
) -> None:
    """A configured table that is absent or lacks columns should fail loudly."""
    with pytest.raises(FileNotFoundError, match="gene-ID fallback table"):
        load_gene_id_fallback_table(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["H2AX"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="needs an Ensembl ID column"):
        load_gene_id_fallback_table(bad)


def test_summarize_gene_id_resolution_matches_ids_across_platforms() -> None:
    """MERSCOPE H2AX should be linked to the Xenium feature sharing its ID."""
    reports = {
        "P_MERSCOPE": {
            "resolved_by_reference_lookup": {},
            "resolved_by_fallback": {
                "H2AX": "ENSG00000188486",
                "LIF": "ENSG00000128342",
            },
            "gene_ids": {"H2AX": "ENSG00000188486", "LIF": "ENSG00000128342"},
        },
        "P_XENIUM": {
            "resolved_by_reference_lookup": {},
            "resolved_by_fallback": {},
            "gene_ids": {"H2AFX": "ENSG00000188486", "MAPT3R": ""},
        },
    }

    summary = summarize_gene_id_resolution(reports)

    assert summary["P_MERSCOPE"]["recovered_gene_ids_in_other_samples"] == {
        "H2AX": {"P_XENIUM": "H2AFX"},
        "LIF": {},
    }
    assert summary["P_XENIUM"]["recovered_gene_ids_in_other_samples"] == {}
    assert "gene_ids" not in summary["P_MERSCOPE"]


def test_run_mapmycells_records_fallback_genes_in_results_manifest(
    tmp_path: Path,
    whb_gene_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pair manifest should list fallback genes and the H2AX/H2AFX match."""
    merscope_h5ad = _write_clustered(
        tmp_path / "PAIR1_MERSCOPE_clustered.h5ad", _merscope_var()
    )
    xenium_symbols = ["GJA1", "H2AFX"]
    xenium_h5ad = _write_clustered(
        tmp_path / "PAIR1_XENIUM_clustered.h5ad",
        pd.DataFrame(
            {
                "gene": xenium_symbols,
                "ensembl_id": ["ENSG00000152661", "ENSG00000188486"],
            },
            index=["ENSG00000152661", "ENSG00000188486"],
        ),
    )
    marker_lookup = tmp_path / "markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "stats.h5"
    precomputed_stats.write_bytes(b"stats")
    commands: list[list[str]] = []

    def fake_run_command(command: list[str], **_: Any) -> None:
        commands.append(command)

    monkeypatch.setattr("merxen.analysis.mapmycells._run_command", fake_run_command)
    monkeypatch.setattr(
        "merxen.analysis.mapmycells.annotate_h5ad_with_mapmycells",
        lambda *args, **kwargs: None,
    )
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[
            MapMyCellsSampleConfig(
                sample_id="PAIR1_MERSCOPE",
                platform="MERSCOPE",
                anndata_path=merscope_h5ad,
                gene_id_column="ensembl_id",
            ),
            MapMyCellsSampleConfig(
                sample_id="PAIR1_XENIUM",
                platform="XENIUM",
                anndata_path=xenium_h5ad,
                gene_id_column="ensembl_id",
            ),
        ],
        reference_mode="whole_brain",
        marker_lookup_path=marker_lookup,
        precomputed_stats_path=precomputed_stats,
        region_cache_dir=tmp_path / "empty_cache",
        gene_id_fallback_csv=whb_gene_csv,
    )

    results = run_mapmycells(cfg)

    assert len(commands) == 2
    query = ad.read_h5ad(results["PAIR1_MERSCOPE"]["whole_brain"]["query_h5ad"])
    assert set(MERSCOPE_MISSING_IDS.values()) <= set(query.var_names)
    manifest = json.loads(
        (cfg.output_dir / "PAIR1_mapmycells_manifest.json").read_text()
    )
    assert manifest["gene_id_fallback_csv"] == str(whb_gene_csv)
    merscope = manifest["gene_id_resolution"]["PAIR1_MERSCOPE"]
    assert merscope["resolved_by_fallback"] == MERSCOPE_MISSING_IDS
    assert merscope["recovered_gene_ids_in_other_samples"] == {
        "H2AX": {"PAIR1_XENIUM": "H2AFX"},
        "LIF": {},
        "LIFR": {},
        "SQSTM1": {},
    }
    assert manifest["gene_id_resolution"]["PAIR1_XENIUM"]["n_unresolved"] == 0
    assert (
        "query_gene_ids_json" in (manifest["samples"]["PAIR1_MERSCOPE"]["whole_brain"])
    )


def test_run_mapmycells_fails_fast_on_missing_fallback_table(tmp_path: Path) -> None:
    """A configured but absent fallback table should stop before any mapping."""
    marker_lookup = tmp_path / "markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "stats.h5"
    precomputed_stats.write_bytes(b"stats")
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[
            MapMyCellsSampleConfig(
                sample_id="PAIR1_MERSCOPE",
                platform="MERSCOPE",
                anndata_path=tmp_path / "unused.h5ad",
            )
        ],
        reference_mode="whole_brain",
        marker_lookup_path=marker_lookup,
        precomputed_stats_path=precomputed_stats,
        gene_id_fallback_csv=tmp_path / "missing_gene.csv",
    )

    with pytest.raises(FileNotFoundError, match="gene-ID fallback table"):
        run_mapmycells(cfg)


def test_prepare_query_does_not_apply_fallback_to_non_ensembl_id_column(
    tmp_path: Path,
    whb_gene_csv: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A symbol-valued ID column must not be partly rewritten to Ensembl IDs."""
    symbols = ["APP", "H2AX", "NOTX"]
    input_h5ad = _write_clustered(
        tmp_path / "symbols.h5ad",
        pd.DataFrame({"feature_name": symbols}, index=symbols),
    )
    report_path = tmp_path / "query_gene_ids.json"

    with caplog.at_level("INFO", logger="merxen.analysis.mapmycells"):
        output_h5ad = prepare_mapmycells_query(
            input_h5ad,
            tmp_path / "query.h5ad",
            gene_id_column="feature_name",
            gene_id_fallback=load_gene_id_fallback_table(whb_gene_csv),
            gene_id_report_path=report_path,
        )

    assert list(ad.read_h5ad(output_h5ad).var_names) == symbols
    report = json.loads(report_path.read_text())
    assert report["gene_id_fallback_applied"] is False
    assert report["ids_used_as_provided"] is True
    assert "resolved_by_fallback" not in report
    assert "no Ensembl ID" not in caplog.text
    assert "is not 'ensembl_id'" in caplog.text


def test_prepare_query_reports_symbols_left_for_gene_mapping_db(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With a gene-mapping DB, unresolved symbols are handed on, not ignored."""
    input_h5ad = _write_clustered(tmp_path / "merscope.h5ad", _merscope_var())
    report_path = tmp_path / "query_gene_ids.json"

    with caplog.at_level("WARNING", logger="merxen.analysis.mapmycells"):
        output_h5ad = prepare_mapmycells_query(
            input_h5ad,
            tmp_path / "query.h5ad",
            gene_id_column="ensembl_id",
            allow_gene_symbol_fallback=True,
            gene_id_report_path=report_path,
        )

    assert list(ad.read_h5ad(output_h5ad).var_names[2:]) == list(MERSCOPE_MISSING_IDS)
    report = json.loads(report_path.read_text())
    assert report["unresolved_handling"] == "left_as_symbols_for_gene_mapping_db"
    assert report["unresolved"] == dict.fromkeys(
        MERSCOPE_MISSING_IDS, "left_as_symbol_for_gene_mapping_db"
    )
    assert "left as symbols for the gene-mapping database" in caplog.text
    assert "ignored by MapMyCells" not in caplog.text


def test_run_mapmycells_ignores_fallback_table_without_query_species_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human-only table configured for a mouse run is dropped and recorded."""
    human_gene_csv = tmp_path / "human_gene.csv"
    pd.DataFrame(
        {
            "gene_identifier": list(MERSCOPE_MISSING_IDS.values()),
            "gene_symbol": list(MERSCOPE_MISSING_IDS),
        }
    ).to_csv(human_gene_csv, index=False)
    mouse_ids = ["ENSMUSG00000034394", "ENSMUSG00000020932"]
    mouse_h5ad = _write_clustered(
        tmp_path / "PAIR1_MERSCOPE_clustered.h5ad",
        pd.DataFrame(
            {"gene": ["Lif", "Gfap"], "ensembl_id": mouse_ids}, index=mouse_ids
        ),
    )
    marker_lookup = tmp_path / "markers.json"
    marker_lookup.write_text("{}\n")
    precomputed_stats = tmp_path / "stats.h5"
    precomputed_stats.write_bytes(b"stats")
    monkeypatch.setattr(
        "merxen.analysis.mapmycells._run_command", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "merxen.analysis.mapmycells.annotate_h5ad_with_mapmycells",
        lambda *args, **kwargs: None,
    )
    cfg = MapMyCellsConfig(
        pair_id="PAIR1",
        output_dir=tmp_path / "mapmycells_out",
        samples=[
            MapMyCellsSampleConfig(
                sample_id="PAIR1_MERSCOPE",
                platform="MERSCOPE",
                anndata_path=mouse_h5ad,
                gene_id_column="ensembl_id",
            )
        ],
        reference_mode="whole_brain",
        reference_atlas="wmb",
        query_species="mouse",
        marker_lookup_path=marker_lookup,
        precomputed_stats_path=precomputed_stats,
        region_cache_dir=tmp_path / "empty_cache",
        gene_id_fallback_csv=human_gene_csv,
    )

    run_mapmycells(cfg)

    manifest = json.loads(
        (cfg.output_dir / "PAIR1_mapmycells_manifest.json").read_text()
    )
    assert manifest["gene_id_fallback_csv"] is None
    assert manifest["gene_id_fallback_ignored"] == {
        "path": str(human_gene_csv),
        "reason": "no mouse Ensembl IDs",
    }
    sample = manifest["gene_id_resolution"]["PAIR1_MERSCOPE"]
    assert sample["gene_id_fallback"] is None
    assert sample["gene_id_fallback_applied"] is False
    assert sample["n_unresolved"] == 0
