"""Tests for the analysis gene sets (Atera plan AT-8 §1b, §1f)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import sparse

from merxen.analysis_gene_set import (
    ANALYSIS_GENE_SET_CSV,
    ANALYSIS_GENE_SET_JSON,
    ANALYSIS_GENE_SET_MISSING_CSV,
    AnalysisGeneSet,
    GeneSetError,
    GeneSetRow,
    Recipe,
    build_analysis_gene_set,
    default_recipe,
    join_on_gene_ids,
    read_source_panel,
    write_analysis_gene_set,
)
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mapmycells_engine import write_query_h5ad
from merxen.annotation.panel import (
    PANEL_GENES_FILE,
    compute_panel_hash,
    load_annotation_panel,
    panel_from_gene_list,
)
from merxen.annotation.pipeline import MapSample, build_sample_query, load_samples
from merxen.annotation.public_panels import PUBLIC_PANEL_LISTS
from merxen.annotation.sim_inputs import prime_panel_lists
from merxen.cli import main
from merxen.gene_sets import WHOLE_TRANSCRIPTOME_MIN_GENES

N_REGULAR = 70
REGULAR_IDS = [f"ENSG{100 + index:011d}" for index in range(N_REGULAR)]
REGULAR_SYMBOLS = [f"GENE{index}" for index in range(N_REGULAR)]
# Atera has 17 genes whose symbol is their own Ensembl ID.
ID_SYMBOL_IDS = ["ENSG00000170846", "ENSG00000187186", "ENSG00000188897"]
# A trap: an ID-shaped symbol that names another gene's ID.
TRAP_SYMBOL = "ENSG00000000777"
TRAP_ID = "ENSG00000000888"
PANEL_IDS = sorted([*REGULAR_IDS, *ID_SYMBOL_IDS, TRAP_ID])
N_CELLS = 12


def _atera_var(order: list[int] | None = None) -> pd.DataFrame:
    names = [*REGULAR_SYMBOLS, *ID_SYMBOL_IDS, TRAP_SYMBOL]
    ids = [*REGULAR_IDS, *ID_SYMBOL_IDS, TRAP_ID]
    var = pd.DataFrame(
        {
            "gene_ids": ids,
            "feature_types": pd.Categorical(["Gene Expression"] * len(ids)),
            "genome": pd.Categorical(["hg38"] * len(ids)),
        },
        index=pd.Index(names, dtype=str),
    )
    return var if order is None else var.iloc[order]


def _counts(n_genes: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(7)
    return sparse.csr_matrix(rng.integers(0, 4, size=(N_CELLS, n_genes)))


def _atera_adata(order: list[int] | None = None) -> ad.AnnData:
    var = _atera_var(order)
    obs = pd.DataFrame(index=[f"cell{index}" for index in range(N_CELLS)])
    obs["instance_id"] = np.arange(N_CELLS, dtype=np.int64) + 1
    return ad.AnnData(X=_counts(len(var)), obs=obs, var=var)


def _write_h5ad(path: Path, order: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atera_adata(order).write_h5ad(path)
    return path


def _write_store(root: Path) -> Path:
    """A SpatialData-like store holding one table, ``tables/table_original``."""
    (root / "tables").mkdir(parents=True, exist_ok=True)
    # Zarr v3, as the SpatialData stores of the pipeline.
    with ad.settings.override(zarr_write_format=3):
        _atera_adata().write_zarr(root / "tables" / "table_original")
    return root


def _all_genes(source_path: Path, **kwargs: object) -> AnalysisGeneSet:
    source = read_source_panel(source_path, species="human", platform="ATERA")
    return build_analysis_gene_set(
        source,
        "all_panel_genes",
        species="human",
        **kwargs,  # type: ignore[arg-type]
    )


def _public_dir(root: Path, rows: list[tuple[str, str]]) -> Path:
    """An annotation-panel-fetch directory with a fake v1 brain list."""
    item = PUBLIC_PANEL_LISTS["xenium_human_brain_v1"]
    filler = [
        (f"FILL{index}", f"ENSG{900_000 + index:011d}")
        for index in range(item.n_genes - len(rows))
    ]
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([*rows, *filler], columns=["gene_symbol", "gene_id"]).to_csv(
        root / f"{item.key}.gene_list.csv", index=False
    )
    return root


# --------------------------------------------------------------------------
# Recipes


def test_recipe_parsing() -> None:
    assert Recipe.parse("all_panel_genes") == Recipe("all_panel_genes")
    parsed = Recipe.parse("intersect_public_panel:xenium_prime_5k_human")
    assert parsed.argument == "xenium_prime_5k_human"
    assert parsed.default_name == "intersect_public_panel__xenium_prime_5k_human"
    explicit = Recipe.parse("explicit:/some/where/genes.csv")
    # The record keeps the file name only; the full path is an input.
    assert explicit.label == "explicit:genes.csv"
    for bad in ("all_panel_genes:x", "explicit:", "hvg", "intersect_public_panel"):
        with pytest.raises(GeneSetError):
            Recipe.parse(bad)


def test_every_platform_defaults_to_all_panel_genes() -> None:
    # ATERA: user decision of 2026-11-07 (RCA uses all 18,049 panel genes).
    for platform in ("ATERA", "XENIUM", "MERSCOPE", None):
        assert default_recipe(platform) == "all_panel_genes"


# --------------------------------------------------------------------------
# all_panel_genes


def test_all_panel_genes_is_the_identity_on_store_and_h5ad(tmp_path: Path) -> None:
    from_store = _all_genes(_write_store(tmp_path / "store.zarr"))
    from_h5ad = _all_genes(_write_h5ad(tmp_path / "table.h5ad"))

    for gene_set in (from_store, from_h5ad):
        assert gene_set.gene_ids == PANEL_IDS
        assert gene_set.n_requested is None
        assert gene_set.missing_genes == []
        assert gene_set.n_excluded_panel_genes == 0
        assert gene_set.gene_set_hash == gene_set.source_panel.panel_hash
        assert gene_set.whole_transcriptome is False
        assert gene_set.whole_transcriptome_min_genes == WHOLE_TRANSCRIPTOME_MIN_GENES
        assert gene_set.source_panel.id_sources == {"native": len(PANEL_IDS)}
        symbols = dict(zip(gene_set.gene_ids, gene_set.gene_symbols, strict=True))
        assert symbols[TRAP_ID] == TRAP_SYMBOL
        assert symbols[ID_SYMBOL_IDS[0]] == ID_SYMBOL_IDS[0]
    assert from_store.inputs[0].kind == "spatialdata_table"
    assert from_store.inputs[0].key == "table_original"
    assert from_h5ad.inputs[0].kind == "h5ad_var"
    assert from_store.gene_set_hash == from_h5ad.gene_set_hash
    assert (
        from_store.source_panel.declared_features_sha256
        == from_h5ad.source_panel.declared_features_sha256
    )


def test_the_hash_depends_on_the_gene_ids_only(tmp_path: Path) -> None:
    order = list(range(len(PANEL_IDS)))[::-1]
    first = _all_genes(_write_h5ad(tmp_path / "a.h5ad"))
    shuffled = _all_genes(_write_h5ad(tmp_path / "b.h5ad", order))
    renamed = build_analysis_gene_set(
        read_source_panel(_write_h5ad(tmp_path / "a.h5ad"), species="human"),
        "all_panel_genes",
        species="human",
        name="other_name",
        version=3,
    )

    expected = hashlib.sha256("\n".join(PANEL_IDS).encode("utf-8")).hexdigest()
    assert first.gene_set_hash == expected == compute_panel_hash(PANEL_IDS)
    assert shuffled.gene_set_hash == expected
    assert renamed.gene_set_hash == expected
    # Order-independent: the CSV and the declared-feature hash do not move.
    assert shuffled.csv_text() == first.csv_text()
    assert (
        shuffled.source_panel.declared_features_sha256
        == first.source_panel.declared_features_sha256
    )
    # A pinned value guards the formula itself.
    assert (
        compute_panel_hash(["ENSG00000000005", "ENSG00000000003", "ENSG00000000005"])
        == "12538bfc48bc7f74ddf0198390e4bd06400f9a0417c6ab3688b31495901f7781"
    )


def test_a_refused_panel_raises(tmp_path: Path) -> None:
    path = tmp_path / "symbols.csv"
    # Symbols stored in the ID column: the resolver refuses the panel.
    pd.DataFrame({"gene_ids": REGULAR_SYMBOLS}).to_csv(path, index=False)
    with pytest.raises(GeneSetError, match="refused"):
        read_source_panel(path, species="human")


# --------------------------------------------------------------------------
# intersect_public_panel


def test_intersect_joins_on_gene_ids_never_on_symbols(tmp_path: Path) -> None:
    drift_id = "ENSG00000000555"
    rows = [
        *zip(REGULAR_SYMBOLS[:60], REGULAR_IDS[:60], strict=True),
        (ID_SYMBOL_IDS[0], ID_SYMBOL_IDS[0]),
        # The trap's symbol is an ID the panel does not have: not selected,
        # although a panel gene carries that text as its symbol.
        ("SOMEGENE", TRAP_SYMBOL),
        # GENE65 under another ID (release drift): missing, reported only.
        (REGULAR_SYMBOLS[65], drift_id),
    ]
    public = _public_dir(tmp_path / "panels", rows)
    source = read_source_panel(_write_h5ad(tmp_path / "t.h5ad"), species="human")

    gene_set = build_analysis_gene_set(
        source,
        "intersect_public_panel:xenium_human_brain_v1",
        species="human",
        public_panel_dir=public,
    )

    assert gene_set.gene_ids == sorted([*REGULAR_IDS[:60], ID_SYMBOL_IDS[0]])
    assert TRAP_ID not in gene_set.gene_ids
    assert gene_set.n_requested == 266
    missing = {gene.gene_id for gene in gene_set.missing_genes}
    assert len(missing) == 266 - 61
    assert {TRAP_SYMBOL, drift_id} <= missing
    assert gene_set.missing_symbol_matches == {drift_id: REGULAR_IDS[65]}
    assert gene_set.n_excluded_panel_genes == len(PANEL_IDS) - 61
    assert gene_set.inputs[1].kind == "public_panel_gene_list"
    assert gene_set.inputs[1].key == "xenium_human_brain_v1"


def test_join_ignores_symbols_entirely() -> None:
    panel = [
        GeneSetRow(gene_id="ENSG00000000002", gene_symbol="ENSG00000000001"),
        GeneSetRow(gene_id="ENSG00000000003", gene_symbol="ABC"),
    ]
    requested = [
        GeneSetRow(gene_id="ENSG00000000001", gene_symbol=""),
        GeneSetRow(gene_id="ENSG00000000009", gene_symbol="ABC"),
    ]
    join = join_on_gene_ids(panel, requested)
    assert join.genes == ()
    assert [gene.gene_id for gene in join.missing] == [
        "ENSG00000000001",
        "ENSG00000000009",
    ]
    assert join.symbol_matches == {"ENSG00000000009": "ENSG00000000003"}


def test_intersect_with_the_packaged_prime_list(tmp_path: Path) -> None:
    _species, prime_ids = prime_panel_lists()["xenium_prime_5k_human"]
    chosen = sorted(prime_ids)[:55]
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "gene_symbol": [f"P{index}" for index in range(len(chosen) + 5)],
            "gene_id": [*chosen, *(f"ENSG{990 + i:011d}" for i in range(5))],
        }
    ).to_csv(path, index=False)
    source = read_source_panel(path, species="human")

    gene_set = build_analysis_gene_set(
        source, "intersect_public_panel:xenium_prime_5k_human", species="human"
    )

    assert gene_set.gene_ids == chosen
    assert gene_set.n_requested == 5001
    assert len(gene_set.missing_genes) == 5001 - 55
    assert gene_set.inputs[1].kind == "packaged_asset"
    assert not gene_set.inputs[1].path.startswith("/")
    with pytest.raises(GeneSetError, match="mouse"):
        build_analysis_gene_set(
            source, "intersect_public_panel:xenium_prime_5k_mouse", species="human"
        )
    with pytest.raises(GeneSetError, match="not packaged"):
        build_analysis_gene_set(
            source, "intersect_public_panel:xenium_human_brain_v1", species="human"
        )
    with pytest.raises(GeneSetError, match="unknown public panel"):
        build_analysis_gene_set(source, "intersect_public_panel:nope", species="human")


def test_a_fetched_list_must_match_its_manifest_and_pin(tmp_path: Path) -> None:
    public = _public_dir(tmp_path / "panels", [(REGULAR_SYMBOLS[0], REGULAR_IDS[0])])
    source = read_source_panel(_write_h5ad(tmp_path / "t.h5ad"), species="human")
    recipe = "intersect_public_panel:xenium_human_brain_v1"
    (public / "xenium_human_brain_v1.manifest.json").write_text(
        json.dumps({"gene_list_sha256": "0" * 64})
    )
    with pytest.raises(GeneSetError, match="manifest"):
        build_analysis_gene_set(
            source, recipe, species="human", public_panel_dir=public
        )
    (public / "xenium_human_brain_v1.manifest.json").unlink()
    table = pd.read_csv(public / "xenium_human_brain_v1.gene_list.csv")
    table.iloc[:-1].to_csv(public / "xenium_human_brain_v1.gene_list.csv", index=False)
    with pytest.raises(GeneSetError, match="pinned 266"):
        build_analysis_gene_set(
            source, recipe, species="human", public_panel_dir=public
        )


# --------------------------------------------------------------------------
# explicit


@pytest.mark.parametrize("form", ["csv", "txt", "json_list", "panel_json"])
def test_explicit_lists(tmp_path: Path, form: str) -> None:
    wanted = [*REGULAR_IDS[:5], "ENSG00000000555.4"]
    if form == "csv":
        path = tmp_path / "genes.csv"
        pd.DataFrame({"gene_symbol": ["A"] * 6, "ensembl_id": wanted}).to_csv(
            path, index=False
        )
    elif form == "txt":
        path = tmp_path / "genes.txt"
        path.write_text("# a comment\n" + "\n".join(wanted) + "\n")
    elif form == "json_list":
        path = tmp_path / "genes.json"
        path.write_text(json.dumps(wanted))
    else:
        path = tmp_path / "panel_genes.json"
        path.write_text(json.dumps({"ensembl_ids": wanted, "symbols": ["S"] * 6}))
    source = read_source_panel(_write_h5ad(tmp_path / "t.h5ad"), species="human")

    gene_set = build_analysis_gene_set(source, f"explicit:{path}", species="human")

    assert gene_set.gene_ids == REGULAR_IDS[:5]
    # The version suffix is stripped before the join.
    assert [gene.gene_id for gene in gene_set.missing_genes] == ["ENSG00000000555"]
    assert gene_set.recipe == f"explicit:{path.name}"
    assert gene_set.inputs[1].role == "explicit_list"


def test_explicit_lists_refuse_symbols_and_other_species(tmp_path: Path) -> None:
    source = read_source_panel(_write_h5ad(tmp_path / "t.h5ad"), species="human")
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("GENE1\nGENE2\n")
    with pytest.raises(GeneSetError, match="never on symbols"):
        build_analysis_gene_set(source, f"explicit:{symbols}", species="human")
    no_id_column = tmp_path / "symbols.csv"
    pd.DataFrame({"gene_symbol": ["GENE1"]}).to_csv(no_id_column, index=False)
    with pytest.raises(GeneSetError, match="never join on symbols"):
        build_analysis_gene_set(source, f"explicit:{no_id_column}", species="human")
    mouse = tmp_path / "mouse.txt"
    mouse.write_text("ENSMUSG00000000001\n")
    with pytest.raises(GeneSetError, match="human Ensembl gene IDs"):
        build_analysis_gene_set(source, f"explicit:{mouse}", species="human")
    absent = tmp_path / "absent.txt"
    absent.write_text("ENSG00000000555\n")
    with pytest.raises(GeneSetError, match="selects no gene"):
        build_analysis_gene_set(source, f"explicit:{absent}", species="human")


# --------------------------------------------------------------------------
# Writer and record


def test_writer_round_trip_and_missing_report(tmp_path: Path) -> None:
    explicit = tmp_path / "genes.txt"
    explicit.write_text("\n".join([*REGULAR_IDS[:3], "ENSG00000000555"]) + "\n")
    source = read_source_panel(_write_h5ad(tmp_path / "t.h5ad"), species="human")
    gene_set = build_analysis_gene_set(source, f"explicit:{explicit}", species="human")

    paths = write_analysis_gene_set(gene_set, tmp_path / "out")

    assert paths.record.name == ANALYSIS_GENE_SET_JSON
    assert AnalysisGeneSet.load(paths.record) == gene_set
    csv_text = paths.csv.read_text()
    assert paths.csv.name == ANALYSIS_GENE_SET_CSV
    assert csv_text.splitlines()[0] == "gene_symbol,gene_id"
    assert hashlib.sha256(csv_text.encode()).hexdigest() == gene_set.csv_sha256
    missing = pd.read_csv(tmp_path / "out" / ANALYSIS_GENE_SET_MISSING_CSV, dtype=str)
    assert missing["gene_id"].tolist() == ["ENSG00000000555"]
    # Writing again gives the same bytes.
    first = paths.record.read_bytes()
    write_analysis_gene_set(gene_set, tmp_path / "out")
    assert paths.record.read_bytes() == first


def test_a_tampered_record_does_not_load(tmp_path: Path) -> None:
    gene_set = _all_genes(_write_h5ad(tmp_path / "t.h5ad"))
    paths = write_analysis_gene_set(gene_set, tmp_path / "out")
    payload = json.loads(paths.record.read_text())
    payload["genes"] = payload["genes"][1:]
    payload["n_genes"] -= 1
    paths.record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="gene_set_hash"):
        AnalysisGeneSet.load(paths.record)


def test_an_empty_symbol_is_written_as_the_id() -> None:
    rows = [
        GeneSetRow(gene_id="ENSG00000000001", gene_symbol=""),
        GeneSetRow(gene_id="ENSG00000000002", gene_symbol="NaN"),
    ]
    from merxen.analysis_gene_set import gene_set_csv_text

    assert gene_set_csv_text(rows).splitlines()[1:] == [
        "ENSG00000000001,ENSG00000000001",
        "ENSG00000000002,ENSG00000000002",
    ]


# --------------------------------------------------------------------------
# Hand-off to annotation-panel and the MapMyCells query (§1f)


def _cli_gene_set(tmp_path: Path, source: Path, *extra: str) -> AnalysisGeneSet:
    output = tmp_path / "gene_set"
    result = CliRunner().invoke(
        main,
        [
            "analysis-gene-set",
            "--source",
            str(source),
            "--species",
            "human",
            "--platform",
            "atera",
            "--output-dir",
            str(output),
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "gene_set_hash" in result.output
    return AnalysisGeneSet.load(output / ANALYSIS_GENE_SET_JSON)


@pytest.mark.parametrize("recipe", ["all_panel_genes", "explicit"])
def test_round_trip_into_annotation_panel(tmp_path: Path, recipe: str) -> None:
    store = _write_store(tmp_path / "store.zarr")
    extra: list[str] = []
    if recipe == "explicit":
        listed = tmp_path / "genes.txt"
        listed.write_text("\n".join([*REGULAR_IDS[:58], *ID_SYMBOL_IDS, TRAP_ID]))
        extra = ["--recipe", f"explicit:{listed}"]
    gene_set = _cli_gene_set(tmp_path, store, *extra)
    output = tmp_path / "panel"

    result = CliRunner().invoke(
        main,
        [
            "annotation-panel",
            "--panel-genes-path",
            str(tmp_path / "gene_set" / ANALYSIS_GENE_SET_CSV),
            "--species",
            "human",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    panel = load_annotation_panel(output / PANEL_GENES_FILE)
    assert panel.ensembl_ids == gene_set.gene_ids
    assert panel.panel_hash == gene_set.gene_set_hash
    assert panel.n_genes == gene_set.n_genes
    report = json.loads((output / "panel_report.json").read_text())
    assert report["status"] == "ok"


def test_cli_refuses_a_bad_recipe(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "analysis-gene-set",
            "--source",
            str(_write_h5ad(tmp_path / "t.h5ad")),
            "--species",
            "human",
            "--recipe",
            "hvg",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown recipe" in result.output


@pytest.mark.parametrize("recipe", ["all_panel_genes", "explicit"])
def test_an_atera_query_has_exactly_the_gene_set_columns(
    tmp_path: Path, recipe: str
) -> None:
    h5ad = _write_h5ad(tmp_path / "prepared" / "ATERA_prepared.h5ad")
    source = read_source_panel(h5ad, species="human", platform="ATERA")
    if recipe == "explicit":
        listed = tmp_path / "genes.txt"
        listed.write_text("\n".join([*REGULAR_IDS[10:65], ID_SYMBOL_IDS[1], TRAP_ID]))
        recipe = f"explicit:{listed}"
    gene_set = build_analysis_gene_set(source, recipe, species="human")
    paths = write_analysis_gene_set(gene_set, tmp_path / "gene_set")
    panel_from_gene_list(paths.csv, "human", output_dir=tmp_path / "panel")
    panel = load_annotation_panel(tmp_path / "panel" / PANEL_GENES_FILE)
    config = AnnotationConfig(species="human").coupled_to_clustering(1)

    (loaded,) = load_samples(
        [MapSample("ATERA_PUB_ATERA", "ATERA", h5ad, "prepared")], config, min_counts=1
    )
    query = build_sample_query(loaded, panel)
    written = write_query_h5ad(
        query.counts, query.cell_ids, query.gene_ids, tmp_path / "query.h5ad"
    )

    # The query is the gene set: never trimmed to markers, nothing missing.
    assert query.gene_ids == gene_set.gene_ids
    assert query.missing_gene_ids == []
    assert list(ad.read_h5ad(written).var_names) == gene_set.gene_ids
    # Its columns are the gene set's counts, by gene ID.
    adata = ad.read_h5ad(h5ad)
    columns = [list(adata.var["gene_ids"]).index(i) for i in gene_set.gene_ids]
    np.testing.assert_array_equal(
        query.counts.toarray(), adata.X[:, columns].toarray()[loaded.in_table]
    )
    # total_counts are over every non-control feature of the sample, not the
    # gene set (the depth question handed to AT-11).
    np.testing.assert_array_equal(
        query.total_counts, np.asarray(adata.X.sum(axis=1)).ravel()[loaded.in_table]
    )
