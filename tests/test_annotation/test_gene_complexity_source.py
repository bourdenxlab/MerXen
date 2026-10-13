"""The gene-complexity check's simulated source (M13 chunk C16; decision D19 (a)).

A version-7 self-map stores the simulated ``n_genes`` per (member, test
cell, depth), counted on the bundle's query genes (the test cells' genes);
RESOLVE counts native ``n_genes`` on the same genes and builds the
``gene_complexity_check`` inputs from both. Version-6 bundles (the seeded
families) and version-7 bundles built before the artefact store none, so the
check is ``not_evaluable`` for them; the version-6 path and its build hashes
are unchanged (``test_resolvability_v6_golden``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import real_qc as qc
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationConfig,
    AnnotationRealQcConfig,
    AnnotationResolvabilityConfig,
)
from merxen.annotation.store import compute_build_hash

from .test_resolvability import (
    bootstrap_mapper,
    make_test_cells,
    recipes,
    synthetic_specs,
)

GRID = (10, 30, 100)


def _members() -> list[res.EnsembleMember]:
    """Two emission members and clean (enough to tell the roles apart)."""
    every = res.ensemble_members(
        AnnotationResolvabilityConfig(), species="human", chemistry="unknown"
    )
    emission = [member for member in every if member.role == "emission"][:2]
    return [*emission, *(member for member in every if member.role != "emission")]


@pytest.fixture(scope="module")
def v7_run() -> res.ResolvabilityResultV7:
    return res.run_resolvability_v7(
        make_test_cells(30),
        specs=synthetic_specs(),
        depths=GRID,
        members=_members(),
        map_fn=bootstrap_mapper,
        settings=res.RuleSettings(),
        ensemble=res.EnsembleSettings(),
        species="human",
        neuronal={"A": True, "B": False},
    )


@pytest.fixture(scope="module")
def v7_dir(
    v7_run: res.ResolvabilityResultV7, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    directory = tmp_path_factory.mktemp("v7_bundle")
    v7_run.write(directory)
    return directory


# --------------------------------------------------------------------------
# PREP: what a version-7 self-map stores


def test_simulated_gene_counts_counts_detected_query_genes_per_simulated_cell() -> None:
    # Row b holds an explicit zero entry, which is not a detected gene.
    counts = sparse.csr_matrix(
        (
            np.array([3.0, 1.0, 0.0, 1.0, 2.0, 5.0]),
            np.array([0, 2, 2, 0, 1, 3]),
            np.array([0, 2, 3, 6]),
        ),
        shape=(3, 4),
    )
    assert counts.nnz == 6
    query = res.SimulatedQuery(
        recipe=recipes()[0],
        counts=counts,
        genes=["G0", "G1", "G2", "G3"],
        obs=pd.DataFrame(
            {"cell_id": ["a", "b", "c"], "depth": [10, 10, 30]},
            index=["a|D10", "b|D10", "c|D30"],
        ),
        n_by_depth={10: 2, 30: 1},
    )
    table = res.simulated_gene_counts(query)
    assert table["cell_id"].tolist() == ["a", "b", "c"]
    assert table["depth"].tolist() == [10, 10, 30]
    assert table["n_genes"].tolist() == [2, 0, 3]
    assert table["total_counts"].tolist() == [4, 0, 8]


def test_every_member_stores_its_simulated_cells_with_their_role(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    table = v7_run.sim_genes
    assert table is not None
    assert list(table.columns) == list(res.SIM_GENES_COLUMNS)
    summary = v7_run.summary
    per_member = table.groupby(res.MEMBER_COLUMN, observed=True).size().to_dict()
    assert per_member == summary["n_simulated_cells"]
    roles = {
        str(name): str(role)
        for name, role in table[[res.MEMBER_COLUMN, res.MEMBER_ROLE_COLUMN]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    assert roles == {item["member"]: item["role"] for item in summary["members"]}
    # One row per (member, test cell, depth): the cells table's sim ids.
    keys = table[[res.MEMBER_COLUMN, "cell_id", "depth"]].astype(str)
    assert not keys.duplicated().any()
    cells = v7_run.cells[v7_run.cells["seed"] == 0]
    first_level = str(cells["level"].iloc[0])
    expected = cells[cells["level"] == first_level]
    assert len(expected) == len(table)
    np.testing.assert_allclose(
        np.sort(expected["total_counts"].to_numpy(np.float64)),
        np.sort(table["total_counts"].to_numpy(np.float64)),
    )


def test_stored_n_genes_are_the_detected_genes_of_the_member_draw(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    # The draws are keyed, so re-simulating a member reproduces its cells.
    member = _members()[1]
    test = make_test_cells(30)
    query = res.thin_and_contaminate_v7(test, GRID, member.recipe)
    detected = np.asarray((query.counts > 0).sum(axis=1)).ravel()
    stored = v7_run.sim_genes
    assert stored is not None
    rows = stored[stored[res.MEMBER_COLUMN].astype(str) == member.name]
    by_id = dict(
        zip(
            rows["cell_id"].astype(str) + "|D" + rows["depth"].astype(str),
            rows["n_genes"].astype(int),
            strict=True,
        )
    )
    assert [by_id[sim_id] for sim_id in query.obs.index] == detected.tolist()
    # The spill partner's genes count: host + spill, never the host alone.
    assert member.recipe.spill_fraction > 0


def test_the_summary_records_the_counted_genes_and_the_file(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    record = v7_run.summary[res.SIM_GENES_RECORD]
    genes = make_test_cells(30).genes
    assert record["version"] == res.SIM_GENES_VERSION
    assert record["file"] == res.SIM_GENES_FILE
    assert record["query_genes"] == list(genes)
    assert record["n_query_genes"] == len(genes)
    assert record["query_genes_sha256"] == res.genes_sha256(genes)
    assert record["n_rows"] == len(v7_run.sim_genes)
    output = v7_run.bundle_output()
    assert output["files"]["sim_genes"] == res.SIM_GENES_FILE
    assert "query_genes" not in output[res.SIM_GENES_RECORD]
    assert output[res.SIM_GENES_RECORD]["query_genes_sha256"] == res.genes_sha256(genes)


def test_the_artefact_round_trips_through_the_bundle(
    v7_run: res.ResolvabilityResultV7, v7_dir: Path
) -> None:
    assert (v7_dir / res.SIM_GENES_FILE).is_file()
    loaded = res.load_simulated_genes(v7_dir)
    assert loaded is not None
    assert loaded.version == res.SIM_GENES_VERSION
    assert loaded.query_genes == tuple(make_test_cells(30).genes)
    assert loaded.depth_grid == GRID
    assert loaded.emission_members == tuple(v7_run.summary["emission_members"])
    stored = v7_run.sim_genes
    assert stored is not None
    pd.testing.assert_frame_equal(
        loaded.table.reset_index(drop=True),
        stored.astype(
            {
                res.MEMBER_COLUMN: str,
                res.MEMBER_ROLE_COLUMN: str,
                "cell_id": str,
            }
        ).reset_index(drop=True),
        check_dtype=False,
    )


def test_per_cell_values_average_the_emission_members_only() -> None:
    table = pd.DataFrame(
        {
            res.MEMBER_COLUMN: ["R1@0", "R1@6", "clean@0", "R1@0", "R1@6"],
            res.MEMBER_ROLE_COLUMN: [
                "emission",
                "emission",
                "reported",
                "emission",
                "emission",
            ],
            "cell_id": ["a", "a", "a", "a", "a"],
            "depth": [10, 10, 10, 30, 30],
            "total_counts": [10, 10, 10, 30, 31],
            "n_genes": [8, 6, 10, 20, 23],
        }
    )
    simulated = res.SimulatedGenes(
        table=table,
        query_genes=("G0",),
        depth_grid=(10, 30),
        emission_members=("R1@0", "R1@6"),
        version=res.SIM_GENES_VERSION,
    )
    per_cell = simulated.per_cell()
    assert per_cell["depth"].tolist() == [10, 30]
    assert per_cell["n_genes"].tolist() == [7.0, 21.5]
    assert per_cell["n_members"].tolist() == [2, 2]
    clean = simulated.per_cell(members=["clean@0"])
    assert clean["n_genes"].tolist() == [10.0]


# --------------------------------------------------------------------------
# Bundles without the artefact: version 6 (seeded families) and version 7
# built before it


def test_a_version_6_bundle_stores_none_and_the_check_is_not_evaluable(
    tmp_path: Path,
) -> None:
    run = res.run_resolvability(
        make_test_cells(30),
        specs=synthetic_specs(),
        depths=GRID,
        recipes=recipes(),
        map_fn=bootstrap_mapper,
        settings=res.RuleSettings(),
        species="human",
    )
    files = run.write(tmp_path)
    assert "sim_genes" not in files
    assert not (tmp_path / res.SIM_GENES_FILE).exists()
    assert res.SIM_GENES_RECORD not in run.summary
    assert res.load_simulated_genes(tmp_path) is None
    signal = qc.gene_complexity_signal(
        res.load_simulated_genes(tmp_path),
        native_counts=np.ones((3, 2)),
        native_gene_ids=["G00", "G01"],
    )
    assert signal is None
    outcome = _complexity_outcome(qc.real_data_qc(_signals(signal), None, _config()))
    assert outcome.state == "not_evaluable"
    assert "D19" in str(outcome.reason)


def _strip_artefact(source: Path, target: Path) -> None:
    """Copy a version-7 bundle as a pre-artefact build wrote it."""
    target.mkdir()
    for path in source.iterdir():
        if path.name != res.SIM_GENES_FILE:
            (target / path.name).write_bytes(path.read_bytes())
    summary_path = target / res.RESOLVABILITY_SUMMARY_FILE
    summary = json.loads(summary_path.read_text())
    summary.pop(res.SIM_GENES_RECORD)
    summary_path.write_text(json.dumps(summary))


def test_a_version_7_bundle_built_before_the_artefact_is_not_evaluable(
    v7_dir: Path, tmp_path: Path
) -> None:
    old = tmp_path / "old"
    _strip_artefact(v7_dir, old)
    assert res.load_simulated_genes(old) is None
    # The rest of the bundle is still read as before.
    tables = res.load_resolvability(old, allow_version_7=True)
    assert tables is not None and tables.version == 7
    assert qc.gene_complexity_signal(None, np.ones((1, 1)), ["G00"]) is None


def test_a_directory_without_resolvability_outputs_has_no_artefact(
    tmp_path: Path,
) -> None:
    assert res.load_simulated_genes(tmp_path) is None


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("drop_file", "missing"),
        ("version", "version"),
        ("genes", "sha256"),
        ("columns", "columns"),
        # C16 review 5: the table must be the one the summary counts.
        ("truncated", "rows but the summary records"),
        ("member_rows", "rows per member"),
        ("emission_member_absent", "no rows of the emission members"),
        ("off_grid", "off the grid"),
        ("repeated_cell", "more than once"),
        ("n_genes_range", "outside"),
    ],
)
def test_a_damaged_artefact_is_refused(
    v7_dir: Path, tmp_path: Path, change: str, match: str
) -> None:
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    for path in v7_dir.iterdir():
        (damaged / path.name).write_bytes(path.read_bytes())
    summary_path = damaged / res.RESOLVABILITY_SUMMARY_FILE
    summary = json.loads(summary_path.read_text())
    record = summary[res.SIM_GENES_RECORD]
    table_path = damaged / res.SIM_GENES_FILE
    table = pd.read_parquet(table_path)
    table = table.astype({res.MEMBER_COLUMN: str, res.MEMBER_ROLE_COLUMN: str})
    table["cell_id"] = table["cell_id"].astype(str)
    first = summary["emission_members"][0]
    second = summary["emission_members"][1]
    if change == "drop_file":
        table_path.unlink()
    elif change == "version":
        record["version"] = res.SIM_GENES_VERSION + 1
    elif change == "genes":
        record["query_genes"][0] = "OTHER"
    elif change == "columns":
        table.drop(columns=["n_genes"]).to_parquet(table_path)
    elif change == "truncated":
        # A file cut short and rewritten: readable, with fewer rows.
        table.iloc[:-3].to_parquet(table_path)
    elif change == "member_rows":
        # Same row count, one row moved to another member.
        row = table.index[table[res.MEMBER_COLUMN] == first][0]
        table.loc[row, res.MEMBER_COLUMN] = second
        table.loc[row, "cell_id"] = "moved"
        table.to_parquet(table_path)
    elif change == "emission_member_absent":
        # Table and record agree, but an emission member was never stored.
        kept = table[table[res.MEMBER_COLUMN] != first]
        kept.to_parquet(table_path)
        record["n_rows"] = len(kept)
        record["n_rows_per_member"].pop(first)
    elif change == "off_grid":
        table.loc[table.index[0], "depth"] = GRID[0] + 1
        table.to_parquet(table_path)
    elif change == "repeated_cell":
        # Same rows per member, one simulated cell twice.
        rows = table.index[table[res.MEMBER_COLUMN] == first]
        table.loc[rows[1], ["cell_id", "depth"]] = table.loc[
            rows[0], ["cell_id", "depth"]
        ].to_numpy()
        table.to_parquet(table_path)
    else:
        table.loc[table.index[0], "n_genes"] = len(record["query_genes"]) + 1
        table.to_parquet(table_path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(res.ResolvabilityError, match=match):
        res.load_simulated_genes(damaged)


# --------------------------------------------------------------------------
# The build hash: only version-7 bundles change


def _payload(version: int | None = None) -> dict[str, Any]:
    payload = res.v7_simulation_payload(
        members=_members(),
        assets=[],
        chemistry="unknown",
        depth_grid=GRID,
    )
    if version is not None:
        payload["simulated_n_genes_version"] = version
    return payload


def test_the_artefact_version_enters_the_version_7_build_hash() -> None:
    payload = _payload()
    assert payload["simulated_n_genes_version"] == res.SIM_GENES_VERSION
    before = _payload()
    del before["simulated_n_genes_version"]
    assert compute_build_hash(payload) != compute_build_hash(before)  # type: ignore[arg-type]
    assert compute_build_hash(payload) != compute_build_hash(  # type: ignore[arg-type]
        _payload(res.SIM_GENES_VERSION + 1)
    )
    # A version-6 self-map's recipes hold no version-7 input.
    assert all(
        "simulated_n_genes_version" not in recipe.to_json() for recipe in recipes()
    )


# --------------------------------------------------------------------------
# RESOLVE: native n_genes on the same genes, and the check's inputs


def test_native_genes_are_counted_on_the_query_genes_only() -> None:
    counts = np.array(
        [
            [2.0, 0.0, 5.0, 1.0],
            [0.0, 0.0, 0.0, 7.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    gene_ids = ["G0", "G1", "X9", "G2"]
    query_genes = ["G0", "G1", "G2", "G3"]
    for matrix in (counts, sparse.csr_matrix(counts)):
        native = qc.native_gene_complexity(matrix, gene_ids, query_genes)
        # X9 is not a query gene; G3 is a query gene the dataset lacks.
        assert native.n_genes.tolist() == [2, 1, 2]
        assert native.totals.tolist() == [3.0, 7.0, 2.0]
        assert native.n_query_genes == 4
        assert native.missing_genes == ("G3",)


def test_native_counting_refuses_mismatched_genes() -> None:
    with pytest.raises(ValueError, match="columns"):
        qc.native_gene_complexity(np.ones((2, 3)), ["G0", "G1"], ["G0"])
    with pytest.raises(ValueError, match="query genes hold duplicates"):
        qc.native_gene_complexity(np.ones((2, 2)), ["G0", "G1"], ["G0", "G0"])


def test_native_counting_takes_loaded_feature_ids_as_the_query_sums_them() -> None:
    """C16 review 3: LoadedSample.feature_ids hold "" and summed aliases.

    Unresolved features ("") and features of no query gene are ignored;
    two features resolving to one query gene (H2AX and H2AFX) are one gene,
    detected when their sum is > 0, as build_sample_query sums them.
    """
    from merxen.annotation.gene_ids import summing_matrix

    counts = np.array(
        [
            # G0, unresolved, H2AX -> GA, unresolved, H2AFX -> GA, G1, X9
            [1.0, 4.0, 2.0, 0.0, 3.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 5.0, 1.0, 2.0, 0.0],
            [0.0, 9.0, 0.0, 9.0, 0.0, 0.0, 9.0],
        ]
    )
    feature_ids = ["G0", "", "GA", "", "GA", "G1", "X9"]
    query_genes = ["G0", "GA", "G1", "G3"]
    present = ["G0", "GA", "G1"]
    summed = counts @ summing_matrix(feature_ids, present, dtype=np.float64).toarray()
    for matrix in (counts, sparse.csr_matrix(counts)):
        native = qc.native_gene_complexity(matrix, feature_ids, query_genes)
        # GA counts once per cell however many of its aliases hold counts.
        assert native.n_genes.tolist() == [2, 2, 0]
        assert native.totals.tolist() == [6.0, 3.0, 0.0]
        assert native.missing_genes == ("G3",)
        # The same as counting the summed query (SampleQuery) columns.
        on_query = qc.native_gene_complexity(summed, present, query_genes)
        np.testing.assert_array_equal(native.n_genes, on_query.n_genes)
        np.testing.assert_array_equal(native.totals, on_query.totals)
        assert native.missing_genes == on_query.missing_genes


def test_native_counting_without_any_query_gene_counts_nothing() -> None:
    counts = np.ones((2, 2))
    for matrix in (counts, sparse.csr_matrix(counts)):
        native = qc.native_gene_complexity(matrix, ["", "X9"], ["G0"])
        assert native.n_genes.tolist() == [0, 0]
        assert native.totals.tolist() == [0.0, 0.0]
        assert native.missing_genes == ("G0",)


def test_the_signal_takes_emission_members_at_their_grid_depth(
    v7_dir: Path,
) -> None:
    simulated = res.load_simulated_genes(v7_dir)
    assert simulated is not None
    test = make_test_cells(30)
    signal = qc.gene_complexity_signal(
        simulated, native_counts=test.counts, native_gene_ids=test.genes
    )
    assert signal is not None
    per_cell = simulated.per_cell()
    np.testing.assert_array_equal(signal.simulated_n_genes, per_cell["n_genes"])
    # Simulated cells sit in their grid bin D, never in the bin of the
    # realised total (which may fall just below D).
    assert set(np.asarray(signal.simulated_depth).tolist()) <= set(GRID)
    assert list(signal.grid) == list(GRID)
    expected = qc.native_gene_complexity(test.counts, test.genes, test.genes)
    np.testing.assert_array_equal(signal.native_n_genes, expected.n_genes)
    np.testing.assert_array_equal(signal.native_totals, expected.totals)
    source = signal.source
    assert source["artefact_version"] == res.SIM_GENES_VERSION
    assert source["members"] == list(simulated.emission_members)
    assert source["n_query_genes"] == len(test.genes)
    assert source["n_query_genes_missing"] == 0


def _config(**real_qc: Any) -> AnnotationConfig:
    return AnnotationConfig(species="human", real_qc=AnnotationRealQcConfig(**real_qc))


def _signals(gene_complexity: qc.GeneComplexitySignal | None) -> qc.RealQcSignals:
    return qc.RealQcSignals(
        resolvability_version=7,
        paired=False,
        prefilter_applied=False,
        has_r3_member=False,
        gene_complexity=gene_complexity,
    )


def _complexity_outcome(result: qc.RealQcResult) -> qc.QcOutcome:
    (outcome,) = [
        outcome
        for outcome in result.outcomes
        if outcome.check == qc.GENE_COMPLEXITY_CHECK
    ]
    return outcome


def _bundle_like(simulated_n_genes: int) -> res.SimulatedGenes:
    """Sixty test cells simulated at D = 30 and 100 with a fixed n_genes.

    NR7 interpolates a test cell's genes between the native bin's edges
    (30 and 100) at the bin's native median total (C3 (b)).
    """
    table = pd.DataFrame(
        {
            res.MEMBER_COLUMN: "R1_contam_HO@0",
            res.MEMBER_ROLE_COLUMN: "emission",
            "cell_id": [f"c{index}" for index in range(60)] * 2,
            "depth": np.repeat([30, 100], 60),
            "total_counts": np.repeat([30, 100], 60),
            "n_genes": simulated_n_genes,
        }
    )
    return res.SimulatedGenes(
        table=table,
        query_genes=tuple(f"G{index}" for index in range(40)),
        depth_grid=(10, 30, 100),
        emission_members=("R1_contam_HO@0",),
        version=res.SIM_GENES_VERSION,
    )


def _native(n_cells: int, n_detected: int) -> np.ndarray:
    """Native cells with 30 counts over ``n_detected`` of 40 query genes."""
    counts = np.zeros((n_cells, 40))
    per_gene = np.full(n_detected, 30 // n_detected)
    per_gene[: 30 - per_gene.sum()] += 1
    counts[:, :n_detected] = per_gene
    return counts


def test_the_check_warns_from_the_stored_source_at_the_configured_gap() -> None:
    simulated = _bundle_like(simulated_n_genes=10)
    genes = list(simulated.query_genes)
    # Native cells carry 15 genes against 10 simulated: a 50% gap.
    signal = qc.gene_complexity_signal(simulated, _native(60, 15), genes)
    assert signal is not None
    warned = _complexity_outcome(qc.real_data_qc(_signals(signal), None, _config()))
    assert warned.state == "evaluated"
    assert warned.fired
    assert warned.details["max_gap"] == pytest.approx(0.5)
    assert warned.details["source"]["n_query_genes"] == 40
    # config.real_qc.genes_per_count_gap_warn decides the warning.
    quiet = _complexity_outcome(
        qc.real_data_qc(_signals(signal), None, _config(genes_per_count_gap_warn=0.6))
    )
    assert quiet.state == "evaluated"
    assert not quiet.fired
    assert quiet.details["gap_warn"] == 0.6
    # Fewer than 50 native cells in the bin: not evaluable.
    thin = qc.gene_complexity_signal(simulated, _native(40, 15), genes)
    assert thin is not None
    thin_outcome = _complexity_outcome(qc.real_data_qc(_signals(thin), None, _config()))
    assert thin_outcome.state == "not_evaluable"
    assert thin_outcome.details["source"]["artefact_version"] == res.SIM_GENES_VERSION
