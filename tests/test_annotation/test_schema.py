"""Tests for the per-cell label table contract (merxen.annotation.schema)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.schema import (
    FINAL_LEVELS,
    HUMAN_BRANCHES,
    LABEL_TABLE_VERSION,
    LEVEL_FIELDS,
    LEVELS,
    REPORT_ONLY_LEVELS,
    CellStatus,
    Columns,
    FinalLevel,
    LabelTableError,
    branch_categories,
    coerce_label_table_dtypes,
    column_specs,
    label_table_filename,
    leaf_categories,
    safe_token,
    soft_columns,
    validate_label_table,
)
from merxen.annotation.vocab import UNASSIGNED_LABEL, load_vocab

TableFactory = Callable[[str], pd.DataFrame]


def _table_ids(table: pd.DataFrame) -> list[str]:
    return table.loc[table[Columns.IN_TABLE], Columns.CELL_ID].tolist()


def _set(table: pd.DataFrame, row: int, column: str, value: object) -> None:
    """Set one cell, adding the value to a categorical's categories first."""
    series = table[column]
    if (
        isinstance(series.dtype, pd.CategoricalDtype)
        and value is not None
        and not (isinstance(value, float) and np.isnan(value))
        and value not in series.cat.categories
    ):
        table[column] = series.cat.add_categories([value])
    table.loc[row, column] = value


def _problems(table: pd.DataFrame, species: str, **kwargs: object) -> list[str]:
    with pytest.raises(LabelTableError) as caught:
        validate_label_table(table, species, **kwargs)  # type: ignore[arg-type]
    return caught.value.problems


def test_cell_status_matches_plan_vocabulary() -> None:
    assert [status.value for status in CellStatus] == [
        "confident",
        "low_confidence",
        "below_floor",
        "not_resolvable",
        "method_disagree",
        "single_method",
        "implausible",
        "parent_unresolved",
        "not_attempted_gate",
        "not_applicable",
        "low_counts",
    ]
    assert CellStatus("confident") is CellStatus.CONFIDENT
    assert CellStatus.CONFIDENT == "confident"


def test_final_level_covers_both_species_orders() -> None:
    values = {level.value for level in FinalLevel}
    for species in ("human", "mouse"):
        assert set(FINAL_LEVELS[species]) <= values
        assert FINAL_LEVELS[species][0] == "none"
    assert FINAL_LEVELS["human"][-1] == "supercluster"
    assert FINAL_LEVELS["mouse"][-1] == "subclass"


def test_label_table_version_and_filename() -> None:
    assert LABEL_TABLE_VERSION == 1
    assert label_table_filename("P7513_MERSCOPE") == (
        "P7513_MERSCOPE_celltype_labels.parquet"
    )


def test_columns_level_builds_names_and_rejects_unknown_fields() -> None:
    assert Columns.level("broad", "status") == "ct_broad_status"
    assert [Columns.level("nt", field) for field in LEVEL_FIELDS][0] == "ct_nt_name"
    with pytest.raises(ValueError, match="field must be one of"):
        Columns.level("broad", "probability")


def test_safe_token_removes_slashes_and_spaces() -> None:
    assert safe_token("Astrocytes/Ependymal") == "astrocytes_ependymal"
    assert safe_token("Oligodendrocyte precursors") == "oligodendrocyte_precursors"
    assert safe_token("01 IT-ET Glut") == "01_it_et_glut"
    assert safe_token(" / ") == "value"


def test_column_specs_cover_the_plan_columns() -> None:
    human = column_specs("human")
    mouse = column_specs("mouse")
    for level in LEVELS["human"]:
        for field in LEVEL_FIELDS:
            assert Columns.level(level, field) in human
    for level in LEVELS["mouse"]:
        for field in LEVEL_FIELDS:
            assert Columns.level(level, field) in mouse
    assert Columns.FLAG_COP_SUPPRESSED in human
    assert Columns.FLAG_COP_SUPPRESSED not in mouse
    for column in (
        Columns.FLAG_REGION_INCOHERENT,
        Columns.REGION_COHERENCE,
        Columns.FLAG_ASTRO_LOWCOUNT,
    ):
        assert column in mouse
        assert column not in human
    fine = column_specs("human", include_fine_levels=True)
    assert Columns.level("cluster", "status") in fine
    assert Columns.level("cluster", "status") not in human
    assert human[Columns.GENES_PER_COUNT].kind == "float32"
    assert human[Columns.CT_CONSENSUS_TIER].kind == "int8"
    with pytest.raises(ValueError, match="species"):
        column_specs("rat")  # type: ignore[arg-type]


def test_branch_leaf_and_soft_vocabularies() -> None:
    assert branch_categories("human") == HUMAN_BRANCHES
    assert UNASSIGNED_LABEL in HUMAN_BRANCHES
    mouse_branches = branch_categories("mouse")
    assert len(mouse_branches) == 35
    assert mouse_branches[-1] == UNASSIGNED_LABEL
    human_leaves = leaf_categories("human")
    assert human_leaves is not None
    assert set(load_vocab("whb_supercluster").names) < set(human_leaves)
    assert "unresolved" in human_leaves
    assert leaf_categories("mouse") is None
    human_soft = soft_columns("human")
    assert "soft_broad_oligodendrocyte_precursors" in human_soft
    assert human_soft[-1] == Columns.SOFT_BROAD_UNALLOCATED
    assert len(soft_columns("mouse")) == 34
    assert all("/" not in column and " " not in column for column in human_soft)


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_valid_table_passes(species: str, make_label_table: TableFactory) -> None:
    table = make_label_table(species)
    validate_label_table(table, species, h5ad_index=_table_ids(table))  # type: ignore[arg-type]


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_parquet_round_trip_keeps_the_contract(
    species: str, make_label_table: TableFactory, tmp_path: Path
) -> None:
    table = make_label_table(species)
    path = tmp_path / label_table_filename("sample")
    table.to_parquet(path)
    restored = pd.read_parquet(path)
    validate_label_table(restored, species, h5ad_index=_table_ids(table))  # type: ignore[arg-type]
    for column in table.columns:
        if table[column].isna().all() and isinstance(
            table[column].dtype, pd.CategoricalDtype
        ):
            # All-null categoricals come back as null-typed object columns.
            assert restored[column].isna().all(), column
            continue
        assert restored[column].dtype == table[column].dtype, column


def test_h5ad_index_is_order_insensitive(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    validate_label_table(table, "human", h5ad_index=pd.Index(_table_ids(table)[::-1]))


def test_h5ad_index_mismatch_is_reported(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    ids = _table_ids(table)
    problems = _problems(table, "human", h5ad_index=[*ids[1:], "cell_extra"])
    assert any("missing from the H5AD" in problem for problem in problems)
    assert any("not in_table" in problem for problem in problems)
    duplicated = _problems(table, "human", h5ad_index=[*ids, ids[0]])
    assert any("duplicated" in problem for problem in duplicated)


def test_missing_column_and_wrong_dtype_are_reported(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("human").drop(columns=[Columns.CT_LEAF])
    table[Columns.GENES_PER_COUNT] = table[Columns.GENES_PER_COUNT].astype("float64")
    broad_name = Columns.level("broad", "name")
    table[broad_name] = table[broad_name].astype(object)
    problems = _problems(table, "human")
    assert any(
        "missing columns" in problem and "ct_leaf" in problem for problem in problems
    )
    assert any("genes_per_count: dtype float64" in problem for problem in problems)
    assert any("ct_broad_name: dtype object" in problem for problem in problems)


def test_unknown_values_are_reported(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    status = Columns.level("broad", "status")
    _set(table, 1, status, "sure")
    table[Columns.PLATFORM] = pd.Categorical(["COSMX"] * len(table))
    table[Columns.SPECIES] = pd.Categorical(["mouse"] * len(table))
    _set(table, 1, Columns.CT_BRANCH, "Neurons")
    problems = _problems(table, "human")
    assert any(
        problem.startswith("ct_broad_status: values outside") for problem in problems
    )
    assert any(problem.startswith("platform: values outside") for problem in problems)
    assert any(problem.startswith("species: values outside") for problem in problems)
    assert any(problem.startswith("ct_branch: values outside") for problem in problems)


def test_probability_and_soft_ranges_are_checked(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("human")
    _set(table, 1, Columns.level("broad", "raw"), np.float32(1.5))
    _set(table, 1, Columns.SOFT_BROAD_UNALLOCATED, np.float32(-0.1))
    problems = _problems(table, "human")
    assert any(
        problem.startswith("ct_broad_raw: 1 values outside") for problem in problems
    )
    assert any(
        problem.startswith("soft_broad_unallocated: 1 values") for problem in problems
    )


def test_duplicated_cell_ids_are_reported(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    _set(table, 2, Columns.CELL_ID, table.loc[1, Columns.CELL_ID])
    problems = _problems(table, "human")
    assert any("duplicated ids" in problem for problem in problems)


def test_missing_status_is_reported(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    _set(table, 1, Columns.level("nt", "status"), np.nan)
    problems = _problems(table, "human")
    assert any(problem.startswith("ct_nt_status: 1 missing") for problem in problems)


def test_low_counts_must_match_in_table(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    _set(table, 0, Columns.FLAG_LOW_COUNTS, False)
    _set(table, 1, Columns.level("lineage", "status"), CellStatus.LOW_COUNTS.value)
    _set(table, 0, Columns.level("broad", "status"), CellStatus.LOW_CONFIDENCE.value)
    problems = _problems(table, "human")
    assert "flag_low_counts must equal ~in_table" in problems
    assert "lineage: low_counts status on table cells" in problems
    assert "broad: objects outside the table must be low_counts" in problems


def test_validated_only_on_confident_cells(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    _set(table, 2, Columns.level("supercluster", "validated"), True)
    problems = _problems(table, "human")
    assert "supercluster: validated is true on non-confident cells" in problems


def test_names_absent_where_not_applicable(make_label_table: TableFactory) -> None:
    table = make_label_table("human")
    name = Columns.level("nt", "name")
    _set(table, 2, name, "Excitatory")
    problems = _problems(table, "human")
    assert any(problem.startswith("nt: a name is set") for problem in problems)


def test_final_level_must_be_confident_and_named(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("human")
    _set(table, 2, Columns.CT_FINAL_LEVEL, "supercluster")
    _set(table, 1, Columns.CT_FINAL_NAME, "Astrocyte x")
    _set(table, 4, Columns.CT_FINAL_NAME, "Neurons")
    problems = _problems(table, "human")
    assert "ct_final_level 'supercluster' on cells not confident there" in problems
    assert "ct_final_name differs from ct_supercluster_name" in problems
    assert f"ct_final_name must be {UNASSIGNED_LABEL!r} when level is none" in problems


def test_final_level_is_the_deepest_confident_level(
    make_label_table: TableFactory,
) -> None:
    """A confident level deeper than ct_final_level is a violation (§4.1)."""
    table = make_label_table("human")
    # Astrocyte row: confident at supercluster, but the final level stays broad.
    _set(table, 2, Columns.level("supercluster", "status"), CellStatus.CONFIDENT.value)

    problems = _problems(table, "human")

    assert any(
        "coarser than 'supercluster' on cells confident at 'supercluster'" in problem
        for problem in problems
    )


def test_final_level_needs_confident_parents(make_label_table: TableFactory) -> None:
    """Every applicable level above ct_final_level must be confident (§4.2)."""
    table = make_label_table("human")
    # Neuron row: final supercluster, but broad is only low_confidence.
    _set(table, 1, Columns.level("broad", "status"), CellStatus.LOW_CONFIDENCE.value)
    _set(table, 1, Columns.level("broad", "validated"), False)

    problems = _problems(table, "human")

    assert any(
        "deeper than 'broad' on cells not confident at 'broad'" in problem
        for problem in problems
    )


def test_final_level_chain_skips_not_applicable_levels(
    make_label_table: TableFactory,
) -> None:
    """nt is not_applicable for glia, so their final level may lie below it."""
    table = make_label_table("human")
    _set(table, 2, Columns.level("supercluster", "status"), CellStatus.CONFIDENT.value)
    _set(table, 2, Columns.CT_FINAL_LEVEL, "supercluster")
    _set(table, 2, Columns.CT_FINAL_NAME, "Astrocyte")
    _set(table, 2, Columns.CT_LEAF, "Astrocyte")

    validate_label_table(table, "human")
    mouse = make_label_table("mouse")
    _set(mouse, 1, Columns.level("class", "status"), CellStatus.BELOW_FLOOR.value)
    _set(mouse, 1, Columns.level("class", "validated"), False)
    assert any(
        "deeper than 'class'" in problem for problem in _problems(mouse, "mouse")
    )


def test_fine_levels_are_detected_and_can_be_required(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("human")
    with pytest.raises(LabelTableError, match="missing columns"):
        validate_label_table(table, "human", include_fine_levels=True)
    fine = REPORT_ONLY_LEVELS["human"][0]
    table[Columns.level(fine, "name")] = [None] * len(table)
    for field in ("raw", "conf", "corr", "margin"):
        table[Columns.level(fine, field)] = np.full(len(table), np.nan)
    table[Columns.level(fine, "runner_up")] = [None] * len(table)
    table[Columns.level(fine, "status")] = [CellStatus.LOW_COUNTS.value] + [
        CellStatus.NOT_RESOLVABLE.value
    ] * (len(table) - 1)
    table[Columns.level(fine, "validated")] = False
    table = coerce_label_table_dtypes(table, "human", include_fine_levels=True)
    validate_label_table(table, "human")
    _set(table, 0, Columns.level(fine, "status"), CellStatus.NOT_RESOLVABLE.value)
    with pytest.raises(LabelTableError, match="cluster: objects outside the table"):
        validate_label_table(table, "human")


def test_coerce_label_table_dtypes_fixes_default_dtypes(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("mouse")
    loose = table.astype(
        {
            Columns.TOTAL_COUNTS: "int64",
            Columns.GENES_PER_COUNT: "float64",
            Columns.PLATFORM: object,
            Columns.CT_CONSENSUS_TIER: "int64",
        }
    )
    loose[Columns.SOFT_CLASS_PREFIX + "01_it_et_glut"] = loose[
        Columns.SOFT_CLASS_PREFIX + "01_it_et_glut"
    ].astype("float64")
    with pytest.raises(LabelTableError):
        validate_label_table(loose, "mouse")
    validate_label_table(coerce_label_table_dtypes(loose, "mouse"), "mouse")


def test_coercion_keeps_missing_values_for_validation(
    make_label_table: TableFactory,
) -> None:
    """A null id or NaN flag is not cast into 'None' or True and passes unseen."""
    table = make_label_table("human")
    loose = table.astype({Columns.CELL_ID: object, Columns.EXCLUDE_HARD: object})
    loose.loc[1, Columns.CELL_ID] = None
    loose.loc[2, Columns.EXCLUDE_HARD] = np.nan
    loose[Columns.IN_TABLE] = loose[Columns.IN_TABLE].astype(float)
    loose.loc[3, Columns.IN_TABLE] = np.nan

    coerced = coerce_label_table_dtypes(loose, "human")

    assert coerced.loc[1, Columns.CELL_ID] is None
    assert coerced[Columns.CELL_ID].tolist()[0] == "cell_0"
    assert pd.isna(coerced.loc[2, Columns.EXCLUDE_HARD])
    assert pd.isna(coerced.loc[3, Columns.IN_TABLE])
    with pytest.raises(LabelTableError) as error:
        validate_label_table(coerced, "human")
    problems = "\n".join(error.value.problems)
    assert "cell_id: 1 missing values" in problems
    assert "exclude_hard: 1 missing values" in problems
    assert "in_table: 1 missing values" in problems


def test_coercion_casts_string_ids_without_touching_nulls(
    make_label_table: TableFactory,
) -> None:
    table = make_label_table("mouse")
    table[Columns.CELL_ID] = pd.array(
        [*range(len(table) - 1), None], dtype="Int64"
    ).astype(object)

    coerced = coerce_label_table_dtypes(table, "mouse")

    assert coerced[Columns.CELL_ID].tolist()[:-1] == [
        str(index) for index in range(len(table) - 1)
    ]
    assert coerced[Columns.CELL_ID].iloc[-1] is None


def test_error_message_is_truncated() -> None:
    error = LabelTableError([f"problem {index}" for index in range(25)])
    assert "and 5 more" in str(error)
    assert len(error.problems) == 25
