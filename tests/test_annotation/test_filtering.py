"""Tests for ``merxen.annotation.filtering.select_cells`` (always-label M5)."""

from __future__ import annotations

import pandas as pd
import pytest

from merxen.annotation.filtering import select_cells


def _labels() -> pd.DataFrame:
    """Four mouse cells over two datasets (the second failed G1)."""
    return pd.DataFrame(
        {
            "species": ["mouse"] * 4,
            "ct_subclass_status": [
                "confident",
                "low_confidence",
                "confident",
                "confident",
            ],
            "ct_subclass_status_strict": [
                "confident",
                "low_confidence",
                "not_attempted_gate",
                "not_attempted_gate",
            ],
            "ct_broad_status": ["confident"] * 4,
            "qc_dataset_max_level": ["subclass", "subclass", "none", "none"],
            "qc_dataset_warnings": [
                "",
                "",
                "g1_registration_fail;panel_provisional_warn",
                "g1_registration_fail;panel_provisional_warn",
            ],
            "flag_microglial_spillover": pd.array(
                [False, False, None, True], dtype="boolean"
            ),
            "exclude_hard": [False, False, False, False],
            "exclude_hard_strict": [False, False, True, True],
        },
        index=["a", "b", "c", "d"],
    )


def test_select_cells_keeps_confident_cells_at_the_level_by_default() -> None:
    keep = select_cells(_labels(), level="subclass")
    assert keep.index.tolist() == ["a", "b", "c", "d"]
    assert keep.tolist() == [True, False, True, True]


def test_select_cells_drops_datasets_whose_qc_does_not_support_the_level() -> None:
    labels = _labels()
    assert select_cells(
        labels, level="subclass", within_dataset_support=True
    ).tolist() == [True, False, False, False]


def test_select_cells_matches_warning_tokens_by_suffix_and_prefix() -> None:
    labels = _labels()
    by_suffix = select_cells(labels, level="broad", exclude_dataset_warnings=["_fail"])
    by_prefix = select_cells(labels, level="broad", exclude_dataset_warnings=["g1_"])
    by_other = select_cells(labels, level="broad", exclude_dataset_warnings=["g2_"])
    assert by_suffix.tolist() == [True, True, False, False]
    assert by_prefix.tolist() == by_suffix.tolist()
    assert by_other.all()


def test_select_cells_keeps_cells_whose_flag_was_not_evaluated() -> None:
    keep = select_cells(
        _labels(), level="subclass", exclude_flags=["flag_microglial_spillover"]
    )
    assert keep.tolist() == [True, False, True, False]


def test_select_cells_strict_reads_the_strict_view() -> None:
    keep = select_cells(_labels(), level="subclass", strict=True)
    assert keep.tolist() == [True, False, False, False]


def test_select_cells_drops_exclude_hard_cells_unless_asked() -> None:
    labels = _labels()
    labels["exclude_hard"] = [True, False, False, False]
    assert not select_cells(labels, level="broad")["a"]
    assert select_cells(labels, level="broad", exclude_hard=False)["a"]


def test_select_cells_raises_on_a_missing_level() -> None:
    with pytest.raises(KeyError, match="ct_class_status"):
        select_cells(_labels(), level="class")


def test_select_cells_raises_on_a_version_1_table_for_dataset_filters() -> None:
    labels = _labels().drop(columns=["qc_dataset_max_level", "qc_dataset_warnings"])
    with pytest.raises(KeyError, match="qc_dataset_max_level"):
        select_cells(labels, level="subclass", within_dataset_support=True)
    with pytest.raises(KeyError, match="qc_dataset_warnings"):
        select_cells(labels, level="subclass", exclude_dataset_warnings=["_fail"])


def test_select_cells_refuses_a_level_outside_the_chain_for_dataset_support() -> None:
    labels = _labels()
    labels["ct_supertype_status"] = "confident"
    with pytest.raises(ValueError, match="chain level"):
        select_cells(labels, level="supertype", within_dataset_support=True)


def test_select_cells_reads_the_species_of_a_clustered_obs_from_its_leaf() -> None:
    obs = _labels().drop(columns=["species"])
    assert select_cells(
        obs, level="subclass", within_dataset_support=True
    ).tolist() == [True, False, False, False]
