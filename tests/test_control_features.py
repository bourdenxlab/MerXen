"""Tests for the platform control-feature registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merxen.analysis import clustering_squidpy
from merxen.control_features import (
    CONTROL_TOKENS,
    control_token_mask,
    control_transcript_mask,
    has_control_token,
    is_registered_control_name,
    matches_control_name_pattern,
)


def test_control_tokens_are_shared_with_clustering() -> None:
    """Clustering keeps importing the registry's token tuple."""
    assert clustering_squidpy.CONTROL_TOKENS is CONTROL_TOKENS


def test_has_control_token_is_case_insensitive_substring() -> None:
    """Token matching keeps the legacy case-insensitive substring rule."""
    assert has_control_token("Blank-12")
    assert has_control_token("NegControlProbe_00002")
    assert has_control_token("control_probe_counts")
    assert not has_control_token("GFAP")
    assert not has_control_token("HLA-DMB")


def test_control_token_mask_matches_scalar_rule_and_handles_missing() -> None:
    """The vectorised mask equals the scalar rule; missing names never match."""
    values = pd.Series(
        ["GeneA", "BLANK_0006", None, "UnassignedCodeword_0003", "GeneA", np.nan]
    )

    mask = control_token_mask(values)

    assert mask.tolist() == [False, True, False, True, False, False]
    assert control_token_mask([]).tolist() == []


@pytest.mark.parametrize(
    "name",
    [
        "NegControlProbe_00002",
        "NegControlCodeword_0500",
        "UnassignedCodeword_0003",
        "DeprecatedCodeword_0001",
        "BLANK_0006",
        "Intergenic_Region_12",
        "GenomicControlProbe_0001",
        "antisense_PROKR2",
    ],
)
def test_xenium_name_pattern_matches_documented_controls(name: str) -> None:
    """Every documented Xenium control prefix is recognised by name."""
    assert matches_control_name_pattern(name, "XENIUM")
    assert matches_control_name_pattern(name, "xenium")


@pytest.mark.parametrize(
    "name",
    [
        "GFAP",
        "HLA-DMB",
        "ENST00000262410.10_MAPT3R_exon9_exon11",
        "Blank-12",
        "blank_0006",
    ],
)
def test_xenium_name_pattern_keeps_genes_and_other_platform_names(name: str) -> None:
    """Xenium name rules are anchored and case-sensitive."""
    assert not matches_control_name_pattern(name, "XENIUM")


def test_merscope_name_pattern_is_anchored_to_blank_codewords() -> None:
    """MERSCOPE blanks are ``Blank-<N>``; hyphenated genes are kept."""
    assert matches_control_name_pattern("Blank-0", "MERSCOPE")
    assert matches_control_name_pattern("Blank-84", "MERSCOPE")
    for gene in ["Nkx6-1", "HLA-DQA1", "Blank-", "Blank-1a", "NegControlProbe_1"]:
        assert not matches_control_name_pattern(gene, "MERSCOPE")


def test_unknown_platform_checks_every_platform_pattern() -> None:
    """Without a known platform every anchored rule applies."""
    assert matches_control_name_pattern("Blank-3")
    assert matches_control_name_pattern("NegControlProbe_1", "COSMX")
    assert not matches_control_name_pattern("GFAP")


def test_control_transcript_mask_prefers_xenium_is_gene() -> None:
    """``is_gene`` flags controls whose names the registry does not know."""
    names = ["GFAP", "Odd_1", "UnassignedCodeword_0003", "GFAP"]
    is_gene = pd.Series([True, False, True, True])

    mask = control_transcript_mask(names, platform="XENIUM", is_gene=is_gene)

    # An anchored name is dropped even when is_gene disagrees.
    assert mask.tolist() == [False, True, True, False]


def test_control_transcript_mask_keeps_token_like_gene_with_feature_type() -> None:
    """A usable feature type overrides the substring fallback for real genes."""
    names = ["EGFP_control_reporter", "EGFP_control_reporter"]

    typed = control_transcript_mask(
        names, platform="XENIUM", is_gene=np.array([True, True])
    )
    untyped = control_transcript_mask(names, platform="XENIUM")

    assert typed.tolist() == [False, False]
    assert untyped.tolist() == [True, True]


def test_control_transcript_mask_uses_codeword_category_without_is_gene() -> None:
    """``codeword_category`` decides where ``is_gene`` is absent or missing."""
    names = ["GFAP", "SNAP25", "Odd_1", "Odd_2", "GFAP"]
    categories = [
        "predesigned_gene",
        "custom_gene",
        "genomic_control_probe",
        None,
        "",
    ]
    is_gene = pd.Series([pd.NA, pd.NA, pd.NA, pd.NA, True], dtype="boolean")

    only_category = control_transcript_mask(
        names, platform="XENIUM", codeword_category=categories
    )
    both = control_transcript_mask(
        names, platform="XENIUM", is_gene=is_gene, codeword_category=categories
    )

    assert only_category.tolist() == [False, False, True, False, False]
    assert both.tolist() == [False, False, True, False, False]


def test_control_transcript_mask_merscope_blanks_and_missing_names() -> None:
    """MERSCOPE blanks are dropped; missing names are never controls."""
    names = np.array(["Gad1", "Blank-7", None, "Nkx6-1", "Blank-7"], dtype=object)

    mask = control_transcript_mask(names, platform="MERSCOPE")

    assert mask.tolist() == [False, True, False, False, True]


def test_control_transcript_mask_accepts_string_booleans() -> None:
    """String ``is_gene`` values from CSV exports are understood."""
    mask = control_transcript_mask(
        ["GFAP", "Odd_1", "Odd_2"],
        platform="XENIUM",
        is_gene=["true", "False", "maybe"],
    )

    assert mask.tolist() == [False, True, False]


def test_control_transcript_mask_rejects_misaligned_feature_types() -> None:
    """Feature-type columns must align with the names."""
    with pytest.raises(ValueError, match="is_gene"):
        control_transcript_mask(["GFAP"], is_gene=[True, False])
    with pytest.raises(ValueError, match="codeword_category"):
        control_transcript_mask(["GFAP"], codeword_category=[])


def test_is_registered_control_name_combines_patterns_and_tokens() -> None:
    """Registered names are those the name rules alone would drop."""
    assert is_registered_control_name("Intergenic_Region_1", "XENIUM")
    assert is_registered_control_name("SystemControl_3", "XENIUM")
    assert not is_registered_control_name("NewType_1", "XENIUM")
