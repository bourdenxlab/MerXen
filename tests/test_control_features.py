"""Tests for the platform control-feature registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merxen.analysis import clustering_squidpy
from merxen.control_features import (
    CONTROL_TOKENS,
    classify_control_transcripts,
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


def test_control_tokens_match_genomic_control_names() -> None:
    """10x genomic controls (``Intergenic_Region_*``) carry a control token."""
    assert has_control_token("Intergenic_Region_12")
    assert control_token_mask(["Intergenic_Region_12", "GFAP"]).tolist() == [
        True,
        False,
    ]


def test_atera_platform_uses_the_xenium_pattern() -> None:
    """Atera control names follow Xenium; MERSCOPE blanks are not Atera names."""
    assert matches_control_name_pattern("Intergenic_Region_1", "ATERA")
    assert matches_control_name_pattern("Intergenic_Region_1", "atera")
    assert not matches_control_name_pattern("Blank-3", "ATERA")


# Feature-type categories and control-name prefixes of the Atera cell-feature
# matrix (``var.feature_type``); Xenium uses the same ones.
TENX_CONTROL_CATEGORIES: list[tuple[str, str]] = [
    ("Negative Control Codeword", "NegControlCodeword_0500"),
    ("Negative Control Probe", "NegControlProbe_00002"),
    ("Genomic Control", "Intergenic_Region_12"),
    ("Unassigned Codeword", "UnassignedCodeword_0003"),
    ("Deprecated Codeword", "DeprecatedCodeword_0001"),
]

# Gene symbols that contain fragments of control prefixes or tokens.
TOKEN_LIKE_GENES: list[str] = [
    "GFAP",
    "HLA-DMB",
    "Nkx6-1",
    "BLNK",
    "NEGR1",
    "DEPP1",
    "UNC13A",
    "INTS1",
    "CNTNAP2",
    "INTS6L",
]


@pytest.mark.parametrize("platform", ["XENIUM", "ATERA", None])
@pytest.mark.parametrize(("feature_type", "name"), TENX_CONTROL_CATEGORIES)
def test_every_10x_control_category_is_a_control_by_name(
    platform: str | None, feature_type: str, name: str
) -> None:
    """Each control category is recognised from its name alone."""
    del feature_type
    assert matches_control_name_pattern(name, platform)
    assert is_registered_control_name(name, platform)
    assert control_transcript_mask([name], platform=platform).tolist() == [True]
    assert control_token_mask([name]).tolist() == [True]


@pytest.mark.parametrize("platform", ["XENIUM", "ATERA", "MERSCOPE", None])
def test_token_like_gene_names_are_not_controls(platform: str | None) -> None:
    """Names that only contain control-like fragments stay genes."""
    assert not control_transcript_mask(TOKEN_LIKE_GENES, platform=platform).any()
    assert not control_token_mask(TOKEN_LIKE_GENES).any()


def test_feature_type_flags_every_10x_control_category() -> None:
    """``feature_type`` decides for unknown names; ``Gene Expression`` is a gene."""
    types = [feature_type for feature_type, _ in TENX_CONTROL_CATEGORIES]
    names = [f"Odd_{i}" for i in range(len(types))]

    mask = control_transcript_mask(
        [*names, "GFAP", "gfap2"],
        platform="ATERA",
        feature_type=[*types, "Gene Expression", " gene expression "],
    )

    assert mask.tolist() == [True] * len(types) + [False, False]


def test_feature_type_gene_keeps_control_like_name_and_falls_back() -> None:
    """A gene feature type overrides names; a missing type uses the name."""
    names = ["Intergenic_Region_4", "NegControlProbe_1", "Odd_1", "GFAP"]
    feature_type = pd.Series(["Gene Expression", None, "", "Gene Expression"])

    flags = classify_control_transcripts(
        names, platform="ATERA", feature_type=feature_type
    )

    assert flags.control.tolist() == [False, True, False, False]
    assert flags.kept_by_feature_type.tolist() == [True, False, False, False]


def test_is_gene_and_codeword_category_take_precedence_over_feature_type() -> None:
    """``feature_type`` fills only rows the transcript columns leave unknown."""
    names = ["Odd_1", "Odd_2", "Odd_3"]
    feature_type = ["Genomic Control", "Gene Expression", "Genomic Control"]

    mask = control_transcript_mask(
        names,
        platform="ATERA",
        is_gene=pd.Series([True, pd.NA, pd.NA], dtype="boolean"),
        codeword_category=[None, "negative_control_codeword", None],
        feature_type=feature_type,
    )

    assert mask.tolist() == [False, True, True]


def test_atera_boosted_gene_with_is_gene_is_not_a_control() -> None:
    """``is_gene`` keeps a boosted gene whose category is not a gene category."""
    mask = control_transcript_mask(
        ["GFAP", "Intergenic_Region_1"],
        platform="ATERA",
        is_gene=[True, False],
        codeword_category=["boosted_gene", "genomic_control"],
    )

    assert mask.tolist() == [False, True]


def test_merscope_blanks_are_unaffected_by_the_10x_rules() -> None:
    """MERSCOPE ``Blank-*`` codewords stay controls; hyphenated genes stay genes."""
    names = ["Blank-0", "Blank-84", "Nkx6-1", "Gad1"]

    mask = control_transcript_mask(names, platform="MERSCOPE")

    assert mask.tolist() == [True, True, False, False]


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
    is_gene = pd.Series([True, False, False, True])

    mask = control_transcript_mask(names, platform="XENIUM", is_gene=is_gene)

    assert mask.tolist() == [False, True, True, False]


def test_feature_type_gene_overrides_control_name_and_is_reported() -> None:
    """A gene feature type keeps a control-like name; the override is flagged."""
    names = [
        "antisense_PROKR2",
        "Intergenic_Region_4",
        "antisense_PROKR2",
        "GFAP",
        "BLANK_0006",
    ]
    is_gene = pd.Series([True, True, pd.NA, True, False], dtype="boolean")

    flags = classify_control_transcripts(names, platform="XENIUM", is_gene=is_gene)

    # Rows 0-1 are genes by feature type; row 2 has no type, so its name
    # decides; row 4 is a control by type (and by name, so not an override).
    assert flags.control.tolist() == [False, False, True, False, True]
    assert flags.kept_by_feature_type.tolist() == [True, True, False, False, False]
    assert (
        control_transcript_mask(names, platform="XENIUM", is_gene=is_gene).tolist()
        == flags.control.tolist()
    )


def test_control_transcript_mask_keeps_token_like_gene_with_feature_type() -> None:
    """A usable feature type overrides the substring fallback for real genes."""
    names = ["EGFP_control_reporter", "EGFP_control_reporter"]

    typed = classify_control_transcripts(
        names, platform="XENIUM", is_gene=np.array([True, True])
    )
    untyped = classify_control_transcripts(names, platform="XENIUM")

    assert typed.control.tolist() == [False, False]
    assert typed.kept_by_feature_type.tolist() == [True, True]
    assert untyped.control.tolist() == [True, True]
    assert untyped.kept_by_feature_type.tolist() == [False, False]


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

    flags = classify_control_transcripts(names, platform="MERSCOPE")

    assert flags.control.tolist() == [False, True, False, False, True]
    assert not flags.kept_by_feature_type.any()


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
    with pytest.raises(ValueError, match="feature_type"):
        control_transcript_mask(["GFAP"], feature_type=["a", "b"])


def test_is_registered_control_name_combines_patterns_and_tokens() -> None:
    """Registered names are those the name rules alone would drop."""
    assert is_registered_control_name("Intergenic_Region_1", "XENIUM")
    assert is_registered_control_name("SystemControl_3", "XENIUM")
    assert not is_registered_control_name("NewType_1", "XENIUM")
