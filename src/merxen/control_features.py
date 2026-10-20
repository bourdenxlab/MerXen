"""Registry of platform control features (negative controls, blanks, codewords).

Xenium and MERSCOPE report control features alongside real genes. This module
is the single place that recognises them, so segmentation inputs, clustering
and transcript analyses drop the same features.

Sources for the names and categories below:

* Xenium (10x Genomics, *Understanding Xenium Outputs* and the Xenium Onboard
  Analysis release notes): cell-feature-matrix feature types are
  ``Gene Expression``, ``Negative Control Codeword``, ``Negative Control
  Probe``, ``Genomic Control`` (Xenium Prime), ``Unassigned Codeword`` and
  ``Deprecated Codeword``. Since XOA 3.0 ``transcripts.parquet`` also carries
  ``is_gene`` (whether the feature is ``Gene Expression``) and
  ``codeword_category`` (``predesigned_gene`` / ``custom_gene`` for genes).
  Control feature names start with ``NegControlProbe_``,
  ``NegControlCodeword_``, ``UnassignedCodeword_`` (``BLANK_`` before the
  rename in early XOA releases), ``DeprecatedCodeword_`` and, for the Xenium
  Prime 5K genomic controls, ``Intergenic_Region_``. Pre-release datasets name
  some negative control probes ``antisense_<GENE>``.
* MERSCOPE (Vizgen): blank codewords are listed in the codebook next to the
  genes and appear in ``detected_transcripts`` as ``Blank-<N>``. MERSCOPE
  transcript tables carry no feature-type column.

When a transcript has a usable feature type (``is_gene`` or
``codeword_category``), that type alone decides whether it is a control; the
name rules apply only to transcripts without one. So a name rule can never
drop a feature the platform labels as a gene. This matters most for
``antisense_``: it is kept as a name rule for pre-release Xenium data, which
has no feature-type column, and a custom-panel gene with that prefix is kept
on XOA >= 3.0 data. Rows where a name rule and a gene feature type disagree
are reported by :func:`classify_control_transcripts` so callers can log them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

# Case-insensitive substrings used by clustering and transcript analyses to
# recognise control-like features when no platform-specific rule applies.
CONTROL_TOKENS: tuple[str, ...] = (
    "blank",
    "control",
    "negative",
    "negcontrol",
    "unassigned",
    "deprecated",
)

# ``codeword_category`` values of real genes in Xenium ``transcripts.parquet``.
# Every other non-empty category is a control.
XENIUM_GENE_CODEWORD_CATEGORIES: frozenset[str] = frozenset(
    {"predesigned_gene", "custom_gene"}
)

# Anchored, case-sensitive name rules, used for transcripts without a feature
# type. The Xenium rule keeps ProSeg's own Xenium preset prefixes
# (``Deprecated|NegControl|Unassigned|Intergenic``) and adds the other
# documented control names.
XENIUM_CONTROL_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:NegControl|Unassigned|Deprecated|Intergenic|GenomicControl|BLANK_"
    r"|antisense_)"
)
MERSCOPE_CONTROL_NAME_PATTERN: re.Pattern[str] = re.compile(r"^Blank-\d+$")

_CONTROL_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "XENIUM": (XENIUM_CONTROL_NAME_PATTERN,),
    "MERSCOPE": (MERSCOPE_CONTROL_NAME_PATTERN,),
}
_ALL_CONTROL_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    XENIUM_CONTROL_NAME_PATTERN,
    MERSCOPE_CONTROL_NAME_PATTERN,
)

_TRUE_STRINGS = frozenset({"true", "t", "1", "yes", "y"})
_FALSE_STRINGS = frozenset({"false", "f", "0", "no", "n"})


def _name_patterns(platform: str | None) -> tuple[re.Pattern[str], ...]:
    if platform is None:
        return _ALL_CONTROL_NAME_PATTERNS
    return _CONTROL_NAME_PATTERNS.get(platform.upper(), _ALL_CONTROL_NAME_PATTERNS)


def matches_control_name_pattern(name: str, platform: str | None = None) -> bool:
    """Return whether a feature name matches a documented control-name rule.

    Args:
        name: Feature name as reported by the platform.
        platform: ``"XENIUM"`` or ``"MERSCOPE"`` (case-insensitive). ``None`` or
            an unknown platform checks the rules of every platform.

    Returns:
        ``True`` when an anchored platform control-name pattern matches.
    """
    return any(pattern.match(str(name)) for pattern in _name_patterns(platform))


def has_control_token(name: str) -> bool:
    """Return whether a name contains one of :data:`CONTROL_TOKENS`.

    Args:
        name: Feature, column or key name.

    Returns:
        ``True`` when a control token occurs anywhere in the lower-cased name.
    """
    lower = str(name).lower()
    return any(token in lower for token in CONTROL_TOKENS)


def control_token_mask(values: Any) -> np.ndarray:
    """Vectorised :func:`has_control_token` over feature names.

    Args:
        values: One-dimensional array-like of names. Missing values are
            converted with ``str`` and so never match.

    Returns:
        Boolean array, ``True`` where a name contains a control token.
    """
    names = pd.Series(values, dtype=object).astype(str)
    codes, uniques = pd.factorize(names)
    unique_mask = np.fromiter(
        (has_control_token(name) for name in uniques),
        dtype=bool,
        count=len(uniques),
    )
    return np.asarray(unique_mask[codes], dtype=bool)


def _optional_bool(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(known, value)`` arrays for a boolean-like column."""
    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        known = series.notna().to_numpy(dtype=bool)
        value = series.fillna(False).astype(bool).to_numpy(dtype=bool)
        return known, value
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.notna().to_numpy(dtype=bool), (numeric != 0).to_numpy(dtype=bool)
    text = series.astype("string").str.strip().str.lower()
    is_true = text.isin(_TRUE_STRINGS).fillna(False).to_numpy(dtype=bool)
    is_false = text.isin(_FALSE_STRINGS).fillna(False).to_numpy(dtype=bool)
    return is_true | is_false, is_true


class ControlTranscriptFlags(NamedTuple):
    """Per-transcript result of :func:`classify_control_transcripts`.

    Attributes:
        control: ``True`` for control transcripts.
        kept_by_feature_type: ``True`` where a control-name rule matches but
            the feature type says gene, so the transcript is kept.
    """

    control: np.ndarray
    kept_by_feature_type: np.ndarray


def classify_control_transcripts(
    feature_names: Sequence[Any] | np.ndarray | pd.Series,
    *,
    platform: str | None = None,
    is_gene: Sequence[Any] | np.ndarray | pd.Series | None = None,
    codeword_category: Sequence[Any] | np.ndarray | pd.Series | None = None,
) -> ControlTranscriptFlags:
    """Flag control transcripts, preferring the platform's feature-type columns.

    A transcript with a usable feature type is a control when that type says
    it is not a gene: ``is_gene`` false, else a ``codeword_category`` outside
    :data:`XENIUM_GENE_CODEWORD_CATEGORIES`. A transcript without one is a
    control when its name matches the platform's anchored control-name
    pattern or contains one of :data:`CONTROL_TOKENS`.

    Args:
        feature_names: Per-transcript feature names.
        platform: ``"XENIUM"`` or ``"MERSCOPE"``; selects the name patterns.
        is_gene: Optional per-transcript Xenium ``is_gene`` values.
        codeword_category: Optional per-transcript Xenium
            ``codeword_category`` values, used where ``is_gene`` is missing.

    Returns:
        The control flags, plus the transcripts kept only because their
        feature type overrode a matching control-name rule.

    Raises:
        ValueError: If a feature-type column differs in length from
            ``feature_names``.
    """
    names = pd.Series(np.asarray(feature_names, dtype=object), dtype=object)
    n_rows = len(names)
    codes, uniques = pd.factorize(names)
    unique_names = [str(name) for name in uniques]
    unique_name_rule = np.fromiter(
        (
            matches_control_name_pattern(name, platform) or has_control_token(name)
            for name in unique_names
        ),
        dtype=bool,
        count=len(unique_names),
    )
    # A trailing False lets missing names (code -1) index a non-match.
    name_rows = np.append(unique_name_rule, False)[codes]

    has_type = np.zeros(n_rows, dtype=bool)
    type_is_control = np.zeros(n_rows, dtype=bool)
    if is_gene is not None:
        if len(is_gene) != n_rows:
            raise ValueError("is_gene must have one value per feature name")
        known, gene_values = _optional_bool(is_gene)
        type_is_control[known] = ~gene_values[known]
        has_type |= known
    if codeword_category is not None:
        if len(codeword_category) != n_rows:
            raise ValueError("codeword_category must have one value per feature name")
        categories = pd.Series(codeword_category).astype("string").str.strip()
        known = (categories.notna() & (categories != "")).to_numpy(dtype=bool)
        use = known & ~has_type
        is_gene_category = (
            categories.isin(XENIUM_GENE_CODEWORD_CATEGORIES)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        type_is_control[use] = ~is_gene_category[use]
        has_type |= use

    return ControlTranscriptFlags(
        control=np.asarray(np.where(has_type, type_is_control, name_rows), dtype=bool),
        kept_by_feature_type=np.asarray(
            has_type & ~type_is_control & name_rows, dtype=bool
        ),
    )


def control_transcript_mask(
    feature_names: Sequence[Any] | np.ndarray | pd.Series,
    *,
    platform: str | None = None,
    is_gene: Sequence[Any] | np.ndarray | pd.Series | None = None,
    codeword_category: Sequence[Any] | np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Return the control flags of :func:`classify_control_transcripts`.

    Args:
        feature_names: Per-transcript feature names.
        platform: ``"XENIUM"`` or ``"MERSCOPE"``; selects the name patterns.
        is_gene: Optional per-transcript Xenium ``is_gene`` values.
        codeword_category: Optional per-transcript Xenium
            ``codeword_category`` values, used where ``is_gene`` is missing.

    Returns:
        Boolean array, ``True`` for control transcripts.

    Raises:
        ValueError: If a feature-type column differs in length from
            ``feature_names``.
    """
    return classify_control_transcripts(
        feature_names,
        platform=platform,
        is_gene=is_gene,
        codeword_category=codeword_category,
    ).control


def is_registered_control_name(name: str, platform: str | None = None) -> bool:
    """Return whether the name rules alone recognise a control feature.

    Used to report controls found only through a feature-type column, which
    means this registry is missing their name.

    Args:
        name: Feature name.
        platform: ``"XENIUM"`` or ``"MERSCOPE"``; selects the name patterns.

    Returns:
        ``True`` when the anchored pattern or a control token matches.
    """
    return matches_control_name_pattern(name, platform) or has_control_token(name)
