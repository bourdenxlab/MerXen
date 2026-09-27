"""Contract of the per-cell label table ``<sid>_celltype_labels.parquet``.

One row per segmented object of ``<sid>_prepared.h5ad`` (plan §4.1). Column
names, dtypes and vocabularies live here, together with
``validate_label_table``, so the writer (RESOLVE) and every reader (the
map_first hierarchy, FINALIZE, the report) share one definition.

This module imports only the standard library, numpy and pandas (GPU-env
safe; plan §3.5).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.vocab import (
    FINAL_LEVELS,
    HUMAN_BROAD_CLASSES,
    SPECIES,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
    Species,
    load_vocab,
)

logger = logging.getLogger(__name__)

LABEL_TABLE_VERSION: Final = 1
LABEL_TABLE_SUFFIX: Final = "_celltype_labels.parquet"
PLATFORMS: Final[tuple[str, ...]] = ("MERSCOPE", "XENIUM")
# Probabilities are stored as float32 (tidy MMC parquet, ``ct_*_raw``), so a
# bootstrap probability of exactly 0.69, 0.45 or 0.90 reads back just below
# its threshold. Threshold tests allow this much below the threshold, far
# less than the 1 / bootstrap_iteration grid step.
PROBABILITY_TOLERANCE: Final = 1e-6


def meets_threshold(
    values: np.ndarray | pd.Series | float, threshold: np.ndarray | float
) -> np.ndarray:
    """Return ``values >= threshold``, tolerant of float32 storage.

    Args:
        values: Probabilities (NaN never meets a threshold).
        threshold: A threshold, or one per value.

    Returns:
        Boolean array (``PROBABILITY_TOLERANCE`` below the threshold passes).
    """
    array = np.asarray(values, dtype=np.float64)
    limit = np.asarray(threshold, dtype=np.float64) - PROBABILITY_TOLERANCE
    with np.errstate(invalid="ignore"):
        return np.asarray(np.nan_to_num(array, nan=-np.inf) >= limit)


class CellStatus(StrEnum):
    """Per-cell, per-level annotation status (plan §4.2)."""

    CONFIDENT = "confident"
    LOW_CONFIDENCE = "low_confidence"
    BELOW_FLOOR = "below_floor"
    NOT_RESOLVABLE = "not_resolvable"
    METHOD_DISAGREE = "method_disagree"
    SINGLE_METHOD = "single_method"
    IMPLAUSIBLE = "implausible"
    PARENT_UNRESOLVED = "parent_unresolved"
    NOT_ATTEMPTED_GATE = "not_attempted_gate"
    NOT_APPLICABLE = "not_applicable"
    LOW_COUNTS = "low_counts"


# Statuses whose level carries no assigned name (§4.1: null if not applicable
# or not attempted).
NAMELESS_STATUSES: Final[frozenset[str]] = frozenset(
    {
        CellStatus.LOW_COUNTS,
        CellStatus.NOT_APPLICABLE,
        CellStatus.NOT_ATTEMPTED_GATE,
    }
)


class FinalLevel(StrEnum):
    """Values of ``ct_final_level``; the species order is ``FINAL_LEVELS``."""

    NONE = "none"
    LINEAGE = "lineage"
    BROAD = "broad"
    CLASS = "class"
    NT = "nt"
    SUPERCLUSTER = "supercluster"
    SUBCLASS = "subclass"


GateLevel = Literal["full", "broad_only", "failed"]
GATE_LEVELS: Final[tuple[str, ...]] = ("full", "broad_only", "failed")
PanelTrust = Literal["refused", "broad_only", "provisional", "validated"]
PANEL_TRUST_STATES: Final[tuple[str, ...]] = (
    "refused",
    "broad_only",
    "provisional",
    "validated",
)
ValidationBasis = Literal["real_data", "simulation"]
VALIDATION_BASES: Final[tuple[str, ...]] = ("real_data", "simulation")
ReferenceRole = Literal[
    "primary",
    "secondary",
    "likelihood",
    "sensitivity",
    "resolvability",
    "region_share",
]
REFERENCE_ROLES: Final[tuple[str, ...]] = (
    "primary",
    "secondary",
    "likelihood",
    "sensitivity",
    "resolvability",
    "region_share",
)
# Resolved panel mode of a pair (the config's "auto" resolves to one of these;
# unpaired sections use their own panel).
PanelMode = Literal["intersection", "per_platform", "single_sample"]
PANEL_MODES: Final[tuple[str, ...]] = ("intersection", "per_platform", "single_sample")
SUBCLUSTER_STATUSES: Final[tuple[str, ...]] = (
    "mapped_supercluster",
    "mapped_subclass",
    "unresolved",
    "not_resolvable_panel",
    "dataset_broad_only",
    "unassigned",
)

# Levels per species (§4.1). The report-only fine levels (OD-E4) are present
# only when ``allow_fine_levels`` is set.
LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("lineage", "broad", "nt", "supercluster", "seaad_subclass"),
    "mouse": ("broad", "class", "nt", "subclass"),
}
REPORT_ONLY_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("cluster",),
    "mouse": ("supertype",),
}
LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "raw",
    "conf",
    "corr",
    "runner_up",
    "margin",
    "status",
    "validated",
)

HUMAN_BRANCHES: Final[tuple[str, ...]] = (
    "Neurons/Excitatory",
    "Neurons/Inhibitory",
    "Neurons/unresolved",
    "Astrocytes",
    "Oligodendrocytes",
    "Oligodendrocyte precursors",
    "Oligodendrocyte lineage/unresolved",
    "Microglia",
    "Vascular cells",
    "Fibroblasts",
    UNASSIGNED_LABEL,
)

_SAFE_TOKEN_PATTERN: Final = re.compile(r"[^A-Za-z0-9]+")


def safe_token(value: str) -> str:
    """Return an h5ad/zarr-safe token for a label (no ``/``, no spaces).

    Same rule as the legacy ``clustering_squidpy._safe_token``: lower case,
    runs of other characters become ``_``.

    Args:
        value: Any label, e.g. ``"Astrocytes/Ependymal"``.

    Returns:
        The token, e.g. ``"astrocytes_ependymal"``; ``"value"`` if empty.
    """
    token = _SAFE_TOKEN_PATTERN.sub("_", str(value).strip().lower()).strip("_")
    return token or "value"


def label_table_filename(sample_id: str) -> str:
    """Return the label table file name of a sample.

    Args:
        sample_id: Sample id (``<sid>``).

    Returns:
        ``"<sid>_celltype_labels.parquet"``.
    """
    return f"{sample_id}{LABEL_TABLE_SUFFIX}"


class Columns:
    """Column names of the label table (plan §4.1, §4.3)."""

    CELL_ID: Final = "cell_id"
    INSTANCE_ID: Final = "instance_id"
    PAIR_ID: Final = "pair_id"
    SAMPLE_ID: Final = "sample_id"
    PLATFORM: Final = "platform"
    SEGMENTATION: Final = "segmentation"
    SPECIES: Final = "species"
    ANATOMICAL_REGION: Final = "anatomical_region"
    PANEL_HASH: Final = "panel_hash"
    TOTAL_COUNTS: Final = "total_counts"
    N_GENES: Final = "n_genes"
    GENES_PER_COUNT: Final = "genes_per_count"
    IN_TABLE: Final = "in_table"
    DEPTH_BIN: Final = "depth_bin"
    RESOLVABILITY_EXTRAPOLATED: Final = "resolvability_extrapolated"
    N_MISSING_PANEL_GENES: Final = "n_missing_panel_genes"

    CT_FINAL_LEVEL: Final = "ct_final_level"
    CT_FINAL_NAME: Final = "ct_final_name"
    CT_CONSENSUS_TIER: Final = "ct_consensus_tier"
    CT_BRANCH: Final = "ct_branch"
    CT_LEAF: Final = "ct_leaf"
    CT_MENDER_STATE: Final = "ct_mender_state"

    FLAG_LOW_COUNTS: Final = "flag_low_counts"
    FLAG_BELOW_FLOOR: Final = "flag_below_floor"
    FLAG_METHOD_DISAGREE: Final = "flag_method_disagree"
    FLAG_IMPLAUSIBLE: Final = "flag_implausible"
    FLAG_COP_SUPPRESSED: Final = "flag_cop_suppressed"
    CONTAMINATION_SCORE: Final = "contamination_score"
    NEG_COUNTS: Final = "neg_counts"
    FLAG_CONTAMINATED: Final = "flag_contaminated"
    EXPECTED_GENES_Q95: Final = "expected_genes_q95"
    FLAG_DIFFUSE_PROFILE: Final = "flag_diffuse_profile"
    OOD_Z: Final = "ood_z"
    FLAG_OOD: Final = "flag_ood"
    FLAG_REGION_INCOHERENT: Final = "flag_region_incoherent"
    REGION_COHERENCE: Final = "region_coherence"
    MICROGLIA_STAT: Final = "microglia_stat"
    MICROGLIA_WEIGHT: Final = "microglia_weight"
    FLAG_MICROGLIAL_SPILLOVER: Final = "flag_microglial_spillover"
    FLAG_ASTRO_LOWCOUNT: Final = "flag_astro_lowcount"
    EXCLUDE_HARD: Final = "exclude_hard"
    DISCOVERY_CAUTION: Final = "discovery_caution"

    SOFT_BROAD_PREFIX: Final = "soft_broad_"
    SOFT_CLASS_PREFIX: Final = "soft_class_"
    SOFT_BROAD_UNALLOCATED: Final = "soft_broad_unallocated"
    MMC_PREFIX: Final = "mmc_"
    LL_PREFIX: Final = "ll_"

    @staticmethod
    def level(level: str, field: str) -> str:
        """Return a per-level column name, e.g. ``ct_broad_status``.

        Args:
            level: Annotation level, e.g. ``"broad"``.
            field: One of ``LEVEL_FIELDS``.

        Returns:
            ``f"ct_{level}_{field}"``.

        Raises:
            ValueError: If ``field`` is not a per-level field.
        """
        if field not in LEVEL_FIELDS:
            raise ValueError(f"field must be one of {LEVEL_FIELDS}, got {field!r}")
        return f"ct_{level}_{field}"


ColumnKind = Literal[
    "string",
    "category",
    "int64",
    "int32",
    "int8",
    "nullable_int32",
    "float32",
    "bool",
    "nullable_bool",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Expected dtype and value constraints of one label-table column.

    Attributes:
        name: Column name.
        kind: Expected dtype family (``ColumnKind``).
        nullable: Whether missing values are allowed.
        categories: Allowed values, or ``None`` for any value.
        value_range: Inclusive ``(low, high)`` for numeric columns, NaN
            allowed when ``nullable``.
    """

    name: str
    kind: ColumnKind
    nullable: bool = False
    categories: tuple[str, ...] | None = None
    value_range: tuple[float, float] | None = None


def branch_categories(species: Species) -> tuple[str, ...]:
    """Return the allowed ``ct_branch`` values of a species.

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Human: the §4.1 branch list. Mouse: the WMB classes and
        ``UNASSIGNED_LABEL``.
    """
    if species == "human":
        return HUMAN_BRANCHES
    return (*load_vocab("wmb_class").names, UNASSIGNED_LABEL)


def leaf_categories(species: Species) -> tuple[str, ...] | None:
    """Return the allowed ``ct_leaf`` values, or ``None`` when open-ended.

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Human: WHB superclusters and ``"unresolved"``. Mouse: ``None`` (WMB
        subclasses are not packaged; any non-null value is accepted).
    """
    if species == "human":
        return (*load_vocab("whb_supercluster").names, UNRESOLVED_LABEL)
    return None


def soft_columns(species: Species) -> tuple[str, ...]:
    """Return the soft-composition columns of a species (§4.1, §5.5).

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Human: ``soft_broad_<token>`` per broad class plus
        ``soft_broad_unallocated``. Mouse: ``soft_class_<token>`` per WMB class.
    """
    if species == "human":
        tokens = [safe_token(name) for name in HUMAN_BROAD_CLASSES]
        return (
            *(f"{Columns.SOFT_BROAD_PREFIX}{token}" for token in tokens),
            Columns.SOFT_BROAD_UNALLOCATED,
        )
    return tuple(
        f"{Columns.SOFT_CLASS_PREFIX}{safe_token(name)}"
        for name in load_vocab("wmb_class").names
    )


def _level_specs(level: str) -> dict[str, ColumnSpec]:
    probability = (0.0, 1.0)
    specs = [
        ColumnSpec(Columns.level(level, "name"), "category", nullable=True),
        ColumnSpec(
            Columns.level(level, "raw"), "float32", True, value_range=probability
        ),
        ColumnSpec(
            Columns.level(level, "conf"), "float32", True, value_range=probability
        ),
        ColumnSpec(
            Columns.level(level, "corr"), "float32", True, value_range=(-1.0, 1.0)
        ),
        ColumnSpec(Columns.level(level, "runner_up"), "category", nullable=True),
        ColumnSpec(Columns.level(level, "margin"), "float32", nullable=True),
        ColumnSpec(
            Columns.level(level, "status"),
            "category",
            categories=tuple(status.value for status in CellStatus),
        ),
        ColumnSpec(Columns.level(level, "validated"), "bool"),
    ]
    return {spec.name: spec for spec in specs}


def column_specs(
    species: Species, *, include_fine_levels: bool = False
) -> dict[str, ColumnSpec]:
    """Return the required columns of a species' label table.

    Soft-composition, raw engine (``mmc_*``) and v1.1 (``ll_*``) columns are
    optional and not listed; ``validate_label_table`` checks soft columns when
    present.

    Args:
        species: ``"human"`` or ``"mouse"``.
        include_fine_levels: Also require the report-only fine level
            (``cluster`` / ``supertype``), as when ``allow_fine_levels`` is set.

    Returns:
        Column name to spec, in table order.

    Raises:
        ValueError: If the species is unknown.
    """
    if species not in SPECIES:
        raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
    counts = (0.0, float(np.iinfo(np.int32).max))
    identity = [
        ColumnSpec(Columns.CELL_ID, "string"),
        ColumnSpec(Columns.INSTANCE_ID, "int64"),
        ColumnSpec(Columns.PAIR_ID, "category"),
        ColumnSpec(Columns.SAMPLE_ID, "category"),
        ColumnSpec(Columns.PLATFORM, "category", categories=PLATFORMS),
        ColumnSpec(Columns.SEGMENTATION, "category"),
        ColumnSpec(Columns.SPECIES, "category", categories=(species,)),
        ColumnSpec(Columns.ANATOMICAL_REGION, "category", nullable=True),
        ColumnSpec(Columns.PANEL_HASH, "category"),
        ColumnSpec(Columns.TOTAL_COUNTS, "int32", value_range=counts),
        ColumnSpec(Columns.N_GENES, "int32", value_range=counts),
        ColumnSpec(Columns.GENES_PER_COUNT, "float32", True, value_range=(0.0, 1.0)),
        ColumnSpec(Columns.IN_TABLE, "bool"),
        ColumnSpec(Columns.DEPTH_BIN, "nullable_int32", True, value_range=counts),
        ColumnSpec(Columns.RESOLVABILITY_EXTRAPOLATED, "bool"),
        ColumnSpec(Columns.N_MISSING_PANEL_GENES, "int32", value_range=counts),
    ]
    levels = list(LEVELS[species])
    if include_fine_levels:
        levels += list(REPORT_ONLY_LEVELS[species])
    per_level: list[ColumnSpec] = []
    for level in levels:
        per_level += list(_level_specs(level).values())
    mender_states = branch_categories(species)
    final = [
        ColumnSpec(
            Columns.CT_FINAL_LEVEL, "category", categories=FINAL_LEVELS[species]
        ),
        ColumnSpec(Columns.CT_FINAL_NAME, "category"),
        ColumnSpec(Columns.CT_CONSENSUS_TIER, "int8", value_range=(0.0, 3.0)),
        ColumnSpec(
            Columns.CT_BRANCH, "category", categories=branch_categories(species)
        ),
        ColumnSpec(Columns.CT_LEAF, "category", categories=leaf_categories(species)),
        ColumnSpec(Columns.CT_MENDER_STATE, "category", categories=mender_states),
    ]
    score = (0.0, 1.0)
    flags = [
        ColumnSpec(Columns.FLAG_LOW_COUNTS, "bool"),
        ColumnSpec(Columns.FLAG_BELOW_FLOOR, "bool"),
        ColumnSpec(Columns.FLAG_METHOD_DISAGREE, "bool"),
        ColumnSpec(Columns.FLAG_IMPLAUSIBLE, "bool"),
        ColumnSpec(Columns.CONTAMINATION_SCORE, "float32", True, value_range=score),
        ColumnSpec(Columns.NEG_COUNTS, "nullable_int32", True, value_range=counts),
        ColumnSpec(Columns.FLAG_CONTAMINATED, "nullable_bool", nullable=True),
        ColumnSpec(Columns.EXPECTED_GENES_Q95, "float32", nullable=True),
        ColumnSpec(Columns.FLAG_DIFFUSE_PROFILE, "nullable_bool", nullable=True),
        ColumnSpec(Columns.OOD_Z, "float32", nullable=True),
        ColumnSpec(Columns.FLAG_OOD, "nullable_bool", nullable=True),
        ColumnSpec(Columns.MICROGLIA_STAT, "float32", nullable=True),
        ColumnSpec(Columns.MICROGLIA_WEIGHT, "float32", nullable=True),
        ColumnSpec(Columns.FLAG_MICROGLIAL_SPILLOVER, "nullable_bool", nullable=True),
        ColumnSpec(Columns.EXCLUDE_HARD, "bool"),
        ColumnSpec(Columns.DISCOVERY_CAUTION, "bool"),
    ]
    if species == "human":
        flags.append(ColumnSpec(Columns.FLAG_COP_SUPPRESSED, "bool"))
    else:
        flags += [
            ColumnSpec(Columns.FLAG_REGION_INCOHERENT, "nullable_bool", nullable=True),
            ColumnSpec(Columns.REGION_COHERENCE, "float32", True, value_range=score),
            ColumnSpec(Columns.FLAG_ASTRO_LOWCOUNT, "bool"),
        ]
    return {spec.name: spec for spec in [*identity, *per_level, *final, *flags]}


class LabelTableError(ValueError):
    """A label table violates the contract.

    Attributes:
        problems: One message per violation.
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        summary = "; ".join(self.problems[:20])
        more = len(self.problems) - 20
        if more > 0:
            summary += f"; ... and {more} more"
        super().__init__(f"label table violates the contract: {summary}")


_INTEGER_DTYPES: Final[dict[str, str]] = {
    "int64": "int64",
    "int32": "int32",
    "int8": "int8",
}


def _dtype_problem(series: pd.Series, spec: ColumnSpec) -> str | None:
    dtype = series.dtype
    kind = spec.kind
    ok: bool
    if kind == "string":
        ok = pd.api.types.is_string_dtype(series) and not isinstance(
            dtype, pd.CategoricalDtype
        )
    elif kind == "category":
        # An all-null categorical has no categories, and parquet reads it back
        # as an object (null-typed) column, so accept that for nullable columns.
        ok = isinstance(dtype, pd.CategoricalDtype) or (
            spec.nullable and bool(series.isna().all())
        )
    elif kind in _INTEGER_DTYPES:
        ok = dtype == np.dtype(_INTEGER_DTYPES[kind])
    elif kind == "nullable_int32":
        ok = dtype == np.dtype("int32") or dtype == pd.Int32Dtype()
    elif kind == "float32":
        ok = dtype == np.dtype("float32")
    elif kind == "bool":
        ok = dtype == np.dtype("bool")
    else:
        ok = dtype == np.dtype("bool") or dtype == pd.BooleanDtype()
    if ok:
        return None
    return f"{spec.name}: dtype {dtype} is not {kind}"


def _value_problems(series: pd.Series, spec: ColumnSpec) -> list[str]:
    problems = []
    missing = series.isna()
    if not spec.nullable and bool(missing.any()):
        problems.append(f"{spec.name}: {int(missing.sum())} missing values")
    present = series[~missing]
    if spec.categories is not None and len(present):
        unknown = sorted(set(map(str, present.unique())) - set(spec.categories))
        if unknown:
            problems.append(
                f"{spec.name}: values outside the vocabulary {unknown[:10]}"
            )
    if spec.value_range is not None and len(present):
        values = present.to_numpy(dtype=float)
        low, high = spec.value_range
        outside = int(((values < low) | (values > high)).sum())
        if outside:
            problems.append(f"{spec.name}: {outside} values outside [{low}, {high}]")
    return problems


def _consistency_problems(
    df: pd.DataFrame, species: Species, levels: Iterable[str]
) -> list[str]:
    problems = []
    in_table = df[Columns.IN_TABLE].to_numpy(dtype=bool)
    flag_low = df[Columns.FLAG_LOW_COUNTS].to_numpy(dtype=bool)
    if bool((flag_low == in_table).any()):
        problems.append("flag_low_counts must equal ~in_table")
    for level in levels:
        status = df[Columns.level(level, "status")].astype(str).to_numpy()
        is_low = status == CellStatus.LOW_COUNTS
        if bool((is_low & in_table).any()):
            problems.append(f"{level}: low_counts status on table cells")
        if bool((~is_low & ~in_table).any()):
            problems.append(f"{level}: objects outside the table must be low_counts")
        confident = status == CellStatus.CONFIDENT
        validated = df[Columns.level(level, "validated")].to_numpy(dtype=bool)
        if bool((validated & ~confident).any()):
            problems.append(f"{level}: validated is true on non-confident cells")
        nameless = np.isin(status, list(NAMELESS_STATUSES))
        named = df[Columns.level(level, "name")].notna().to_numpy()
        if bool((nameless & named).any()):
            problems.append(
                f"{level}: a name is set where the status is "
                "low_counts / not_applicable / not_attempted_gate"
            )
    final_level = df[Columns.CT_FINAL_LEVEL].astype(str).to_numpy()
    final_name = df[Columns.CT_FINAL_NAME].astype(str).to_numpy()
    is_none = final_level == FinalLevel.NONE
    if bool((is_none & (final_name != UNASSIGNED_LABEL)).any()):
        problems.append(
            f"ct_final_name must be {UNASSIGNED_LABEL!r} when level is none"
        )
    for level in FINAL_LEVELS[species][1:]:
        rows = final_level == level
        if not rows.any():
            continue
        status = df[Columns.level(level, "status")].astype(str).to_numpy()[rows]
        if bool((status != CellStatus.CONFIDENT).any()):
            problems.append(f"ct_final_level {level!r} on cells not confident there")
        names = df[Columns.level(level, "name")].astype(object).to_numpy()[rows]
        if bool((names != final_name[rows]).any()):
            problems.append(f"ct_final_name differs from ct_{level}_name")
    problems += _final_chain_problems(df, species, final_level)
    return problems


def _final_chain_problems(
    df: pd.DataFrame, species: Species, final_level: np.ndarray
) -> list[str]:
    """Check ``ct_final_level`` against the statuses along the level chain.

    ``ct_final_level`` is the deepest confident level (§4.1), and a level is
    confident only when every applicable coarser level is (§4.2: otherwise
    ``parent_unresolved``). So every applicable level coarser than the final
    level must be confident, and no deeper level may be. ``not_applicable``
    levels (``nt`` for non-neurons) are skipped.
    """
    problems = []
    chain = FINAL_LEVELS[species]
    final_rank = np.array(
        [chain.index(level) if level in chain else -1 for level in final_level]
    )
    for rank, level in enumerate(chain[1:], start=1):
        status = df[Columns.level(level, "status")].astype(str).to_numpy()
        confident = status == CellStatus.CONFIDENT
        applicable = status != CellStatus.NOT_APPLICABLE
        coarser = rank < final_rank
        if bool((coarser & applicable & ~confident).any()):
            problems.append(
                f"ct_final_level is deeper than {level!r} on cells not confident at "
                f"{level!r} (a confident level needs confident parents)"
            )
        if bool(((rank > final_rank) & confident).any()):
            problems.append(
                f"ct_final_level is coarser than {level!r} on cells confident at "
                f"{level!r} (the final level is the deepest confident level)"
            )
    return problems


def _soft_problems(df: pd.DataFrame) -> list[str]:
    problems = []
    for column in df.columns:
        if not str(column).startswith(
            (Columns.SOFT_BROAD_PREFIX, Columns.SOFT_CLASS_PREFIX)
        ):
            continue
        spec = ColumnSpec(str(column), "float32", True, value_range=(0.0, 1.0))
        dtype_problem = _dtype_problem(df[column], spec)
        if dtype_problem:
            problems.append(dtype_problem)
        else:
            problems += _value_problems(df[column], spec)
    return problems


def validate_label_table(
    df: pd.DataFrame,
    species: Species,
    h5ad_index: Iterable[str] | pd.Index | None = None,
    *,
    include_fine_levels: bool | None = None,
) -> None:
    """Check a label table against the §4.1–§4.3 contract.

    Checks: required columns and their dtypes; vocabularies (statuses,
    platforms, final levels, branches, leaves, MENDER states); value ranges;
    unique non-null ``cell_id``; ``flag_low_counts == ~in_table``;
    ``low_counts`` exactly outside the table; ``ct_<L>_validated`` only on
    confident cells; no names where the level was not attempted or does not
    apply; ``ct_final_level`` the deepest confident level, with every
    applicable coarser level confident, and ``ct_final_name`` its name;
    soft columns (if present) float32 in [0, 1]; and, when ``h5ad_index`` is
    given, that the ``in_table`` ids equal the clustered H5AD index (same ids,
    any order).

    Args:
        df: The label table.
        species: ``"human"`` or ``"mouse"``.
        h5ad_index: ``obs_names`` of the clustered H5AD, or ``None`` to skip
            the membership check.
        include_fine_levels: Require the report-only fine level. ``None``
            requires it when its status column is present.

    Raises:
        LabelTableError: Listing every violation found.
    """
    specs = column_specs(species, include_fine_levels=False)
    fine = REPORT_ONLY_LEVELS[species]
    if include_fine_levels is None:
        include_fine_levels = all(
            Columns.level(level, "status") in df.columns for level in fine
        )
    if include_fine_levels:
        specs = column_specs(species, include_fine_levels=True)
    problems: list[str] = []
    missing = [name for name in specs if name not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    for name, spec in specs.items():
        if name not in df.columns:
            continue
        dtype_problem = _dtype_problem(df[name], spec)
        if dtype_problem:
            problems.append(dtype_problem)
            n_missing = int(df[name].isna().sum())
            if n_missing and not spec.nullable:
                problems.append(f"{name}: {n_missing} missing values")
            continue
        problems += _value_problems(df[name], spec)
    problems += _soft_problems(df)
    if Columns.CELL_ID in df.columns:
        duplicated = df[Columns.CELL_ID][df[Columns.CELL_ID].duplicated()]
        if len(duplicated):
            problems.append(
                f"cell_id: {len(duplicated)} duplicated ids, e.g. "
                f"{duplicated.astype(str).tolist()[:5]}"
            )
    if not problems:
        levels = list(LEVELS[species]) + (list(fine) if include_fine_levels else [])
        problems += _consistency_problems(df, species, levels)
    if h5ad_index is not None and Columns.CELL_ID in df.columns and not problems:
        problems += _membership_problems(df, h5ad_index)
    if problems:
        raise LabelTableError(problems)


def _membership_problems(
    df: pd.DataFrame, h5ad_index: Iterable[str] | pd.Index
) -> list[str]:
    index = pd.Index([str(value) for value in h5ad_index])
    problems = []
    if index.has_duplicates:
        problems.append("the clustered H5AD index has duplicated ids")
    in_table = df.loc[df[Columns.IN_TABLE].to_numpy(dtype=bool), Columns.CELL_ID]
    table_ids = set(in_table.astype(str))
    h5ad_ids = set(index)
    only_table = sorted(table_ids - h5ad_ids)
    only_h5ad = sorted(h5ad_ids - table_ids)
    if only_table:
        problems.append(
            f"{len(only_table)} in_table ids missing from the H5AD, e.g. "
            f"{only_table[:5]}"
        )
    if only_h5ad:
        problems.append(
            f"{len(only_h5ad)} H5AD ids not in_table in the label table, e.g. "
            f"{only_h5ad[:5]}"
        )
    return problems


def coerce_label_table_dtypes(
    df: pd.DataFrame, species: Species, *, include_fine_levels: bool = False
) -> pd.DataFrame:
    """Cast a label table's columns to the contract dtypes (copy).

    Useful for writers that build columns with default dtypes; values are not
    checked (run ``validate_label_table`` afterwards). Missing values survive
    the cast, so validation still sees them: string columns cast only their
    non-null values, and a non-nullable integer, float or bool column with
    missing values is left uncast (a cast would fail or turn NaN into
    ``True``), which ``validate_label_table`` then reports.

    Args:
        df: The label table.
        species: ``"human"`` or ``"mouse"``.
        include_fine_levels: Also cast the report-only fine level.

    Returns:
        A copy with every present contract column cast.
    """
    out = df.copy()
    specs: Mapping[str, ColumnSpec] = column_specs(
        species, include_fine_levels=include_fine_levels
    )
    for name, spec in specs.items():
        if name not in out.columns:
            continue
        column = out[name]
        has_missing = bool(column.isna().any())
        if spec.kind == "string":
            as_text = column.astype(object).map(str)
            out[name] = as_text.where(column.notna(), None).astype(object)
        elif spec.kind == "category":
            out[name] = column.astype("category")
        elif spec.kind == "nullable_int32":
            out[name] = column.astype(pd.Int32Dtype())
        elif spec.kind == "nullable_bool":
            out[name] = column.astype(pd.BooleanDtype())
        elif has_missing and not spec.nullable:
            logger.debug("%s has missing values; left uncast for validation", name)
        else:
            out[name] = column.astype(spec.kind)
    for column in out.columns:
        if str(column).startswith(
            (Columns.SOFT_BROAD_PREFIX, Columns.SOFT_CLASS_PREFIX)
        ):
            out[column] = out[column].astype("float32")
    return out
