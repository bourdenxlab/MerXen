"""Shadow evaluation of the annotation MAP outputs (plan §12 M3, §5.2-§5.5, §14).

The M3 shadow programme scores the standalone ``merxen annotate`` outputs on
published datasets before any default changes. This module holds the
reusable pieces; ``scripts/acceptance/shadow_baselines.py`` runs them over the
human datasets and writes the baselines that
``docs/acceptance/annotation-v1-preregistration.md`` records.

- **Composition** (§5.5): soft (the assigned WHB supercluster's bootstrap
  probability plus its runner-ups', aggregated to the seven broad classes
  through the vocab; the residual is ``unallocated``), argmax and
  confident-only broad compositions; the Jensen-Shannon distance (base 2,
  as ``scipy.spatial.distance.jensenshannon`` and E1 ``real_jsd.csv``) of
  the renormalised 7-class vectors; spatial block-bootstrap CIs over square
  tiles of one grid shared by the co-registered sections, resampled jointly
  (the same tile-location weights for both sections).
- **Human rules** (§5.2 rules 1, 2 and 4; §5.4): a shadow evaluation of the
  v1 lineage, broad and supercluster rules on raw thresholds with the
  packaged floors, the COP rule, the implausible fallback, a choice of second
  vote (none, SEA-AD below 60 counts only, the full v1 SEA-AD rule, or E2's
  likelihood-typer rule) and the dataset gate. It has no resolvability
  (M3b) and no flags; RESOLVE (M4) replaces it, and M4's exit compares its
  coverage with these baselines.
- **SEA-AD broad calls**: the 7-class label from the vocab ("VLMC &
  Perivascular" split by supertype) and the aggregated broad probability,
  on E2's definition (class bp for neurons, class bp x subclass mass
  otherwise; the v1 ``seaad_broad`` threshold's basis) or E1's
  (``e1lib.load_mmc_seaad``, subclass mass only).
- **Marker referee** (E1 ``09_marker_referee.py``): for cells where two
  labellings disagree on two of the seven classes, the label whose canonical
  panel markers hold the larger fraction of the cell's counts wins.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.composition import (
    BOOTSTRAP_SEED,
    COMPOSITION_COLUMNS,
    N_BOOTSTRAP,
    Resampling,
    _bootstrap_weights,
    jensen_shannon_distance,
    soft_broad_matrix,
    whb_broad_of,
)
from merxen.annotation.composition import (
    TILE_UM as TILE_UM,
)
from merxen.annotation.composition import (
    UNALLOCATED as UNALLOCATED,
)
from merxen.annotation.composition import (
    BootstrapJsd as BootstrapJsd,
)
from merxen.annotation.composition import (
    block_bootstrap_jsd as block_bootstrap_jsd,
)
from merxen.annotation.composition import (
    broad_class_index as broad_class_index,
)
from merxen.annotation.composition import (
    composition_shares as composition_shares,
)
from merxen.annotation.composition import (
    one_hot_broad_matrix as one_hot_broad_matrix,
)
from merxen.annotation.composition import (
    shared_tile_codes as shared_tile_codes,
)
from merxen.annotation.composition import (
    tile_codes as tile_codes,
)
from merxen.annotation.composition import (
    tile_sums as tile_sums,
)
from merxen.annotation.config import AnnotationGate, AnnotationThresholds
from merxen.annotation.flags import (
    CONTAMINATION_ALPHA as CONTAMINATION_ALPHA,
)
from merxen.annotation.flags import (
    CONTAMINATION_MAX_INFORMATIVE as CONTAMINATION_MAX_INFORMATIVE,
)
from merxen.annotation.flags import (
    CONTAMINATION_MIN_NEGATIVE as CONTAMINATION_MIN_NEGATIVE,
)
from merxen.annotation.flags import (
    CONTAMINATION_MIN_NULL_CELLS as CONTAMINATION_MIN_NULL_CELLS,
)
from merxen.annotation.flags import (
    CONTAMINATION_NULL_QUANTILE as CONTAMINATION_NULL_QUANTILE,
)
from merxen.annotation.flags import (
    DIFFUSE_MAX_INFORMATIVE as DIFFUSE_MAX_INFORMATIVE,
)
from merxen.annotation.flags import (
    DIFFUSE_N_SIMULATIONS as DIFFUSE_N_SIMULATIONS,
)
from merxen.annotation.flags import (
    DIFFUSE_QUANTILE as DIFFUSE_QUANTILE,
)
from merxen.annotation.flags import (
    BetaBinomialFit as BetaBinomialFit,
)
from merxen.annotation.flags import (
    ContaminationFlags as ContaminationFlags,
)
from merxen.annotation.flags import (
    beta_binomial_upper_tail as beta_binomial_upper_tail,
)
from merxen.annotation.flags import (
    class_profiles as class_profiles,
)
from merxen.annotation.flags import (
    contamination_flags as contamination_flags,
)
from merxen.annotation.flags import (
    depth_grid as depth_grid,
)
from merxen.annotation.flags import (
    distinct_gene_quantiles as distinct_gene_quantiles,
)
from merxen.annotation.flags import (
    expected_genes_quantile as expected_genes_quantile,
)
from merxen.annotation.flags import (
    fit_beta_binomial as fit_beta_binomial,
)
from merxen.annotation.flags import (
    negative_counts as negative_counts,
)
from merxen.annotation.flags import (
    negative_gene_mask as negative_gene_mask,
)
from merxen.annotation.flags import (
    realised_rates as realised_rates,
)
from merxen.annotation.schema import meets_threshold
from merxen.annotation.vocab import (
    COP_SUPERCLUSTER,
    HUMAN_BROAD_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    HUMAN_LINEAGES,
    UNASSIGNED_LABEL,
    VocabTable,
    floor_class_for,
    load_floor_table,
    load_vocab,
    primary_vocab,
)

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.panel import AnnotationPanel
    from merxen.annotation.pipeline import SampleQuery

logger = logging.getLogger(__name__)

AGREEMENT_MIN_COUNTS: Final = 20
SEED_PANEL_FAMILY: Final = "human_set_a"
OPC: Final = "Oligodendrocyte precursors"

# Canonical markers of the E1 referee (exp/E1/09_marker_referee.py, MK); a
# class scores the fraction of a cell's counts on its markers in the panel.
E1_REFEREE_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "Neurons": (
        "SLC17A7",
        "GAD1",
        "GAD2",
        "SST",
        "VIP",
        "PVALB",
        "LAMP5",
        "CUX2",
        "RORB",
        "CCK",
        "SYNPR",
    ),
    "Oligodendrocytes": (
        "MOBP",
        "MOG",
        "OPALIN",
        "MAG",
        "ST18",
        "ERMN",
        "KLK6",
        "CLDN11",
        "MAL",
        "UGT8",
    ),
    "Oligodendrocyte precursors": ("PDGFRA", "CSPG4", "VCAN", "PTPRZ1"),
    "Astrocytes": ("AQP4", "GJA1", "SOX9", "FGFR3"),
    "Microglia": (
        "CX3CR1",
        "P2RY12",
        "AIF1",
        "CTSS",
        "GPR34",
        "P2RY13",
        "SPI1",
        "LY86",
    ),
    "Vascular cells": ("FLT1", "PECAM1", "ABCC9", "CALCRL"),
    "Fibroblasts": ("DCN", "FBLN1", "COL12A1"),
}

# E2's likelihood-typer broad scheme (exp/E2/common.py MAP): neurons merged
# (the lenient rule: OtherNeuron counts as a neuron), fibroblasts pooled with
# vascular cells. E2's LL rule compares WHB and LL in this scheme.
E2_MERGED_CLASS: Final[dict[str, str]] = {
    "Exc": "N",
    "Inh": "N",
    "OtherNeuron": "N",
    "Oligo": "Oligo",
    "OPC": "OPC",
    "Astro": "Astro",
    "Immune": "Immune",
    "Vascular": "Vascular",
}
E2_CLASS_OF_HUMAN_BROAD: Final[dict[str, str]] = {
    "Neurons": "N",
    "Oligodendrocytes": "Oligo",
    OPC: "OPC",
    "Astrocytes": "Astro",
    "Microglia": "Immune",
    "Vascular cells": "Vascular",
    "Fibroblasts": "Vascular",
}
E2_LINEAGE_OF_CLASS: Final[dict[str, str]] = {
    "N": "N",
    "Oligo": "OL",
    "OPC": "OL",
    "Astro": "Astro",
    "Immune": "Immune",
    "Vascular": "Vascular",
}

SecondVote = Literal[
    "none", "seaad_from60", "seaad", "ll_below60", "seaad_ll_below60", "seaad_or_ll"
]
Below60Vote = Literal["none", "seaad", "ll", "seaad_or_ll"]
LikelihoodScheme = Literal["e2", "broad7"]
# Second-vote variants: (method that must agree below 60 counts, whether
# SEA-AD also applies its other v1 roles: vetoing from 60 counts when it
# confidently disagrees, the COP rescue and the implausible-lineage rescue).
# ``seaad_or_ll`` is the v1.1 row of the degraded-mode table (§5.3): below 60
# counts SEA-AD **or** the likelihood typer must agree.
SECOND_VOTE_VARIANTS: Final[dict[str, tuple[Below60Vote, bool]]] = {
    "none": ("none", False),
    "seaad_from60": ("none", True),
    "seaad": ("seaad", True),
    "ll_below60": ("ll", False),
    "seaad_ll_below60": ("ll", True),
    "seaad_or_ll": ("seaad_or_ll", True),
}
SECOND_VOTES: Final[tuple[str, ...]] = tuple(SECOND_VOTE_VARIANTS)
GateLevel = Literal["full", "broad_only", "failed"]


# --------------------------------------------------------------------------
# Composition and JSD


def _object_array(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    missing = pd.isna(array)
    if missing.any():
        array = array.copy()
        array[missing] = None
    return array


# Composition, JSD and the spatial block bootstrap live in
# ``merxen.annotation.composition`` (imported above).


# --------------------------------------------------------------------------
# SEA-AD broad calls


def _runner_up_columns(frame: pd.DataFrame, n_runners_up: int) -> list[tuple[str, str]]:
    columns = []
    for rank in range(1, n_runners_up + 1):
        name_column = f"runner_up_{rank}_name"
        probability_column = f"runner_up_{rank}_probability"
        if name_column in frame.columns and probability_column in frame.columns:
            columns.append((name_column, probability_column))
    return columns


def seaad_broad_calls(
    subclass: pd.DataFrame,
    supertype: pd.DataFrame | None = None,
    *,
    class_level: pd.DataFrame | None = None,
    vocab: VocabTable | None = None,
    n_runners_up: int = 5,
) -> pd.DataFrame:
    """Return SEA-AD's 7-class label and aggregated broad probability per cell.

    The label is the assigned subclass's broad class (vocab; "VLMC &
    Perivascular" split by the assigned supertype). The subclass-level mass
    sums the bootstrap probability of the assigned subclass and of the
    runner-up subclasses that count toward the same class
    (``broad_classes_any``); for a split subclass it is multiplied by the
    supertype-level mass of the supertypes that give the same class.

    - **E2 definition** (``class_level`` given; the v1 ``seaad_broad``
      threshold 0.68 was derived on it, ``exp/E2/build_tables.py``): neurons
      take the SEA-AD class-level bootstrap probability; every other class
      takes class bp x the subclass-level mass.
    - **E1 definition** (``class_level=None``; E1 ``load_mmc_seaad``): the
      subclass-level mass alone, with no class-level factor.

    Args:
        subclass: SEA-AD subclass level of a tidy table (``level_frame``).
        supertype: The supertype level (any index order; aligned by cell).
        class_level: The class level (aligned by cell); selects the E2
            definition.
        vocab: The SEA-AD vocab (default: packaged).
        n_runners_up: Runner-ups to aggregate.

    Returns:
        Indexed like ``subclass``: ``subclass``, ``supertype``, ``broad``
        (``Mixed/Unknown`` outside the seven classes) and ``broad_raw``
        (0 outside them).
    """
    table = vocab or load_vocab("seaad_mr_subclass")
    names = _object_array(subclass["name"])
    bp = np.nan_to_num(subclass["bp"].to_numpy(np.float64), nan=0.0)
    aligned = None if supertype is None else supertype.reindex(subclass.index)
    supertype_names = (
        np.full(len(names), None, dtype=object)
        if aligned is None
        else _object_array(aligned["name"])
    )
    label_cache: dict[tuple[object, object], str] = {}
    classes_cache: dict[object, tuple[str, ...]] = {}

    def label_of(name: object, supertype_name: object) -> str:
        key = (name, supertype_name)
        if key not in label_cache:
            known = name is not None and str(name) in table
            label_cache[key] = (
                table.broad_class(
                    str(name), None if supertype_name is None else str(supertype_name)
                )
                if known
                else UNASSIGNED_LABEL
            )
        return label_cache[key]

    def classes_of(name: object) -> tuple[str, ...]:
        if name not in classes_cache:
            known = name is not None and str(name) in table
            classes_cache[name] = table.broad_classes_any(str(name)) if known else ()
        return classes_cache[name]

    broad = np.array(
        [
            label_of(name, supertype_name)
            for name, supertype_name in zip(names, supertype_names, strict=True)
        ],
        dtype=object,
    )
    raw = bp.copy()
    for name_column, probability_column in _runner_up_columns(subclass, n_runners_up):
        runner_names = _object_array(subclass[name_column])
        runner_bp = np.nan_to_num(
            subclass[probability_column].to_numpy(np.float64), nan=0.0
        )
        same_class = np.array(
            [
                cls in classes_of(name)
                for cls, name in zip(broad, runner_names, strict=True)
            ],
            dtype=bool,
        )
        raw += np.where(same_class, runner_bp, 0.0)
    split_names = {
        str(name) for name in table.supertype_overrides.get(table.key_column, [])
    }
    is_split = np.array(
        [name is not None and str(name) in split_names for name in names], dtype=bool
    )
    if aligned is not None and is_split.any():
        supertype_bp = np.nan_to_num(aligned["bp"].to_numpy(np.float64), nan=0.0)
        mass = np.where(
            [
                label_of(name, supertype_name) == cls
                for name, supertype_name, cls in zip(
                    names, supertype_names, broad, strict=True
                )
            ],
            supertype_bp,
            0.0,
        )
        for name_column, probability_column in _runner_up_columns(
            aligned, n_runners_up
        ):
            runner_names = _object_array(aligned[name_column])
            runner_bp = np.nan_to_num(
                aligned[probability_column].to_numpy(np.float64), nan=0.0
            )
            same_class = np.array(
                [
                    runner is not None and label_of(name, runner) == cls
                    for name, runner, cls in zip(
                        names, runner_names, broad, strict=True
                    )
                ],
                dtype=bool,
            )
            mass += np.where(same_class, runner_bp, 0.0)
        raw = np.where(is_split, raw * np.minimum(mass, 1.0), raw)
    if class_level is not None:
        class_bp = np.nan_to_num(
            class_level.reindex(subclass.index)["bp"].to_numpy(np.float64), nan=0.0
        )
        raw = np.where(broad == "Neurons", class_bp, class_bp * np.minimum(raw, 1.0))
    in_classes = np.isin(broad, np.asarray(HUMAN_BROAD_CLASSES, dtype=object))
    raw = np.where(in_classes, np.minimum(raw, 1.0), 0.0)
    return pd.DataFrame(
        {
            "subclass": names,
            "supertype": supertype_names,
            "broad": broad,
            "broad_raw": raw,
        },
        index=subclass.index,
    )


# --------------------------------------------------------------------------
# Human rules (shadow evaluation of §5.2 rules 1, 2, 4 and the §5.4 gate)


@dataclass(frozen=True)
class FloorLookup:
    """Count floors per (level, platform, floor class) (§5.4).

    Attributes:
        floors: ``{(level, platform): {floor_class: min_counts}}``.
        panel_family: Family the floors were read for.
    """

    floors: Mapping[tuple[str, str], Mapping[str, int]]
    panel_family: str = SEED_PANEL_FAMILY

    @classmethod
    def packaged(cls, panel_family: str = SEED_PANEL_FAMILY) -> FloorLookup:
        """Load the packaged human floors of one panel family.

        Args:
            panel_family: ``floors_human.csv`` family.

        Returns:
            The lookup.

        Raises:
            ValueError: If the table has no row for the family.
        """
        table = load_floor_table("human")
        rows = table[table["panel_family"].astype(str) == panel_family]
        if rows.empty:
            raise ValueError(f"floors_human.csv has no rows for {panel_family!r}")
        floors: dict[tuple[str, str], dict[str, int]] = {}
        for row in rows.itertuples(index=False):
            key = (str(row.level), str(row.platform).upper())
            floors.setdefault(key, {})[str(row.floor_class)] = int(row.min_counts)
        return cls(floors=floors, panel_family=panel_family)

    def per_cell(
        self, level: str, platform: str, supercluster: np.ndarray
    ) -> np.ndarray:
        """Return each cell's floor at one level (inf where none applies).

        Args:
            level: ``"broad"`` or ``"supercluster"``.
            platform: ``MERSCOPE`` or ``XENIUM``.
            supercluster: Assigned WHB supercluster names.

        Returns:
            Floors as floats; nodes without a floor class (sinks, nodes
            outside the seven classes) get ``inf`` (never confident).

        Raises:
            KeyError: If the table has no floors for the level and platform.
        """
        key = (level, platform.upper())
        if key not in self.floors:
            raise KeyError(f"no {level} floors for platform {platform!r}")
        table = self.floors[key]
        vocab = primary_vocab("human")
        cache: dict[object, float] = {}
        values = np.empty(len(supercluster), dtype=np.float64)
        for index, name in enumerate(_object_array(supercluster)):
            if name not in cache:
                floor_class = (
                    floor_class_for(str(name), species="human", level=level)
                    if name is not None and str(name) in vocab
                    else None
                )
                cache[name] = (
                    float(table[floor_class])
                    if floor_class is not None and floor_class in table
                    else math.inf
                )
            values[index] = cache[name]
        return values


@dataclass(frozen=True)
class HumanRuleInputs:
    """Per-cell inputs of the shadow human rules (table cells only).

    Attributes:
        total_counts: Counts per cell (after control removal).
        whb_supercluster: Assigned WHB supercluster name.
        whb_supercluster_bp: Its bootstrap probability.
        whb_lineage: Lineage of the assigned node (vocab).
        whb_lineage_raw: Aggregated lineage probability (E2 definition).
        whb_broad: Broad class of the assigned node (``Mixed/Unknown`` for
            sinks and nodes outside the seven classes).
        whb_broad_raw: Aggregated broad probability.
        sea_broad: SEA-AD 7-class label (``seaad_broad_calls``), or ``None``
            when SEA-AD is not available.
        sea_broad_raw: SEA-AD aggregated broad probability.
        ll_broad: Likelihood-typer broad label, for the ``ll`` second-vote
            variants: in E2's scheme (``Exc``, ``Inh``, ``OtherNeuron``,
            ``Oligo``, ``OPC``, ``Astro``, ``Immune``, ``Vascular``) or the
            seven broad classes (``ll_scheme``).
        ll_scheme: ``"e2"`` (E2's typer: neurons merged, fibroblasts pooled
            with vascular cells) or ``"broad7"`` (LL (vii) on the bundle
            profiles: the seven broad classes, compared as SEA-AD is).
    """

    total_counts: np.ndarray
    whb_supercluster: np.ndarray
    whb_supercluster_bp: np.ndarray
    whb_lineage: np.ndarray
    whb_lineage_raw: np.ndarray
    whb_broad: np.ndarray
    whb_broad_raw: np.ndarray
    sea_broad: np.ndarray | None = None
    sea_broad_raw: np.ndarray | None = None
    ll_broad: np.ndarray | None = None
    ll_scheme: LikelihoodScheme = "e2"

    def __len__(self) -> int:
        return len(self.total_counts)


@dataclass(frozen=True)
class DatasetGate:
    """The dataset gate verdict (§5.4).

    Attributes:
        level: ``full``, ``broad_only`` or ``failed``.
        frac_ge30: A, the share of table cells with >= 30 counts.
        broad_coverage_table: Confident broad coverage of table cells.
        broad_coverage_segmented: Coverage of segmented objects (NaN when
            the object count is unknown).
        warning: Whether the warning flag is set.
        reasons: Warning and level reasons.
    """

    level: GateLevel
    frac_ge30: float
    broad_coverage_table: float
    broad_coverage_segmented: float
    warning: bool
    reasons: tuple[str, ...]


def dataset_gate(
    total_counts: np.ndarray,
    broad_confident: np.ndarray,
    *,
    n_segmented: int | None = None,
    gate: AnnotationGate | None = None,
) -> DatasetGate:
    """Return the dataset gate verdict from table-cell counts and coverage.

    Args:
        total_counts: Counts of the table cells.
        broad_confident: Confident broad flag per table cell.
        n_segmented: Segmented objects of the sample (the warning's
            denominator); ``None`` skips the warning.
        gate: Gate settings (default: the pre-registered values).

    Returns:
        The verdict.
    """
    settings = gate or AnnotationGate()
    counts = np.asarray(total_counts, dtype=np.float64)
    confident = np.asarray(broad_confident, dtype=bool)
    n_table = max(len(counts), 1)
    frac = float((counts >= settings.depth_counts).sum() / n_table)
    coverage = float(confident.sum() / n_table)
    segmented = float(confident.sum() / n_segmented) if n_segmented else math.nan
    reasons: list[str] = []
    level: GateLevel = "full"
    if coverage < settings.min_table_broad_coverage:
        level = "failed"
        reasons.append(
            f"broad coverage of table cells {coverage:.3f} < "
            f"{settings.min_table_broad_coverage}"
        )
    elif frac < settings.min_frac_ge30:
        level = "broad_only"
        reasons.append(f"A = {frac:.3f} < {settings.min_frac_ge30}")
    warning = bool(
        math.isfinite(segmented) and segmented < settings.warn_segmented_broad_coverage
    )
    if warning:
        reasons.append(
            f"broad coverage of segmented objects {segmented:.3f} < "
            f"{settings.warn_segmented_broad_coverage}"
        )
    return DatasetGate(
        level=level,
        frac_ge30=frac,
        broad_coverage_table=coverage,
        broad_coverage_segmented=segmented,
        warning=warning,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class HumanRuleResult:
    """Shadow statuses of the human rules, per table cell.

    Attributes:
        second_vote: The second-vote variant applied.
        lineage_confident: Lineage confident.
        broad_confident: Broad confident.
        supercluster_confident: Supercluster confident (after the gate).
        broad_name: Confident broad class (``None`` elsewhere).
        implausible: Sink or region-implausible WHB node.
        cop_suppressed: Lineage-confident WHB COP call that fails the COP
            rule (it stays at lineage; ``flag_cop_suppressed``).
        gate: The dataset gate verdict.
    """

    second_vote: str
    lineage_confident: np.ndarray
    broad_confident: np.ndarray
    supercluster_confident: np.ndarray
    broad_name: np.ndarray
    implausible: np.ndarray
    cop_suppressed: np.ndarray
    gate: DatasetGate

    def final_level(self) -> np.ndarray:
        """Return the deepest confident level (``none`` < ``lineage`` < ...)."""
        level = np.full(len(self.broad_confident), "none", dtype=object)
        level[self.lineage_confident] = "lineage"
        level[self.broad_confident] = "broad"
        level[self.supercluster_confident] = "supercluster"
        return level


def _in(values: np.ndarray, allowed: Sequence[str]) -> np.ndarray:
    return np.isin(_object_array(values), np.asarray(allowed, dtype=object))


def _mapped(values: np.ndarray | None, mapping: Mapping[str, str]) -> np.ndarray:
    if values is None:
        return np.full(0, None, dtype=object)
    return np.array(
        [
            None if value is None else mapping.get(str(value))
            for value in _object_array(values)
        ],
        dtype=object,
    )


def _keep_where(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    kept = np.full(len(values), None, dtype=object)
    kept[mask] = values[mask]
    return kept


def _equal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a is not None and b is not None and a == b
            for a, b in zip(first, second, strict=True)
        ],
        dtype=bool,
    )


def _likelihood_agreement(
    ll_broad: np.ndarray,
    whb_broad: np.ndarray,
    whb_lineage: np.ndarray,
    *,
    scheme: LikelihoodScheme,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lineage, broad) agreement of the likelihood typer with WHB."""
    if scheme == "broad7":
        ll_class = _object_array(ll_broad)
        ll_class = _keep_where(_in(ll_class, HUMAN_BROAD_CLASSES), ll_class)
        return (
            _equal(_mapped(ll_class, HUMAN_LINEAGE_OF_BROAD_CLASS), whb_lineage),
            _equal(ll_class, whb_broad),
        )
    ll_e2 = _mapped(ll_broad, E2_MERGED_CLASS)
    whb_e2 = _mapped(whb_broad, E2_CLASS_OF_HUMAN_BROAD)
    return (
        _equal(
            _mapped(ll_e2, E2_LINEAGE_OF_CLASS), _mapped(whb_e2, E2_LINEAGE_OF_CLASS)
        ),
        _equal(ll_e2, whb_e2),
    )


def evaluate_human_rules(
    inputs: HumanRuleInputs,
    *,
    platform: str,
    second_vote: SecondVote = "seaad",
    thresholds: AnnotationThresholds | None = None,
    floors: FloorLookup | None = None,
    region: str = "frontal_cortex",
    gate: AnnotationGate | None = None,
    n_segmented: int | None = None,
    min_counts: int = 10,
) -> HumanRuleResult:
    """Evaluate the v1 human lineage, broad and supercluster rules (shadow).

    Rules (plan §5.2, raw thresholds, packaged floors, no resolvability):

    1. **Lineage** confident if the aggregated WHB lineage probability is
       >= ``whb_broad``, counts >= the hard floor, the node is plausible (an
       implausible node keeps its lineage when the second vote agrees at
       lineage) and the second vote passes at lineage.
    2. **Broad** needs a confident lineage, a plausible node, aggregated
       broad probability >= ``whb_broad``, counts >= the broad floor of its
       class and platform and the second vote at the 7-class level. **COP:**
       a COP call becomes broad OPC only with >= the COP supercluster floor
       (120) and supercluster probability >= ``whb_supercluster``, or when
       SEA-AD confidently calls OPC; otherwise it stays at lineage
       (``cop_suppressed``).
    4. **Supercluster** needs a confident broad, gate level ``full``,
       probability >= ``whb_supercluster`` and counts >= the supercluster
       floor of its class and platform.

    Second-vote variants (``SECOND_VOTE_VARIANTS``; "below 60" is
    ``second_vote_below_counts``):

    - ``seaad`` (v1): below 60 counts SEA-AD must agree; from 60 SEA-AD must
      not confidently disagree (broad probability >= ``seaad_broad`` on a
      different class); SEA-AD also rescues COP (confident OPC) and the
      lineage of implausible nodes (agreement at lineage).
    - ``seaad_from60``: v1 without the below-60 agreement requirement (its
      coverage cost is ``seaad_from60 - seaad``).
    - ``seaad_ll_below60``: v1 with the likelihood typer, not SEA-AD, as the
      below-60 vote (the OD-B13 comparison).
    - ``ll_below60`` (E2's rule as ``exp/E2/final_coverage.py``): below 60
      counts the likelihood typer must agree; no SEA-AD role.
    - ``seaad_or_ll`` (v1.1, §5.3): v1 with SEA-AD **or** the likelihood
      typer agreeing below 60 counts.
    - ``none``: no second method (implausible nodes are never rescued).

    With ``ll_scheme="e2"`` the likelihood typer agrees in E2's scheme:
    neurons merged (the lenient rule, OtherNeuron counts as a neuron) and
    fibroblasts pooled with vascular cells; with ``"broad7"`` it agrees at the
    seven broad classes and their lineages, as SEA-AD does.

    Args:
        inputs: Per-cell inputs.
        platform: ``MERSCOPE`` or ``XENIUM`` (floors).
        second_vote: Variant.
        thresholds: Thresholds (default: the v1 raw values).
        floors: Floors (default: the packaged seeded set-a floors).
        region: Anatomical region (plausibility column).
        gate: Gate settings (default: pre-registered).
        n_segmented: Segmented objects of the sample (gate warning).
        min_counts: Hard floor when ``thresholds.hard_min_counts`` is unset
            (the clustering ``min_counts``).

    Returns:
        Statuses per cell and the gate verdict.

    Raises:
        ValueError: If the variant's second method is missing.
    """
    if second_vote not in SECOND_VOTES:
        raise ValueError(f"second_vote must be one of {SECOND_VOTES}")
    settings = thresholds or AnnotationThresholds()
    floor_lookup = floors or FloorLookup.packaged()
    vocab = primary_vocab("human")
    counts = np.asarray(inputs.total_counts, dtype=np.float64)
    supercluster = _object_array(inputs.whb_supercluster)
    supercluster_bp = np.nan_to_num(
        np.asarray(inputs.whb_supercluster_bp, dtype=np.float64), nan=-1.0
    )
    lineage = _object_array(inputs.whb_lineage)
    broad = _object_array(inputs.whb_broad)
    lineage_raw = np.nan_to_num(
        np.asarray(inputs.whb_lineage_raw, dtype=np.float64), nan=-1.0
    )
    broad_raw = np.nan_to_num(
        np.asarray(inputs.whb_broad_raw, dtype=np.float64), nan=-1.0
    )
    below = counts < settings.second_vote_below_counts
    hard_floor = (
        settings.hard_min_counts if settings.hard_min_counts is not None else min_counts
    )

    plausibility: dict[object, bool] = {}
    for name in set(supercluster.tolist()):
        known = name is not None and str(name) in vocab
        plausibility[name] = (
            known
            and not vocab.is_sink(str(name))
            and vocab.is_region_plausible(str(name), region)
        )
    implausible = np.array(
        [not plausibility[name] for name in supercluster], dtype=bool
    )

    n_cells = len(counts)
    below60_vote, uses_seaad = SECOND_VOTE_VARIANTS[second_vote]
    agree_lineage = np.ones(n_cells, dtype=bool)
    agree_broad = np.ones(n_cells, dtype=bool)
    rescued = np.zeros(n_cells, dtype=bool)
    disagree_lineage = np.zeros(n_cells, dtype=bool)
    disagree_broad = np.zeros(n_cells, dtype=bool)
    sea_confident_opc = np.zeros(n_cells, dtype=bool)
    if uses_seaad or below60_vote in ("seaad", "seaad_or_ll"):
        if inputs.sea_broad is None or inputs.sea_broad_raw is None:
            raise ValueError(f"second_vote={second_vote!r} needs SEA-AD calls")
        sea_broad = _object_array(inputs.sea_broad)
        sea_in_classes = _in(sea_broad, HUMAN_BROAD_CLASSES)
        sea_broad = _keep_where(sea_in_classes, sea_broad)
        sea_raw = np.nan_to_num(
            np.asarray(inputs.sea_broad_raw, dtype=np.float64), nan=0.0
        )
        sea_confident = meets_threshold(sea_raw, settings.seaad_broad) & sea_in_classes
        sea_agree_lineage = _equal(
            _mapped(sea_broad, HUMAN_LINEAGE_OF_BROAD_CLASS), lineage
        )
        sea_agree_broad = _equal(sea_broad, broad)
        if uses_seaad:
            disagree_lineage = sea_confident & ~sea_agree_lineage
            disagree_broad = sea_confident & ~sea_agree_broad
            sea_confident_opc = sea_confident & (sea_broad == OPC)
            rescued = sea_agree_lineage
        if below60_vote in ("seaad", "seaad_or_ll"):
            agree_lineage, agree_broad = sea_agree_lineage, sea_agree_broad
    if below60_vote in ("ll", "seaad_or_ll"):
        if inputs.ll_broad is None:
            raise ValueError(
                f"second_vote={second_vote!r} needs likelihood-typer calls"
            )
        ll_lineage, ll_broad = _likelihood_agreement(
            inputs.ll_broad, broad, lineage, scheme=inputs.ll_scheme
        )
        if below60_vote == "ll":
            agree_lineage, agree_broad = ll_lineage, ll_broad
        else:
            agree_lineage = agree_lineage | ll_lineage
            agree_broad = agree_broad | ll_broad
        if not uses_seaad:
            rescued = agree_lineage
    vote_lineage = np.where(below, agree_lineage, ~disagree_lineage)
    vote_broad = np.where(below, agree_broad, ~disagree_broad)

    has_lineage = _in(lineage, HUMAN_LINEAGES)
    lineage_confident = (
        has_lineage
        & meets_threshold(lineage_raw, settings.whb_broad)
        & (counts >= hard_floor)
        & (~implausible | rescued)
        & vote_lineage
    )
    broad_floor = floor_lookup.per_cell("broad", platform, supercluster)
    broad_candidate = (
        lineage_confident
        & ~implausible
        & _in(broad, HUMAN_BROAD_CLASSES)
        & meets_threshold(broad_raw, settings.whb_broad)
        & (counts >= broad_floor)
        & vote_broad
    )
    supercluster_floor = floor_lookup.per_cell("supercluster", platform, supercluster)
    is_cop = supercluster == COP_SUPERCLUSTER
    cop_passes = (
        (counts >= supercluster_floor)
        & meets_threshold(supercluster_bp, settings.whb_supercluster)
    ) | sea_confident_opc
    cop_suppressed = lineage_confident & is_cop & ~cop_passes
    broad_confident = broad_candidate & ~(is_cop & ~cop_passes)
    verdict = dataset_gate(counts, broad_confident, n_segmented=n_segmented, gate=gate)
    supercluster_confident = (
        broad_confident
        & (verdict.level == "full")
        & meets_threshold(supercluster_bp, settings.whb_supercluster)
        & (counts >= supercluster_floor)
    )
    broad_name = _keep_where(broad_confident, broad)
    return HumanRuleResult(
        second_vote=second_vote,
        lineage_confident=lineage_confident,
        broad_confident=broad_confident,
        supercluster_confident=supercluster_confident,
        broad_name=broad_name,
        implausible=implausible,
        cop_suppressed=cop_suppressed,
        gate=verdict,
    )


# --------------------------------------------------------------------------
# Agreement and the marker referee


def label_agreement(
    first: Sequence[object] | np.ndarray | pd.Series,
    second: Sequence[object] | np.ndarray | pd.Series,
    mask: np.ndarray | None = None,
    *,
    strict: bool = False,
) -> tuple[float, int]:
    """Return the share of cells whose two labels are equal.

    Args:
        first: Labels of the first method.
        second: Labels of the second method.
        mask: Cells to score (default: all).
        strict: Count a cell whose label is outside the seven broad classes
            as a disagreement (E1 ``real_agreement.csv`` counts two
            out-of-class labels as agreeing, the default here).

    Returns:
        ``(agreement, n_cells)``; agreement is NaN without cells.
    """
    a = _object_array(first)
    b = _object_array(second)
    keep = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    a_norm = np.where(_in(a, HUMAN_BROAD_CLASSES), a, UNASSIGNED_LABEL)
    b_norm = np.where(_in(b, HUMAN_BROAD_CLASSES), b, UNASSIGNED_LABEL)
    same = a_norm == b_norm
    if strict:
        same &= a_norm != UNASSIGNED_LABEL
    n_cells = int(keep.sum())
    return (float(same[keep].mean()) if n_cells else math.nan), n_cells


def marker_class_scores(
    counts: sparse.spmatrix | np.ndarray,
    gene_symbols: Sequence[str],
    markers: Mapping[str, Sequence[str]] = E1_REFEREE_MARKERS,
) -> tuple[np.ndarray, dict[str, tuple[str, ...]]]:
    """Return each cell's fraction of counts on each class's canonical markers.

    Args:
        counts: Cells x genes raw counts.
        gene_symbols: Gene symbol per column.
        markers: Class to canonical marker symbols (default: E1's referee
            list); markers absent from the panel are ignored.

    Returns:
        ``(scores, present)``: ``(n_cells, n_classes)`` fractions in
        ``markers`` order, and the markers present per class.
    """
    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts)
    symbols = np.asarray([str(symbol) for symbol in gene_symbols], dtype=object)
    totals = np.maximum(np.asarray(matrix.sum(axis=1)).ravel(), 1.0)
    scores = np.zeros((matrix.shape[0], len(markers)), dtype=np.float64)
    present: dict[str, tuple[str, ...]] = {}
    for column, (cls, genes) in enumerate(markers.items()):
        selected = np.isin(symbols, np.asarray(list(genes), dtype=object))
        present[cls] = tuple(str(symbol) for symbol in symbols[selected])
        if selected.any():
            scores[:, column] = (
                np.asarray(matrix[:, selected].sum(axis=1)).ravel() / totals
            )
    return scores, present


@dataclass(frozen=True)
class RefereeResult:
    """Marker-referee outcome of one labelling pair (E1 method).

    Attributes:
        n_cells: Cells scored.
        n_disputes: Cells where both labels are in the seven classes and differ.
        first_wins: Share of disputes whose first label has the higher score.
        second_wins: Share won by the second label.
        undecided: Share of ties.
        consensus_top_marker_ok: Share of agreeing cells whose label has the
            top marker score (sanity).
        top_disputes: The most frequent ``"first | second"`` disputes.
    """

    n_cells: int
    n_disputes: int
    first_wins: float
    second_wins: float
    undecided: float
    consensus_top_marker_ok: float
    top_disputes: dict[str, int]


def marker_referee(
    scores: np.ndarray,
    first: Sequence[object] | np.ndarray | pd.Series,
    second: Sequence[object] | np.ndarray | pd.Series,
    *,
    classes: Sequence[str] = tuple(E1_REFEREE_MARKERS),
    n_top: int = 3,
) -> RefereeResult:
    """Referee disputed broad labels by canonical-marker fractions (E1).

    Args:
        scores: ``marker_class_scores`` output, columns in ``classes`` order.
        first: First labelling per cell.
        second: Second labelling per cell.
        classes: Classes of the score columns.
        n_top: Disputes to list.

    Returns:
        Shares of disputes won by each labelling.
    """
    column_of = {cls: index for index, cls in enumerate(classes)}
    a = _object_array(first)
    b = _object_array(second)
    in_a = _in(a, list(classes))
    in_b = _in(b, list(classes))
    core = in_a & in_b
    disputed = core & (a != b)
    agree = core & (a == b)
    rows = np.flatnonzero(disputed)
    score_a = np.array([scores[row, column_of[str(a[row])]] for row in rows])
    score_b = np.array([scores[row, column_of[str(b[row])]] for row in rows])
    n_disputes = len(rows)
    agree_rows = np.flatnonzero(agree)
    top_ok = (
        float(
            np.mean(
                [
                    int(np.argmax(scores[row])) == column_of[str(a[row])]
                    for row in agree_rows
                ]
            )
        )
        if len(agree_rows)
        else math.nan
    )
    pairs = pd.Series([f"{a[row]} | {b[row]}" for row in rows], dtype=object)
    top = {
        str(key): int(value) for key, value in pairs.value_counts().head(n_top).items()
    }
    return RefereeResult(
        n_cells=len(a),
        n_disputes=n_disputes,
        first_wins=float((score_a > score_b).mean()) if n_disputes else math.nan,
        second_wins=float((score_b > score_a).mean()) if n_disputes else math.nan,
        undecided=float((score_a == score_b).mean()) if n_disputes else math.nan,
        consensus_top_marker_ok=top_ok,
        top_disputes=top,
    )


# --------------------------------------------------------------------------
# Inputs from the MAP outputs


def rule_inputs_from_provisional(
    labels: pd.DataFrame,
    sea: pd.DataFrame | None = None,
    ll_broad: pd.Series | None = None,
    *,
    prefix: str = "mmc_whb",
    ll_scheme: LikelihoodScheme = "e2",
) -> HumanRuleInputs:
    """Build rule inputs from a ``<sid>_ct_provisional.parquet`` table.

    Args:
        labels: Provisional labels of table cells, indexed by cell id
            (``ct_lineage_*``, ``ct_broad_*`` and the ``<prefix>_supercluster_*``
            engine columns).
        sea: ``seaad_broad_calls`` output (any order; aligned by cell id).
        ll_broad: Likelihood-typer broad labels by cell id (cells without
            one never agree).
        prefix: WHB engine-column prefix (``mmc_whb``; ``mmc_whb_setc`` for
            the set-c run).
        ll_scheme: Scheme of ``ll_broad`` (``HumanRuleInputs.ll_scheme``).

    Returns:
        The inputs, in ``labels`` order.
    """
    aligned_sea = None if sea is None else sea.reindex(labels.index)
    return HumanRuleInputs(
        total_counts=labels["total_counts"].to_numpy(np.float64),
        whb_supercluster=_object_array(labels[f"{prefix}_supercluster_name"]),
        whb_supercluster_bp=labels[f"{prefix}_supercluster_bp"].to_numpy(np.float64),
        whb_lineage=_object_array(labels["ct_lineage_name"]),
        whb_lineage_raw=labels["ct_lineage_raw"].to_numpy(np.float64),
        whb_broad=_object_array(labels["ct_broad_name"]),
        whb_broad_raw=labels["ct_broad_raw"].to_numpy(np.float64),
        sea_broad=None if aligned_sea is None else _object_array(aligned_sea["broad"]),
        sea_broad_raw=(
            None
            if aligned_sea is None
            else aligned_sea["broad_raw"].to_numpy(np.float64)
        ),
        ll_broad=(
            None if ll_broad is None else _object_array(ll_broad.reindex(labels.index))
        ),
        ll_scheme=ll_scheme,
    )


def soft_matrix_from_provisional(
    labels: pd.DataFrame,
    *,
    prefix: str = "mmc_whb",
    n_runners_up: int = 5,
    broad_of: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Return the soft broad matrix of a provisional table's WHB run (§5.5).

    Args:
        labels: Provisional labels with ``<prefix>_supercluster_{name,bp}``
            and ``<prefix>_supercluster_runner_up_<k>_{name,bp}``.
        prefix: Engine-column prefix.
        n_runners_up: Runner-ups to use.
        broad_of: Supercluster to broad class (default: the WHB vocab).

    Returns:
        ``(n, 8)`` soft rows over ``COMPOSITION_COLUMNS``.
    """
    stem = f"{prefix}_supercluster"
    name_columns = [f"{stem}_name"]
    bp_columns = [f"{stem}_bp"]
    for rank in range(1, n_runners_up + 1):
        name_column = f"{stem}_runner_up_{rank}_name"
        if name_column in labels.columns:
            name_columns.append(name_column)
            bp_columns.append(f"{stem}_runner_up_{rank}_bp")
    names = np.column_stack([_object_array(labels[column]) for column in name_columns])
    probabilities = np.column_stack(
        [labels[column].to_numpy(np.float64) for column in bp_columns]
    )
    return soft_broad_matrix(names, probabilities, broad_of=broad_of or whb_broad_of())


def argmax_broad_names(labels: pd.DataFrame, *, prefix: str = "mmc_whb") -> np.ndarray:
    """Return the broad class of each cell's assigned WHB supercluster.

    Args:
        labels: Provisional labels with ``<prefix>_supercluster_name``.
        prefix: Engine-column prefix.

    Returns:
        Broad class per cell (``Mixed/Unknown`` for sinks and missing calls).
    """
    broad_of = whb_broad_of()
    return np.array(
        [
            UNASSIGNED_LABEL
            if name is None
            else broad_of.get(str(name), UNASSIGNED_LABEL)
            for name in _object_array(labels[f"{prefix}_supercluster_name"])
        ],
        dtype=object,
    )


# --------------------------------------------------------------------------
# SEA-AD soft composition and paired JSD differences (M3 item 7)

GLIAL_CLASSES: Final[tuple[str, ...]] = (
    "Astrocytes",
    "Oligodendrocytes",
    OPC,
    "Microglia",
)
GLIAL_COLUMNS: Final[tuple[int, ...]] = tuple(
    HUMAN_BROAD_CLASSES.index(name) for name in GLIAL_CLASSES
)


def seaad_soft_broad_matrix(
    subclass: pd.DataFrame,
    supertype: pd.DataFrame | None = None,
    *,
    class_level: pd.DataFrame | None = None,
    vocab: VocabTable | None = None,
    n_runners_up: int = 5,
) -> np.ndarray:
    """Return per-cell soft broad-class mass of a SEA-AD run (as §5.5 for WHB).

    The assigned subclass's bootstrap probability and its runner-ups' are
    aggregated to the seven broad classes through the SEA-AD vocab. A
    subclass whose class depends on the supertype ("VLMC & Perivascular")
    splits its mass by the supertype level's probabilities when it is the
    assigned subclass; as a runner-up its mass is ``unallocated``. With
    ``class_level`` the subclass-level mass is multiplied by the assigned
    class's bootstrap probability (class x subclass mass, the root-level
    mass as WHB's soft composition and E2's SEA-AD broad probability; the
    class-level residual is ``unallocated``).

    Args:
        subclass: SEA-AD subclass level of a tidy table (``level_frame``).
        supertype: The supertype level (aligned by cell).
        class_level: The class level (aligned by cell).
        vocab: The SEA-AD vocab (default: packaged).
        n_runners_up: Runner-ups to aggregate.

    Returns:
        ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``, each summing to 1.
    """
    table = vocab or load_vocab("seaad_mr_subclass")
    position = {name: index for index, name in enumerate(HUMAN_BROAD_CLASSES)}
    unallocated = len(HUMAN_BROAD_CLASSES)
    split = {str(name) for name in table.supertype_overrides.get(table.key_column, [])}
    fixed_cache: dict[object, int] = {}

    def fixed_index(name: object) -> int:
        if name not in fixed_cache:
            if name is None or str(name) not in table or str(name) in split:
                fixed_cache[name] = unallocated
            else:
                fixed_cache[name] = position.get(
                    table.broad_class(str(name), None), unallocated
                )
        return fixed_cache[name]

    n_cells = len(subclass)
    matrix = np.zeros((n_cells, len(COMPOSITION_COLUMNS)), dtype=np.float64)
    rows = np.arange(n_cells)
    columns = [("name", "bp"), *_runner_up_columns(subclass, n_runners_up)]
    for name_column, probability_column in columns:
        names = _object_array(subclass[name_column])
        values = np.clip(
            np.nan_to_num(subclass[probability_column].to_numpy(np.float64), nan=0.0),
            0.0,
            1.0,
        )
        index = np.array([fixed_index(name) for name in names], dtype=np.int64)
        allocated = index < unallocated
        np.add.at(matrix, (rows[allocated], index[allocated]), values[allocated])
    assigned = _object_array(subclass["name"])
    is_split = np.array(
        [name is not None and str(name) in split for name in assigned], dtype=bool
    )
    if supertype is not None and is_split.any():
        aligned = supertype.reindex(subclass.index)
        bp = np.clip(
            np.nan_to_num(subclass["bp"].to_numpy(np.float64), nan=0.0), 0.0, 1.0
        )
        shares = np.zeros((n_cells, len(HUMAN_BROAD_CLASSES)))
        supertype_columns = [
            ("name", "bp"),
            *_runner_up_columns(aligned, n_runners_up),
        ]
        for name_column, probability_column in supertype_columns:
            names = _object_array(aligned[name_column])
            values = np.nan_to_num(
                aligned[probability_column].to_numpy(np.float64), nan=0.0
            )
            for row in np.flatnonzero(is_split):
                name = names[row]
                if name is None:
                    continue
                cls = table.broad_class(str(assigned[row]), str(name))
                if cls in position:
                    shares[row, position[cls]] += values[row]
        totals = shares.sum(axis=1)
        good = is_split & (totals > 0)
        matrix[good, : len(HUMAN_BROAD_CLASSES)] += (
            shares[good] / totals[good, None] * bp[good, None]
        )
    total = matrix[:, :unallocated].sum(axis=1)
    over = total > 1.0
    if over.any():
        matrix[over, :unallocated] /= total[over, None]
        total = np.minimum(total, 1.0)
    if class_level is not None:
        class_bp = np.clip(
            np.nan_to_num(
                class_level.reindex(subclass.index)["bp"].to_numpy(np.float64),
                nan=0.0,
            ),
            0.0,
            1.0,
        )
        matrix[:, :unallocated] *= class_bp[:, None]
        total = matrix[:, :unallocated].sum(axis=1)
    matrix[:, unallocated] = 1.0 - total
    return matrix


@dataclass(frozen=True)
class BootstrapJsdDifference:
    """The difference of two labellings' JSDs with a paired block bootstrap.

    Attributes:
        first_jsd: JSD of the first labelling (all cells).
        second_jsd: JSD of the second labelling.
        difference: ``first_jsd - second_jsd``.
        ci_low: 2.5th percentile of the replicate differences.
        ci_high: 97.5th percentile.
        first_ci: Percentile CI of the first JSD.
        second_ci: Percentile CI of the second JSD.
        share_positive: Share of replicates with a positive difference.
        n_reps: Replicates.
        resampling: ``joint`` or ``independent`` (``block_bootstrap_jsd``).
    """

    first_jsd: float
    second_jsd: float
    difference: float
    ci_low: float
    ci_high: float
    first_ci: tuple[float, float]
    second_ci: tuple[float, float]
    share_positive: float
    n_reps: int
    resampling: Resampling = "joint"


def paired_block_bootstrap_jsd_difference(
    first_a: np.ndarray,
    first_b: np.ndarray,
    second_a: np.ndarray,
    second_b: np.ndarray,
    *,
    columns: Sequence[int] | None = None,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    resampling: Resampling = "joint",
) -> BootstrapJsdDifference:
    """Bootstrap the JSD difference of two labellings of the same sections.

    One draw per replicate is applied to both labellings (a paired
    bootstrap), so the interval is that of the difference, not of two
    independent estimates. With ``joint`` resampling (the registered method)
    the draw is over shared tile locations and the same weights apply to
    both sections as well (``block_bootstrap_jsd``); with ``independent``
    each section's tiles are drawn on their own (sensitivity).

    Args:
        first_a: Tile sums of section A under the first labelling.
        first_b: Tile sums of section B under the first labelling.
        second_a: Tile sums of section A under the second labelling (same
            tiles, same row order as ``first_a``).
        second_b: Tile sums of section B under the second labelling.
        columns: Classes compared (``jensen_shannon_distance``).
        n_reps: Replicates.
        seed: RNG seed.
        resampling: ``joint`` (all four tables on one grid) or
            ``independent``.

    Returns:
        The point estimates and the paired percentile CI.

    Raises:
        ValueError: If the paired tile tables differ in shape or a section
            has no tile.
    """
    first_a, first_b, second_a, second_b = (
        np.asarray(item, dtype=np.float64)
        for item in (first_a, first_b, second_a, second_b)
    )
    if first_a.shape[0] != second_a.shape[0] or first_b.shape[0] != second_b.shape[0]:
        raise ValueError("paired tile tables must have the same tiles")
    nonempty_a = (first_a.sum(axis=1) > 0) | (second_a.sum(axis=1) > 0)
    nonempty_b = (first_b.sum(axis=1) > 0) | (second_b.sum(axis=1) > 0)
    if not nonempty_a.any() or not nonempty_b.any():
        raise ValueError("each section needs at least one non-empty tile")
    point_first = float(
        jensen_shannon_distance(first_a.sum(0), first_b.sum(0), columns=columns)
    )
    point_second = float(
        jensen_shannon_distance(second_a.sum(0), second_b.sum(0), columns=columns)
    )
    rng = np.random.default_rng(seed)
    if resampling == "joint":
        if first_a.shape[0] != first_b.shape[0]:
            raise ValueError(
                "joint resampling needs both sections on one tile grid "
                "(shared_tile_codes)"
            )
        keep = nonempty_a | nonempty_b
        weights = _bootstrap_weights(rng, int(keep.sum()), n_reps)
        weights_a = weights_b = weights
        first_a, second_a = first_a[keep], second_a[keep]
        first_b, second_b = first_b[keep], second_b[keep]
    elif resampling == "independent":
        first_a, second_a = first_a[nonempty_a], second_a[nonempty_a]
        first_b, second_b = first_b[nonempty_b], second_b[nonempty_b]
        weights_a = _bootstrap_weights(rng, len(first_a), n_reps)
        weights_b = _bootstrap_weights(rng, len(first_b), n_reps)
    else:
        raise ValueError(f"unknown resampling {resampling!r}")
    reps_first = np.asarray(
        jensen_shannon_distance(
            weights_a @ first_a, weights_b @ first_b, columns=columns
        ),
        dtype=np.float64,
    ).reshape(-1)
    reps_second = np.asarray(
        jensen_shannon_distance(
            weights_a @ second_a, weights_b @ second_b, columns=columns
        ),
        dtype=np.float64,
    ).reshape(-1)
    differences = reps_first - reps_second
    low, high = np.nanpercentile(differences, [2.5, 97.5])
    first_low, first_high = np.nanpercentile(reps_first, [2.5, 97.5])
    second_low, second_high = np.nanpercentile(reps_second, [2.5, 97.5])
    return BootstrapJsdDifference(
        first_jsd=point_first,
        second_jsd=point_second,
        difference=point_first - point_second,
        ci_low=float(low),
        ci_high=float(high),
        first_ci=(float(first_low), float(first_high)),
        second_ci=(float(second_low), float(second_high)),
        share_positive=float(np.mean(differences > 0)),
        n_reps=int(n_reps),
        resampling=resampling,
    )


# --------------------------------------------------------------------------
# Held-out-gene enrichment (§5.8; M3 item 4)

# Marker classes of ``heldout_markers_human.csv`` (E2's scheme) to the seven
# broad classes; neurons pool the excitatory and inhibitory markers.
HELDOUT_BROAD_CLASS: Final[dict[str, str]] = {
    "Exc": "Neurons",
    "Inh": "Neurons",
    "Astro": "Astrocytes",
    "Oligo": "Oligodendrocytes",
    "OPC": OPC,
    "Immune": "Microglia",
    "Vascular": "Vascular cells",
    "Fibroblast": "Fibroblasts",
}
MIN_HELDOUT_MARKERS: Final = 2
MAX_HELDOUT_MARKERS: Final = 3
H4_MIN_FOLD: Final = 3.0
H4_MIN_AUROC: Final = 0.70


@dataclass(frozen=True)
class HeldoutMarkerSelection:
    """Held-out markers of one panel and platform (§5.8 step 1).

    Attributes:
        platform: ``MERSCOPE`` or ``XENIUM``.
        markers: Broad class to the selected marker symbols (>= the minimum).
        skipped: Broad class to the reason it is not scored.
        not_on_panel: Listed markers absent from the panel, per broad class.
        avoided: Listed markers avoided on this platform, per broad class.
    """

    platform: str
    markers: dict[str, tuple[str, ...]]
    skipped: dict[str, str]
    not_on_panel: dict[str, tuple[str, ...]]
    avoided: dict[str, tuple[str, ...]]

    @property
    def all_markers(self) -> tuple[str, ...]:
        """Return every selected marker, in class order."""
        return tuple(gene for genes in self.markers.values() for gene in genes)


def select_heldout_markers(
    table: pd.DataFrame,
    panel_symbols: Sequence[str],
    platform: str,
    *,
    classes: Sequence[str] = HUMAN_BROAD_CLASSES,
    class_of: Mapping[str, str] = HELDOUT_BROAD_CLASS,
    min_markers: int = MIN_HELDOUT_MARKERS,
    max_markers: int = MAX_HELDOUT_MARKERS,
) -> HeldoutMarkerSelection:
    """Choose each broad class's held-out markers on one panel and platform.

    Markers are taken in the table's ``rank`` order (then file order), if
    they are on the panel and not avoided on the platform
    (``avoid_platforms``), up to ``max_markers`` per broad class; a class with
    fewer than ``min_markers`` is skipped (§5.8: "classes with < 2 held-out
    markers on a panel are skipped and reported").

    Args:
        table: ``load_heldout_markers`` output (``marker_class``,
            ``gene_symbol``, ``avoid_platforms``, optional ``rank``).
        panel_symbols: Symbols of the panel on this platform.
        platform: ``MERSCOPE`` or ``XENIUM``.
        classes: Broad classes to score.
        class_of: Marker class to broad class.
        min_markers: Minimum markers per scored class.
        max_markers: Maximum markers per class.

    Returns:
        The selection.
    """
    on_panel = {str(symbol).upper() for symbol in panel_symbols if symbol}
    frame = table.copy()
    frame["_order"] = np.arange(len(frame))
    if "rank" not in frame.columns:
        frame["rank"] = 1
    frame = frame.sort_values(["rank", "_order"], kind="stable")
    wanted = platform.upper()
    chosen: dict[str, list[str]] = {cls: [] for cls in classes}
    absent: dict[str, list[str]] = {cls: [] for cls in classes}
    avoided: dict[str, list[str]] = {cls: [] for cls in classes}
    for row in frame.itertuples(index=False):
        cls = class_of.get(str(row.marker_class))
        if cls not in chosen:
            continue
        symbol = str(row.gene_symbol)
        avoid = {
            item.strip().upper()
            for item in str(getattr(row, "avoid_platforms", "") or "").split(";")
            if item.strip() and item.strip().lower() != "nan"
        }
        if symbol.upper() not in on_panel:
            absent[cls].append(symbol)
        elif wanted in avoid:
            avoided[cls].append(symbol)
        elif len(chosen[cls]) < max_markers and symbol not in chosen[cls]:
            chosen[cls].append(symbol)
    markers = {
        cls: tuple(genes) for cls, genes in chosen.items() if len(genes) >= min_markers
    }
    skipped = {
        cls: f"{len(genes)} held-out marker(s) on the panel (< {min_markers})"
        for cls, genes in chosen.items()
        if len(genes) < min_markers
    }
    return HeldoutMarkerSelection(
        platform=wanted,
        markers=markers,
        skipped=skipped,
        not_on_panel={cls: tuple(genes) for cls, genes in absent.items() if genes},
        avoided={cls: tuple(genes) for cls, genes in avoided.items() if genes},
    )


def cop_rule_broad_names(
    broad_names: Sequence[object] | np.ndarray,
    supercluster: Sequence[object] | np.ndarray,
    supercluster_bp: np.ndarray,
    total_counts: np.ndarray,
    *,
    platform: str,
    sea_confident_opc: np.ndarray | None = None,
    thresholds: AnnotationThresholds | None = None,
    floors: FloorLookup | None = None,
) -> np.ndarray:
    """Apply the §5.2 COP rule to a broad labelling (H4 option (b)).

    A WHB COP call is broad OPC only with >= the COP supercluster floor and
    supercluster bp >= ``whb_supercluster``, or when SEA-AD confidently calls
    OPC (``sea_confident_opc``); otherwise it stays at lineage and leaves the
    seven classes (``None``), as ``evaluate_human_rules`` suppresses it.

    Args:
        broad_names: Broad class per cell (e.g. the argmax labelling).
        supercluster: Assigned WHB supercluster per cell.
        supercluster_bp: Its bootstrap probability.
        total_counts: Counts per cell.
        platform: ``MERSCOPE`` or ``XENIUM`` (floors).
        sea_confident_opc: SEA-AD's confident OPC calls (the rescue), if any.
        thresholds: Thresholds (default: v1).
        floors: Floors (default: the packaged set-a floors).

    Returns:
        The labelling with suppressed COP calls set to ``None``.
    """
    settings = thresholds or AnnotationThresholds()
    floor_lookup = floors or FloorLookup.packaged()
    names = _object_array(supercluster)
    labels = _object_array(broad_names).copy()
    counts = np.asarray(total_counts, dtype=np.float64)
    passes = (counts >= floor_lookup.per_cell("supercluster", platform, names)) & (
        meets_threshold(supercluster_bp, settings.whb_supercluster)
    )
    if sea_confident_opc is not None:
        passes |= np.asarray(sea_confident_opc, dtype=bool)
    labels[(names == COP_SUPERCLUSTER) & ~passes] = None
    return labels


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Return the area under the ROC curve (Mann-Whitney, average ranks).

    Args:
        scores: Score per item.
        positive: Whether each item is a positive.

    Returns:
        The AUROC; NaN without positives or negatives.
    """
    from scipy.stats import rankdata

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positive, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(values)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class HeldoutEnrichment:
    """Enrichment of one class's held-out markers in its assigned cells.

    Attributes:
        broad_class: The class.
        markers: Its held-out markers.
        n_assigned: Cells assigned the class.
        n_other: Cells assigned another of the seven classes.
        rate_assigned: Held-out counts per count in the assigned cells
            (pooled).
        rate_other: The same in the other cells.
        fold: ``rate_assigned / rate_other``.
        detection_assigned: Share of assigned cells with a held-out count.
        detection_other: The same in the other cells.
        auroc: AUROC of the per-cell held-out fraction, assigned vs other.
        passes: Fold and AUROC at or above the H4 thresholds.
    """

    broad_class: str
    markers: tuple[str, ...]
    n_assigned: int
    n_other: int
    rate_assigned: float
    rate_other: float
    fold: float
    detection_assigned: float
    detection_other: float
    auroc: float
    passes: bool


def heldout_enrichment(
    marker_counts: np.ndarray | pd.DataFrame,
    total_counts: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    markers: Mapping[str, Sequence[str]],
    *,
    classes: Sequence[str] = HUMAN_BROAD_CLASSES,
    include: np.ndarray | None = None,
    min_fold: float = H4_MIN_FOLD,
    min_auroc: float = H4_MIN_AUROC,
) -> list[HeldoutEnrichment]:
    """Score held-out markers in the assigned class vs the others (§5.8 step 3).

    Args:
        marker_counts: Cells x held-out markers (a DataFrame with marker
            symbols as columns).
        total_counts: Counts per cell (all panel genes).
        labels: Broad class per cell (from the held-out re-map).
        markers: Broad class to its held-out marker symbols.
        classes: The broad classes (cells labelled outside them are left
            out).
        include: Cells to score (default: all).
        min_fold: H4 fold threshold.
        min_auroc: H4 AUROC threshold.

    Returns:
        One result per class of ``markers``.
    """
    frame = pd.DataFrame(marker_counts)
    totals = np.maximum(np.asarray(total_counts, dtype=np.float64), 1.0)
    names = _object_array(labels)
    keep = _in(names, classes)
    if include is not None:
        keep &= np.asarray(include, dtype=bool)
    results = []
    for cls, genes in markers.items():
        heldout = frame[list(genes)].to_numpy(np.float64).sum(axis=1)
        assigned = keep & (names == cls)
        other = keep & (names != cls)
        rate_a = (
            float(heldout[assigned].sum() / totals[assigned].sum())
            if assigned.any()
            else math.nan
        )
        rate_o = (
            float(heldout[other].sum() / totals[other].sum())
            if other.any()
            else math.nan
        )
        fold = rate_a / rate_o if rate_o > 0 else (math.inf if rate_a > 0 else math.nan)
        score = heldout / totals
        area = auroc(score[assigned | other], assigned[assigned | other])
        results.append(
            HeldoutEnrichment(
                broad_class=cls,
                markers=tuple(genes),
                n_assigned=int(assigned.sum()),
                n_other=int(other.sum()),
                rate_assigned=rate_a,
                rate_other=rate_o,
                fold=float(fold),
                detection_assigned=float((heldout[assigned] > 0).mean())
                if assigned.any()
                else math.nan,
                detection_other=float((heldout[other] > 0).mean())
                if other.any()
                else math.nan,
                auroc=area,
                passes=bool(
                    math.isfinite(area)
                    and not math.isnan(fold)
                    and fold >= min_fold
                    and area >= min_auroc
                ),
            )
        )
    return results


# --------------------------------------------------------------------------
# X1: reference-pseudobulk platform factors (M3 item 3)

X1_CAP_LOG2: Final = 2.0
X1_LABEL_MIN_BP: Final = 0.8
X1_COUNT_SCALE: Final = 10.0


def profile_matrix(
    profiles: pd.DataFrame,
    level: str,
    gene_ids: Sequence[str],
    *,
    key: Literal["node", "node_name"] = "node_name",
) -> pd.DataFrame:
    """Return a bundle profile level as nodes x genes expected fractions.

    Args:
        profiles: ``profiles.parquet`` (long).
        level: Level to take.
        gene_ids: Query genes; the fractions are renormalised over them.
        key: Node labels or names as the index.

    Returns:
        ``nodes x gene_ids`` expected fractions (rows sum to 1; genes the
        reference lacks are 0).

    Raises:
        ValueError: If the level has no rows.
    """
    rows = profiles[profiles["level"].astype(str) == level]
    if rows.empty:
        raise ValueError(f"profiles have no rows at level {level!r}")
    wide = rows.pivot_table(
        index=key, columns="gene_id", values="mean_cpm", aggfunc="first"
    )
    wide = wide.reindex(columns=[str(gene) for gene in gene_ids]).fillna(0.0)
    totals = wide.sum(axis=1)
    return wide.div(totals.where(totals > 0, 1.0), axis=0)


def reference_pseudobulk_log2_factors(
    counts: sparse.spmatrix | np.ndarray,
    labels: Sequence[object] | np.ndarray,
    profiles: pd.DataFrame,
    *,
    include: np.ndarray | None = None,
    cap_log2: float | None = X1_CAP_LOG2,
    pseudocount: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-gene platform factors against a reference pseudobulk (X1).

    For the labelled cells the observed gene totals are compared with the
    reference expectation ``sum_c n_c * p_{label(c), g}`` (``n_c`` the cell's
    counts on the query genes, ``p`` the label's expected fraction); the
    log2 ratio is centred on its median and capped.

    Args:
        counts: Cells x query genes.
        labels: Reference node (``profiles`` index) per cell.
        profiles: ``profile_matrix`` output over the query genes.
        include: Cells used (e.g. bp >= 0.8); unlabelled cells are left out.
        cap_log2: Cap on the centred log2 factor; ``None`` for none.
        pseudocount: Added to observed and expected totals.

    Returns:
        ``(log2_factors, uncapped)``, one per query gene.

    Raises:
        ValueError: If the profiles do not have one column per gene or no
            cell is usable.
    """
    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts, dtype=np.float64)
    if profiles.shape[1] != matrix.shape[1]:
        raise ValueError("profiles must have one column per query gene")
    names = _object_array(labels)
    row_of = {str(name): index for index, name in enumerate(profiles.index)}
    rows = np.array(
        [row_of.get(str(name), -1) if name is not None else -1 for name in names]
    )
    usable = rows >= 0
    if include is not None:
        usable &= np.asarray(include, dtype=bool)
    if not usable.any():
        raise ValueError("no labelled cell to estimate the factors from")
    subset = matrix[usable]
    depth = np.asarray(subset.sum(axis=1)).ravel()
    observed = np.asarray(subset.sum(axis=0)).ravel()
    weight = np.bincount(rows[usable], weights=depth, minlength=len(profiles))
    expected = weight @ profiles.to_numpy(np.float64)
    uncapped = np.log2((observed + pseudocount) / (expected + pseudocount))
    uncapped -= np.median(uncapped)
    capped = uncapped if cap_log2 is None else np.clip(uncapped, -cap_log2, cap_log2)
    return capped, uncapped


def rescale_counts(
    counts: sparse.spmatrix | np.ndarray,
    log2_factors: np.ndarray,
    *,
    scale: float = X1_COUNT_SCALE,
) -> sparse.csr_matrix:
    """Divide each gene by its platform factor and round to integers.

    MapMyCells normalises raw counts to CPM, so a common ``scale`` leaves the
    mapping unchanged while keeping the rounding error small.

    Args:
        counts: Cells x genes.
        log2_factors: Per-gene log2 factors.
        scale: Common multiplier before rounding.

    Returns:
        Integer-valued CSR counts ``round(scale * counts / 2**factor)``.

    Raises:
        ValueError: If there is not one factor per gene.
    """
    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts, dtype=np.float64)
    factors = np.asarray(log2_factors, dtype=np.float64)
    if factors.shape != (matrix.shape[1],):
        raise ValueError("one log2 factor per gene is needed")
    scaled = matrix @ sp.diags(scale * np.power(2.0, -factors))
    scaled = sp.csr_matrix(scaled)
    scaled.data = np.rint(scaled.data)
    scaled.eliminate_zeros()
    return scaled.astype(np.int32)


# --------------------------------------------------------------------------
# Contamination and diffuse-profile flags (§5.6; M3 item 5): the primitives
# live in ``merxen.annotation.flags`` (imported above).


# --------------------------------------------------------------------------
# E8 segmentation comparison helpers (M3 item 2)


def occupied_area_mm2(
    xy: np.ndarray, *, bin_um: float = 100.0, min_cells: int = 3
) -> float:
    """Return the tissue area covered by cells, from an occupancy grid.

    Args:
        xy: ``(n, 2)`` coordinates in µm.
        bin_um: Grid bin edge.
        min_cells: Cells a bin needs to count as tissue.

    Returns:
        Occupied area in mm².

    Raises:
        ValueError: If ``bin_um`` is not positive.
    """
    if bin_um <= 0:
        raise ValueError("bin_um must be positive")
    points = np.asarray(xy, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return 0.0
    cells = np.floor(points / bin_um).astype(np.int64)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    return float((counts >= min_cells).sum() * (bin_um / 1000.0) ** 2)


def foreign_marker_fraction(
    scores: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    classes: Sequence[str] = tuple(E1_REFEREE_MARKERS),
) -> np.ndarray:
    """Return the share of a cell's counts on other classes' canonical markers.

    Args:
        scores: ``marker_class_scores`` output (columns in ``classes`` order).
        labels: Broad class per cell.
        classes: Classes of the score columns.

    Returns:
        Per cell ``sum(scores) - scores[label]``; NaN outside ``classes``.
    """
    names = _object_array(labels)
    column = {cls: index for index, cls in enumerate(classes)}
    total = np.asarray(scores, dtype=np.float64).sum(axis=1)
    values = np.full(len(names), math.nan)
    for row, name in enumerate(names):
        if name is not None and str(name) in column:
            values[row] = total[row] - scores[row, column[str(name)]]
    return values


@dataclass(frozen=True)
class MouseConfidence:
    """Mouse v1 confident levels without region pruning (§7.3, D-M4).

    Attributes:
        class_confident: WMB class bp >= 0.9 and >= 20 counts.
        subclass_confident: Confident class, subclass bp >= 0.8 and >= 50
            counts.
    """

    class_confident: np.ndarray
    subclass_confident: np.ndarray


def mouse_confidence(
    labels: pd.DataFrame,
    *,
    class_bp: float = 0.9,
    class_min_counts: int = 20,
    subclass_bp: float = 0.8,
    subclass_min_counts: int = 50,
) -> MouseConfidence:
    """Return the D-M4 mouse confident class and subclass flags.

    Args:
        labels: Provisional mouse labels (``total_counts``,
            ``mmc_wmb_class_bp``, ``mmc_wmb_subclass_bp``).
        class_bp: Class threshold.
        class_min_counts: Class floor.
        subclass_bp: Subclass threshold.
        subclass_min_counts: Subclass floor.

    Returns:
        The flags (region pruning, M6, is not applied).
    """
    counts = labels["total_counts"].to_numpy(np.float64)
    class_ok = meets_threshold(labels["mmc_wmb_class_bp"], class_bp) & (
        counts >= class_min_counts
    )
    subclass_ok = (
        class_ok
        & meets_threshold(labels["mmc_wmb_subclass_bp"], subclass_bp)
        & (counts >= subclass_min_counts)
    )
    return MouseConfidence(class_confident=class_ok, subclass_confident=subclass_ok)


# --------------------------------------------------------------------------
# Production queries of published samples (variant re-maps)


def published_queries(
    clustered: Mapping[str, Path | str],
    panel: AnnotationPanel,
    *,
    species: Literal["human", "mouse"],
    gene_id_fallback_csv: Path | str | None = None,
    min_counts: int = 10,
) -> dict[str, SampleQuery]:
    """Rebuild MAP's query of published clustered samples on a panel.

    The same loading, control removal, gene-ID resolution and table-cell
    selection as ``merxen annotate --from-clustered-h5ad``
    (``pipeline.load_samples`` and ``build_sample_query``), so a shadow
    variant (held-out genes, X1 rescaling, LL) starts from the production
    query.

    Args:
        clustered: Platform (``MERSCOPE`` / ``XENIUM``) to its published
            ``<sid>_clustered.h5ad``; the sample id is the file stem without
            ``_clustered``.
        panel: The annotation panel (``panel_genes.json``).
        species: Species.
        gene_id_fallback_csv: The gene-ID fallback table used by the run.
        min_counts: Table-cell threshold.

    Returns:
        Platform to its query.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.pipeline import MapSample, build_sample_query, load_samples

    config = AnnotationConfig(species=species)
    if gene_id_fallback_csv is not None:
        config = config.model_copy(
            update={
                "panel": config.panel.model_copy(
                    update={"gene_id_fallback_csv": Path(gene_id_fallback_csv)}
                )
            }
        )
    config = config.coupled_to_clustering(min_counts)
    samples = [
        MapSample(
            sample_id=Path(path).name.removesuffix(".h5ad").removesuffix("_clustered"),
            platform=platform.upper(),
            h5ad_path=Path(path),
            source="clustered",
        )
        for platform, path in clustered.items()
    ]
    loaded = load_samples(samples, config, min_counts=min_counts)
    return {item.sample.platform: build_sample_query(item, panel) for item in loaded}


def variant_sha256(
    cell_ids: Sequence[object], gene_ids: Sequence[str], counts: sparse.spmatrix
) -> str:
    """Return the sha256 of a variant query (cells, genes and every value).

    Unlike ``query_fingerprint`` (per-cell totals), it changes with any
    per-gene value, so two rescalings of one query differ.

    Args:
        cell_ids: Query cell ids.
        gene_ids: Query gene IDs.
        counts: Cells x genes (sparse or dense).

    Returns:
        Hex digest.
    """
    import hashlib

    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts, dtype=np.float64)
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(cell) for cell in cell_ids).encode("utf-8"))
    digest.update(b"\x1e")
    digest.update("\x1f".join(str(gene) for gene in gene_ids).encode("utf-8"))
    for part in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def map_query_variant(
    query: SampleQuery,
    bundle: MmcBundle,
    output_parquet: Path | str,
    *,
    work_dir: Path | str,
    drop_gene_ids: Sequence[str] = (),
    log2_factors: Mapping[str, float] | None = None,
    n_processors: int = 6,
    expected_ctm_version: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    reuse: bool = True,
) -> Path:
    """Map a variant of a production query with the production engine.

    The variant drops genes (held-out markers, §5.8: removed from the query
    **and**, through ``restrict_lookup``, from the lookup) and/or divides each
    gene by a platform factor (X1, ``rescale_counts``); everything else is
    MAP's configuration (bootstrap 0.5 x 100, seed 0, raw normalisation,
    BLAS threads 1).

    An existing parquet is reused only when its metadata records the same
    variant query (``variant_sha256``), dropped genes, rescaling, bundle
    ``build_hash``, engine parameters, ctm version and ``run_metadata``.

    Args:
        query: The production query (``published_queries``).
        bundle: The reference bundle.
        output_parquet: The tidy parquet to write.
        work_dir: Scratch directory (its query and lookup files are removed,
            also on failure).
        drop_gene_ids: Gene IDs removed from query and lookup.
        log2_factors: Gene ID to log2 factor (missing genes: 0).
        n_processors: MapMyCells processes.
        expected_ctm_version: ctm version (default: the configured one).
        run_metadata: Extra parquet metadata.
        reuse: Keep an existing parquet of the same variant instead of
            re-mapping.

    Returns:
        The tidy parquet.

    Raises:
        ValueError: If every gene would be dropped.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mapmycells_engine import (
        MmcEngineParams,
        read_tidy_parquet,
        restrict_lookup,
        run_mmc,
        write_query_h5ad,
    )

    output = Path(output_parquet)
    dropped = {str(gene) for gene in drop_gene_ids}
    keep = [i for i, gene in enumerate(query.gene_ids) if gene not in dropped]
    if not keep:
        raise ValueError("the variant drops every query gene")
    genes = [query.gene_ids[i] for i in keep]
    counts = query.counts[:, keep]
    if log2_factors is not None:
        factors = np.array([float(log2_factors.get(gene, 0.0)) for gene in genes])
        counts = rescale_counts(counts, factors)
    version = expected_ctm_version or AnnotationConfig(species="human").ctm_version
    params = MmcEngineParams(n_processors=n_processors)
    metadata = {
        "variant_sha256": variant_sha256(query.cell_ids, genes, counts),
        "variant_dropped_genes": sorted(dropped & set(query.gene_ids)),
        "variant_rescaled": log2_factors is not None,
        **dict(run_metadata or {}),
    }
    expected = {
        **metadata,
        "build_hash": bundle.build_hash,
        "engine_params": params.reuse_key(),
        "ctm_version": version,
    }
    if reuse and output.is_file():
        _, recorded = read_tidy_parquet(output)
        differing = sorted(
            key for key, value in expected.items() if recorded.get(key) != value
        )
        if not differing:
            logger.info("reusing %s", output)
            return output
        logger.info(
            "%s: not reused (%s differ); re-mapping", output, ", ".join(differing)
        )
    scratch = Path(work_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    query_path = scratch / f"{output.stem}.query.h5ad"
    lookup_path = scratch / f"{output.stem}.lookup.json"
    try:
        write_query_h5ad(counts, query.cell_ids, genes, query_path)
        restricted = restrict_lookup(bundle, genes, lookup_path)
        run_mmc(
            query_path,
            bundle,
            params,
            output_parquet=output,
            work_dir=scratch,
            expected_ctm_version=version,
            lookup_path=restricted.path,
            run_metadata=metadata,
        )
    finally:
        query_path.unlink(missing_ok=True)
        lookup_path.unlink(missing_ok=True)
    return output


def whb_labels_from_tidy(
    tidy: pd.DataFrame,
    vocab: pd.DataFrame,
    total_counts: pd.Series,
    *,
    level: str = "CCN202210140_SUPC",
    prefix: str = "mmc_whb",
    n_runners_up: int = 5,
) -> pd.DataFrame:
    """Return provisional-style WHB columns of a tidy MMC table.

    Lineage and broad aggregate the supercluster bootstrap probabilities as
    MAP's provisional labels do (``aggregate_parent_probability`` over the
    bundle's vocab snapshot), so the shadow rules and compositions run on a
    variant re-map as on the production one.

    Args:
        tidy: Tidy MMC table of a WHB run.
        vocab: The bundle's ``vocab_snapshot.csv``.
        total_counts: Counts per cell (after control removal), by cell id.
        level: The supercluster level.
        prefix: Engine-column prefix of the output.
        n_runners_up: Runner-up columns to copy.

    Returns:
        Indexed by cell id: ``total_counts``, ``ct_lineage_{name,raw}``,
        ``ct_broad_{name,raw}``, ``<prefix>_supercluster_{name,bp}`` and the
        runner-up ``_name`` / ``_bp`` columns.
    """
    from merxen.annotation.mapmycells_engine import (
        aggregate_parent_probability,
        level_frame,
    )

    frame = level_frame(tidy, level)
    rows = vocab[vocab["level"].astype(str) == level]

    def lookup(column: str) -> dict[str, str | None]:
        return {
            str(node): (None if str(value) in {"", "nan", "None"} else str(value))
            for node, value in zip(rows["node"], rows[column], strict=True)
        }

    lineage = aggregate_parent_probability(frame, lookup("lineage"))
    broad = aggregate_parent_probability(frame, lookup("broad_class"))
    output = pd.DataFrame(
        {
            "total_counts": total_counts.reindex(frame.index).to_numpy(np.float64),
            "ct_lineage_name": lineage.classes,
            "ct_lineage_raw": lineage.probability,
            "ct_broad_name": broad.classes,
            "ct_broad_raw": broad.probability,
            f"{prefix}_supercluster_name": frame["name"].astype(object).to_numpy(),
            f"{prefix}_supercluster_bp": frame["bp"].to_numpy(np.float64),
        },
        index=frame.index.astype(str),
    )
    for rank in range(1, n_runners_up + 1):
        name_column = f"runner_up_{rank}_name"
        if name_column in frame.columns:
            output[f"{prefix}_supercluster_runner_up_{rank}_name"] = (
                frame[name_column].astype(object).to_numpy()
            )
            output[f"{prefix}_supercluster_runner_up_{rank}_bp"] = frame[
                f"runner_up_{rank}_probability"
            ].to_numpy(np.float64)
    return output
