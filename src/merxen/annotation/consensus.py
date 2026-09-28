"""Human consensus rules v1: statuses, final label and tier (plan §5.2, §5.3).

``resolve_human`` turns the per-cell calls of the WHB-frontal primary and the
SEA-AD Multiregion second vote into one status per level (``CellStatus``,
§4.2), the deepest confident label and the consensus tier:

1. **Lineage** is confident when the WHB lineage-aggregated probability meets
   the broad threshold (0.73), the node is plausible (not a vocab sink and
   plausible in the anatomical region), counts reach the hard floor, the
   panel resolves the level for the cell's class at its depth bin, and the
   second vote passes: below 60 counts SEA-AD must agree at lineage, from 60
   it must not confidently disagree (SEA broad probability >= 0.68 on a
   different class). An implausible node keeps its lineage when SEA-AD
   agrees at lineage.
2. **Broad** needs a confident lineage, the broad probability, the broad
   floor of the class x platform x panel, resolvability and the same vote at
   the 7-class level. **COP rule:** a WHB "Committed oligodendrocyte
   precursor" call is broad "Oligodendrocyte precursors" only when the COP
   supercluster rule passes (counts >= the COP supercluster floor, 120, and
   supercluster probability >= the supercluster threshold) or SEA-AD
   confidently calls OPC; otherwise the cell stays at lineage
   ("Oligodendrocyte lineage") with ``flag_cop_suppressed``.
3. **NT** (neurons): a confident broad, the aggregated Exc / Inh probability,
   the class's broad floor (Inh: MERSCOPE 10, Xenium 30) and resolvability;
   no second vote. Non-neurons are ``not_applicable``.
4. **Supercluster:** a confident parent (NT for neurons, else broad), gate
   level ``full`` (a warning flag does not block), the supercluster
   probability, the supercluster floor (COP 120) and resolvability.
5. **SEA-AD subclass** (secondary name, never in ``ct_final``): gate level
   ``full``, a confident WHB broad that SEA-AD's 7-class call agrees with,
   the supercluster floor of the broad class, SEA's raw threshold (0.55
   below 60 counts, 0.45 from 60) and the leaf's resolvability for the
   broad class.
6. **WHB cluster** is never a leaf or in ``ct_final``; it is reported only
   with ``allow_fine_levels`` and a resolvability pass (OD-E4).
7. **Final label** = the deepest confident level along lineage -> broad ->
   NT -> supercluster (``Mixed/Unknown`` for none). 8. **Tier**: methods
   agreeing at the 7-class level, computed independently of the statuses.

**Degraded modes** (§5.3) are one explicit truth table, ``DEGRADED_MODES``:
which method must agree below 60 counts, which may veto from 60, which
rescues an implausible lineage, the status when no second method exists and
the maximum tier. ``annotation_allow_single_method`` lets WHB decide alone
below 60 counts in the WHB-only mode (recorded). The likelihood-typer rows
exist for the truth table only: by OD-B8 (decided 2026-09-27) no v1 or v1.1
production path enables the LL vote.

When several checks fail, the status is the first in this order:
``low_counts`` > ``not_attempted_gate`` > ``not_applicable`` >
``implausible`` > ``parent_unresolved`` > ``below_floor`` >
``not_resolvable`` > ``low_confidence`` > (COP rule) > ``single_method`` /
``method_disagree``; a level passing every check is ``confident``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.config import (
    DEFAULT_HUMAN_REGION,
    AnnotationConfig,
    AnnotationGate,
    AnnotationThresholds,
)
from merxen.annotation.schema import (
    NAMELESS_STATUSES,
    CellStatus,
    Columns,
    meets_threshold,
)
from merxen.annotation.thresholds import (
    EmissionPlan,
    FloorPlan,
    GateVerdict,
    LevelEmission,
    dataset_gate,
)
from merxen.annotation.vocab import (
    COP_SUPERCLUSTER,
    FINAL_LEVELS,
    HUMAN_BROAD_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    HUMAN_LINEAGES,
    NEURONS,
    UNASSIGNED_LABEL,
    human_floor_class,
    load_vocab,
    primary_vocab,
)

if TYPE_CHECKING:
    from merxen.annotation.diagnostics import TrustDecision

logger = logging.getLogger(__name__)

OPC: Final = "Oligodendrocyte precursors"
COP_FLOOR_CLASS: Final = "COP"
WHB: Final = "whb"
SEA: Final = "sea"
LL: Final = "ll"
METHODS: Final[tuple[str, ...]] = (WHB, SEA, LL)
HUMAN_CHAIN: Final[tuple[str, ...]] = FINAL_LEVELS["human"][1:]
HUMAN_SECONDARY_LEVELS: Final[tuple[str, ...]] = ("seaad_subclass",)
HUMAN_FINE_LEVELS: Final[tuple[str, ...]] = ("cluster",)
# Statuses that say nothing about how far a cell's chain got: the deepest
# attempted level is the deepest one outside this set (flags, §4.3).
_NOT_ATTEMPTED: Final[frozenset[str]] = frozenset(
    {
        CellStatus.LOW_COUNTS,
        CellStatus.NOT_ATTEMPTED_GATE,
        CellStatus.NOT_APPLICABLE,
        CellStatus.PARENT_UNRESOLVED,
    }
)
# LL labels that count as neurons (§5.3: "LL OtherNeuron counts as Neurons").
_LL_NEURON_LABELS: Final[frozenset[str]] = frozenset(
    {NEURONS, "Exc", "Inh", "OtherNeuron", "N"}
)


# --------------------------------------------------------------------------
# Degraded modes (§5.3)


@dataclass(frozen=True)
class DegradedModeRule:
    """One row of the §5.3 degraded-mode truth table.

    Attributes:
        name: Mode id.
        methods: The methods available (``whb``, ``sea``, ``ll``).
        primary_available: Whether WHB (the primary) is available; without
            it every table cell is ``not_attempted_gate`` (statuses only).
        below60_votes: Methods of which at least one must agree below 60
            counts (empty: no second method).
        veto_methods: Methods whose confident disagreement blocks a level
            from 60 counts.
        rescue_methods: Methods whose agreement at lineage keeps an
            implausible node's lineage.
        missing_status: Status below 60 counts when no second method exists
            (``single_method``), unless ``allow_single_method``.
        max_tier: The largest consensus tier the mode can give.
        description: The table row in words.
    """

    name: str
    methods: frozenset[str]
    primary_available: bool
    below60_votes: tuple[str, ...]
    veto_methods: tuple[str, ...]
    rescue_methods: tuple[str, ...]
    missing_status: CellStatus | None
    max_tier: int
    description: str

    def to_json(self) -> dict[str, Any]:
        """Return the rule as JSON (provenance)."""
        return {
            "name": self.name,
            "methods": sorted(self.methods),
            "primary_available": self.primary_available,
            "below60_votes": list(self.below60_votes),
            "veto_methods": list(self.veto_methods),
            "rescue_methods": list(self.rescue_methods),
            "missing_status": (
                None if self.missing_status is None else self.missing_status.value
            ),
            "max_tier": self.max_tier,
            "description": self.description,
        }


DEGRADED_MODES: Final[dict[str, DegradedModeRule]] = {
    "whb_sea": DegradedModeRule(
        name="whb_sea",
        methods=frozenset({WHB, SEA}),
        primary_available=True,
        below60_votes=(SEA,),
        veto_methods=(SEA,),
        rescue_methods=(SEA,),
        missing_status=None,
        max_tier=2,
        description="WHB + SEA (v1 default): below 60 counts SEA must agree",
    ),
    "whb_sea_ll": DegradedModeRule(
        name="whb_sea_ll",
        methods=frozenset({WHB, SEA, LL}),
        primary_available=True,
        below60_votes=(SEA, LL),
        veto_methods=(SEA,),
        rescue_methods=(SEA,),
        missing_status=None,
        max_tier=3,
        description=(
            "WHB + SEA + LL (v1.1 table row; OD-B8: not enabled): below 60 "
            "counts SEA or LL agrees (LL OtherNeuron counts as Neurons)"
        ),
    ),
    "whb_ll": DegradedModeRule(
        name="whb_ll",
        methods=frozenset({WHB, LL}),
        primary_available=True,
        below60_votes=(LL,),
        veto_methods=(),
        rescue_methods=(LL,),
        missing_status=None,
        max_tier=2,
        description="WHB + LL (table row; OD-B8: not enabled): LL must agree",
    ),
    "whb_only": DegradedModeRule(
        name="whb_only",
        methods=frozenset({WHB}),
        primary_available=True,
        below60_votes=(),
        veto_methods=(),
        rescue_methods=(),
        missing_status=CellStatus.SINGLE_METHOD,
        max_tier=1,
        description=(
            "WHB only (SEA failed, disabled or refused for this panel): below 60 "
            "counts not confident, single_method (override "
            "annotation_allow_single_method, recorded)"
        ),
    ),
    "primary_missing": DegradedModeRule(
        name="primary_missing",
        methods=frozenset(),
        primary_available=False,
        below60_votes=(),
        veto_methods=(),
        rescue_methods=(),
        missing_status=CellStatus.NOT_ATTEMPTED_GATE,
        max_tier=0,
        description=(
            "SEA only or nothing (WHB failed or refused): statuses only, the "
            "primary is required (not_attempted_gate)"
        ),
    ),
}


def degraded_mode(*, whb: bool, sea: bool, ll: bool = False) -> DegradedModeRule:
    """Return the §5.3 mode of a set of available methods.

    Args:
        whb: The WHB primary is available (mapped and not refused).
        sea: SEA-AD is available (mapped and not refused).
        ll: The likelihood typer's vote is enabled and available (never in
            v1 / v1.1 production, OD-B8).

    Returns:
        The truth-table row.
    """
    if not whb:
        return DEGRADED_MODES["primary_missing"]
    if sea and ll:
        return DEGRADED_MODES["whb_sea_ll"]
    if sea:
        return DEGRADED_MODES["whb_sea"]
    if ll:
        return DEGRADED_MODES["whb_ll"]
    return DEGRADED_MODES["whb_only"]


# --------------------------------------------------------------------------
# Inputs


def _objects(
    values: Sequence[object] | np.ndarray | pd.Series | None, n: int
) -> np.ndarray:
    if values is None:
        return np.full(n, None, dtype=object)
    array = np.asarray(values, dtype=object).copy()
    for index, value in enumerate(array):
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value) or value is pd.NA:
            array[index] = None
        else:
            array[index] = str(value)
    return array


def _floats(
    values: Sequence[float] | np.ndarray | pd.Series | None, n: int
) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class LevelScores:
    """Per-object probability columns of one engine level.

    Attributes:
        raw: Raw probability (coarse levels: the E2 aggregate of the
            assigned node and its same-class runner-ups).
        corr: MMC ``avg_correlation``.
        runner_up: Best alternative (class at coarse levels).
        margin: Raw margin to it.
    """

    raw: np.ndarray
    corr: np.ndarray | None = None
    runner_up: np.ndarray | None = None
    margin: np.ndarray | None = None

    @classmethod
    def of(
        cls,
        raw: Sequence[float] | np.ndarray | pd.Series,
        *,
        corr: Sequence[float] | np.ndarray | pd.Series | None = None,
        runner_up: Sequence[object] | np.ndarray | pd.Series | None = None,
        margin: Sequence[float] | np.ndarray | pd.Series | None = None,
    ) -> LevelScores:
        """Return scores with the columns coerced (floats, object labels)."""
        values = np.asarray(raw, dtype=np.float64)
        n = len(values)
        return cls(
            raw=values,
            corr=None if corr is None else _floats(corr, n),
            runner_up=None if runner_up is None else _objects(runner_up, n),
            margin=None if margin is None else _floats(margin, n),
        )

    def __len__(self) -> int:
        return len(self.raw)


@dataclass(frozen=True)
class LevelCall:
    """An engine's node call at one level: the node name and its scores.

    Attributes:
        name: Assigned node name per object (``None``: not mapped).
        scores: Its probability columns.
    """

    name: np.ndarray
    scores: LevelScores

    @classmethod
    def of(
        cls,
        name: Sequence[object] | np.ndarray | pd.Series,
        raw: Sequence[float] | np.ndarray | pd.Series,
        *,
        corr: Sequence[float] | np.ndarray | pd.Series | None = None,
        runner_up: Sequence[object] | np.ndarray | pd.Series | None = None,
        margin: Sequence[float] | np.ndarray | pd.Series | None = None,
    ) -> LevelCall:
        """Return a call with its columns coerced."""
        scores = LevelScores.of(raw, corr=corr, runner_up=runner_up, margin=margin)
        return cls(name=_objects(name, len(scores)), scores=scores)

    def __len__(self) -> int:
        return len(self.name)


@dataclass(frozen=True)
class WhbCalls:
    """The WHB-frontal primary's calls (§5.2).

    Coarse-level names come from the vocab of the assigned supercluster;
    their probabilities aggregate the supercluster bootstrap probabilities
    of the assigned node and its runner-ups of the same class (E1 / E2).

    Attributes:
        supercluster: Assigned supercluster and its bootstrap probability.
        lineage: Lineage-aggregated scores.
        broad: Broad-aggregated scores.
        nt: NT-aggregated scores (over neuronal nodes).
        cluster: WHB cluster call (report-only fine level), if mapped.
    """

    supercluster: LevelCall
    lineage: LevelScores
    broad: LevelScores
    nt: LevelScores
    cluster: LevelCall | None = None


@dataclass(frozen=True)
class SeaCalls:
    """The SEA-AD Multiregion second vote's calls (§5.2).

    Attributes:
        broad: The 7-class label (vocab; "VLMC & Perivascular" split by
            supertype) and E2's broad probability (``shadow.seaad_broad_calls``).
        subclass: Subclass name and its ``aggregate_probability`` (E2's
            subclass definition, on which 0.55 / 0.45 were derived).
    """

    broad: LevelCall
    subclass: LevelCall


@dataclass(frozen=True)
class HumanCalls:
    """Everything ``resolve_human`` reads per object (all segmented objects).

    Attributes:
        total_counts: Counts after control removal.
        in_table: ``total_counts >= min_counts`` (table cells).
        whb: WHB calls, or ``None`` when WHB failed or was not mapped.
        sea: SEA-AD calls, or ``None`` when SEA-AD failed, was disabled or
            was refused for the panel.
        ll_broad: Likelihood-typer broad labels (7 classes; ``OtherNeuron``
            counts as Neurons). Truth-table use only (OD-B8).
    """

    total_counts: np.ndarray
    in_table: np.ndarray
    whb: WhbCalls | None
    sea: SeaCalls | None = None
    ll_broad: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check that every column has one value per object."""
        n = len(self.total_counts)
        lengths = [len(self.in_table)]
        if self.whb is not None:
            lengths += [
                len(self.whb.supercluster),
                len(self.whb.lineage),
                len(self.whb.broad),
                len(self.whb.nt),
            ]
            if self.whb.cluster is not None:
                lengths.append(len(self.whb.cluster))
        if self.sea is not None:
            lengths += [len(self.sea.broad), len(self.sea.subclass)]
        if self.ll_broad is not None:
            lengths.append(len(self.ll_broad))
        if any(length != n for length in lengths):
            raise ValueError("every HumanCalls column needs one value per object")

    def __len__(self) -> int:
        return len(self.total_counts)


@dataclass(frozen=True)
class HumanResolveSettings:
    """The settings and bundle-derived plans ``resolve_human`` applies.

    Attributes:
        platform: ``MERSCOPE`` or ``XENIUM``.
        min_counts: The hard floor and table threshold.
        emission: The primary's resolvability-gated emission (§8.3).
        floors: The count floors (§5.4).
        thresholds: Threshold settings.
        gate: Dataset gate settings.
        trust: The primary reference's trust decision.
        secondary_trust: SEA-AD's trust decision (``refused``: degraded mode).
        region: Anatomical region (vocab plausibility column).
        allow_single_method: ``annotation_allow_single_method`` (§5.3).
        allow_fine_levels: Report the WHB cluster level (OD-E4).
        use_likelihood_vote: Let the LL typer vote (never in v1 / v1.1
            production, OD-B8; the truth table's LL rows).
        n_segmented: Segmented objects (gate warning denominator); ``None``
            uses the number of objects given (the label table covers every
            segmented object).
        allow_table_below_min_counts: Accept table cells below
            ``min_counts``: a published clustered H5AD's table cells can sum
            lower on its ``min_cells``-filtered genes (MAP's
            ``n_below_min_in_clustered``); they stay table cells and are
            ``below_floor`` at every level.
    """

    platform: str
    min_counts: int
    emission: EmissionPlan
    floors: FloorPlan
    thresholds: AnnotationThresholds = field(default_factory=AnnotationThresholds)
    gate: AnnotationGate = field(default_factory=AnnotationGate)
    trust: TrustDecision | None = None
    secondary_trust: TrustDecision | None = None
    region: str = DEFAULT_HUMAN_REGION
    allow_single_method: bool = False
    allow_fine_levels: bool = False
    use_likelihood_vote: bool = False
    n_segmented: int | None = None
    allow_table_below_min_counts: bool = False

    @classmethod
    def from_config(
        cls,
        config: AnnotationConfig,
        *,
        platform: str,
        emission: EmissionPlan,
        floors: FloorPlan,
        trust: TrustDecision | None,
        secondary_trust: TrustDecision | None = None,
        n_segmented: int | None = None,
    ) -> HumanResolveSettings:
        """Return the settings of a coupled annotation config.

        Args:
            config: The annotation config (``min_counts`` coupled).
            platform: ``MERSCOPE`` or ``XENIUM``.
            emission: The primary's emission plan.
            floors: The floor plan.
            trust: The primary's trust decision.
            secondary_trust: SEA-AD's trust decision.
            n_segmented: Segmented objects, if not every object is given.

        Returns:
            The settings.
        """
        if config.species != "human":
            raise ValueError("resolve_human needs a human annotation config")
        return cls(
            platform=platform,
            min_counts=config.require_min_counts(),
            emission=emission,
            floors=floors,
            thresholds=config.thresholds,
            gate=config.gate,
            trust=trust,
            secondary_trust=secondary_trust,
            region=config.anatomical_region or DEFAULT_HUMAN_REGION,
            allow_single_method=config.allow_single_method,
            allow_fine_levels=config.thresholds.allow_fine_levels,
            n_segmented=n_segmented,
        )


# --------------------------------------------------------------------------
# Outputs


@dataclass
class LevelResult:
    """The ``ct_<L>_*`` values of one level, per object (§4.1).

    Attributes:
        name: Assigned label (``None`` where not applicable / not attempted).
        raw: Raw engine probability.
        conf: Confidence (v1 = raw; ``calibration="none"``).
        corr: MMC ``avg_correlation``.
        runner_up: Best alternative.
        margin: Raw margin.
        status: ``CellStatus`` values.
        validated: Confident and inside the family's validated region.
        threshold: The threshold applied (NaN where none).
        floor: The count floor applied (NaN where none).
        class_key: The class the resolvability table and floors were read
            for.
        extrapolated: The emission rests on a pooled deep set's verdict.
    """

    name: np.ndarray
    raw: np.ndarray
    conf: np.ndarray
    corr: np.ndarray
    runner_up: np.ndarray
    margin: np.ndarray
    status: np.ndarray
    validated: np.ndarray
    threshold: np.ndarray
    floor: np.ndarray
    class_key: np.ndarray
    extrapolated: np.ndarray

    @property
    def confident(self) -> np.ndarray:
        """Whether each object is confident at this level."""
        return np.asarray(self.status == CellStatus.CONFIDENT.value, dtype=bool)

    def to_columns(self, level: str) -> dict[str, Any]:
        """Return the level's label-table columns (schema dtypes)."""
        return {
            Columns.level(level, "name"): pd.Categorical(self.name),
            Columns.level(level, "raw"): self.raw.astype(np.float32),
            Columns.level(level, "conf"): self.conf.astype(np.float32),
            Columns.level(level, "corr"): self.corr.astype(np.float32),
            Columns.level(level, "runner_up"): pd.Categorical(self.runner_up),
            Columns.level(level, "margin"): self.margin.astype(np.float32),
            Columns.level(level, "status"): pd.Categorical(
                self.status, categories=[status.value for status in CellStatus]
            ),
            Columns.level(level, "validated"): self.validated.astype(bool),
        }


@dataclass
class HumanResolution:
    """The output of ``resolve_human`` (one entry per object).

    Attributes:
        levels: Per level: ``LevelResult`` (chain levels, ``seaad_subclass``,
            and ``cluster`` with ``allow_fine_levels``).
        final_level: ``ct_final_level``.
        final_name: ``ct_final_name``.
        consensus_tier: ``ct_consensus_tier``.
        flags: ``flag_low_counts``, ``flag_below_floor``,
            ``flag_method_disagree``, ``flag_implausible``,
            ``flag_cop_suppressed`` and ``exclude_hard``.
        gate: The dataset gate verdict.
        mode: The degraded mode applied.
        single_method_override: WHB decided alone below 60 counts
            (``allow_single_method`` in the WHB-only mode).
        depth_bin: Each object's grid bin (NaN below the grid).
        resolvability_extrapolated: A confident level rests on a pooled deep
            set's verdict.
        emissions: The emission of every level (provenance).
        in_table: Table cells.
        floor_warnings: Levels whose floors follow the unknown-panel rule.
    """

    levels: dict[str, LevelResult]
    final_level: np.ndarray
    final_name: np.ndarray
    consensus_tier: np.ndarray
    flags: dict[str, np.ndarray]
    gate: GateVerdict
    mode: DegradedModeRule
    single_method_override: bool
    depth_bin: np.ndarray
    resolvability_extrapolated: np.ndarray
    emissions: dict[str, LevelEmission]
    in_table: np.ndarray
    floor_warnings: list[str] = field(default_factory=list)

    def status(self, level: str) -> np.ndarray:
        """Return one level's statuses."""
        return self.levels[level].status

    def to_columns(self) -> dict[str, Any]:
        """Return the label-table columns ``resolve_human`` decides (§4.1, §4.3).

        Identity, soft-composition, raw-engine and report-only flag columns
        are added by the caller.

        Returns:
            Column name to values (schema dtypes).
        """
        columns: dict[str, Any] = {}
        for level, result in self.levels.items():
            columns.update(result.to_columns(level))
        depth = pd.array(
            [
                None if not np.isfinite(value) else int(value)
                for value in self.depth_bin
            ],
            dtype=pd.Int32Dtype(),
        )
        columns[Columns.DEPTH_BIN] = depth
        columns[Columns.RESOLVABILITY_EXTRAPOLATED] = self.resolvability_extrapolated
        columns[Columns.CT_FINAL_LEVEL] = pd.Categorical(
            self.final_level, categories=list(FINAL_LEVELS["human"])
        )
        columns[Columns.CT_FINAL_NAME] = pd.Categorical(self.final_name)
        columns[Columns.CT_CONSENSUS_TIER] = self.consensus_tier.astype(np.int8)
        for name, values in self.flags.items():
            columns[name] = values.astype(bool)
        return columns

    def summary(self) -> dict[str, Any]:
        """Return the provenance summary (§4.6, resolve summary §3.4).

        Returns:
            Degraded mode, gate, confident share per level of table cells and
            of segmented objects, resolvable share per level, thresholds and
            emission per level, COP control shares (H5) and flag counts.
        """
        table = self.in_table
        n_table = int(table.sum())
        n_objects = len(table)
        # The gate's denominator (§4.4 / §5.4): the segmented objects when the
        # caller gave them (published clustered inputs hold only table
        # cells), else every object of the label table.
        n_segmented = (
            self.gate.n_segmented if self.gate.n_segmented is not None else n_objects
        )
        per_level: dict[str, Any] = {}
        for level, result in self.levels.items():
            confident = result.confident
            emission = self.emissions.get(level)
            per_level[level] = {
                "confident_share_table": (
                    float(confident[table].mean()) if n_table else None
                ),
                "confident_share_segmented": (
                    float(confident.sum() / n_segmented) if n_segmented else None
                ),
                "validated_share": (
                    float(result.validated[confident].mean())
                    if confident.any()
                    else None
                ),
                "status_counts": _counts(result.status[table]),
                "emission": None if emission is None else emission.summary(table),
            }
        supercluster = self.levels["supercluster"]
        broad = self.levels["broad"]
        cop_confident = supercluster.confident & (supercluster.name == COP_SUPERCLUSTER)
        opc_confident = broad.confident & (broad.name == OPC)
        from_cop = opc_confident & (self.levels["supercluster"].class_key == "COP")
        return {
            "degraded_mode": self.mode.to_json(),
            "single_method_override": self.single_method_override,
            "gate": self.gate.to_json(),
            "n_objects": n_objects,
            "n_table": n_table,
            "final_level_counts": _counts(self.final_level[table]),
            "levels": per_level,
            "cop_control": {
                "confident_cop_supercluster_share": (
                    float(cop_confident[table].mean()) if n_table else None
                ),
                "confident_broad_opc_share": (
                    float(opc_confident[table].mean()) if n_table else None
                ),
                "cop_derived_opc_share": (
                    float(from_cop.sum() / opc_confident.sum())
                    if opc_confident.any()
                    else None
                ),
                "cop_suppressed": int(self.flags[Columns.FLAG_COP_SUPPRESSED].sum()),
            },
            "flag_counts": {
                name: int(values[table].sum()) for name, values in self.flags.items()
            },
            "resolvability_extrapolated_share": (
                float(self.resolvability_extrapolated[table].mean())
                if n_table
                else None
            ),
            "floor_warnings": list(self.floor_warnings),
        }


def _counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


# --------------------------------------------------------------------------
# Generic pieces (human and, in M6, mouse)


class _StatusBuilder:
    """First failing check wins (the module's precedence order)."""

    def __init__(self, n: int) -> None:
        self.values = np.full(n, CellStatus.CONFIDENT.value, dtype=object)
        self._unset = np.ones(n, dtype=bool)

    def fail(self, mask: np.ndarray, status: CellStatus) -> None:
        hit = self._unset & np.asarray(mask, dtype=bool)
        self.values[hit] = status.value
        self._unset &= ~hit

    def unset(self) -> np.ndarray:
        return self._unset.copy()

    def finish(self) -> np.ndarray:
        return self.values.copy()


def final_label(
    levels: Mapping[str, LevelResult], *, species: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``ct_final_level`` and ``ct_final_name``: the deepest confident level.

    Args:
        levels: Per-level results of the species' chain (``FINAL_LEVELS``;
            report-only and secondary levels are ignored).
        species: ``"human"`` or ``"mouse"``.

    Returns:
        ``(final_level, final_name)``; ``none`` / ``Mixed/Unknown`` where no
        level is confident.
    """
    chain = FINAL_LEVELS[species][1:]
    first = next(iter(levels.values()))
    n = len(first.status)
    final_level = np.full(n, "none", dtype=object)
    final_name = np.full(n, UNASSIGNED_LABEL, dtype=object)
    for level in chain:
        result = levels.get(level)
        if result is None:
            continue
        confident = result.confident
        final_level[confident] = level
        final_name[confident] = result.name[confident]
    return final_level, final_name


def consensus_tier(
    labels: Sequence[np.ndarray],
    informative: Sequence[np.ndarray],
    *,
    max_tier: int,
) -> np.ndarray:
    """Return the consensus tier: informative methods agreeing (§4.1).

    A method is informative for a cell when its 7-class call meets its
    threshold. The tier is the size of the largest group of informative
    methods on one label, 0 when two or more are informative and all
    disagree (confident disagreement) or none is, 1 when only one is;
    capped at the degraded mode's ``max_tier``. Agreement, not accuracy.

    Args:
        labels: One label array per method (``None``: no 7-class call).
        informative: One boolean array per method.
        max_tier: The mode's maximum tier.

    Returns:
        ``int8`` tiers.
    """
    if not labels:
        return np.zeros(0, dtype=np.int8)
    n = len(labels[0])
    tier = np.zeros(n, dtype=np.int8)
    for index in range(n):
        votes = [
            str(label[index])
            for label, usable in zip(labels, informative, strict=True)
            if bool(usable[index]) and label[index] is not None
        ]
        if not votes:
            continue
        if len(votes) == 1:
            tier[index] = 1
            continue
        largest = max(votes.count(value) for value in set(votes))
        tier[index] = largest if largest >= 2 else 0
    return np.minimum(tier, max_tier).astype(np.int8)


# --------------------------------------------------------------------------
# resolve_human


@dataclass(frozen=True)
class _NodeInfo:
    """Vocab properties of each object's assigned WHB supercluster."""

    has_call: np.ndarray
    implausible: np.ndarray
    lineage: np.ndarray
    broad: np.ndarray
    nt: np.ndarray
    neuron_lineage: np.ndarray
    broad_key: np.ndarray
    supercluster_key: np.ndarray
    is_cop: np.ndarray


def _node_info(names: np.ndarray, region: str) -> _NodeInfo:
    vocab = primary_vocab("human")
    n = len(names)
    cache: dict[str, tuple[Any, ...]] = {}
    has_call = np.array([name is not None for name in names], dtype=bool)
    implausible = np.zeros(n, dtype=bool)
    lineage = np.full(n, None, dtype=object)
    broad = np.full(n, None, dtype=object)
    nt = np.full(n, None, dtype=object)
    broad_key = np.full(n, None, dtype=object)
    supercluster_key = np.full(n, None, dtype=object)
    for index, name in enumerate(names):
        if name is None:
            continue
        if name not in cache:
            if name not in vocab:
                logger.warning("WHB node %r is not in the vocab: implausible", name)
                cache[name] = (True, None, None, None, None, None)
            else:
                bad = vocab.is_sink(name) or not vocab.is_region_plausible(name, region)
                node_broad = vocab.broad_class(name)
                node_nt = vocab.nt(name)
                key = None if bad else human_floor_class(node_broad, node_nt)
                supc_key = (
                    COP_FLOOR_CLASS if (key and name == COP_SUPERCLUSTER) else key
                )
                cache[name] = (
                    bad,
                    vocab.lineage(name),
                    node_broad,
                    node_nt,
                    key,
                    supc_key,
                )
        entry = cache[name]
        implausible[index] = entry[0]
        lineage[index], broad[index], nt[index] = entry[1], entry[2], entry[3]
        broad_key[index], supercluster_key[index] = entry[4], entry[5]
    neuron_lineage = np.array([value == NEURONS for value in lineage], dtype=bool)
    is_cop = np.array([name == COP_SUPERCLUSTER for name in names], dtype=bool)
    return _NodeInfo(
        has_call=has_call,
        implausible=implausible,
        lineage=lineage,
        broad=broad,
        nt=nt,
        neuron_lineage=neuron_lineage,
        broad_key=broad_key,
        supercluster_key=supercluster_key,
        is_cop=is_cop,
    )


def _in(values: np.ndarray, allowed: Sequence[str]) -> np.ndarray:
    allowed_set = set(allowed)
    return np.array([value in allowed_set for value in values], dtype=bool)


def _equal(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a is not None and b is not None and a == b
            for a, b in zip(first, second, strict=True)
        ],
        dtype=bool,
    )


def _keep(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return ``values`` where ``mask`` holds, ``None`` elsewhere (objects)."""
    kept = np.full(len(values), None, dtype=object)
    keep = np.asarray(mask, dtype=bool)
    kept[keep] = np.asarray(values, dtype=object)[keep]
    return kept


def _map(values: np.ndarray, mapping: Mapping[str, str]) -> np.ndarray:
    return np.array(
        [None if value is None else mapping.get(str(value)) for value in values],
        dtype=object,
    )


@dataclass(frozen=True)
class _Votes:
    """Per-cell second-vote outcomes at lineage and at the 7-class level."""

    lineage_ok: np.ndarray
    broad_ok: np.ndarray
    lineage_status: np.ndarray
    broad_status: np.ndarray
    rescued: np.ndarray
    sea_confident_opc: np.ndarray
    sea_class: np.ndarray
    sea_informative: np.ndarray
    ll_class: np.ndarray
    sea_key: np.ndarray


def _second_votes(
    calls: HumanCalls,
    info: _NodeInfo,
    mode: DegradedModeRule,
    settings: HumanResolveSettings,
    *,
    use_sea: bool,
    use_ll: bool,
) -> _Votes:
    """Apply the mode's truth-table row to every cell (§5.3)."""
    n = len(calls)
    thresholds = settings.thresholds
    counts = np.asarray(calls.total_counts, dtype=np.float64)
    below = counts < thresholds.second_vote_below_counts
    agree_lineage: dict[str, np.ndarray] = {}
    agree_broad: dict[str, np.ndarray] = {}
    disagree_lineage: dict[str, np.ndarray] = {}
    disagree_broad: dict[str, np.ndarray] = {}
    sea_class = np.full(n, None, dtype=object)
    sea_confident = np.zeros(n, dtype=bool)
    sea_key = np.full(n, None, dtype=object)
    if use_sea and calls.sea is not None:
        labels = calls.sea.broad.name
        sea_class = _keep(_in(labels, HUMAN_BROAD_CLASSES), labels)
        sea_confident = np.array(
            [label is not None for label in sea_class], dtype=bool
        ) & meets_threshold(calls.sea.broad.scores.raw, thresholds.seaad_broad)
        sea_lineage = _map(sea_class, HUMAN_LINEAGE_OF_BROAD_CLASS)
        agree_lineage[SEA] = _equal(sea_lineage, info.lineage)
        agree_broad[SEA] = _equal(sea_class, info.broad)
        disagree_lineage[SEA] = sea_confident & ~agree_lineage[SEA]
        disagree_broad[SEA] = sea_confident & ~agree_broad[SEA]
        sea_vocab = load_vocab("seaad_mr_subclass")
        subclasses = calls.sea.subclass.name
        for index in range(n):
            label = sea_class[index]
            if label is None:
                continue
            subclass = subclasses[index]
            sea_nt = (
                sea_vocab.nt(subclass)
                if subclass is not None and subclass in sea_vocab
                else None
            )
            sea_key[index] = human_floor_class(label, sea_nt)
    ll_class: np.ndarray = np.full(n, None, dtype=object)
    if use_ll and calls.ll_broad is not None:
        raw_ll = _objects(calls.ll_broad, n)
        ll_class = np.array(
            [
                NEURONS
                if value in _LL_NEURON_LABELS
                else (value if value in HUMAN_BROAD_CLASSES else None)
                for value in raw_ll
            ],
            dtype=object,
        )
        agree_lineage[LL] = _equal(
            _map(ll_class, HUMAN_LINEAGE_OF_BROAD_CLASS), info.lineage
        )
        agree_broad[LL] = _equal(ll_class, info.broad)
    missing = np.zeros(n, dtype=bool)

    def ok_and_status(
        agree: Mapping[str, np.ndarray], disagree: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        votes = [agree[method] for method in mode.below60_votes if method in agree]
        if votes:
            below_ok = np.logical_or.reduce(votes)
            below_status = CellStatus.METHOD_DISAGREE
        elif mode.missing_status is not None and not settings.allow_single_method:
            below_ok = missing.copy()
            below_status = mode.missing_status
        else:
            below_ok = ~missing
            below_status = CellStatus.METHOD_DISAGREE
        vetoes = [
            disagree[method] for method in mode.veto_methods if method in disagree
        ]
        from60_ok = ~np.logical_or.reduce(vetoes) if vetoes else ~missing
        ok = np.where(below, below_ok, from60_ok)
        status = np.where(
            below, below_status.value, CellStatus.METHOD_DISAGREE.value
        ).astype(object)
        return np.asarray(ok, dtype=bool), status

    lineage_ok, lineage_status = ok_and_status(agree_lineage, disagree_lineage)
    broad_ok, broad_status = ok_and_status(agree_broad, disagree_broad)
    rescues = [
        agree_lineage[method]
        for method in mode.rescue_methods
        if method in agree_lineage
    ]
    rescued = np.logical_or.reduce(rescues) if rescues else missing.copy()
    return _Votes(
        lineage_ok=lineage_ok,
        broad_ok=broad_ok,
        lineage_status=lineage_status,
        broad_status=broad_status,
        rescued=np.asarray(rescued, dtype=bool),
        sea_confident_opc=sea_confident & (sea_class == OPC),
        sea_class=sea_class,
        sea_informative=sea_confident,
        ll_class=ll_class,
        sea_key=sea_key,
    )


def _level_result(
    n: int,
    *,
    name: np.ndarray,
    scores: LevelScores | None,
    status: np.ndarray,
    threshold: np.ndarray,
    floor: np.ndarray,
    class_key: np.ndarray,
    extrapolated: np.ndarray,
) -> LevelResult:
    raw = np.full(n, np.nan) if scores is None else scores.raw.astype(np.float64).copy()
    corr = (
        np.full(n, np.nan)
        if scores is None or scores.corr is None
        else scores.corr.astype(np.float64).copy()
    )
    runner_up = (
        np.full(n, None, dtype=object)
        if scores is None or scores.runner_up is None
        else scores.runner_up.copy()
    )
    margin = (
        np.full(n, np.nan)
        if scores is None or scores.margin is None
        else scores.margin.astype(np.float64).copy()
    )
    names = np.asarray(name, dtype=object).copy()
    nameless = _in(status, [str(value) for value in NAMELESS_STATUSES])
    names[nameless] = None
    runner_up[nameless] = None
    not_applicable = status == CellStatus.NOT_APPLICABLE.value
    low_counts = status == CellStatus.LOW_COUNTS.value
    blank = not_applicable | low_counts
    raw[blank] = np.nan
    corr[blank] = np.nan
    margin[blank] = np.nan
    return LevelResult(
        name=names,
        raw=raw,
        conf=raw.copy(),
        corr=corr,
        runner_up=runner_up,
        margin=margin,
        status=status,
        validated=np.zeros(n, dtype=bool),
        threshold=np.where(blank, np.nan, threshold),
        floor=np.where(blank, np.nan, floor),
        class_key=np.asarray(class_key, dtype=object),
        extrapolated=np.asarray(extrapolated, dtype=bool) & (status == "confident"),
    )


def resolve_human(calls: HumanCalls, settings: HumanResolveSettings) -> HumanResolution:
    """Apply the human v1 rules to one sample (plan §5.2-§5.4, §8.2-§8.3).

    Args:
        calls: Per-object calls (every segmented object).
        settings: Thresholds, floors, emission, gate, trust and degraded-mode
            settings.

    Returns:
        Statuses, labels, tier, flags and the gate verdict per object.
    """
    n = len(calls)
    thresholds = settings.thresholds
    counts = np.asarray(calls.total_counts, dtype=np.float64)
    table = np.asarray(calls.in_table, dtype=bool)
    below_min = table & (counts < settings.min_counts)
    if bool(below_min.any()):
        if not settings.allow_table_below_min_counts:
            raise ValueError("in_table objects must reach min_counts")
        logger.warning(
            "%d table cell(s) below min_counts %d (a min_cells-filtered published "
            "table): below_floor at every level",
            int(below_min.sum()),
            settings.min_counts,
        )
    primary_refused = settings.trust is not None and settings.trust.state == "refused"
    sea_refused = (
        settings.secondary_trust is not None
        and settings.secondary_trust.state == "refused"
    )
    use_ll = bool(settings.use_likelihood_vote and calls.ll_broad is not None)
    if use_ll:
        logger.warning(
            "The likelihood-typer vote is enabled; OD-B8 keeps it off in v1.1"
        )
    mode = degraded_mode(
        whb=calls.whb is not None and not primary_refused,
        sea=calls.sea is not None and not sea_refused,
        ll=use_ll,
    )
    use_sea = SEA in mode.methods
    single_method_override = bool(
        mode.missing_status == CellStatus.SINGLE_METHOD and settings.allow_single_method
    )
    if single_method_override:
        logger.warning(
            "Degraded mode %s with annotation_allow_single_method: WHB decides "
            "alone below %d counts (recorded)",
            mode.name,
            thresholds.second_vote_below_counts,
        )
    whb = calls.whb
    names = whb.supercluster.name if whb is not None else np.full(n, None, dtype=object)
    info = _node_info(names, settings.region)
    votes = _second_votes(calls, info, mode, settings, use_sea=use_sea, use_ll=use_ll)
    emission = settings.emission
    floors = settings.floors
    low_counts = ~table
    no_call = ~info.has_call

    # Class keys (the called class; §8.3): E2 floor classes of plausible nodes
    # (COP separate at supercluster); a rescued implausible lineage is keyed by
    # the SEA-AD class it rests on.
    lineage_key = info.broad_key.copy()
    rescue_key = info.implausible & votes.rescued
    lineage_key[rescue_key] = votes.sea_key[rescue_key]

    def emit(level: str, keys: np.ndarray) -> LevelEmission:
        return emission.level(level, keys, counts)

    emissions: dict[str, LevelEmission] = {}
    levels: dict[str, LevelResult] = {}

    # 1. Lineage
    em = emit("lineage", lineage_key)
    emissions["lineage"] = em
    floor = floors.per_cell("lineage", lineage_key)
    builder = _StatusBuilder(n)
    builder.fail(low_counts, CellStatus.LOW_COUNTS)
    builder.fail(no_call, CellStatus.LOW_CONFIDENCE)
    builder.fail(info.implausible & ~votes.rescued, CellStatus.IMPLAUSIBLE)
    builder.fail(~_in(info.lineage, HUMAN_LINEAGES), CellStatus.IMPLAUSIBLE)
    builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
    builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
    lineage_raw = whb.lineage.raw if whb is not None else np.full(n, np.nan)
    builder.fail(~meets_threshold(lineage_raw, em.threshold), CellStatus.LOW_CONFIDENCE)
    for status in (CellStatus.SINGLE_METHOD, CellStatus.METHOD_DISAGREE):
        builder.fail(~votes.lineage_ok & (votes.lineage_status == status.value), status)
    levels["lineage"] = _level_result(
        n,
        name=info.lineage,
        scores=None if whb is None else whb.lineage,
        status=builder.finish(),
        threshold=em.threshold,
        floor=floor,
        class_key=lineage_key,
        extrapolated=em.extrapolated,
    )
    lineage_confident = levels["lineage"].confident

    # 2. Broad, with the COP rule
    em = emit("broad", info.broad_key)
    emissions["broad"] = em
    supercluster_em = emit("supercluster", info.supercluster_key)
    emissions["supercluster"] = supercluster_em
    floor = floors.per_cell("broad", info.broad_key)
    cop_floor = floors.per_cell(
        "supercluster", np.full(n, COP_FLOOR_CLASS, dtype=object)
    )
    supercluster_raw = (
        whb.supercluster.scores.raw if whb is not None else np.full(n, np.nan)
    )
    cop_supercluster = (counts >= cop_floor) & meets_threshold(
        supercluster_raw, supercluster_em.threshold
    )
    cop_passes = cop_supercluster | votes.sea_confident_opc
    builder = _StatusBuilder(n)
    builder.fail(low_counts, CellStatus.LOW_COUNTS)
    builder.fail(no_call, CellStatus.LOW_CONFIDENCE)
    builder.fail(info.implausible, CellStatus.IMPLAUSIBLE)
    builder.fail(~lineage_confident, CellStatus.PARENT_UNRESOLVED)
    builder.fail(~_in(info.broad, HUMAN_BROAD_CLASSES), CellStatus.IMPLAUSIBLE)
    builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
    builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
    broad_raw = whb.broad.raw if whb is not None else np.full(n, np.nan)
    builder.fail(~meets_threshold(broad_raw, em.threshold), CellStatus.LOW_CONFIDENCE)
    cop_suppressed = builder.unset() & info.is_cop & ~cop_passes
    builder.fail(cop_suppressed & (counts < cop_floor), CellStatus.BELOW_FLOOR)
    builder.fail(cop_suppressed, CellStatus.LOW_CONFIDENCE)
    for status in (CellStatus.SINGLE_METHOD, CellStatus.METHOD_DISAGREE):
        builder.fail(~votes.broad_ok & (votes.broad_status == status.value), status)
    levels["broad"] = _level_result(
        n,
        name=info.broad,
        scores=None if whb is None else whb.broad,
        status=builder.finish(),
        threshold=em.threshold,
        floor=floor,
        class_key=info.broad_key,
        extrapolated=em.extrapolated,
    )
    broad_confident = levels["broad"].confident

    # 3. NT
    em = emit("nt", info.broad_key)
    emissions["nt"] = em
    floor = floors.per_cell("nt", info.broad_key)
    nt_names = _keep(info.neuron_lineage, info.nt)
    builder = _StatusBuilder(n)
    builder.fail(low_counts, CellStatus.LOW_COUNTS)
    builder.fail(no_call, CellStatus.LOW_CONFIDENCE)
    builder.fail(~info.neuron_lineage, CellStatus.NOT_APPLICABLE)
    builder.fail(info.implausible, CellStatus.IMPLAUSIBLE)
    builder.fail(~broad_confident, CellStatus.PARENT_UNRESOLVED)
    builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
    builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
    nt_raw = whb.nt.raw if whb is not None else np.full(n, np.nan)
    builder.fail(~meets_threshold(nt_raw, em.threshold), CellStatus.LOW_CONFIDENCE)
    levels["nt"] = _level_result(
        n,
        name=nt_names,
        scores=None if whb is None else whb.nt,
        status=builder.finish(),
        threshold=em.threshold,
        floor=floor,
        class_key=info.broad_key,
        extrapolated=em.extrapolated,
    )
    nt_applicable = levels["nt"].status != CellStatus.NOT_APPLICABLE.value
    leaf_parent = np.where(nt_applicable, levels["nt"].confident, broad_confident)

    # Dataset gate (depends on broad only; the leaf levels depend on it).
    gate_first = dataset_gate(
        counts[table],
        broad_confident[table],
        n_segmented=settings.n_segmented if settings.n_segmented is not None else n,
        gate=settings.gate,
        trust=settings.trust,
    )
    leaf_gated = np.full(n, not gate_first.attempts_leaf, dtype=bool) & table

    # 4. Supercluster
    em = supercluster_em
    floor = floors.per_cell("supercluster", info.supercluster_key)
    builder = _StatusBuilder(n)
    builder.fail(low_counts, CellStatus.LOW_COUNTS)
    builder.fail(leaf_gated, CellStatus.NOT_ATTEMPTED_GATE)
    builder.fail(no_call, CellStatus.LOW_CONFIDENCE)
    builder.fail(info.implausible, CellStatus.IMPLAUSIBLE)
    builder.fail(~leaf_parent, CellStatus.PARENT_UNRESOLVED)
    builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
    builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
    builder.fail(
        ~meets_threshold(supercluster_raw, em.threshold), CellStatus.LOW_CONFIDENCE
    )
    levels["supercluster"] = _level_result(
        n,
        name=names,
        scores=None if whb is None else whb.supercluster.scores,
        status=builder.finish(),
        threshold=em.threshold,
        floor=floor,
        class_key=info.supercluster_key,
        extrapolated=em.extrapolated,
    )

    # 5. SEA-AD subclass (secondary name; never in ct_final)
    em = emit("seaad_subclass", info.broad_key)
    emissions["seaad_subclass"] = em
    floor = floors.per_cell("seaad_subclass", info.broad_key)
    builder = _StatusBuilder(n)
    builder.fail(low_counts, CellStatus.LOW_COUNTS)
    sea = calls.sea if use_sea else None
    builder.fail(
        np.full(n, sea is None, dtype=bool) | leaf_gated, CellStatus.NOT_ATTEMPTED_GATE
    )
    sea_names = sea.subclass.name if sea is not None else np.full(n, None, dtype=object)
    builder.fail(
        np.array([value is None for value in sea_names]), CellStatus.LOW_CONFIDENCE
    )
    builder.fail(~broad_confident, CellStatus.PARENT_UNRESOLVED)
    builder.fail(~_equal(votes.sea_class, info.broad), CellStatus.METHOD_DISAGREE)
    builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
    builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
    sea_raw = sea.subclass.scores.raw if sea is not None else np.full(n, np.nan)
    builder.fail(~meets_threshold(sea_raw, em.threshold), CellStatus.LOW_CONFIDENCE)
    levels["seaad_subclass"] = _level_result(
        n,
        name=sea_names,
        scores=None if sea is None else sea.subclass.scores,
        status=builder.finish(),
        threshold=em.threshold,
        floor=floor,
        class_key=info.broad_key,
        extrapolated=em.extrapolated,
    )

    # 6. WHB cluster (report-only; never a leaf or in ct_final)
    if settings.allow_fine_levels:
        em = emit("cluster", info.supercluster_key)
        emissions["cluster"] = em
        floor = floors.per_cell("cluster", info.supercluster_key)
        cluster = whb.cluster if whb is not None else None
        builder = _StatusBuilder(n)
        builder.fail(low_counts, CellStatus.LOW_COUNTS)
        builder.fail(leaf_gated, CellStatus.NOT_ATTEMPTED_GATE)
        cluster_names = (
            cluster.name if cluster is not None else np.full(n, None, dtype=object)
        )
        builder.fail(
            np.array([value is None for value in cluster_names]),
            CellStatus.LOW_CONFIDENCE,
        )
        builder.fail(info.implausible, CellStatus.IMPLAUSIBLE)
        builder.fail(~levels["supercluster"].confident, CellStatus.PARENT_UNRESOLVED)
        builder.fail(counts < floor, CellStatus.BELOW_FLOOR)
        builder.fail(~em.emitted, CellStatus.NOT_RESOLVABLE)
        cluster_raw = cluster.scores.raw if cluster is not None else np.full(n, np.nan)
        builder.fail(
            ~meets_threshold(cluster_raw, em.threshold), CellStatus.LOW_CONFIDENCE
        )
        levels["cluster"] = _level_result(
            n,
            name=cluster_names,
            scores=None if cluster is None else cluster.scores,
            status=builder.finish(),
            threshold=em.threshold,
            floor=floor,
            class_key=info.supercluster_key,
            extrapolated=em.extrapolated,
        )

    # Statuses only without the primary, and nothing attempted when the gate
    # fails (§4.2 not_attempted_gate; exclude_hard, §4.3).
    blocked = table & (
        np.full(n, not mode.primary_available, dtype=bool)
        | np.full(n, gate_first.level == "failed", dtype=bool)
    )
    if blocked.any():
        for result in levels.values():
            result.status[blocked] = CellStatus.NOT_ATTEMPTED_GATE.value
            result.name[blocked] = None
            result.runner_up[blocked] = None
            result.extrapolated[blocked] = False
        cop_suppressed = cop_suppressed & ~blocked

    # ct_<L>_validated (§4.1) and the simulation-family gate warning.
    validated_share: dict[str, float | None] = {}
    for level, result in levels.items():
        if settings.trust is None:
            continue
        result.validated = settings.trust.validated_mask(
            level, result.class_key, counts, result.confident
        )
        if level in HUMAN_CHAIN:
            confident = result.confident
            validated_share[level] = (
                float(result.validated[confident].mean()) if confident.any() else None
            )
    gate = dataset_gate(
        counts[table],
        broad_confident[table],
        n_segmented=settings.n_segmented if settings.n_segmented is not None else n,
        gate=settings.gate,
        trust=settings.trust,
        validated_share=validated_share,
    )
    if gate.level != gate_first.level:
        raise AssertionError("the gate level cannot depend on the leaf levels")

    chain = {level: levels[level] for level in HUMAN_CHAIN}
    final_level, final_name = final_label(chain, species="human")
    whb_informative = (
        table
        & info.has_call
        & ~info.implausible
        & _in(info.broad, HUMAN_BROAD_CLASSES)
        & meets_threshold(broad_raw, thresholds.whb_broad)
    )
    method_labels = [_keep(whb_informative, info.broad)]
    method_informative = [whb_informative]
    if use_sea:
        method_labels.append(votes.sea_class)
        method_informative.append(votes.sea_informative & table)
    if use_ll:
        method_labels.append(votes.ll_class)
        method_informative.append(
            np.array([value is not None for value in votes.ll_class]) & table
        )
    tier = consensus_tier(method_labels, method_informative, max_tier=mode.max_tier)
    tier[~table] = 0
    if not mode.primary_available:
        tier[:] = 0

    flags = _human_flags(chain, table, cop_suppressed, gate.level)
    extrapolated = np.logical_or.reduce(
        [result.extrapolated & result.confident for result in chain.values()]
    )
    resolution = HumanResolution(
        levels=levels,
        final_level=final_level,
        final_name=final_name,
        consensus_tier=tier,
        flags=flags,
        gate=gate,
        mode=mode,
        single_method_override=single_method_override,
        depth_bin=emission.depth_bins(counts),
        resolvability_extrapolated=np.asarray(extrapolated, dtype=bool) & table,
        emissions=emissions,
        in_table=table,
        floor_warnings=floors.warnings(),
    )
    logger.info(
        "resolve_human: mode %s, gate %s%s; confident share of %d table cells: %s",
        mode.name,
        gate.level,
        " + warning" if gate.warning else "",
        int(table.sum()),
        ", ".join(
            f"{level} {levels[level].confident[table].mean():.3f}"
            for level in HUMAN_CHAIN
            if table.any()
        ),
    )
    return resolution


def _human_flags(
    chain: Mapping[str, LevelResult],
    table: np.ndarray,
    cop_suppressed: np.ndarray,
    gate_level: str,
) -> dict[str, np.ndarray]:
    """Return the status-mirroring flags and ``exclude_hard`` (§4.3)."""
    n = len(table)
    deepest = np.full(n, None, dtype=object)
    for result in chain.values():
        attempted = ~_in(result.status, list(_NOT_ATTEMPTED))
        deepest[attempted] = result.status[attempted]
    statuses = np.stack([result.status for result in chain.values()], axis=1)
    implausible_any = (statuses == CellStatus.IMPLAUSIBLE.value).any(axis=1)
    only_implausible = np.isin(
        statuses,
        [
            CellStatus.IMPLAUSIBLE.value,
            CellStatus.NOT_APPLICABLE.value,
            CellStatus.NOT_ATTEMPTED_GATE.value,
        ],
    ).all(axis=1)
    exclude_hard = ~table | (implausible_any & only_implausible)
    if gate_level == "failed":
        exclude_hard = np.ones(n, dtype=bool)
    return {
        Columns.FLAG_LOW_COUNTS: ~table,
        Columns.FLAG_BELOW_FLOOR: deepest == CellStatus.BELOW_FLOOR.value,
        Columns.FLAG_METHOD_DISAGREE: deepest == CellStatus.METHOD_DISAGREE.value,
        Columns.FLAG_IMPLAUSIBLE: deepest == CellStatus.IMPLAUSIBLE.value,
        Columns.FLAG_COP_SUPPRESSED: np.asarray(cop_suppressed, dtype=bool) & table,
        Columns.EXCLUDE_HARD: np.asarray(exclude_hard, dtype=bool),
    }
