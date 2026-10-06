"""Downgrade-only QC of real datasets against the simulation (plan §8.8; M3c).

Real in-house datasets of every panel family run label-free production QC
that can only **warn or downgrade, never promote** (OD-E1 as amended on
2026-09-28, D-G9 / D-G13): a passing check changes neither the trust state
nor the margins, and no check here changes emission, a threshold, a floor or
a label. The functions are pure (tables in, records out); RESOLVE wires them
per dataset (the M4 follow-up, plan §12 M3c) and M13 wires the first
in-house dataset of a family.

Checks added by M3c (plan §8.8 table; §8.3 v7.5, v7.9; §8.10):

* ``coverage_vs_simulation`` -- per (level, called class) with at least
  ``min_cells`` (200) dataset cells, the dataset's confident share against
  the class-depth prediction ``sum_d s_c(d) cov(L, c, d)`` at the dataset's
  **own** per-class bin shares ``s_c(d)`` (``predicted_class_coverage``, the
  sums of ``resolvability_class_depth.parquet``), the pre-registered
  predictor. A warning per (level, class) when real < simulated - 0.10 (user
  decision 4); never an offset. Where profile mode has run, its per-class
  prediction is reported beside it (``profile_coverage_table``) and never
  decides a warning (orchestrator decision D4 of 2026-09-29, pending the
  user's confirmation; pre-registration §22.2). The warning is worded per
  class, not as a glial warning: it fires on vendor-segmented 5K glia
  (simulated glial coverage is an upper bound, -.06 to -.18; -.04 to -.14
  re-segmented with ProSeg), on small hypothalamic classes (stage D:
  CNU-HYa GABA, HY GABA, CNU-HYa Glut, HY Glut) and on v1-type large-mask
  (nucleus-expansion) segmentation, where simulation over-predicts coverage
  by +.16 to +.22 whatever the efficiency model (``5k_real/sim/REPORT.txt``
  §7).
* ``nonneuronal_high_depth_flags`` -- the report-only per-cell
  ``flag_nonneuronal_high_depth``: a non-neuronal cell at >= 1,000 counts
  whose (level, class, bin) was emitted on its own ensemble verdict and is
  marked ``nonneuronal_high_depth`` (§8.3 v7.9); and
  ``nonneuronal_depth_trend``, the dataset-level glial large-mask /
  high-depth flag: real non-neuronal coverage falling above 1,000 counts
  (5K: class .916 -> .897 -> .886 over 500-999 / 1,000-1,999 / >= 2,000,
  REVIEW_CHECKS C3), the signature of merged masks reference cells cannot
  show. Report-only.
* ``factor_remeasure`` -- on the first in-house dataset of a family with a
  measured factor table (an R3 member), the per-gene factors re-measured
  against the reference pseudobulk of its confident calls
  (``shadow.reference_pseudobulk_totals``, the X1 code) and compared with
  the stored table on the informative genes: a warning when Pearson r < 0.9
  that recommends re-running PREP with the in-house table as a new
  simulation-input asset (inputs, never trust evidence; not automatic).
* ``gene_complexity_check`` -- median genes per cell of native cells vs
  simulated cells per depth bin; a warning when native cells carry > 45%
  more genes (beyond E2's 30-45%) which, from M3c, also says that simulated
  coverage predictions are unreliable for the dataset (v1-type cells carried
  2-73% more genes than simulated ones at matched depth).

``apply_qc_outcomes`` combines outcomes with a trust state and can only keep
or lower it (refused < broad_only < provisional < validated); every M3c
check is warning- or report-only and never lowers it.

M13 (chunk C12) adds the outcome model, the combinators and the checks the
§8.8 table still lacked, so that every check records an outcome per dataset
(pre-registration §23.5 P4: ``pass``, ``warn``, ``fail``, ``not_applicable``
or ``not_evaluable``):

* ``QcOutcome`` gains a ``state`` (``evaluated``, ``not_applicable``,
  ``not_evaluable``) and the lowering effects ``gate_cap`` (the dataset gate
  level), ``withhold_level`` (an emitted level made ``not_resolvable`` for
  the dataset) and ``withhold_pair_stats`` (the pair's supercluster-level
  cross-platform statistics);
* ``qc_effects`` combines outcomes; ``apply_qc_to_gate`` can only lower a
  gate verdict and ``apply_qc_to_statuses`` gives the per-level statuses a
  QC-applied RESOLVE must produce (confident sets only shrink, names of the
  cells that stay confident unchanged); ``qc_to_provenance`` maps outcomes to
  ``RealQcProvenance``;
* the checks: ``marker_consistency_outcome`` (human referee, D18: warning
  < 0.75, gate cap ``broad_only`` < 0.70), ``registration_g1_outcome``
  (human G1 as §7.6 defines it: its fail rule warn-only in M13 by D23 (b),
  its warning rule a warning), ``paired_concordance`` (soft
  broad JSD on the shared mask, D21; ``not_applicable`` for an unpaired
  section), ``flag_rate_summary`` (§8.8's literal reading, D22 / CHECK K6),
  ``prefilter_spotcheck`` and ``factor_remeasure_outcome``
  (``not_applicable`` where they do not apply);
* ``real_data_qc(signals, trust, config)`` runs them all from
  ``AnnotationRealQcConfig`` and demotes the lowering effects to warnings
  on the seeded ``real_data`` families until their species gate has merged
  (D20 (b)).

M13 chunk C16 adds the gene-complexity source (D19 (a)):
``native_gene_complexity`` counts native genes and totals on a bundle's query
genes, and ``gene_complexity_signal`` builds the check's inputs from the
simulated genes per cell a version-7 PREP stores
(``resolvability.load_simulated_genes``). Version-6 bundles (the seeded
families) and version-7 bundles built before the artefact store none, so the
check is ``not_evaluable`` for them.

RESOLVE wires them per dataset in M13 chunk C15.

This module imports only the standard library, numpy and pandas (and
``merxen.annotation.resolvability`` / ``schema`` / ``provenance``, which
need no more), so it imports in the GPU clustering environment.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeVar, cast

import numpy as np
import pandas as pd

from merxen.annotation.resolvability import STATUS_EMITTED, SimulatedGenes, depth_bin
from merxen.annotation.schema import GATE_LEVELS, PANEL_TRUST_STATES, CellStatus

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.diagnostics import TrustDecision
    from merxen.annotation.provenance import RealQcOutcome, RealQcProvenance

logger = logging.getLogger(__name__)

REAL_QC_VERSION: Final = 1
# Pre-registered constants (plan §8.8, user decision 4; [L]).
COVERAGE_WARN_MARGIN: Final = 0.10
COVERAGE_MIN_CELLS: Final = 200
# A called class needs this many dataset cells for its own per-class depth
# s_c(d); a thinner class takes the label-free total-count histogram of every
# cell (the RESOLVE follow-up of plan §12 M3c). Not pre-registered: the value
# was chosen for the follow-up (M13 chunk C14) and awaits the user's
# confirmation (pre-registration §22.9). It reuses the registered per-class
# minimum of a depth profile (plan §8.3 v7.5,
# ``sim_inputs.PROFILE_MIN_CLASS_CELLS``). It decides no coverage warning only
# while ``real_qc.coverage_min_cells`` (default ``COVERAGE_MIN_CELLS``, 200)
# is at least this: a lower setting lets the warning judge classes whose
# prediction uses the label-free histogram.
CLASS_DEPTH_MIN_CLASS_CELLS: Final = 100
SHARE_SOURCE_OWN: Final = "own"
SHARE_SOURCE_LABEL_FREE: Final = "label_free"
NONNEURONAL_HIGH_DEPTH_COUNTS: Final = 1000
FACTOR_REMEASURE_MIN_R: Final = 0.9
FACTOR_INFORMATIVE_MIN_EXPECTED: Final = 1000.0
FACTOR_CAP_LOG2: Final = 3.0
GENE_COMPLEXITY_GAP_WARN: Final = 0.45
GENE_COMPLEXITY_MIN_CELLS: Final = 50
# The depth bands of the non-neuronal trend (REVIEW_CHECKS C3).
DEPTH_BANDS: Final[tuple[tuple[int, int | None], ...]] = (
    (500, 1000),
    (1000, 2000),
    (2000, None),
)
TREND_MIN_BAND_CELLS: Final = 200
TREND_Z: Final = 2.0
FLOAT_TOL: Final = 1e-9

Effect = Literal[
    "warning",
    "report_only",
    "downgrade",
    "gate_cap",
    "withhold_level",
    "withhold_pair_stats",
]
EFFECTS: Final[tuple[str, ...]] = (
    "warning",
    "report_only",
    "downgrade",
    "gate_cap",
    "withhold_level",
    "withhold_pair_stats",
)
# Effects that lower what a dataset emits or reports (plan §8.8): a trust
# cap, a dataset gate-level cap, an emitted level made ``not_resolvable`` for
# the dataset, and the pair's supercluster-level cross-platform statistics
# withheld. Every other effect leaves the dataset's outputs unchanged.
LOWERING_EFFECTS: Final[frozenset[str]] = frozenset(
    {"downgrade", "gate_cap", "withhold_level", "withhold_pair_stats"}
)
QcState = Literal["evaluated", "not_applicable", "not_evaluable"]
QC_STATES: Final[tuple[str, ...]] = ("evaluated", "not_applicable", "not_evaluable")
# Gate levels a QC outcome may cap a dataset at (never ``full``).
QC_GATE_CAPS: Final[tuple[str, ...]] = ("broad_only", "failed")
# Provenance outcome tokens (``provenance.RealQcOutcome``), worst first: a
# check with several outcomes (one per level or class) records the worst, and
# a check that could not be evaluated in part is never recorded as a pass.
OUTCOME_ORDER: Final[tuple[str, ...]] = (
    "fail",
    "warn",
    "not_evaluable",
    "pass",
    "not_applicable",
)

# Worded per class (D4 of 2026-09-29): the warning fires for glia, vascular
# and immune cells and for small hypothalamic neuron classes alike.
COVERAGE_WARNING_TEXT: Final = (
    "{cls} at {level}: the real confident share {real:.3f} of the {n} cells "
    "called {cls} is below the class-depth prediction {predicted:.3f} - "
    "{margin:.2f} at the dataset's own per-class depth{profile}. Simulated "
    "coverage of {cls} at {level} is an upper bound for this dataset; its "
    "precision is unmeasured on real data. The warning also fires on v1-type "
    "large-mask (nucleus-expansion) segmentation, where simulation "
    "over-predicts coverage by +.16 to +.22 whatever the efficiency model. No "
    "offset is applied; emission, thresholds and trust are unchanged."
)
COVERAGE_PROFILE_TEXT: Final = (
    "; the profile-mode prediction {profile:.3f} is reported beside it "
    "(it does not decide the warning)"
)
NONNEURONAL_TREND_TEXT: Final = (
    "real {level} coverage of cells called {cls} falls above {limit:,} counts "
    "({low:.3f} at {low_band} vs {high:.3f} at >= {limit:,}; z {z:.1f}): the "
    "signature of merged or large masks that reference cells cannot show "
    "(5K vendor segmentation: glial class .916 -> .897 -> .886). Report-only; "
    "non-neuronal bins at >= {limit:,} counts are emitted only on their own "
    "ensemble verdict and their cells carry flag_nonneuronal_high_depth."
)
FACTOR_WARNING_TEXT: Final = (
    "re-measured per-gene factors disagree with the stored table "
    "{asset} (Pearson r {r:.3f} < {min_r:.2f} on {n} informative genes). "
    "Recommended, not automatic: re-run PREP with this dataset's table as a new "
    "simulation-input asset (an input, never trust evidence; plan §8.8, M13)."
)
GENE_COMPLEXITY_TEXT: Final = (
    "native cells carry {gap:.0%} more genes per cell than simulated cells at "
    "matched depth (bin {depth}; > {limit:.0%}, beyond E2's 30-45%). Simulated "
    "coverage predictions are unreliable for this dataset: real cells with more "
    "genes than simulated ones (v1-type large masks carried 2-73% more) are "
    "resolved differently from their simulation (5k_real/sim/REPORT.txt §7)."
)


@dataclass(frozen=True)
class QcOutcome:
    """One real-data QC outcome (plan §8.8).

    An outcome is ``evaluated`` (it passed or fired), ``not_applicable`` (the
    check does not apply to the dataset: an unpaired section has no paired
    concordance, a family without an R3 member no factor re-measure, a panel
    without a prefilter no spot check) or ``not_evaluable`` (it applies but
    its input is missing). Only an evaluated outcome can fire.

    Attributes:
        check: The check (``coverage_vs_simulation``, ...).
        fired: Whether the check fired.
        effect: What a fired outcome does: ``warning`` (the dataset gate's
            warning flag) or ``report_only`` (neither changes an output);
            ``downgrade`` (trust lowered to ``trust_cap``), ``gate_cap``
            (the dataset gate level lowered to ``gate_cap``),
            ``withhold_level`` (the emitted ``level`` becomes
            ``not_resolvable`` for the dataset) or ``withhold_pair_stats``
            (the pair's supercluster-level cross-platform statistics are
            withheld). None raises a trust state, a gate level or an
            emission, and none changes a margin, a threshold or a floor.
        message: Human-readable text (empty when not fired).
        trust_cap: For ``downgrade``, the highest trust state kept.
        level: The level concerned, if any (required for
            ``withhold_level``).
        cls: The class concerned, if any.
        details: Numbers behind the outcome.
        gate_cap: For ``gate_cap``, the highest gate level kept
            (``broad_only`` or ``failed``).
        state: ``evaluated``, ``not_applicable`` or ``not_evaluable``.
        reason: Why the check was not evaluated (required then).
    """

    check: str
    fired: bool
    effect: Effect
    message: str = ""
    trust_cap: str | None = None
    level: str | None = None
    cls: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    gate_cap: str | None = None
    state: QcState = "evaluated"
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the effect, its cap and the state."""
        if self.effect not in EFFECTS:
            raise ValueError(f"unknown QC effect {self.effect!r}")
        if self.effect == "downgrade":
            if self.trust_cap not in PANEL_TRUST_STATES:
                raise ValueError(
                    f"a downgrade needs a trust cap, got {self.trust_cap!r}"
                )
        elif self.trust_cap is not None:
            raise ValueError(f"a {self.effect} outcome has no trust cap")
        if self.effect == "gate_cap":
            if self.gate_cap not in QC_GATE_CAPS:
                raise ValueError(
                    f"a gate_cap outcome needs a gate cap in {QC_GATE_CAPS}, got "
                    f"{self.gate_cap!r}"
                )
        elif self.gate_cap is not None:
            raise ValueError(f"a {self.effect} outcome has no gate cap")
        if (
            self.effect == "withhold_level"
            and self.state == "evaluated"
            and not self.level
        ):
            raise ValueError("a withhold_level outcome needs its level")
        if self.state not in QC_STATES:
            raise ValueError(f"unknown QC state {self.state!r}")
        if self.state != "evaluated":
            if self.fired:
                raise ValueError(f"a {self.state} outcome cannot fire")
            if not self.reason:
                raise ValueError(f"a {self.state} outcome needs a reason")

    @classmethod
    def not_applicable(
        cls,
        check: str,
        reason: str,
        *,
        effect: Effect = "warning",
        **fields: Any,
    ) -> QcOutcome:
        """Return the outcome of a check that does not apply to the dataset.

        Args:
            check: The check.
            reason: Why it does not apply.
            effect: The effect it would have had.
            **fields: Further ``QcOutcome`` fields (a cap for capping
                effects, ``details``).

        Returns:
            The ``not_applicable`` outcome.
        """
        return cls(
            check, False, effect, state="not_applicable", reason=reason, **fields
        )

    @classmethod
    def not_evaluable(
        cls,
        check: str,
        reason: str,
        *,
        effect: Effect = "warning",
        **fields: Any,
    ) -> QcOutcome:
        """Return the outcome of a check that applies but cannot be evaluated.

        Args:
            check: The check.
            reason: Why it cannot be evaluated (the missing input).
            effect: The effect it would have had.
            **fields: Further ``QcOutcome`` fields.

        Returns:
            The ``not_evaluable`` outcome.
        """
        return cls(check, False, effect, state="not_evaluable", reason=reason, **fields)

    @property
    def lowers(self) -> bool:
        """Whether the outcome lowers the dataset (fired, lowering effect)."""
        return bool(self.fired) and self.effect in LOWERING_EFFECTS

    @property
    def outcome(self) -> str:
        """Return the provenance token (``provenance.RealQcOutcome``).

        ``not_applicable`` / ``not_evaluable`` for checks not evaluated;
        ``pass`` when not fired; ``fail`` when fired with a lowering effect;
        ``warn`` when fired with a warning or report-only effect.
        """
        if self.state != "evaluated":
            return self.state
        if not self.fired:
            return "pass"
        return "fail" if self.effect in LOWERING_EFFECTS else "warn"

    def warn_only(self, note: str) -> QcOutcome:
        """Return the outcome with a lowering effect demoted to a warning.

        The seeded ``real_data`` families only warn until their species gate
        has merged (plan §8.8, ``seeded_families_warn_only_until_gate``): the
        outcome keeps its message (prefixed with ``note``) and records the
        effect it would have had in ``details`` (``warn_only_effect`` and its
        cap), so the report can say what was withheld.

        Args:
            note: Why the effect is withheld.

        Returns:
            ``self`` unless it lowers; else the warning.
        """
        if not self.lowers:
            return self
        details = dict(self.details)
        details["warn_only_effect"] = self.effect
        if self.gate_cap is not None:
            details["warn_only_gate_cap"] = self.gate_cap
        if self.trust_cap is not None:
            details["warn_only_trust_cap"] = self.trust_cap
        return QcOutcome(
            check=self.check,
            fired=True,
            effect="warning",
            message=f"{note}: {self.message}" if self.message else note,
            level=self.level,
            cls=self.cls,
            details=details,
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe record."""
        return {
            "check": self.check,
            "fired": bool(self.fired),
            "effect": self.effect,
            "outcome": self.outcome,
            "state": self.state,
            "reason": self.reason,
            "message": self.message,
            "trust_cap": self.trust_cap,
            "gate_cap": self.gate_cap,
            "level": self.level,
            "class": self.cls,
            "details": _json_safe(dict(self.details)),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def trust_rank(state: str) -> int:
    """Return the rank of a trust state (refused 0 < ... < validated 3).

    Raises:
        ValueError: For an unknown state.
    """
    if state not in PANEL_TRUST_STATES:
        raise ValueError(f"unknown trust state {state!r}")
    return PANEL_TRUST_STATES.index(state)


def apply_qc_outcomes(state: str, outcomes: Iterable[QcOutcome]) -> str:
    """Return the trust state after real-data QC: never higher than ``state``.

    Only fired ``downgrade`` outcomes lower it (to their ``trust_cap``);
    every other effect (a gate cap, a withheld level or pair statistics, a
    warning, a report), and every passing, ``not_applicable`` or
    ``not_evaluable`` check, leaves it unchanged (plan §8.8: real data stay
    downgrade-only; the marker referee caps the gate, not trust, D18 (a)).

    Args:
        state: The trust state from PREP and the validated tables.
        outcomes: QC outcomes of the dataset.

    Returns:
        The trust state, at most ``state``.
    """
    rank = trust_rank(state)
    for outcome in outcomes:
        if outcome.fired and outcome.effect == "downgrade" and outcome.trust_cap:
            rank = min(rank, trust_rank(outcome.trust_cap))
    return PANEL_TRUST_STATES[rank]


def _bins(totals: np.ndarray, grid: Sequence[int]) -> np.ndarray:
    """Return ``resolvability.depth_bin`` with missing totals in no bin."""
    values = np.asarray(totals, dtype=np.float64)
    bins = depth_bin(values, grid)
    bins[~np.isfinite(values)] = np.nan
    return bins


# --------------------------------------------------------------------------
# Per-class real vs simulated coverage (user decision 4)


def class_bin_shares(
    totals: Sequence[float] | np.ndarray,
    called_class: Sequence[object] | np.ndarray,
    grid: Sequence[int],
) -> pd.DataFrame:
    """Return a dataset's own per-class depth-bin shares ``s_c(d)`` (§8.3 v7.5).

    Each cell is binned by its total counts with RESOLVE's floor-bin lookup
    (``resolvability.depth_bin``: the largest grid value <= its total); cells
    below the smallest grid value stay in their class's denominator and in no
    bin, as ``resolvability.class_depth_table`` counts profile cells.

    Args:
        totals: Total counts per cell.
        called_class: The cell's called class (the resolvability parent
            class: mouse WMB class, human floor class); missing values are
            left out.
        grid: The bundle's depth grid.

    Returns:
        ``class``, ``depth``, ``share``, ``n_cells`` (cells of the class) per
        (class, grid value).
    """
    values = np.asarray(totals, dtype=np.float64)
    classes = pd.Series(np.asarray(called_class, dtype=object))
    if len(values) != len(classes):
        raise ValueError("totals and called_class differ in length")
    bins = _bins(values, grid)
    frame = pd.DataFrame({"class": classes, "bin": bins})
    frame = frame[frame["class"].notna()]
    frame["class"] = frame["class"].astype(str)
    depths = sorted(int(value) for value in grid)
    rows: list[dict[str, Any]] = []
    for cls, group in frame.groupby("class", sort=True):
        n_cells = len(group)
        counts = group["bin"].value_counts()
        for depth in depths:
            rows.append(
                {
                    "class": str(cls),
                    "depth": depth,
                    "share": float(counts.get(float(depth), 0) / n_cells),
                    "n_cells": int(n_cells),
                }
            )
    return pd.DataFrame(rows, columns=["class", "depth", "share", "n_cells"])


def predicted_class_coverage(
    class_depth: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    regime: str = "provisional",
    levels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the class-depth prediction at a dataset's own bin shares.

    ``predicted_coverage(L, c) = sum_d s_c(d) cov(L, c, d)`` and
    ``resolvable_share(L, c) = sum_d s_c(d)`` over the bins where (L, c, d)
    is emitted in ``regime`` (``resolvability_class_depth.parquet``: a
    monotone-filled bin contributes its fill measurement, a pooled bin its
    pool's), the sums RESOLVE computes with the dataset's cells called c
    (plan §8.3 v7.5). A class without any class-depth row gets no row.

    Args:
        class_depth: ``resolvability_class_depth.parquet``.
        shares: ``class_bin_shares`` of the dataset.
        regime: The dataset's regime (``provisional`` or ``validated``).
        levels: Levels to predict (default: every level of the table).

    Returns:
        ``level``, ``class``, ``n_cells``, ``predicted_coverage``,
        ``resolvable_share``.
    """
    columns = ["level", "class", "n_cells", "predicted_coverage", "resolvable_share"]
    table = class_depth[class_depth["regime"].astype(str) == regime]
    if levels is not None:
        table = table[table["level"].astype(str).isin([str(item) for item in levels])]
    if table.empty or shares.empty:
        return pd.DataFrame(columns=columns)
    table = table.assign(
        _class=table["class"].astype(str), _depth=table["depth"].astype(int)
    )
    share = shares.assign(_class=shares["class"].astype(str), _depth=shares["depth"])
    merged = table.merge(
        share[["_class", "_depth", "share", "n_cells"]],
        on=["_class", "_depth"],
        how="inner",
    )
    emitted = (merged["status"].astype(str) == STATUS_EMITTED).to_numpy()
    coverage = np.nan_to_num(merged["coverage"].to_numpy(np.float64), nan=0.0)
    merged["_term"] = np.where(emitted, merged["share"] * coverage, 0.0)
    merged["_resolvable"] = np.where(emitted, merged["share"], 0.0)
    rows = []
    for (level, cls), group in merged.groupby(["level", "_class"], sort=True):
        rows.append(
            {
                "level": str(level),
                "class": str(cls),
                "n_cells": int(group["n_cells"].iloc[0]),
                "predicted_coverage": float(group["_term"].sum()),
                "resolvable_share": float(group["_resolvable"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def dataset_class_bin_shares(
    totals: Sequence[float] | np.ndarray,
    called_class: Sequence[object] | np.ndarray,
    grid: Sequence[int],
    *,
    min_class_cells: int = CLASS_DEPTH_MIN_CLASS_CELLS,
) -> pd.DataFrame:
    """Return a dataset's per-class depth ``s_c(d)``, label-free for thin classes.

    A called class with at least ``min_class_cells`` cells takes its own bin
    shares (``class_bin_shares``, ``share_source`` ``own``); a thinner class
    takes the bin shares of every cell with a total (a label-free total-count
    histogram, ``label_free``), so a handful of calls never sets a class's
    depth (the RESOLVE follow-up of plan §12 M3c). ``n_cells`` stays the
    class's own cell count either way.

    Args:
        totals: Total counts per cell.
        called_class: The cell's called class (missing values are left out
            of the classes, never of the label-free histogram).
        grid: The bundle's depth grid.
        min_class_cells: Cells a class needs for its own shares.

    Returns:
        ``class``, ``depth``, ``share``, ``n_cells``, ``share_source`` per
        (class, grid value).
    """
    if min_class_cells < 1:
        raise ValueError("min_class_cells must be >= 1")
    columns = ["class", "depth", "share", "n_cells", "share_source"]
    own = class_bin_shares(totals, called_class, grid)
    if own.empty:
        return pd.DataFrame(columns=columns)
    values = np.asarray(totals, dtype=np.float64)
    pooled = class_bin_shares(
        values, np.full(len(values), "all", dtype=object), grid
    ).set_index("depth")["share"]
    thin = own["n_cells"].to_numpy() < int(min_class_cells)
    shares = own["share"].to_numpy(np.float64).copy()
    shares[thin] = pooled.reindex(own["depth"][thin].to_numpy()).to_numpy(np.float64)
    return pd.DataFrame(
        {
            "class": own["class"].astype(str).to_numpy(),
            "depth": own["depth"].astype(int).to_numpy(),
            "share": shares,
            "n_cells": own["n_cells"].astype(int).to_numpy(),
            "share_source": np.where(thin, SHARE_SOURCE_LABEL_FREE, SHARE_SOURCE_OWN),
        },
        columns=columns,
    )


def dataset_class_depth_prediction(
    class_depth: pd.DataFrame,
    totals: Sequence[float] | np.ndarray,
    class_keys: Mapping[str, Sequence[object] | np.ndarray],
    regimes: Mapping[str, str],
    grid: Sequence[int],
    *,
    min_class_cells: int = CLASS_DEPTH_MIN_CLASS_CELLS,
) -> pd.DataFrame:
    """Return the class-depth prediction of each (level, called class) of a dataset.

    ``predicted_class_coverage`` per level, at the dataset's own per-class
    depth (``dataset_class_bin_shares`` of the cells whose class key at the
    level is c, label-free for thin classes) and in the dataset's regime at
    that level (plan §8.3 v7.5; the RESOLVE follow-up of §12 M3c). It is the
    predictor of ``coverage_vs_simulation``; computing it changes nothing.

    Args:
        class_depth: The class-depth table of the decisions applied
            (``resolvability.class_depth_table``).
        totals: Total counts per cell.
        class_keys: Per level, each cell's class key there (the class its
            emission was read for).
        regimes: Per level, the dataset's regime there.
        grid: The bundle's depth grid.
        min_class_cells: Cells a class needs for its own depth.

    Returns:
        ``level``, ``regime``, ``class``, ``n_cells``, ``share_source``,
        ``predicted_coverage``, ``resolvable_share`` per (level, class).
    """
    columns = [
        "level",
        "regime",
        "class",
        "n_cells",
        "share_source",
        "predicted_coverage",
        "resolvable_share",
    ]
    frames = []
    for level, keys in class_keys.items():
        regime = str(regimes[level])
        shares = dataset_class_bin_shares(
            totals, keys, grid, min_class_cells=min_class_cells
        )
        predicted = predicted_class_coverage(
            class_depth, shares, regime=regime, levels=[str(level)]
        )
        if predicted.empty:
            continue
        source = shares.drop_duplicates("class").set_index("class")["share_source"]
        frames.append(
            predicted.assign(
                regime=regime,
                share_source=predicted["class"].map(source).to_numpy(),
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def real_class_coverage(
    called_class: Sequence[object] | np.ndarray,
    confident: Mapping[str, Sequence[bool] | np.ndarray],
) -> pd.DataFrame:
    """Return the dataset's confident share per (level, called class).

    Args:
        called_class: The cell's called class (as ``class_bin_shares``).
        confident: Per level, whether each cell is confidently labelled there
            (after RESOLVE's thresholds, floors and emission).

    Returns:
        ``level``, ``class``, ``n_cells``, ``real_coverage``.
    """
    classes = pd.Series(np.asarray(called_class, dtype=object))
    rows = []
    for level, flags in confident.items():
        values = np.asarray(flags, dtype=bool)
        if len(values) != len(classes):
            raise ValueError(f"confident[{level!r}] differs in length")
        frame = pd.DataFrame({"class": classes, "confident": values})
        frame = frame[frame["class"].notna()]
        for cls, group in frame.groupby(frame["class"].astype(str), sort=True):
            rows.append(
                {
                    "level": str(level),
                    "class": str(cls),
                    "n_cells": len(group),
                    "real_coverage": float(group["confident"].mean()),
                }
            )
    return pd.DataFrame(rows, columns=["level", "class", "n_cells", "real_coverage"])


@dataclass(frozen=True)
class CoverageComparison:
    """``coverage_vs_simulation`` output (warning-only; plan §8.8).

    Attributes:
        table: Per (level, class): ``n_cells``, ``real_coverage``,
            ``predicted_coverage`` (the class-depth predictor, which decides
            the warning), ``resolvable_share``, ``difference`` (real -
            predicted), ``profile_coverage`` and ``profile_difference`` (the
            profile-mode prediction, reported only; ``nan`` without one),
            ``judged`` (>= ``min_cells``), ``warn``.
        outcomes: One fired ``warning`` per flagged (level, class).
        margin: The warning margin (0.10).
        min_cells: Dataset cells a (level, class) needs to be judged (200).
        profile_reported: Whether a profile-mode prediction was given.
    """

    table: pd.DataFrame
    outcomes: tuple[QcOutcome, ...]
    margin: float
    min_cells: int
    profile_reported: bool = False

    @property
    def flagged(self) -> list[tuple[str, str]]:
        """Return the flagged (level, class) pairs."""
        rows = self.table[self.table["warn"]]
        return [
            (str(level), str(cls))
            for level, cls in zip(rows["level"], rows["class"], strict=True)
        ]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary (``<pair>_resolve_summary.json``)."""
        return {
            "check": "coverage_vs_simulation",
            "version": REAL_QC_VERSION,
            "margin": self.margin,
            "min_cells": self.min_cells,
            "n_judged": int(self.table["judged"].sum()) if len(self.table) else 0,
            "n_flagged": len(self.flagged),
            "flagged": [{"level": level, "class": cls} for level, cls in self.flagged],
            "predictor": "class_depth",
            "profile_mode_reported": bool(self.profile_reported),
            "trust_effect": "none",
            "note": (
                "warning only: never an offset, never a change of emission, "
                "thresholds or trust; fires on v1-type large-mask segmentation too"
            ),
        }


def profile_coverage_table(
    predictions: pd.DataFrame,
    metrics: Mapping[str, str],
    *,
    member: str = "member_mean",
) -> pd.DataFrame:
    """Return profile mode's per-class coverage in ``coverage_vs_simulation`` form.

    Args:
        predictions: ``annotation-panel-simulate``'s ``profile_predictions.csv``
            (per member and ``member_mean``: ``class`` and ``cov_*`` columns).
        metrics: Level -> the coverage column of that level (mouse:
            ``{"class": "cov_class_prov", "subclass": "cov_subclass_prov"}``).
        member: The member whose rows are used (default: the member mean).

    Returns:
        ``level``, ``class``, ``profile_coverage`` (the ALL row left out).
    """
    columns = ["level", "class", "profile_coverage"]
    if predictions.empty or "class" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    rows = predictions
    if "member" in rows.columns:
        rows = rows[rows["member"].astype(str) == member]
    rows = rows[rows["class"].astype(str) != "ALL"]
    records = []
    for level, column in metrics.items():
        if column not in rows.columns:
            continue
        for cls, value in zip(rows["class"].astype(str), rows[column], strict=True):
            records.append(
                {
                    "level": str(level),
                    "class": cls,
                    "profile_coverage": float(value)
                    if value is not None and np.isfinite(float(value))
                    else math.nan,
                }
            )
    return pd.DataFrame(records, columns=columns)


def coverage_vs_simulation(
    real: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    profile_predicted: pd.DataFrame | None = None,
    margin: float = COVERAGE_WARN_MARGIN,
    min_cells: int = COVERAGE_MIN_CELLS,
) -> CoverageComparison:
    """Compare a dataset's real coverage with the simulation, per class (§8.8).

    A (level, called class) with at least ``min_cells`` dataset cells is
    flagged when its real confident share is below the class-depth
    prediction at the dataset's own per-class depth minus ``margin`` (user
    decision 4: real < simulated - 0.10; the pre-registered predictor). The
    profile-mode prediction, when given, is reported beside it and never
    decides a warning (D4 of 2026-09-29). The warning is worded per class.
    The check can only warn: it returns warnings and never a trust change,
    an offset or a change of emission.

    Args:
        real: ``real_class_coverage`` output.
        predicted: ``predicted_class_coverage`` output (the class-depth
            predictor).
        profile_predicted: Optional ``profile_coverage_table`` output.
        margin: Warning margin.
        min_cells: Cells a (level, class) needs to be judged.

    Returns:
        The comparison.
    """
    if margin < 0:
        raise ValueError("the margin must be >= 0")
    columns = [
        "level",
        "class",
        "n_cells",
        "real_coverage",
        "predicted_coverage",
        "resolvable_share",
        "difference",
        "profile_coverage",
        "profile_difference",
        "judged",
        "warn",
    ]
    reported = profile_predicted is not None and not profile_predicted.empty
    if real.empty or predicted.empty:
        return CoverageComparison(
            pd.DataFrame(columns=columns), (), margin, min_cells, reported
        )
    left = real.assign(
        level=real["level"].astype(str), **{"class": real["class"].astype(str)}
    )
    right = predicted.assign(
        level=predicted["level"].astype(str),
        **{"class": predicted["class"].astype(str)},
    )[["level", "class", "predicted_coverage", "resolvable_share"]]
    merged = left.merge(right, on=["level", "class"], how="inner")
    if profile_predicted is not None and not profile_predicted.empty:
        profile = profile_predicted.assign(
            level=profile_predicted["level"].astype(str),
            **{"class": profile_predicted["class"].astype(str)},
        )[["level", "class", "profile_coverage"]]
        merged = merged.merge(profile, on=["level", "class"], how="left")
    else:
        merged["profile_coverage"] = math.nan
    merged["profile_coverage"] = pd.to_numeric(
        merged["profile_coverage"], errors="coerce"
    ).astype(np.float64)
    merged["difference"] = merged["real_coverage"] - merged["predicted_coverage"]
    merged["profile_difference"] = merged["real_coverage"] - merged["profile_coverage"]
    merged["judged"] = merged["n_cells"].astype(int) >= int(min_cells)
    # A float tolerance, so that real = predicted - margin never warns.
    merged["warn"] = merged["judged"] & (
        merged["real_coverage"] < merged["predicted_coverage"] - margin - FLOAT_TOL
    )
    merged = merged.sort_values(["level", "class"]).reset_index(drop=True)
    outcomes = tuple(
        QcOutcome(
            check="coverage_vs_simulation",
            fired=True,
            effect="warning",
            message=COVERAGE_WARNING_TEXT.format(
                real=row["real_coverage"],
                n=int(row["n_cells"]),
                cls=row["class"],
                level=row["level"],
                predicted=row["predicted_coverage"],
                margin=margin,
                profile=COVERAGE_PROFILE_TEXT.format(profile=row["profile_coverage"])
                if math.isfinite(float(row["profile_coverage"]))
                else "",
            ),
            level=str(row["level"]),
            cls=str(row["class"]),
            details={
                "n_cells": int(row["n_cells"]),
                "real_coverage": float(row["real_coverage"]),
                "predicted_coverage": float(row["predicted_coverage"]),
                "difference": float(row["difference"]),
                "predictor": "class_depth",
                "profile_coverage": float(row["profile_coverage"]),
            },
        )
        for row in merged[merged["warn"]].to_dict("records")
    )
    return CoverageComparison(merged[columns], outcomes, margin, min_cells, reported)


# --------------------------------------------------------------------------
# Non-neuronal high depth (§8.3 v7.9)


def nonneuronal_high_depth_flags(
    totals: Sequence[float] | np.ndarray,
    called_class: Sequence[object] | np.ndarray,
    class_depth: pd.DataFrame,
    *,
    regime: str = "provisional",
    min_counts: int = NONNEURONAL_HIGH_DEPTH_COUNTS,
    levels: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the report-only per-cell ``flag_nonneuronal_high_depth`` (v7.9).

    A cell is flagged when its called class is non-neuronal (the class-depth
    table's ``neuronal`` is not true: unknown lineage counts as
    non-neuronal, as in the monotone fill), its total counts are at least
    ``min_counts`` and at least one level's (class, bin) is emitted in
    ``regime`` and marked ``nonneuronal_high_depth`` (emitted on its own
    ensemble verdict, never by the fill).

    Args:
        totals: Total counts per cell.
        called_class: The cell's called class.
        class_depth: ``resolvability_class_depth.parquet``.
        regime: The dataset's regime.
        min_counts: The high-depth limit (1,000).
        levels: Only these levels' bins (default: every level), for callers
            whose class key differs by level (RESOLVE).

    Returns:
        Boolean flags, one per cell.
    """
    values = np.asarray(totals, dtype=np.float64)
    classes = np.asarray(called_class, dtype=object)
    if len(values) != len(classes):
        raise ValueError("totals and called_class differ in length")
    neuronal = np.array(
        [_known_true(value) for value in class_depth["neuronal"]], dtype=bool
    )
    high_depth = np.array(
        [_known_true(value) for value in class_depth["nonneuronal_high_depth"]],
        dtype=bool,
    )
    table = class_depth[
        (class_depth["regime"].astype(str) == regime).to_numpy()
        & (class_depth["status"].astype(str) == STATUS_EMITTED).to_numpy()
        & high_depth
        & ~neuronal
    ]
    if levels is not None:
        table = table[table["level"].astype(str).isin([str(item) for item in levels])]
    if table.empty:
        return np.zeros(len(values), dtype=bool)
    grid = sorted({int(value) for value in class_depth["depth"].astype(int)})
    bins = _bins(values, grid)
    marked = {
        (str(cls), float(depth))
        for cls, depth in zip(table["class"].astype(str), table["depth"], strict=True)
    }
    flags = np.zeros(len(values), dtype=bool)
    for index, (cls, value, bin_value) in enumerate(
        zip(classes, values, bins, strict=True)
    ):
        if cls is None or (isinstance(cls, float) and math.isnan(cls)):
            continue
        if value < min_counts or math.isnan(bin_value):
            continue
        flags[index] = (str(cls), float(bin_value)) in marked
    return flags


def _known_true(value: object) -> bool:
    """Whether a (possibly nullable) boolean value is known to be true.

    ``class_depth_table`` stores ``neuronal`` as a nullable ``boolean`` column
    (unknown lineage is missing) and the parquet round trip may give objects
    or floats; a missing value is never true.
    """
    if value is None or value is pd.NA:
        return False
    try:
        return bool(value == 1)
    except (TypeError, ValueError):
        return False


def _band_label(low: int, high: int | None) -> str:
    return f"{low:,}-{high - 1:,}" if high is not None else f">= {low:,}"


def nonneuronal_depth_trend(
    totals: Sequence[float] | np.ndarray,
    called_class: Sequence[object] | np.ndarray,
    confident: Mapping[str, Sequence[bool] | np.ndarray],
    nonneuronal_classes: Iterable[str],
    *,
    limit: int = NONNEURONAL_HIGH_DEPTH_COUNTS,
    min_band_cells: int = TREND_MIN_BAND_CELLS,
    z_min: float = TREND_Z,
) -> tuple[pd.DataFrame, tuple[QcOutcome, ...]]:
    """Return the glial large-mask / high-depth trend (report-only; §8.8).

    Per non-neuronal class and level: the real confident share in the depth
    bands 500-999, 1,000-1,999 and >= 2,000 counts and pooled at >= ``limit``.
    A (level, class) is reported as falling when both the 500-999 band and
    the >= ``limit`` pool hold at least ``min_band_cells`` cells and the
    pool's share is below the band's by more than ``z_min`` standard errors
    of the difference (report-only; real vendor-segmented 5K glia fall
    .916 -> .897 -> .886 at class level, REVIEW_CHECKS C3).

    Args:
        totals: Total counts per cell.
        called_class: The cell's called class.
        confident: Per level, whether each cell is confidently labelled.
        nonneuronal_classes: The non-neuronal classes (vocab lineage).
        limit: The high-depth limit (1,000).
        min_band_cells: Cells each compared band needs.
        z_min: Standard errors the fall must exceed.

    Returns:
        ``(table, outcomes)``: per (level, class, band) ``n_cells`` and
        ``coverage`` (bands and the ``>= limit`` pool), and one fired
        report-only outcome per falling (level, class).
    """
    values = np.asarray(totals, dtype=np.float64)
    classes = pd.Series(np.asarray(called_class, dtype=object)).astype(str)
    wanted = {str(item) for item in nonneuronal_classes}
    rows: list[dict[str, Any]] = []
    outcomes: list[QcOutcome] = []
    bands = [*DEPTH_BANDS, (limit, None)]
    for level, flags in confident.items():
        confident_values = np.asarray(flags, dtype=bool)
        if len(confident_values) != len(values):
            raise ValueError(f"confident[{level!r}] differs in length")
        for cls in sorted(wanted):
            in_class = (classes == cls).to_numpy()
            if not in_class.any():
                continue
            measured: dict[str, tuple[int, float]] = {}
            for index, (low, high) in enumerate(bands):
                in_band = in_class & (values >= low)
                if high is not None:
                    in_band &= values < high
                n_cells = int(in_band.sum())
                share = float(confident_values[in_band].mean()) if n_cells else math.nan
                label = (
                    f">= {low:,} (pooled)"
                    if index == len(bands) - 1
                    else _band_label(low, high)
                )
                measured[label] = (n_cells, share)
                rows.append(
                    {
                        "level": str(level),
                        "class": cls,
                        "band": label,
                        "n_cells": n_cells,
                        "coverage": share,
                    }
                )
            low_label = _band_label(*DEPTH_BANDS[0])
            n_low, share_low = measured[low_label]
            n_high, share_high = measured[f">= {limit:,} (pooled)"]
            if n_low < min_band_cells or n_high < min_band_cells:
                continue
            pooled = (share_low * n_low + share_high * n_high) / (n_low + n_high)
            se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / n_low + 1 / n_high))
            z = (share_low - share_high) / se
            if z > z_min:
                outcomes.append(
                    QcOutcome(
                        check="nonneuronal_depth_trend",
                        fired=True,
                        effect="report_only",
                        message=NONNEURONAL_TREND_TEXT.format(
                            level=level,
                            cls=cls,
                            limit=limit,
                            low=share_low,
                            low_band=low_label,
                            high=share_high,
                            z=z,
                        ),
                        level=str(level),
                        cls=cls,
                        details={
                            "n_low": n_low,
                            "coverage_low": share_low,
                            "n_high": n_high,
                            "coverage_high": share_high,
                            "z": z,
                        },
                    )
                )
    table = pd.DataFrame(
        rows, columns=["level", "class", "band", "n_cells", "coverage"]
    )
    return table, tuple(outcomes)


# --------------------------------------------------------------------------
# Per-gene factor re-measure (first in-house dataset of an R3 family)


@dataclass(frozen=True)
class FactorRemeasure:
    """``factor_remeasure`` output (warning-only; plan §8.8, §8.10).

    Attributes:
        applies: Whether the check ran (first dataset of a family with a
            stored measured table).
        reason: Why it did not run, if it did not.
        asset_id: The stored table compared with.
        n_informative: Informative genes compared (both tables).
        pearson_r: Pearson r of the log2 factors on them.
        spearman_r: Spearman r (reported).
        outcome: The warning or the passing outcome.
        factors: Per gene: ``gene_id``, ``log2_factor`` (re-measured,
            centred on the median informative gene, capped +-3),
            ``expected_counts``, ``observed_counts``, ``stored_log2_factor``,
            ``informative``.
    """

    applies: bool
    reason: str | None
    asset_id: str | None
    n_informative: int
    pearson_r: float | None
    spearman_r: float | None
    outcome: QcOutcome | None
    factors: pd.DataFrame | None = None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""
        return {
            "check": "factor_remeasure",
            "version": REAL_QC_VERSION,
            "applies": self.applies,
            "reason": self.reason,
            "asset_id": self.asset_id,
            "n_informative": self.n_informative,
            "pearson_r": _json_safe(self.pearson_r),
            "spearman_r": _json_safe(self.spearman_r),
            "warn": bool(self.outcome is not None and self.outcome.fired),
            "message": "" if self.outcome is None else self.outcome.message,
            "trust_effect": "none",
            "recommendation": (
                "re-run PREP with this dataset's table as a new simulation-input "
                "asset (inputs, never trust evidence); not automatic"
                if self.outcome is not None and self.outcome.fired
                else None
            ),
        }


def factor_remeasure(
    counts: sparse.spmatrix | np.ndarray,
    gene_ids: Sequence[str],
    labels: Sequence[object] | np.ndarray,
    profiles: pd.DataFrame,
    stored: pd.DataFrame | None,
    *,
    include: np.ndarray | None = None,
    first_dataset_of_family: bool = True,
    asset_id: str | None = None,
    min_r: float = FACTOR_REMEASURE_MIN_R,
    informative_min_expected: float = FACTOR_INFORMATIVE_MIN_EXPECTED,
    cap_log2: float = FACTOR_CAP_LOG2,
    pseudocount: float = 1.0,
) -> FactorRemeasure:
    """Re-measure per-gene factors and compare them with the stored table (§8.8).

    Runs on the first in-house dataset of a family with a measured factor
    table (an R3 member). The factors are measured with the X1 code
    (``shadow.reference_pseudobulk_totals``: observed gene totals of the
    included, labelled cells against the reference pseudobulk of their
    calls), as the stored table was, then centred on the median informative
    gene and capped at +-3 log2 (the stored table's conventions). The
    comparison uses the genes informative in both tables (expected >= 1,000
    counts here, tier ``informative`` in the stored table). Pearson r <
    ``min_r`` gives a warning that recommends re-running PREP with this
    dataset's table as a new asset; nothing is re-run automatically and
    trust never changes.

    Args:
        counts: Cells x genes (raw counts).
        gene_ids: The genes of ``counts`` (Ensembl ids).
        labels: The called class per cell (``profiles`` index).
        profiles: ``shadow.profile_matrix`` of the reference at the called
            level over ``gene_ids``.
        stored: The stored ``member`` table (``sim_inputs.efficiency_table``:
            indexed by gene id, ``tier``, ``log2_factor``), or ``None``.
        include: Cells used (confident calls).
        first_dataset_of_family: Whether this is the family's first in-house
            dataset (the check runs only then).
        asset_id: The stored asset's id (reported).
        min_r: The warning threshold (0.9).
        informative_min_expected: Expected counts of an informative gene.
        cap_log2: Cap of the centred log2 factor (3).
        pseudocount: Added to observed and expected totals.

    Returns:
        The re-measure.
    """
    if stored is None:
        return FactorRemeasure(False, "no_stored_table", asset_id, 0, None, None, None)
    if not first_dataset_of_family:
        return FactorRemeasure(
            False, "not_first_dataset_of_family", asset_id, 0, None, None, None
        )
    from merxen.annotation.shadow import reference_pseudobulk_totals

    genes = [str(gene) for gene in gene_ids]
    observed, expected = reference_pseudobulk_totals(
        counts, labels, profiles, include=include
    )
    raw = np.log2((observed + pseudocount) / (expected + pseudocount))
    informative_here = expected >= informative_min_expected
    centre = float(np.median(raw[informative_here])) if informative_here.any() else 0.0
    factors = np.clip(raw - centre, -cap_log2, cap_log2)
    stored_factor = stored["log2_factor"].reindex(genes).to_numpy(np.float64)
    stored_tier = stored["tier"].reindex(genes).astype(object).to_numpy()
    informative = (
        informative_here & (stored_tier == "informative") & np.isfinite(stored_factor)
    )
    table = pd.DataFrame(
        {
            "gene_id": genes,
            "log2_factor": factors,
            "expected_counts": expected,
            "observed_counts": observed,
            "stored_log2_factor": stored_factor,
            "informative": informative,
        }
    )
    n_informative = int(informative.sum())
    if n_informative < 3:
        return FactorRemeasure(
            False,
            "too_few_informative_genes",
            asset_id,
            n_informative,
            None,
            None,
            None,
            table,
        )
    here = factors[informative]
    there = stored_factor[informative]
    pearson = float(np.corrcoef(here, there)[0, 1])
    spearman = float(
        np.corrcoef(
            pd.Series(here).rank().to_numpy(), pd.Series(there).rank().to_numpy()
        )[0, 1]
    )
    fired = not (pearson >= min_r)
    outcome = QcOutcome(
        check="factor_remeasure",
        fired=fired,
        effect="warning",
        message=FACTOR_WARNING_TEXT.format(
            asset=asset_id or "(stored table)", r=pearson, min_r=min_r, n=n_informative
        )
        if fired
        else "",
        details={
            "pearson_r": pearson,
            "spearman_r": spearman,
            "n_informative": n_informative,
            "min_r": min_r,
        },
    )
    return FactorRemeasure(
        True, None, asset_id, n_informative, pearson, spearman, outcome, table
    )


# --------------------------------------------------------------------------
# Gene complexity (native vs simulated genes per cell)


def gene_complexity_check(
    native_n_genes: Sequence[float] | np.ndarray,
    native_totals: Sequence[float] | np.ndarray,
    simulated_n_genes: Sequence[float] | np.ndarray,
    simulated_depth: Sequence[float] | np.ndarray,
    grid: Sequence[int],
    *,
    gap_warn: float = GENE_COMPLEXITY_GAP_WARN,
    min_cells: int = GENE_COMPLEXITY_MIN_CELLS,
) -> tuple[pd.DataFrame, QcOutcome]:
    """Compare genes per cell of native and simulated cells per depth bin (§8.8).

    Native cells are binned by their total counts (``depth_bin``), simulated
    cells by their depth (the grid value, or the realised total binned the
    same way). A bin with at least ``min_cells`` cells of each is judged:
    ``gap`` = native median / simulated median - 1. The check warns when a
    judged bin's gap exceeds ``gap_warn`` (45%, beyond E2's 30-45%); from
    M3c the warning also says that simulated coverage predictions are
    unreliable for the dataset. Every judged bin records whether native cells
    carry more genes than simulated ones at all (report-only).

    Args:
        native_n_genes: Genes detected per native cell.
        native_totals: Total counts per native cell.
        simulated_n_genes: Genes detected per simulated cell.
        simulated_depth: Depth (total counts) per simulated cell.
        grid: The depth grid.
        gap_warn: Warning gap (0.45).
        min_cells: Cells of each kind a judged bin needs.

    Returns:
        ``(per-bin table, outcome)``.
    """
    native_bins = _bins(np.asarray(native_totals, dtype=np.float64), grid)
    simulated_bins = _bins(np.asarray(simulated_depth, dtype=np.float64), grid)
    native = np.asarray(native_n_genes, dtype=np.float64)
    simulated = np.asarray(simulated_n_genes, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for depth in sorted(int(value) for value in grid):
        in_native = native_bins == float(depth)
        in_simulated = simulated_bins == float(depth)
        n_native = int(in_native.sum())
        n_simulated = int(in_simulated.sum())
        judged = n_native >= min_cells and n_simulated >= min_cells
        native_median = float(np.median(native[in_native])) if n_native else math.nan
        simulated_median = (
            float(np.median(simulated[in_simulated])) if n_simulated else math.nan
        )
        gap = (
            native_median / simulated_median - 1.0
            if judged and simulated_median > 0
            else math.nan
        )
        rows.append(
            {
                "depth": depth,
                "n_native": n_native,
                "n_simulated": n_simulated,
                "native_median_genes": native_median,
                "simulated_median_genes": simulated_median,
                "gap": gap,
                "judged": judged,
                "native_exceeds_simulated": bool(judged and gap > 0),
                "warn": bool(judged and gap > gap_warn),
            }
        )
    table = pd.DataFrame(rows)
    warned = table[table["warn"]] if len(table) else table
    if len(warned):
        worst = warned.loc[warned["gap"].idxmax()]
        outcome = QcOutcome(
            check="gene_complexity",
            fired=True,
            effect="warning",
            message=GENE_COMPLEXITY_TEXT.format(
                gap=float(worst["gap"]), depth=int(worst["depth"]), limit=gap_warn
            ),
            details={
                "bins_warned": [int(value) for value in warned["depth"]],
                "max_gap": float(worst["gap"]),
                "gap_warn": gap_warn,
                "coverage_predictions_reliable": False,
            },
        )
    else:
        exceeds = bool(len(table) and table["native_exceeds_simulated"].any())
        outcome = QcOutcome(
            check="gene_complexity",
            fired=False,
            effect="warning",
            details={
                "gap_warn": gap_warn,
                "native_exceeds_simulated_in_some_bin": exceeds,
            },
        )
    return table, outcome


@dataclass(frozen=True)
class NativeGeneComplexity:
    """Native genes and counts per cell on a bundle's query genes (M13 C16).

    Attributes:
        n_genes: Query genes with a count > 0, per cell.
        totals: Counts over the query genes, per cell.
        n_query_genes: The bundle's query genes.
        missing_genes: Query genes the dataset lacks (they count as not
            detected), in the bundle's order.
    """

    n_genes: np.ndarray
    totals: np.ndarray
    n_query_genes: int
    missing_genes: tuple[str, ...]


def native_gene_complexity(
    counts: sparse.spmatrix | np.ndarray,
    gene_ids: Sequence[str],
    query_genes: Sequence[str],
) -> NativeGeneComplexity:
    """Count native genes and counts per cell on a bundle's query genes.

    D19 (a): native ``n_genes`` is counted on the genes the simulated cells
    were counted on (the bundle's query genes, ``SimulatedGenes.query_genes``:
    the test cells' genes), never on the dataset's whole panel, so that a
    gene the reference lacks cannot make native cells look more complex.
    Totals are summed over the same genes, so native and simulated depth
    bins mean the same counts.

    Args:
        counts: Cells x genes raw counts (dense or scipy sparse); the
            caller passes the cells to compare (RESOLVE: the table cells).
        gene_ids: The genes of ``counts`` (unique ids).
        query_genes: The bundle's query genes.

    Returns:
        The per-cell genes and totals and the query genes the dataset lacks.

    Raises:
        ValueError: If ``gene_ids`` does not match the columns of ``counts``
            or holds duplicates.
    """
    genes = [str(gene) for gene in gene_ids]
    n_columns = int(counts.shape[1])
    if len(genes) != n_columns:
        raise ValueError(
            f"{len(genes)} gene ids for {n_columns} columns of the native counts"
        )
    if len(set(genes)) != len(genes):
        raise ValueError("the native gene ids hold duplicates")
    position = {gene: index for index, gene in enumerate(genes)}
    query = [str(gene) for gene in query_genes]
    columns = [position[gene] for gene in query if gene in position]
    missing = tuple(gene for gene in query if gene not in position)
    if hasattr(counts, "tocsr"):
        # scipy sparse (not imported here: this module imports in the GPU
        # clustering environment, which has no scipy).
        selected = counts.tocsr()[:, columns]
        n_genes = np.asarray((selected > 0).sum(axis=1)).ravel()
        totals = np.asarray(selected.sum(axis=1), dtype=np.float64).ravel()
    else:
        dense = np.asarray(counts, dtype=np.float64)[:, columns]
        n_genes = (dense > 0).sum(axis=1)
        totals = dense.sum(axis=1)
    return NativeGeneComplexity(
        n_genes=np.asarray(n_genes, dtype=np.int64),
        totals=np.asarray(totals, dtype=np.float64),
        n_query_genes=len(query),
        missing_genes=missing,
    )


def qc_summary(outcomes: Iterable[QcOutcome]) -> dict[str, Any]:
    """Return the JSON-safe record of a dataset's real-data QC outcomes."""
    items = [outcome.to_json() for outcome in outcomes]
    return {
        "version": REAL_QC_VERSION,
        "outcomes": items,
        "n_fired": sum(1 for item in items if item["fired"]),
        "promotes": False,
    }


# --------------------------------------------------------------------------
# Effects and the downgrade-only combinators (M13)

# The levels a ``broad_only`` gate leaves unattempted (``leaf_gated`` in
# ``consensus.resolve_human`` / ``resolve_mouse``).
LEAF_GATED_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("supercluster", "seaad_subclass", "cluster"),
    "mouse": ("subclass", "supertype"),
}
# RESOLVE's levels, parents first.
LEVEL_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "human": ("lineage", "broad", "nt", "supercluster", "seaad_subclass", "cluster"),
    "mouse": ("broad", "class", "nt", "subclass", "supertype"),
}
# Per level, its parents in RESOLVE: a confident call needs a confident
# parent, the first listed one whose status at the cell is not
# ``not_applicable`` (a neuron's leaf hangs off its NT, a non-neuron's off its
# broad or class level).
LEVEL_PARENTS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "human": {
        "broad": ("lineage",),
        "nt": ("broad",),
        "supercluster": ("nt", "broad"),
        "seaad_subclass": ("broad",),
        "cluster": ("supercluster",),
    },
    "mouse": {
        "class": ("broad",),
        "nt": ("class",),
        "subclass": ("nt", "class"),
        "supertype": ("subclass",),
    },
}


class _GateVerdictLike(Protocol):
    """The fields ``apply_qc_to_gate`` reads (``GateVerdict``, ``MouseGateVerdict``)."""

    @property
    def level(self) -> str: ...

    @property
    def warning(self) -> bool: ...

    @property
    def level_reasons(self) -> tuple[str, ...]: ...

    @property
    def warning_reasons(self) -> tuple[str, ...]: ...


GateT = TypeVar("GateT", bound=_GateVerdictLike)


def gate_severity(level: str) -> int:
    """Return the severity of a gate level (full 0 < broad_only 1 < failed 2).

    Args:
        level: ``full``, ``broad_only`` or ``failed``.

    Returns:
        Its index in ``schema.GATE_LEVELS``.

    Raises:
        ValueError: For an unknown level.
    """
    if level not in GATE_LEVELS:
        raise ValueError(f"unknown gate level {level!r}")
    return GATE_LEVELS.index(level)


@dataclass(frozen=True)
class QcEffects:
    """The combined effects of a dataset's QC outcomes (downgrade-only).

    Attributes:
        gate_cap: The most severe gate level a fired ``gate_cap`` outcome
            caps the dataset at (``None``: no cap).
        trust_cap: The lowest trust state a fired ``downgrade`` keeps.
        withheld_levels: Emitted levels made ``not_resolvable`` for the
            dataset.
        withhold_pair_stats: Whether the pair's supercluster-level
            cross-platform statistics are withheld.
        level_reasons: Gate level reasons of the gate caps.
        warning_reasons: Gate warning reasons: fired warnings and every
            lowering effect other than a gate cap.
        downgrades: ``<effect>:<check>`` tokens of the lowering effects
            (``RealQcProvenance.downgrades``).
    """

    gate_cap: str | None = None
    trust_cap: str | None = None
    withheld_levels: tuple[str, ...] = ()
    withhold_pair_stats: bool = False
    level_reasons: tuple[str, ...] = ()
    warning_reasons: tuple[str, ...] = ()
    downgrades: tuple[str, ...] = ()

    @property
    def lowers(self) -> bool:
        """Whether any effect lowers the dataset."""
        return bool(
            self.gate_cap
            or self.trust_cap
            or self.withheld_levels
            or self.withhold_pair_stats
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe record."""
        return {
            "gate_cap": self.gate_cap,
            "trust_cap": self.trust_cap,
            "withheld_levels": list(self.withheld_levels),
            "withhold_pair_stats": self.withhold_pair_stats,
            "level_reasons": list(self.level_reasons),
            "warning_reasons": list(self.warning_reasons),
            "downgrades": list(self.downgrades),
        }


def _scope(outcome: QcOutcome) -> str:
    parts = [str(part) for part in (outcome.level, outcome.cls) if part]
    return f"[{', '.join(parts)}]" if parts else ""


def qc_effects(outcomes: Iterable[QcOutcome]) -> QcEffects:
    """Combine a dataset's QC outcomes into their effects.

    Only fired outcomes act: a ``warning`` adds a gate warning reason, a
    ``gate_cap`` the lowest cap and a level reason, a ``downgrade`` the
    lowest trust cap, ``withhold_level`` its level and ``withhold_pair_stats``
    the pair flag (each of the last three also a warning reason). A
    ``report_only`` outcome and every passing, ``not_applicable`` or
    ``not_evaluable`` outcome have no effect.

    Args:
        outcomes: The dataset's outcomes.

    Returns:
        The effects.
    """
    gate_cap: str | None = None
    trust_cap: str | None = None
    withheld: list[str] = []
    pair_stats = False
    level_reasons: list[str] = []
    warning_reasons: list[str] = []
    downgrades: list[str] = []
    for outcome in outcomes:
        if not outcome.fired or outcome.effect == "report_only":
            continue
        reason = f"real_qc_{outcome.check}{_scope(outcome)}: " + (
            outcome.message or outcome.effect
        )
        if outcome.effect == "warning":
            warning_reasons.append(reason)
        elif outcome.effect == "gate_cap":
            cap = str(outcome.gate_cap)
            if gate_cap is None or gate_severity(cap) > gate_severity(gate_cap):
                gate_cap = cap
            level_reasons.append(reason)
            downgrades.append(f"{cap}:{outcome.check}")
        elif outcome.effect == "downgrade":
            cap = str(outcome.trust_cap)
            if trust_cap is None or trust_rank(cap) < trust_rank(trust_cap):
                trust_cap = cap
            warning_reasons.append(reason)
            downgrades.append(f"trust_{cap}:{outcome.check}")
        elif outcome.effect == "withhold_level":
            level = str(outcome.level)
            if level not in withheld:
                withheld.append(level)
            warning_reasons.append(reason)
            downgrades.append(f"not_resolvable_{level}:{outcome.check}")
        elif outcome.effect == "withhold_pair_stats":
            pair_stats = True
            warning_reasons.append(reason)
            downgrades.append(f"withhold_pair_stats:{outcome.check}")
    return QcEffects(
        gate_cap=gate_cap,
        trust_cap=trust_cap,
        withheld_levels=tuple(withheld),
        withhold_pair_stats=pair_stats,
        level_reasons=tuple(level_reasons),
        warning_reasons=tuple(warning_reasons),
        downgrades=tuple(dict.fromkeys(downgrades)),
    )


def _as_effects(qc: Iterable[QcOutcome] | QcEffects) -> QcEffects:
    return qc if isinstance(qc, QcEffects) else qc_effects(qc)


def apply_qc_to_gate(verdict: GateT, qc: Iterable[QcOutcome] | QcEffects) -> GateT:
    """Return a dataset gate verdict after real-data QC: never a higher level.

    The level becomes the worse of the verdict's and the QC gate cap; the QC
    level and warning reasons are appended and the warning flag is set when
    QC adds a warning reason (it is never cleared). Works for the human
    ``GateVerdict`` and the ``MouseGateVerdict`` (plan §5.4, §7.6, §8.8).

    Args:
        verdict: The dataset gate verdict before QC.
        qc: The dataset's QC outcomes or their effects.

    Returns:
        The verdict after QC (``verdict`` itself when QC adds nothing).
    """
    effects = _as_effects(qc)
    if not effects.level_reasons and not effects.warning_reasons:
        return verdict
    level = str(verdict.level)
    if effects.gate_cap is not None and gate_severity(effects.gate_cap) > gate_severity(
        level
    ):
        level = effects.gate_cap
    replaced = dataclasses.replace(
        cast(Any, verdict),
        level=level,
        warning=bool(verdict.warning or effects.warning_reasons),
        level_reasons=(*verdict.level_reasons, *effects.level_reasons),
        warning_reasons=(*verdict.warning_reasons, *effects.warning_reasons),
    )
    return cast(GateT, replaced)


def apply_qc_to_statuses(
    statuses: Mapping[str, np.ndarray | Sequence[object]],
    names: Mapping[str, np.ndarray | Sequence[object]],
    qc: Iterable[QcOutcome] | QcEffects,
    *,
    in_table: np.ndarray | Sequence[bool],
    species: str,
    gate_level: str = "full",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return per-level statuses and names after a dataset's QC effects.

    The expected statuses of RESOLVE with the QC effects applied, from the
    QC-free run's (``gate_level``: that run's gate level):

    * a gate cap worse than ``gate_level``: ``failed`` makes every table cell
      ``not_attempted_gate`` at every level, ``broad_only`` at the leaf
      levels (``LEAF_GATED_LEVELS``), without names, as RESOLVE writes them;
    * a withheld level: its confident cells, and those of the levels whose
      emission reads its table (``thresholds.DERIVED_EMISSION_LEVELS``: human
      ``seaad_subclass`` reads ``supercluster``), become ``not_resolvable``;
      the confident cells of the levels that hang off them
      (``LEVEL_PARENTS``) ``parent_unresolved``, names kept.

    Statuses only fall from ``confident``: no cell becomes confident and no
    cell that stays confident changes its name, and nothing here reads or
    changes an emission plan, a floor plan, a threshold or a margin. Exact
    for the gate caps; for a withheld level exact on the confident sets,
    while a non-confident status that RESOLVE orders after
    ``not_resolvable`` (a threshold miss) may differ from an in-pass
    application. The gate level is the caller's: withholding the lineage or
    broad level also lowers the confident broad coverage the dataset gate
    reads, which RESOLVE re-evaluates (with no confident broad call the gate
    fails); pass that gate's cap with the outcomes.

    Args:
        statuses: Per level, the QC-free run's ``CellStatus`` values.
        names: Per level, the QC-free run's names (same levels).
        qc: The dataset's QC outcomes or their effects.
        in_table: Table cells.
        species: ``"human"`` or ``"mouse"``.
        gate_level: The QC-free run's gate level.

    Returns:
        ``(statuses, names)`` after QC (copies).

    Raises:
        ValueError: For an unknown species, a level without names or arrays
            of different lengths.
    """
    if species not in LEVEL_PARENTS:
        raise ValueError(f"unknown species {species!r}")
    effects = _as_effects(qc)
    table = np.asarray(in_table, dtype=bool)
    n = len(table)
    out_status: dict[str, np.ndarray] = {}
    out_names: dict[str, np.ndarray] = {}
    for level, values in statuses.items():
        if level not in names:
            raise ValueError(f"no names for level {level!r}")
        out_status[level] = np.asarray(values, dtype=object).copy()
        out_names[level] = np.asarray(names[level], dtype=object).copy()
        if len(out_status[level]) != n or len(out_names[level]) != n:
            raise ValueError(f"level {level!r} differs in length from in_table")
    confident = CellStatus.CONFIDENT.value
    applicable = {
        level: values != CellStatus.NOT_APPLICABLE.value
        for level, values in out_status.items()
    }
    before = {level: values == confident for level, values in out_status.items()}
    cap = effects.gate_cap
    if cap is not None and gate_severity(cap) > gate_severity(gate_level):
        blocked = (
            list(out_status)
            if cap == "failed"
            else [level for level in out_status if level in LEAF_GATED_LEVELS[species]]
        )
        for level in blocked:
            out_status[level][table] = CellStatus.NOT_ATTEMPTED_GATE.value
            out_names[level][table] = None
    from merxen.annotation.thresholds import DERIVED_EMISSION_LEVELS

    withheld = set(effects.withheld_levels)
    withheld |= {
        derived
        for derived, source in DERIVED_EMISSION_LEVELS[species].items()
        if source in withheld
    }
    lost: dict[str, np.ndarray] = {}
    order = [level for level in LEVEL_ORDER[species] if level in out_status]
    order += [level for level in out_status if level not in order]
    for level in order:
        now = out_status[level] == confident
        if level in withheld:
            out_status[level][now] = CellStatus.NOT_RESOLVABLE.value
        else:
            parent_lost = np.zeros(n, dtype=bool)
            chosen = np.zeros(n, dtype=bool)
            for parent in LEVEL_PARENTS[species].get(level, ()):
                if parent not in lost:
                    continue
                use = ~chosen & applicable[parent]
                parent_lost |= use & lost[parent]
                chosen |= use
            out_status[level][now & parent_lost] = CellStatus.PARENT_UNRESOLVED.value
        lost[level] = before[level] & (out_status[level] != confident)
    return out_status, out_names


def worst_outcome(tokens: Iterable[str]) -> str:
    """Return the worst provenance token (``OUTCOME_ORDER``).

    Args:
        tokens: ``pass``, ``warn``, ``fail``, ``not_applicable`` or
            ``not_evaluable`` values.

    Returns:
        The worst of them.

    Raises:
        ValueError: For no token or an unknown one.
    """
    values = set(tokens)
    unknown = values - set(OUTCOME_ORDER)
    if unknown:
        raise ValueError(f"unknown QC outcome(s) {sorted(unknown)}")
    for token in OUTCOME_ORDER:
        if token in values:
            return token
    raise ValueError("no QC outcome given")


_SIGNAL_OUTCOME: Final[dict[str, str]] = {
    "pass": "pass",
    "warn": "warn",
    "fail": "fail",
    "not_evaluated": "not_evaluable",
}


def gate_outcomes(verdict: Any) -> dict[str, str]:
    """Return the provenance outcomes a dataset gate verdict records itself.

    The dataset gate applies its own effects (§5.4, §7.6); QC only records
    them: ``dataset_gate`` is ``fail`` below ``full``, ``warn`` with the
    warning flag, else ``pass``. A mouse verdict also gives
    ``registration_g1`` (G1) and ``marker_consistency`` (G2, the mouse
    marker referee of §8.8) from its signal statuses.

    Args:
        verdict: ``GateVerdict`` or ``MouseGateVerdict`` before QC.

    Returns:
        Check -> outcome token.
    """
    level = str(verdict.level)
    record = {
        "dataset_gate": "fail"
        if level != "full"
        else ("warn" if verdict.warning else "pass")
    }
    signal_status = getattr(verdict, "signal_status", None)
    if isinstance(signal_status, Mapping):
        for signal, check in (("g1", "registration_g1"), ("g2", "marker_consistency")):
            value = signal_status.get(signal)
            if value is not None:
                record[check] = _SIGNAL_OUTCOME.get(str(value), "not_evaluable")
    return record


def qc_to_provenance(
    outcomes: Iterable[QcOutcome],
    *,
    warn_only: bool | None = None,
    gate: Any = None,
) -> RealQcProvenance:
    """Return ``PanelProvenance.real_data_qc`` for a dataset's outcomes.

    Each check records its worst outcome (``worst_outcome``: a check with one
    outcome per level or class fails if any fails), the dataset gate's own
    outcomes are added from ``gate`` (``gate_outcomes``), and ``downgrades``
    lists the lowering effects (``qc_effects``).

    Args:
        outcomes: The dataset's outcomes (after any warn-only demotion).
        warn_only: Whether the checks only warned (seeded family).
        gate: The dataset gate verdict before QC, if any.

    Returns:
        The provenance record.

    Raises:
        ValueError: If a check name is not a safe provenance key.
    """
    from merxen.annotation.provenance import RealQcProvenance, is_safe_key

    items = list(outcomes)
    tokens: dict[str, list[str]] = {}
    for outcome in items:
        tokens.setdefault(outcome.check, []).append(outcome.outcome)
    if gate is not None:
        for check, token in gate_outcomes(gate).items():
            tokens.setdefault(check, []).append(token)
    for check in tokens:
        if not is_safe_key(check):
            raise ValueError(f"QC check {check!r} is not a safe provenance key")
    return RealQcProvenance(
        outcomes=cast(
            "dict[str, RealQcOutcome]",
            {check: worst_outcome(values) for check, values in tokens.items()},
        ),
        downgrades=list(qc_effects(items).downgrades),
        warn_only=warn_only,
    )


# --------------------------------------------------------------------------
# The checks M13 adds (plan §8.8)

MARKER_CONSISTENCY_CHECK: Final = "marker_consistency"
REGISTRATION_G1_CHECK: Final = "registration_g1"
PAIRED_CONCORDANCE_CHECK: Final = "paired_concordance"
FLAG_RATES_CHECK: Final = "flag_rates"
GENE_COMPLEXITY_CHECK: Final = "gene_complexity"
PREFILTER_SPOTCHECK_CHECK: Final = "prefilter_spotcheck"
COVERAGE_CHECK: Final = "coverage_vs_simulation"
NONNEURONAL_TREND_CHECK: Final = "nonneuronal_depth_trend"
FACTOR_REMEASURE_CHECK: Final = "factor_remeasure"
# The dataset gate's own outcome (``gate_outcomes``).
DATASET_GATE_CHECK: Final = "dataset_gate"
# §8.8: "the checks this module adds (marker referee, paired concordance,
# flag rates, gene complexity, prefilter spot check) only warn" on the seeded
# real-data families until their species gate has merged.
SEEDED_WARN_ONLY_CHECKS: Final[frozenset[str]] = frozenset(
    {
        MARKER_CONSISTENCY_CHECK,
        PAIRED_CONCORDANCE_CHECK,
        FLAG_RATES_CHECK,
        GENE_COMPLEXITY_CHECK,
        PREFILTER_SPOTCHECK_CHECK,
    }
)
# The flags whose rates §8.8 reads: "realised contamination, diffuse and
# spill-over rates (§5.6)" (the OOD flag's rates are reported only).
FLAG_RATE_FLAGS: Final[tuple[str, ...]] = (
    "contaminated",
    "diffuse_profile",
    "microglial_spillover",
)
# The pair statistic of §8.5 / §8.8 and D21: the intersection run's soft
# broad JSD, scored on the shared-tissue mask (point estimate), the whole
# section reported.
PAIRED_KIND: Final = "soft"
PAIRED_SCORED_REGION: Final = "shared_mask"
PAIRED_REPORTED_REGION: Final = "whole_section"

MARKER_CONSISTENCY_TEXT: Final = (
    "marker consistency {value:.3f} of confident broad calls over {n} "
    "marker-pseudo-labelled cells ({groups} marker groups) is below {limit:.2f}"
)
REGISTRATION_G1_TEXT: Final = (
    "registration G1 fails ({failed}; M0a guard with the §7.6 fail rule)"
)
REGISTRATION_G1_WARN_TEXT: Final = (
    "registration G1 warns (density ratio {ratio:.3f} < {limit}; M0a guard with "
    "the §7.6 warning rule)"
)
PAIRED_TEXT: Final = (
    "the pair's soft broad JSD {value:.3f} on the {region} is above {limit:.2f} "
    "({other_region} {other}): supercluster-level cross-platform statistics "
    "are withheld for this pair"
)
FLAG_RATES_TEXT: Final = (
    "{k} of {n} (class x platform) strata are uninformative under H16's 15% "
    "marking ({flags}), more than {frac:.0%}: most of the dataset's flag "
    "rates cannot be read (the flags stay report-only)"
)
PREFILTER_TEXT: Final = (
    "{level}: prefiltered vs unfiltered lookup agreement {value:.3f} < "
    "{limit:.2f} on {n} cells: the level is not_resolvable for this dataset"
)


@dataclass(frozen=True)
class MarkerConsistencySignal:
    """The human marker referee of one dataset (§8.8; computed by M13 C13).

    ``human_referee.human_marker_referee`` computes it (``.signal()``): the
    G2 port on per-panel marker sets derived from the WHB profiles (D18 (a)).

    Attributes:
        consistency: The share of marker-pseudo-labelled cells whose
            confident ``ct_broad`` agrees with their pseudo-label (``None``
            when not evaluable).
        n_marker_groups: Marker groups with at least 3 markers.
        n_pseudo_labelled: Marker-pseudo-labelled cells scored.
        reason: Why the statistic is not evaluable, if it is not.
        details: Further numbers (reported).
    """

    consistency: float | None
    n_marker_groups: int = 0
    n_pseudo_labelled: int = 0
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def marker_consistency_outcome(
    signal: MarkerConsistencySignal | None, *, warn: float, broad_only: float
) -> QcOutcome:
    """Return the human marker-referee outcome (§8.8; decision D18).

    A warning below ``warn`` (0.75); below ``broad_only`` (0.70) a dataset
    gate-level cap at ``broad_only`` (D18 (a)): trust and margins unchanged.
    ``not_evaluable`` without a statistic (fewer than 2 marker groups of >= 3
    markers, or too few pseudo-labelled cells; the referee says why).

    Args:
        signal: The referee, or ``None`` when it did not run.
        warn: The warning threshold.
        broad_only: The gate-cap threshold.

    Returns:
        The outcome.

    Raises:
        ValueError: If ``broad_only`` exceeds ``warn``.
    """
    if broad_only > warn:
        raise ValueError("the broad_only threshold must not exceed the warning")
    cap: dict[str, Any] = {"effect": "gate_cap", "gate_cap": "broad_only"}
    if signal is None:
        return QcOutcome.not_evaluable(
            MARKER_CONSISTENCY_CHECK, "the marker referee did not run", **cap
        )
    value = signal.consistency
    if value is None or not math.isfinite(float(value)):
        return QcOutcome.not_evaluable(
            MARKER_CONSISTENCY_CHECK,
            signal.reason or "no marker-consistency statistic",
            details=dict(signal.details),
            **cap,
        )
    number = float(value)
    details = {
        **dict(signal.details),
        "consistency": number,
        "n_marker_groups": int(signal.n_marker_groups),
        "n_pseudo_labelled": int(signal.n_pseudo_labelled),
        "warn": warn,
        "broad_only": broad_only,
    }
    text = MARKER_CONSISTENCY_TEXT.format(
        value=number,
        n=int(signal.n_pseudo_labelled),
        groups=int(signal.n_marker_groups),
        limit=broad_only if number < broad_only else warn,
    )
    if number < broad_only:
        return QcOutcome(
            MARKER_CONSISTENCY_CHECK,
            True,
            "gate_cap",
            message=text + ": the dataset gate is capped at broad_only (trust "
            "and margins unchanged)",
            gate_cap="broad_only",
            details=details,
        )
    if number < warn:
        return QcOutcome(
            MARKER_CONSISTENCY_CHECK, True, "warning", message=text, details=details
        )
    return QcOutcome(
        MARKER_CONSISTENCY_CHECK,
        False,
        "gate_cap",
        gate_cap="broad_only",
        details=details,
    )


def registration_g1_outcome(
    signal: Any,
    *,
    density_ratio_fail: float,
    density_ratio_warn: float,
    shift_fail_um: float,
    effect: str = "warning",
) -> QcOutcome:
    """Return the human registration G1 outcome (§8.8; NR9; decision D23).

    G1 is §7.6's, as defined there (NR9): it fails when M0a's density ratio
    is below ``density_ratio_fail`` (1.5) or the shift against the
    platform's own segmentation exceeds ``shift_fail_um`` (5 µm), and
    otherwise warns when the ratio is below ``density_ratio_warn`` (2.0).
    ``effect`` decides only what the fail rule does: ``warning`` keeps it
    warn-only (D23 (b), M13); ``gate_failed`` applies §8.8's effect (the
    dataset gate fails: every cell ``not_attempted_gate`` and
    ``exclude_hard``). The warning rule is a warning under either effect.
    ``not_evaluable`` without a check or a ratio.

    Args:
        signal: ``mouse_gate.RegistrationSignal`` (``density_ratio``,
            ``shift_um``, ``status``), or ``None``.
        density_ratio_fail: The fail ratio
            (``mouse_gate.g1_density_ratio_fail``).
        density_ratio_warn: The warning ratio
            (``mouse_gate.g1_density_ratio_warn``).
        shift_fail_um: The fail shift (``mouse_gate.g1_shift_fail_um``).
        effect: ``warning`` or ``gate_failed``.

    Returns:
        The outcome.

    Raises:
        ValueError: For an unknown effect, or a warning ratio below the fail
            ratio.
    """
    if effect not in ("warning", "gate_failed"):
        raise ValueError(f"unknown registration G1 effect {effect!r}")
    if density_ratio_warn < density_ratio_fail:
        raise ValueError("the G1 warning ratio must be at least the fail ratio")
    fields: dict[str, Any] = (
        {"effect": "gate_cap", "gate_cap": "failed"}
        if effect == "gate_failed"
        else {"effect": "warning"}
    )
    if signal is None:
        return QcOutcome.not_evaluable(
            REGISTRATION_G1_CHECK, "no registration check given", **fields
        )
    ratio = getattr(signal, "density_ratio", None)
    shift = getattr(signal, "shift_um", None)
    if ratio is None or not math.isfinite(float(ratio)):
        status = getattr(signal, "status", None)
        return QcOutcome.not_evaluable(
            REGISTRATION_G1_CHECK,
            f"registration check {status or 'without a density ratio'}",
            **fields,
        )
    number = float(ratio)
    failed: list[str] = []
    if number < density_ratio_fail:
        failed.append(f"density ratio {number:.3f} < {density_ratio_fail}")
    if shift is not None and float(shift) > shift_fail_um:
        failed.append(f"shift {float(shift):.1f} um > {shift_fail_um} um")
    warned = not failed and number < density_ratio_warn
    details = {
        "density_ratio": number,
        "shift_um": None if shift is None else float(shift),
        "density_ratio_fail": density_ratio_fail,
        "density_ratio_warn": density_ratio_warn,
        "shift_fail_um": shift_fail_um,
        "effect_setting": effect,
        "rule": "fail" if failed else ("warn" if warned else None),
        "source": getattr(signal, "source", None),
    }
    if warned:
        return QcOutcome(
            REGISTRATION_G1_CHECK,
            True,
            "warning",
            message=REGISTRATION_G1_WARN_TEXT.format(
                ratio=number, limit=density_ratio_warn
            ),
            details=details,
        )
    if not failed:
        return QcOutcome(REGISTRATION_G1_CHECK, False, details=details, **fields)
    message = REGISTRATION_G1_TEXT.format(failed="; ".join(failed))
    message += (
        ": the dataset gate fails"
        if effect == "gate_failed"
        else ": warn-only in M13 (decision D23 (b))"
    )
    return QcOutcome(
        REGISTRATION_G1_CHECK, True, message=message, details=details, **fields
    )


def _jsd_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def paired_concordance(
    jsd_rows: Sequence[Mapping[str, Any]] | None,
    *,
    paired: bool,
    warn_above: float,
    kind: str = PAIRED_KIND,
    scored_region: str = PAIRED_SCORED_REGION,
    reported_region: str = PAIRED_REPORTED_REGION,
) -> QcOutcome:
    """Return the paired-platform concordance outcome (§8.5, §8.8; D21).

    The pair's intersection-run soft broad JSD (``<pair>_resolve_summary.json``
    ``pair.jsd`` rows of ``kind``): its point estimate on ``scored_region``
    (the shared-tissue mask, D21) above ``warn_above`` (0.20) withholds the
    pair's supercluster-level cross-platform statistics; the other region is
    reported beside it. Only the registered statistic decides: a pair
    without a ``scored_region`` row is ``not_evaluable`` (the other region's
    value is reported in ``details``, never scored). An unpaired section (a
    single-platform dataset) is ``not_applicable``.

    Args:
        jsd_rows: The pair's JSD rows (``kind``, ``region``, ``jsd``,
            ``ci_low``, ``ci_high``).
        paired: Whether the dataset has a section of the other platform.
        warn_above: The warning threshold.
        kind: The composition kind scored.
        scored_region: The region scored.
        reported_region: The region reported beside it.

    Returns:
        The outcome (the same for both sections of the pair).
    """
    effect: dict[str, Any] = {"effect": "withhold_pair_stats"}
    if not paired:
        return QcOutcome.not_applicable(
            PAIRED_CONCORDANCE_CHECK,
            "unpaired: no section of the other platform",
            **effect,
        )
    rows = {
        str(row.get("region")): row
        for row in (jsd_rows or ())
        if str(row.get("kind")) == kind and _jsd_number(row.get("jsd")) is not None
    }
    other = rows.get(reported_region)
    other_value = None if other is None else _jsd_number(other.get("jsd"))
    if scored_region not in rows:
        reported = (
            ""
            if other_value is None
            else f"; {reported_region} {other_value:.3f} reported, not scored"
        )
        return QcOutcome.not_evaluable(
            PAIRED_CONCORDANCE_CHECK,
            f"no {kind} broad JSD of the pair on the {scored_region} (the "
            f"registered statistic, D21){reported}",
            details={
                "kind": kind,
                "scored_region": scored_region,
                f"{reported_region}_jsd": other_value,
                "warn_above": warn_above,
            },
            **effect,
        )
    row = rows[scored_region]
    value = float(row["jsd"])
    details = {
        "kind": kind,
        "statistic": "point",
        "scored_region": scored_region,
        "jsd": value,
        "ci_low": _jsd_number(row.get("ci_low")),
        "ci_high": _jsd_number(row.get("ci_high")),
        f"{reported_region}_jsd": other_value,
        "warn_above": warn_above,
    }
    if not value > warn_above:
        return QcOutcome(PAIRED_CONCORDANCE_CHECK, False, details=details, **effect)
    return QcOutcome(
        PAIRED_CONCORDANCE_CHECK,
        True,
        message=PAIRED_TEXT.format(
            value=value,
            region=scored_region,
            limit=warn_above,
            other_region=reported_region,
            other="not measured" if other_value is None else f"{other_value:.3f}",
        ),
        details=details,
        **effect,
    )


def _stratum_record(stratum: Any) -> dict[str, Any]:
    """Return a flag stratum as ``FlagStratum.to_json`` keys."""
    if isinstance(stratum, Mapping):
        return {
            "flag": str(stratum.get("flag")),
            "class": str(stratum.get("class")),
            "platform": str(stratum.get("platform")),
            "rate": _jsd_number(stratum.get("rate")),
            "informative": bool(stratum.get("informative")),
            "informative_h16": bool(stratum.get("informative_h16")),
        }
    return {
        "flag": str(stratum.flag),
        "class": str(stratum.cls),
        "platform": str(stratum.platform),
        "rate": None if stratum.rate is None else float(stratum.rate),
        "informative": bool(stratum.informative),
        "informative_h16": bool(stratum.informative_h16),
    }


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


@dataclass(frozen=True)
class FlagRateSummary:
    """``flag_rate_summary`` output (§8.8 flag rates; H16; decision D22).

    Attributes:
        table: Per (class, platform) stratum: ``class``, ``platform``,
            ``flags`` (the §8.8 flags with a stratum there),
            ``uninformative_flags`` (those above H16's 15% marking or
            without basis cells), ``uninformative``.
        outcome: The warning outcome.
        n_strata: (class, platform) strata counted.
        n_uninformative: Of those, uninformative.
        warn_frac: The warning share (0.5).
        reported: The other readings, reported only: per flag, pooled over
            flag x class x platform, and D22 option (a) (each flag's own null
            switch over every flag x class x platform stratum).
    """

    table: pd.DataFrame
    outcome: QcOutcome
    n_strata: int
    n_uninformative: int
    warn_frac: float
    reported: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary."""
        return {
            "check": FLAG_RATES_CHECK,
            "version": REAL_QC_VERSION,
            "reading": "h16_class_platform",
            "n_strata": self.n_strata,
            "n_uninformative": self.n_uninformative,
            "share_uninformative": _share(self.n_uninformative, self.n_strata),
            "warn_frac": self.warn_frac,
            "warn": bool(self.outcome.fired),
            "uninformative_strata": [
                {"class": row["class"], "platform": row["platform"]}
                for row in self.table.to_dict("records")
                if row["uninformative"]
            ],
            "reported": _json_safe(dict(self.reported)),
            "trust_effect": "none",
        }


def flag_rate_summary(
    strata: Iterable[Any],
    *,
    warn_frac: float,
    flags: Sequence[str] = FLAG_RATE_FLAGS,
) -> FlagRateSummary:
    """Return the flag-rate check of one dataset (§8.8; H16; D22 literal).

    The literal reading of §8.8 (CHECK K6, adopted 2026-10-06): the
    dataset's (class x platform) strata of the §8.8 flags (``flags``:
    contamination, diffuse and spill-over; ``FlagSet.strata``), each
    uninformative under H16's 15% marking when any of its flags' rate is
    above 15% or has no confident basis cells (``informative_h16`` false);
    a warning when more than ``warn_frac`` of them are uninformative. The
    readings counted per flag and pooled over flag x class x platform, and
    D22 option (a) (each flag's own null switch, every flag x class x
    platform stratum pooled), are reported beside it; none decides the
    warning.

    Args:
        strata: ``FlagStratum`` objects or their ``to_json`` records (the
            resolve summary's ``flags.strata``).
        warn_frac: The warning share (``uninformative_strata_warn_frac``).
        flags: The flags counted.

    Returns:
        The summary with its outcome.
    """
    records = [_stratum_record(item) for item in strata]
    counted = [record for record in records if record["flag"] in set(flags)]
    columns = ["class", "platform", "flags", "uninformative_flags", "uninformative"]
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in counted:
        groups.setdefault((record["class"], record["platform"]), []).append(record)
    for (cls, platform), members in sorted(groups.items()):
        uninformative = sorted(
            record["flag"] for record in members if not record["informative_h16"]
        )
        rows.append(
            {
                "class": cls,
                "platform": platform,
                "flags": sorted(record["flag"] for record in members),
                "uninformative_flags": uninformative,
                "uninformative": bool(uninformative),
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    per_flag: dict[str, dict[str, Any]] = {}
    for flag in sorted({record["flag"] for record in counted}):
        members = [record for record in counted if record["flag"] == flag]
        k = sum(1 for record in members if not record["informative_h16"])
        per_flag[flag] = {
            "n_strata": len(members),
            "n_uninformative": k,
            "share": _share(k, len(members)),
        }
    k_pooled = sum(1 for record in counted if not record["informative_h16"])
    k_own = sum(1 for record in records if not record["informative"])
    reported = {
        "per_flag_h16": per_flag,
        "pooled_h16": {
            "n_strata": len(counted),
            "n_uninformative": k_pooled,
            "share": _share(k_pooled, len(counted)),
        },
        "own_switch_all_flags": {
            "flags": sorted({record["flag"] for record in records}),
            "n_strata": len(records),
            "n_uninformative": k_own,
            "share": _share(k_own, len(records)),
            "would_warn": bool(records) and k_own / len(records) > warn_frac,
        },
    }
    n = len(rows)
    k = int(table["uninformative"].sum()) if n else 0
    details = {"n_strata": n, "n_uninformative": k, "warn_frac": warn_frac}
    if not n:
        outcome = QcOutcome.not_evaluable(
            FLAG_RATES_CHECK,
            f"no flag strata of {', '.join(flags)}",
            details=details,
        )
    elif k / n > warn_frac:
        outcome = QcOutcome(
            FLAG_RATES_CHECK,
            True,
            "warning",
            message=FLAG_RATES_TEXT.format(
                k=k, n=n, flags=", ".join(sorted(per_flag)), frac=warn_frac
            ),
            details=details,
        )
    else:
        outcome = QcOutcome(FLAG_RATES_CHECK, False, "warning", details=details)
    return FlagRateSummary(table, outcome, n, k, warn_frac, reported)


@dataclass(frozen=True)
class PrefilterSpotcheckSignal:
    """The 5K prefilter spot check of one dataset (§8.7, §8.8).

    Attributes:
        agreement: Per emitted level, the agreement of the prefiltered with
            the unfiltered lookup on the subsample.
        n_cells: Cells of the subsample (10k).
    """

    agreement: Mapping[str, float]
    n_cells: int = 0


def prefilter_spotcheck(
    signal: PrefilterSpotcheckSignal | None,
    *,
    applies: bool,
    emitted_levels: Sequence[str],
    min_agreement: float,
) -> tuple[QcOutcome, ...]:
    """Return the prefilter spot-check outcomes (§8.8), one per emitted level.

    An emitted level whose agreement is below ``min_agreement`` (0.95)
    becomes ``not_resolvable`` for the dataset (``withhold_level``). A
    bundle without a marker prefilter (panels of <= 1,000 genes, or the
    prefilter off) is ``not_applicable``; a prefiltered bundle without a spot
    check, or a level without an agreement, ``not_evaluable``.

    Args:
        signal: The spot check, or ``None``.
        applies: Whether the bundle's marker lookup is prefiltered.
        emitted_levels: The levels the dataset emits.
        min_agreement: The agreement threshold.

    Returns:
        The outcomes.
    """
    effect: dict[str, Any] = {"effect": "withhold_level"}
    if not applies:
        return (
            QcOutcome.not_applicable(
                PREFILTER_SPOTCHECK_CHECK,
                "no marker prefilter in the bundle (panels of <= 1,000 genes, "
                "or the prefilter off)",
                **effect,
            ),
        )
    if signal is None:
        return (
            QcOutcome.not_evaluable(
                PREFILTER_SPOTCHECK_CHECK,
                "a prefiltered bundle without a spot check",
                **effect,
            ),
        )
    if not emitted_levels:
        return (
            QcOutcome.not_evaluable(
                PREFILTER_SPOTCHECK_CHECK, "no emitted level", **effect
            ),
        )
    outcomes: list[QcOutcome] = []
    for level in emitted_levels:
        value = signal.agreement.get(level)
        if value is None or not math.isfinite(float(value)):
            outcomes.append(
                QcOutcome.not_evaluable(
                    PREFILTER_SPOTCHECK_CHECK,
                    f"no spot-check agreement at {level}",
                    level=level,
                    **effect,
                )
            )
            continue
        number = float(value)
        fired = number < min_agreement
        outcomes.append(
            QcOutcome(
                PREFILTER_SPOTCHECK_CHECK,
                fired,
                message=PREFILTER_TEXT.format(
                    level=level, value=number, limit=min_agreement, n=signal.n_cells
                )
                if fired
                else "",
                level=level,
                details={
                    "agreement": number,
                    "min_agreement": min_agreement,
                    "n_cells": int(signal.n_cells),
                },
                **effect,
            )
        )
    return tuple(outcomes)


def factor_remeasure_outcome(
    result: FactorRemeasure | None, *, min_r: float, has_r3_member: bool
) -> QcOutcome:
    """Return the factor re-measure outcome (§8.8) at the configured ``min_r``.

    Applicability comes from the family, never from a missing input: a
    family without an R3 member (no measured factor table) is
    ``not_applicable``, and so is a dataset other than the family's first. A
    family with an R3 member whose re-measure did not run, or ran without the
    stored table, is ``not_evaluable``, as is one with too few informative
    genes. Otherwise a warning when Pearson r < ``min_r`` (recommending a
    PREP re-run with the in-house table; never automatic, never trust
    evidence).

    Args:
        result: ``factor_remeasure`` output, or ``None`` when it did not run.
        min_r: The warning threshold (``factor_remeasure_min_r``).
        has_r3_member: Whether the family's bundle has an R3 member (a
            measured factor table).

    Returns:
        The outcome.

    Raises:
        ValueError: For a re-measure against a stored table on a family
            without an R3 member.
    """
    if not has_r3_member:
        if result is not None and result.applies:
            raise ValueError(
                "a factor re-measure against a stored table for a family without "
                "an R3 member"
            )
        return QcOutcome.not_applicable(
            FACTOR_REMEASURE_CHECK,
            "no R3 member: the family has no measured factor table",
        )
    if result is None:
        return QcOutcome.not_evaluable(
            FACTOR_REMEASURE_CHECK,
            "the family has an R3 member but the factor re-measure did not run",
        )
    if not result.applies:
        reason = str(result.reason)
        if reason == "not_first_dataset_of_family":
            return QcOutcome.not_applicable(FACTOR_REMEASURE_CHECK, reason)
        if reason == "no_stored_table":
            reason = (
                "no_stored_table: the family has an R3 member but its stored "
                "factor table was not given"
            )
        return QcOutcome.not_evaluable(FACTOR_REMEASURE_CHECK, reason)
    pearson = result.pearson_r
    fired = pearson is None or not pearson >= min_r
    details = {
        "pearson_r": pearson,
        "spearman_r": result.spearman_r,
        "n_informative": result.n_informative,
        "min_r": min_r,
        "asset_id": result.asset_id,
    }
    return QcOutcome(
        FACTOR_REMEASURE_CHECK,
        fired,
        "warning",
        message=FACTOR_WARNING_TEXT.format(
            asset=result.asset_id or "(stored table)",
            r=math.nan if pearson is None else pearson,
            min_r=min_r,
            n=result.n_informative,
        )
        if fired
        else "",
        details=details,
    )


# --------------------------------------------------------------------------
# The orchestrator (plan §8.8; M13 chunk C12)


@dataclass(frozen=True)
class GeneComplexitySignal:
    """``gene_complexity_check`` inputs (simulated cells: M13 C16, D19).

    ``gene_complexity_signal`` builds it from a version-7 bundle's stored
    simulated genes per cell and the dataset's counts.

    Attributes:
        native_n_genes: Genes per native table cell (the bundle's query
            genes).
        native_totals: Total counts per native table cell.
        simulated_n_genes: Genes per simulated cell stored by PREP.
        simulated_depth: Depth per simulated cell.
        grid: The bundle's depth grid.
        source: Where the inputs came from (recorded in the outcome's
            ``details``).
    """

    native_n_genes: Sequence[float] | np.ndarray
    native_totals: Sequence[float] | np.ndarray
    simulated_n_genes: Sequence[float] | np.ndarray
    simulated_depth: Sequence[float] | np.ndarray
    grid: Sequence[int]
    source: Mapping[str, Any] = field(default_factory=dict)


# How many missing query genes the gene-complexity source lists by name.
GENE_COMPLEXITY_MISSING_LISTED: Final = 20


def gene_complexity_signal(
    simulated: SimulatedGenes | None,
    native_counts: sparse.spmatrix | np.ndarray,
    native_gene_ids: Sequence[str],
) -> GeneComplexitySignal | None:
    """Build the gene-complexity inputs from a bundle and a dataset (D19 (a)).

    Simulated cells: one value per (test cell, grid depth), the mean
    ``n_genes`` over the bundle's emission members
    (``SimulatedGenes.per_cell``), at the grid depth D (a simulated cell's
    bin, v7.2; its realised total can fall just below D). Native cells: the
    given cells' genes and totals on the bundle's query genes
    (``native_gene_complexity``).

    Args:
        simulated: ``resolvability.load_simulated_genes`` of the primary
            bundle (``None``: a version-6 bundle, i.e. a seeded family, or a
            version-7 bundle built before the artefact).
        native_counts: Cells x genes raw counts of the cells to compare
            (RESOLVE: the table cells).
        native_gene_ids: The genes of ``native_counts``.

    Returns:
        The signal, or ``None`` without a stored source (the check is then
        ``not_evaluable``).
    """
    if simulated is None:
        return None
    native = native_gene_complexity(
        native_counts, native_gene_ids, simulated.query_genes
    )
    per_cell = simulated.per_cell()
    return GeneComplexitySignal(
        native_n_genes=native.n_genes,
        native_totals=native.totals,
        simulated_n_genes=per_cell["n_genes"].to_numpy(np.float64),
        simulated_depth=per_cell["depth"].to_numpy(np.float64),
        grid=list(simulated.depth_grid),
        source={
            "artefact_version": simulated.version,
            "members": list(simulated.emission_members),
            "simulated_cells": (
                "one per (test cell, grid depth): the mean n_genes over the "
                "emission members, binned at the grid depth"
            ),
            "native_cells": "genes and total counts on the bundle's query genes",
            "n_query_genes": native.n_query_genes,
            "n_query_genes_missing": len(native.missing_genes),
            "missing_genes": list(
                native.missing_genes[:GENE_COMPLEXITY_MISSING_LISTED]
            ),
        },
    )


@dataclass(frozen=True)
class CoverageSignal:
    """``coverage_vs_simulation`` inputs (version-7 families).

    Attributes:
        real: ``real_class_coverage`` of the dataset.
        predicted: The class-depth prediction
            (``dataset_class_depth_prediction``).
        profile: Profile mode's prediction, reported only.
    """

    real: pd.DataFrame
    predicted: pd.DataFrame
    profile: pd.DataFrame | None = None


@dataclass(frozen=True)
class NonneuronalTrendSignal:
    """``nonneuronal_depth_trend`` inputs (version-7 families).

    Attributes:
        totals: Total counts per table cell.
        called_class: The called class per table cell.
        confident: Per level, whether each table cell is confident.
        nonneuronal_classes: The non-neuronal classes.
    """

    totals: Sequence[float] | np.ndarray
    called_class: Sequence[object] | np.ndarray
    confident: Mapping[str, Sequence[bool] | np.ndarray]
    nonneuronal_classes: Sequence[str]


@dataclass(frozen=True, kw_only=True)
class RealQcSignals:
    """The label-free inputs of one dataset's real-data QC (plan §8.8).

    Whether a check applies is decided only from the bundle and dataset
    facts, which have no default and must be stated: ``resolvability_version``
    (the version-7 checks), ``paired`` (paired concordance),
    ``prefilter_applied`` (the prefilter spot check) and ``has_r3_member``
    (the factor re-measure). A check that applies but whose input is missing
    is ``not_evaluable``, never ``not_applicable``: an input left out is
    never recorded as a check that does not apply (pre-registration §23.5
    P4).

    Attributes:
        resolvability_version: The primary bundle's resolvability version
            (``None``: no resolvability tables).
        paired: Whether the dataset has a section of the other platform.
        prefilter_applied: Whether the bundle's marker lookup is
            prefiltered.
        has_r3_member: Whether the family's bundle has an R3 member (a
            measured factor table, version 7).
        pair_jsd: The pair's JSD rows (``<pair>_resolve_summary.json``
            ``pair.jsd``).
        marker_consistency: The human marker referee (C13).
        flag_strata: ``FlagSet.strata`` (or their JSON records).
        gene_complexity: Native and simulated genes per cell (C16).
        prefilter: The 5K prefilter spot check.
        emitted_levels: The levels the dataset emits (spot check).
        factor: ``factor_remeasure`` output (families with an R3 member).
        coverage: Real and predicted per-class coverage (version 7).
        nonneuronal_trend: The non-neuronal depth-trend inputs (version 7).
        registration: Human G1 (``mouse_gate.RegistrationSignal``); mouse
            G1 is the mouse gate's.
        gate: The dataset gate verdict before QC (its own outcomes are
            recorded; QC never re-applies them). Without it the gate's
            outcomes are ``not_evaluable``.
    """

    resolvability_version: int | None
    paired: bool
    prefilter_applied: bool
    has_r3_member: bool
    pair_jsd: Sequence[Mapping[str, Any]] = ()
    marker_consistency: MarkerConsistencySignal | None = None
    flag_strata: Sequence[Any] | None = None
    gene_complexity: GeneComplexitySignal | None = None
    prefilter: PrefilterSpotcheckSignal | None = None
    emitted_levels: Sequence[str] = ()
    factor: FactorRemeasure | None = None
    coverage: CoverageSignal | None = None
    nonneuronal_trend: NonneuronalTrendSignal | None = None
    registration: Any = None
    gate: Any = None


@dataclass(frozen=True)
class RealQcResult:
    """The real-data QC of one dataset (``real_data_qc``).

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        outcomes: Every outcome, after any warn-only demotion.
        seeded: Whether the family is a seeded ``real_data`` family.
        warn_only: Whether the checks of ``SEEDED_WARN_ONLY_CHECKS`` only
            warn (a seeded family whose species gate is pending).
        gate: The dataset gate verdict before QC, if given.
        flag_rates: The flag-rate summary, if strata were given.
        tables: Per-check tables (coverage, trend, gene complexity).
    """

    species: str
    outcomes: tuple[QcOutcome, ...]
    seeded: bool
    warn_only: bool
    gate: Any = None
    flag_rates: FlagRateSummary | None = None
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def effects(self) -> QcEffects:
        """Return the combined effects."""
        return qc_effects(self.outcomes)

    def apply_to_gate(self, verdict: GateT) -> GateT:
        """Return ``verdict`` after QC (``apply_qc_to_gate``)."""
        return apply_qc_to_gate(verdict, self.effects)

    def apply_to_trust(self, state: str) -> str:
        """Return the trust state after QC (``apply_qc_outcomes``)."""
        return apply_qc_outcomes(state, self.outcomes)

    def provenance(self) -> RealQcProvenance:
        """Return ``PanelProvenance.real_data_qc``."""
        return qc_to_provenance(self.outcomes, warn_only=self.warn_only, gate=self.gate)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe record (``<pair>_resolve_summary.json``)."""
        provenance = self.provenance()
        return {
            "version": REAL_QC_VERSION,
            "species": self.species,
            "seeded": self.seeded,
            "warn_only": self.warn_only,
            "per_check": dict(provenance.outcomes),
            "downgrades": list(provenance.downgrades),
            "effects": self.effects.to_json(),
            "outcomes": [outcome.to_json() for outcome in self.outcomes],
            "flag_rates": None
            if self.flag_rates is None
            else self.flag_rates.summary(),
            "promotes": False,
        }


def _is_seeded(trust: TrustDecision | None) -> bool:
    return (
        trust is not None
        and trust.state == "validated"
        and trust.validation_basis == "real_data"
    )


def _coverage_outcomes(
    signal: CoverageSignal | None,
    *,
    version: int | None,
    margin: float,
    min_cells: int,
) -> tuple[tuple[QcOutcome, ...], pd.DataFrame | None]:
    if version != 7:
        return (
            QcOutcome.not_applicable(
                COVERAGE_CHECK, "version-7 families only (no class-depth table)"
            ),
        ), None
    if signal is None:
        return (
            QcOutcome.not_evaluable(COVERAGE_CHECK, "no class-depth prediction"),
        ), None
    comparison = coverage_vs_simulation(
        signal.real,
        signal.predicted,
        profile_predicted=signal.profile,
        margin=margin,
        min_cells=min_cells,
    )
    if comparison.outcomes:
        return comparison.outcomes, comparison.table
    n_judged = int(comparison.table["judged"].sum()) if len(comparison.table) else 0
    details = {"n_judged": n_judged, "margin": margin, "min_cells": min_cells}
    if not n_judged:
        return (
            QcOutcome.not_evaluable(
                COVERAGE_CHECK,
                f"no (level, class) with >= {min_cells} cells and a prediction",
                details=details,
            ),
        ), comparison.table
    return (
        QcOutcome(COVERAGE_CHECK, False, "warning", details=details),
    ), comparison.table


def _trend_outcomes(
    signal: NonneuronalTrendSignal | None, *, version: int | None, limit: int
) -> tuple[tuple[QcOutcome, ...], pd.DataFrame | None]:
    effect: dict[str, Any] = {"effect": "report_only"}
    if version != 7:
        return (
            QcOutcome.not_applicable(
                NONNEURONAL_TREND_CHECK, "version-7 families only", **effect
            ),
        ), None
    if signal is None:
        return (
            QcOutcome.not_evaluable(
                NONNEURONAL_TREND_CHECK, "no depth-trend input", **effect
            ),
        ), None
    table, fired = nonneuronal_depth_trend(
        signal.totals,
        signal.called_class,
        signal.confident,
        signal.nonneuronal_classes,
        limit=limit,
    )
    if fired:
        return fired, table
    low = _band_label(*DEPTH_BANDS[0])
    high = f">= {limit:,} (pooled)"
    compared = 0
    if len(table):
        for _, group in table.groupby(["level", "class"]):
            counts = group.set_index("band")["n_cells"]
            if (
                int(counts.get(low, 0)) >= TREND_MIN_BAND_CELLS
                and int(counts.get(high, 0)) >= TREND_MIN_BAND_CELLS
            ):
                compared += 1
    if not compared:
        return (
            QcOutcome.not_evaluable(
                NONNEURONAL_TREND_CHECK,
                f"no non-neuronal class with >= {TREND_MIN_BAND_CELLS} cells in "
                f"both the {low} band and at >= {limit:,} counts",
                **effect,
            ),
        ), table
    return (
        QcOutcome(
            NONNEURONAL_TREND_CHECK, False, details={"n_compared": compared}, **effect
        ),
    ), table


def _gene_complexity_outcome(
    signal: GeneComplexitySignal | None, *, gap_warn: float
) -> tuple[QcOutcome, pd.DataFrame | None]:
    if signal is None:
        return (
            QcOutcome.not_evaluable(
                GENE_COMPLEXITY_CHECK,
                "no simulated n_genes: the bundle stores none (version 6, or "
                "version 7 built before the artefact; decision D19)",
            ),
            None,
        )
    table, outcome = gene_complexity_check(
        signal.native_n_genes,
        signal.native_totals,
        signal.simulated_n_genes,
        signal.simulated_depth,
        signal.grid,
        gap_warn=gap_warn,
    )
    source = {"source": dict(signal.source)} if signal.source else {}
    if not outcome.fired and not (len(table) and table["judged"].any()):
        return (
            QcOutcome.not_evaluable(
                GENE_COMPLEXITY_CHECK,
                f"no depth bin with >= {GENE_COMPLEXITY_MIN_CELLS} native and "
                "simulated cells",
                details=source,
            ),
            table,
        )
    if source:
        outcome = dataclasses.replace(outcome, details={**outcome.details, **source})
    return outcome, table


def real_data_qc(
    signals: RealQcSignals,
    trust: TrustDecision | None,
    config: AnnotationConfig,
) -> RealQcResult:
    """Run the downgrade-only real-data QC of one dataset (plan §8.8; M13).

    Records one outcome (or one per level or class) for every check of plan
    §8.8 that ``real_qc`` evaluates, reading ``config.real_qc``
    (``AnnotationRealQcConfig``): the human marker referee and registration
    G1, paired concordance, flag rates, gene complexity, the prefilter spot
    check, per-class coverage against simulation, the non-neuronal depth
    trend and the factor re-measure. The dataset gate's own outcomes (and
    mouse G1 / G2) come from ``signals.gate`` and are recorded, never
    re-applied (``not_evaluable`` without a verdict). On a seeded
    ``real_data`` family whose species gate is pending
    (``seeded_families_warn_only_until_gate``, D20 (b)), the lowering effects
    of ``SEEDED_WARN_ONLY_CHECKS`` are demoted to warnings; every other check
    applies as defined.

    Nothing here raises a trust state or a gate level, changes a margin, a
    threshold, a floor or an emission plan, or promotes a family: real data
    only warn or downgrade (OD-E1 / OD-E9).

    Args:
        signals: The dataset's inputs.
        trust: The primary panel's trust decision (seeded families:
            ``validated`` with ``validation_basis`` ``real_data``).
        config: The annotation config (``species``, ``real_qc``; human G1
            reads §7.6's fail rule from ``mouse_gate``).

    Returns:
        The QC result.

    Raises:
        ValueError: If the human marker-referee threshold is unset.
    """
    species = config.species
    settings = config.real_qc
    version = signals.resolvability_version
    outcomes: list[QcOutcome] = []
    tables: dict[str, pd.DataFrame] = {}
    if species == "human":
        warn = settings.marker_consistency_warn
        if warn is None:
            raise ValueError(
                "real_qc.marker_consistency_warn is unset: pass an AnnotationConfig, "
                "which fills the species default"
            )
        outcomes.append(
            marker_consistency_outcome(
                signals.marker_consistency,
                warn=float(warn),
                broad_only=settings.marker_consistency_broad_only,
            )
        )
        outcomes.append(
            registration_g1_outcome(
                signals.registration,
                density_ratio_fail=config.mouse_gate.g1_density_ratio_fail,
                density_ratio_warn=config.mouse_gate.g1_density_ratio_warn,
                shift_fail_um=config.mouse_gate.g1_shift_fail_um,
                effect=settings.registration_g1_effect,
            )
        )
    outcomes.append(
        paired_concordance(
            signals.pair_jsd,
            paired=signals.paired,
            warn_above=settings.paired_broad_jsd_warn,
        )
    )
    flag_rates = None
    if signals.flag_strata is None:
        outcomes.append(
            QcOutcome.not_evaluable(FLAG_RATES_CHECK, "no flag strata given")
        )
    else:
        flag_rates = flag_rate_summary(
            signals.flag_strata, warn_frac=settings.uninformative_strata_warn_frac
        )
        outcomes.append(flag_rates.outcome)
    complexity, complexity_table = _gene_complexity_outcome(
        signals.gene_complexity, gap_warn=settings.genes_per_count_gap_warn
    )
    outcomes.append(complexity)
    if complexity_table is not None:
        tables[GENE_COMPLEXITY_CHECK] = complexity_table
    outcomes.extend(
        prefilter_spotcheck(
            signals.prefilter,
            applies=signals.prefilter_applied,
            emitted_levels=signals.emitted_levels,
            min_agreement=settings.prefilter_spotcheck_min_agreement,
        )
    )
    coverage, coverage_table = _coverage_outcomes(
        signals.coverage,
        version=version,
        margin=settings.coverage_warn_margin,
        min_cells=settings.coverage_min_cells,
    )
    outcomes.extend(coverage)
    if coverage_table is not None:
        tables[COVERAGE_CHECK] = coverage_table
    trend, trend_table = _trend_outcomes(
        signals.nonneuronal_trend,
        version=version,
        limit=settings.nonneuronal_high_depth_counts,
    )
    outcomes.extend(trend)
    if trend_table is not None:
        tables[NONNEURONAL_TREND_CHECK] = trend_table
    outcomes.append(
        factor_remeasure_outcome(
            signals.factor,
            min_r=settings.factor_remeasure_min_r,
            has_r3_member=signals.has_r3_member,
        )
    )
    if signals.gate is None:
        # The dataset gate records its own outcomes (and mouse G1 / G2):
        # without its verdict they are recorded, not left out (P4).
        gate_checks = (
            (DATASET_GATE_CHECK, REGISTRATION_G1_CHECK, MARKER_CONSISTENCY_CHECK)
            if species == "mouse"
            else (DATASET_GATE_CHECK,)
        )
        outcomes.extend(
            QcOutcome.not_evaluable(check, "no dataset gate verdict given")
            for check in gate_checks
        )
    seeded = _is_seeded(trust)
    warn_only = seeded and settings.seeded_warn_only(species)
    if warn_only:
        note = (
            f"warn-only until the {species} gate merges into main (seeded "
            "real-data family; seeded_families_warn_only_until_gate)"
        )
        outcomes = [
            outcome.warn_only(note)
            if outcome.check in SEEDED_WARN_ONLY_CHECKS
            else outcome
            for outcome in outcomes
        ]
    result = RealQcResult(
        species=species,
        outcomes=tuple(outcomes),
        seeded=seeded,
        warn_only=warn_only,
        gate=signals.gate,
        flag_rates=flag_rates,
        tables=tables,
    )
    logger.info(
        "Real-data QC (%s%s): %s",
        species,
        ", warn-only" if warn_only else "",
        ", ".join(
            f"{check} {token}" for check, token in result.provenance().outcomes.items()
        ),
    )
    return result
