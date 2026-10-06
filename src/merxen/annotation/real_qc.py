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

This module imports only the standard library, numpy and pandas (and
``merxen.annotation.resolvability`` / ``schema``, which need no more), so it
imports in the GPU clustering environment.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.resolvability import STATUS_EMITTED, depth_bin
from merxen.annotation.schema import PANEL_TRUST_STATES

if TYPE_CHECKING:
    from scipy import sparse

logger = logging.getLogger(__name__)

REAL_QC_VERSION: Final = 1
# Pre-registered constants (plan §8.8, user decision 4; [L]).
COVERAGE_WARN_MARGIN: Final = 0.10
COVERAGE_MIN_CELLS: Final = 200
# A called class needs this many dataset cells for its own per-class depth
# s_c(d); a thinner class takes the label-free total-count histogram of every
# cell (the RESOLVE follow-up of plan §12 M3c). 100 is the registered
# per-class minimum of a depth profile (plan §8.3 v7.5,
# ``sim_inputs.PROFILE_MIN_CLASS_CELLS``).
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

Effect = Literal["warning", "report_only", "downgrade"]

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

    Attributes:
        check: The check (``coverage_vs_simulation``, ...).
        fired: Whether the check fired.
        effect: ``warning`` / ``report_only`` (never changes trust) or
            ``downgrade`` (lowers trust to ``trust_cap``).
        message: Human-readable text (empty when not fired).
        trust_cap: For ``downgrade``, the highest trust state kept.
        level: The level concerned, if any.
        cls: The class concerned, if any.
        details: Numbers behind the outcome.
    """

    check: str
    fired: bool
    effect: Effect
    message: str = ""
    trust_cap: str | None = None
    level: str | None = None
    cls: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the effect and the trust cap."""
        if self.effect == "downgrade":
            if self.trust_cap not in PANEL_TRUST_STATES:
                raise ValueError(
                    f"a downgrade needs a trust cap, got {self.trust_cap!r}"
                )
        elif self.trust_cap is not None:
            raise ValueError(f"a {self.effect} outcome has no trust cap")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe record."""
        return {
            "check": self.check,
            "fired": bool(self.fired),
            "effect": self.effect,
            "message": self.message,
            "trust_cap": self.trust_cap,
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
    warnings and report-only outcomes, and every passing check, leave it
    unchanged (plan §8.8: real data stay downgrade-only).

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


def qc_summary(outcomes: Iterable[QcOutcome]) -> dict[str, Any]:
    """Return the JSON-safe record of a dataset's real-data QC outcomes."""
    items = [outcome.to_json() for outcome in outcomes]
    return {
        "version": REAL_QC_VERSION,
        "outcomes": items,
        "n_fired": sum(1 for item in items if item["fired"]),
        "promotes": False,
    }
