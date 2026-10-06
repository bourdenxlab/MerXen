"""Gate-P scoring (M13, plan §14 new panel family).

Pure functions on cells tables: NP3 precision and coverage, NP4 donor / draw
stability, NP5 resolvability consistency, NP7 error structure.

Gate P validates a new panel family by simulation only (plan §8.8, §14). The
thresholds and emission are derived once from the default donor (human) or
draw (mouse) at seed 0 and then frozen; the held-out replicates are scored
at those frozen thresholds on the pooled tested sets (``gate_p_tested_sets``:
each set holds >= ``gate_p_min_confident_n`` confident calls, deep bins
pooled with each test cell counted once at its deepest bin).

NP3, precision and coverage at the panel's depth (§14 NP3): per (level,
class), at every tested set of the pooled held-out calls
(``pooled_held_out_cells``) from ``validated_min_depth`` up to and including
the ">= D_P" set, the point precision reaches target+_L (the margin of the
set's shallowest bin), its Wilson bound on the Kish effective n reaches
target_L and the coverage reaches ``gate_p_min_coverage`` (0.30), each under
two weightings of the set: the reference's natural composition and the
class-balanced one (equal total weight per truth type within the called
class's tested set; pre-registration §23.9 item 2, confirmed with D12). The
unweighted values, a family dataset's depth histogram (label-free) and the
two weightings read on the test cells of a set's scope are reported only. A
family dataset's composition is not used: it needs the dataset's labels, and
no real-label input reaches gate P other than NP5's registered profile
(§23.9 item 6).

Open (M13 review of NP3, to be put to the user before the set a dry run):
§14 asks for "the reference's natural composition within the class", and
neither the pre-registration nor the decisions of 2026-10-06 say how it is
read. The scored ``natural`` scheme is this implementation's reading: every
truth type of the called class's tested set, a wrong call's type included,
takes its share of the reference's composition. ``natural_test_cells``
(report-only) is the other candidate. On a set that is almost all right,
the set weightings' result is set mainly by the judged-set trim (10 x the
set's median weight), which caps the weight of a wrong-call type of a few
calls: on 1,000 calls, 990 right in three types and the 10 wrong ones in two
others, the class-balanced precision is .918 with the trim and .60 without
it. Until the user chooses, the code scores ``natural`` and
``class_balanced`` as written here.

NP4, stability across held-out donors (or draws) and seeds (§14 NP4): per
(level, class), at every tested set,

- each replicate with >= ``gate_p_replicate_min_confident_n`` (100)
  confident calls there has point precision >= target_L (the floor); and
- where every replicate has >= 100 calls, the range of the seed-averaged
  donor (or draw) precisions is <= max(0.03, 3.5 x pooled SE), with
  pooled SE = sqrt(p_bar (1 - p_bar) / n_bar) (the range rule).

A set where some replicate has fewer than 100 calls is scored by the floor
alone, and one where no replicate has 100 passes NP4 vacuously (``vacuous``,
reported); this is §14's literal reading.

NP4's seed criterion (seed 0 vs 1 changes <= 2% of confident labels) is a
separate check. A replicate is keyed by an opaque (group, seed label) pair:
the group is a human donor or a mouse draw, so the functions are
species-agnostic. The frozen thresholds were fitted on the default group's
fit half, so that group is scored on its check half only
(``held_out_replicates``; §14 "the default donor's check half").
Version-7 families score NP4 in every emission member (``member=``) and
combine the members with ``every_member_verdict``.

The averaging conventions of the range rule (p_bar and n_bar as means of
seed-averaged group values, unweighted precision) are D12 of the M13 plan,
confirmed by the user on 2026-10-06 (pre-registration §23.9, §23.10).

NP5, resolvability consistency (§14 NP5): per (level, class),

- the §8.3 emission decisions re-derived in each replicate
  (``np5_rederive``; version 7 ``np5_rederive_ensemble``) agree with the
  base run at every depth bin with >= 50 test cells, except at most the bin
  adjacent to the emission boundary (``np5_decision_agreement``);
- each t* varies <= 0.05 across the replicates at the tested sets
  (``np5_set_thresholds``, fitted on each replicate's rows of the set by the
  shared membership rule; ``np5_tstar_spread``); and
- its cells at the family's expected depth are at most 50%
  ``resolvability_extrapolated`` under the frozen decisions
  (``np5_extrapolated_share``). The depths are an input per class. A
  family with a per-class profile (the frozen ``sim_inputs`` profile asset
  of D8) passes each class's profile depths: the test then uses the class's
  profile shares (plan §8.3 v7.5: "uses the class's profile shares"), and
  the class's profile median is reported as its expected depth (§14
  "Version-7 families"; pre-registration §23.9 item 6, §23.10). One
  expected depth per class, whose share is 0 or 1, is only the label-free
  fallback of a family without a profile: the pooled median of its
  sections for every class.

Readings this implementation takes where §14 is not explicit (strict where
there is a choice; to be put to the user with the set a dry run): a bin is
compared when the base run or the replicate has its 50 test cells; the
emission boundaries are the base run's status changes inside the grid (its
edges are none), and at most one compared bin may flip, one that flanks a
boundary; a replicate whose t* fit exists but never reaches the target
fails the t* range, and one without a fit (too few fit-half calls) is left
out of it; "each t* ... at tested sets" is re-fitted in each replicate on
the replicate's calls of each tested set (the shared membership rule,
``np5_set_thresholds``), not read from the ``t_star`` of the replicate's
re-derived decisions, whose bins and pools are the replicate's own and need
not match the frozen tested sets; and a class without cells in the profile
takes D8's overall median as one depth (the registered words), so its
share is 0 or 1; the pooled profile's shares are the alternative.

NP7, error structure (§14 NP7), on the pooled held-out calls at seed 0 at
the frozen thresholds, per emission member for version 7
(``np7_error_structure``; unweighted, as §14 reweights NP3 only):

- human: per level, the confident calls to sink or region-implausible nodes
  are at most 1% of the level's confident calls over all classes and
  emitted bins (CHECK K9.1). Such calls have no class (``parent`` null), so
  they lie outside every tested set and are counted beside the
  denominator; a node is read against the bundle's vocab snapshot with the
  frontal-cortex plausibility column (CHECK K4), and a coarse level's call
  takes the assigned supercluster of its simulated cell;
- both species: at every tested set of a (level, class), no single wrong
  node receives more than 5% of the truth class's confident calls (D12,
  pre-registration §23.9 item 5); the called-class view is reported.

A failure of the 1% part fails NP7 for every class of the level. The
readings taken where §14 is not explicit (the confidence of a call that has
no class, a sink in the single-wrong-node view, a node outside the vocab)
are listed in ``np7_error_structure``'s docstring, for the user with the
set a dry run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
)
from merxen.annotation.thresholds import level_target
from merxen.annotation.vocab import REGION_COLUMN_PREFIX, Species

# §14 NP4: "the range ... is <= max(0.03, 3.5 x pooled SE)".
GATE_P_SPREAD_FLOOR: Final = 0.03
# A replicate: (group, seed label). The group is a held-out donor (human) or
# a test draw (mouse); the seed label names the replicate's seed. Both are
# opaque: each replicate table must hold exactly one replicate.
ReplicateKey = tuple[str, int]

_TOLERANCE: Final = 1e-9
STATS_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "set_min_depth",
    "group",
    "seed",
    "n_confident",
    "n_correct",
    "precision",
)
NP4_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "target",
    "n_replicates",
    "n_groups",
    "n_evaluated",
    "vacuous",
    "floor_ok",
    "floor_failures",
    "range_evaluable",
    "p_bar",
    "n_bar",
    "donor_range",
    "pooled_se",
    "limit",
    "range_ok",
    "passed",
)


@dataclass(frozen=True)
class Np4Settings:
    """The NP4 constants (§14 NP4; plan §3.7 ``gate_p_*``).

    Attributes:
        replicate_min_confident_n: Confident calls a replicate needs in a
            tested set for the floor to apply to it; the range rule applies
            only when every replicate has them
            (``gate_p_replicate_min_confident_n``, 100).
        spread_se_multiplier: The range limit in pooled standard errors
            (``gate_p_spread_se_multiplier``, 3.5).
        spread_floor: The smallest range limit (0.03).
    """

    replicate_min_confident_n: int
    spread_se_multiplier: float
    spread_floor: float = GATE_P_SPREAD_FLOOR

    def __post_init__(self) -> None:
        """Validate that every constant is positive.

        Raises:
            ValueError: If a constant is not > 0.
        """
        for name in (
            "replicate_min_confident_n",
            "spread_se_multiplier",
            "spread_floor",
        ):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"Np4Settings.{name} must be > 0, got {value!r}")

    @classmethod
    def from_config(cls, config: AnnotationResolvabilityConfig) -> Np4Settings:
        """Read the NP4 constants from the resolvability config (§14 NP4).

        Args:
            config: The resolvability config.

        Returns:
            The settings (the floor is the §14 constant 0.03).
        """
        return cls(
            replicate_min_confident_n=config.gate_p_replicate_min_confident_n,
            spread_se_multiplier=config.gate_p_spread_se_multiplier,
        )


def level_targets(
    thresholds: AnnotationThresholds, levels: Iterable[str]
) -> dict[str, float]:
    """Return target_L per level, the precision NP4's floor requires (§14).

    Args:
        thresholds: The threshold settings.
        levels: Levels to look up.

    Returns:
        ``{level: target}``: 0.90 for broad, lineage, NT and class, 0.85 for
        supercluster and subclass with the defaults.

    Raises:
        ValueError: For a level without a precision target.
    """
    return {level: level_target(level, thresholds) for level in levels}


def tested_set_label(item: res.GatePTestedSet) -> str:
    """Return a tested set's label: ``">=<D_P>"`` when pooled, else its depth.

    Args:
        item: A tested set (§14 evaluation rules).

    Returns:
        The label, e.g. ``">=30"`` or ``"10"``.
    """
    if item.pooled:
        return f">={min(item.depths)}"
    return str(item.depths[0])


def tested_set_mask(
    frame: pd.DataFrame, confident: np.ndarray, item: res.GatePTestedSet
) -> np.ndarray:
    """Return which rows of one replicate belong to a tested set (§14 NP3-NP7).

    One membership rule for every gate-P criterion, the one
    ``gate_p_tested_sets`` uses to build the sets:

    - a single bin holds the confident rows of the set's level, class and
      depth;
    - a pooled ">= D_P" set holds each test cell once, at its deepest row of
      the level (``deepest_rows`` over every row of the level, before any
      class or confidence filter), when that row is a confident call of the
      class at a depth >= D_P. A cell called into another class, or
      unconfident, at its deepest bin is not rescued by a shallower row.

    Args:
        frame: One replicate's rows with a fresh ``RangeIndex``
            (``reset_index(drop=True)``): the mask is positional.
        confident: ``frozen_confident_mask`` of ``frame``.
        item: The tested set.

    Returns:
        A bool array aligned with the rows of ``frame``.

    Raises:
        ValueError: If ``frame`` has another index, or ``confident`` another
            length.
    """
    mask = np.zeros(len(frame), dtype=bool)
    mask[_ReplicateIndex(frame, confident).positions(item)] = True
    return mask


class _ReplicateIndex:
    """One replicate's columns, coded once for all of its tested sets.

    ``tested_set_mask``'s membership rule on integer codes: each level's rows
    and deepest rows (``deepest_rows``) are found once per replicate, not
    once per tested set (a few hundred sets per replicate in gate P).
    """

    def __init__(self, frame: pd.DataFrame, confident: np.ndarray) -> None:
        """Code the columns of one replicate's rows.

        Args:
            frame: The replicate's rows with a fresh ``RangeIndex``.
            confident: ``frozen_confident_mask`` of ``frame``.

        Raises:
            ValueError: If ``frame`` has another index, or ``confident``
                another length.
        """
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError(
                "tested_set_mask is positional: pass the replicate's rows with a "
                "fresh RangeIndex (reset_index(drop=True))"
            )
        is_confident = np.asarray(confident, dtype=bool)
        if is_confident.shape != (len(frame),):
            raise ValueError(
                f"confident has {is_confident.shape[0]} entries for {len(frame)} rows"
            )
        self._frame = frame
        self._confident = is_confident
        level_codes, level_values = pd.factorize(frame["level"].astype(str).to_numpy())
        self._level_codes = np.asarray(level_codes, dtype=np.int64)
        self._level_code = {str(value): code for code, value in enumerate(level_values)}
        # A null parent (a sink or no call) gets code -1 and matches no class.
        parent_codes, parent_values = pd.factorize(
            frame["parent"].astype(object).to_numpy()
        )
        self._parent_codes = np.asarray(parent_codes, dtype=np.int64)
        self._parent_code = {value: code for code, value in enumerate(parent_values)}
        self._depth = frame["depth"].to_numpy(np.int64)
        self._levels: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _level_rows(self, level: str) -> tuple[np.ndarray, np.ndarray]:
        """The positions of a level's rows and of each test cell's deepest row."""
        if level not in self._levels:
            code = self._level_code.get(level)
            rows = (
                np.flatnonzero(self._level_codes == code)
                if code is not None
                else np.empty(0, dtype=np.int64)
            )
            deepest = res.deepest_rows(self._frame.iloc[rows])
            self._levels[level] = (rows, deepest.index.to_numpy(np.int64))
        return self._levels[level]

    def scope_positions(self, item: res.GatePTestedSet) -> np.ndarray:
        """Return the positions of the test cells a tested set is drawn from.

        A single bin's scope is every row of the level at its depth; a pooled
        ">= D_P" set's is each test cell's deepest row of the level when it
        lies at a depth >= D_P. Every call is kept (any class, a sink, no
        call, unconfident).

        Args:
            item: The tested set.

        Returns:
            Row positions, one per test cell.
        """
        rows, deepest = self._level_rows(item.level)
        if item.pooled:
            keep = self._depth[deepest] >= min(item.depths)
            return np.asarray(deepest[keep], dtype=np.int64)
        return np.asarray(rows[self._depth[rows] == item.depths[0]], dtype=np.int64)

    def called_positions(self, item: res.GatePTestedSet) -> np.ndarray:
        """Return the positions of the scope's calls of the set's class.

        These are the calls the coverage of the set is measured on (the
        judged set of ``decide``), confident or not.

        Args:
            item: The tested set.

        Returns:
            Row positions; empty when the replicate has no call of the class.
        """
        code = self._parent_code.get(item.cls)
        if code is None:
            return np.empty(0, dtype=np.int64)
        scope = self.scope_positions(item)
        return np.asarray(scope[self._parent_codes[scope] == code], dtype=np.int64)

    def positions(self, item: res.GatePTestedSet) -> np.ndarray:
        """Return the positions of the rows in a tested set (one per test cell).

        Args:
            item: The tested set.

        Returns:
            Row positions: the confident ones of ``called_positions``.
        """
        called = self.called_positions(item)
        return np.asarray(called[self._confident[called]], dtype=np.int64)


def held_out_replicates(
    replicates: Mapping[ReplicateKey, pd.DataFrame], *, default_group: str | None
) -> dict[ReplicateKey, pd.DataFrame]:
    """Return the replicates as gate P scores them (§14 pooled held-out calls).

    The frozen thresholds are fitted on the fit half (``half == 0``) of the
    default group's base run, so that group is scored on its check half
    only, at every seed (§14: "the default donor's check half"; human the
    default donor, mouse the draw the thresholds came from). Pre-registration
    §23.2 D2 (d) makes the replicates disjoint, so a fit-half test cell of
    the default group found in another group's table is a leak and raises.
    Use the same tables to build the pooled seed-0 tested sets
    (``gate_p_tested_sets``) and to score the replicates
    (``replicate_set_stats``, which applies this itself).

    Args:
        replicates: Per (group, seed label), that replicate's cells table;
            the default group's tables in full (both halves), so that its
            fit-half cells can be checked for in the other tables.
        default_group: The group whose fit half the frozen thresholds were
            fitted on; ``None`` when no replicate holds those cells.

    Returns:
        The tables in the order of ``replicates``, the default group's
        restricted to its check half (``half == 1``); the others unchanged.

    Raises:
        ValueError: If ``default_group`` is not a group of ``replicates``, a
            table of it has no ``half`` column or a value other than 0 and
            1, or another group's table holds one of its fit-half cells.
    """
    if default_group is None:
        return dict(replicates)
    groups = sorted({str(group) for group, _ in replicates})
    if default_group not in groups:
        raise ValueError(
            f"default_group {default_group!r} is not a group of the replicates {groups}"
        )
    result: dict[ReplicateKey, pd.DataFrame] = {}
    fit_cells: set[str] = set()
    for key, table in replicates.items():
        if str(key[0]) != default_group:
            continue
        if "half" not in table.columns:
            raise ValueError(
                f"replicate {key[0]}/{key[1]}: the default group's table needs "
                "the split-half column 'half' (0 = fit, 1 = check)"
            )
        half = table["half"].to_numpy()
        if not bool(np.isin(half, (0, 1)).all()):
            raise ValueError(
                f"replicate {key[0]}/{key[1]}: 'half' holds values other than "
                "0 (fit) and 1 (check)"
            )
        fit = half == 0
        fit_cells.update(table["cell_id"].astype(str).to_numpy()[fit])
        result[key] = table[~fit]
    for key, table in replicates.items():
        if str(key[0]) == default_group:
            continue
        leaked = table["cell_id"].astype(str).isin(fit_cells).to_numpy(bool)
        if bool(leaked.any()):
            raise ValueError(
                f"replicate {key[0]}/{key[1]} holds {int(leaked.sum())} rows of "
                f"fit-half test cells of the default group {default_group!r}: "
                "the frozen thresholds were fitted on them"
            )
        result[key] = table
    return {key: result[key] for key in replicates}


def pooled_held_out_cells(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    *,
    default_group: str | None,
    seed: int = 0,
) -> pd.DataFrame:
    """Return the pooled held-out calls of one seed (§14 "Pooled held-out calls").

    The tested sets are fixed, and NP3, NP6 and NP7 scored, at the frozen
    thresholds on every held-out call at seed 0: human, the default donor's
    check half plus both other donors (each mapped against the bundle built
    without it); mouse, both draws, the draw the thresholds came from on its
    check half (pre-registration §23.9 item 3). The tables are pooled after
    ``held_out_replicates``, so the default group's fit half never enters.

    Args:
        replicates: Per (group, seed label), that replicate's cells table
            (the default group's in full; ``held_out_replicates``).
        default_group: The group whose fit half the frozen thresholds were
            fitted on (``None`` when no replicate holds those cells).
        seed: The seed label to pool.

    Returns:
        The rows of every replicate with seed label ``seed``, in key order,
        with a fresh ``RangeIndex``.

    Raises:
        ValueError: If no replicate has seed label ``seed``, or for the
            default group's inputs (``held_out_replicates``).
    """
    scored = held_out_replicates(replicates, default_group=default_group)
    tables = [scored[key] for key in sorted(scored) if key[1] == seed]
    if not tables:
        raise ValueError(f"pooled_held_out_cells: no replicate has seed label {seed!r}")
    return pd.concat(tables, ignore_index=True)


def _check_tested(
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
) -> None:
    """Check that each key's tested sets are ``None`` or non-empty, of its key.

    Raises:
        ValueError: For an empty list (``gate_p_tested_sets`` gives ``None``
            for a key without a tested set, so ``[]`` would pass untested),
            or a tested set whose level or class differs from its key.
    """
    for key, items in tested.items():
        if items is None:
            continue
        if len(items) == 0:
            raise ValueError(
                f"{key}: an empty list of tested sets; a (level, class) that is "
                "not evaluable is None"
            )
        for item in items:
            if (item.level, item.cls) != tuple(key):
                raise ValueError(
                    f"the tested set {item.level}/{item.cls} "
                    f"{tested_set_label(item)} belongs to another key than {key}"
                )


def replicate_set_stats(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    decisions: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    *,
    default_group: str | None,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    member: str | None = None,
) -> pd.DataFrame:
    """Count each replicate's confident calls in every tested set (§14 NP4).

    The replicates are scored at the frozen thresholds of ``decisions`` on
    the tested sets fixed from the pooled seed-0 calls (``tested``, from
    ``gate_p_tested_sets`` on ``held_out_replicates``). The default group is
    scored on its check half only (``held_out_replicates``). Every (tested
    set, replicate) pair gets a row, a replicate without calls in the set
    included (``n_confident`` 0), so the range rule can see that not every
    replicate has enough calls.

    Args:
        replicates: Per (group, seed label), that replicate's cells table.
            Each table must hold exactly one replicate.
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's).
        tested: The tested sets per (level, class); ``None`` marks a
            (level, class) that is not evaluable and has no rows.
        default_group: The group whose fit half the frozen thresholds were
            fitted on (required, so that no caller leaks it by omission;
            ``None`` when no replicate holds those cells).
        regime: The regime whose thresholds are frozen.
        recipe: The recipe of the scored rows (``None``: every recipe, so
            the table must hold one).
        member: The version-7 emission member of the scored rows.

    Returns:
        One row per (tested set, replicate), sorted by level, class, set,
        group and seed, with columns ``STATS_COLUMNS``; ``precision`` is
        ``nan`` without calls.

    Raises:
        ValueError: If ``replicates`` is empty, a replicate has no rows after
            the filters, a key's tested sets are an empty list or a tested
            set's level or class differs from its key, or for the default
            group's inputs (``held_out_replicates``).
        ResolvabilityError: If a table holds more than one replicate
            (``replicate_rows``).
    """
    if not replicates:
        raise ValueError("replicate_set_stats: no replicates")
    _check_tested(tested)
    scored = held_out_replicates(replicates, default_group=default_group)
    lookup = res.emission_lookup(decisions, regime)
    ordered = sorted(tested.items(), key=lambda pair: pair[0])
    records: list[dict[str, object]] = []
    for group, seed in sorted(scored):
        frame = res.replicate_rows(
            scored[(group, seed)], recipe=recipe, seed=None, member=member
        ).reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"replicate {group}/{seed} has no rows after the filters "
                f"(recipe={recipe!r}, member={member!r}, "
                f"default_group={default_group!r}: its check half only)"
            )
        index = _ReplicateIndex(frame, res.frozen_confident_mask(frame, lookup))
        correct = frame["correct"].to_numpy(bool)
        for (level, cls), items in ordered:
            for item in items or ():
                positions = index.positions(item)
                n_confident = int(len(positions))
                n_correct = int(correct[positions].sum())
                records.append(
                    {
                        "level": level,
                        "class": cls,
                        "set": tested_set_label(item),
                        "pooled": bool(item.pooled),
                        "set_min_depth": int(min(item.depths)),
                        "group": str(group),
                        "seed": seed,
                        "n_confident": n_confident,
                        "n_correct": n_correct,
                        "precision": n_correct / n_confident
                        if n_confident
                        else math.nan,
                    }
                )
    stats = pd.DataFrame.from_records(records, columns=list(STATS_COLUMNS))
    return stats.sort_values(
        ["level", "class", "set", "group", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def _check_replicate_grid(stats: pd.DataFrame) -> tuple[list[str], list[object]]:
    """The groups and seeds of NP4's stats, which must form a full grid.

    Every tested set must hold one row per (group, seed) of the product of
    the stats' groups and seeds, with at least two groups: a missing
    replicate would otherwise shrink the range silently.
    """
    groups = sorted({str(group) for group in stats["group"]})
    seeds = sorted(set(stats["seed"]))
    if len(groups) < 2:
        raise ValueError(f"NP4 needs at least 2 groups (donors or draws), got {groups}")
    expected = sorted((group, seed) for group in groups for seed in seeds)
    for (level, cls, label), rows in stats.groupby(
        ["level", "class", "set"], sort=True
    ):
        pairs = sorted(zip(rows["group"].astype(str), rows["seed"], strict=True))
        if pairs != expected:
            raise ValueError(
                f"NP4 {level}/{cls} {label}: the replicates are not the full "
                f"grid of groups {groups} x seeds {seeds}"
            )
    return groups, seeds


def _np4_range(
    rows: pd.DataFrame, settings: Np4Settings
) -> tuple[float, float, float, float, float, bool]:
    """NP4's range rule on one tested set (D12).

    M13 plan D12, confirmed on 2026-10-06 (pre-registration §23.9 item 3,
    §23.10): per group, p_g and n_g are the arithmetic means over its seeds
    of the per-seed (unweighted) precision and confident n; p_bar and n_bar
    are the means of the group values; the limit is
    max(floor, k x sqrt(p_bar (1 - p_bar) / n_bar)). This is the registered
    E2 convention (``member_spread``; pre-registration §22.3).

    Returns:
        ``(p_bar, n_bar, donor_range, pooled_se, limit, range_ok)``.
    """
    per_group = rows.groupby(rows["group"].astype(str), sort=True).agg(
        p_g=("precision", "mean"), n_g=("n_confident", "mean")
    )
    p_bar = float(per_group["p_g"].mean())
    n_bar = float(per_group["n_g"].mean())
    donor_range = float(per_group["p_g"].max() - per_group["p_g"].min())
    pooled_se = res.pooled_standard_error(p_bar, n_bar)
    limit = max(settings.spread_floor, settings.spread_se_multiplier * pooled_se)
    return (
        p_bar,
        n_bar,
        donor_range,
        pooled_se,
        limit,
        donor_range <= limit + _TOLERANCE,
    )


def np4_set_verdicts(
    stats: pd.DataFrame, targets: Mapping[str, float], settings: Np4Settings
) -> pd.DataFrame:
    """Score NP4's floor and range rule at every tested set (§14 NP4).

    Per (level, class, set): the floor fails for each replicate with at
    least ``replicate_min_confident_n`` confident calls whose point
    precision is below target_L; the range rule applies only when every
    replicate has that many calls (otherwise it passes), and fails when the
    range of the seed-averaged group precisions exceeds
    max(``spread_floor``, ``spread_se_multiplier`` x pooled SE). A set where
    no replicate reaches the minimum is ``vacuous`` and passes (§14's literal
    reading; reported, so that a pass on nothing is visible).

    Args:
        stats: ``replicate_set_stats`` output.
        targets: target_L per level (``level_targets``).
        settings: The NP4 constants.

    Returns:
        One row per (level, class, set), columns ``NP4_COLUMNS``;
        ``floor_failures`` lists the failing replicates as ``group/seed``
        joined by ``;``. The range statistics are ``nan`` where the range
        rule does not apply.

    Raises:
        ValueError: If a level has no target, or the replicates are not the
            full grid of at least 2 groups x the same seeds.
    """
    if stats.empty:
        return pd.DataFrame(columns=list(NP4_COLUMNS))
    missing = sorted({str(level) for level in stats["level"]} - set(targets))
    if missing:
        raise ValueError(f"NP4: no precision target for levels {missing}")
    groups, _ = _check_replicate_grid(stats)
    minimum = settings.replicate_min_confident_n
    records: list[dict[str, object]] = []
    for (level, cls, label), rows in stats.groupby(
        ["level", "class", "set"], sort=True
    ):
        target = float(targets[str(level)])
        n_confident = rows["n_confident"].to_numpy(np.int64)
        precision = rows["precision"].to_numpy(np.float64)
        evaluated = n_confident >= minimum
        below = evaluated & (precision < target - _TOLERANCE)
        failures = ";".join(
            f"{group}/{seed}"
            for group, seed in zip(
                rows["group"][below].astype(str), rows["seed"][below], strict=True
            )
        )
        n_evaluated = int(evaluated.sum())
        range_evaluable = n_evaluated == len(rows)
        if range_evaluable:
            p_bar, n_bar, donor_range, pooled_se, limit, range_ok = _np4_range(
                rows, settings
            )
        else:
            p_bar = n_bar = donor_range = pooled_se = limit = math.nan
            range_ok = True
        floor_ok = not bool(below.any())
        records.append(
            {
                "level": level,
                "class": cls,
                "set": label,
                "pooled": bool(rows["pooled"].iloc[0]),
                "target": target,
                "n_replicates": len(rows),
                "n_groups": len(groups),
                "n_evaluated": n_evaluated,
                "vacuous": n_evaluated == 0,
                "floor_ok": floor_ok,
                "floor_failures": failures,
                "range_evaluable": range_evaluable,
                "p_bar": p_bar,
                "n_bar": n_bar,
                "donor_range": donor_range,
                "pooled_se": pooled_se,
                "limit": limit,
                "range_ok": range_ok,
                "passed": floor_ok and range_ok,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(NP4_COLUMNS))


def np4_class_verdicts(
    set_verdicts: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
) -> dict[tuple[str, str], bool | None]:
    """Combine NP4's set verdicts per (level, class) (§14 NP4: every tested set).

    The literal reading of §14 (M13 plan CHECK K10): an NP4 failure at any
    tested set fails the (level, class); it does not only raise
    ``validated_min_depth``. The result has the per-member shape that
    ``resolvability.every_member_verdict`` combines over the version-7
    emission members.

    Args:
        set_verdicts: ``np4_set_verdicts`` output (``level``, ``class``,
            ``set``, ``passed``).
        tested: The tested sets per (level, class).

    Returns:
        Per (level, class) of ``tested``: ``None`` when it has no tested set
        (not evaluable), ``False`` when any of its sets fails, else ``True``.

    Raises:
        ValueError: If a tested set of a key has no verdict row, a key's
            tested sets are an empty list, or a tested set's level or class
            differs from its key.
    """
    _check_tested(tested)
    passed_by_set = {
        (str(level), str(cls), str(label)): bool(passed)
        for level, cls, label, passed in zip(
            set_verdicts["level"],
            set_verdicts["class"],
            set_verdicts["set"],
            set_verdicts["passed"],
            strict=True,
        )
    }
    result: dict[tuple[str, str], bool | None] = {}
    for key, items in tested.items():
        if items is None:
            result[key] = None
            continue
        labels = [tested_set_label(item) for item in items]
        missing = [
            label for label in labels if (key[0], key[1], label) not in passed_by_set
        ]
        if missing:
            raise ValueError(f"{key}: no NP4 verdict for the tested sets {missing}")
        result[key] = all(passed_by_set[(key[0], key[1], label)] for label in labels)
    return result


# --------------------------------------------------------------------------
# NP3: precision and coverage at the panel's depth (§14 NP3)

# The weightings NP3 scores (§14 NP3: "reweighted both to the reference's
# natural composition within the class and to a class-balanced one";
# pre-registration §23.9 item 2: equal total weight to each truth type within
# the called class's tested set).
NP3_NATURAL: Final = "natural"
NP3_CLASS_BALANCED: Final = "class_balanced"
# Reported only: the unweighted values; a family dataset's depth histogram
# (§14: "values on a family dataset's depth histogram ... reported when one
# exists"); and the two weightings read on the test cells of a set's scope
# rather than on the set's calls (``np3_test_cell_weights``).
NP3_UNWEIGHTED: Final = "unweighted"
NP3_DEPTH_HISTOGRAM: Final = "depth_histogram"
NP3_NATURAL_TEST_CELLS: Final = "natural_test_cells"
NP3_CLASS_BALANCED_TEST_CELLS: Final = "class_balanced_test_cells"
NP3_SCORED_SCHEMES: Final[tuple[str, ...]] = (NP3_NATURAL, NP3_CLASS_BALANCED)
NP3_SET_SCHEMES: Final[tuple[str, ...]] = (
    NP3_UNWEIGHTED,
    NP3_NATURAL,
    NP3_CLASS_BALANCED,
    NP3_DEPTH_HISTOGRAM,
)
NP3_TEST_CELL_SCHEMES: Final[tuple[str, ...]] = (
    NP3_NATURAL_TEST_CELLS,
    NP3_CLASS_BALANCED_TEST_CELLS,
)
NP3_REASON_NOT_EVALUABLE: Final = "not_evaluable"
NP3_REASON_DEEP_SET_FAILED: Final = "deep_set_failed"
# The truth class of a test cell without one at a level (null truth_parent).
_NO_TRUTH_CLASS: Final = "<none>"
NP3_STATS_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "set_min_depth",
    "scheme",
    "scored",
    "n_called",
    "n_confident",
    "n_correct",
    "precision",
    "kish_n",
    "wilson_lb",
    "coverage",
    "max_weight_share",
)
NP3_VERDICT_COLUMNS: Final[tuple[str, ...]] = (
    *NP3_STATS_COLUMNS,
    "target",
    "target_plus",
    "min_coverage",
    "point_ok",
    "wilson_ok",
    "coverage_ok",
    "passed",
)
NP3_DEPTH_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "tested_max_depth",
    "validated_min_depth",
    "passed",
    "reason",
    "n_sets",
    "failed_sets",
    "stop_depth",
    "stop_reason",
)
# Columns of ``validated_min_depth`` that hold None (kept as Python objects).
_NULLABLE_DEPTH_COLUMNS: Final = frozenset(
    {"tested_max_depth", "validated_min_depth", "passed", "stop_depth"}
)


@dataclass(frozen=True)
class Np3Settings:
    """The NP3 constants (§14 NP3; plan §3.7).

    Attributes:
        min_coverage: Coverage every tested set needs under each weighting
            (``gate_p_min_coverage``, 0.30).
        weight_min_type_cells: Calls a truth type needs in a set to be
            weighted on its own; rarer types take the weight of their broad
            class (``weight_min_type_cells``, 20; ``composition_weights``).
        weight_trim_factor: Weights are capped at this multiple of the set's
            median positive weight, as ``decide`` trims each judged set
            (``weight_trim_factor``, 10; 0: no cap).
    """

    min_coverage: float
    weight_min_type_cells: int
    weight_trim_factor: float

    def __post_init__(self) -> None:
        """Validate the constants.

        Raises:
            ValueError: If ``min_coverage`` is outside [0, 1],
                ``weight_min_type_cells`` is below 1 or ``weight_trim_factor``
                is negative.
        """
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(
                "Np3Settings.min_coverage must lie in [0, 1], got "
                f"{self.min_coverage!r}"
            )
        if self.weight_min_type_cells < 1:
            raise ValueError(
                "Np3Settings.weight_min_type_cells must be >= 1, got "
                f"{self.weight_min_type_cells!r}"
            )
        if not self.weight_trim_factor >= 0.0:
            raise ValueError(
                "Np3Settings.weight_trim_factor must be >= 0, got "
                f"{self.weight_trim_factor!r}"
            )

    @classmethod
    def from_config(cls, config: AnnotationResolvabilityConfig) -> Np3Settings:
        """Read the NP3 constants from the resolvability config (§14 NP3).

        Args:
            config: The resolvability config.

        Returns:
            The settings.
        """
        return cls(
            min_coverage=config.gate_p_min_coverage,
            weight_min_type_cells=config.weight_min_type_cells,
            weight_trim_factor=config.weight_trim_factor,
        )


@dataclass(frozen=True)
class WeightedTestedSet:
    """A gate-P tested set scored under one weighting (§14 NP3).

    The weighted variant of ``resolvability.GatePTestedSet``: the Wilson
    bound uses the Kish effective n of the weights (§14: "Wilson bounds on
    reweighted sets use the Kish effective n"). Only calls with a positive
    weight count, as in ``check_threshold``.

    Attributes:
        level: Level.
        cls: Class.
        depths: Depth bins in the set (one bin, or a pooled ">= D_P" set).
        pooled: Whether bins were pooled from the deep end.
        scheme: The weighting (``NP3_SET_SCHEMES``, ``NP3_TEST_CELL_SCHEMES``).
        n_called: Calls of the class in the set's scope (the coverage's
            denominator).
        n_confident: Confident calls in the set.
        n_correct: The correct ones among them.
        precision: Their weighted precision (``nan`` without weight).
        kish_n: The Kish effective n of their weights.
        wilson_lb: The Wilson 95% lower bound of the precision on ``kish_n``.
        coverage: The weighted share of the scope's calls that are confident.
        max_weight_share: The largest single confident call's share of the
            confident weight.
    """

    level: str
    cls: str
    depths: tuple[int, ...]
    pooled: bool
    scheme: str
    n_called: int
    n_confident: int
    n_correct: int
    precision: float
    kish_n: float
    wilson_lb: float
    coverage: float
    max_weight_share: float


def _one_set(types: np.ndarray) -> pd.DataFrame:
    """A frame ``composition_weights`` reweights as one group (one set)."""
    return pd.DataFrame(
        {
            "recipe": "set",
            "seed": 0,
            "level": "set",
            "depth": 0,
            res.TRUTH_LEAF_COLUMN: types,
        }
    )


def _natural_shares(
    composition: Mapping[str, float] | None, types: np.ndarray
) -> dict[str, float]:
    """Check the reference's natural composition against the test types.

    Raises:
        ValueError: Without a composition, for a share that is not finite or
            is negative, or when a test type has no positive share: its
            calls would take weight 0 and drop out of the set, which could
            only raise the set's precision.
    """
    if composition is None:
        raise ValueError(
            "the natural weighting needs the reference's natural composition "
            "(share per truth type)"
        )
    shares = {str(name): float(value) for name, value in composition.items()}
    invalid = sorted(
        name
        for name, value in shares.items()
        if not math.isfinite(value) or value < 0.0
    )
    if invalid:
        raise ValueError(
            f"the natural composition has shares that are not finite and >= 0: "
            f"{invalid}"
        )
    missing = sorted(
        {str(name) for name in np.unique(types) if not shares.get(str(name), 0.0) > 0}
    )
    if missing:
        raise ValueError(
            f"the natural composition has no positive share for the truth types "
            f"{missing}: their calls would weigh nothing"
        )
    return shares


def _depth_histogram(histogram: Mapping[int, float]) -> dict[int, float]:
    """Check a family dataset's depth histogram (mass per grid bin).

    Raises:
        ValueError: For a mass that is not finite or is negative.
    """
    masses = {int(depth): float(mass) for depth, mass in histogram.items()}
    invalid = sorted(
        depth for depth, mass in masses.items() if not math.isfinite(mass) or mass < 0.0
    )
    if invalid:
        raise ValueError(
            f"the depth histogram has masses that are not finite and >= 0 at the "
            f"bins {invalid}"
        )
    return masses


def _depth_weights(depths: np.ndarray, histogram: Mapping[int, float]) -> np.ndarray:
    """Weights that give each depth bin of a set its share of the histogram.

    ``w(d) = h(d) / p(d)``, with ``h`` the histogram's mass over the set's
    bins normalised and ``p(d)`` the set's share of rows at ``d`` (mean 1);
    all zero when the histogram has no mass on the set's bins.
    """
    values, inverse, counts = np.unique(depths, return_inverse=True, return_counts=True)
    mass = np.array([histogram.get(int(value), 0.0) for value in values])
    if not float(mass.sum()) > 0.0:
        return np.zeros(len(depths), dtype=np.float64)
    per_depth = (mass / mass.sum()) / (counts / counts.sum())
    return np.asarray(per_depth[inverse], dtype=np.float64)


def np3_set_weights(
    rows: pd.DataFrame,
    scheme: str,
    *,
    composition: Mapping[str, float] | None = None,
    depth_histogram: Mapping[int, float] | None = None,
    class_of: Mapping[str, str] | None = None,
    min_type_cells: int = 20,
    trim_factor: float = 0.0,
) -> np.ndarray:
    """Return the NP3 weights of one set of calls, weighted as one set (§14 NP3).

    The schemes, each over the truth types (``truth_leaf``) of ``rows``:

    - ``natural``: each type's total weight follows its share in the
      reference's natural composition (this implementation's reading of
      §14's "natural composition within the class"; open, see the module
      docstring);
    - ``class_balanced``: each type gets the same total weight
      (pre-registration §23.9 item 2: "equal total weight to each truth type
      within the called class's tested set");
    - ``unweighted``: weight 1;
    - ``depth_histogram`` (report-only): each depth bin of the set gets its
      share of a family dataset's depth histogram, restricted to the set's
      bins (label-free).

    ``natural`` and ``class_balanced`` reuse ``composition_weights``: a type
    with fewer than ``min_type_cells`` rows takes the weight of its broad
    class's common types (or of its broad class as a whole), so a handful of
    calls cannot stand for a whole type. Every scheme but ``unweighted`` is
    then capped at ``trim_factor`` x the set's median positive weight
    (``trim_weights``; the judged-set trim of ``decide``).

    Args:
        rows: The set's rows (``truth_leaf``; ``depth`` for the histogram;
            ``truth_parent`` and ``level`` when ``class_of`` is not given).
        scheme: One of ``NP3_SET_SCHEMES``.
        composition: The reference's natural share per truth type (any
            scale; ``natural`` only). Every type of ``rows`` needs a positive
            share.
        depth_histogram: The family dataset's cell mass per grid bin
            (``depth_histogram`` only).
        class_of: Broad class per truth type, the pooling unit of rare types
            (default ``leaf_class_map(rows)``).
        min_type_cells: ``weight_min_type_cells``.
        trim_factor: ``weight_trim_factor`` (``np3_set_stats`` passes the
            config's; 0: no cap).

    Returns:
        Weights aligned with ``rows``.

    Raises:
        ValueError: For an unknown scheme, a missing or invalid composition or
            histogram.
    """
    if scheme not in NP3_SET_SCHEMES:
        raise ValueError(
            f"np3_set_weights: unknown scheme {scheme!r}, expected one of "
            f"{NP3_SET_SCHEMES} (test-cell schemes: np3_test_cell_weights)"
        )
    if scheme == NP3_UNWEIGHTED:
        return np.ones(len(rows), dtype=np.float64)
    if scheme == NP3_DEPTH_HISTOGRAM:
        if depth_histogram is None:
            raise ValueError("the depth_histogram weighting needs a depth histogram")
        histogram = _depth_histogram(depth_histogram)
        if rows.empty:
            return np.zeros(0, dtype=np.float64)
        weights = _depth_weights(rows["depth"].to_numpy(np.int64), histogram)
        return res.trim_weights(weights, trim_factor)
    types = rows[res.TRUTH_LEAF_COLUMN].astype(str).to_numpy()
    if scheme == NP3_NATURAL:
        target = _natural_shares(composition, types)
    else:
        target = dict.fromkeys((str(name) for name in np.unique(types)), 1.0)
    if rows.empty:
        return np.zeros(0, dtype=np.float64)
    weights = res.composition_weights(
        _one_set(types),
        target,
        class_of=res.leaf_class_map(rows) if class_of is None else class_of,
        min_type_cells=min_type_cells,
    )
    return res.trim_weights(weights, trim_factor)


def np3_test_cell_weights(
    scope: pd.DataFrame,
    scheme: str,
    *,
    composition: Mapping[str, float] | None = None,
    min_type_cells: int = 20,
) -> np.ndarray:
    """Return the report-only test-cell weights of a tested set's scope.

    Reweights the test cells a tested set is drawn from (one level: a bin's
    rows, or each cell's deepest row at >= D_P), as RESOLVE reweights test
    cells to a dataset's composition, rather than the set's calls: within
    each truth class of the level (``truth_parent``), its truth types are
    rebalanced to the reference's natural shares (``natural_test_cells``) or
    to equal shares (``class_balanced_test_cells``), and each class keeps its
    share of the test cells. A class's tested set then takes the weights of
    its calls, so a wrong call weighs what its type's test cells weigh, not
    a share of the set. ``class_balanced_test_cells`` is not the registered
    class-balanced weighting (§23.9 item 2). ``natural_test_cells`` is the
    other reading of §14's natural composition "within the class"; which
    reading NP3 scores is open (module docstring), and until the user
    chooses, the set reading (``np3_set_weights``) is scored and this one is
    reported beside it.

    Args:
        scope: The scope's rows (``truth_leaf``, ``truth_parent``, ``level``).
        scheme: One of ``NP3_TEST_CELL_SCHEMES``.
        composition: The reference's natural share per truth type
            (``natural_test_cells`` only).
        min_type_cells: ``weight_min_type_cells``: rarer types take the weight
            of their truth class's common types (``composition_weights``).

    Returns:
        Weights aligned with ``scope`` (mean 1).

    Raises:
        ValueError: For an unknown scheme, rows of several levels, a truth
            type with more than one truth class, or a missing or invalid
            composition.
    """
    if scheme not in NP3_TEST_CELL_SCHEMES:
        raise ValueError(
            f"np3_test_cell_weights: unknown scheme {scheme!r}, expected one of "
            f"{NP3_TEST_CELL_SCHEMES}"
        )
    if scope.empty:
        return np.zeros(0, dtype=np.float64)
    if scope["level"].astype(str).nunique() > 1:
        raise ValueError(
            "np3_test_cell_weights: the scope holds rows of several levels"
        )
    types = scope[res.TRUTH_LEAF_COLUMN].astype(str).to_numpy()
    classes = np.array(
        [
            _NO_TRUTH_CLASS if pd.isna(value) else str(value)
            for value in scope["truth_parent"].astype(object)
        ],
        dtype=object,
    )
    pairs = pd.DataFrame({"type": types, "class": classes}).drop_duplicates()
    duplicated = pairs["type"].duplicated(keep=False)
    if bool(duplicated.any()):
        raise ValueError(
            f"the truth types {sorted(set(pairs.loc[duplicated, 'type']))} have "
            "more than one truth class at this level"
        )
    shares = (
        _natural_shares(composition, types)
        if scheme == NP3_NATURAL_TEST_CELLS
        else None
    )
    class_share = pd.Series(classes).value_counts(normalize=True)
    target: dict[str, float] = {}
    for cls, members in pairs.groupby("class", sort=True)["type"]:
        names = [str(name) for name in members]
        if shares is None:
            within = dict.fromkeys(names, 1.0 / len(names))
        else:
            total = sum(shares[name] for name in names)
            within = {name: shares[name] / total for name in names}
        for name in names:
            target[name] = float(class_share[cls]) * within[name]
    return res.composition_weights(
        _one_set(types),
        target,
        class_of=dict(zip(pairs["type"], pairs["class"], strict=True)),
        min_type_cells=min_type_cells,
    )


def _weighted_set(
    item: res.GatePTestedSet,
    scheme: str,
    correct: np.ndarray,
    weights: np.ndarray,
    called_confident: np.ndarray,
    called_weights: np.ndarray,
) -> WeightedTestedSet:
    """Score one tested set under one weighting (``check_threshold`` rules).

    ``correct`` and ``weights`` are the set's confident calls; the coverage
    is the confident share of the scope's calls of the class, on their own
    weights (``called_weights``; ``called_confident`` marks the confident
    ones).
    """
    positive = weights > 0
    kept = weights[positive]
    total = float(kept.sum())
    n_confident = int(positive.sum())
    if total > 0:
        precision = float((kept * correct[positive]).sum()) / total
        kish_n = res.kish_effective_n(kept)
        wilson_lb = res.wilson_lower_bound(precision, kish_n)
        max_weight_share = float(kept.max()) / total
    else:
        precision = wilson_lb = max_weight_share = math.nan
        kish_n = 0.0
    called_positive = called_weights > 0
    called_total = float(called_weights[called_positive].sum())
    coverage = (
        float(called_weights[called_positive & called_confident].sum()) / called_total
        if called_total > 0
        else math.nan
    )
    return WeightedTestedSet(
        level=item.level,
        cls=item.cls,
        depths=tuple(int(depth) for depth in item.depths),
        pooled=bool(item.pooled),
        scheme=scheme,
        n_called=int(called_positive.sum()),
        n_confident=n_confident,
        n_correct=int(correct[positive].sum()),
        precision=precision,
        kish_n=kish_n,
        wilson_lb=wilson_lb,
        coverage=coverage,
        max_weight_share=max_weight_share,
    )


def np3_set_stats(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    decisions: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    *,
    default_group: str | None,
    composition: Mapping[str, float],
    settings: Np3Settings,
    depth_histogram: Mapping[int, float] | None = None,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    seed: int = 0,
    member: str | None = None,
) -> pd.DataFrame:
    """Score every tested set of the pooled held-out calls (§14 NP3).

    NP3 is scored on the base recipe at seed 0, on all held-out calls at the
    frozen thresholds of ``decisions``, on the tested sets fixed from the
    same calls (``gate_p_tested_sets`` on ``pooled_held_out_cells``; version
    7: per emission member, ``gate_p_member_sets``). The calls are pooled
    here (``pooled_held_out_cells``), so the default group's fit half, on
    which the frozen thresholds were fitted, cannot enter NP3 by omission:
    ``default_group`` is required, as in ``replicate_set_stats``. Each set is
    scored under every scheme of ``np3_set_weights`` (``depth_histogram``
    only with a histogram) and of ``np3_test_cell_weights``; NP3's verdict
    uses ``NP3_SCORED_SCHEMES`` only (``np3_verdicts``).

    A set's precision, Kish n and Wilson bound are on its confident calls,
    weighted as one set; its coverage is the confident share of the calls of
    the class in the set's scope (a bin's calls, or the cells' deepest calls
    at >= D_P), weighted the same way as a set of their own.

    Args:
        replicates: Per (group, seed label), that replicate's cells table
            (the default group's in full; ``held_out_replicates``). Pooled
            donors or draws need distinct cell ids.
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's).
        tested: The tested sets per (level, class); ``None`` marks a
            (level, class) that is not evaluable and has no rows.
        default_group: The group whose fit half the frozen thresholds were
            fitted on (required, so that no caller leaks it by omission;
            ``None`` when no replicate holds those cells).
        composition: The reference's natural share per truth type. Every
            truth type of the held-out calls needs a positive share, the
            types of each donor's non-frontal top-up cells included
            (pre-registration §23.2 D2 (d)): a share table of the frontal
            cortex alone raises when a top-up type is missing from it.
        settings: The NP3 constants.
        depth_histogram: A family dataset's cell mass per grid bin
            (label-free; report-only).
        regime: The regime whose thresholds are frozen.
        recipe: The recipe of the scored rows (``None``: every recipe, so
            the tables must hold one).
        seed: The seed label of the replicates pooled
            (``pooled_held_out_cells``); their rows are kept at the same
            mapping seed, as ``gate_p_tested_sets`` keeps its rows.
        member: The version-7 emission member of the scored rows.

    Returns:
        One row per (tested set, scheme), sorted by level, class and set,
        columns ``NP3_STATS_COLUMNS`` (``scored``: the scheme decides NP3).

    Raises:
        ValueError: If no replicate has seed label ``seed``, no row is left
            after the filters, a key's tested sets are an empty list or of
            another key, for an invalid composition or histogram, or for the
            default group's inputs (``held_out_replicates``).
        ResolvabilityError: If the rows mix replicates (``replicate_rows``).
    """
    _check_tested(tested)
    histogram = None if depth_histogram is None else _depth_histogram(depth_histogram)
    cells = pooled_held_out_cells(replicates, default_group=default_group, seed=seed)
    frame = res.replicate_rows(
        cells, recipe=recipe, seed=seed, member=member
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"np3_set_stats: no rows after the filters (recipe={recipe!r}, "
            f"seed={seed!r}, member={member!r})"
        )
    _natural_shares(composition, frame[res.TRUTH_LEAF_COLUMN].astype(str).to_numpy())
    confident = res.frozen_confident_mask(frame, res.emission_lookup(decisions, regime))
    index = _ReplicateIndex(frame, confident)
    correct = frame["correct"].to_numpy(bool)
    columns = [res.TRUTH_LEAF_COLUMN, "truth_parent", "level", "depth"]
    slim = frame[columns]
    class_of = res.leaf_class_map(frame)
    set_schemes = tuple(
        scheme
        for scheme in NP3_SET_SCHEMES
        if scheme != NP3_DEPTH_HISTOGRAM or histogram is not None
    )
    trim = settings.weight_trim_factor
    options: dict[str, Any] = {
        "composition": composition,
        "depth_histogram": histogram,
        "class_of": class_of,
        "min_type_cells": settings.weight_min_type_cells,
        "trim_factor": trim,
    }
    # Per scope (level, pooled, D_P or depth) and test-cell scheme: the
    # scope's positions in ascending order and their weights.
    scopes: dict[tuple[str, bool, int, str], tuple[np.ndarray, np.ndarray]] = {}

    def scope_weights(
        item: res.GatePTestedSet, scheme: str
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (item.level, bool(item.pooled), int(min(item.depths)), scheme)
        if key not in scopes:
            positions = index.scope_positions(item)
            weights = np3_test_cell_weights(
                slim.iloc[positions],
                scheme,
                composition=composition,
                min_type_cells=settings.weight_min_type_cells,
            )
            order = np.argsort(positions, kind="mergesort")
            scopes[key] = (positions[order], weights[order])
        return scopes[key]

    records: list[dict[str, object]] = []
    for (level, cls), items in sorted(tested.items(), key=lambda pair: pair[0]):
        for item in items or ():
            called = index.called_positions(item)
            called_confident = confident[called]
            in_set = called[called_confident]
            set_correct = correct[in_set]
            results: list[WeightedTestedSet] = []
            for scheme in set_schemes:
                weights = np3_set_weights(slim.iloc[in_set], scheme, **options)
                called_weights = np3_set_weights(slim.iloc[called], scheme, **options)
                results.append(
                    _weighted_set(
                        item,
                        scheme,
                        set_correct,
                        weights,
                        called_confident,
                        called_weights,
                    )
                )
            for scheme in NP3_TEST_CELL_SCHEMES:
                positions, weights = scope_weights(item, scheme)
                called_weights = res.trim_weights(
                    weights[np.searchsorted(positions, called)], trim
                )
                results.append(
                    _weighted_set(
                        item,
                        scheme,
                        set_correct,
                        called_weights[called_confident],
                        called_confident,
                        called_weights,
                    )
                )
            for result in results:
                records.append(
                    {
                        "level": level,
                        "class": cls,
                        "set": tested_set_label(item),
                        "pooled": result.pooled,
                        "set_min_depth": int(min(item.depths)),
                        "scheme": result.scheme,
                        "scored": result.scheme in NP3_SCORED_SCHEMES,
                        "n_called": result.n_called,
                        "n_confident": result.n_confident,
                        "n_correct": result.n_correct,
                        "precision": result.precision,
                        "kish_n": result.kish_n,
                        "wilson_lb": result.wilson_lb,
                        "coverage": result.coverage,
                        "max_weight_share": result.max_weight_share,
                    }
                )
    stats = pd.DataFrame.from_records(records, columns=list(NP3_STATS_COLUMNS))
    return stats.sort_values(["level", "class", "set"], kind="mergesort").reset_index(
        drop=True
    )


def np3_verdicts(
    stats: pd.DataFrame, thresholds: AnnotationThresholds, settings: Np3Settings
) -> pd.DataFrame:
    """Judge every (tested set, scheme) of ``np3_set_stats`` (§14 NP3).

    A row passes when its point precision reaches target+_L, the provisional
    target with the margin of the set's shallowest bin (+0.05 at >= 60
    counts, +0.10 below, capped at 0.97: ``provisional_target``, with the
    60-count switch of ``RuleSettings``), its Wilson bound on the Kish n
    reaches target_L, and its coverage reaches ``min_coverage``; a ``nan``
    fails. Report-only schemes are judged too (``scored`` False).

    Args:
        stats: ``np3_set_stats`` output.
        thresholds: The threshold settings (targets and margins).
        settings: The NP3 constants.

    Returns:
        ``stats`` with ``target``, ``target_plus``, ``min_coverage``,
        ``point_ok``, ``wilson_ok``, ``coverage_ok`` and ``passed``
        (columns ``NP3_VERDICT_COLUMNS``).

    Raises:
        ValueError: For a level without a precision target.
    """
    if stats.empty:
        return pd.DataFrame(columns=list(NP3_VERDICT_COLUMNS))
    frame = stats.copy()
    targets = level_targets(
        thresholds, sorted({str(level) for level in frame["level"]})
    )
    below = thresholds.second_vote_below_counts
    target = np.array([targets[str(level)] for level in frame["level"]])
    target_plus = np.array(
        [
            thresholds.provisional_target(
                targets[str(level)], below60=int(depth) < below
            )
            for level, depth in zip(frame["level"], frame["set_min_depth"], strict=True)
        ]
    )
    frame["target"] = target
    frame["target_plus"] = target_plus
    frame["min_coverage"] = settings.min_coverage
    precision = frame["precision"].to_numpy(np.float64)
    wilson = frame["wilson_lb"].to_numpy(np.float64)
    coverage = frame["coverage"].to_numpy(np.float64)
    frame["point_ok"] = precision >= target_plus - _TOLERANCE
    frame["wilson_ok"] = wilson >= target - _TOLERANCE
    frame["coverage_ok"] = coverage >= settings.min_coverage - _TOLERANCE
    frame["passed"] = frame["point_ok"] & frame["wilson_ok"] & frame["coverage_ok"]
    return frame[list(NP3_VERDICT_COLUMNS)]


def _np3_set_passed(
    verdicts: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[str, bool]]:
    """The scored verdicts per (level, class, set) and scheme."""
    result: dict[tuple[str, str, str], dict[str, bool]] = {}
    if verdicts.empty:
        return result
    scored = verdicts[verdicts["scored"].astype(bool).to_numpy()]
    for level, cls, label, scheme, passed in zip(
        scored["level"],
        scored["class"],
        scored["set"],
        scored["scheme"],
        scored["passed"],
        strict=True,
    ):
        result.setdefault((str(level), str(cls), str(label)), {})[str(scheme)] = bool(
            passed
        )
    return result


def validated_min_depth(
    verdicts: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    depths: Sequence[int],
) -> pd.DataFrame:
    """Return NP3's per-(level, class) record: D_P and ``validated_min_depth``.

    §14 per-class records: a tested set passes NP3 when it passes under every
    scheme of ``NP3_SCORED_SCHEMES``. D_P (``tested_max_depth``) is the
    shallowest bin of the pooled ">= D_P" set, or the deepest bin when that
    bin is tested on its own (no pool). NP3 holds for the (level, class)
    when every tested set at or above D_P passes (the ">= D_P" set, and any
    bin there tested on its own); ``validated_min_depth`` is then the
    shallowest bin from which every grid bin below D_P is tested on its own
    and passes: the walk down from D_P stops at the first bin that is not
    tested on its own (``untested``) or fails (``failed``). A failing class
    does not block the other classes.

    Args:
        verdicts: ``np3_verdicts`` output (``level``, ``class``, ``set``,
            ``scheme``, ``scored``, ``passed``).
        tested: The tested sets per (level, class).
        depths: The simulation's depth grid (the bundle's ``depth_grid``).

    Returns:
        One row per (level, class) of ``tested``, columns
        ``NP3_DEPTH_COLUMNS``: ``passed`` True, False or None (not
        evaluable); ``validated_min_depth`` None unless passed;
        ``failed_sets`` the labels of every failing tested set, ``;``
        joined; ``stop_depth`` and ``stop_reason`` where the walk stopped.

    Raises:
        ValueError: If a tested set has no verdict for a scored scheme, a
            key has more than one pooled set or a tested depth is not in
            the grid, or a key's tested sets are an empty list or of
            another key.
    """
    _check_tested(tested)
    grid = sorted({int(depth) for depth in depths})
    passed_by_set = _np3_set_passed(verdicts)
    records: list[dict[str, object]] = []
    for key in sorted(tested):
        items = tested[key]
        record: dict[str, object] = {
            "level": key[0],
            "class": key[1],
            "tested_max_depth": None,
            "validated_min_depth": None,
            "passed": None,
            "reason": NP3_REASON_NOT_EVALUABLE,
            "n_sets": 0,
            "failed_sets": "",
            "stop_depth": None,
            "stop_reason": "",
        }
        records.append(record)
        if items is None:
            continue
        pooled = [item for item in items if item.pooled]
        if len(pooled) > 1:
            raise ValueError(
                f"{key}: {len(pooled)} pooled tested sets; a (level, class) has "
                "at most one"
            )
        off_grid = sorted(
            {int(depth) for item in items for depth in item.depths} - set(grid)
        )
        if off_grid:
            raise ValueError(
                f"{key}: the tested depths {off_grid} are not in the grid {grid}"
            )
        set_passed: dict[str, bool] = {}
        for item in items:
            label = tested_set_label(item)
            schemes = passed_by_set.get((key[0], key[1], label), {})
            missing = [scheme for scheme in NP3_SCORED_SCHEMES if scheme not in schemes]
            if missing:
                raise ValueError(
                    f"{key}: no NP3 verdict for the tested set {label} under {missing}"
                )
            set_passed[label] = all(schemes[scheme] for scheme in NP3_SCORED_SCHEMES)
        d_p = (
            int(min(pooled[0].depths))
            if pooled
            else max(int(item.depths[0]) for item in items)
        )
        deep_ok = all(
            set_passed[tested_set_label(item)]
            for item in items
            if min(item.depths) >= d_p
        )
        record.update(
            tested_max_depth=d_p,
            n_sets=len(items),
            failed_sets=";".join(
                label for label, passed in set_passed.items() if not passed
            ),
        )
        if not deep_ok:
            record.update(passed=False, reason=NP3_REASON_DEEP_SET_FAILED)
            continue
        single = {
            int(item.depths[0]): tested_set_label(item)
            for item in items
            if not item.pooled
        }
        minimum = d_p
        for depth in reversed([value for value in grid if value < d_p]):
            own = single.get(depth)
            if own is None or not set_passed[own]:
                record.update(
                    stop_depth=depth,
                    stop_reason="untested" if own is None else "failed",
                )
                break
            minimum = depth
        record.update(passed=True, reason="", validated_min_depth=minimum)
    return pd.DataFrame(
        {
            column: pd.Series(
                [record[column] for record in records],
                dtype=object if column in _NULLABLE_DEPTH_COLUMNS else None,
            )
            for column in NP3_DEPTH_COLUMNS
        }
    )


def np3_class_verdicts(
    table: pd.DataFrame,
) -> dict[tuple[str, str], bool | None]:
    """Return NP3's verdict per (level, class) (``validated_min_depth`` rows).

    The result has the per-member shape that
    ``resolvability.every_member_verdict`` combines over the version-7
    emission members.

    A missing ``passed`` (``None``, or the ``nan`` a CSV round trip turns it
    into) stays not evaluable; it is never read as a pass.

    Args:
        table: ``validated_min_depth`` output.

    Returns:
        Per (level, class): ``True`` (passes), ``False`` (fails) or ``None``
        (not evaluable).
    """
    return {
        (str(level), str(cls)): None
        if passed is None or pd.isna(passed)
        else bool(passed)
        for level, cls, passed in zip(
            table["level"], table["class"], table["passed"], strict=True
        )
    }


# --------------------------------------------------------------------------
# NP5: resolvability consistency (§14 NP5)

# §14 NP5: "each t* varies <= 0.05 across replicates at tested sets".
NP5_MAX_TSTAR_SPREAD: Final = 0.05
# §14 NP5: "> 50% resolvability_extrapolated" at the expected depth fails.
NP5_MAX_EXTRAPOLATED_SHARE: Final = 0.5
# Where a class's expected depth comes from (``np5_extrapolated_share``).
NP5_DEPTH_FROM_CLASS: Final = "class"
NP5_DEPTH_FROM_DEFAULT: Final = "default"
NP5_AGREEMENT_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "group",
    "seed",
    "n_bins",
    "n_compared",
    "boundary_depths",
    "flipped_depths",
    "n_flipped",
    "passed",
)
NP5_THRESHOLD_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "set_min_depth",
    "group",
    "seed",
    "n_called",
    "n_fit",
    "fitted",
    "target",
    "t_star",
    "threshold",
    "threshold_source",
)
NP5_SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "n_replicates",
    "n_fitted",
    "n_thresholds",
    "unfitted",
    "missing",
    "threshold_min",
    "threshold_max",
    "spread",
    "max_spread",
    "evaluable",
    "passed",
)
NP5_EXTRAPOLATED_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "depth_source",
    "n_depths",
    "expected_depth",
    "expected_bin",
    "extrapolated_share",
    "above_grid_share",
    "max_share",
    "passed",
)
_DECISION_KEY: Final[tuple[str, ...]] = ("level", "class", "depth")


@dataclass(frozen=True)
class Np5Settings:
    """The NP5 constants (§14 NP5; plan §3.7).

    Attributes:
        min_test_cells: Test cells a depth bin needs, in the base run or the
            replicate, for its emission decision to be compared (§14: "every
            depth bin with >= 50 test cells"; ``min_cells_per_bin``, 50, the
            rule of D_max).
        max_tstar_spread: The largest range of t* across the replicates at a
            tested set (0.05).
        max_extrapolated_share: The largest share of a class's cells at the
            family's expected depth that may be ``resolvability_extrapolated``
            (0.5).
    """

    min_test_cells: int
    max_tstar_spread: float = NP5_MAX_TSTAR_SPREAD
    max_extrapolated_share: float = NP5_MAX_EXTRAPOLATED_SHARE

    def __post_init__(self) -> None:
        """Validate the constants.

        Raises:
            ValueError: If ``min_test_cells`` is below 1, ``max_tstar_spread``
                is negative or ``max_extrapolated_share`` is outside [0, 1].
        """
        if self.min_test_cells < 1:
            raise ValueError(
                f"Np5Settings.min_test_cells must be >= 1, got {self.min_test_cells!r}"
            )
        if not self.max_tstar_spread >= 0.0:
            raise ValueError(
                "Np5Settings.max_tstar_spread must be >= 0, got "
                f"{self.max_tstar_spread!r}"
            )
        if not 0.0 <= self.max_extrapolated_share <= 1.0:
            raise ValueError(
                "Np5Settings.max_extrapolated_share must lie in [0, 1], got "
                f"{self.max_extrapolated_share!r}"
            )

    @classmethod
    def from_config(cls, config: AnnotationResolvabilityConfig) -> Np5Settings:
        """Read the NP5 constants from the resolvability config (§14 NP5).

        Args:
            config: The resolvability config.

        Returns:
            The settings (the t* spread and the extrapolated share are the §14
            constants 0.05 and 0.5).
        """
        return cls(min_test_cells=config.min_cells_per_bin)


def _one_seed(frame: pd.DataFrame, name: str) -> int:
    """The one mapping seed of a replicate's rows.

    Raises:
        ValueError: If the rows are empty or hold more than one seed.
    """
    if frame.empty:
        raise ValueError(f"{name}: no rows after the filters")
    seeds = pd.unique(frame["seed"])
    if len(seeds) != 1:
        raise ValueError(
            f"{name}: the rows hold the mapping seeds {sorted(seeds.tolist())}; "
            "pass one replicate (one seed) per table"
        )
    return int(seeds[0])


def _require_fit_half(frame: pd.DataFrame, name: str) -> None:
    """Refuse a replicate's rows that hold check-half rows but no fit-half rows.

    NP5 re-derives each replicate's emission and t* on the replicate's own
    fit half, so it takes every replicate's table in full, the default
    group's included. ``held_out_replicates`` keeps only the default group's
    check half (``half == 1``): given that output, the group's fit would be
    empty, and its replicates would be left out of the t* range as
    ``unfitted`` without an error.

    Raises:
        ValueError: If ``frame`` has rows with ``half == 1`` and none with
            ``half == 0``.
    """
    if "half" not in frame.columns:
        return
    half = frame["half"].to_numpy()
    if bool((half == 1).any()) and not bool((half == 0).any()):
        raise ValueError(
            f"{name}: the rows hold check-half rows (half == 1) and no fit-half "
            "rows; NP5 re-derives each replicate on its own fit half, so pass "
            "the replicate tables in full, not held_out_replicates' output"
        )


def np5_rederive(
    cells: pd.DataFrame,
    levels: Sequence[res.LevelMeta],
    depths: Sequence[int],
    settings: res.RuleSettings,
    *,
    recipe: str | None = res.DECISION_RECIPE,
    member: str | None = None,
    saturated_bp_share: float | None = None,
) -> pd.DataFrame:
    """Re-derive one replicate's emission decisions by ``decide`` (§14 NP5).

    NP5 re-derives the §8.3 decisions in each replicate (another donor or
    draw, another mapping seed) from its own rows, with the base run's rule
    settings, and compares them with the base run
    (``np5_decision_agreement``). The replicate's thresholds are fitted on
    its own fit half; nothing here is scored at the frozen thresholds, so
    the default group's table is used in full. The rows are kept at their
    own mapping seed.

    Version 7: one emission member's decisions (``member``, ``recipe=None``
    and the ensemble's ``saturated_bp_share``, as ``ensemble_decide`` decides
    each member), or the ensemble's (``np5_rederive_ensemble``).

    Args:
        cells: One replicate's cells table.
        levels: The bundle's level metadata.
        depths: The bundle's depth grid.
        settings: The bundle's rule settings.
        recipe: The recipe of the rows (``None``: the rows must hold one).
        member: The version-7 member of the rows.
        saturated_bp_share: The saturated-bp rule (v7.8; ``None``: version 6).

    Returns:
        ``decide`` output.

    Raises:
        ValueError: If no row is left after the filters, the rows hold more
            than one recipe or mapping seed, or they hold check-half rows and
            no fit-half rows (``held_out_replicates`` output).
        ResolvabilityError: If a (level, cell, depth) occurs more than once
            (``replicate_rows``).
    """
    frame = res.replicate_rows(cells, recipe=recipe, seed=None, member=member)
    name = f"np5_rederive (recipe={recipe!r}, member={member!r})"
    seed = _one_seed(frame, name)
    _require_fit_half(frame, name)
    recipes = sorted({str(value) for value in frame["recipe"]})
    if len(recipes) != 1:
        raise ValueError(
            f"np5_rederive: the rows hold the recipes {recipes}; pass recipe= "
            "or member= to select one replicate"
        )
    return res.decide(
        frame,
        levels,
        depths,
        settings,
        recipe=recipes[0],
        seed=seed,
        saturated_bp_share=saturated_bp_share,
    )


def np5_rederive_ensemble(
    cells: pd.DataFrame,
    levels: Sequence[res.LevelMeta],
    depths: Sequence[int],
    settings: res.RuleSettings,
    ensemble: res.EnsembleSettings,
    *,
    members: Sequence[str],
    neuronal: res.NeuronalOf | None = None,
) -> pd.DataFrame:
    """Re-derive one replicate's version-7 ensemble decisions (§14 NP5, v7.7-v7.9).

    The ensemble rule of ``ensemble_decide`` on the replicate's rows of the
    emission members. ``ensemble_decide`` decides on mapping seed 0, so a
    replicate at another mapping seed is relabelled to 0 first (its rows
    are unchanged otherwise).

    Args:
        cells: One replicate's version-7 cells table (``member`` column).
        levels: The bundle's level metadata.
        depths: The bundle's depth grid.
        settings: The bundle's rule settings.
        ensemble: The bundle's ensemble settings.
        members: The emission members.
        neuronal: Per class, whether it is neuronal (the monotone fill's
            non-neuronal limit, v7.9).

    Returns:
        The ensemble decisions (``EnsembleDecisions.decisions``).

    Raises:
        ValueError: Without members, when a member has no rows or check-half
            rows and no fit-half rows (``held_out_replicates`` output), or
            when the rows hold more than one mapping seed.
        ResolvabilityError: If a member holds a (level, cell, depth) more
            than once (``replicate_rows``).
    """
    if not members:
        raise ValueError("np5_rederive_ensemble needs at least one emission member")
    if res.MEMBER_COLUMN not in cells.columns:
        raise ValueError("np5_rederive_ensemble: the cells have no member column")
    names = cells[res.MEMBER_COLUMN].astype(str)
    frame = cells[names.isin([str(name) for name in members]).to_numpy()]
    for name in members:
        rows = res.replicate_rows(frame, recipe=None, seed=None, member=str(name))
        if rows.empty:
            raise ValueError(f"np5_rederive_ensemble: member {name!r} has no rows")
        _require_fit_half(rows, f"np5_rederive_ensemble (member={name!r})")
    _one_seed(frame, "np5_rederive_ensemble")
    relabelled = frame.copy()
    relabelled["seed"] = np.zeros(len(frame), dtype=frame["seed"].to_numpy().dtype)
    return res.ensemble_decide(
        relabelled,
        levels,
        depths,
        settings,
        ensemble,
        members=list(members),
        neuronal=neuronal,
    ).decisions


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise ``ValueError`` naming the columns ``frame`` lacks."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _regime_rows(decisions: pd.DataFrame, regime: str, name: str) -> pd.DataFrame:
    """One regime's rows of a decisions table, one per (level, class, depth).

    Raises:
        ValueError: If the regime has no rows, or a (level, class, depth)
            occurs more than once (several members or replicates in one
            table).
    """
    _require_columns(decisions, ("regime", *_DECISION_KEY, "status"), name)
    frame = decisions[(decisions["regime"].astype(str) == regime).to_numpy()]
    if frame.empty:
        raise ValueError(f"{name}: no decisions of the regime {regime!r}")
    key = pd.DataFrame(
        {
            "level": frame["level"].astype(str).to_numpy(),
            "class": frame["class"].astype(str).to_numpy(),
            "depth": frame["depth"].to_numpy(np.int64),
        }
    )
    duplicated = key.duplicated()
    if bool(duplicated.any()):
        raise ValueError(
            f"{name}: {int(duplicated.sum())} (level, class, depth) decisions occur "
            "more than once; pass one decisions table per replicate (version 7: "
            "the ensemble's, or one member's)"
        )
    return frame


def _emission_view(
    decisions: pd.DataFrame, regime: str, name: str
) -> tuple[dict[tuple[str, str], dict[int, tuple[bool, int]]], set[int]]:
    """Per (level, class) and depth: (emitted, n_test); and the depths seen."""
    _require_columns(decisions, ("n_test",), name)
    frame = _regime_rows(decisions, regime, name)
    view: dict[tuple[str, str], dict[int, tuple[bool, int]]] = {}
    for level, cls, depth, status, n_test in zip(
        frame["level"].astype(str),
        frame["class"].astype(str),
        frame["depth"].to_numpy(np.int64),
        frame["status"].astype(object),
        frame["n_test"].astype(object),
        strict=True,
    ):
        count = 0 if n_test is None or pd.isna(n_test) else int(n_test)
        view.setdefault((level, cls), {})[int(depth)] = (
            status == res.STATUS_EMITTED,
            count,
        )
    return view, {int(depth) for depth in frame["depth"]}


def _boundary_depths(emitted: Sequence[bool], grid: Sequence[int]) -> set[int]:
    """The bins flanking a change of emission status between grid neighbours."""
    flanking: set[int] = set()
    for index in range(1, len(grid)):
        if emitted[index] != emitted[index - 1]:
            flanking.update((grid[index - 1], grid[index]))
    return flanking


def _joined(depths: Iterable[int]) -> str:
    """Depths in ascending order, ``;`` joined."""
    return ";".join(str(depth) for depth in sorted(depths))


def np5_decision_agreement(
    base: pd.DataFrame,
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    settings: Np5Settings,
    *,
    regime: res.Regime = "provisional",
) -> pd.DataFrame:
    """Compare each replicate's re-derived emission with the base run (§14 NP5).

    §14 NP5: per (level, class), the §8.3 emission decisions re-derived in
    each replicate (``np5_rederive``) agree with the base run at every depth
    bin with >= 50 test cells, except at most the bin adjacent to the
    emission boundary. Read here as:

    - a bin is compared when the base run or the replicate has at least
      ``min_test_cells`` test cells of the class there (``n_test``; the
      union of the two readings "the base's" and "the replicate's" bins);
    - a decision is the bin's status at ``regime`` (emitted or not); a
      (level, class) or bin that a table lacks is not emitted there;
    - the emission boundaries are the base run's status changes between
      neighbouring bins of the grid; the bins adjacent to one are the two
      bins flanking it. The grid's edges are no boundary, so a base that
      emits every bin (or none) allows no flip;
    - a replicate passes when no compared bin flips, or exactly one does and
      it is adjacent to a boundary.

    Args:
        base: The base run's decisions (``decide``; version 7: the frozen
            ensemble's, or one member's).
        replicates: Per (group, seed label), that replicate's re-derived
            decisions of the same kind as ``base``.
        settings: The NP5 constants.
        regime: The regime compared (gate P freezes the provisional one).

    Returns:
        One row per (level, class) of either table and replicate, sorted by
        level, class, group and seed, columns ``NP5_AGREEMENT_COLUMNS``:
        ``boundary_depths`` and ``flipped_depths`` are ``;`` joined.

    Raises:
        ValueError: Without replicates, for a table without ``n_test`` or
            rows of the regime, or with a (level, class, depth) more than
            once.
    """
    if not replicates:
        raise ValueError("np5_decision_agreement: no replicates")
    base_view, base_depths = _emission_view(base, regime, "the base decisions")
    minimum = settings.min_test_cells
    records: list[dict[str, object]] = []
    for group, seed in sorted(replicates):
        name = f"replicate {group}/{seed}"
        view, depths = _emission_view(replicates[(group, seed)], regime, name)
        grid = sorted(base_depths | depths)
        for key in sorted(set(base_view) | set(view)):
            mine = base_view.get(key, {})
            theirs = view.get(key, {})
            base_emitted = [mine.get(depth, (False, 0))[0] for depth in grid]
            boundary = _boundary_depths(base_emitted, grid)
            compared = [
                depth
                for depth in grid
                if max(mine.get(depth, (False, 0))[1], theirs.get(depth, (False, 0))[1])
                >= minimum
            ]
            flipped = [
                depth
                for depth in compared
                if mine.get(depth, (False, 0))[0] != theirs.get(depth, (False, 0))[0]
            ]
            records.append(
                {
                    "level": key[0],
                    "class": key[1],
                    "group": str(group),
                    "seed": seed,
                    "n_bins": len(grid),
                    "n_compared": len(compared),
                    "boundary_depths": _joined(boundary),
                    "flipped_depths": _joined(flipped),
                    "n_flipped": len(flipped),
                    "passed": not flipped
                    or (len(flipped) == 1 and flipped[0] in boundary),
                }
            )
    table = pd.DataFrame.from_records(records, columns=list(NP5_AGREEMENT_COLUMNS))
    return table.sort_values(
        ["level", "class", "group", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def _set_threshold(
    bp: np.ndarray,
    correct: np.ndarray,
    half: np.ndarray,
    meta: res.LevelMeta,
    target: float,
    settings: res.RuleSettings,
    saturated_bp_share: float | None,
) -> tuple[int, bool, float | None, float | None, str | None]:
    """Fit one tested set's t* as ``decide`` fits a bin or a pooled set.

    Returns:
        ``(n_fit, fitted, t_star, threshold, threshold_source)``.
    """
    fit_mask = (half == 0) if settings.split_halves else np.ones(len(bp), dtype=bool)
    fit_mask = fit_mask & np.isfinite(bp)
    n_fit = int(fit_mask.sum())
    fit = (
        res.isotonic_fit(bp[fit_mask], correct[fit_mask])
        if n_fit >= settings.min_cells_per_bin
        else None
    )
    t_star = res.local_threshold(
        fit,
        default=meta.default_threshold,
        target=target,
        cap=settings.threshold_cap,
    )
    if t_star is not None:
        return n_fit, True, t_star, t_star, res.THRESHOLD_SOURCE_LOCAL
    if (
        fit is not None
        and saturated_bp_share is not None
        and res.saturated_bp_fraction(bp[fit_mask]) > saturated_bp_share
    ):
        cap = float(settings.threshold_cap)
        return n_fit, True, None, cap, res.THRESHOLD_SOURCE_SATURATED
    return n_fit, fit is not None, None, None, None


def np5_set_thresholds(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    levels: Sequence[res.LevelMeta],
    settings: res.RuleSettings,
    *,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    member: str | None = None,
    saturated_bp_share: float | None = None,
) -> pd.DataFrame:
    """Re-derive t* at every tested set in each replicate (§14 NP5).

    A tested set's t* is fitted on the replicate's calls of the class in the
    set's scope (``tested_set_mask``'s membership rule, so NP3-NP7 cannot
    drift apart: a bin's calls, or each test cell's deepest call at >= D_P),
    as ``decide`` fits a bin or a pooled ">= d" set: the isotonic fit on the
    fit half (``half == 0``; every call without split halves) with at least
    ``min_cells_per_bin`` calls, then ``local_threshold`` at the regime's
    target of the set's shallowest bin. The default group's table is used
    in full: its t* is re-derived on its own fit half, never scored at the
    frozen thresholds, so a table of check-half rows alone (the output of
    ``held_out_replicates``, which ``replicate_set_stats`` and
    ``pooled_held_out_cells`` apply themselves) raises rather than leaving
    the group out as ``unfitted``. With ``saturated_bp_share`` (version 7, v7.8) a
    fitted set without t* whose fit-half calls are saturated takes the cap
    (``threshold_source = saturated_cap``).

    Args:
        replicates: Per (group, seed label), that replicate's cells table.
            Each table must hold exactly one replicate.
        tested: The tested sets per (level, class) (``gate_p_tested_sets``
            on the pooled held-out calls; version 7: the member's).
        levels: The bundle's level metadata (default thresholds, targets).
        settings: The bundle's rule settings.
        regime: A fitted regime (``provisional`` in gate P, or ``trust``).
        recipe: The recipe of the rows (``None``: every recipe, so each table
            must hold one).
        member: The version-7 emission member of the rows.
        saturated_bp_share: The saturated-bp rule (``None``: version 6).

    Returns:
        One row per (tested set, replicate), sorted by level, class, set,
        group and seed, columns ``NP5_THRESHOLD_COLUMNS``: ``fitted`` is
        whether the fit exists; ``t_star`` and ``threshold`` (the applied
        one: t* or the saturated cap) are ``nan`` without one.

    Raises:
        ValueError: For the ``validated`` regime (it applies the default,
            which has no t*), without replicates, for a level without
            metadata, a replicate without rows after the filters or with
            check-half rows and no fit-half rows, or a key's tested sets
            that are an empty list or of another key.
        ResolvabilityError: If a table holds more than one replicate
            (``replicate_rows``).
    """
    if regime == "validated":
        raise ValueError(
            "np5_set_thresholds: the validated regime applies the default "
            "threshold, which has no t* to vary"
        )
    if not replicates:
        raise ValueError("np5_set_thresholds: no replicates")
    _check_tested(tested)
    meta_of = {meta.level: meta for meta in levels}
    unknown = sorted({level for level, _ in tested} - set(meta_of))
    if unknown:
        raise ValueError(f"np5_set_thresholds: no level metadata for {unknown}")
    ordered = sorted(tested.items(), key=lambda pair: pair[0])
    records: list[dict[str, object]] = []
    for group, seed in sorted(replicates):
        frame = res.replicate_rows(
            replicates[(group, seed)], recipe=recipe, seed=None, member=member
        ).reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"replicate {group}/{seed} has no rows after the filters "
                f"(recipe={recipe!r}, member={member!r})"
            )
        _require_fit_half(frame, f"replicate {group}/{seed}")
        index = _ReplicateIndex(frame, np.zeros(len(frame), dtype=bool))
        bp = frame["bp"].to_numpy(np.float64)
        correct = frame["correct"].to_numpy(bool).astype(np.float64)
        half = frame["half"].to_numpy(np.int64)
        for (level, cls), items in ordered:
            meta = meta_of[level]
            for item in items or ():
                called = index.called_positions(item)
                set_min_depth = int(min(item.depths))
                target = settings.target(regime, meta.base_target, set_min_depth)
                n_fit, fitted, t_star, threshold, source = _set_threshold(
                    bp[called],
                    correct[called],
                    half[called],
                    meta,
                    target,
                    settings,
                    saturated_bp_share,
                )
                records.append(
                    {
                        "level": level,
                        "class": cls,
                        "set": tested_set_label(item),
                        "pooled": bool(item.pooled),
                        "set_min_depth": set_min_depth,
                        "group": str(group),
                        "seed": seed,
                        "n_called": int(len(called)),
                        "n_fit": n_fit,
                        "fitted": fitted,
                        "target": target,
                        "t_star": math.nan if t_star is None else t_star,
                        "threshold": math.nan if threshold is None else threshold,
                        "threshold_source": source,
                    }
                )
    table = pd.DataFrame.from_records(records, columns=list(NP5_THRESHOLD_COLUMNS))
    return table.sort_values(
        ["level", "class", "set", "group", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def _replicate_labels(rows: pd.DataFrame, mask: np.ndarray) -> str:
    """``group/seed`` of the masked rows, ``;`` joined."""
    return ";".join(
        f"{group}/{seed}"
        for group, seed in zip(
            rows["group"][mask].astype(str), rows["seed"][mask], strict=True
        )
    )


def np5_tstar_spread(thresholds: pd.DataFrame, settings: Np5Settings) -> pd.DataFrame:
    """Judge the range of t* across the replicates at every tested set (§14 NP5).

    §14 NP5: each t* varies <= ``max_tstar_spread`` (0.05) across replicates
    at tested sets. The range is taken over the replicates with a threshold
    (t*, or the saturated cap of version 7). A replicate whose fit exists
    but never reaches the target has no threshold, so it would not emit the
    set at all: the set fails (``missing``). A replicate without a fit (too
    few fit-half calls) is left out of the range (``unfitted``). With fewer
    than two thresholds the range is not evaluable and the set passes; this
    is reported (``evaluable``), so that a pass on nothing is visible.

    Args:
        thresholds: ``np5_set_thresholds`` output (or rows of its columns).
        settings: The NP5 constants.

    Returns:
        One row per (level, class, set), columns ``NP5_SPREAD_COLUMNS``.

    Raises:
        ValueError: If a column is missing.
    """
    _require_columns(
        thresholds,
        ("level", "class", "set", "pooled", "group", "seed", "fitted", "threshold"),
        "the t* table",
    )
    if thresholds.empty:
        return pd.DataFrame(columns=list(NP5_SPREAD_COLUMNS))
    records: list[dict[str, object]] = []
    for (level, cls, label), rows in thresholds.groupby(
        ["level", "class", "set"], sort=True
    ):
        values = rows["threshold"].to_numpy(np.float64)
        fitted = rows["fitted"].astype(bool).to_numpy()
        has = np.isfinite(values)
        missing = fitted & ~has
        kept = values[has]
        evaluable = len(kept) >= 2
        spread = float(kept.max() - kept.min()) if evaluable else math.nan
        records.append(
            {
                "level": level,
                "class": cls,
                "set": label,
                "pooled": bool(rows["pooled"].iloc[0]),
                "n_replicates": len(rows),
                "n_fitted": int(fitted.sum()),
                "n_thresholds": int(has.sum()),
                "unfitted": _replicate_labels(rows, ~fitted),
                "missing": _replicate_labels(rows, missing),
                "threshold_min": float(kept.min()) if len(kept) else math.nan,
                "threshold_max": float(kept.max()) if len(kept) else math.nan,
                "spread": spread,
                "max_spread": settings.max_tstar_spread,
                "evaluable": evaluable,
                "passed": not bool(missing.any())
                and (not evaluable or spread <= settings.max_tstar_spread + _TOLERANCE),
            }
        )
    return pd.DataFrame.from_records(records, columns=list(NP5_SPREAD_COLUMNS))


def _depth_values(value: float | Sequence[float], cls: str) -> np.ndarray:
    """A class's expected depth (one value) or its cells' depths, checked.

    Raises:
        ValueError: For no value, or a value that is not finite and >= 0.
    """
    values = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if values.size == 0 or not bool(np.all(np.isfinite(values) & (values >= 0.0))):
        raise ValueError(
            f"np5_extrapolated_share: the depths of class {cls!r} must be one or "
            "more finite counts >= 0"
        )
    return values


def _is_true(value: object) -> bool:
    """A boolean cell of a decisions table (missing: False)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value)


def np5_extrapolated_share(
    decisions: pd.DataFrame,
    expected_depth: Mapping[str, float | Sequence[float]],
    depths: Sequence[int],
    settings: Np5Settings,
    *,
    regime: res.Regime = "provisional",
    default_depth: float | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Share of a class's cells at the expected depth that are extrapolated (§14 NP5).

    §14 NP5: a class is not validated at L if its cells at the family's
    expected depth would be more than 50% ``resolvability_extrapolated``. A
    cell takes the bin of its depth (``depth_bin``) and is
    ``resolvability_extrapolated`` when the frozen decisions mark that
    (level, class, bin) ``extrapolated`` (``cell_emission``: a pooled bin
    deeper than D_P, or a version-7 monotone-filled bin); a cell below the
    grid is not. The share is over the class's depths, and the class passes
    at L when it is at most ``max_extrapolated_share``.

    The registered input (§14 "Version-7 families"; plan §8.3 v7.5;
    pre-registration §23.9 item 6 and §23.10, D8):

    - a family with a per-class profile (its frozen ``sim_inputs`` profile
      asset) passes each class's profile depths, a sequence: the share is
      the class's profile share of extrapolated bins (§8.3 v7.5: the test
      "uses the class's profile shares"), and the profile median is
      reported as ``expected_depth``. Passing the median alone instead
      would loosen the test: a profile of 10% at 20, 20% at 40, 25% at 80,
      25% at 150 and 20% at 300 counts, against decisions that mark 30, 120
      and 250 extrapolated, has a share of 0.65 and fails, while its median
      (80, in the 60 bin) has a share of 0 and passes. A class without
      cells in the profile takes D8's overall median (``default_depth``;
      a reading, see the module docstring);
    - a family without a profile passes the label-free pooled median of its
      sections for every class (``{}`` and ``default_depth``, one value).
      The class's cells at one depth all take its bin, so the share is then
      0 or 1. This is the only use of a single depth.

    The same input gives D9's report: the share of the class's depths above
    the grid's deepest bin, which take that bin (``above_grid_share``; a
    depth at the deepest bin is not above it). With one depth it is 0 or 1
    as well.

    Args:
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's, monotone-filled bins included).
        expected_depth: Per class, its cells' depths in the profile (a
            sequence), or one expected depth (counts).
        depths: The bundle's depth grid.
        settings: The NP5 constants.
        regime: The regime of the frozen decisions.
        default_depth: The depth(s) of every class without an entry (D8:
            the overall median, or the label-free pooled median).

    Returns:
        One row per (level, class) of the decisions at the regime, sorted,
        columns ``NP5_EXTRAPOLATED_COLUMNS``: ``expected_depth`` is the median
        of the class's depths and ``expected_bin`` its bin (``None`` below
        the grid); ``depth_source`` is ``class`` or ``default``.

    Raises:
        ValueError: For an empty grid, a class without depths and no
            default, depths that are not finite counts >= 0, decisions
            without ``extrapolated`` or rows of the regime, a (level, class,
            depth) more than once, or a bin a class's depths fall in that its
            decisions lack.
    """
    grid = sorted({int(depth) for depth in depths})
    if not grid:
        raise ValueError("np5_extrapolated_share: an empty depth grid")
    _require_columns(decisions, ("extrapolated",), "the decisions")
    frame = _regime_rows(decisions, regime, "the decisions")
    marked: dict[tuple[str, str], dict[int, bool]] = {}
    for level, cls, depth, flag in zip(
        frame["level"].astype(str),
        frame["class"].astype(str),
        frame["depth"].to_numpy(np.int64),
        frame["extrapolated"].astype(object),
        strict=True,
    ):
        marked.setdefault((level, cls), {})[int(depth)] = _is_true(flag)
    records: list[dict[str, object]] = []
    for (level, cls), by_depth in sorted(marked.items()):
        if cls in expected_depth:
            source, value = NP5_DEPTH_FROM_CLASS, expected_depth[cls]
        elif default_depth is not None:
            source, value = NP5_DEPTH_FROM_DEFAULT, default_depth
        else:
            raise ValueError(
                f"np5_extrapolated_share: no expected depth for class {cls!r} "
                "and no default_depth"
            )
        values = _depth_values(value, cls)
        bins = res.depth_bin(values, grid)
        lacking = sorted(
            {int(value) for value in bins[np.isfinite(bins)]} - set(by_depth)
        )
        if lacking:
            raise ValueError(
                f"{level}/{cls}: the bins {lacking} are not in the decisions"
            )
        flags = np.array(
            [bool(np.isfinite(value)) and by_depth[int(value)] for value in bins],
            dtype=bool,
        )
        median = float(np.median(values))
        median_bin = res.depth_bin([median], grid)[0]
        share = float(flags.mean())
        records.append(
            {
                "level": level,
                "class": cls,
                "depth_source": source,
                "n_depths": int(values.size),
                "expected_depth": median,
                "expected_bin": int(median_bin) if np.isfinite(median_bin) else None,
                "extrapolated_share": share,
                "above_grid_share": float(np.mean(values > grid[-1])),
                "max_share": settings.max_extrapolated_share,
                "passed": share <= settings.max_extrapolated_share + _TOLERANCE,
            }
        )
    return pd.DataFrame(
        {
            column: pd.Series(
                [record[column] for record in records],
                dtype=object if column == "expected_bin" else None,
            )
            for column in NP5_EXTRAPOLATED_COLUMNS
        }
    )


def _passed_by(
    table: pd.DataFrame, columns: Sequence[str], name: str
) -> dict[tuple[str, ...], list[bool]]:
    """The ``passed`` values of a table per key of ``columns``.

    ``name`` names the criterion and table in the error (``"NP5 agreement"``).

    Raises:
        ValueError: If a ``passed`` value is missing (``None`` or ``nan``).
    """
    result: dict[tuple[str, ...], list[bool]] = {}
    keys = zip(*(table[column].astype(str) for column in columns), strict=True)
    for key, passed in zip(keys, table["passed"].astype(object), strict=True):
        if passed is None or (isinstance(passed, float) and math.isnan(passed)):
            raise ValueError(f"{name} row {key} has no passed value")
        result.setdefault(tuple(key), []).append(bool(passed))
    return result


def np5_class_verdicts(
    agreement: pd.DataFrame,
    spread: pd.DataFrame,
    extrapolated: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
) -> dict[tuple[str, str], bool | None]:
    """Combine NP5's three parts per (level, class) (§14 NP5).

    A (level, class) passes NP5 when every replicate's re-derived emission
    agrees with the base run (``np5_decision_agreement``), the t* range is
    within the limit at each of its tested sets (``np5_tstar_spread``) and
    its cells at the expected depth are at most 50% extrapolated
    (``np5_extrapolated_share``). The result has the per-member shape that
    ``resolvability.every_member_verdict`` combines over the version-7
    emission members.

    Args:
        agreement: ``np5_decision_agreement`` output.
        spread: ``np5_tstar_spread`` output.
        extrapolated: ``np5_extrapolated_share`` output.
        tested: The tested sets per (level, class).

    Returns:
        Per (level, class) of ``tested``: ``None`` when it has no tested set
        (not evaluable), ``False`` when any part fails, else ``True``.

    Raises:
        ValueError: If a (level, class) with tested sets has no agreement or
            extrapolated row, a tested set has no t* spread row, a row has
            no ``passed`` value, or a key's tested sets are an empty list or
            of another key.
    """
    _check_tested(tested)
    agreed = _passed_by(agreement, ("level", "class"), "NP5 agreement")
    spread_ok = _passed_by(spread, ("level", "class", "set"), "NP5 t* spread")
    depth_ok = _passed_by(extrapolated, ("level", "class"), "NP5 extrapolated share")
    result: dict[tuple[str, str], bool | None] = {}
    for key, items in tested.items():
        if items is None:
            result[key] = None
            continue
        level, cls = str(key[0]), str(key[1])
        if (level, cls) not in agreed:
            raise ValueError(f"{key}: no NP5 agreement rows")
        if (level, cls) not in depth_ok:
            raise ValueError(f"{key}: no NP5 extrapolated share row")
        labels = [tested_set_label(item) for item in items]
        missing = [label for label in labels if (level, cls, label) not in spread_ok]
        if missing:
            raise ValueError(
                f"{key}: no NP5 t* spread row for the tested sets {missing}"
            )
        result[key] = (
            all(agreed[(level, cls)])
            and all(depth_ok[(level, cls)])
            and all(all(spread_ok[(level, cls, label)]) for label in labels)
        )
    return result


# --------------------------------------------------------------------------
# NP7: error structure (§14 NP7)

# §14 NP7 (human): "confident calls to sink or region-implausible nodes <= 1%
# of confident calls".
NP7_MAX_EXCLUDED_SHARE: Final = 0.01
# §14 NP7: "no single wrong node receives > 5% of the class's confident calls".
NP7_MAX_WRONG_NODE_SHARE: Final = 0.05
# CHECK K4, pre-registration §23.9 item 5: gate P simulates from the frontal
# WHB reference, so NP7's region part reads this column whatever the
# family's sections are.
NP7_REGION: Final = "frontal_cortex"
# The cells-table level whose ``call`` is the assigned WHB supercluster (a
# node; ``node_level_calls``), and the vocab snapshot level of those nodes.
NP7_NODE_LEVEL: Final = "supercluster"
NP7_VOCAB_LEVEL: Final = res.WHB_SUPC
# Why a node is excluded (a node that is both a sink and region-implausible,
# as WHB Splatter, counts as a sink).
NP7_REASON_SINK: Final = "sink"
NP7_REASON_REGION: Final = "region_implausible"
NP7_REASON_NOT_IN_VOCAB: Final = "not_in_vocab"
NP7_EXCLUDED_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "n_confident",
    "n_excluded_calls",
    "n_excluded_confident",
    "n_sink",
    "n_region_implausible",
    "n_not_in_vocab",
    "excluded_share",
    "max_share",
    "nodes",
    "passed",
)
NP7_WRONG_NODE_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "class",
    "set",
    "pooled",
    "set_min_depth",
    "n_truth_confident",
    "n_truth_excluded",
    "n_truth_wrong",
    "wrong_node",
    "n_wrong_node",
    "wrong_node_share",
    "n_called_confident",
    "called_wrong_node",
    "n_called_wrong_node",
    "called_wrong_node_share",
    "max_share",
    "passed",
)
# The node of a wrong call without a call value (kept apart from any label).
_NO_NODE: Final = "<none>"


@dataclass(frozen=True)
class Np7Settings:
    """The NP7 constants (§14 NP7).

    Attributes:
        max_excluded_share: The largest share of a level's confident calls
            that confident calls to sink or region-implausible nodes may
            reach (human; 0.01).
        max_wrong_node_share: The largest share of a truth class's confident
            calls at a tested set that one wrong node may receive (0.05).
        region: The region of the vocab's plausibility column
            (``region_plausible_<region>``; ``frontal_cortex``, CHECK K4).
    """

    max_excluded_share: float = NP7_MAX_EXCLUDED_SHARE
    max_wrong_node_share: float = NP7_MAX_WRONG_NODE_SHARE
    region: str = NP7_REGION

    def __post_init__(self) -> None:
        """Validate the constants.

        Raises:
            ValueError: If a share is outside [0, 1] or the region is empty.
        """
        for name in ("max_excluded_share", "max_wrong_node_share"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Np7Settings.{name} must lie in [0, 1], got {value!r}"
                )
        if not self.region.strip():
            raise ValueError("Np7Settings.region must not be empty")

    @property
    def region_column(self) -> str:
        """The vocab column of the region (``region_plausible_<region>``)."""
        return f"{REGION_COLUMN_PREFIX}{self.region}"


@dataclass(frozen=True)
class Np7Tables:
    """NP7's two tables for one emission member (``np7_error_structure``).

    Attributes:
        excluded: Per level, the confident calls to sink or region-implausible
            nodes against the level's confident calls (columns
            ``NP7_EXCLUDED_COLUMNS``); ``None`` for mouse, where §14 NP7's
            1% part does not apply.
        wrong_node: Per tested set, the share of the truth class's confident
            calls on its most frequent wrong node (columns
            ``NP7_WRONG_NODE_COLUMNS``).
    """

    excluded: pd.DataFrame | None
    wrong_node: pd.DataFrame


def _label(value: object) -> str | None:
    """A label cell of a table (``None``, ``nan`` and blank: ``None``)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return text if text.strip() else None


def _labels(values: pd.Series) -> np.ndarray:
    """``_label`` of every cell of a column, as an object array.

    The distinct values are cleaned once (``pd.factorize``); a missing value
    takes code -1, which indexes the trailing ``None``.
    """
    codes, uniques = pd.factorize(
        values.astype(object).to_numpy(), use_na_sentinel=True
    )
    cleaned = np.array([*(_label(value) for value in uniques), None], dtype=object)
    return np.asarray(cleaned[codes], dtype=object)


def _vocab_flag(value: object, column: str, node: str) -> bool | None:
    """A boolean vocab cell (blank: ``None``, a node the vocab does not know).

    Raises:
        ValueError: For a value other than true, false or blank.
    """
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = _label(value)
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    raise ValueError(
        f"NP7: the vocab's {column!r} of node {node!r} is {text!r}, not true or false"
    )


def np7_excluded_nodes(
    vocab: pd.DataFrame,
    *,
    region: str = NP7_REGION,
    vocab_level: str = NP7_VOCAB_LEVEL,
) -> dict[str, str | None]:
    """Return why each node of a vocab level is excluded from the calls (§14 NP7).

    Production never emits a call to a sink or region-implausible node, and
    reads a node outside the vocab as implausible (``consensus.resolve_human``),
    so the self-map's level specs give such calls no class (``parent`` is
    null; ``whb_level_specs``). NP7 counts them from the bundle's vocab
    snapshot: per node, ``sink`` true gives ``sink``; else
    ``region_plausible_<region>`` false gives ``region_implausible``; else a
    blank flag (a node the vocab does not know) gives ``not_in_vocab``;
    else ``None`` (the node is plausible).

    Args:
        vocab: The mapped bundle's vocab snapshot (``MmcBundle.vocab``:
            ``level``, ``node``, ``sink``, ``region_plausible_<region>``).
        region: The region of the plausibility column.
        vocab_level: The vocab level of the nodes (the WHB supercluster).

    Returns:
        Per node label of ``vocab_level``, the reason or ``None``. A label
        missing here is a node outside the vocab (``not_in_vocab``).

    Raises:
        ValueError: If the vocab lacks a column, has no row at
            ``vocab_level``, holds a node twice or a flag that is not true,
            false or blank.
    """
    column = f"{REGION_COLUMN_PREFIX}{region}"
    _require_columns(vocab, ("level", "node", "sink", column), "NP7: the vocab")
    frame = vocab.reset_index(drop=True)
    rows = frame[(frame["level"].astype(str) == vocab_level).to_numpy()]
    if rows.empty:
        raise ValueError(f"NP7: the vocab has no node at the level {vocab_level!r}")
    nodes = rows["node"].astype(str)
    if bool(nodes.duplicated().any()):
        repeated = sorted(set(nodes[nodes.duplicated()]))
        raise ValueError(
            f"NP7: the vocab lists the nodes {repeated} more than once at "
            f"{vocab_level!r}"
        )
    result: dict[str, str | None] = {}
    for node, sink_value, region_value in zip(
        nodes, rows["sink"].astype(object), rows[column].astype(object), strict=True
    ):
        sink = _vocab_flag(sink_value, "sink", node)
        plausible = _vocab_flag(region_value, column, node)
        if sink:
            result[node] = NP7_REASON_SINK
        elif plausible is False:
            result[node] = NP7_REASON_REGION
        elif sink is None or plausible is None:
            result[node] = NP7_REASON_NOT_IN_VOCAB
        else:
            result[node] = None
    return result


def _assigned_nodes(frame: pd.DataFrame, node_level: str) -> np.ndarray:
    """Each row's assigned node: the call of its cell's row at ``node_level``.

    A lineage, broad or NT call names a group of nodes (``group_level_calls``)
    and the WHB cluster call a child of the assigned supercluster, so the
    assigned node of every row is the supercluster call of the same simulated
    cell (one replicate: (cell, depth) is unique per level).

    Raises:
        ValueError: If ``frame`` has no row at ``node_level``, or a row's cell
            has none there.
    """
    levels = frame["level"].astype(str).to_numpy()
    is_node = levels == node_level
    if not bool(is_node.any()):
        raise ValueError(
            f"NP7: the rows hold no {node_level!r} level, whose calls name the "
            "assigned nodes"
        )
    node_rows = frame[is_node]
    index = pd.MultiIndex.from_arrays(
        [
            node_rows["cell_id"].astype(str).to_numpy(),
            node_rows["depth"].to_numpy(np.int64),
        ]
    )
    keys = pd.MultiIndex.from_arrays(
        [frame["cell_id"].astype(str).to_numpy(), frame["depth"].to_numpy(np.int64)]
    )
    positions = index.get_indexer(keys)
    missing = positions < 0
    if bool(missing.any()):
        raise ValueError(
            f"NP7: {int(missing.sum())} rows have no {node_level!r} row of the same "
            "simulated cell (cell_id, depth), so their assigned node is unknown"
        )
    return np.asarray(_labels(node_rows["call"])[positions], dtype=object)


def _lowest_emitted_thresholds(
    lookup: Mapping[tuple[str, str, int], tuple[str, float | None, bool]],
) -> dict[tuple[str, int], float]:
    """The lowest frozen threshold of the bins emitted per (level, depth)."""
    lowest: dict[tuple[str, int], float] = {}
    for (level, _, depth), (status, threshold, _) in lookup.items():
        if status != res.STATUS_EMITTED or threshold is None:
            continue
        key = (str(level), int(depth))
        lowest[key] = min(lowest.get(key, math.inf), float(threshold))
    return lowest


@dataclass(frozen=True)
class _Np7Rows:
    """One member's pooled held-out rows with NP7's per-row arrays."""

    frame: pd.DataFrame
    index: _ReplicateIndex
    confident: np.ndarray
    correct: np.ndarray
    excluded: np.ndarray
    excluded_confident: np.ndarray
    reason: np.ndarray
    node: np.ndarray


def _np7_rows(
    frame: pd.DataFrame,
    lookup: Mapping[tuple[str, str, int], tuple[str, float | None, bool]],
    reasons: Mapping[str, str | None] | None,
    node_level: str,
) -> _Np7Rows:
    """NP7's arrays of one member's rows (``reasons`` None: no excluded nodes).

    Raises:
        ValueError: If a call to an excluded node has a class (``parent``),
            or for the assigned nodes (``_assigned_nodes``).
    """
    n_rows = len(frame)
    confident = res.frozen_confident_mask(frame, lookup)
    node = _labels(frame["call"])
    node[pd.isna(node)] = _NO_NODE
    reason: np.ndarray = np.full(n_rows, None, dtype=object)
    excluded: np.ndarray = np.zeros(n_rows, dtype=bool)
    excluded_confident = np.zeros(n_rows, dtype=bool)
    if reasons is not None:
        codes, uniques = pd.factorize(
            _assigned_nodes(frame, node_level), use_na_sentinel=True
        )
        why = np.array(
            [*(reasons.get(str(value), NP7_REASON_NOT_IN_VOCAB) for value in uniques)]
            + [None],
            dtype=object,
        )
        reason = np.asarray(why[codes], dtype=object)
        excluded = np.asarray(pd.notna(reason), dtype=bool)
        with_class = excluded & frame["parent"].notna().to_numpy()
        if bool(with_class.any()):
            raise ValueError(
                f"NP7: {int(with_class.sum())} calls to sink or region-implausible "
                "nodes have a class (parent): the cells table was built with "
                "another region or vocab than NP7 reads"
            )
        hit = np.flatnonzero(excluded)
        node[hit] = np.asarray(uniques, dtype=object)[codes[hit]]
        lowest = _lowest_emitted_thresholds(lookup)
        thresholds = np.array(
            [
                lowest.get((str(level), int(depth)), math.nan)
                for level, depth in zip(
                    frame["level"].to_numpy()[hit],
                    frame["depth"].to_numpy()[hit],
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        bp = np.nan_to_num(frame["bp"].to_numpy(np.float64)[hit], nan=-1.0)
        excluded_confident[hit] = np.isfinite(thresholds) & (
            bp >= thresholds - _TOLERANCE
        )
    return _Np7Rows(
        frame=frame,
        index=_ReplicateIndex(frame, confident),
        confident=confident,
        correct=frame["correct"].to_numpy(bool),
        excluded=excluded,
        excluded_confident=excluded_confident,
        reason=reason,
        node=node,
    )


def _node_counts(nodes: np.ndarray) -> list[tuple[str, int]]:
    """Counts per node, the largest first (ties by label)."""
    if len(nodes) == 0:
        return []
    values, counts = np.unique(nodes.astype(str), return_counts=True)
    return sorted(
        ((str(value), int(count)) for value, count in zip(values, counts, strict=True)),
        key=lambda pair: (-pair[1], pair[0]),
    )


def _np7_excluded_table(rows: _Np7Rows, settings: Np7Settings) -> pd.DataFrame:
    """The 1% part per level (§14 NP7, human; K9.1 denominator)."""
    levels = rows.frame["level"].astype(str).to_numpy()
    records: list[dict[str, object]] = []
    for level in sorted(set(levels)):
        at_level = levels == level
        n_confident = int((rows.confident & at_level).sum())
        hit = rows.excluded_confident & at_level
        n_excluded = int(hit.sum())
        reasons = rows.reason[hit]
        records.append(
            {
                "level": level,
                "n_confident": n_confident,
                "n_excluded_calls": int((rows.excluded & at_level).sum()),
                "n_excluded_confident": n_excluded,
                "n_sink": int((reasons == NP7_REASON_SINK).sum()),
                "n_region_implausible": int((reasons == NP7_REASON_REGION).sum()),
                "n_not_in_vocab": int((reasons == NP7_REASON_NOT_IN_VOCAB).sum()),
                "excluded_share": n_excluded / n_confident if n_confident else math.nan,
                "max_share": settings.max_excluded_share,
                "nodes": ";".join(
                    f"{node}:{count}" for node, count in _node_counts(rows.node[hit])
                ),
                "passed": n_excluded
                <= settings.max_excluded_share * n_confident + _TOLERANCE,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(NP7_EXCLUDED_COLUMNS))


def _np7_wrong_node_table(
    rows: _Np7Rows,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    settings: Np7Settings,
) -> pd.DataFrame:
    """The 5% part per tested set (§14 NP7; D12 truth view, called view reported)."""
    truth = _labels(rows.frame["truth_parent"])
    counted = rows.confident | rows.excluded_confident
    wrong = rows.excluded_confident | (rows.confident & ~rows.correct)
    limit = settings.max_wrong_node_share
    records: list[dict[str, object]] = []
    for (level, cls), items in sorted(tested.items(), key=lambda pair: pair[0]):
        for item in items or ():
            scope = rows.index.scope_positions(item)
            mine = scope[(truth[scope] == cls) & counted[scope]]
            mine_wrong = mine[wrong[mine]]
            top = _node_counts(rows.node[mine_wrong])
            node, n_node = top[0] if top else (None, 0)
            called = rows.index.positions(item)
            called_wrong = called[~rows.correct[called]]
            called_top = _node_counts(rows.node[called_wrong])
            called_node, n_called_node = called_top[0] if called_top else (None, 0)
            n_mine = int(len(mine))
            records.append(
                {
                    "level": level,
                    "class": cls,
                    "set": tested_set_label(item),
                    "pooled": bool(item.pooled),
                    "set_min_depth": int(min(item.depths)),
                    "n_truth_confident": n_mine,
                    "n_truth_excluded": int(rows.excluded_confident[mine].sum()),
                    "n_truth_wrong": int(len(mine_wrong)),
                    "wrong_node": node,
                    "n_wrong_node": n_node,
                    "wrong_node_share": n_node / n_mine if n_mine else math.nan,
                    "n_called_confident": int(len(called)),
                    "called_wrong_node": called_node,
                    "n_called_wrong_node": n_called_node,
                    "called_wrong_node_share": n_called_node / len(called)
                    if len(called)
                    else math.nan,
                    "max_share": limit,
                    "passed": n_mine > 0 and n_node <= limit * n_mine + _TOLERANCE,
                }
            )
    return pd.DataFrame.from_records(records, columns=list(NP7_WRONG_NODE_COLUMNS))


def np7_error_structure(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    decisions: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    *,
    default_group: str | None,
    species: Species,
    settings: Np7Settings,
    vocab: pd.DataFrame | None = None,
    node_level: str = NP7_NODE_LEVEL,
    vocab_level: str = NP7_VOCAB_LEVEL,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    seed: int = 0,
    member: str | None = None,
) -> Np7Tables:
    """Score NP7's error structure on the pooled held-out calls (§14 NP7).

    NP7 is scored at the frozen thresholds on all held-out calls at seed 0
    (``pooled_held_out_cells``; the default group on its check half), per
    emission member for version 7. Two parts:

    - **Calls to excluded nodes** (human only): per level, the confident calls
      to sink or region-implausible nodes are at most 1% of the level's
      confident calls (``excluded``). The denominator is the level's
      confident calls at the frozen thresholds over all classes and emitted
      bins (CHECK K9.1; ``frozen_confident_mask``). A call to such a node
      has no class (``parent`` null), so it lies outside every tested set
      and is counted here beside the denominator, never in it. Its node is
      the assigned supercluster of the simulated cell (``node_level``; a
      coarse level's call names a group), read against the vocab snapshot
      (``np7_excluded_nodes``; ``Np7Settings.region``, frontal cortex for
      gate P, CHECK K4). A failure here fails NP7 for every class of the
      level.
    - **Single wrong node** (both species): at every tested set of a
      (level, class), the share of truth class c's confident calls that
      land on one wrong node is at most 5% (``wrong_node``; D12, pre-
      registration §23.9 item 5). These are the confident calls of the
      set's scope (a bin's rows, or each test cell's deepest row at >= D_P;
      ``tested_set_mask``'s scope) whose truth class is c, whatever class
      they were called into. A call's node is its call at the level; a call
      to an excluded node is always wrong and its node is that node (at
      broad a call to a region-implausible neuron node names "Neurons" and
      is "correct" in the cells table, yet production never emits it). The
      called-class view (the share of the set's own confident calls that
      are wrong and name one node) is reported only.

    Readings this implementation takes where §14 is not explicit (strict
    where there is a choice; for the user with the set a dry run):

    - a call to an excluded node has no frozen threshold of its own, so it
      counts as confident when its bp reaches the lowest frozen threshold
      emitted at its level and depth (any class; the "emitted bins" of
      K9.1), and never at a depth where nothing is emitted or with a NaN
      bp. Alternatives: the threshold of the cell's truth class at that
      bin (never stricter), or the level's raw default (.73 / .69).
      ``n_excluded_calls`` reports every call to an excluded node, whatever
      its bp (the H2 count of production);
    - those confident calls also enter the single-wrong-node truth view, so
      a sink that absorbs more than 5% of a class fails the class even when
      the level stays below 1% (§12 M13: "a planted sink absorbing > 5% of
      a class fails NP7");
    - a node outside the vocab, or with a blank flag, counts as
      implausible, as production reads it (``not_in_vocab``);
    - a tested set whose truth class has no confident call in its scope
      fails (a ``nan`` share never passes).

    Args:
        replicates: Per (group, seed label), that replicate's cells table
            (the default group's in full; ``held_out_replicates``), every
            level of a simulated cell in the same table.
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's).
        tested: The tested sets per (level, class) (``gate_p_tested_sets``
            on the same pooled calls; version 7 ``gate_p_member_sets``);
            ``None`` marks a (level, class) that is not evaluable.
        default_group: The group whose fit half the frozen thresholds were
            fitted on (required; ``None`` when no replicate holds those
            cells).
        species: ``human`` scores both parts and needs ``vocab``; ``mouse``
            scores the single-wrong-node part only (§14 NP7: the 1% part is
            "Human").
        settings: The NP7 constants.
        vocab: The mapped bundle's vocab snapshot (human).
        node_level: The cells-table level whose call is the assigned node.
        vocab_level: The vocab level of those nodes.
        regime: The regime whose thresholds are frozen.
        recipe: The recipe of the scored rows (``None``: every recipe, so
            the tables must hold one).
        seed: The seed label of the replicates pooled.
        member: The version-7 emission member of the scored rows.

    Returns:
        The two tables (``excluded`` is ``None`` for mouse), sorted by level
        (and class and set).

    Raises:
        ValueError: If a human run has no vocab or a mouse run has one, no
            row is left after the filters, a key's tested sets are an empty
            list or of another key, a call to an excluded node has a class,
            for the vocab (``np7_excluded_nodes``), the assigned nodes or
            the default group's inputs (``held_out_replicates``).
        ResolvabilityError: If the rows mix replicates (``replicate_rows``).
    """
    if species == "human" and vocab is None:
        raise ValueError(
            "NP7 for a human family needs the vocab snapshot (sink and region "
            "plausibility of the nodes)"
        )
    if species != "human" and vocab is not None:
        raise ValueError(
            "NP7's calls to sink or region-implausible nodes are human only "
            "(§14 NP7); pass no vocab for mouse"
        )
    _check_tested(tested)
    reasons = (
        None
        if vocab is None
        else np7_excluded_nodes(vocab, region=settings.region, vocab_level=vocab_level)
    )
    cells = pooled_held_out_cells(replicates, default_group=default_group, seed=seed)
    frame = res.replicate_rows(
        cells, recipe=recipe, seed=seed, member=member
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"np7_error_structure: no rows after the filters (recipe={recipe!r}, "
            f"seed={seed!r}, member={member!r})"
        )
    rows = _np7_rows(frame, res.emission_lookup(decisions, regime), reasons, node_level)
    return Np7Tables(
        excluded=None if reasons is None else _np7_excluded_table(rows, settings),
        wrong_node=_np7_wrong_node_table(rows, tested, settings),
    )


def np7_class_verdicts(
    excluded: pd.DataFrame | None,
    wrong_node: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
) -> dict[tuple[str, str], bool | None]:
    """Combine NP7's two parts per (level, class) (§14 NP7).

    A (level, class) passes NP7 when its level's confident calls to sink or
    region-implausible nodes are within 1% (human) and no single wrong node
    takes more than 5% of its truth class's confident calls at any of its
    tested sets. The result has the per-member shape that
    ``resolvability.every_member_verdict`` combines over the version-7
    emission members.

    Args:
        excluded: ``Np7Tables.excluded`` (``None`` for mouse).
        wrong_node: ``Np7Tables.wrong_node``.
        tested: The tested sets per (level, class).

    Returns:
        Per (level, class) of ``tested``: ``None`` when it has no tested set
        (not evaluable), ``False`` when any part fails, else ``True``.

    Raises:
        ValueError: If a (level, class) with tested sets has no excluded-share
            row for its level (human), a tested set has no wrong-node row, a
            row has no ``passed`` value, or a key's tested sets are an empty
            list or of another key.
    """
    _check_tested(tested)
    level_ok = (
        None
        if excluded is None
        else _passed_by(excluded, ("level",), "NP7 excluded share")
    )
    set_ok = _passed_by(wrong_node, ("level", "class", "set"), "NP7 wrong node")
    result: dict[tuple[str, str], bool | None] = {}
    for key, items in tested.items():
        if items is None:
            result[key] = None
            continue
        level, cls = str(key[0]), str(key[1])
        if level_ok is not None and (level,) not in level_ok:
            raise ValueError(f"{key}: no NP7 excluded-share row for its level")
        labels = [tested_set_label(item) for item in items]
        missing = [label for label in labels if (level, cls, label) not in set_ok]
        if missing:
            raise ValueError(
                f"{key}: no NP7 wrong-node row for the tested sets {missing}"
            )
        result[key] = (level_ok is None or all(level_ok[(level,)])) and all(
            all(set_ok[(level, cls, label)]) for label in labels
        )
    return result
