"""Gate-P scoring (M13, plan §14 new panel family).

Pure functions on cells tables: NP3 precision and coverage, NP4 donor / draw
stability.

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
class's tested set; pre-registration §23.9 item 2). The unweighted values,
a family dataset's depth histogram (label-free) and the two weightings read
on the test cells of a set's scope are reported only. A family dataset's
composition is not used: it needs the dataset's labels, and no real-label
input reaches gate P other than NP5's registered profile (§23.9 item 6).

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
      reference's natural composition;
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
    a share of the set. This is not the registered NP3 weighting (§23.9 item
    2); it is reported beside it.

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
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    *,
    composition: Mapping[str, float],
    settings: Np3Settings,
    depth_histogram: Mapping[int, float] | None = None,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    seed: int | None = 0,
    member: str | None = None,
) -> pd.DataFrame:
    """Score every tested set of the pooled held-out calls (§14 NP3).

    NP3 is scored on the base recipe at seed 0, on all held-out calls
    (``pooled_held_out_cells``) at the frozen thresholds of ``decisions``,
    on the tested sets fixed from the same calls (``gate_p_tested_sets``;
    version 7: per emission member, ``gate_p_member_sets``). Each set is
    scored under every scheme of ``np3_set_weights`` (``depth_histogram``
    only with a histogram) and of ``np3_test_cell_weights``; NP3's verdict
    uses ``NP3_SCORED_SCHEMES`` only (``np3_verdicts``).

    A set's precision, Kish n and Wilson bound are on its confident calls,
    weighted as one set; its coverage is the confident share of the calls of
    the class in the set's scope (a bin's calls, or the cells' deepest calls
    at >= D_P), weighted the same way as a set of their own.

    Args:
        cells: The pooled held-out calls (one replicate's worth of rows:
            pooled donors or draws with distinct cell ids).
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's).
        tested: The tested sets per (level, class); ``None`` marks a
            (level, class) that is not evaluable and has no rows.
        composition: The reference's natural share per truth type; every
            truth type of the held-out calls needs a positive share.
        settings: The NP3 constants.
        depth_histogram: A family dataset's cell mass per grid bin
            (label-free; report-only).
        regime: The regime whose thresholds are frozen.
        recipe: The recipe of the scored rows (``None``: every recipe, so
            the table must hold one).
        seed: The mapping seed of the scored rows (``None``: likewise).
        member: The version-7 emission member of the scored rows.

    Returns:
        One row per (tested set, scheme), sorted by level, class and set,
        columns ``NP3_STATS_COLUMNS`` (``scored``: the scheme decides NP3).

    Raises:
        ValueError: If no row is left after the filters, a key's tested sets
            are an empty list or of another key, or for an invalid
            composition or histogram.
        ResolvabilityError: If the rows mix replicates (``replicate_rows``).
    """
    _check_tested(tested)
    histogram = None if depth_histogram is None else _depth_histogram(depth_histogram)
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

    Args:
        table: ``validated_min_depth`` output.

    Returns:
        Per (level, class): ``True`` (passes), ``False`` (fails) or ``None``
        (not evaluable).
    """
    return {
        (str(level), str(cls)): None if passed is None else bool(passed)
        for level, cls, passed in zip(
            table["level"], table["class"], table["passed"], strict=True
        )
    }
