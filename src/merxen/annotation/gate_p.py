"""Gate-P scoring (M13, plan §14 new panel family).

Pure functions on cells tables; NP4 donor / draw stability first.

Gate P validates a new panel family by simulation only (plan §8.8, §14). The
thresholds and emission are derived once from the default donor (human) or
draw (mouse) at seed 0 and then frozen; the held-out replicates are scored
at those frozen thresholds on the pooled tested sets (``gate_p_tested_sets``:
each set holds >= ``gate_p_min_confident_n`` confident calls, deep bins
pooled with each test cell counted once at its deepest bin).

This module holds NP4, stability across held-out donors (or draws) and
seeds (§14 NP4): per (level, class), at every tested set,

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
from typing import Final

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

    def positions(self, item: res.GatePTestedSet) -> np.ndarray:
        """Return the positions of the rows in a tested set (one per test cell).

        Args:
            item: The tested set.

        Returns:
            Row positions; empty when the replicate has no call of the class.
        """
        rows, deepest = self._level_rows(item.level)
        code = self._parent_code.get(item.cls)
        if code is None:
            return np.empty(0, dtype=np.int64)
        if item.pooled:
            keep = (
                (self._parent_codes[deepest] == code)
                & self._confident[deepest]
                & (self._depth[deepest] >= min(item.depths))
            )
            return np.asarray(deepest[keep], dtype=np.int64)
        keep = (
            (self._parent_codes[rows] == code)
            & (self._depth[rows] == item.depths[0])
            & self._confident[rows]
        )
        return np.asarray(rows[keep], dtype=np.int64)


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
