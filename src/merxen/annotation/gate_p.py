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

NP4's seed criterion (seed 0 vs 1 changes <= 2% of confident labels) is a
separate check. A replicate is keyed by an opaque (group, seed label) pair:
the group is a human donor or a mouse draw, so the functions are
species-agnostic. Version-7 families score NP4 in every emission member
(``member=``) and combine the members with ``every_member_verdict``.

The averaging conventions of the range rule (p_bar and n_bar as means of
seed-averaged group values, unweighted precision) are the D12 proposal of
the M13 plan, pending the user's confirmation (pre-registration §23.9).
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
    level_mask = (frame["level"].astype(str) == item.level).to_numpy(bool)
    of_class = (frame["parent"].astype(object) == item.cls).to_numpy(bool)
    depth = frame["depth"].to_numpy(np.int64)
    if not item.pooled:
        single = level_mask & of_class & (depth == item.depths[0]) & is_confident
        return np.asarray(single, dtype=bool)
    mask = np.zeros(len(frame), dtype=bool)
    deepest = res.deepest_rows(frame[level_mask])
    if deepest.empty:
        return mask
    positions = deepest.index.to_numpy(np.int64)
    keep = (
        of_class[positions]
        & is_confident[positions]
        & (depth[positions] >= min(item.depths))
    )
    mask[positions[keep]] = True
    return mask


def replicate_set_stats(
    replicates: Mapping[ReplicateKey, pd.DataFrame],
    decisions: pd.DataFrame,
    tested: Mapping[tuple[str, str], Sequence[res.GatePTestedSet] | None],
    *,
    regime: res.Regime = "provisional",
    recipe: str | None = res.DECISION_RECIPE,
    member: str | None = None,
) -> pd.DataFrame:
    """Count each replicate's confident calls in every tested set (§14 NP4).

    The replicates are scored at the frozen thresholds of ``decisions`` on
    the tested sets fixed from the pooled seed-0 calls (``tested``, from
    ``gate_p_tested_sets``). Every (tested set, replicate) pair gets a row,
    a replicate without calls in the set included (``n_confident`` 0), so
    the range rule can see that not every replicate has enough calls.

    Args:
        replicates: Per (group, seed label), that replicate's cells table.
            Each table must hold exactly one replicate.
        decisions: The frozen decisions of the base run (version 7: the
            ensemble's).
        tested: The tested sets per (level, class); ``None`` marks a
            (level, class) that is not evaluable and has no rows.
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
            the filters, or a tested set's level or class differs from its
            key.
        ResolvabilityError: If a table holds more than one replicate
            (``replicate_rows``).
    """
    if not replicates:
        raise ValueError("replicate_set_stats: no replicates")
    for key, items in tested.items():
        for item in items or ():
            if (item.level, item.cls) != tuple(key):
                raise ValueError(
                    f"the tested set {item.level}/{item.cls} "
                    f"{tested_set_label(item)} belongs to another key than {key}"
                )
    lookup = res.emission_lookup(decisions, regime)
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
        confident = res.frozen_confident_mask(frame, lookup)
        correct = frame["correct"].to_numpy(bool)
        for (level, cls), items in sorted(tested.items(), key=lambda pair: pair[0]):
            for item in items or ():
                mask = tested_set_mask(frame, confident, item)
                n_confident = int(mask.sum())
                n_correct = int(correct[mask].sum())
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
    """NP4's range rule on one tested set (the D12 proposal, pending).

    Pending the user's confirmation (M13 plan D12; pre-registration §23.9):
    per group, p_g and n_g are the arithmetic means over its seeds of the
    per-seed (unweighted) precision and confident n; p_bar and n_bar are the
    means of the group values; the limit is
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
    no replicate reaches the minimum is ``vacuous``.

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
        ValueError: If a tested set of a key has no verdict row.
    """
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
