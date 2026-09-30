"""The draw spread of a resolvability self-map (M8 D2; pre-registration §18).

The gate scores the pre-registered simulation draw: per-cell seed 0 (the
thinning, spill and spill-partner keys) and efficiency seed 0 (the per-gene
detection efficiency), mapped with seed 0 (plan §8.3 step 3). The M3b
review-3 factorial showed that some verdicts depend on that draw, so the
user decided on 2026-09-30 (M8 D2) that the seed-0 draw is scored and the
spread over a 3 x 3 grid of per-cell seeds x efficiency seeds (0-2 each) is
reported beside it. Every draw is decided alone with the unchanged rules
(``ResolvabilityTables.decisions``: unweighted for PREP, reweighted to a
dataset's soft composition for RESOLVE); the spread never changes a verdict.

This module holds what the M8 acceptance scripts share:

- ``draw_grid`` and ``Draw``: the grid; ``c0e0`` is the scored draw;
- ``simulate_draw``: one draw's decision-recipe cells table, through the
  production simulation, mapper and cells rules;
- ``tables_with_draw``: a bundle's tables with one draw's cells in place of
  the stored decision-recipe rows;
- ``dataset_composition``: the soft composition RESOLVE reweights to;
- ``h18_human_rows`` and ``validated_bins``: H18's human rule (broad and
  supercluster emitted from one grid step above the class's floor up to
  D_max, for every class with D_max) and the validated bins per decision set;
- ``decision_metric_rows``, ``spread_table`` and ``bin_spread``: the long
  per-draw rows, and the per (dataset, metric) and per-bin spread tables.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation import resolvability as res

GRID_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SCORED_SEEDS: Final[tuple[int, int]] = (0, 0)
# The mapping seed of every draw: the production self-map maps with seed 0.
MAPPING_SEED: Final = 0
# H18's human levels and the depths from which its simulated precision is
# judged (plan §14 H18: "at D >= 15 / 30"; the would-raise count).
H18_LEVELS: Final[tuple[str, ...]] = ("broad", "supercluster")
H18_MIN_DEPTH: Final[dict[str, int]] = {"broad": 15, "supercluster": 30}
VALIDATED: Final = "validated"
# Metric names of the decision rows (``decision_metric_rows``).
METRIC_H18_FAILING: Final = "H18/failing_pairs"
METRIC_H18_WOULD_RAISE: Final = "H18/would_raise"
METRIC_BINS_EMITTED: Final = "validated_bins_emitted"
ROW_COLUMNS: Final[tuple[str, ...]] = (
    "draw",
    "cell_seed",
    "efficiency_seed",
    "scored_draw",
    "dataset",
    "metric",
    "value",
    "threshold",
    "comparator",
    "passes",
    "note",
)
SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "metric",
    "threshold",
    "comparator",
    "scored_value",
    "scored_passes",
    "min",
    "max",
    "n_draws",
    "n_draws_failing",
    "failing_draws",
    "verdict_depends_on_draw",
)
BIN_COLUMNS: Final[tuple[str, ...]] = (
    "draw",
    "dataset",
    "level",
    "class",
    "depth",
    "emitted",
    "precision",
    "wilson_lb",
    "n_confident",
    "reason",
)
BIN_SPREAD_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "level",
    "class",
    "depth",
    "scored_emitted",
    "n_draws_emitting",
    "n_draws",
    "draws_emitting",
)


@dataclass(frozen=True)
class Draw:
    """One simulation draw of the grid.

    Attributes:
        cell_seed: Seed of the per-cell keys (thinning, spill, partner).
        efficiency_seed: Seed of the per-gene efficiency draw.
    """

    cell_seed: int
    efficiency_seed: int

    @property
    def tag(self) -> str:
        """The draw's name, ``c<cell seed>e<efficiency seed>``."""
        return f"c{self.cell_seed}e{self.efficiency_seed}"

    @property
    def is_scored(self) -> bool:
        """Whether this is the pre-registered draw the gate scores."""
        return (self.cell_seed, self.efficiency_seed) == SCORED_SEEDS


def draw_grid(seeds: Sequence[int] = GRID_SEEDS) -> list[Draw]:
    """Return the per-cell seed x efficiency seed grid, the scored draw first.

    Args:
        seeds: Seeds of each factor.

    Returns:
        ``len(seeds) ** 2`` draws.

    Raises:
        ValueError: If the seeds repeat or omit 0 (the scored draw).
    """
    values = [int(seed) for seed in seeds]
    if len(set(values)) != len(values) or 0 not in values:
        raise ValueError(f"the grid seeds must be distinct and include 0: {values}")
    draws = [Draw(cell, efficiency) for cell in values for efficiency in values]
    return sorted(draws, key=lambda draw: (not draw.is_scored, draw.tag))


def simulate_draw(
    test: res.HeldOutCells,
    depths: Sequence[int],
    recipe: res.SimulationRecipe,
    draw: Draw,
    *,
    specs: Sequence[res.LevelSpec],
    map_fn: res.MapFunction,
    cells_rules: Sequence[res.CellsRule] = (),
) -> pd.DataFrame:
    """Simulate, map and tabulate one draw of the decision recipe.

    The production path of ``resolvability.run_resolvability`` for one
    recipe: ``thin_and_contaminate`` with the draw's per-cell seed (the
    recipe's ``seed``) and efficiency seed, the production mapper at mapping
    seed 0, ``level_cells`` and the cells rules (WHB: the COP rule). The
    ``c0e0`` draw reproduces a bundle's stored decision-recipe rows.

    Args:
        test: The self-map's test cells (``reference.self_map_test_cells``).
        depths: The bundle's depth grid.
        recipe: The decision recipe (``simulation_recipes(...)[0]``).
        draw: The draw.
        specs: The self-map's level specs.
        map_fn: ``(query, tag, seed) -> MMC tidy table`` (production mapper).
        cells_rules: Production rules applied to the cells table.

    Returns:
        The draw's cells rows, as a bundle stores them (``as_stored``).
    """
    drawn = dataclasses.replace(recipe, seed=int(draw.cell_seed))
    query = res.thin_and_contaminate(
        test, depths, drawn, efficiency_seed=int(draw.efficiency_seed)
    )
    tidy = map_fn(query, f"{recipe.name}_{draw.tag}", MAPPING_SEED)
    frame = res.level_cells(tidy, query, test, specs, seed=MAPPING_SEED)
    for rule in cells_rules:
        frame = rule(frame)
    return res.as_stored(frame)


def tables_with_draw(
    tables: res.ResolvabilityTables, draw_cells: pd.DataFrame
) -> res.ResolvabilityTables:
    """Return a bundle's tables with one draw's decision-recipe cells.

    The stored rows of the decision recipe (every mapping seed) are replaced
    by the draw's; the other recipes' rows (``clean``) stay. The summary,
    levels and rule settings are the bundle's, so the draw is decided with
    the unchanged rules.

    Args:
        tables: ``resolvability.load_resolvability`` output.
        draw_cells: ``simulate_draw`` output (or its parquet).

    Returns:
        The substituted tables.

    Raises:
        ValueError: If the draw's rows are not of the decision recipe.
    """
    recipe = str(tables.summary["decision_recipe"])
    recipes = set(draw_cells["recipe"].astype(str))
    if recipes != {recipe}:
        raise ValueError(
            f"draw cells hold recipes {sorted(recipes)}, not the decision recipe "
            f"{recipe!r}"
        )
    kept = tables.cells[tables.cells["recipe"].astype(str) != recipe]
    cells = pd.concat(
        [kept, res.restore_labels(res.as_stored(draw_cells.copy()))],
        ignore_index=True,
    )
    return dataclasses.replace(tables, cells=res.restore_labels(cells))


def dataset_composition(
    leaf: pd.DataFrame,
    labels: pd.DataFrame,
    depths: Sequence[int],
    *,
    min_bin_mass: float,
) -> res.DatasetComposition:
    """Return the composition RESOLVE reweights a dataset's tables to.

    As ``pipeline.resolve_human_sample``: the primary leaf level of the
    table cells (``in_table``) with their total counts
    (``composition.dataset_type_composition``).

    Args:
        leaf: ``level_frame`` of the primary run's leaf level (index cell id).
        labels: The sample's label table (``cell_id``, ``in_table``,
            ``total_counts``).
        depths: The bundle's depth grid.
        min_bin_mass: ``resolvability.composition_min_bin_cells``.

    Returns:
        The composition.
    """
    from merxen.annotation.composition import dataset_type_composition

    table = labels[labels["in_table"].to_numpy(bool)]
    frame = leaf.copy()
    frame.index = frame.index.astype(str)
    frame = frame.reindex(table["cell_id"].astype(str).to_numpy())
    return dataset_type_composition(
        frame,
        table["total_counts"].to_numpy(np.float64),
        list(depths),
        min_bin_mass=float(min_bin_mass),
    )


def _floor(floors: pd.DataFrame, level: str, cls: str, platform: str) -> int | None:
    rows = floors[
        (floors["level"].astype(str) == level)
        & (floors["floor_class"].astype(str) == cls)
        & (floors["platform"].astype(str).str.upper() == platform.upper())
    ]
    return int(rows["min_counts"].max()) if len(rows) else None


def h18_human_rows(
    decisions: pd.DataFrame,
    depths: Sequence[int],
    platform: str,
    floors: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Return H18's human emission check per (level, class) of one decision set.

    Plan §14 H18, as M3b and the M8 prep scored it: broad and supercluster
    are emitted (validated regime) for every class with D_max (>= 50 test
    cells in some bin) at every grid depth above the class's packaged floor
    on ``platform`` up to D_max; a class without D_max is not required.

    Args:
        decisions: ``ResolvabilityTables.decisions`` output (unweighted PREP
            or reweighted to a dataset).
        depths: The bundle's depth grid.
        platform: Whose packaged floors apply (``MERSCOPE`` / ``XENIUM``).
        floors: ``vocab.load_floor_table("human")``.
        dataset: Label recorded in the rows (``PREP[MERSCOPE floors]``, a
            sample id).

    Returns:
        One row per (level, class): ``dataset``, ``level``, ``class``,
        ``d_max``, ``floor``, ``expected``, ``emitted``, ``missing``,
        ``required`` and ``passes``.
    """
    grid = [int(depth) for depth in depths]
    rows: list[dict[str, Any]] = []
    for level in H18_LEVELS:
        trust = decisions[
            (decisions["regime"].astype(str) == "trust")
            & (decisions["level"].astype(str) == level)
        ]
        validated = decisions[
            (decisions["regime"].astype(str) == VALIDATED)
            & (decisions["level"].astype(str) == level)
            & (decisions["status"].astype(str) == res.STATUS_EMITTED)
        ]
        emitted_of = {
            str(cls): sorted(int(depth) for depth in group["depth"])
            for cls, group in validated.groupby(validated["class"].astype(str))
        }
        for cls, group in trust.groupby(trust["class"].astype(str), sort=True):
            values = pd.to_numeric(group["d_max"], errors="coerce").dropna()
            d_max = int(values.iloc[0]) if len(values) else None
            got = emitted_of.get(str(cls), [])
            if d_max is None:
                rows.append(
                    {
                        "dataset": dataset,
                        "level": level,
                        "class": str(cls),
                        "d_max": None,
                        "floor": None,
                        "expected": "",
                        "emitted": ",".join(map(str, got)),
                        "missing": "",
                        "required": False,
                        "passes": True,
                    }
                )
                continue
            floor = _floor(floors, level, str(cls), platform)
            start = floor if floor is not None else grid[0]
            expected = [depth for depth in grid if start < depth <= d_max]
            missing = [depth for depth in expected if depth not in got]
            rows.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "class": str(cls),
                    "d_max": d_max,
                    "floor": floor,
                    "expected": ",".join(map(str, expected)),
                    "emitted": ",".join(map(str, got)),
                    "missing": ",".join(map(str, missing)),
                    "required": True,
                    "passes": not missing,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "level",
            "class",
            "d_max",
            "floor",
            "expected",
            "emitted",
            "missing",
            "required",
            "passes",
        ],
    )


def validated_bins(
    decisions: pd.DataFrame,
    *,
    draw: Draw,
    dataset: str,
    levels: Sequence[str] = H18_LEVELS,
) -> pd.DataFrame:
    """Return the validated-regime bins of one decision set (``BIN_COLUMNS``).

    Args:
        decisions: ``ResolvabilityTables.decisions`` output.
        draw: The draw.
        dataset: ``PREP`` or a sample id.
        levels: Levels to keep.

    Returns:
        One row per (level, class, depth).
    """
    frame = decisions[
        (decisions["regime"].astype(str) == VALIDATED)
        & decisions["level"].astype(str).isin(list(levels))
    ]
    return pd.DataFrame(
        {
            "draw": draw.tag,
            "dataset": dataset,
            "level": frame["level"].astype(str).to_numpy(),
            "class": frame["class"].astype(str).to_numpy(),
            "depth": frame["depth"].astype(int).to_numpy(),
            "emitted": (frame["status"].astype(str) == res.STATUS_EMITTED).to_numpy(),
            "precision": pd.to_numeric(frame["precision"], errors="coerce").to_numpy(),
            "wilson_lb": pd.to_numeric(frame["wilson_lb"], errors="coerce").to_numpy(),
            "n_confident": pd.to_numeric(
                frame["n_confident"], errors="coerce"
            ).to_numpy(),
            "reason": frame["reason"].astype(object).to_numpy(),
        },
        columns=list(BIN_COLUMNS),
    )


def metric_row(
    draw: Draw,
    dataset: str,
    metric: str,
    value: float | None,
    *,
    threshold: float | None = None,
    comparator: str = "",
    passes: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Return one long per-draw row (``ROW_COLUMNS``).

    Args:
        draw: The draw.
        dataset: ``PREP[...]`` or a sample id.
        metric: Metric name (``H7``, ``H8/warning_value``, ``H18/...``).
        value: The measured value.
        threshold: The pre-registered threshold, if any.
        comparator: ``>=``, ``<=`` or ``""``.
        passes: Override; default from the comparison when a threshold is
            given.
        note: Free text.

    Returns:
        The row.
    """
    measured = value is not None and not (
        isinstance(value, float) and math.isnan(value)
    )
    if passes is None and measured and threshold is not None and comparator:
        passes = (
            bool(float(value) >= threshold - 1e-12)  # type: ignore[arg-type]
            if comparator == ">="
            else bool(float(value) <= threshold + 1e-12)  # type: ignore[arg-type]
        )
    return {
        "draw": draw.tag,
        "cell_seed": draw.cell_seed,
        "efficiency_seed": draw.efficiency_seed,
        "scored_draw": draw.is_scored,
        "dataset": dataset,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "comparator": comparator,
        "passes": passes,
        "note": note,
    }


def decision_metric_rows(
    decisions: pd.DataFrame,
    *,
    draw: Draw,
    dataset: str,
    platform: str,
    depths: Sequence[int],
    floors: pd.DataFrame,
    settings: res.RuleSettings,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Return one decision set's per-draw rows, H18 rows and validated bins.

    Metrics: the number of failing H18 (level, class) pairs (passes at 0),
    the validated thresholds the local rule would raise at H18's depths
    (``would_raise_bins``, broad from 15 and supercluster from 30 counts;
    listed by the gate, never a pass or fail here) and the validated bins
    emitted at the H18 levels.

    Args:
        decisions: ``ResolvabilityTables.decisions`` output.
        draw: The draw.
        dataset: ``PREP[<platform> floors]`` or a sample id.
        platform: Whose floors H18 uses.
        depths: The depth grid.
        floors: The human packaged floors.
        settings: The bundle's rule settings.

    Returns:
        ``(rows, h18, bins)``.
    """
    h18 = h18_human_rows(decisions, depths, platform, floors, dataset=dataset)
    h18.insert(0, "draw", draw.tag)
    failing = h18[h18["required"].astype(bool) & ~h18["passes"].astype(bool)]
    raised, _ = res.would_raise_bins(decisions, settings)
    raised = raised[
        [
            str(level) in H18_MIN_DEPTH and int(depth) >= H18_MIN_DEPTH[str(level)]
            for level, depth in zip(raised["level"], raised["depth"], strict=True)
        ]
    ]
    bins = validated_bins(decisions, draw=draw, dataset=dataset)
    rows = [
        metric_row(
            draw,
            dataset,
            METRIC_H18_FAILING,
            float(len(failing)),
            threshold=0.0,
            comparator="<=",
            note="; ".join(
                f"{level} {cls} missing {missing}"
                for level, cls, missing in failing[
                    ["level", "class", "missing"]
                ].itertuples(index=False, name=None)
            ),
        ),
        metric_row(
            draw,
            dataset,
            METRIC_H18_WOULD_RAISE,
            float(len(raised)),
            note="validated thresholds the local rule would raise (listed, D4)",
        ),
        metric_row(
            draw,
            dataset,
            METRIC_BINS_EMITTED,
            float(bins["emitted"].sum()),
            note=f"of {len(bins)} validated broad / supercluster bins",
        ),
    ]
    return rows, h18, bins


def spread_table(rows: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return the per (dataset, metric) spread over the draws (``SPREAD_COLUMNS``).

    The scored value is the ``c0e0`` draw's; ``verdict_depends_on_draw`` is
    true when some draw's verdict differs from it (report only: the gate
    scores the seed-0 draw, M8 D2).

    Args:
        rows: Long per-draw rows (``ROW_COLUMNS``).

    Returns:
        One row per (dataset, metric).
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(SPREAD_COLUMNS))
    records: list[dict[str, Any]] = []
    for (dataset, metric), group in frame.groupby(
        ["dataset", "metric"], sort=True, dropna=False
    ):
        scored = group[group["scored_draw"].astype(bool)]
        values = pd.to_numeric(group["value"], errors="coerce")
        passes = group["passes"]
        judged = passes.notna()
        failing = sorted(
            str(tag)
            for tag, ok in zip(group["draw"], passes, strict=True)
            if ok is not None
            and not (isinstance(ok, float) and math.isnan(ok))
            and not bool(ok)
        )
        scored_passes = (
            None
            if scored.empty or pd.isna(scored["passes"].iloc[0])
            else bool(scored["passes"].iloc[0])
        )
        verdicts = {bool(ok) for ok in passes[judged]}
        records.append(
            {
                "dataset": dataset,
                "metric": metric,
                "threshold": group["threshold"].iloc[0],
                "comparator": group["comparator"].iloc[0],
                "scored_value": (
                    None if scored.empty else _as_float(scored["value"].iloc[0])
                ),
                "scored_passes": scored_passes,
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
                "n_draws": int(group["draw"].nunique()),
                "n_draws_failing": len(failing),
                "failing_draws": ",".join(failing),
                "verdict_depends_on_draw": len(verdicts) > 1,
            }
        )
    return pd.DataFrame(records, columns=list(SPREAD_COLUMNS))


def bin_spread(bins: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return, per validated bin, how many draws emit it (``BIN_SPREAD_COLUMNS``).

    Args:
        bins: ``validated_bins`` rows of every draw.

    Returns:
        One row per (dataset, level, class, depth).
    """
    frame = pd.DataFrame(bins)
    if frame.empty:
        return pd.DataFrame(columns=list(BIN_SPREAD_COLUMNS))
    scored_tag = Draw(*SCORED_SEEDS).tag
    records: list[dict[str, Any]] = []
    for (dataset, level, cls, depth), group in frame.groupby(
        ["dataset", "level", "class", "depth"], sort=True
    ):
        emitted = group["emitted"].astype(bool)
        scored = group[group["draw"].astype(str) == scored_tag]
        records.append(
            {
                "dataset": dataset,
                "level": level,
                "class": cls,
                "depth": int(depth),
                "scored_emitted": (
                    None if scored.empty else bool(scored["emitted"].iloc[0])
                ),
                "n_draws_emitting": int(emitted.sum()),
                "n_draws": int(group["draw"].nunique()),
                "draws_emitting": ",".join(
                    sorted(group.loc[emitted, "draw"].astype(str))
                ),
            }
        )
    return pd.DataFrame(records, columns=list(BIN_SPREAD_COLUMNS))


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


__all__ = [
    "BIN_COLUMNS",
    "BIN_SPREAD_COLUMNS",
    "GRID_SEEDS",
    "MAPPING_SEED",
    "METRIC_BINS_EMITTED",
    "METRIC_H18_FAILING",
    "METRIC_H18_WOULD_RAISE",
    "ROW_COLUMNS",
    "SCORED_SEEDS",
    "SPREAD_COLUMNS",
    "Draw",
    "bin_spread",
    "dataset_composition",
    "decision_metric_rows",
    "draw_grid",
    "h18_human_rows",
    "metric_row",
    "simulate_draw",
    "spread_table",
    "tables_with_draw",
    "validated_bins",
]
