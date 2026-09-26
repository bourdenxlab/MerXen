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
  tiles.
- **Human rules** (§5.2 rules 1, 2 and 4; §5.4): a shadow evaluation of the
  v1 lineage, broad and supercluster rules on raw thresholds with the
  packaged floors, the COP rule, the implausible fallback, a choice of second
  vote (none, SEA-AD below 60 counts only, the full v1 SEA-AD rule, or E2's
  likelihood-typer rule) and the dataset gate. It has no resolvability
  (M3b) and no flags; RESOLVE (M4) replaces it, and M4's exit compares its
  coverage with these baselines.
- **SEA-AD broad calls** (E1 ``e1lib.load_mmc_seaad``): the 7-class label
  from the vocab ("VLMC & Perivascular" split by supertype) and the
  aggregated broad probability.
- **Marker referee** (E1 ``09_marker_referee.py``): for cells where two
  labellings disagree on two of the seven classes, the label whose canonical
  panel markers hold the larger fraction of the cell's counts wins.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.config import AnnotationGate, AnnotationThresholds
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

logger = logging.getLogger(__name__)

UNALLOCATED: Final = "unallocated"
COMPOSITION_COLUMNS: Final[tuple[str, ...]] = (*HUMAN_BROAD_CLASSES, UNALLOCATED)
TILE_UM: Final = 500.0
N_BOOTSTRAP: Final = 200
BOOTSTRAP_SEED: Final = 0
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

SecondVote = Literal["none", "seaad_from60", "seaad", "ll_below60", "seaad_ll_below60"]
# Second-vote variants: (method that must agree below 60 counts, whether
# SEA-AD also applies its other v1 roles: vetoing from 60 counts when it
# confidently disagrees, the COP rescue and the implausible-lineage rescue).
SECOND_VOTE_VARIANTS: Final[dict[str, tuple[Literal["none", "seaad", "ll"], bool]]] = {
    "none": ("none", False),
    "seaad_from60": ("none", True),
    "seaad": ("seaad", True),
    "ll_below60": ("ll", False),
    "seaad_ll_below60": ("ll", True),
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


def broad_class_index(
    names: Sequence[object] | np.ndarray | pd.Series,
    broad_of: Mapping[str, str],
) -> np.ndarray:
    """Return the ``COMPOSITION_COLUMNS`` index of each node's broad class.

    Args:
        names: Node names (``None`` / NaN for missing).
        broad_of: Node name to human broad class.

    Returns:
        Integer indices; nodes outside the seven broad classes (sinks,
        unknown or missing names) get the ``unallocated`` index.
    """
    position = {name: index for index, name in enumerate(HUMAN_BROAD_CLASSES)}
    unallocated = len(HUMAN_BROAD_CLASSES)
    lookup = {
        str(name): position.get(str(broad), unallocated)
        for name, broad in broad_of.items()
    }
    return np.array(
        [
            unallocated if value is None else lookup.get(str(value), unallocated)
            for value in _object_array(names)
        ],
        dtype=np.int64,
    )


def whb_broad_of(vocab: VocabTable | None = None) -> dict[str, str]:
    """Return the WHB supercluster name to broad class map (sinks unassigned)."""
    table = vocab or primary_vocab("human")
    return {
        name: (UNASSIGNED_LABEL if table.is_sink(name) else table.broad_class(name))
        for name in table.names
    }


def soft_broad_matrix(
    names: np.ndarray,
    probabilities: np.ndarray,
    *,
    broad_of: Mapping[str, str],
) -> np.ndarray:
    """Return per-cell soft broad-class mass (plan §5.5).

    Args:
        names: ``(n, k)`` node names: the assigned supercluster, then its
            runner-ups (``None`` / NaN where absent).
        probabilities: ``(n, k)`` bootstrap probabilities of those nodes.
        broad_of: Node name to broad class.

    Returns:
        ``(n, 8)`` mass over ``COMPOSITION_COLUMNS``; mass on nodes outside
        the seven classes and the residual ``1 - sum`` go to ``unallocated``,
        so every row sums to 1 (cells without a call are all unallocated).

    Raises:
        ValueError: If the shapes differ.
    """
    names = np.asarray(names, dtype=object)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if names.ndim != 2 or names.shape != probabilities.shape:
        raise ValueError("names and probabilities must be (n, k) arrays of one shape")
    n_cells, n_nodes = names.shape
    matrix = np.zeros((n_cells, len(COMPOSITION_COLUMNS)), dtype=np.float64)
    rows = np.arange(n_cells)
    n_classes = len(HUMAN_BROAD_CLASSES)
    for column in range(n_nodes):
        index = broad_class_index(names[:, column], broad_of)
        values = np.nan_to_num(probabilities[:, column], nan=0.0)
        values = np.clip(values, 0.0, 1.0)
        allocated = index < n_classes
        np.add.at(matrix, (rows[allocated], index[allocated]), values[allocated])
    total = matrix[:, :n_classes].sum(axis=1)
    over = total > 1.0
    if over.any():
        matrix[over, :n_classes] /= total[over, None]
        total = np.minimum(total, 1.0)
    matrix[:, n_classes] = 1.0 - total
    return matrix


def one_hot_broad_matrix(
    broad_names: Sequence[object] | np.ndarray | pd.Series,
    *,
    include: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-cell one-hot broad-class rows (argmax or confident-only).

    Args:
        broad_names: Broad class per cell (anything outside the seven classes,
            incl. ``Mixed/Unknown`` and missing, is ``unallocated``).
        include: Cells that count; excluded cells get an all-zero row
            (confident-only composition).

    Returns:
        ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``.
    """
    identity = {name: name for name in HUMAN_BROAD_CLASSES}
    index = broad_class_index(broad_names, identity)
    matrix = np.zeros((len(index), len(COMPOSITION_COLUMNS)), dtype=np.float64)
    matrix[np.arange(len(index)), index] = 1.0
    if include is not None:
        matrix[~np.asarray(include, dtype=bool)] = 0.0
    return matrix


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the base-2 Jensen-Shannon distance of two compositions.

    Both vectors are restricted to the seven broad classes (the first seven
    entries when ``unallocated`` is appended) and renormalised, as the plan's
    JSD (§5.5) and ``scipy.spatial.distance.jensenshannon(p, q, base=2)``.

    Args:
        p: Class masses (7 or 8 entries).
        q: Class masses of the same length.

    Returns:
        The distance in [0, 1]; NaN if either vector has no mass.
    """
    n_classes = len(HUMAN_BROAD_CLASSES)
    first = np.asarray(p, dtype=np.float64)[..., :n_classes]
    second = np.asarray(q, dtype=np.float64)[..., :n_classes]
    first_total = first.sum(axis=-1, keepdims=True)
    second_total = second.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        first = first / first_total
        second = second / second_total
        middle = 0.5 * (first + second)
        left = np.where(first > 0, first * np.log2(first / middle), 0.0)
        right = np.where(second > 0, second * np.log2(second / middle), 0.0)
    divergence = 0.5 * (left.sum(axis=-1) + right.sum(axis=-1))
    distance = np.sqrt(np.clip(divergence, 0.0, None))
    empty = (first_total[..., 0] <= 0) | (second_total[..., 0] <= 0)
    distance = np.where(empty, np.nan, distance)
    return float(distance) if np.ndim(distance) == 0 else distance  # type: ignore[return-value]


def composition_shares(matrix: np.ndarray) -> dict[str, float]:
    """Return the class shares of a per-cell matrix, over all its mass.

    Args:
        matrix: ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``.

    Returns:
        Share per column (the seven classes and ``unallocated``) of the total
        mass; the seven-class shares are not renormalised.
    """
    totals = np.asarray(matrix, dtype=np.float64).sum(axis=0)
    mass = float(totals.sum())
    if mass <= 0:
        return {name: math.nan for name in COMPOSITION_COLUMNS}
    return {
        name: float(value / mass)
        for name, value in zip(COMPOSITION_COLUMNS, totals, strict=True)
    }


# --------------------------------------------------------------------------
# Spatial block bootstrap


def tile_codes(xy: np.ndarray, tile_um: float = TILE_UM) -> np.ndarray:
    """Return the square-tile id of each cell (plan §5.5: 500 µm tiles).

    Args:
        xy: ``(n, 2)`` coordinates in µm.
        tile_um: Tile edge.

    Returns:
        Dense integer tile ids ``0..n_tiles-1``; cells with non-finite
        coordinates get ``-1``.

    Raises:
        ValueError: If ``tile_um`` is not positive or ``xy`` is not ``(n, 2)``.
    """
    if tile_um <= 0:
        raise ValueError("tile_um must be positive")
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must be an (n, 2) array")
    finite = np.isfinite(points).all(axis=1)
    codes = np.full(len(points), -1, dtype=np.int64)
    if not finite.any():
        return codes
    cells = np.floor(points[finite] / tile_um).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    codes[finite] = inverse.reshape(-1)
    return codes


def tile_sums(matrix: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Sum per-cell composition rows by tile.

    Args:
        matrix: ``(n, k)`` per-cell rows.
        codes: Tile id per cell (``tile_codes``); ``-1`` cells are dropped.

    Returns:
        ``(n_tiles, k)`` sums.
    """
    codes = np.asarray(codes, dtype=np.int64)
    values = np.asarray(matrix, dtype=np.float64)
    keep = codes >= 0
    n_tiles = int(codes[keep].max()) + 1 if keep.any() else 0
    sums = np.zeros((n_tiles, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        sums[:, column] = np.bincount(
            codes[keep], weights=values[keep, column], minlength=n_tiles
        )
    return sums


@dataclass(frozen=True)
class BootstrapJsd:
    """A Jensen-Shannon distance with its spatial block-bootstrap CI.

    Attributes:
        jsd: Point estimate on all cells.
        ci_low: 2.5th percentile of the replicates.
        ci_high: 97.5th percentile of the replicates.
        n_reps: Replicates.
        n_tiles_a: Non-empty tiles of the first section.
        n_tiles_b: Non-empty tiles of the second section.
        replicates: Replicate distances.
    """

    jsd: float
    ci_low: float
    ci_high: float
    n_reps: int
    n_tiles_a: int
    n_tiles_b: int
    replicates: np.ndarray


def block_bootstrap_jsd(
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    *,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapJsd:
    """Bootstrap the JSD of two sections by resampling each one's tiles.

    Each replicate draws, independently per section, as many tiles as the
    section has, with replacement, and recomputes both compositions from the
    drawn tiles' sums (plan §5.5).

    Args:
        tiles_a: ``(n_tiles, k)`` tile sums of the first section.
        tiles_b: ``(n_tiles, k)`` tile sums of the second section.
        n_reps: Replicates (200 in the plan).
        seed: RNG seed.

    Returns:
        Point estimate, percentile CI and replicates.

    Raises:
        ValueError: If a section has no tile.
    """
    first = np.asarray(tiles_a, dtype=np.float64)
    second = np.asarray(tiles_b, dtype=np.float64)
    first = first[first.sum(axis=1) > 0]
    second = second[second.sum(axis=1) > 0]
    if len(first) == 0 or len(second) == 0:
        raise ValueError("each section needs at least one non-empty tile")
    point = jensen_shannon_distance(first.sum(axis=0), second.sum(axis=0))
    rng = np.random.default_rng(seed)
    weights_a = rng.multinomial(len(first), np.full(len(first), 1 / len(first)), n_reps)
    weights_b = rng.multinomial(
        len(second), np.full(len(second), 1 / len(second)), n_reps
    )
    replicates = np.asarray(
        jensen_shannon_distance(weights_a @ first, weights_b @ second),
        dtype=np.float64,
    ).reshape(-1)
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return BootstrapJsd(
        jsd=float(point),
        ci_low=float(low),
        ci_high=float(high),
        n_reps=int(n_reps),
        n_tiles_a=len(first),
        n_tiles_b=len(second),
        replicates=replicates,
    )


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
    vocab: VocabTable | None = None,
    n_runners_up: int = 5,
) -> pd.DataFrame:
    """Return SEA-AD's 7-class label and aggregated broad probability per cell.

    The label is the assigned subclass's broad class (vocab; "VLMC &
    Perivascular" split by the assigned supertype). The probability sums the
    bootstrap probability of the assigned subclass and of the runner-up
    subclasses that count toward the same class (``broad_classes_any``); for
    a split subclass it is multiplied by the supertype-level mass of the
    supertypes that give the same class (E1 ``load_mmc_seaad``).

    Args:
        subclass: SEA-AD subclass level of a tidy table (``level_frame``).
        supertype: The supertype level (any index order; aligned by cell).
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
        ll_broad: Likelihood-typer broad label in E2's scheme (``Exc``,
            ``Inh``, ``OtherNeuron``, ``Oligo``, ``OPC``, ``Astro``,
            ``Immune``, ``Vascular``), for the ``ll`` second-vote variants.
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
    - ``none``: no second method (implausible nodes are never rescued).

    The likelihood typer agrees in E2's scheme: neurons merged (the lenient
    rule, OtherNeuron counts as a neuron) and fibroblasts pooled with
    vascular cells.

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
    if uses_seaad or below60_vote == "seaad":
        if inputs.sea_broad is None or inputs.sea_broad_raw is None:
            raise ValueError(f"second_vote={second_vote!r} needs SEA-AD calls")
        sea_broad = _object_array(inputs.sea_broad)
        sea_in_classes = _in(sea_broad, HUMAN_BROAD_CLASSES)
        sea_broad = _keep_where(sea_in_classes, sea_broad)
        sea_raw = np.nan_to_num(
            np.asarray(inputs.sea_broad_raw, dtype=np.float64), nan=0.0
        )
        sea_confident = (sea_raw >= settings.seaad_broad) & sea_in_classes
        sea_agree_lineage = _equal(
            _mapped(sea_broad, HUMAN_LINEAGE_OF_BROAD_CLASS), lineage
        )
        sea_agree_broad = _equal(sea_broad, broad)
        if uses_seaad:
            disagree_lineage = sea_confident & ~sea_agree_lineage
            disagree_broad = sea_confident & ~sea_agree_broad
            sea_confident_opc = sea_confident & (sea_broad == OPC)
            rescued = sea_agree_lineage
        if below60_vote == "seaad":
            agree_lineage, agree_broad = sea_agree_lineage, sea_agree_broad
    if below60_vote == "ll":
        if inputs.ll_broad is None:
            raise ValueError(
                f"second_vote={second_vote!r} needs likelihood-typer calls"
            )
        ll_class = _mapped(inputs.ll_broad, E2_MERGED_CLASS)
        whb_class = _mapped(broad, E2_CLASS_OF_HUMAN_BROAD)
        agree_broad = _equal(ll_class, whb_class)
        agree_lineage = _equal(
            _mapped(ll_class, E2_LINEAGE_OF_CLASS),
            _mapped(whb_class, E2_LINEAGE_OF_CLASS),
        )
        if not uses_seaad:
            rescued = agree_lineage
    vote_lineage = np.where(below, agree_lineage, ~disagree_lineage)
    vote_broad = np.where(below, agree_broad, ~disagree_broad)

    has_lineage = _in(lineage, HUMAN_LINEAGES)
    lineage_confident = (
        has_lineage
        & (lineage_raw >= settings.whb_broad)
        & (counts >= hard_floor)
        & (~implausible | rescued)
        & vote_lineage
    )
    broad_floor = floor_lookup.per_cell("broad", platform, supercluster)
    broad_candidate = (
        lineage_confident
        & ~implausible
        & _in(broad, HUMAN_BROAD_CLASSES)
        & (broad_raw >= settings.whb_broad)
        & (counts >= broad_floor)
        & vote_broad
    )
    supercluster_floor = floor_lookup.per_cell("supercluster", platform, supercluster)
    is_cop = supercluster == COP_SUPERCLUSTER
    cop_passes = (
        (counts >= supercluster_floor) & (supercluster_bp >= settings.whb_supercluster)
    ) | sea_confident_opc
    cop_suppressed = lineage_confident & is_cop & ~cop_passes
    broad_confident = broad_candidate & ~(is_cop & ~cop_passes)
    verdict = dataset_gate(counts, broad_confident, n_segmented=n_segmented, gate=gate)
    supercluster_confident = (
        broad_confident
        & (verdict.level == "full")
        & (supercluster_bp >= settings.whb_supercluster)
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
) -> HumanRuleInputs:
    """Build rule inputs from a ``<sid>_ct_provisional.parquet`` table.

    Args:
        labels: Provisional labels of table cells, indexed by cell id
            (``ct_lineage_*``, ``ct_broad_*`` and the ``<prefix>_supercluster_*``
            engine columns).
        sea: ``seaad_broad_calls`` output (any order; aligned by cell id).
        ll_broad: E2 likelihood-typer broad labels by cell id (cells without
            one never agree).
        prefix: WHB engine-column prefix (``mmc_whb``; ``mmc_whb_setc`` for
            the set-c run).

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
