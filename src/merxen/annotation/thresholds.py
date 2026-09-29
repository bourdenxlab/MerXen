"""Raw thresholds, count floors, resolvability-gated emission and the gate (§5.4).

RESOLVE decides per cell and level whether a label may be emitted. This
module holds the numeric side of that decision; ``consensus`` applies it.

- **Raw v1 thresholds** (plan §3.7, D-C1): the pre-registered bootstrap
  probability thresholds per level (WHB lineage / broad / NT 0.73,
  supercluster 0.69; SEA-AD broad 0.68, subclass 0.55 below 60 counts and
  0.45 from 60; WMB class 0.90, subclass 0.80). ``raw_threshold`` returns
  them per cell and ``apply_raw_thresholds`` tests probabilities against
  them with the float32 tolerance of ``schema.meets_threshold``.
- **Floors** (§5.4, D-C2): ``load_floors`` reads the packaged
  ``floors_<species>.csv``; ``derive_floors`` is the one floor algorithm
  (smallest grid depth with precision >= target and >= 50 test cells; a
  class with fewer inherits the other platforms' floor, else the hard floor,
  ``default_low_n``). ``FloorPlan`` applies them per (level, class,
  platform, panel family): real-data-validated families use the packaged
  rows of their family and platform; every other panel or platform takes
  ``max(maximum over the known platforms and validated panels, the panel's
  simulated floor)`` (``unknown_panel``, with a warning;
  ``simulation_validated`` without one), and the mouse subclass floor is at
  least 60 there (§8.2).
- **Resolvability-gated emission** (§8.3, §8.2): ``EmissionPlan`` reads the
  bundle's resolvability decisions (reweighted to the dataset's composition
  in RESOLVE) and returns, per cell, whether a level is emitted for the
  cell's class at its depth bin and with which threshold. Real-data-validated
  families (up to ``validated_max_level``) use the pre-registered defaults
  (``threshold_source = validated_default``; the local rule's raises are only
  reported, H18); every other panel, and families validated by simulation,
  use the local isotonic thresholds of the table, whose targets carry the
  provisional margins (``resolvability_local``). Merging is raise-only: a
  threshold is never below the configured default or the bundle's recorded
  default.
- **Dataset gate** (§5.4, rev3): the verdict is a **level** (``full``,
  ``broad_only``, ``failed``) plus a separate **warning flag** with reasons.
  A = share of table cells with >= 30 counts; ``broad_only`` if A < 0.30;
  ``failed`` if confident broad coverage of table cells < 0.25; warning if
  confident broad coverage of segmented objects < 0.15. The primary panel's
  trust state can force ``broad_only`` or ``failed``; a provisional panel
  sets the warning flag but never lowers the level; a family validated by
  simulation warns only when more than 10% of a dataset's confident labels
  at an emitted level fall outside its validated region.

The module needs numpy and pandas (and the resolvability tables' helpers);
the trust decision is read through its attributes, so ``diagnostics`` is
imported for type checking only.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import pandas as pd

from merxen.annotation.config import AnnotationGate, AnnotationThresholds
from merxen.annotation.resolvability import (
    LevelMeta,
    ResolvabilityTables,
    RuleSettings,
    depth_bin,
    level_emission,
    simulated_floors,
)
from merxen.annotation.schema import GateLevel, meets_threshold
from merxen.annotation.vocab import (
    FLOOR_COLUMNS,
    FLOOR_FILES,
    SPECIES,
    Species,
    asset_path,
    load_floor_table,
)

if TYPE_CHECKING:
    from merxen.annotation.diagnostics import TrustDecision
    from merxen.annotation.resolvability import DatasetComposition

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Raw v1 thresholds (plan §3.7)

# The AnnotationThresholds field holding each primary level's raw default.
# Coarse levels aggregate the leaf's bootstrap probability (E2), so they share
# the broad / class threshold; the report-only fine levels (OD-E4) share the
# leaf's.
PRIMARY_THRESHOLD_FIELDS: Final[dict[str, dict[str, str]]] = {
    "human": {
        "lineage": "whb_broad",
        "broad": "whb_broad",
        "nt": "whb_broad",
        "supercluster": "whb_supercluster",
        "cluster": "whb_supercluster",
    },
    "mouse": {
        "broad": "wmb_class",
        "class": "wmb_class",
        "nt": "wmb_class",
        "subclass": "wmb_subclass",
        "supertype": "wmb_subclass",
    },
}
# Levels of the human second vote (SEA-AD). ``seaad_broad`` is the 7-class
# call ("SEA confidently disagrees" / "confidently calls OPC", §5.2);
# ``seaad_subclass`` is the secondary name (rule 5), whose threshold depends
# on the depth.
SECONDARY_LEVELS: Final[tuple[str, ...]] = ("seaad_broad", "seaad_subclass")
# Precision targets per level (§3.7): broad, lineage, NT and class 0.90;
# supercluster and subclass 0.85 (fine levels follow their leaf).
TARGET_FIELDS: Final[dict[str, str]] = {
    "lineage": "target_lineage",
    "broad": "target_broad",
    "nt": "target_nt",
    "class": "target_class",
    "supercluster": "target_supercluster",
    "subclass": "target_subclass",
    "cluster": "target_supercluster",
    "supertype": "target_subclass",
    "seaad_broad": "target_broad",
    "seaad_subclass": "target_supercluster",
}
# Levels whose emission is read from another level's resolvability table:
# SEA-AD's subclass has no in-silico truth (E2: real data only), so it is
# emitted where the primary resolves the leaf for the cell's broad class
# (§5.2 "every level is also subject to the panel's resolvability"; fails
# towards coarser labels, §8.1).
DERIVED_EMISSION_LEVELS: Final[dict[str, dict[str, str]]] = {
    "human": {"seaad_subclass": "supercluster"},
    "mouse": {},
}

ThresholdSource = Literal["validated_default", "resolvability_local"]
EmissionRegime = Literal["validated", "provisional"]

REASON_RESOLVABILITY_NOT_RUN: Final = "resolvability_not_run"


def _check_species(species: str) -> Species:
    if species not in SPECIES:
        raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
    return cast(Species, species)


def _counts_array(counts: np.ndarray | Sequence[float] | None, n: int) -> np.ndarray:
    if counts is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(counts, dtype=np.float64)


def level_target(level: str, thresholds: AnnotationThresholds) -> float:
    """Return a level's precision target on real-data-validated panels (§3.7).

    Args:
        level: Level name (primary or secondary).
        thresholds: The threshold settings.

    Returns:
        The target (0.90 or 0.85 with the defaults).

    Raises:
        ValueError: For an unknown level.
    """
    if level not in TARGET_FIELDS:
        raise ValueError(f"no precision target for level {level!r}")
    return float(getattr(thresholds, TARGET_FIELDS[level]))


def raw_threshold(
    level: str,
    *,
    thresholds: AnnotationThresholds,
    species: Species,
    counts: np.ndarray | Sequence[float] | None = None,
    n_cells: int | None = None,
) -> np.ndarray:
    """Return the pre-registered raw threshold of a level, per cell.

    Args:
        level: A primary level of the species, or ``seaad_broad`` /
            ``seaad_subclass`` (human).
        thresholds: The threshold settings (v1 raw defaults).
        species: ``"human"`` or ``"mouse"``.
        counts: Each cell's total counts; needed for ``seaad_subclass``
            (0.55 below ``second_vote_below_counts``, 0.45 from it).
        n_cells: Number of cells when ``counts`` is not given.

    Returns:
        One threshold per cell (float64).

    Raises:
        ValueError: For an unknown level, a mode other than ``raw`` or a
            depth-dependent level without counts.
    """
    checked = _check_species(species)
    if thresholds.mode != "raw":
        raise ValueError(
            f"thresholds.mode {thresholds.mode!r} is v1.1 (M10); v1 applies raw "
            "thresholds only"
        )
    if counts is None and n_cells is None:
        raise ValueError("raw_threshold needs counts or n_cells")
    n = len(counts) if counts is not None else int(n_cells or 0)
    if level == "seaad_subclass":
        if checked != "human":
            raise ValueError("seaad_subclass is a human level")
        if counts is None:
            raise ValueError("seaad_subclass thresholds depend on the counts")
        values = np.asarray(counts, dtype=np.float64)
        return np.where(
            values < thresholds.second_vote_below_counts,
            thresholds.seaad_subclass_below60,
            thresholds.seaad_subclass_from60,
        ).astype(np.float64)
    if level == "seaad_broad":
        if checked != "human":
            raise ValueError("seaad_broad is a human level")
        return np.full(n, float(thresholds.seaad_broad), dtype=np.float64)
    fields = PRIMARY_THRESHOLD_FIELDS[checked]
    if level not in fields:
        raise ValueError(f"{level!r} is not a {checked} level {tuple(fields)}")
    return np.full(n, float(getattr(thresholds, fields[level])), dtype=np.float64)


def apply_raw_thresholds(
    raw: np.ndarray | Sequence[float],
    threshold: np.ndarray | float,
) -> np.ndarray:
    """Return which raw probabilities meet their thresholds (float32-tolerant).

    Args:
        raw: Raw engine probabilities (NaN never passes).
        threshold: One threshold, or one per cell (NaN never passes).

    Returns:
        Boolean array.
    """
    values = np.asarray(raw, dtype=np.float64)
    limit = np.asarray(threshold, dtype=np.float64)
    passes = meets_threshold(values, np.nan_to_num(limit, nan=np.inf))
    return np.asarray(passes, dtype=bool)


def merge_thresholds(
    default: np.ndarray | float, *local: np.ndarray | float | None
) -> np.ndarray:
    """Merge thresholds raise-only: the largest defined value wins.

    Args:
        default: The pre-registered default (per cell or scalar).
        local: Raised candidates (e.g. the bundle's recorded default, a local
            isotonic ``t*``); NaN or ``None`` entries are ignored.

    Returns:
        ``max(default, *local)`` per cell, never below ``default``.
    """
    merged = np.asarray(default, dtype=np.float64).copy()
    for candidate in local:
        if candidate is None:
            continue
        merged = np.fmax(merged, np.asarray(candidate, dtype=np.float64))
    return np.asarray(merged, dtype=np.float64)


# --------------------------------------------------------------------------
# Floors (§5.4)

# The floor-table level of each annotation level. ``None``: the hard floor
# only (human lineage). NT uses its class's broad floor (§5.2 rule 3: Inh
# MERSCOPE 10, Xenium 30); SEA-AD's subclass uses the supercluster floor of
# its broad class (rule 5); fine levels use their leaf's.
FLOOR_LEVELS: Final[dict[str, dict[str, str | None]]] = {
    "human": {
        "lineage": None,
        "broad": "broad",
        "nt": "broad",
        "supercluster": "supercluster",
        "seaad_subclass": "supercluster",
        "cluster": "supercluster",
    },
    "mouse": {
        "broad": "class",
        "class": "class",
        "nt": "class",
        "subclass": "subclass",
        "supertype": "subclass",
    },
}
# Mouse levels that take ``provisional_mouse_subclass_floor`` while the panel
# is provisional or validated by simulation (§8.2).
MOUSE_SUBCLASS_FLOOR_LEVELS: Final[frozenset[str]] = frozenset(
    {"subclass", "supertype"}
)
FloorPolicy = Literal["packaged", "simulation_validated", "unknown_panel"]
FLOOR_POLICIES: Final[tuple[str, ...]] = (
    "packaged",
    "simulation_validated",
    "unknown_panel",
)
# Per-value floor sources (provenance): the packaged row's own source
# (``real_e2``, ``design_v1``, ...) or one of these.
SOURCE_HARD_FLOOR: Final = "hard_floor"
SOURCE_UNKNOWN_PLATFORM: Final = "unknown_platform"
DERIVED_SOURCE: Final = "derived"
INHERITED_SOURCE: Final = "inherited"
LOW_N_SOURCE: Final = "default_low_n"
NOT_REACHED_SOURCE: Final = "not_reached"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class FloorTable:
    """The count floors of one species (``floors_<species>.csv``).

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        frame: One row per (level, floor_class, platform, panel_family) with
            an integer ``min_counts`` (``vocab.FLOOR_COLUMNS``).
        path: The file read.
        sha256: Its digest (provenance: floors sha256, §4.6).
    """

    species: Species
    frame: pd.DataFrame
    path: Path
    sha256: str

    def _rows(self, floor_level: str, floor_class: str) -> pd.DataFrame:
        frame = self.frame
        return frame[
            (frame["level"] == floor_level) & (frame["floor_class"] == floor_class)
        ]

    def packaged(
        self,
        floor_level: str,
        floor_class: str,
        *,
        platform: str,
        panel_family: str | None,
    ) -> tuple[int, str] | None:
        """Return a family's packaged floor on one platform, if listed.

        Args:
            floor_level: Floor-table level.
            floor_class: Floor class.
            platform: ``MERSCOPE`` or ``XENIUM`` (case-insensitive).
            panel_family: The family id (``floors_<species>.csv``
                ``panel_family``).

        Returns:
            ``(min_counts, floor_source)`` or ``None`` when no row matches.
        """
        if panel_family is None:
            return None
        rows = self._rows(floor_level, floor_class)
        rows = rows[
            (rows["platform"].str.upper() == str(platform).upper())
            & (rows["panel_family"] == panel_family)
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return int(row["min_counts"]), str(row["floor_source"] or "packaged")

    def known(self, floor_level: str, floor_class: str) -> int | None:
        """Return the maximum floor over every platform and validated panel.

        Args:
            floor_level: Floor-table level.
            floor_class: Floor class.

        Returns:
            The maximum ``min_counts``, or ``None`` when the class is not
            listed at that level.
        """
        rows = self._rows(floor_level, floor_class)
        if rows.empty:
            return None
        return int(rows["min_counts"].max())

    @property
    def panel_families(self) -> tuple[str, ...]:
        """The families with packaged floors."""
        return tuple(sorted(set(self.frame["panel_family"].astype(str))))


def load_floors(species: Species, path: Path | str | None = None) -> FloorTable:
    """Load the count floors of a species (§5.4).

    Args:
        species: ``"human"`` or ``"mouse"``.
        path: A floor table with the packaged columns
            (``AnnotationThresholds.floors_path``); ``None`` reads the
            packaged ``floors_<species>.csv``.

    Returns:
        The floor table.

    Raises:
        ValueError: If a column is missing, a key is duplicated or a floor
            is negative.
    """
    checked = _check_species(species)
    if path is None:
        source = asset_path(FLOOR_FILES[checked])
        frame = load_floor_table(checked)
    else:
        source = Path(path)
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        missing = [column for column in FLOOR_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{source}: missing columns {missing}")
        frame["min_counts"] = frame["min_counts"].astype(int)
        key = ["level", "floor_class", "platform", "panel_family"]
        duplicated = frame[frame.duplicated(key)]
        if not duplicated.empty:
            raise ValueError(
                f"{source}: duplicated keys {duplicated[key].values.tolist()}"
            )
    if bool((frame["min_counts"] < 0).any()):
        raise ValueError(f"{source}: negative floors")
    frame = frame.copy()
    frame["platform"] = frame["platform"].astype(str).str.upper()
    return FloorTable(
        species=checked, frame=frame, path=source, sha256=_file_sha256(source)
    )


DERIVE_FLOOR_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "floor_class",
    "platform",
    "min_counts",
    "floor_source",
    "inherited_from",
    "max_n",
)


def derive_floors(
    stats: pd.DataFrame,
    *,
    targets: Mapping[str, float] | AnnotationThresholds,
    hard_floor: int,
    min_n: int = 50,
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Derive count floors with the one floor algorithm (§5.4, D-C2).

    Per (level, class, platform[, group]): the smallest grid depth whose
    precision reaches the level's target with at least ``min_n`` test cells
    there (``derived``). A (level, class, platform) with fewer than
    ``min_n`` test cells at every depth inherits the largest floor of the
    other platforms of the same (level, class[, group]) that have enough
    test cells and a floor (``inherited``, ``inherited_from``); otherwise it
    takes the hard floor (``default_low_n``). A class with enough test cells
    that never reaches its target has no floor (``not_reached``,
    ``min_counts`` missing): the level is not emitted for it.

    Args:
        stats: One row per (level, class, platform, depth) with columns
            ``level``, ``class``, ``platform``, ``depth``, ``n`` (test cells
            at that depth) and ``precision``, plus ``group_columns``.
        targets: Precision target per level, or the threshold settings
            (``level_target``).
        hard_floor: The hard floor (``min_counts``).
        min_n: Test cells a depth needs (50).
        group_columns: Extra key columns (e.g. ``panel_family``); inheritance
            stays within a group.

    Returns:
        ``DERIVE_FLOOR_COLUMNS`` (+ ``group_columns``); ``min_counts`` is a
        nullable integer.

    Raises:
        ValueError: If a required column is missing.
    """
    required = ["level", "class", "platform", "depth", "n", "precision"]
    missing = [column for column in [*required, *group_columns] if column not in stats]
    if missing:
        raise ValueError(f"derive_floors: missing columns {missing}")

    def target_of(level: str) -> float:
        if isinstance(targets, AnnotationThresholds):
            return level_target(level, targets)
        return float(targets[level])

    groups = list(group_columns)
    own: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, rows in stats.groupby(
        ["level", "class", "platform", *groups], observed=True, sort=True
    ):
        level, cls, platform = str(key[0]), str(key[1]), str(key[2]).upper()
        extra = tuple(key[3:])
        n = rows["n"].to_numpy(np.float64)
        precision = rows["precision"].to_numpy(np.float64)
        depth = rows["depth"].to_numpy(np.int64)
        enough = n >= min_n
        with np.errstate(invalid="ignore"):
            passes = enough & (precision >= target_of(level) - 1e-12)
        own[(level, cls, platform, *extra)] = {
            "floor": int(depth[passes].min()) if passes.any() else None,
            "supported": bool(enough.any()),
            "max_n": int(np.nanmax(n)) if len(n) and np.isfinite(n).any() else 0,
        }
    records = []
    for (level, cls, platform, *rest), entry in own.items():
        inherited_from = ""
        if entry["floor"] is not None:
            value: int | None = max(int(entry["floor"]), hard_floor)
            source = DERIVED_SOURCE
        elif entry["supported"]:
            value, source = None, NOT_REACHED_SOURCE
        else:
            donors = [
                (other_platform, other["floor"])
                for (o_level, o_cls, other_platform, *o_rest), other in own.items()
                if o_level == level
                and o_cls == cls
                and o_rest == rest
                and other_platform != platform
                and other["supported"]
                and other["floor"] is not None
            ]
            if donors:
                value = max(hard_floor, max(int(floor) for _, floor in donors))
                source = INHERITED_SOURCE
                inherited_from = ";".join(sorted(name for name, _ in donors))
            else:
                value, source = hard_floor, LOW_N_SOURCE
        records.append(
            {
                "level": level,
                "floor_class": cls,
                "platform": platform,
                **dict(zip(groups, rest, strict=True)),
                "min_counts": value,
                "floor_source": source,
                "inherited_from": inherited_from,
                "max_n": entry["max_n"],
            }
        )
    columns = [*DERIVE_FLOOR_COLUMNS[:3], *groups, *DERIVE_FLOOR_COLUMNS[3:]]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["min_counts"] = frame["min_counts"].astype(pd.Int64Dtype())
    return frame


@dataclass(frozen=True)
class FloorValue:
    """One (level, class) floor on the run's platform and panel.

    Attributes:
        min_counts: The floor (``inf``: no table cell reaches it).
        source: ``hard_floor``, the packaged row's source (``real_e2``,
            ``design_v1``), ``unknown_platform``, ``unknown_panel`` or
            ``simulation_validated``.
        known: The maximum packaged floor over platforms and panels.
        simulated: The panel's simulated floor (smallest emitted depth), if
            the resolvability table has one.
        warning: Whether the choice warns (``unknown_panel``, or a family
            floor missing for the platform).
    """

    min_counts: float
    source: str
    known: int | None = None
    simulated: int | None = None
    warning: bool = False

    def to_json(self) -> dict[str, Any]:
        """Return the floor as JSON."""
        return {
            "min_counts": None if math.isinf(self.min_counts) else self.min_counts,
            "source": self.source,
            "known": self.known,
            "simulated": self.simulated,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class FloorPlan:
    """The count floors RESOLVE applies to one dataset (§5.4, §8.2).

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        platform: The dataset's platform.
        table: The packaged floors.
        hard_floor: ``min_counts`` (table cells reach it by definition).
        policies: Floor policy per level (``TrustDecision.floor_source``);
            a level not listed uses ``default_policy``.
        default_policy: ``unknown_panel`` (fail-safe) unless given.
        panel_family: The panel's family (packaged rows).
        simulated: Smallest emitted depth per (resolvability level, class)
            under the provisional regime (``EmissionPlan.simulated_floors``);
            ``None`` values mark classes never emitted.
        provisional_mouse_subclass_floor: Mouse subclass floor while
            provisional or validated by simulation (60).
    """

    species: Species
    platform: str
    table: FloorTable
    hard_floor: int
    policies: Mapping[str, FloorPolicy] = field(default_factory=dict)
    default_policy: FloorPolicy = "unknown_panel"
    panel_family: str | None = None
    simulated: Mapping[tuple[str, str], int | None] = field(default_factory=dict)
    provisional_mouse_subclass_floor: int = 60

    @classmethod
    def build(
        cls,
        *,
        species: Species,
        platform: str,
        hard_floor: int,
        trust: TrustDecision | None,
        thresholds: AnnotationThresholds | None = None,
        table: FloorTable | None = None,
        emission: EmissionPlan | None = None,
    ) -> FloorPlan:
        """Return the floors a trust decision and emission plan imply.

        Args:
            species: ``"human"`` or ``"mouse"``.
            platform: ``MERSCOPE`` or ``XENIUM``.
            hard_floor: ``min_counts``.
            trust: The primary reference's trust decision (``None``: every
                level ``unknown_panel``, the fail-safe).
            thresholds: Threshold settings (floors path, mouse subclass
                floor); default settings when ``None``.
            table: The floor table (default: ``load_floors`` of the
                settings' ``floors_path``).
            emission: The emission plan (simulated floors).

        Returns:
            The plan.
        """
        settings = thresholds or AnnotationThresholds()
        checked = _check_species(species)
        floors = table or load_floors(checked, settings.floors_path)
        policies: dict[str, FloorPolicy] = {}
        if trust is not None:
            for level in FLOOR_LEVELS[checked]:
                policies[level] = trust.floor_source(level)
        return cls(
            species=checked,
            platform=str(platform).upper(),
            table=floors,
            hard_floor=int(hard_floor),
            policies=policies,
            panel_family=None if trust is None else trust.family_id,
            simulated={} if emission is None else emission.simulated_floors(),
            provisional_mouse_subclass_floor=settings.provisional_mouse_subclass_floor,
        )

    def policy(self, level: str) -> FloorPolicy:
        """Return a level's floor policy."""
        return self.policies.get(level, self.default_policy)

    def _simulated(self, level: str, floor_class: str) -> tuple[bool, int | None]:
        key_level = DERIVED_EMISSION_LEVELS[self.species].get(level, level)
        key = (key_level, floor_class)
        if key not in self.simulated:
            return False, None
        return True, self.simulated[key]

    def floor(self, level: str, floor_class: str | None) -> FloorValue:
        """Return the floor of one (level, class) on this platform and panel.

        Args:
            level: Annotation level.
            floor_class: The cell's floor class at the level (``None``:
                nodes outside the vocabulary, which are never confident).

        Returns:
            The floor.

        Raises:
            ValueError: For a level the species does not have.
        """
        levels = FLOOR_LEVELS[self.species]
        if level not in levels:
            raise ValueError(f"{level!r} is not a {self.species} level {tuple(levels)}")
        floor_level = levels[level]
        if floor_level is None:
            return FloorValue(float(self.hard_floor), SOURCE_HARD_FLOOR)
        if floor_class is None:
            return FloorValue(math.inf, SOURCE_HARD_FLOOR)
        known = self.table.known(floor_level, floor_class)
        policy = self.policy(level)
        if policy == "packaged":
            packaged = self.table.packaged(
                floor_level,
                floor_class,
                platform=self.platform,
                panel_family=self.panel_family,
            )
            if packaged is not None:
                value, source = packaged
                return FloorValue(
                    float(max(value, self.hard_floor)), source, known=known
                )
            # A validated family without a row for this platform or class:
            # the unknown-platform max rule, with a warning.
            value = self.hard_floor if known is None else max(known, self.hard_floor)
            return FloorValue(
                float(value), SOURCE_UNKNOWN_PLATFORM, known=known, warning=True
            )
        value = self.hard_floor if known is None else max(known, self.hard_floor)
        tabulated, simulated = self._simulated(level, floor_class)
        if tabulated and simulated is not None:
            value = max(value, int(simulated))
        if self.species == "mouse" and level in MOUSE_SUBCLASS_FLOOR_LEVELS:
            value = max(value, self.provisional_mouse_subclass_floor)
        return FloorValue(
            float(value),
            policy,
            known=known,
            simulated=simulated,
            warning=policy == "unknown_panel",
        )

    def per_cell(
        self, level: str, floor_classes: Sequence[str | None] | np.ndarray
    ) -> np.ndarray:
        """Return each cell's floor at one level (``inf`` where none applies).

        Args:
            level: Annotation level.
            floor_classes: Each cell's floor class at the level.

        Returns:
            Floors as float64.
        """
        classes = np.asarray(floor_classes, dtype=object)
        values = np.full(len(classes), math.inf, dtype=np.float64)
        cache: dict[object, float] = {}
        for index, cls in enumerate(classes):
            key = None if _missing(cls) else str(cls)
            if key not in cache:
                cache[key] = self.floor(level, key).min_counts
            values[index] = cache[key]
        return values

    def summary(self, levels: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return every (level, class) floor with its source (provenance).

        Args:
            levels: Levels to list (default: every level of the species).

        Returns:
            One record per (level, class) the table or the simulation knows.
        """
        records = []
        for level in levels or tuple(FLOOR_LEVELS[self.species]):
            floor_level = FLOOR_LEVELS[self.species][level]
            if floor_level is None:
                classes: list[str | None] = [None]
            else:
                table_classes = set(
                    self.table.frame.loc[
                        self.table.frame["level"] == floor_level, "floor_class"
                    ].astype(str)
                )
                key_level = DERIVED_EMISSION_LEVELS[self.species].get(level, level)
                sim_classes = {cls for (lvl, cls) in self.simulated if lvl == key_level}
                classes = sorted(table_classes | sim_classes)
            for cls in classes:
                value = self.floor(level, cls)
                records.append(
                    {
                        "level": level,
                        "class": cls,
                        "policy": self.policy(level),
                        **value.to_json(),
                    }
                )
        return records

    def warnings(self) -> list[str]:
        """Return one reason per level whose floors warn (§5.4)."""
        reasons = []
        for level in FLOOR_LEVELS[self.species]:
            if self.policy(level) == "unknown_panel":
                reasons.append(f"floors_unknown_panel:{level}")
        return reasons

    def to_json(self) -> dict[str, Any]:
        """Return the floor provenance (sha256, policies, values)."""
        return {
            "floors_path": str(self.table.path),
            "floors_sha256": self.table.sha256,
            "platform": self.platform,
            "panel_family": self.panel_family,
            "hard_floor": self.hard_floor,
            "policies": {
                level: self.policy(level) for level in FLOOR_LEVELS[self.species]
            },
            "floors": self.summary(),
        }


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(isinstance(value, str) and not value.strip())


# --------------------------------------------------------------------------
# Resolvability-gated emission (§8.3)


@dataclass(frozen=True)
class LevelEmission:
    """Per-cell emission of one level (§8.3; ``EmissionPlan.level``).

    Attributes:
        level: Level name.
        regime: ``validated`` or ``provisional`` (``None``: no resolvability
            table; every cell is emitted at the defaults).
        threshold_source: ``validated_default`` or ``resolvability_local``.
        emitted: Whether the level may be emitted for the cell's class at
            its depth bin.
        threshold: The threshold to apply (raise-only merge).
        default_threshold: The pre-registered default per cell.
        local_threshold: The table's threshold (``t*`` in the provisional
            regime, the checked default in the validated one); NaN where the
            table has none.
        extrapolated: The cell's bin takes a pooled deep set's verdict
            (``resolvability_extrapolated``).
        depth_bin: The cell's grid bin (NaN below the grid).
        reason: Why the level is not emitted (``None`` where emitted).
    """

    level: str
    regime: EmissionRegime | None
    threshold_source: ThresholdSource
    emitted: np.ndarray
    threshold: np.ndarray
    default_threshold: np.ndarray
    local_threshold: np.ndarray
    extrapolated: np.ndarray
    depth_bin: np.ndarray
    reason: np.ndarray

    def __len__(self) -> int:
        return len(self.emitted)

    def summary(self, mask: np.ndarray | None = None) -> dict[str, Any]:
        """Return the emitted and extrapolated shares (provenance).

        Args:
            mask: Cells to summarise (default: all).

        Returns:
            ``regime``, ``threshold_source``, ``n_cells``, ``emitted_share``,
            ``extrapolated_share`` and the not-emitted reasons with counts.
        """
        keep = (
            np.ones(len(self), dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        n = int(keep.sum())
        reasons: dict[str, int] = {}
        for value in self.reason[keep]:
            if value is None:
                continue
            reasons[str(value)] = reasons.get(str(value), 0) + 1
        return {
            "level": self.level,
            "regime": self.regime,
            "threshold_source": self.threshold_source,
            "n_cells": n,
            "emitted_share": float(self.emitted[keep].mean()) if n else None,
            "extrapolated_share": float(self.extrapolated[keep].mean()) if n else None,
            "not_emitted_reasons": dict(sorted(reasons.items())),
        }


@dataclass(frozen=True)
class EmissionPlan:
    """Resolvability-gated emission and thresholds of one primary bundle (§8.3).

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        thresholds: The threshold settings (pre-registered defaults).
        decisions: ``resolvability.decide`` output (RESOLVE: reweighted to
            the dataset's composition); ``None`` when the bundle has no
            self-map (every level emitted at the defaults; a trust state
            without resolvability is already ``broad_only`` for panels
            outside the validated families, §8.2).
        levels: The bundle's level metadata.
        grid: The bundle's depth grid.
        trust: The primary reference's trust decision (``None``: every level
            in the provisional regime, the fail-safe).
        fine_seed_stability: ``resolvability_summary.json``
            ``fine_level_seed_stability``.
        seed_stability_max_change: ``seed_stability_max_change`` (0.02).
    """

    species: Species
    thresholds: AnnotationThresholds
    decisions: pd.DataFrame | None = None
    levels: tuple[LevelMeta, ...] = ()
    grid: tuple[int, ...] = ()
    trust: TrustDecision | None = None
    fine_seed_stability: Mapping[str, float] | None = None
    seed_stability_max_change: float = 0.02

    @classmethod
    def from_tables(
        cls,
        tables: ResolvabilityTables | None,
        *,
        species: Species,
        thresholds: AnnotationThresholds,
        trust: TrustDecision | None,
        composition: Mapping[str, float] | DatasetComposition | None = None,
        settings: RuleSettings | None = None,
        seed_stability_max_change: float = 0.02,
    ) -> EmissionPlan:
        """Return the plan of a bundle's resolvability tables.

        Args:
            tables: ``resolvability.load_resolvability`` output, or ``None``.
            species: ``"human"`` or ``"mouse"``.
            thresholds: The threshold settings.
            trust: The primary reference's trust decision.
            composition: The dataset's soft composition over truth types (per
                depth bin); ``None`` keeps PREP's unweighted decisions.
            settings: Rule settings (default: the bundle's).
            seed_stability_max_change: Fine-level seed stability limit.

        Returns:
            The plan.
        """
        checked = _check_species(species)
        if tables is None:
            return cls(species=checked, thresholds=thresholds, trust=trust)
        decisions = tables.decisions(composition=composition, settings=settings)
        stability = tables.summary.get("fine_level_seed_stability") or {}
        return cls(
            species=checked,
            thresholds=thresholds,
            decisions=decisions,
            levels=tuple(tables.levels),
            grid=tuple(tables.depth_grid),
            trust=trust,
            fine_seed_stability={str(k): float(v) for k, v in stability.items()},
            seed_stability_max_change=seed_stability_max_change,
        )

    @property
    def has_tables(self) -> bool:
        """Whether a resolvability table gates emission."""
        return self.decisions is not None

    def table_level(self, level: str) -> str:
        """Return the resolvability level whose table gates ``level``."""
        return DERIVED_EMISSION_LEVELS[self.species].get(level, level)

    def regime(self, level: str) -> EmissionRegime:
        """Return a level's decision regime (``TrustDecision.emission_regime``).

        Args:
            level: Level name.

        Returns:
            ``validated`` for a real-data-validated family up to its
            ``validated_max_level``; ``provisional`` otherwise (also without
            a trust decision).
        """
        if self.trust is None:
            return "provisional"
        return self.trust.emission_regime(level)

    def threshold_source(self, level: str) -> ThresholdSource:
        """Return ``threshold_source`` of a level (§8.3)."""
        if level in SECONDARY_LEVELS or not self.has_tables:
            return "validated_default"
        if self.regime(level) == "validated":
            return "validated_default"
        return "resolvability_local"

    def _meta(self, level: str) -> LevelMeta | None:
        return next((item for item in self.levels if item.level == level), None)

    def default_threshold(
        self, level: str, counts: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        """Return a level's pre-registered threshold, raised to the bundle's.

        The bundle records the default it decided with; a configured
        default below it would apply a threshold the table never checked,
        so the larger of the two applies (raise-only).

        Args:
            level: Level name.
            counts: Each cell's total counts.

        Returns:
            One threshold per cell.
        """
        values = _counts_array(counts, 0)
        configured = raw_threshold(
            level, thresholds=self.thresholds, species=self.species, counts=values
        )
        if level in SECONDARY_LEVELS:
            return configured
        meta = self._meta(level)
        if meta is None:
            return configured
        if meta.default_threshold > float(np.max(configured, initial=-np.inf)) + 1e-12:
            logger.warning(
                "%s: the bundle decided at default %.3f, above the configured "
                "%.3f; the bundle's value applies (raise-only)",
                level,
                meta.default_threshold,
                float(np.max(configured, initial=0.0)),
            )
        return merge_thresholds(configured, meta.default_threshold)

    def level(
        self,
        level: str,
        class_keys: Sequence[str | None] | np.ndarray,
        counts: np.ndarray | Sequence[float],
    ) -> LevelEmission:
        """Return per-cell emission and thresholds of one level.

        Args:
            level: Level name (a primary level, or a derived one such as
                human ``seaad_subclass``).
            class_keys: Each cell's class key in the level's table (the
                called class: human E2 floor classes, ``COP`` at
                supercluster; mouse WMB classes); ``None`` never emits.
            counts: Each cell's total counts.

        Returns:
            The emission.
        """
        values = _counts_array(counts, 0)
        n_cells = len(values)
        keys = np.asarray(class_keys, dtype=object)
        if len(keys) != n_cells:
            raise ValueError("class_keys and counts must have one value per cell")
        default = self.default_threshold(level, values)
        source = self.threshold_source(level)
        grid = self.grid
        if not self.has_tables:
            fine = level in ("cluster", "supertype")
            emitted = np.full(n_cells, not fine, dtype=bool)
            reason = np.full(
                n_cells,
                None if not fine else REASON_RESOLVABILITY_NOT_RUN,
                dtype=object,
            )
            return LevelEmission(
                level=level,
                regime=None,
                threshold_source=source,
                emitted=emitted,
                threshold=default,
                default_threshold=default,
                local_threshold=np.full(n_cells, np.nan),
                extrapolated=np.zeros(n_cells, dtype=bool),
                depth_bin=np.full(n_cells, np.nan),
                reason=reason,
            )
        table_level = self.table_level(level)
        regime = self.regime(table_level)
        assert self.decisions is not None
        table = level_emission(
            self.decisions,
            list(self.levels),
            table_level,
            [None if _missing(key) else str(key) for key in keys],
            values,
            list(grid),
            regime=regime,
            allow_fine_levels=self.thresholds.allow_fine_levels,
            fine_seed_stability=self.fine_seed_stability,
            seed_stability_max_change=self.seed_stability_max_change,
        )
        emitted = table["emitted"].to_numpy(dtype=bool)
        local = table["threshold"].to_numpy(dtype=np.float64)
        if level in SECONDARY_LEVELS or regime == "validated":
            threshold = default
        else:
            threshold = merge_thresholds(default, local)
        return LevelEmission(
            level=level,
            regime=regime,
            threshold_source=source,
            emitted=emitted,
            threshold=threshold,
            default_threshold=default,
            local_threshold=local,
            extrapolated=table["resolvability_extrapolated"].to_numpy(dtype=bool),
            depth_bin=table["depth_bin"].to_numpy(dtype=np.float64),
            reason=table["reason"].to_numpy(dtype=object),
        )

    def simulated_floors(self) -> dict[tuple[str, str], int | None]:
        """Return the smallest emitted depth per (level, class), provisional regime.

        Returns:
            ``resolvability.simulated_floors`` of the decisions (empty
            without a table).
        """
        if self.decisions is None:
            return {}
        return simulated_floors(self.decisions, "provisional")

    def depth_bins(self, counts: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return each cell's grid bin (NaN below the grid or without a grid)."""
        values = _counts_array(counts, 0)
        if not self.grid:
            return np.full(len(values), np.nan)
        return depth_bin(values, list(self.grid))


# --------------------------------------------------------------------------
# Dataset gate (§5.4; rev3 level + warning flag)

GATE_SEVERITY: Final[dict[str, int]] = {"full": 0, "broad_only": 1, "failed": 2}


@dataclass(frozen=True)
class GateVerdict:
    """The human dataset gate verdict: a level plus a warning flag (§5.4).

    Attributes:
        level: ``full``, ``broad_only`` or ``failed``.
        warning: The warning flag (never lowers the level).
        level_reasons: Why the level is below ``full`` (``code: detail``).
        warning_reasons: Why the warning flag is set.
        frac_ge30: A, the share of table cells with >= ``depth_counts``.
        broad_coverage_table: Confident broad coverage of table cells.
        broad_coverage_segmented: Coverage of segmented objects (NaN when
            the object count is unknown).
        n_table: Table cells.
        n_segmented: Segmented objects (``None`` when unknown).
        trust_state: The primary panel's trust state, if given.
    """

    level: GateLevel
    warning: bool
    level_reasons: tuple[str, ...]
    warning_reasons: tuple[str, ...]
    frac_ge30: float
    broad_coverage_table: float
    broad_coverage_segmented: float
    n_table: int
    n_segmented: int | None = None
    trust_state: str | None = None

    @property
    def reasons(self) -> tuple[str, ...]:
        """Level reasons followed by warning reasons."""
        return (*self.level_reasons, *self.warning_reasons)

    @property
    def attempts_leaf(self) -> bool:
        """Whether the leaf and SEA-AD subclass levels are attempted (``full``)."""
        return self.level == "full"

    def to_json(self) -> dict[str, Any]:
        """Return the verdict as JSON (provenance, resolve summary)."""

        def number(value: float) -> float | None:
            return None if not math.isfinite(value) else round(float(value), 6)

        return {
            "level": self.level,
            "warning": self.warning,
            "level_reasons": list(self.level_reasons),
            "warning_reasons": list(self.warning_reasons),
            "frac_ge30": number(self.frac_ge30),
            "broad_coverage_table": number(self.broad_coverage_table),
            "broad_coverage_segmented": number(self.broad_coverage_segmented),
            "n_table": self.n_table,
            "n_segmented": self.n_segmented,
            "trust_state": self.trust_state,
        }


def _worse(current: GateLevel, candidate: GateLevel) -> GateLevel:
    return candidate if GATE_SEVERITY[candidate] > GATE_SEVERITY[current] else current


def dataset_gate(
    total_counts: np.ndarray | Sequence[float],
    broad_confident: np.ndarray | Sequence[bool],
    *,
    n_segmented: int | None = None,
    gate: AnnotationGate | None = None,
    trust: TrustDecision | None = None,
    validated_share: Mapping[str, float | None] | None = None,
    extra_warnings: Sequence[str] = (),
) -> GateVerdict:
    """Return the dataset gate verdict (§5.4, pre-registered; rev3).

    Level (the worst applies): ``failed`` if confident broad coverage of
    table cells < ``min_table_broad_coverage`` (0.25), or the primary panel
    is refused (``panel_refused``); ``broad_only`` if A (share of table
    cells with >= 30 counts) < ``min_frac_ge30`` (0.30), or the primary
    panel is ``broad_only`` (``panel_broad_only``); ``full`` otherwise.
    Warning flag: confident broad coverage of segmented objects <
    ``warn_segmented_broad_coverage`` (0.15); a provisional primary panel
    (``panel_provisional``; never a lower level); a family validated by
    simulation with more than ``warn_unvalidated_share`` of confident labels
    outside its validated region at an emitted level; ``extra_warnings``
    (e.g. real-data QC, unknown-panel floors).

    Args:
        total_counts: Counts of the table cells.
        broad_confident: Confident broad flag per table cell (before the gate
            is applied to the statuses).
        n_segmented: Segmented objects of the sample (the warning's
            denominator); ``None`` skips that warning.
        gate: Gate settings (default: the pre-registered values).
        trust: The primary reference's trust decision.
        validated_share: Share of confident labels inside the validated
            region per emitted level (simulation-validated families).
        extra_warnings: Further warning reasons.

    Returns:
        The verdict.
    """
    settings = gate or AnnotationGate()
    counts = np.asarray(total_counts, dtype=np.float64)
    confident = np.asarray(broad_confident, dtype=bool)
    if len(confident) != len(counts):
        raise ValueError("total_counts and broad_confident must align")
    n_table = len(counts)
    frac = float((counts >= settings.depth_counts).sum() / n_table) if n_table else 0.0
    coverage = float(confident.sum() / n_table) if n_table else 0.0
    segmented = (
        float(confident.sum() / n_segmented)
        if n_segmented is not None and n_segmented > 0
        else math.nan
    )
    level: GateLevel = "full"
    level_reasons: list[str] = []
    warnings: list[str] = []
    if coverage < settings.min_table_broad_coverage:
        level = _worse(level, "failed")
        level_reasons.append(
            f"broad_coverage_table: {coverage:.3f} < "
            f"{settings.min_table_broad_coverage} (confident broad coverage of "
            f"{n_table} table cells)"
        )
    if frac < settings.min_frac_ge30:
        level = _worse(level, "broad_only")
        level_reasons.append(
            f"frac_ge{settings.depth_counts}: A = {frac:.3f} < {settings.min_frac_ge30}"
        )
    if trust is not None:
        cap = trust.gate_level_cap
        if cap is not None:
            level = _worse(level, cap)
            level_reasons.append(
                f"{trust.gate_reason}: primary panel trust {trust.state} "
                f"({', '.join(trust.reason_codes) or 'no reason'})"
            )
        if trust.gate_warning:
            warnings.append(
                f"panel_provisional: primary panel trust {trust.state} (banner; "
                "never a lower gate level)"
            )
        if validated_share:
            warn, reasons = trust.unvalidated_share_warning(
                validated_share, max_share=settings.warn_unvalidated_share
            )
            if warn:
                warnings.extend(
                    f"{reason}: > {settings.warn_unvalidated_share} of confident "
                    "labels outside the validated region"
                    for reason in reasons
                )
    if math.isfinite(segmented) and segmented < settings.warn_segmented_broad_coverage:
        warnings.append(
            f"broad_coverage_segmented: {segmented:.3f} < "
            f"{settings.warn_segmented_broad_coverage} (of {n_segmented} segmented "
            "objects)"
        )
    warnings.extend(str(item) for item in extra_warnings)
    verdict = GateVerdict(
        level=level,
        warning=bool(warnings),
        level_reasons=tuple(level_reasons),
        warning_reasons=tuple(warnings),
        frac_ge30=frac,
        broad_coverage_table=coverage,
        broad_coverage_segmented=segmented,
        n_table=n_table,
        n_segmented=n_segmented,
        trust_state=None if trust is None else trust.state,
    )
    logger.info(
        "Dataset gate: %s%s (A %.3f, broad coverage %.3f of table cells)",
        verdict.level,
        " + warning" if verdict.warning else "",
        frac,
        coverage,
    )
    return verdict
