"""Human marker-consistency referee (plan §8.8, §8.6; M13 chunk C13).

The human row of plan §8.8's marker referee: the canonical-marker
consistency of confident ``ct_broad`` calls over marker-pseudo-confident
table cells, on marker sets derived per panel (§8.6; decision D18 (a),
pre-registration §23.6 NR5). It ports mouse G2 (``mouse_gate``
``class_group_markers`` and ``marker_pseudo_labels``) to the primary WHB
bundle's ``profiles.parquet``, with the WHB superclusters of the bundle's
leaf level grouped to the seven human broad classes by the WHB vocab
(``composition.whb_broad_of``; sinks and the nodes outside the seven
classes belong to none).

* **Marker sets** (D27 (a)). For each broad class, the panel genes (the
  bundle's query genes) that pass the §8.6 / E3 specificity rule
  (``mouse_flags.specific_genes`` with ``AnnotationFlagsConfig``
  ``specific_gene_ratio`` 20 and ``specific_gene_min_share`` 1/1000;
  immediate-early genes excluded, in human case), most specific first.
  Classes with fewer than ``real_qc.marker_referee_min_group_markers`` (3)
  are left out. The comparator (``real_qc.marker_referee_comparator``):

  - ``node`` (the C13 specification, the default): mouse G2's rule. The
    class's profile is the unweighted mean of its supercluster profiles,
    compared with every other supercluster, sinks and nodes of no broad
    class included (G2 compares a group with every class outside it).
  - ``class``: the cell-weighted broad-class profile (``n_cells`` of each
    supercluster; the codebase's human class profile, as
    ``flags.class_profiles``) compared with the other broad classes'
    profiles; nodes of no broad class are not compared.

  The two differ where one broad class holds a small node that shares
  another class's genes (WHB's Committed oligodendrocyte precursor, in the
  OPC class, shares oligodendrocyte genes) or a sink resembles a class
  (Splatter, neuronal): ``node`` then leaves that class without markers.
  The sets depend only on the bundle and the panel, so they are fixed
  before any dataset is read: ``RefereeMarkers.to_frame`` writes them and
  ``RefereeMarkers.fingerprint`` identifies them; a run with supplied
  (frozen or hand-curated, D27 (b)) sets records the same fingerprint.
* **Pseudo-labels.** ``data/P1212``'s rule, as mouse G2: each marker's
  counts over its mean positive count among the dataset's table cells;
  class score = the sum; a label when the top class has
  ``marker_referee_min_marker_units`` (1.5) units and
  ``marker_referee_min_marker_share`` (60%) of the summed units.
* **Consistency.** Among the table cells that are marker-pseudo-confident
  and have a confident ``ct_broad`` call, the share whose call equals the
  pseudo-label: H9's denominator (confident cells with a resolved
  pseudo-label). A confident call to a class without markers can only
  disagree, as in G2. ``not_evaluable`` with fewer than two classes with
  markers, or fewer than ``marker_referee_min_pseudo_confident`` (200)
  scored cells.

``HumanMarkerReferee.signal`` gives ``real_qc.MarkerConsistencySignal``;
``real_qc.marker_consistency_outcome`` turns it into the outcome (a warning
below 0.75, the dataset gate capped at ``broad_only`` below 0.70; D18 (a)).
Those thresholds came from the H9 hand lists; the derived statistic is
re-measured on set a (M13 C17) before the new-panel human MERSCOPE family is
scored, and the thresholds go back to the user if set a falls below 0.75.

The module needs numpy, pandas and (for the pseudo-labels) scipy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.mouse_flags import (
    IMMEDIATE_EARLY_GENES,
    SpecificGenes,
    specific_genes,
)
from merxen.annotation.mouse_gate import marker_pseudo_labels
from merxen.annotation.real_qc import MarkerConsistencySignal
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import AnnotationFlagsConfig, AnnotationRealQcConfig

logger = logging.getLogger(__name__)

# E3's immediate-early genes (``mouse_flags.IMMEDIATE_EARLY_GENES``) in human
# case: never markers.
HUMAN_IMMEDIATE_EARLY_GENES: Final[frozenset[str]] = frozenset(
    symbol.upper() for symbol in IMMEDIATE_EARLY_GENES
)
COMPARATOR_NODE: Final = "node"
COMPARATOR_CLASS: Final = "class"
REFEREE_COMPARATORS: Final[tuple[str, ...]] = (COMPARATOR_NODE, COMPARATOR_CLASS)
# The referee needs at least two classes with markers (D18 (a)).
MIN_MARKER_GROUPS: Final = 2
SOURCE_DERIVED: Final = "derived"
SOURCE_SUPPLIED: Final = "supplied"
PROFILES_FILE: Final = "profiles.parquet"
PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "node_name",
    "n_cells",
    "gene_id",
    "expected_fraction",
)
MARKER_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "rank",
    "gene_id",
    "symbol",
    "ratio",
)
_BROAD_CLASSES: Final[frozenset[str]] = frozenset(HUMAN_BROAD_CLASSES)


# --------------------------------------------------------------------------
# Settings


@dataclass(frozen=True)
class HumanRefereeSettings:
    """The human referee's rule (``AnnotationRealQcConfig``; D18 (a)).

    Attributes:
        min_group_markers: Broad classes with fewer derived markers are left
            out.
        min_marker_units: Units the top class of a pseudo-label needs.
        min_marker_share: Its share of the summed units.
        min_pseudo_confident: Scored cells the statistic needs.
        comparator: ``node`` (mouse G2's rule on the superclusters) or
            ``class`` (broad-class profiles).
    """

    min_group_markers: int
    min_marker_units: float
    min_marker_share: float
    min_pseudo_confident: int
    comparator: str

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.comparator not in REFEREE_COMPARATORS:
            raise ValueError(
                f"unknown referee comparator {self.comparator!r}; expected one of "
                f"{REFEREE_COMPARATORS}"
            )
        if self.min_group_markers < 1:
            raise ValueError("min_group_markers must be at least 1")
        if self.min_pseudo_confident < 1:
            raise ValueError("min_pseudo_confident must be at least 1")
        if not self.min_marker_units > 0:
            raise ValueError("min_marker_units must be positive")
        if not 0.0 < self.min_marker_share <= 1.0:
            raise ValueError("min_marker_share must lie in (0, 1]")

    @classmethod
    def from_config(cls, config: AnnotationRealQcConfig) -> HumanRefereeSettings:
        """Read the settings from ``AnnotationConfig.real_qc``.

        Args:
            config: The real-data QC config.

        Returns:
            The settings.
        """
        return cls(
            min_group_markers=int(config.marker_referee_min_group_markers),
            min_marker_units=float(config.marker_referee_min_marker_units),
            min_marker_share=float(config.marker_referee_min_marker_share),
            min_pseudo_confident=int(config.marker_referee_min_pseudo_confident),
            comparator=str(config.marker_referee_comparator),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the settings as JSON."""
        return {
            "min_group_markers": self.min_group_markers,
            "min_marker_units": self.min_marker_units,
            "min_marker_share": self.min_marker_share,
            "min_pseudo_confident": self.min_pseudo_confident,
            "comparator": self.comparator,
        }


# --------------------------------------------------------------------------
# Profiles


@dataclass(frozen=True)
class HumanRefereeProfiles:
    """WHB node profiles on a query's genes, grouped to the broad classes.

    Attributes:
        gene_ids: Query genes (columns of ``node_profiles``).
        symbols: Their symbols (immediate-early exclusion and reporting).
        node_names: Nodes of the level (rows), sorted.
        node_classes: Each node's human broad class, ``None`` for sinks and
            nodes outside the seven classes.
        node_cells: Reference cells per node (``n_cells``; NaN if absent).
        node_profiles: Expected fraction per node x gene (``profiles.parquet``
            ``expected_fraction``; genes the table lacks are zero).
    """

    gene_ids: tuple[str, ...]
    symbols: tuple[str, ...]
    node_names: tuple[str, ...]
    node_classes: tuple[str | None, ...]
    node_cells: np.ndarray
    node_profiles: np.ndarray

    @classmethod
    def from_table(
        cls,
        profiles: pd.DataFrame,
        *,
        level: str,
        gene_ids: Sequence[str],
        symbols: Sequence[str],
        broad_of: Mapping[str, str] | None = None,
    ) -> HumanRefereeProfiles:
        """Build the profiles from a bundle's ``profiles.parquet``.

        Args:
            profiles: The long table (``level``, ``node_name``, ``gene_id``,
                ``expected_fraction``; ``n_cells`` for the ``class``
                comparator).
            level: The level whose nodes are grouped (the primary bundle's
                leaf level, the WHB supercluster).
            gene_ids: Query genes; genes the table lacks get zero.
            symbols: Their symbols.
            broad_of: Node name to broad class (default: the WHB vocab,
                ``composition.whb_broad_of``).

        Returns:
            The profiles.

        Raises:
            ValueError: If a column or the level is missing, or the symbols
                do not fit the genes.
        """
        if len(symbols) != len(gene_ids):
            raise ValueError("symbols must have one entry per gene")
        needed = {"level", "node_name", "gene_id", "expected_fraction"}
        missing = sorted(needed - set(profiles.columns))
        if missing:
            raise ValueError(f"profiles table lacks columns {missing}")
        rows = profiles[profiles["level"].astype(str) == level]
        if rows.empty:
            raise ValueError(f"profiles table has no {level} rows")
        rows = rows.assign(
            node_name=rows["node_name"].astype(str), gene_id=rows["gene_id"].astype(str)
        )
        genes = [str(gene) for gene in gene_ids]
        wide = rows.pivot_table(
            index="node_name",
            columns="gene_id",
            values="expected_fraction",
            aggfunc="first",
        )
        wide = wide.sort_index().reindex(columns=genes).fillna(0.0)
        names = tuple(str(name) for name in wide.index)
        if "n_cells" in rows.columns:
            cells = (
                rows.drop_duplicates("node_name")
                .set_index("node_name")["n_cells"]
                .reindex(list(names))
                .to_numpy(np.float64)
            )
        else:
            cells = np.full(len(names), np.nan)
        if broad_of is None:
            from merxen.annotation.composition import whb_broad_of

            broad_of = whb_broad_of()
        classes = tuple(_broad_class(broad_of.get(name)) for name in names)
        return cls(
            gene_ids=tuple(genes),
            symbols=tuple(str(symbol) for symbol in symbols),
            node_names=names,
            node_classes=classes,
            node_cells=cells,
            node_profiles=wide.to_numpy(np.float64),
        )

    def present_classes(self) -> tuple[str, ...]:
        """Return the broad classes with a node, in ``HUMAN_BROAD_CLASSES`` order."""
        present = set(self.node_classes)
        return tuple(cls for cls in HUMAN_BROAD_CLASSES if cls in present)

    def class_profile(self, cls: str) -> np.ndarray:
        """Return a broad class's cell-weighted profile (``class`` comparator).

        Args:
            cls: A broad class with a node.

        Returns:
            The ``n_cells``-weighted mean of its node profiles.

        Raises:
            ValueError: If the class has no node or its nodes have no
                positive ``n_cells``.
        """
        rows = np.array([value == cls for value in self.node_classes], dtype=bool)
        if not rows.any():
            raise ValueError(f"no {cls} node in the profiles")
        weights = self.node_cells[rows]
        if not (np.isfinite(weights).all() and weights.sum() > 0):
            raise ValueError(
                f"the {cls} nodes have no positive n_cells: the class comparator "
                "weights each node by its reference cells"
            )
        profile: np.ndarray = (self.node_profiles[rows] * weights[:, None]).sum(
            axis=0
        ) / weights.sum()
        return profile


def _broad_class(value: object) -> str | None:
    return str(value) if value is not None and str(value) in _BROAD_CLASSES else None


def load_referee_profiles(
    bundle_dir: Path | str,
    *,
    level: str,
    gene_ids: Sequence[str],
    symbols: Sequence[str],
) -> HumanRefereeProfiles | None:
    """Return a WHB bundle's referee profiles on the query genes.

    Args:
        bundle_dir: The primary bundle directory.
        level: Its leaf level (the WHB supercluster).
        gene_ids: Query genes.
        symbols: Their symbols.

    Returns:
        The profiles, or ``None`` when the bundle has no usable
        ``profiles.parquet`` (the referee is then ``not_evaluable``).
    """
    path = Path(bundle_dir) / PROFILES_FILE
    if not path.is_file():
        return None
    try:
        table = pd.read_parquet(path, columns=list(PROFILE_COLUMNS))
        return HumanRefereeProfiles.from_table(
            table, level=level, gene_ids=gene_ids, symbols=symbols
        )
    except ValueError as error:
        logger.warning("no human referee profiles from %s: %s", path, error)
        return None


# --------------------------------------------------------------------------
# Marker sets


@dataclass(frozen=True)
class RefereeMarkers:
    """The referee's marker sets on one panel (D27).

    Attributes:
        markers: Broad class to its markers (positions in the query genes),
            in ``HUMAN_BROAD_CLASSES`` order; only classes with at least
            ``min_group_markers``.
        n_query_genes: Query genes the positions refer to.
        source: ``derived`` (from the profiles) or ``supplied`` (a frozen or
            hand-curated table).
        min_group_markers: The rule that left classes out.
        comparator: The derivation's comparator (``None`` when supplied
            without one).
        left_out: Classes with too few markers.
        absent: Broad classes without a node in the profiles.
        missing_symbols: Per class, supplied symbols absent from the panel.
        ratio: The specificity ratio of a derivation.
        min_share: Its minimum share.
    """

    markers: Mapping[str, SpecificGenes]
    n_query_genes: int
    source: str
    min_group_markers: int
    comparator: str | None = None
    left_out: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    missing_symbols: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ratio: float | None = None
    min_share: float | None = None

    @property
    def fingerprint(self) -> str:
        """Return the sha256 of the marker sets (class to sorted gene IDs)."""
        sets = {group: sorted(genes.gene_ids) for group, genes in self.markers.items()}
        text = json.dumps(sets, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def symbol_sets(self) -> dict[str, list[str]]:
        """Return each class's marker symbols, most specific first."""
        return {group: list(genes.symbols) for group, genes in self.markers.items()}

    def to_frame(self) -> pd.DataFrame:
        """Return the sets as a table (written and fixed before a run).

        Returns:
            One row per marker: ``group``, ``rank`` (1 = most specific),
            ``gene_id``, ``symbol``, ``ratio`` (NaN for hand-curated sets).
        """
        rows = [
            {
                "group": group,
                "rank": rank,
                "gene_id": gene_id,
                "symbol": symbol,
                "ratio": ratio,
            }
            for group, genes in self.markers.items()
            for rank, (gene_id, symbol, ratio) in enumerate(
                zip(genes.gene_ids, genes.symbols, genes.ratios, strict=True), start=1
            )
        ]
        return pd.DataFrame(rows, columns=list(MARKER_FRAME_COLUMNS))

    def to_json(self) -> dict[str, Any]:
        """Return the sets and their provenance as JSON."""
        return {
            "source": self.source,
            "comparator": self.comparator,
            "fingerprint": self.fingerprint,
            "min_group_markers": self.min_group_markers,
            "ratio": self.ratio,
            "min_share": self.min_share,
            "markers": self.symbol_sets(),
            "marker_gene_ids": {
                group: list(genes.gene_ids) for group, genes in self.markers.items()
            },
            "left_out": list(self.left_out),
            "absent": list(self.absent),
            "missing_symbols": {
                group: list(values) for group, values in self.missing_symbols.items()
            },
        }

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        gene_ids: Sequence[str],
        symbols: Sequence[str],
        min_group_markers: int = 3,
        comparator: str | None = None,
    ) -> RefereeMarkers:
        """Return frozen sets (``to_frame``) on a run's query genes.

        Args:
            frame: The table (``group``, ``gene_id``; ``rank`` and ``ratio``
                when present).
            gene_ids: The run's query genes.
            symbols: Their symbols (the run's panel file).
            min_group_markers: Classes with fewer markers are left out.
            comparator: The comparator the sets were derived with, if known.

        Returns:
            The sets (``source`` ``supplied``).

        Raises:
            ValueError: If a column is missing, a class is not a human broad
                class or a gene is not among the query genes.
        """
        missing = sorted({"group", "gene_id"} - set(frame.columns))
        if missing:
            raise ValueError(f"marker table lacks columns {missing}")
        position = _positions(gene_ids, symbols)
        table = frame.assign(
            group=frame["group"].astype(str), gene_id=frame["gene_id"].astype(str)
        )
        if "rank" not in table.columns:
            table = table.assign(rank=np.arange(len(table)))
        if "ratio" not in table.columns:
            table = table.assign(ratio=np.nan)
        _check_groups(table["group"])
        absent = sorted(set(table["gene_id"]) - set(position))
        if absent:
            raise ValueError(
                f"marker genes {absent[:5]} are not among the query genes "
                f"({len(absent)} missing)"
            )
        sets: dict[str, list[tuple[str, float]]] = {}
        for group in HUMAN_BROAD_CLASSES:
            rows = table[table["group"] == group].sort_values(["rank", "gene_id"])
            if len(rows):
                sets[group] = [
                    (str(gene), float(ratio))
                    for gene, ratio in zip(rows["gene_id"], rows["ratio"], strict=True)
                ]
        return _supplied(
            sets,
            gene_ids=gene_ids,
            symbols=symbols,
            min_group_markers=min_group_markers,
            comparator=comparator,
        )

    @classmethod
    def from_symbols(
        cls,
        sets: Mapping[str, Sequence[str]],
        *,
        gene_ids: Sequence[str],
        symbols: Sequence[str],
        min_group_markers: int = 3,
    ) -> RefereeMarkers:
        """Return hand-curated sets (D27 (b), reported) on a run's query genes.

        Args:
            sets: Broad class to marker symbols.
            gene_ids: The run's query genes.
            symbols: Their symbols.
            min_group_markers: Classes with fewer panel markers are left out.

        Returns:
            The sets; symbols absent from the panel are dropped and listed.

        Raises:
            ValueError: If a class is not a human broad class.
        """
        _check_groups(sets)
        gene_of: dict[str, str] = {}
        for gene_id, symbol in zip(gene_ids, symbols, strict=True):
            gene_of.setdefault(str(symbol), str(gene_id))
        found: dict[str, list[tuple[str, float]]] = {}
        missing: dict[str, tuple[str, ...]] = {}
        for group in HUMAN_BROAD_CLASSES:
            if group not in sets:
                continue
            listed = [str(symbol) for symbol in sets[group]]
            found[group] = [
                (gene_of[symbol], math.nan) for symbol in listed if symbol in gene_of
            ]
            lacking = tuple(symbol for symbol in listed if symbol not in gene_of)
            if lacking:
                missing[group] = lacking
        return _supplied(
            found,
            gene_ids=gene_ids,
            symbols=symbols,
            min_group_markers=min_group_markers,
            missing_symbols=missing,
        )


def _positions(gene_ids: Sequence[str], symbols: Sequence[str]) -> dict[str, int]:
    if len(symbols) != len(gene_ids):
        raise ValueError("symbols must have one entry per gene")
    position: dict[str, int] = {}
    for index, gene_id in enumerate(gene_ids):
        position.setdefault(str(gene_id), index)
    return position


def _check_groups(groups: Any) -> None:
    unknown = sorted({str(group) for group in groups} - _BROAD_CLASSES)
    if unknown:
        raise ValueError(
            f"{unknown} is not a human broad class; expected {HUMAN_BROAD_CLASSES}"
        )


def _supplied(
    sets: Mapping[str, Sequence[tuple[str, float]]],
    *,
    gene_ids: Sequence[str],
    symbols: Sequence[str],
    min_group_markers: int,
    comparator: str | None = None,
    missing_symbols: Mapping[str, tuple[str, ...]] | None = None,
) -> RefereeMarkers:
    position = _positions(gene_ids, symbols)
    markers: dict[str, SpecificGenes] = {}
    left_out: list[str] = []
    for group, genes in sets.items():
        if len(genes) < min_group_markers:
            left_out.append(group)
            continue
        indices = np.array([position[gene] for gene, _ in genes], dtype=np.int64)
        markers[group] = SpecificGenes(
            target=group,
            indices=indices,
            gene_ids=tuple(str(gene_ids[index]) for index in indices),
            symbols=tuple(str(symbols[index]) for index in indices),
            ratios=tuple(float(ratio) for _, ratio in genes),
        )
    return RefereeMarkers(
        markers=markers,
        n_query_genes=len(gene_ids),
        source=SOURCE_SUPPLIED,
        min_group_markers=min_group_markers,
        comparator=comparator,
        left_out=tuple(left_out),
        missing_symbols=dict(missing_symbols or {}),
    )


def derive_referee_markers(
    profiles: HumanRefereeProfiles,
    flags_config: AnnotationFlagsConfig,
    settings: HumanRefereeSettings,
) -> RefereeMarkers:
    """Return each broad class's panel markers (§8.6 specificity rule; D27 (a)).

    Args:
        profiles: The primary bundle's node profiles on the query genes.
        flags_config: ``specific_gene_ratio`` and ``specific_gene_min_share``.
        settings: ``min_group_markers`` and the comparator.

    Returns:
        The sets (``source`` ``derived``).

    Raises:
        ValueError: If the ``class`` comparator meets nodes without cells.
    """
    present = profiles.present_classes()
    classes = np.array(profiles.node_classes, dtype=object)
    class_profiles: dict[str, np.ndarray] = {}
    if settings.comparator == COMPARATOR_CLASS:
        class_profiles = {cls: profiles.class_profile(cls) for cls in present}
    markers: dict[str, SpecificGenes] = {}
    left_out: list[str] = []
    for cls in present:
        if settings.comparator == COMPARATOR_NODE:
            inside = np.array([value == cls for value in classes], dtype=bool)
            target = profiles.node_profiles[inside].mean(axis=0)
            others = profiles.node_profiles[~inside]
        else:
            target = class_profiles[cls]
            others = np.array(
                [class_profiles[other] for other in present if other != cls]
            )
        if not len(others):
            left_out.append(cls)
            continue
        genes = specific_genes(
            target,
            others,
            gene_ids=profiles.gene_ids,
            symbols=profiles.symbols,
            ratio=flags_config.specific_gene_ratio,
            min_share=flags_config.specific_gene_min_share,
            target=cls,
            exclude_symbols=HUMAN_IMMEDIATE_EARLY_GENES,
        )
        if len(genes) >= settings.min_group_markers:
            markers[cls] = genes
        else:
            left_out.append(cls)
    return RefereeMarkers(
        markers=markers,
        n_query_genes=len(profiles.gene_ids),
        source=SOURCE_DERIVED,
        min_group_markers=settings.min_group_markers,
        comparator=settings.comparator,
        left_out=tuple(left_out),
        absent=tuple(cls for cls in HUMAN_BROAD_CLASSES if cls not in present),
        ratio=float(flags_config.specific_gene_ratio),
        min_share=float(flags_config.specific_gene_min_share),
    )


# --------------------------------------------------------------------------
# The referee


@dataclass(frozen=True)
class HumanMarkerReferee:
    """The human marker referee of one dataset (§8.8; D18 (a)).

    Attributes:
        consistency: Share of the scored cells whose confident ``ct_broad``
            equals the marker pseudo-label (``None`` when not evaluable).
        n_cells: Table cells.
        n_confident: Table cells with a confident ``ct_broad`` call.
        n_pseudo_confident: Marker-pseudo-confident table cells.
        n_scored: Cells both (the denominator).
        markers: The marker sets (``None`` without inputs).
        settings: The rule.
        recall: Per pseudo-label class, the share of its scored cells whose
            call agrees.
        precision: Per called class, the share of its scored cells whose
            pseudo-label agrees.
        confusion: Scored cells per pseudo-label class and called class.
        reason: Why the consistency is ``None``.
    """

    consistency: float | None
    n_cells: int
    n_confident: int
    n_pseudo_confident: int
    n_scored: int
    markers: RefereeMarkers | None
    settings: HumanRefereeSettings
    recall: Mapping[str, float] = field(default_factory=dict)
    precision: Mapping[str, float] = field(default_factory=dict)
    confusion: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    reason: str | None = None

    @property
    def n_marker_groups(self) -> int:
        """Return the classes with markers."""
        return 0 if self.markers is None else len(self.markers.markers)

    def to_json(self) -> dict[str, Any]:
        """Return the referee as JSON (resolve summary, QC details)."""
        markers = None if self.markers is None else self.markers.to_json()
        return {
            "consistency": _round(self.consistency),
            "n_cells": self.n_cells,
            "n_confident": self.n_confident,
            "n_pseudo_confident": self.n_pseudo_confident,
            "n_scored": self.n_scored,
            "n_marker_groups": self.n_marker_groups,
            "comparator": None if self.markers is None else self.markers.comparator,
            "fingerprint": None if self.markers is None else self.markers.fingerprint,
            "marker_sets": markers,
            "recall": {key: _round(value) for key, value in self.recall.items()},
            "precision": {key: _round(value) for key, value in self.precision.items()},
            "confusion": {
                key: dict(value.items()) for key, value in self.confusion.items()
            },
            "settings": self.settings.to_json(),
            "reason": self.reason,
        }

    def signal(self) -> MarkerConsistencySignal:
        """Return the input of ``real_qc.marker_consistency_outcome``."""
        details = self.to_json()
        details.pop("consistency")
        return MarkerConsistencySignal(
            consistency=self.consistency,
            n_marker_groups=self.n_marker_groups,
            n_pseudo_labelled=self.n_scored,
            reason=self.reason,
            details=details,
        )


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)


def _names(values: Sequence[object] | np.ndarray) -> np.ndarray:
    return np.array(
        [None if value is None or pd.isna(value) else str(value) for value in values],
        dtype=object,
    )


def human_marker_referee(
    counts: sparse.spmatrix | np.ndarray | None,
    profiles: HumanRefereeProfiles | None,
    broad_names: Sequence[object] | np.ndarray,
    confident: Sequence[bool] | np.ndarray,
    *,
    flags_config: AnnotationFlagsConfig,
    settings: HumanRefereeSettings,
    markers: RefereeMarkers | None = None,
) -> HumanMarkerReferee:
    """Return the human marker referee of one dataset's table cells (D18 (a)).

    Args:
        counts: Table cells x query genes (``None``: not evaluable). The
            pseudo-labels normalise each marker by its mean positive count
            over these cells, so pass every table cell.
        profiles: The primary WHB bundle's profiles on the query genes
            (``load_referee_profiles``); not needed with ``markers``.
        broad_names: ``ct_broad_name`` per table cell.
        confident: Whether each table cell's ``ct_broad`` is confident.
        flags_config: The specificity rule.
        settings: The referee rule (``HumanRefereeSettings.from_config``).
        markers: Frozen or hand-curated sets used instead of a derivation.

    Returns:
        The referee.

    Raises:
        ValueError: If the inputs do not have one entry per table cell or
            the counts do not have one column per query gene.
    """
    names = _names(broad_names)
    called_mask = np.asarray(confident, dtype=bool).reshape(-1)
    n_cells = len(names)
    if len(called_mask) != n_cells:
        raise ValueError("broad_names and confident must have one entry per table cell")
    in_class = np.array([name in _BROAD_CLASSES for name in names], dtype=bool)
    called = called_mask & in_class
    n_confident = int(called.sum())

    def empty(reason: str, sets: RefereeMarkers | None) -> HumanMarkerReferee:
        return HumanMarkerReferee(
            consistency=None,
            n_cells=n_cells,
            n_confident=n_confident,
            n_pseudo_confident=0,
            n_scored=0,
            markers=sets,
            settings=settings,
            reason=reason,
        )

    if counts is None:
        return empty("no query counts", markers)
    if counts.shape[0] != n_cells:
        raise ValueError("counts must have one row per table cell")
    if markers is None:
        if profiles is None:
            return empty("no profiles and no supplied marker sets", None)
        if counts.shape[1] != len(profiles.gene_ids):
            raise ValueError(
                f"counts have {counts.shape[1]} columns for "
                f"{len(profiles.gene_ids)} query genes"
            )
        markers = derive_referee_markers(profiles, flags_config, settings)
    elif counts.shape[1] != markers.n_query_genes:
        raise ValueError(
            f"counts have {counts.shape[1]} columns for {markers.n_query_genes} "
            "query genes"
        )
    labels, pseudo = marker_pseudo_labels(
        counts,
        markers.markers,
        min_units=settings.min_marker_units,
        min_share=settings.min_marker_share,
    )
    scored = pseudo & called
    agree = scored & (labels == names)
    recall: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = {}
    for group in markers.markers:
        rows = scored & (labels == group)
        if rows.any():
            recall[group] = float(agree[rows].mean())
            calls, n = np.unique(names[rows].astype(str), return_counts=True)
            confusion[group] = {
                str(name): int(value) for name, value in zip(calls, n, strict=True)
            }
    precision = {
        cls: float(agree[scored & (names == cls)].mean())
        for cls in HUMAN_BROAD_CLASSES
        if (scored & (names == cls)).any()
    }
    n_scored = int(scored.sum())
    reason: str | None = None
    consistency: float | None = None
    if len(markers.markers) < MIN_MARKER_GROUPS:
        reason = (
            f"{len(markers.markers)} marker group(s) with >= "
            f"{markers.min_group_markers} markers ({MIN_MARKER_GROUPS} needed)"
        )
    elif n_scored < settings.min_pseudo_confident:
        reason = (
            f"{n_scored} scored pseudo-confident cells < "
            f"{settings.min_pseudo_confident}"
        )
    else:
        consistency = float(agree.sum() / n_scored)
    return HumanMarkerReferee(
        consistency=consistency,
        n_cells=n_cells,
        n_confident=n_confident,
        n_pseudo_confident=int(pseudo.sum()),
        n_scored=n_scored,
        markers=markers,
        settings=settings,
        recall=recall,
        precision=precision,
        confusion=confusion,
        reason=reason,
    )
