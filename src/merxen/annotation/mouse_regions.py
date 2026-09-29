"""Mouse section-region inference and two-tier pruning (plan §7.2; E7).

The mouse region step of MAP (``pipeline.annotate_map``; plan §3.3 step 5)
works on the unpruned WMB mapping of one section:

1. **Confident neurons**: table cells whose WMB class is neuronal (classes
   01-29, broad class ``Neurons`` in the packaged vocab) with class and
   subclass bootstrap probability >= 0.9 (``confident_neuron_min_bp``).
2. **Tiles**: each confident neuron votes for its subclass's MERFISH home
   division (``wmb_region_share`` bundle, ``region_home``; OB split from OLF);
   the section is cut into 150 µm tiles (``tile_um``) and every tile with
   >= 3 such neurons (``min_neurons_per_tile``) is *assigned* its majority
   home (ties: the first division in sorted order, as E7's
   ``DataFrame.idxmax``).
3. **Presence**: a division is present when it holds >= 1% of the assigned
   tiles (``min_tile_fraction``) and its largest 8-connected component has
   >= 3 tiles (``min_component_tiles``) (``exp/E7/09_autoregion_v2.py``).
4. **Guard**: fewer than 200 assigned tiles (``min_assigned_tiles``), or no
   coordinates, skip pruning with a warning (fails safe; plan [L]).
   ``mouse_section_regions`` overrides the inference with a ``;``-separated
   division list, and ``none`` disables pruning.
5. **Two-tier drop rule** (E7 §7, ``05c_hybrid_droplist.py``): a class with
   less than 0.2 of its MERFISH grey-matter cells in the present divisions
   (``class_low_share``; and at least 100 such cells,
   ``min_merfish_cells_class``) drops its subclasses whose present share is
   below 0.3, and its subclasses with fewer than 20 MERFISH cells
   (``min_merfish_cells_subclass``) follow it; in every other class,
   subclasses below 0.1 are dropped. A class whose subclasses are all
   dropped is dropped as one node. The never-drop classes (Astro-Epen,
   OPC-Oligo, Vascular, Immune, Pineal) and classes without MERFISH grey
   cells are never dropped.
6. **Pruned re-map**: only the table cells whose unpruned class or subclass
   was dropped are mapped again, with ``--nodes_to_drop`` and the bundle's
   lookup filtered to the pruned tree (``pruned_lookup``: keys of dropped
   nodes removed, markers restricted to the query, marker-less parents
   collapsed), so ``cell_type_mapper`` never meets a zero-marker parent;
   every other cell keeps its label by construction (``merge_pruned_tidy``).

The unpruned labels stay in the run's tidy parquet (``mmc_wmb_unpruned_*``
in the label table), the re-mapped cells' tidy table in
``<sid>_mmc_<run_id>_pruned.parquet`` and the per-cell region columns
(``inferred_region``, ``region_dropped_level``, ``region_pruned_changed``) in
``<sid>_mouse_regions.parquet``; ``MouseRegionRecord`` in
``map_manifest.json`` records the decision.

Region coupling (``coupled_regions``) and every other rule variant are M6b
(§7.8); this module refuses them rather than ignoring them.

Imports: numpy, pandas, pydantic and the annotation modules; ``pyarrow``
inside the parquet helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.config import (
    MOUSE_CCF_REGIONS,
    MouseRegionConfig,
    normalise_mouse_section_regions,
)
from merxen.annotation.mapmycells_engine import coerce_tidy_dtypes, level_frame
from merxen.annotation.reference import (
    REGION_HOME_FILE,
    REGION_SHARE_FILE,
    LookupValidation,
    TaxonomyTreeView,
    validate_lookup,
)
from merxen.annotation.schema import meets_threshold
from merxen.annotation.store import BUNDLE_MANIFEST_NAME, BUNDLE_STATUS_COMPLETE
from merxen.annotation.vocab import NEURONS, primary_vocab

logger = logging.getLogger(__name__)

REGION_SHARE_REFERENCE_ID: Final = "wmb_region_share"
RULE_VARIANTS: Final[tuple[str, ...]] = ("v1",)
REGION_CELLS_SUFFIX: Final = "_mouse_regions.parquet"
REGION_CELLS_METADATA_KEY: Final = b"merxen_mouse_regions"
REGION_CELLS_SCHEMA_VERSION: Final = 1
# The re-map run id: the primary run id + this suffix (its tidy parquet holds
# the re-mapped cells only).
REMAP_RUN_SUFFIX: Final = "_pruned"
UNPRUNED_TOKEN: Final = "unpruned"

RegionSource = Literal["auto", "override", "none"]
RegionStepStatus = Literal[
    "pruned",
    "no_nodes_dropped",
    "skipped_few_tiles",
    "skipped_no_coordinates",
    "disabled",
]
DropKind = Literal["class", "subclass"]
DropRule = Literal[
    "all_subclasses_dropped",
    "absent_class_low_share",
    "absent_class_small_subclass",
    "low_share",
]

# Per-cell columns of <sid>_mouse_regions.parquet.
CELL_ID: Final = "cell_id"
TILE_I: Final = "tile_i"
TILE_J: Final = "tile_j"
TILE_REGION: Final = "tile_region"
INFERRED_REGION: Final = "inferred_region"
CONFIDENT_NEURON: Final = "confident_neuron"
REGION_DROPPED_LEVEL: Final = "region_dropped_level"
REGION_PRUNED_CHANGED: Final = "region_pruned_changed"
REGION_CELL_COLUMNS: Final[tuple[str, ...]] = (
    CELL_ID,
    TILE_I,
    TILE_J,
    TILE_REGION,
    INFERRED_REGION,
    CONFIDENT_NEURON,
    REGION_DROPPED_LEVEL,
    REGION_PRUNED_CHANGED,
)


class MouseRegionError(ValueError):
    """The mouse region step cannot run on its inputs."""


# --------------------------------------------------------------------------
# The per-sample request (samplesheet column / param)


@dataclass(frozen=True)
class SectionRegionsRequest:
    """What ``mouse_section_regions`` asks for one section.

    Attributes:
        value: The canonical value (``auto``, ``none`` or divisions).
        source: ``auto`` (infer), ``override`` (the listed divisions) or
            ``none`` (no pruning).
        regions: The override's divisions (empty otherwise).
    """

    value: str
    source: RegionSource
    regions: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> SectionRegionsRequest:
        """Parse a ``mouse_section_regions`` value.

        Args:
            value: ``auto``, ``none`` or ``;``-separated CCF divisions (case
                insensitive; ``normalise_mouse_section_regions``).

        Returns:
            The request.

        Raises:
            ValueError: On a blank value or an unknown division.
        """
        canonical = normalise_mouse_section_regions(value)
        if canonical == "auto":
            return cls(value=canonical, source="auto")
        if canonical == "none":
            return cls(value=canonical, source="none")
        return cls(
            value=canonical, source="override", regions=tuple(canonical.split(";"))
        )

    @property
    def needs_region_shares(self) -> bool:
        """Whether the request needs the ``wmb_region_share`` bundle."""
        return self.source != "none"


def check_rule_variant(config: MouseRegionConfig) -> None:
    """Refuse region-rule settings this version does not implement.

    Args:
        config: The mouse region settings.

    Raises:
        MouseRegionError: For a rule variant other than ``v1`` or any
            ``coupled_regions`` (both M6b, plan §7.8).
    """
    if config.rule_variant not in RULE_VARIANTS:
        raise MouseRegionError(
            f"mouse region rule variant {config.rule_variant!r} is not "
            f"implemented (known: {RULE_VARIANTS}); M6b selects the variants "
            "(plan §7.8)"
        )
    if config.coupled_regions:
        raise MouseRegionError(
            "coupled_regions is an M6b rule variant (plan §7.8, variant (a)) and "
            "is not implemented in v1; leave it empty"
        )


# --------------------------------------------------------------------------
# Region shares (wmb_region_share bundle)


@dataclass(frozen=True)
class RegionShares:
    """MERFISH grey-matter division shares of the WMB classes and subclasses.

    Attributes:
        class_share: Class name x division (``MOUSE_CCF_REGIONS``): share of
            the class's grey-matter MERFISH cells.
        class_n_grey: Grey-matter MERFISH cells per class.
        subclass_share: The same per subclass.
        subclass_n_grey: Grey-matter MERFISH cells per subclass.
        subclass_home: Home division per subclass (largest share).
    """

    class_share: pd.DataFrame
    class_n_grey: pd.Series
    subclass_share: pd.DataFrame
    subclass_n_grey: pd.Series
    subclass_home: pd.Series

    @classmethod
    def from_tables(
        cls, region_share: pd.DataFrame, region_home: pd.DataFrame
    ) -> RegionShares:
        """Build the shares from the bundle's long tables.

        Args:
            region_share: ``region_share.parquet`` (level, node_name, region,
                n, share, n_grey, n_all).
            region_home: ``region_home.parquet`` (level, node_name, home,
                home_share, n_grey).

        Returns:
            The shares.

        Raises:
            MouseRegionError: If a table lacks a column or a level.
        """
        needed = {"level", "node_name", "region", "share", "n_grey"}
        missing = sorted(needed - set(region_share.columns))
        if missing:
            raise MouseRegionError(f"region_share table lacks columns {missing}")
        if not {"level", "node_name", "home"} <= set(region_home.columns):
            raise MouseRegionError("region_home table lacks level/node_name/home")
        frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
        for level in ("class", "subclass"):
            rows = region_share[region_share["level"].astype(str) == level]
            if rows.empty:
                raise MouseRegionError(f"region_share table has no {level} rows")
            share = (
                rows.pivot_table(
                    index="node_name",
                    columns="region",
                    values="share",
                    aggfunc="first",
                )
                .reindex(columns=list(MOUSE_CCF_REGIONS))
                .fillna(0.0)
                .astype(np.float64)
            )
            share.index = share.index.astype(str)
            n_grey = (
                rows.drop_duplicates("node_name")
                .set_index("node_name")["n_grey"]
                .astype(np.int64)
            )
            n_grey.index = n_grey.index.astype(str)
            frames[level] = (share, n_grey.reindex(share.index).fillna(0))
        homes = region_home[region_home["level"].astype(str) == "subclass"]
        home = homes.set_index("node_name")["home"].astype(str)
        home.index = home.index.astype(str)
        return cls(
            class_share=frames["class"][0],
            class_n_grey=frames["class"][1].astype(np.int64),
            subclass_share=frames["subclass"][0],
            subclass_n_grey=frames["subclass"][1].astype(np.int64),
            subclass_home=home,
        )

    def present_share(self, level: DropKind, present: Iterable[str]) -> pd.Series:
        """Return each node's share of grey-matter cells in present divisions.

        Args:
            level: ``class`` or ``subclass``.
            present: Present divisions.

        Returns:
            Share per node name.
        """
        regions = [region for region in MOUSE_CCF_REGIONS if region in set(present)]
        table = self.class_share if level == "class" else self.subclass_share
        if not regions:
            return pd.Series(0.0, index=table.index)
        return table[regions].sum(axis=1)


@dataclass(frozen=True)
class RegionShareBundle:
    """A complete ``wmb_region_share`` bundle.

    Attributes:
        path: Bundle directory.
        build_hash: Its ``build_hash``.
        reference_id: ``wmb_region_share``.
        shares: Its region shares and homes.
    """

    path: Path
    build_hash: str
    reference_id: str
    shares: RegionShares

    @classmethod
    def from_dir(cls, path: Path | str) -> RegionShareBundle:
        """Read a complete region-share bundle directory.

        Every file ``bundle.json`` lists must exist with its recorded size.

        Args:
            path: ``<store>/wmb_region_share/<build_hash>``.

        Returns:
            The bundle.

        Raises:
            MouseRegionError: If the directory is not a complete, intact
                ``wmb_region_share`` bundle.
        """
        bundle_dir = Path(path)
        manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
        try:
            manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MouseRegionError(f"cannot read {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise MouseRegionError(f"{manifest_path} does not hold a JSON object")
        if manifest.get("status") != BUNDLE_STATUS_COMPLETE:
            raise MouseRegionError(f"bundle {bundle_dir} is not complete")
        reference_id = str(manifest.get("reference_id"))
        if reference_id != REGION_SHARE_REFERENCE_ID:
            raise MouseRegionError(
                f"bundle {bundle_dir} is a {reference_id} bundle, not "
                f"{REGION_SHARE_REFERENCE_ID}"
            )
        problems = []
        for record in manifest.get("files", []) or []:
            file_path = bundle_dir / str(record.get("path", ""))
            if not file_path.is_file() or file_path.is_symlink():
                problems.append(f"missing {record.get('path')!r}")
            elif file_path.stat().st_size != record.get("size"):
                problems.append(f"size of {record.get('path')!r} changed")
        if problems:
            raise MouseRegionError(
                f"bundle {bundle_dir} is not intact: {'; '.join(problems)}"
            )
        shares = RegionShares.from_tables(
            pd.read_parquet(bundle_dir / REGION_SHARE_FILE),
            pd.read_parquet(bundle_dir / REGION_HOME_FILE),
        )
        return cls(
            path=bundle_dir,
            build_hash=str(manifest.get("build_hash")),
            reference_id=reference_id,
            shares=shares,
        )

    def provenance(self) -> dict[str, str]:
        """Return the bundle's identity for ``map_manifest.json``.

        Returns:
            ``reference_id``, ``build_hash`` (so ``annotation-store prune``
            counts it as referenced) and ``path``.
        """
        return {
            "reference_id": self.reference_id,
            "build_hash": self.build_hash,
            "path": str(self.path),
        }


# --------------------------------------------------------------------------
# WMB taxonomy (class -> subclasses; names <-> labels)


@dataclass(frozen=True)
class WmbTaxonomy:
    """The class and subclass levels of a WMB mapping tree.

    Attributes:
        class_level: Class level key (``CCN20230722_CLAS``).
        subclass_level: Subclass level key (``CCN20230722_SUBC``).
        class_label: Class name to label.
        subclass_label: Subclass name to label.
        subclasses: Class name to its subclass names (sorted).
    """

    class_level: str
    subclass_level: str
    class_label: dict[str, str]
    subclass_label: dict[str, str]
    subclasses: dict[str, tuple[str, ...]]

    @classmethod
    def from_tree(
        cls, tree: TaxonomyTreeView, *, class_level: str, subclass_level: str
    ) -> WmbTaxonomy:
        """Read the two levels from a mapping tree.

        Args:
            tree: The bundle's mapping tree (supertype level dropped).
            class_level: Class level key.
            subclass_level: Subclass level key; must be the class level's
                child level.

        Returns:
            The taxonomy.

        Raises:
            MouseRegionError: If the levels are not adjacent in the tree or
                two nodes share a name.
        """
        hierarchy = tree.hierarchy
        if (
            class_level not in hierarchy
            or subclass_level not in hierarchy
            or hierarchy.index(subclass_level) != hierarchy.index(class_level) + 1
        ):
            raise MouseRegionError(
                f"the mapping tree {hierarchy} has no {class_level} -> "
                f"{subclass_level} levels"
            )
        class_label: dict[str, str] = {}
        subclass_label: dict[str, str] = {}
        subclasses: dict[str, tuple[str, ...]] = {}
        for label in tree.nodes(class_level):
            name = tree.name(class_level, label)
            if name in class_label:
                raise MouseRegionError(f"two classes are named {name!r}")
            class_label[name] = label
            children = []
            for child in tree.children_of(class_level, label):
                child_name = tree.name(subclass_level, child)
                if child_name in subclass_label:
                    raise MouseRegionError(f"two subclasses are named {child_name!r}")
                subclass_label[child_name] = child
                children.append(child_name)
            subclasses[name] = tuple(sorted(children))
        return cls(
            class_level=class_level,
            subclass_level=subclass_level,
            class_label=class_label,
            subclass_label=subclass_label,
            subclasses=subclasses,
        )


# --------------------------------------------------------------------------
# Region inference


def confident_neurons(
    class_names: Sequence[Any] | np.ndarray,
    class_bp: Sequence[float] | np.ndarray,
    subclass_bp: Sequence[float] | np.ndarray,
    *,
    min_bp: float,
) -> np.ndarray:
    """Return the confident neurons of E7's region rule.

    Args:
        class_names: Unpruned WMB class name per cell.
        class_bp: Class bootstrap probability per cell.
        subclass_bp: Subclass bootstrap probability per cell.
        min_bp: ``confident_neuron_min_bp`` (0.9); float32-tolerant.

    Returns:
        Boolean mask: a neuronal class (vocab broad class ``Neurons``, i.e.
        classes 01-29) with class and subclass bp >= ``min_bp``.
    """
    vocab = primary_vocab("mouse")
    cache: dict[str, bool] = {}

    def is_neuron(name: Any) -> bool:
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return False
        key = str(name)
        if key not in cache:
            cache[key] = key in vocab and vocab.broad_class(key) == NEURONS
        return cache[key]

    neuron = np.array([is_neuron(name) for name in class_names], dtype=bool)
    confident = (
        neuron
        & meets_threshold(np.asarray(class_bp, dtype=np.float64), min_bp)
        & meets_threshold(np.asarray(subclass_bp, dtype=np.float64), min_bp)
    )
    return np.asarray(confident, dtype=bool)


def _largest_component(tiles: set[tuple[int, int]]) -> int:
    """Return the size of the largest 8-connected component of a tile set."""
    seen: set[tuple[int, int]] = set()
    largest = 0
    for start in tiles:
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        size = 0
        while stack:
            i, j = stack.pop()
            size += 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    neighbour = (i + di, j + dj)
                    if neighbour in tiles and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        largest = max(largest, size)
    return largest


@dataclass(frozen=True)
class RegionInference:
    """E7's section-region inference on one section.

    Attributes:
        n_confident_neurons: Confident neurons (with finite coordinates).
        n_voting_neurons: Those whose subclass has a MERFISH home.
        n_assigned_tiles: Tiles with >= ``min_neurons_per_tile`` voters.
        tile_counts: Assigned tiles per majority division.
        largest_component: Largest 8-connected component per division.
        inferred_regions: Present divisions (``MOUSE_CCF_REGIONS`` order).
        tiles: One row per assigned tile (``tile_i``, ``tile_j``,
            ``region``, ``n_neurons``).
        cell_tile: Tile index per cell (``(n, 2)`` int64; ``-1`` rows for
            cells without finite coordinates).
        cell_tile_region: Majority division of each cell's tile, ``None``
            when the tile is not assigned.
    """

    n_confident_neurons: int
    n_voting_neurons: int
    n_assigned_tiles: int
    tile_counts: dict[str, int]
    largest_component: dict[str, int]
    inferred_regions: tuple[str, ...]
    tiles: pd.DataFrame
    cell_tile: np.ndarray
    cell_tile_region: np.ndarray


def infer_regions(
    xy: np.ndarray,
    subclass_names: Sequence[Any] | np.ndarray,
    confident: np.ndarray,
    shares: RegionShares,
    config: MouseRegionConfig,
) -> RegionInference:
    """Infer a section's present divisions from its confident neurons.

    Args:
        xy: ``(n, 2)`` cell coordinates in µm.
        subclass_names: Unpruned WMB subclass name per cell.
        confident: ``confident_neurons`` mask.
        shares: The region shares (subclass homes).
        config: Tile size, neurons per tile, presence thresholds.

    Returns:
        The inference.

    Raises:
        MouseRegionError: If the arrays do not fit together.
    """
    coordinates = np.asarray(xy, dtype=np.float64)
    n = len(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise MouseRegionError("coordinates must be an (n, 2) array")
    names = np.asarray(subclass_names, dtype=object)
    mask = np.asarray(confident, dtype=bool)
    if len(names) != n or len(mask) != n:
        raise MouseRegionError(
            f"{n} coordinates, {len(names)} subclass names, {len(mask)} flags"
        )
    finite = np.isfinite(coordinates[:, :2]).all(axis=1)
    cell_tile = np.full((n, 2), -1, dtype=np.int64)
    cell_tile[finite] = np.floor(coordinates[finite, :2] / config.tile_um).astype(
        np.int64
    )
    home = shares.subclass_home.reindex(
        [None if name is None else str(name) for name in names]
    ).to_numpy(dtype=object)
    has_home = np.array([value is not None and value == value for value in home])
    voters = mask & finite & has_home
    empty_tiles = pd.DataFrame(
        {
            TILE_I: pd.Series(dtype=np.int64),
            TILE_J: pd.Series(dtype=np.int64),
            "region": pd.Series(dtype=object),
            "n_neurons": pd.Series(dtype=np.int64),
        }
    )
    cell_tile_region = np.full(n, None, dtype=object)
    if not voters.any():
        return RegionInference(
            n_confident_neurons=int((mask & finite).sum()),
            n_voting_neurons=0,
            n_assigned_tiles=0,
            tile_counts={},
            largest_component={},
            inferred_regions=(),
            tiles=empty_tiles,
            cell_tile=cell_tile,
            cell_tile_region=cell_tile_region,
        )
    votes = pd.DataFrame(
        {
            TILE_I: cell_tile[voters, 0],
            TILE_J: cell_tile[voters, 1],
            "home": home[voters].astype(str),
        }
    )
    counts = votes.groupby([TILE_I, TILE_J, "home"]).size().unstack(fill_value=0)
    # E7 takes DataFrame.idxmax over the sorted home columns: a tie goes to
    # the first division in sorted order.
    counts = counts.reindex(columns=sorted(counts.columns))
    totals = counts.sum(axis=1)
    counts = counts[totals >= config.min_neurons_per_tile]
    if counts.empty:
        tiles = empty_tiles
    else:
        values = counts.to_numpy()
        majority = np.asarray(counts.columns, dtype=object)[values.argmax(axis=1)]
        index = counts.index.to_frame(index=False)
        tiles = pd.DataFrame(
            {
                TILE_I: index[TILE_I].to_numpy(np.int64),
                TILE_J: index[TILE_J].to_numpy(np.int64),
                "region": majority,
                "n_neurons": values.sum(axis=1).astype(np.int64),
            }
        )
    n_assigned = len(tiles)
    tile_counts = {
        str(region): int(count)
        for region, count in tiles["region"].value_counts().items()
    }
    largest: dict[str, int] = {}
    present: list[str] = []
    for region, count in tile_counts.items():
        cells = {
            (int(i), int(j))
            for i, j, name in zip(
                tiles[TILE_I], tiles[TILE_J], tiles["region"], strict=True
            )
            if name == region
        }
        largest[region] = _largest_component(cells)
        if count / n_assigned < config.min_tile_fraction:
            continue
        if largest[region] >= config.min_component_tiles:
            present.append(region)
    order = {region: index for index, region in enumerate(MOUSE_CCF_REGIONS)}
    inferred = tuple(sorted(present, key=lambda region: order.get(region, 99)))
    if n_assigned:
        lookup = {
            (int(i), int(j)): str(region)
            for i, j, region in zip(
                tiles[TILE_I], tiles[TILE_J], tiles["region"], strict=True
            )
        }
        for index in np.flatnonzero(finite):
            cell_tile_region[index] = lookup.get(
                (int(cell_tile[index, 0]), int(cell_tile[index, 1]))
            )
    return RegionInference(
        n_confident_neurons=int((mask & finite).sum()),
        n_voting_neurons=int(voters.sum()),
        n_assigned_tiles=n_assigned,
        tile_counts=dict(sorted(tile_counts.items())),
        largest_component=dict(sorted(largest.items())),
        inferred_regions=inferred,
        tiles=tiles,
        cell_tile=cell_tile,
        cell_tile_region=cell_tile_region,
    )


# --------------------------------------------------------------------------
# Two-tier drop rule


class DroppedNode(BaseModel):
    """One node of the pruned tree's drop list (``--nodes_to_drop``).

    Attributes:
        level: Taxonomy level key.
        node: Node label.
        name: Node name.
        kind: ``class`` or ``subclass``.
        class_name: The class (itself for a class node).
        present_share: The node's MERFISH grey-matter share in the present
            divisions (a class node: the class's share).
        n_merfish_grey: Its MERFISH grey-matter cells.
        rule: Why it is dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: str
    node: str
    name: str
    kind: DropKind
    class_name: str
    present_share: float
    n_merfish_grey: int
    rule: DropRule

    def as_pair(self) -> tuple[str, str]:
        """Return ``(level, node)`` for ``--nodes_to_drop``."""
        return (self.level, self.node)


def two_tier_drop_list(
    present: Iterable[str],
    shares: RegionShares,
    taxonomy: WmbTaxonomy,
    config: MouseRegionConfig,
) -> list[DroppedNode]:
    """Return the nodes the two-tier rule drops for a set of present divisions.

    Args:
        present: Present divisions.
        shares: MERFISH region shares.
        taxonomy: The mapping tree's classes and subclasses.
        config: Thresholds and the never-drop classes.

    Returns:
        Non-overlapping nodes (a class replaces its subclasses when all are
        dropped), by class name, subclasses sorted by name.
    """
    present_set = set(present)
    class_share = shares.present_share("class", present_set)
    subclass_share = shares.present_share("subclass", present_set)
    never = set(config.never_drop_classes)
    nodes: list[DroppedNode] = []
    for class_name in sorted(taxonomy.subclasses):
        if class_name in never or class_name not in shares.class_share.index:
            continue
        share = float(class_share[class_name])
        n_class = int(shares.class_n_grey.get(class_name, 0))
        absent = share < config.class_low_share and (
            n_class >= config.min_merfish_cells_class
        )
        threshold = (
            config.subclass_share_in_low_class
            if absent
            else config.subclass_share_default
        )
        members = taxonomy.subclasses[class_name]
        dropped: list[DroppedNode] = []
        for subclass in members:
            n_sub = int(shares.subclass_n_grey.get(subclass, 0))
            sub_share = (
                float(subclass_share[subclass])
                if subclass in subclass_share.index
                else 0.0
            )
            rule: DropRule | None = None
            if n_sub >= config.min_merfish_cells_subclass:
                if sub_share < threshold:
                    rule = "absent_class_low_share" if absent else "low_share"
            elif absent:
                rule = "absent_class_small_subclass"
            if rule is not None:
                dropped.append(
                    DroppedNode(
                        level=taxonomy.subclass_level,
                        node=taxonomy.subclass_label[subclass],
                        name=subclass,
                        kind="subclass",
                        class_name=class_name,
                        present_share=sub_share,
                        n_merfish_grey=n_sub,
                        rule=rule,
                    )
                )
        if not dropped:
            continue
        if len(dropped) == len(members):
            nodes.append(
                DroppedNode(
                    level=taxonomy.class_level,
                    node=taxonomy.class_label[class_name],
                    name=class_name,
                    kind="class",
                    class_name=class_name,
                    present_share=share,
                    n_merfish_grey=n_class,
                    rule="all_subclasses_dropped",
                )
            )
        else:
            nodes.extend(dropped)
    return nodes


def pruned_tree(
    tree: TaxonomyTreeView, nodes: Sequence[DroppedNode]
) -> TaxonomyTreeView:
    """Return the mapping tree without the dropped nodes.

    Args:
        tree: The bundle's mapping tree.
        nodes: The drop list.

    Returns:
        The pruned tree (``TaxonomyTreeView.drop_nodes``).
    """
    return tree.drop_nodes([node.as_pair() for node in nodes])


def pruned_lookup(
    lookup: Mapping[str, Any],
    tree: TaxonomyTreeView,
    nodes: Sequence[DroppedNode],
    query_genes: Iterable[str],
) -> LookupValidation:
    """Return the marker lookup of the pruned re-map.

    The keys of dropped nodes (and of their descendants) are removed
    (``filter_lookup_to_tree``), markers are restricted to the query and
    parents left without one are collapsed (``validate_lookup``), so
    ``cell_type_mapper`` meets neither a key its tree lacks nor a
    zero-marker parent (plan §7.2 step 3, R15).

    Args:
        lookup: The bundle's lookup (filtered to the unpruned mapping tree).
        tree: The unpruned mapping tree.
        nodes: The drop list.
        query_genes: Gene IDs of the re-map query.

    Returns:
        The validated lookup.

    Raises:
        LookupValidationError: If the root keeps no marker in the query.
    """
    return validate_lookup(lookup, pruned_tree(tree, nodes), query_genes)


def dropped_levels(
    class_labels: Sequence[Any] | np.ndarray,
    subclass_labels: Sequence[Any] | np.ndarray,
    nodes: Sequence[DroppedNode],
) -> np.ndarray:
    """Return, per cell, the level at which its unpruned call was dropped.

    Args:
        class_labels: Unpruned class label per cell.
        subclass_labels: Unpruned subclass label per cell.
        nodes: The drop list.

    Returns:
        ``"class"`` (its class was dropped whole), ``"subclass"`` (its
        subclass was dropped) or ``None`` per cell.
    """
    classes = {node.node for node in nodes if node.kind == "class"}
    subclasses = {node.node for node in nodes if node.kind == "subclass"}
    result = np.full(len(class_labels), None, dtype=object)
    for index, (class_label, subclass_label) in enumerate(
        zip(class_labels, subclass_labels, strict=True)
    ):
        if class_label is not None and str(class_label) in classes:
            result[index] = "class"
        elif subclass_label is not None and str(subclass_label) in subclasses:
            result[index] = "subclass"
    return result


# --------------------------------------------------------------------------
# The per-section decision


@dataclass(frozen=True)
class RegionPlan:
    """What the region step decided for one section (before the re-map).

    Attributes:
        request: The ``mouse_section_regions`` request.
        status: The step status.
        reasons: Warnings (skip reasons, missed inference).
        inference: The inference, when coordinates allowed one.
        present_regions: Divisions the drop list used (empty when not
            pruned).
        nodes: The drop list (empty when not pruned).
        cell_ids: Table cells, in tidy order.
        dropped_level: Per table cell, ``class`` / ``subclass`` / ``None``.
        confident: Per table cell, whether it is a confident neuron (all
            false without an inference).
    """

    request: SectionRegionsRequest
    status: RegionStepStatus
    reasons: tuple[str, ...]
    inference: RegionInference | None
    present_regions: tuple[str, ...]
    nodes: tuple[DroppedNode, ...]
    cell_ids: pd.Index
    dropped_level: np.ndarray
    confident: np.ndarray

    @property
    def remap_cell_ids(self) -> pd.Index:
        """Return the table cells whose unpruned call was dropped."""
        keep = np.array([value is not None for value in self.dropped_level])
        return self.cell_ids[keep]


def plan_region_step(
    request: SectionRegionsRequest,
    *,
    cell_ids: Sequence[str] | pd.Index,
    class_names: Sequence[Any] | np.ndarray,
    class_labels: Sequence[Any] | np.ndarray,
    class_bp: Sequence[float] | np.ndarray,
    subclass_names: Sequence[Any] | np.ndarray,
    subclass_labels: Sequence[Any] | np.ndarray,
    subclass_bp: Sequence[float] | np.ndarray,
    xy: np.ndarray | None,
    shares: RegionShares | None,
    taxonomy: WmbTaxonomy,
    config: MouseRegionConfig,
) -> RegionPlan:
    """Decide the region step of one section from its unpruned mapping.

    ``auto`` infers the present divisions (skipped with a warning without
    coordinates or with fewer than ``min_assigned_tiles`` assigned tiles);
    an override uses its divisions (the inference still runs, for QC, when
    coordinates exist); ``none`` disables pruning.

    Args:
        request: The section's ``mouse_section_regions`` request.
        cell_ids: Table cells (the unpruned tidy's cells).
        class_names: Unpruned class name per cell.
        class_labels: Unpruned class label per cell.
        class_bp: Class bootstrap probability per cell.
        subclass_names: Unpruned subclass name per cell.
        subclass_labels: Unpruned subclass label per cell.
        subclass_bp: Subclass bootstrap probability per cell.
        xy: ``(n, 2)`` µm coordinates of the table cells, or ``None``.
        shares: Region shares (required unless the request is ``none``).
        taxonomy: The mapping tree's classes and subclasses.
        config: Mouse region settings.

    Returns:
        The plan.

    Raises:
        MouseRegionError: If shares are missing for a request that needs
            them, or the rule variant is not implemented.
    """
    check_rule_variant(config)
    ids = pd.Index([str(cell) for cell in cell_ids])
    n = len(ids)
    no_drop = np.full(n, None, dtype=object)
    reasons: list[str] = []
    inference: RegionInference | None = None
    confident = np.zeros(n, dtype=bool)
    if shares is not None and xy is not None:
        coordinates = np.asarray(xy, dtype=np.float64)
        if coordinates.shape[0] != n:
            raise MouseRegionError(f"{coordinates.shape[0]} coordinates for {n} cells")
        confident = confident_neurons(
            class_names, class_bp, subclass_bp, min_bp=config.confident_neuron_min_bp
        )
        inference = infer_regions(
            coordinates, subclass_names, confident, shares, config
        )
    if request.source == "none":
        return RegionPlan(
            request=request,
            status="disabled",
            reasons=("mouse_section_regions is none: pruning disabled",),
            inference=inference,
            present_regions=(),
            nodes=(),
            cell_ids=ids,
            dropped_level=no_drop,
            confident=confident,
        )
    if shares is None:
        raise MouseRegionError(
            f"mouse_section_regions {request.value!r} needs the "
            f"{REGION_SHARE_REFERENCE_ID} bundle"
        )
    if request.source == "auto":
        if inference is None:
            reasons.append("no cell coordinates: region inference and pruning skipped")
            return RegionPlan(
                request=request,
                status="skipped_no_coordinates",
                reasons=tuple(reasons),
                inference=None,
                present_regions=(),
                nodes=(),
                cell_ids=ids,
                dropped_level=no_drop,
                confident=confident,
            )
        if inference.n_assigned_tiles < config.min_assigned_tiles:
            reasons.append(
                f"{inference.n_assigned_tiles} assigned tiles (< "
                f"{config.min_assigned_tiles}): pruning skipped (small crop or "
                "few confident neurons; give mouse_section_regions to prune)"
            )
            return RegionPlan(
                request=request,
                status="skipped_few_tiles",
                reasons=tuple(reasons),
                inference=inference,
                present_regions=(),
                nodes=(),
                cell_ids=ids,
                dropped_level=no_drop,
                confident=confident,
            )
        present = inference.inferred_regions
    else:
        present = request.regions
        if inference is not None and set(inference.inferred_regions) != set(present):
            reasons.append(
                "the override differs from the inferred divisions "
                f"({';'.join(inference.inferred_regions) or 'none'})"
            )
    nodes = tuple(two_tier_drop_list(present, shares, taxonomy, config))
    levels = dropped_levels(class_labels, subclass_labels, nodes)
    return RegionPlan(
        request=request,
        status="pruned" if nodes else "no_nodes_dropped",
        reasons=tuple(reasons),
        inference=inference,
        present_regions=tuple(present),
        nodes=nodes,
        cell_ids=ids,
        dropped_level=levels,
        confident=confident,
    )


# --------------------------------------------------------------------------
# Tidy tables


def merge_pruned_tidy(unpruned: pd.DataFrame, remap: pd.DataFrame) -> pd.DataFrame:
    """Return the pruned tidy table: re-mapped cells replace their rows.

    Args:
        unpruned: The unpruned run's tidy table (every table cell).
        remap: The pruned re-map's tidy table (the re-mapped cells).

    Returns:
        A tidy table with the unpruned rows of the other cells and the
        re-mapped rows, in the unpruned cell and level order.

    Raises:
        MouseRegionError: If the re-map names cells absent from the unpruned
            table.
    """
    if remap.empty:
        return unpruned.copy()
    unpruned_cells = pd.Index(unpruned["cell_id"].astype(str).unique())
    remap_cells = pd.Index(remap["cell_id"].astype(str).unique())
    unknown = remap_cells.difference(unpruned_cells)
    if len(unknown):
        raise MouseRegionError(
            f"the re-map names {len(unknown)} cells absent from the unpruned "
            f"run (e.g. {list(unknown[:3])})"
        )
    kept = unpruned[~unpruned["cell_id"].astype(str).isin(set(remap_cells))]
    columns = list(unpruned.columns)
    merged = pd.concat(
        [kept.astype(object), remap.reindex(columns=columns).astype(object)],
        ignore_index=True,
    )
    cell_order = {cell: index for index, cell in enumerate(unpruned_cells)}
    level_order = {
        level: index
        for index, level in enumerate(unpruned["level"].astype(str).unique())
    }
    keys = np.lexsort(
        (
            merged["level"].astype(str).map(level_order).fillna(len(level_order)),
            merged["cell_id"].astype(str).map(cell_order),
        )
    )
    return coerce_tidy_dtypes(merged.iloc[keys].reset_index(drop=True))


def changed_cells(
    unpruned: pd.DataFrame,
    pruned: pd.DataFrame,
    *,
    class_level: str,
    subclass_level: str,
    cell_ids: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """Return which cells' class and subclass labels the pruning changed.

    Args:
        unpruned: Unpruned tidy table.
        pruned: Pruned tidy table (``merge_pruned_tidy``).
        class_level: Class level key.
        subclass_level: Subclass level key.
        cell_ids: Cells to compare.

    Returns:
        ``(class_changed, subclass_changed)`` boolean arrays.
    """
    result = []
    for level in (class_level, subclass_level):
        before = level_frame(unpruned, level).reindex(cell_ids)["assignment"]
        after = level_frame(pruned, level).reindex(cell_ids)["assignment"]
        result.append(
            before.astype(object).to_numpy() != after.astype(object).to_numpy()
        )
    return result[0].astype(bool), result[1].astype(bool)


# --------------------------------------------------------------------------
# Records (map_manifest.json)


class _RegionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegionRemapRecord(_RegionModel):
    """The pruned re-map of one section (plan §7.2 step 4).

    Attributes:
        run_id: ``<primary run id>_pruned``.
        reference_id: The primary reference (``wmb_panel``).
        build_hash: The bundle the re-map used (the primary run's).
        bundle_path: Its directory.
        parquet: Tidy parquet of the re-mapped cells, relative to the
            manifest.
        parquet_sha256: Its sha256.
        n_cells: Re-mapped cells.
        n_query_genes: Query genes.
        query_fingerprint: ``query_fingerprint`` of the re-map query (reuse
            key).
        lookup_sha256: Marker-content sha256 of the pruned lookup.
        collapsed_parents: Parents the pruned lookup collapsed.
        nodes_to_drop: ``[level, node]`` pairs passed to ``--nodes_to_drop``.
        engine_params: ``MmcEngineParams`` (reuse key).
        ctm_version: ctm version of the re-map.
        ctm_commit: ctm commit, when known.
        tidy_schema_version: ``TIDY_SCHEMA_VERSION`` of the parquet.
        wall_s: Mapper wall time.
        peak_rss_gb: Mapper peak RSS, when measured.
        extended_json: Kept gzipped extended JSON (relative), if any.
        reused: Whether the re-map was copied from a published manifest.
        reused_from: The published parquet it came from.
    """

    run_id: str
    reference_id: str
    build_hash: str
    bundle_path: str
    parquet: str
    parquet_sha256: str
    n_cells: int
    n_query_genes: int
    query_fingerprint: str
    lookup_sha256: str
    collapsed_parents: list[str] = Field(default_factory=list)
    nodes_to_drop: list[list[str]]
    engine_params: dict[str, Any]
    ctm_version: str | None
    ctm_commit: str | None = None
    tidy_schema_version: int | None = None
    wall_s: float | None = None
    peak_rss_gb: float | None = None
    extended_json: str | None = None
    reused: bool = False
    reused_from: str | None = None


class MouseRegionRecord(_RegionModel):
    """The region step of one mouse section in ``map_manifest.json``.

    Attributes:
        rule_variant: ``MouseRegionConfig.rule_variant``.
        run_id: The primary run the step read (unpruned).
        requested: The effective ``mouse_section_regions`` value.
        source: ``auto``, ``override`` or ``none``.
        status: ``pruned``, ``no_nodes_dropped``, ``skipped_few_tiles``,
            ``skipped_no_coordinates`` or ``disabled``.
        reasons: Warnings.
        params: The ``MouseRegionConfig`` used.
        region_share: The region-share bundle (``reference_id``,
            ``build_hash``, ``path``), when used.
        n_table_cells: Table cells of the section.
        n_confident_neurons: Confident neurons with coordinates.
        n_voting_neurons: Those whose subclass has a MERFISH home.
        n_assigned_tiles: Assigned tiles.
        tile_counts: Assigned tiles per majority division.
        largest_component: Largest 8-connected component per division.
        inferred_regions: Divisions the inference found present.
        present_regions: Divisions the drop list used.
        nodes_to_drop: The drop list.
        n_dropped_classes: Class nodes dropped.
        n_dropped_subclasses: Subclasses dropped (including those of dropped
            classes).
        n_cells_region_dropped: Table cells whose unpruned class or subclass
            was dropped (re-mapped).
        n_cells_region_dropped_class: Those whose class was dropped whole.
        n_cells_class_changed: Table cells whose class the pruning changed.
        n_cells_subclass_changed: Table cells whose subclass it changed.
        remap: The pruned re-map (``None`` when no cell was re-mapped).
        cells_parquet: ``<sid>_mouse_regions.parquet`` (relative).
        cells_parquet_sha256: Its sha256.
    """

    rule_variant: str
    run_id: str
    requested: str
    source: RegionSource
    status: RegionStepStatus
    reasons: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    region_share: dict[str, str] | None = None
    n_table_cells: int
    n_confident_neurons: int = 0
    n_voting_neurons: int = 0
    n_assigned_tiles: int = 0
    tile_counts: dict[str, int] = Field(default_factory=dict)
    largest_component: dict[str, int] = Field(default_factory=dict)
    inferred_regions: list[str] = Field(default_factory=list)
    present_regions: list[str] = Field(default_factory=list)
    nodes_to_drop: list[DroppedNode] = Field(default_factory=list)
    n_dropped_classes: int = 0
    n_dropped_subclasses: int = 0
    n_cells_region_dropped: int = 0
    n_cells_region_dropped_class: int = 0
    n_cells_class_changed: int = 0
    n_cells_subclass_changed: int = 0
    remap: RegionRemapRecord | None = None
    cells_parquet: str
    cells_parquet_sha256: str

    @property
    def is_pruned(self) -> bool:
        """Whether the section was pruned (a drop list was applied)."""
        return self.status == "pruned"


def count_dropped_subclasses(
    nodes: Sequence[DroppedNode], taxonomy: WmbTaxonomy
) -> int:
    """Return the subclasses a drop list removes (a class node: all of its)."""
    total = 0
    for node in nodes:
        total += len(taxonomy.subclasses[node.name]) if node.kind == "class" else 1
    return total


# --------------------------------------------------------------------------
# Per-cell parquet


def region_cells_frame(
    plan: RegionPlan,
    *,
    class_changed: np.ndarray | None = None,
    subclass_changed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return the per-cell region table of one section (table cells).

    Args:
        plan: The section's plan.
        class_changed: Per cell, whether the pruning changed its class.
        subclass_changed: The same for its subclass.

    Returns:
        ``REGION_CELL_COLUMNS``: tile indices (nullable), the tile's
        majority division, ``inferred_region`` (that division when present
        in the inferred set), ``confident_neuron``, ``region_dropped_level``
        and ``region_pruned_changed``.
    """
    n = len(plan.cell_ids)
    inference = plan.inference
    if inference is not None:
        tile = inference.cell_tile
        valid = tile[:, 0] >= 0
        tile_i = pd.array(np.where(valid, tile[:, 0], 0), dtype="Int32")
        tile_j = pd.array(np.where(valid, tile[:, 1], 0), dtype="Int32")
        tile_i[~valid] = pd.NA
        tile_j[~valid] = pd.NA
        tile_region = inference.cell_tile_region
        inferred = set(inference.inferred_regions)
        inferred_region = np.array(
            [value if value in inferred else None for value in tile_region],
            dtype=object,
        )
    else:
        tile_i = pd.array([pd.NA] * n, dtype="Int32")
        tile_j = pd.array([pd.NA] * n, dtype="Int32")
        tile_region = np.full(n, None, dtype=object)
        inferred_region = np.full(n, None, dtype=object)
    confident = np.asarray(plan.confident, dtype=bool)
    changed = np.zeros(n, dtype=bool)
    if class_changed is not None:
        changed |= np.asarray(class_changed, dtype=bool)
    if subclass_changed is not None:
        changed |= np.asarray(subclass_changed, dtype=bool)
    return pd.DataFrame(
        {
            CELL_ID: plan.cell_ids.astype(str),
            TILE_I: tile_i,
            TILE_J: tile_j,
            TILE_REGION: pd.Categorical(tile_region, categories=MOUSE_CCF_REGIONS),
            INFERRED_REGION: pd.Categorical(
                inferred_region, categories=MOUSE_CCF_REGIONS
            ),
            CONFIDENT_NEURON: confident,
            REGION_DROPPED_LEVEL: pd.Categorical(
                plan.dropped_level, categories=["class", "subclass"]
            ),
            REGION_PRUNED_CHANGED: changed,
        }
    )


def region_cells_filename(sample_id: str) -> str:
    """Return ``<sid>_mouse_regions.parquet``."""
    return f"{sample_id}{REGION_CELLS_SUFFIX}"


def write_region_cells(
    cells: pd.DataFrame, path: Path | str, metadata: Mapping[str, Any]
) -> Path:
    """Write the per-cell region table with its metadata in the schema.

    Args:
        cells: ``region_cells_frame`` output.
        path: Output ``<sid>_mouse_regions.parquet``.
        metadata: JSON metadata (status, regions, drop list).

    Returns:
        The written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(cells, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[REGION_CELLS_METADATA_KEY] = json.dumps(
        {"schema_version": REGION_CELLS_SCHEMA_VERSION, **dict(metadata)},
        sort_keys=True,
    ).encode("utf-8")
    table = table.replace_schema_metadata(schema_metadata)
    temporary = output.with_name(output.name + ".partial")
    pq.write_table(table, temporary)
    temporary.replace(output)
    return output


def read_region_cells(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a per-cell region table and its metadata.

    Args:
        path: A ``<sid>_mouse_regions.parquet``.

    Returns:
        ``(cells indexed by cell_id, metadata)``.

    Raises:
        MouseRegionError: If a column is missing.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(REGION_CELLS_METADATA_KEY)
    metadata: dict[str, Any] = json.loads(raw.decode("utf-8")) if raw else {}
    frame = table.to_pandas()
    missing = [column for column in REGION_CELL_COLUMNS if column not in frame]
    if missing:
        raise MouseRegionError(f"{path} lacks columns {missing}")
    frame[CELL_ID] = frame[CELL_ID].astype(str)
    return frame.set_index(CELL_ID, drop=False), metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# RESOLVE-side view


@dataclass(frozen=True)
class MouseRegionOutputs:
    """A section's region step as RESOLVE reads it.

    Attributes:
        record: The manifest record.
        cells: Per-cell region table, indexed by cell id.
        unpruned: The unpruned tidy table.
        pruned: The pruned tidy table (the unpruned one when nothing was
            re-mapped).
        remap: The re-mapped cells' tidy table (empty when none).
    """

    record: MouseRegionRecord
    cells: pd.DataFrame
    unpruned: pd.DataFrame
    pruned: pd.DataFrame
    remap: pd.DataFrame

    def dropped_level(self, obs_names: pd.Index) -> np.ndarray:
        """Return ``region_dropped_level`` per object (``None`` off-table)."""
        values = self.cells[REGION_DROPPED_LEVEL].reindex(obs_names.astype(str))
        return np.array(
            [None if pd.isna(value) else str(value) for value in values], dtype=object
        )

    def engine_columns(
        self,
        obs_names: pd.Index,
        *,
        prefix: str,
        class_level: str,
        subclass_level: str,
    ) -> pd.DataFrame:
        """Return the §4.1 region columns, one row per object.

        Args:
            obs_names: Object ids (non-table objects get nulls / false).
            prefix: Raw engine prefix of the primary run (``mmc_wmb``).
            class_level: Class level key.
            subclass_level: Subclass level key.

        Returns:
            ``<prefix>_unpruned_{class,subclass}_{name,bp}``,
            ``region_pruned_changed`` and ``inferred_region``.
        """
        index = pd.Index([str(name) for name in obs_names])
        columns: dict[str, Any] = {}
        for token, level in (("class", class_level), ("subclass", subclass_level)):
            frame = level_frame(self.unpruned, level).reindex(index)
            stem = f"{prefix}_{UNPRUNED_TOKEN}_{token}"
            columns[f"{stem}_name"] = pd.Categorical(
                frame["name"].astype(object).to_numpy()
            )
            columns[f"{stem}_bp"] = frame["bp"].to_numpy(np.float32)
        changed = self.cells[REGION_PRUNED_CHANGED].reindex(index)
        columns[REGION_PRUNED_CHANGED] = changed.eq(True).to_numpy(bool)
        region = self.cells[INFERRED_REGION].reindex(index)
        columns[INFERRED_REGION] = pd.Categorical(
            region.astype(object).to_numpy(), categories=MOUSE_CCF_REGIONS
        )
        return pd.DataFrame(columns, index=index)


def load_region_outputs(
    map_dir: Path | str,
    record: MouseRegionRecord,
    unpruned: pd.DataFrame,
    *,
    check_sha256: bool = True,
) -> MouseRegionOutputs:
    """Read a section's region outputs next to its ``map_manifest.json``.

    Args:
        map_dir: The directory holding ``map_manifest.json``.
        record: The section's region record.
        unpruned: The primary run's (unpruned) tidy table.
        check_sha256: Check the parquets against their recorded sha256.

    Returns:
        The outputs, with the pruned tidy table merged.

    Raises:
        MouseRegionError: If a parquet changed.
    """
    from merxen.annotation.mapmycells_engine import read_tidy_parquet

    root = Path(map_dir)
    cells_path = root / record.cells_parquet
    if check_sha256 and _sha256(cells_path) != record.cells_parquet_sha256:
        raise MouseRegionError(
            f"{cells_path} differs from the sha256 map_manifest.json recorded"
        )
    cells, _ = read_region_cells(cells_path)
    remap = unpruned.iloc[0:0].copy()
    pruned = unpruned
    if record.remap is not None:
        remap_path = root / record.remap.parquet
        if check_sha256 and _sha256(remap_path) != record.remap.parquet_sha256:
            raise MouseRegionError(
                f"{remap_path} differs from the sha256 map_manifest.json recorded"
            )
        remap, _ = read_tidy_parquet(remap_path)
        pruned = merge_pruned_tidy(unpruned, remap)
    return MouseRegionOutputs(
        record=record, cells=cells, unpruned=unpruned, pruned=pruned, remap=remap
    )
