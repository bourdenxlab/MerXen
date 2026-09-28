"""Reference bundle builders (plan §3.2 builders table, §5.6, §7.1, §7.2, §8.7).

Each builder turns one reference and one declared panel into a bundle inside
``ReferenceStore.get_or_build`` (``merxen.annotation.store``): it writes the
mapping precompute (a checksummed copy, never a symlink), the panel marker
lookup (``query_markers.json`` as ``cell_type_mapper`` wrote it and
``query_markers.filtered.json`` restricted to the mapping tree and the panel,
with zero-marker parents auto-collapsed), ``profiles.parquet``,
``negative_genes.parquet`` (primary references), ``vocab_snapshot.csv``,
``depth_grid.json`` and ``mapping_tree.json``, and returns the diagnostics that
``bundle.json`` records.

Builders (registered with the store on import):

* ``whb_frontal_supc_clus`` (human primary; also the set-c sensitivity bundle,
  which is the same builder on the set-c panel): verify and copy the frontal
  WHB region precompute (or rebuild it from the WHB h5ads), truncate it to
  supercluster -> cluster, and find panel markers (``n_per_utility`` 30;
  ``research/pilot/run_whb_region.sh``).
* ``seaad_mr_panel`` (human secondary): the pinned SEA-AD Multiregion
  precompute (URL + md5 + sha256) and panel markers
  (``research/pilot/run_seaad.sh``).
* ``wmb_panel`` (mouse primary): a marker precompute from at most 50 cells per
  cluster of the local WMB-10Xv3 h5ads restricted to the panel, panel markers
  with the supertype level dropped, and a copy of the Allen means as the
  mapping precompute (``research/selfmap/build_marker_ref.py``,
  ``run_panel_markers.sh``, ``run_map2.sh``). Subclasses without a 10Xv3 cell
  (Multiome-only) are a documented limitation, recorded, never filled in.
* ``wmb_region_share`` (mouse, panel-independent): subclass x CCF-division
  shares from the ABC MERFISH-C57BL6J-638850-CCF cell metadata with OB split
  from OLF, subclass homes and AP-indexed composition windows for every
  section (``exp/E7/01_merfish_tables.py``, ``04_merfish_intrinsic_autoregion.py``).
* ``whb_whole_ctx_panel`` (human, optional): whole-WHB panel markers with the
  cortex-implausible superclusters dropped and the lookup filtered to the
  pruned tree (``research/insilico/06_wb_panel_markers.py``,
  ``exp/E1/06_filter_ctx_lookup.py``); refused above 1,000 genes.
* ``whb_frontal_supc_clus_ho`` (human, resolvability; M3b): the frontal WHB
  reference without the held-out donor (``auto``: the donor with the fewest
  frontal cells, H19.30.002), panel-restricted, with its panel markers, and
  that donor's cells (stratified by supercluster) as the self-map test set
  (``exp/E2/make_ho_ref.py``, ``run_all.sh``, ``research/insilico/01_extract.py``).
* ``wmb_selfmap_testset`` (mouse, resolvability; M3b): the 11,913 self-map
  test cells (``research/selfmap/truth.csv``) plus up to 30 cells per
  non-neuronal supertype that the ``wmb_panel`` marker build did not use,
  panel-restricted.

With resolvability enabled, the primary and secondary builders then run the
self-map (``merxen.annotation.resolvability``) on their test set, which they
get from the store: WHB maps the held-out cells onto the held-out bundle,
SEA-AD maps the same cells onto itself (7-class, via the vocab), WMB maps its
test cells onto itself; each writes ``resolvability.parquet``,
``resolvability_cells.parquet`` and ``resolvability_summary.json``, and a
resolvability trust constraint (``refused`` / ``broad_only``) becomes the
bundle's ``panel_trust``.

``cell_type_mapper``, ``h5py`` and ``anndata`` are imported lazily, inside the
functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.config import (
    KNOWN_REFERENCES,
    LARGE_PANEL_GENES,
    AnnotationConfig,
    AnnotationReferenceSpec,
)
from merxen.annotation.prefilter import (
    MARKER_PREFILTER_FILE,
    PrefilterResult,
    PrefilterSettings,
    SiblingSet,
    per_parent_topk_union,
    settings_from_config,
)
from merxen.annotation.store import (
    BuildContext,
    BundleBuilder,
    StoreError,
    file_sha256,
    make_store_dir,
    register_builder,
)
from merxen.annotation.vocab import (
    ASSET_DIR,
    NEURONS,
    UNASSIGNED_LABEL,
    VOCAB_FILES,
    Species,
    human_floor_class,
    load_state_gene_ids,
    load_vocab,
)

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.panel import AnnotationPanel
    from merxen.annotation.store import BundleRef, ReferenceStore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants

ROOT_KEY: Final = "None"
LOOKUP_METADATA_KEYS: Final[frozenset[str]] = frozenset({"metadata", "log"})

# Bundle file names (plan §3.2 bundle contents).
MAPPING_PRECOMPUTE_FILE: Final = "mapping_precompute.h5"
SOURCE_PRECOMPUTE_FILE: Final = "source_precompute.h5"
MARKER_PRECOMPUTE_FILE: Final = "marker_precompute.h5"
QUERY_MARKERS_FILE: Final = "query_markers.json"
QUERY_MARKERS_FILTERED_FILE: Final = "query_markers.filtered.json"
MAPPING_TREE_FILE: Final = "mapping_tree.json"
PROFILES_FILE: Final = "profiles.parquet"
NEGATIVE_GENES_FILE: Final = "negative_genes.parquet"
VOCAB_SNAPSHOT_FILE: Final = "vocab_snapshot.csv"
DEPTH_GRID_FILE: Final = "depth_grid.json"
PANEL_STUB_FILE: Final = "panel_stub.h5ad"
MARKER_TRAINING_CELLS_FILE: Final = "marker_training_cells.csv"
REFERENCE_MARKERS_DIR: Final = "reference_markers"
CTM_LOG_DIR: Final = "logs"
TAXONOMY_DIR: Final = "taxonomy"
REGION_SHARE_FILE: Final = "region_share.parquet"
REGION_HOME_FILE: Final = "region_home.parquet"
SECTION_COMPOSITION_FILE: Final = "section_composition.parquet"
SECTION_REGIONS_FILE: Final = "section_regions.parquet"
AP_WINDOWS_FILE: Final = "ap_windows.parquet"

# Source names (``AnnotationReferenceSpec.sources`` keys; ``--source NAME=PATH``).
SOURCE_REGION_PRECOMPUTE: Final = "region_precompute"
SOURCE_REGION_MANIFEST: Final = "region_manifest"
SOURCE_WHB_H5AD_DIR: Final = "whb_h5ad_dir"
SOURCE_WHB_METADATA_DIR: Final = "whb_metadata_dir"
SOURCE_WHB_NEURONS_H5AD: Final = "whb_neurons_h5ad"
SOURCE_WHB_NONNEURONS_H5AD: Final = "whb_nonneurons_h5ad"
SOURCE_WHB_CELL_METADATA: Final = "whb_cell_metadata"
SOURCE_WHB_ROI_MAP: Final = "whb_roi_map"
SOURCE_WHB_CLUSTER_ANNOTATION: Final = "whb_cluster_annotation_term"
SOURCE_WHB_CLUSTER_MEMBERSHIP: Final = "whb_cluster_membership"
SOURCE_SEAAD_STATS: Final = "seaad_precomputed_stats"
SOURCE_SEAAD_METADATA_DIR: Final = "seaad_metadata_dir"
SOURCE_WMB_H5AD_DIR: Final = "wmb_h5ad_dir"
SOURCE_WMB_METADATA_DIR: Final = "wmb_metadata_dir"
SOURCE_WMB_CELL_METADATA: Final = "wmb_cell_metadata"
SOURCE_WMB_CLUSTER_ANNOTATION: Final = "wmb_cluster_annotation_term"
SOURCE_WMB_CLUSTER_MEMBERSHIP: Final = "wmb_cluster_membership"
SOURCE_WMB_MAPPING_STATS: Final = "wmb_mapping_stats"
SOURCE_WMB_TEST_CELLS: Final = "wmb_selfmap_test_cells"
SOURCE_WMB_GENE_UNIVERSE: Final = "wmb_marker_gene_universe"
SOURCE_MERFISH_CCF_METADATA: Final = "merfish_ccf_metadata"
SOURCE_WHB_WHOLE_PRECOMPUTE: Final = "whb_whole_precompute"
# Resolvability test sets (M3b). The frontal region cell metadata (donor,
# cluster alias) comes from the region reference directory
# (``region_cell_metadata.csv``); SEA-AD names that directory
# ``whb_region_dir`` because it does not use the WHB precompute itself.
SOURCE_WHB_REGION_CELL_METADATA: Final = "whb_region_cell_metadata"
SOURCE_WHB_REGION_DIR: Final = "whb_region_dir"
REGION_CELL_METADATA_FILE: Final = "region_cell_metadata.csv"

# WHB (Siletti) taxonomy CCN202210140 and the frontal region (plan §3.2).
WHB_TAXONOMY_ID: Final = "CCN202210140"
WHB_SUPC: Final = "CCN202210140_SUPC"
WHB_CLUS: Final = "CCN202210140_CLUS"
WHB_SUBC: Final = "CCN202210140_SUBC"
WHB_SOURCE_HIERARCHY: Final[tuple[str, ...]] = (WHB_SUPC, WHB_CLUS, WHB_SUBC)
WHB_FRONTAL_HIERARCHY: Final[tuple[str, ...]] = (WHB_SUPC, WHB_CLUS)
WHB_FRONTAL_ROI_LABELS: Final[tuple[str, ...]] = (
    "Human A44-A45",
    "Human A46",
    "Human A32",
    "Human ACC",
)
WHB_FRONTAL_MIN_CELLS_PER_LEAF: Final = 10
# The validated frontal precompute (653 leaves, 125,481 cells, 59,357 genes).
WHB_FRONTAL_VALIDATED: Final[dict[str, int]] = {
    "n_leaves": 653,
    "n_cells": 125_481,
    "n_genes": 59_357,
}
# SEA-AD Multiregion taxonomy CCN20260630 (207 supertypes, 36,601 genes).
SEAAD_TAXONOMY_ID: Final = "CCN20260630"
SEAAD_CLASS: Final = "CCN20260630_LEVEL_0"
SEAAD_SUBCLASS: Final = "CCN20260630_LEVEL_1"
SEAAD_SUPERTYPE: Final = "CCN20260630_LEVEL_2"
# WMB (Yao) taxonomy CCN20230722.
WMB_TAXONOMY_ID: Final = "CCN20230722"
WMB_CLAS: Final = "CCN20230722_CLAS"
WMB_SUBC: Final = "CCN20230722_SUBC"
WMB_SUPT: Final = "CCN20230722_SUPT"
WMB_CLUS: Final = "CCN20230722_CLUS"
WMB_HIERARCHY: Final[tuple[str, ...]] = (WMB_CLAS, WMB_SUBC, WMB_SUPT, WMB_CLUS)
WMB_LIBRARY_METHOD: Final = "10Xv3"
WMB_H5AD_SUFFIX: Final = "-raw.h5ad"
# Allen release md5 of the validated WMB mapping means (manifest 20260711).
WMB_MAPPING_STATS_MD5: Final = "d13b316a1755c459d75b8e45ff92ddfc"
WMB_MAPPING_STATS_SIZE: Final = 1_376_366_584
WMB_SAMPLING_SEED: Final = 1
WMB_KNOWN_LIMITATION: Final = (
    "WMB subclasses and clusters without any 10Xv3 marker-training cell (10X "
    "Multiome only, or only self-map test cells) are not in the marker "
    "precompute and are not filled in (user decision 2026-09-26, plan §7.8). "
    "They stay in the mapping tree and the Allen means, and cell_type_mapper "
    "can still assign them with their ancestors' markers (the validated runs "
    "did, e.g. 157 RN Spp1 Glut); such labels have no marker support of "
    "their own and are listed in marker_unsupported_nodes, which RESOLVE "
    "reports as unresolved at that level."
)
# The validated WMB marker-precompute gene universe: the ag7 | VZG2 union of
# research/selfmap/query_{ag7,vzg2}_full.h5ad (899 genes). raw-CPM
# normalisation runs over the genes of the training h5ads, so the universe
# changes the marker choice; a panel inside it uses it (and reproduces the
# validated lookups exactly), any other panel uses its own genes.
WMB_MARKER_UNIVERSE_FILE: Final = "wmb_marker_universe_ag7_vzg2.csv"
WMB_UNIVERSE_VALIDATED: Final = "validated_ag7_vzg2_union"
# MERFISH-C57BL6J-638850-CCF cell metadata (manifest 20231215 / 20260711).
MERFISH_CCF_METADATA_MD5: Final = "51dda47ab139e91357bc09bc4e12c073"
MERFISH_CCF_METADATA_SIZE: Final = 1_606_515_935
# E7 region scheme v2 (``exp/E7/04_merfish_intrinsic_autoregion.py``): the
# grey-matter CCF divisions, with OB = MOB, AOB, the olfactory nerve layer
# ("In") and OLF-unassigned anterior of AP 2.5 mm, split from OLF.
REGION_SCHEME: Final = "e7_v2"
GREY_REGIONS: Final[tuple[str, ...]] = (
    "Isocortex",
    "OLF",
    "OB",
    "HPF",
    "CTXsp",
    "STR",
    "PAL",
    "TH",
    "HY",
    "MB",
    "P",
    "MY",
    "CB",
)
FIBRE_DIVISIONS: Final[frozenset[str]] = frozenset(
    {"lfbs", "cm", "cbf", "mfbs", "scwm", "eps", "fiber tracts-unassigned"}
)
VENTRICLE_DIVISIONS: Final[frozenset[str]] = frozenset({"VL", "V3", "V4", "AQ"})
UNASSIGNED_DIVISIONS: Final[frozenset[str]] = frozenset(
    {"brain-unassigned", "unassigned", "nan"}
)
OB_STRUCTURES: Final[frozenset[str]] = frozenset({"MOB", "AOB"})
OB_NERVE_STRUCTURE: Final = "In"
OLF_UNASSIGNED_STRUCTURE: Final = "OLF-unassigned"
OB_ANTERIOR_AP_MM: Final = 2.5
AP_WINDOW_HALF_WIDTH: Final = 1
MERFISH_COLUMNS: Final[tuple[str, ...]] = (
    "brain_section_label",
    "class",
    "subclass",
    "x_ccf",
    "parcellation_division",
    "parcellation_structure",
)

# Whole-WHB optional bundle: panels above this size are refused (§3.2).
WHB_WHOLE_MAX_GENES: Final = LARGE_PANEL_GENES
WHB_WHOLE_CORTEX_REGION: Final = "frontal_cortex"

# Resolvability test sets (plan §3.2 builders, §8.3; M3b).
HO_REFERENCE_ID: Final = "whb_frontal_supc_clus_ho"
WMB_TESTSET_REFERENCE_ID: Final = "wmb_selfmap_testset"
# E2 make_ho_ref.py: clusters with at least 5 training-donor cells.
HO_MIN_TRAINING_CELLS_PER_CLUSTER: Final = 5
# §8.3 step 1: <= 1,000 test cells per supercluster, all cells of rare ones.
HO_MAX_TEST_CELLS_PER_SUPERCLUSTER: Final = 1000
HO_DEFAULT_TEST_CELLS: Final = 25_000
TEST_SET_SEED: Final = 0
# Other-region non-neuronal test cells (user decision 2026-09-27; E2
# research/insilico/01b_extract_nonneurons.py): the held-out donor holds few
# cells of the thin non-neuronal superclusters (set a: Vascular 23,
# Fibroblast 5, COP 20), so every non-neuronal supercluster of the test set
# is topped up to the per-supercluster cap with WHB non-neuronal nuclei from
# E2's 14 neocortical dissections outside the frontal reference ROIs. Their
# cells are in no frontal reference, training or marker set (checked).
# 2 (user decision 2026-09-27, implemented 2026-09-28): the top-up draws only
# from clusters of the held-out training reference (the kept training
# clusters, >= HO_MIN_TRAINING_CELLS_PER_CLUSTER cells): in version 1, 694 of
# the 2,955 drawn set a cells were of clusters the training reference lacks,
# whose cluster truth no call can name.
HO_OTHER_REGION_VERSION: Final = 2
HO_OTHER_REGION_RULE: Final = (
    "other-region candidates of clusters absent from the held-out training "
    "reference (clusters with fewer than min_training_cells_per_cluster "
    "training cells) are not drawn"
)
HO_OTHER_REGION_ROI_LABELS: Final[tuple[str, ...]] = (
    "Human MTG",
    "Human STG",
    "Human M1C",
    "Human A43",
    "Human A40",
    "Human V1C",
    "Human V2",
    "Human A19",
    "Human S1C",
    "Human A1C",
    "Human A5-A7",
    "Human ITG",
    "Human A38",
    "Human A13",
)
HO_OTHER_REGION_MATRIX: Final = "WHB-10Xv3-Nonneurons"
HO_OTHER_REGION_SEED: Final = 1
# Held-out donor cells whose truth supercluster no production call can name
# are not test cells (M3b review 2; E2 research/insilico/02_make_queries.py
# kept only Neurons-class superclusters other than Amygdala excitatory): the
# sinks and Mixed/Unknown superclusters (Miscellaneous, Splatter) have no
# broad or supercluster truth, so every call of them was scored wrong, and
# calls to region-implausible superclusters (Amygdala excitatory in frontal
# cortex) are excluded, so their supercluster truth could never be called.
HO_TRUTH_EXCLUSION_VERSION: Final = 1
HO_EXCLUDED_SINK: Final = "sink"
HO_EXCLUDED_NO_FLOOR_CLASS: Final = "no_floor_class"
HO_EXCLUDED_REGION_IMPLAUSIBLE: Final = "region_implausible"
# Test-cell provenance column of the human test set.
TEST_SOURCE_COLUMN: Final = "test_source"
TEST_SOURCE_DONOR: Final = "holdout_donor"
TEST_SOURCE_OTHER_REGION: Final = "other_region"
WMB_TESTSET_EXTRA_SEED: Final = 2
# The non-neuronal WMB classes whose supertypes get extra test cells (§3.2:
# Astro-Epen, OPC-Oligo, Vascular, Immune).
WMB_NONNEURONAL_TEST_CLASSES: Final[tuple[str, ...]] = (
    "CS20230722_CLAS_30",
    "CS20230722_CLAS_31",
    "CS20230722_CLAS_33",
    "CS20230722_CLAS_34",
)
HO_SOURCES: Final[tuple[str, ...]] = (
    SOURCE_WHB_REGION_CELL_METADATA,
    SOURCE_WHB_CELL_METADATA,
    SOURCE_WHB_NEURONS_H5AD,
    SOURCE_WHB_NONNEURONS_H5AD,
    SOURCE_WHB_CLUSTER_ANNOTATION,
    SOURCE_WHB_CLUSTER_MEMBERSHIP,
)
WMB_TESTSET_SOURCES: Final[tuple[str, ...]] = (
    SOURCE_WMB_TEST_CELLS,
    SOURCE_WMB_CELL_METADATA,
    SOURCE_WMB_CLUSTER_ANNOTATION,
    SOURCE_WMB_CLUSTER_MEMBERSHIP,
    SOURCE_WMB_H5AD_DIR,
)

# Marker settings (plan §3.2 recipes, §8.7).
DEFAULT_PREP_N_PROCESSORS: Final = 8
DEFAULT_PREP_MAX_GB: Final = 40
# Nextflow passes --max-gb = floor(task.memory x PREP_MAX_GB_FRACTION)
# (workflows/lib/AnnotationReferences.groovy; string-tested), so a PREP task's
# memory reserve is max_gb / PREP_MAX_GB_FRACTION.
PREP_MAX_GB_FRACTION: Final = 0.625
# Peak memory of the WMB query-marker step, the binding PREP step (plan
# §8.7), measured at 8 processes (largest process, /usr/bin/time; M3b stage
# D, evidence m3b/simulate/): VZG2 815 genes 25.8 GB, the Xenium Prime 5K
# Mouse panel prefiltered to 1,992 candidates 37.7 GB and unfiltered (5,006
# genes) 21.3 GB (ag7, 500 genes: 20.2 GB at 16 processes). The peak does
# not grow with the candidate genes up to 5,006 (the plan's ~22 MB per gene
# extrapolation is refuted), so within the measured range the prediction is
# the largest measured peak; beyond it (untested) that peak is scaled with
# the candidate genes. Used only to refuse a large WMB panel built without
# the prefilter when the prediction exceeds the PREP reserve (the prefilter
# is mandatory above it, OD-E8).
WMB_QUERY_MARKER_PEAK_GB_MEASURED: Final = 37.7
WMB_QUERY_MARKER_MEASURED_MAX_GENES: Final = 5006
# Margin on a predicted PREP peak (OD-E8, plan §8.7: the reserve is sized as
# the measured peak + 30%, 37.7 x 1.3 = 49 GB -> 64 GB). A build is refused
# when the prediction times this margin exceeds the reserve.
PREP_MEMORY_MARGIN: Final = 1.3
ROOT_BROAD_MIN_MARKERS: Final = 10
PREP_N_PROCESSORS_ENV: Final = "MERXEN_ANNOTATION_PREP_N_PROCESSORS"
PREP_MAX_GB_ENV: Final = "MERXEN_ANNOTATION_PREP_MAX_GB"


class ReferenceBuildError(StoreError):
    """A reference bundle cannot be built from the given sources."""


class LookupValidationError(ReferenceBuildError):
    """A marker lookup gives the root of the tree no marker in the query."""


# --------------------------------------------------------------------------
# Build resources


@dataclass(frozen=True)
class PrepResources:
    """Processes and memory the ``cell_type_mapper`` marker steps may use.

    Attributes:
        n_processors: ``--n_processors`` of every ctm step (PREP ``task.cpus``).
        max_gb: ``--max_gb`` of the reference-marker step.
    """

    n_processors: int = DEFAULT_PREP_N_PROCESSORS
    max_gb: int = DEFAULT_PREP_MAX_GB


_PREP_RESOURCES: PrepResources | None = None


def set_prep_resources(
    n_processors: int | None = None, max_gb: int | None = None
) -> PrepResources:
    """Set the processes and memory of the ctm steps (``annotation-reference-prep``).

    Resources never enter ``build_hash``: ``cell_type_mapper`` results do not
    depend on them.

    Args:
        n_processors: Processes; ``None`` keeps the current value.
        max_gb: Reference-marker memory bound in GB; ``None`` keeps it.

    Returns:
        The resources now in effect.
    """
    global _PREP_RESOURCES
    current = prep_resources()
    _PREP_RESOURCES = PrepResources(
        n_processors=current.n_processors if n_processors is None else n_processors,
        max_gb=current.max_gb if max_gb is None else max_gb,
    )
    return _PREP_RESOURCES


def prep_resources() -> PrepResources:
    """Return the resources of the ctm steps.

    Returns:
        The value set by ``set_prep_resources``, else the environment
        (``MERXEN_ANNOTATION_PREP_N_PROCESSORS``, ``..._MAX_GB``), else 8
        processes and 40 GB (at most the host's CPU count).
    """
    if _PREP_RESOURCES is not None:
        return _PREP_RESOURCES
    n_processors = _env_int(PREP_N_PROCESSORS_ENV) or min(
        DEFAULT_PREP_N_PROCESSORS, os.cpu_count() or 1
    )
    max_gb = _env_int(PREP_MAX_GB_ENV) or DEFAULT_PREP_MAX_GB
    return PrepResources(n_processors=n_processors, max_gb=max_gb)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, value)
        return None
    return parsed if parsed > 0 else None


# --------------------------------------------------------------------------
# Taxonomy trees


def lookup_key(level: str | None, node: str | None) -> str:
    """Return the MapMyCells lookup key of a parent node.

    Args:
        level: Taxonomy level, or ``None`` for the root.
        node: Node label, or ``None`` for the root.

    Returns:
        ``"None"`` for the root, else ``"<level>/<node>"``.
    """
    if level is None or node is None:
        return ROOT_KEY
    return f"{level}/{node}"


def parse_lookup_key(key: str) -> tuple[str | None, str | None]:
    """Split a lookup key into ``(level, node)``.

    Args:
        key: ``"None"`` or ``"<level>/<node>"``.

    Returns:
        ``(None, None)`` for the root, else ``(level, node)``.

    Raises:
        ValueError: If the key is neither.
    """
    if key == ROOT_KEY:
        return (None, None)
    level, separator, node = key.partition("/")
    if not separator or not level or not node:
        raise ValueError(f"{key!r} is not a lookup key")
    return (level, node)


@dataclass(frozen=True)
class TaxonomyTreeView:
    """A light, ``cell_type_mapper``-free view of a taxonomy tree.

    Attributes:
        hierarchy: Levels, coarse to fine; the last is the leaf level.
        roots: Nodes of the first level (the root's children), sorted.
        children: For every non-leaf level, each node's children (nodes of
            the next level), sorted.
        names: Node names by level and label (from the tree's
            ``name_mapper``), where known.
    """

    hierarchy: tuple[str, ...]
    roots: tuple[str, ...]
    children: dict[str, dict[str, tuple[str, ...]]]
    names: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hierarchy:
            raise ValueError("a taxonomy tree needs at least one level")
        missing = [level for level in self.hierarchy[:-1] if level not in self.children]
        if missing:
            raise ValueError(f"taxonomy tree lacks children for levels {missing}")

    @classmethod
    def from_tree_dict(cls, tree: Mapping[str, Any]) -> TaxonomyTreeView:
        """Build a view from a ``cell_type_mapper`` taxonomy-tree dict.

        Args:
            tree: The ``taxonomy_tree`` JSON of a precompute (``hierarchy``,
                one mapping per level, optional ``name_mapper``).

        Returns:
            The view; the leaf level's cell lists are not kept.
        """
        hierarchy = tuple(str(level) for level in tree["hierarchy"])
        children: dict[str, dict[str, tuple[str, ...]]] = {}
        for level in hierarchy[:-1]:
            children[level] = {
                str(node): tuple(sorted(str(child) for child in kids))
                for node, kids in dict(tree[level]).items()
            }
        roots = tuple(sorted(str(node) for node in dict(tree[hierarchy[0]])))
        return cls(
            hierarchy=hierarchy,
            roots=roots,
            children=children,
            names=_tree_names(tree),
        )

    @classmethod
    def from_precompute(cls, path: Path | str) -> TaxonomyTreeView:
        """Read the taxonomy tree of a precomputed-stats HDF5 file.

        Args:
            path: The precompute.

        Returns:
            The view.
        """
        return cls.from_tree_dict(read_taxonomy_tree(path))

    @property
    def leaf_level(self) -> str:
        """Return the leaf level."""
        return self.hierarchy[-1]

    def nodes(self, level: str) -> tuple[str, ...]:
        """Return the sorted nodes of a level.

        Args:
            level: A level of the tree.

        Returns:
            Node labels.

        Raises:
            KeyError: If the level is not in the tree.
        """
        if level not in self.hierarchy:
            raise KeyError(f"level {level!r} is not in the tree {self.hierarchy}")
        if level == self.hierarchy[0]:
            return self.roots
        if level == self.leaf_level:
            parent_level = self.hierarchy[-2]
            return tuple(
                sorted(
                    {
                        child
                        for kids in self.children[parent_level].values()
                        for child in kids
                    }
                )
            )
        return tuple(sorted(self.children[level]))

    def has_node(self, level: str, node: str) -> bool:
        """Return whether the tree has a node.

        Args:
            level: Level.
            node: Node label.

        Returns:
            ``True`` if the node exists at that level.
        """
        return level in self.hierarchy and node in set(self.nodes(level))

    def child_level(self, level: str | None) -> str:
        """Return the level below a level (``None`` = the root).

        Args:
            level: A non-leaf level, or ``None``.

        Returns:
            The next level.
        """
        if level is None:
            return self.hierarchy[0]
        return self.hierarchy[self.hierarchy.index(level) + 1]

    def children_of(self, level: str | None, node: str | None) -> tuple[str, ...]:
        """Return the children of a node (the root when ``level`` is ``None``).

        Args:
            level: Level, or ``None`` for the root.
            node: Node label, or ``None`` for the root.

        Returns:
            Child labels (empty for leaves).
        """
        if level is None or node is None:
            return self.roots
        if level == self.leaf_level:
            return ()
        return self.children[level].get(node, ())

    def parents(self) -> list[tuple[str | None, str | None]]:
        """Return every parent node, top-down (the root first).

        Returns:
            ``(level, node)`` pairs; ``(None, None)`` is the root.
        """
        result: list[tuple[str | None, str | None]] = [(None, None)]
        for level in self.hierarchy[:-1]:
            result.extend((level, node) for node in self.nodes(level))
        return result

    def parent_keys(self) -> list[str]:
        """Return the lookup keys of every parent, top-down."""
        return [lookup_key(level, node) for level, node in self.parents()]

    def descendants(self, level: str | None, node: str | None) -> list[tuple[str, str]]:
        """Return every descendant of a node, top-down.

        Args:
            level: Level, or ``None`` for the root.
            node: Node label, or ``None`` for the root.

        Returns:
            ``(level, node)`` pairs below the node.
        """
        result: list[tuple[str, str]] = []
        frontier: list[tuple[str | None, str | None]] = [(level, node)]
        while frontier:
            current_level, current_node = frontier.pop()
            if current_level == self.leaf_level:
                continue
            next_level = self.child_level(current_level)
            for child in self.children_of(current_level, current_node):
                result.append((next_level, child))
                frontier.append((next_level, child))
        order = {name: index for index, name in enumerate(self.hierarchy)}
        return sorted(result, key=lambda item: (order[item[0]], item[1]))

    def leaves_under(self, level: str | None, node: str | None) -> list[str]:
        """Return the leaves below (or equal to) a node.

        Args:
            level: Level, or ``None`` for the root.
            node: Node label, or ``None`` for the root.

        Returns:
            Sorted leaf labels.
        """
        if level == self.leaf_level and node is not None:
            return [node]
        return sorted(
            child
            for child_level, child in self.descendants(level, node)
            if child_level == self.leaf_level
        )

    def ancestors_of_leaves(self) -> dict[str, dict[str, str]]:
        """Return each leaf's ancestor at every non-leaf level.

        Returns:
            ``{level: {leaf: ancestor}}`` for the non-leaf levels.
        """
        result: dict[str, dict[str, str]] = {}
        for index, level in enumerate(self.hierarchy[:-1]):
            mapping: dict[str, str] = {}
            for node in self.nodes(level):
                frontier = [node]
                for deeper in self.hierarchy[index:-1]:
                    frontier = [
                        child
                        for current in frontier
                        for child in self.children[deeper].get(current, ())
                    ]
                for leaf in frontier:
                    mapping[leaf] = node
            result[level] = mapping
        return result

    def name(self, level: str, node: str) -> str:
        """Return a node's name, or its label when the tree has no name."""
        return self.names.get(level, {}).get(node, node)

    def drop_level(self, level: str) -> TaxonomyTreeView:
        """Return the tree without a level (``--drop_level``).

        The level's children become the children of its parents.

        Args:
            level: A non-leaf level.

        Returns:
            The new view.

        Raises:
            ValueError: If the level is the leaf level or absent.
        """
        if level not in self.hierarchy:
            raise ValueError(f"cannot drop level {level!r}: not in {self.hierarchy}")
        if level == self.leaf_level:
            raise ValueError(f"cannot drop the leaf level {level!r}")
        index = self.hierarchy.index(level)
        hierarchy = self.hierarchy[:index] + self.hierarchy[index + 1 :]
        children = {name: dict(value) for name, value in self.children.items()}
        dropped = children.pop(level)
        roots = self.roots
        if index == 0:
            roots = tuple(sorted({child for node in roots for child in dropped[node]}))
        else:
            parent_level = self.hierarchy[index - 1]
            children[parent_level] = {
                parent: tuple(
                    sorted(
                        {grand for child in kids for grand in dropped.get(child, ())}
                    )
                )
                for parent, kids in self.children[parent_level].items()
            }
        names = {name: value for name, value in self.names.items() if name != level}
        return TaxonomyTreeView(
            hierarchy=hierarchy, roots=roots, children=children, names=names
        )

    def drop_nodes(self, nodes: Iterable[tuple[str, str]]) -> TaxonomyTreeView:
        """Return the tree without some nodes and their descendants.

        Ancestors left without children are removed too, as
        ``cell_type_mapper``'s ``drop_node_batch`` prunes them.

        Args:
            nodes: ``(level, node)`` pairs.

        Returns:
            The new view.

        Raises:
            ValueError: If a node is not in the tree, or nothing would remain.
        """
        drop_leaves: set[str] = set()
        for level, node in nodes:
            if not self.has_node(level, node):
                raise ValueError(f"cannot drop {level}/{node}: not in the tree")
            drop_leaves.update(self.leaves_under(level, node))
        if not drop_leaves:
            return self
        children: dict[str, dict[str, tuple[str, ...]]] = {}
        kept_below: set[str] = set(self.nodes(self.leaf_level)) - drop_leaves
        for level in reversed(self.hierarchy[:-1]):
            level_children: dict[str, tuple[str, ...]] = {}
            for node, kids in self.children[level].items():
                kept = tuple(child for child in kids if child in kept_below)
                if kept:
                    level_children[node] = kept
            children[level] = level_children
            kept_below = set(level_children)
        roots = tuple(sorted(node for node in self.roots if node in kept_below))
        if not roots:
            raise ValueError("dropping these nodes would empty the tree")
        return TaxonomyTreeView(
            hierarchy=self.hierarchy,
            roots=roots,
            children=children,
            names=dict(self.names),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the tree as ctm-style JSON (leaves map to empty lists)."""
        payload: dict[str, Any] = {"hierarchy": list(self.hierarchy)}
        for level in self.hierarchy[:-1]:
            payload[level] = {
                node: list(self.children[level][node]) for node in self.nodes(level)
            }
        payload[self.leaf_level] = {leaf: [] for leaf in self.nodes(self.leaf_level)}
        payload["name_mapper"] = {
            level: {node: {"name": name} for node, name in sorted(mapping.items())}
            for level, mapping in sorted(self.names.items())
            if level in self.hierarchy
        }
        return payload

    def node_counts(self) -> dict[str, int]:
        """Return the number of nodes per level."""
        return {level: len(self.nodes(level)) for level in self.hierarchy}


def _tree_names(tree: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    names: dict[str, dict[str, str]] = {}
    mapper = tree.get("name_mapper") or {}
    if not isinstance(mapper, Mapping):
        return names
    for level, entries in mapper.items():
        if not isinstance(entries, Mapping):
            continue
        level_names: dict[str, str] = {}
        for node, value in entries.items():
            if isinstance(value, Mapping) and value.get("name"):
                level_names[str(node)] = str(value["name"])
        names[str(level)] = level_names
    return names


def read_taxonomy_tree(path: Path | str) -> dict[str, Any]:
    """Return the ``taxonomy_tree`` JSON of a precomputed-stats file.

    Args:
        path: The precompute (HDF5).

    Returns:
        The decoded tree.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["taxonomy_tree"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    tree: object = json.loads(text)
    if not isinstance(tree, dict):
        raise ReferenceBuildError(f"{path} holds no taxonomy tree")
    return tree


def parse_nodes_to_drop(values: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``nodes_to_drop`` entries written as ``"<level>/<node>"``.

    Args:
        values: Entries.

    Returns:
        ``(level, node)`` pairs.

    Raises:
        ValueError: If an entry is not ``level/node``.
    """
    parsed: list[tuple[str, str]] = []
    for value in values:
        level, node = parse_lookup_key(str(value))
        if level is None or node is None:
            raise ValueError("the root cannot be dropped")
        parsed.append((level, node))
    return parsed


def mapping_tree(
    tree: TaxonomyTreeView,
    *,
    drop_level: str | None = None,
    nodes_to_drop: Iterable[str] = (),
) -> TaxonomyTreeView:
    """Return the tree MapMyCells maps onto: minus a level and some nodes.

    Args:
        tree: The mapping precompute's tree.
        drop_level: Level removed (mouse ``CCN20230722_SUPT``), if any.
        nodes_to_drop: ``"<level>/<node>"`` entries removed with their
            descendants.

    Returns:
        The mapping tree.
    """
    result = tree
    drops = parse_nodes_to_drop(nodes_to_drop)
    if drops:
        result = result.drop_nodes(drops)
    if drop_level is not None:
        result = result.drop_level(drop_level)
    return result


# --------------------------------------------------------------------------
# Marker lookups


def read_lookup(path: Path | str) -> dict[str, Any]:
    """Read a MapMyCells marker lookup (``query_markers*.json``).

    Args:
        path: The lookup.

    Returns:
        The lookup dict.
    """
    payload: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReferenceBuildError(f"{path} does not hold a marker lookup")
    return payload


def lookup_sha256(lookup: Mapping[str, Any]) -> str:
    """Return the sha256 of a lookup's parent-to-markers content.

    ``metadata`` and ``log`` are left out, so the digest depends only on the
    markers.

    Args:
        lookup: The lookup.

    Returns:
        sha256 of the canonical JSON of the marker lists.
    """
    content = {
        key: list(value)
        for key, value in sorted(lookup.items())
        if key not in LOOKUP_METADATA_KEYS
    }
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def filter_lookup_to_tree(
    lookup: Mapping[str, Any], tree: TaxonomyTreeView
) -> tuple[dict[str, Any], list[str]]:
    """Drop the lookup keys of nodes absent from the tree (as parents).

    A lookup built on one tree (the marker precompute, or an unpruned tree)
    keeps keys for nodes the mapping tree lacks (pruned nodes, a dropped
    level, clusters the marker cells did not cover); ``cell_type_mapper``
    fails on such keys when none of their markers is in the query. This
    keeps the root, every non-leaf node of ``tree`` and ``metadata``; ``log``
    is dropped.

    Args:
        lookup: The marker lookup.
        tree: The mapping tree.

    Returns:
        The filtered lookup and the dropped keys (sorted).
    """
    keep = set(tree.parent_keys())
    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in lookup.items():
        if key == "metadata":
            filtered[key] = value
        elif key == "log":
            continue
        elif key in keep:
            filtered[key] = list(value)
        else:
            dropped.append(key)
    return filtered, sorted(dropped)


@dataclass(frozen=True)
class CollapsedParent:
    """A parent auto-collapsed for lack of markers (plan §3.2, R15).

    Attributes:
        key: Lookup key of the parent.
        level: Parent level.
        node: Parent label.
        name: Parent name.
        n_children: Children it had.
        hidden_leaves: Leaves below it, which the bundle cannot resolve.
    """

    key: str
    level: str
    node: str
    name: str
    n_children: int
    hidden_leaves: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "key": self.key,
            "level": self.level,
            "node": self.node,
            "name": self.name,
            "n_children": self.n_children,
            "n_hidden_leaves": len(self.hidden_leaves),
            "hidden_leaves": list(self.hidden_leaves),
        }


@dataclass(frozen=True)
class LookupValidation:
    """Result of ``validate_lookup``.

    Attributes:
        lookup: The lookup restricted to the tree's parents and to the query
            genes, with the descendants of collapsed parents removed.
        dropped_keys: Keys removed because the tree lacks the node.
        collapsed: Parents with more than one child and no marker in the
            query; each becomes a leaf (its children are not resolvable).
        hidden_keys: Keys removed because they lie below a collapsed parent.
        markers_per_parent: Markers in the query per non-trivial parent that
            is not below a collapsed parent (lookup key -> count).
        n_query_genes: Query genes considered.
        n_marker_genes: Distinct marker genes in the result.
    """

    lookup: dict[str, Any]
    dropped_keys: list[str]
    collapsed: list[CollapsedParent]
    hidden_keys: list[str]
    markers_per_parent: dict[str, int]
    n_query_genes: int
    n_marker_genes: int

    @property
    def root_markers(self) -> int:
        """Return the number of root markers in the query."""
        return self.markers_per_parent.get(ROOT_KEY, 0)

    def summary(self, weak_parent_markers: int = 5) -> dict[str, Any]:
        """Return the diagnostics ``bundle.json`` records (plan §8.2).

        Args:
            weak_parent_markers: Parents with fewer markers are weak.

        Returns:
            Markers per parent (min / median / max), weak parents, collapsed
            parents and the leaves they hide, dropped and hidden keys.
        """
        counts = list(self.markers_per_parent.values())
        weak = sorted(
            key
            for key, count in self.markers_per_parent.items()
            if count < weak_parent_markers
        )
        return {
            "n_query_genes": self.n_query_genes,
            "n_marker_genes": self.n_marker_genes,
            "n_parents": len(counts),
            "root_markers": self.root_markers,
            "markers_per_parent_min": min(counts) if counts else None,
            "markers_per_parent_median": (
                float(statistics.median(counts)) if counts else None
            ),
            "markers_per_parent_max": max(counts) if counts else None,
            "weak_parent_markers": weak_parent_markers,
            "weak_parents": weak,
            "collapsed_parents": [parent.key for parent in self.collapsed],
            "collapsed": [parent.to_json() for parent in self.collapsed],
            "n_hidden_leaves": sum(len(p.hidden_leaves) for p in self.collapsed),
            "dropped_keys": self.dropped_keys,
            "hidden_keys": self.hidden_keys,
        }


def validate_lookup(
    lookup: Mapping[str, Any],
    tree: TaxonomyTreeView,
    query_genes: Iterable[str],
) -> LookupValidation:
    """Restrict a lookup to a tree and query genes, auto-collapsing parents.

    Replaces reliance on ``cell_type_mapper``'s zero-marker ``RuntimeError``
    (``type_assignment/marker_cache_v2.py``; plan §3.2, R15):

    1. keys of nodes absent from the tree are dropped
       (``filter_lookup_to_tree``);
    2. every marker list keeps only the query genes (order kept);
    3. top-down, a parent with more than one child and no marker left is
       **collapsed**: it becomes a leaf of this bundle, the keys below it are
       removed and the leaves it hides are recorded. ``cell_type_mapper``
       still assigns inside it with its ancestors' markers (it patches
       parents with too few markers), but RESOLVE must not emit levels below
       it.

    Args:
        lookup: The marker lookup.
        tree: The mapping tree.
        query_genes: Genes of the query (the panel, or a dataset's genes).

    Returns:
        The validated lookup and its diagnostics.

    Raises:
        LookupValidationError: If the root has no marker in the query.
    """
    genes = set(query_genes)
    filtered, dropped = filter_lookup_to_tree(lookup, tree)
    restricted: dict[str, Any] = {}
    for key, value in filtered.items():
        if key == "metadata":
            restricted[key] = value
            continue
        restricted[key] = [gene for gene in value if gene in genes]
    root_markers = restricted.get(ROOT_KEY, [])
    if len(tree.children_of(None, None)) > 1 and not root_markers:
        raise LookupValidationError(
            "the marker lookup gives the root of the taxonomy no marker gene "
            f"among the {len(genes)} query genes; the reference cannot be "
            "mapped on this panel"
        )
    collapsed: list[CollapsedParent] = []
    hidden: set[str] = set()
    markers_per_parent: dict[str, int] = {}
    for level, node in tree.parents():
        key = lookup_key(level, node)
        if key in hidden:
            continue
        children = tree.children_of(level, node)
        if len(children) <= 1:
            continue
        count = len(restricted.get(key, []))
        markers_per_parent[key] = count
        if count == 0 and level is not None and node is not None:
            below = [
                lookup_key(child_level, child)
                for child_level, child in tree.descendants(level, node)
                if child_level != tree.leaf_level
            ]
            hidden.update(below)
            collapsed.append(
                CollapsedParent(
                    key=key,
                    level=level,
                    node=node,
                    name=tree.name(level, node),
                    n_children=len(children),
                    hidden_leaves=tuple(tree.leaves_under(level, node)),
                )
            )
    for key in hidden:
        restricted.pop(key, None)
    for parent in collapsed:
        # Kept, empty, so the lookup itself shows the collapse; ctm patches
        # it with its ancestors' markers instead of failing.
        restricted[parent.key] = []
    marker_genes = {
        gene for key, value in restricted.items() if key != "metadata" for gene in value
    }
    if collapsed:
        logger.warning(
            "Auto-collapsed %d parent(s) without markers in the query: %s",
            len(collapsed),
            ", ".join(parent.key for parent in collapsed),
        )
    return LookupValidation(
        lookup=restricted,
        dropped_keys=dropped,
        collapsed=collapsed,
        hidden_keys=sorted(hidden),
        markers_per_parent=markers_per_parent,
        n_query_genes=len(genes),
        n_marker_genes=len(marker_genes),
    )


MARKER_UNSUPPORTED_POLICY: Final = (
    "cell_type_mapper can still assign these nodes: a parent without markers "
    "of its own is patched with its ancestors' markers, so a label at one of "
    "these nodes is not supported by markers from its own cells. RESOLVE "
    "reports such a label as unresolved at that level (and keeps the parent "
    "level); the nodes stay in the mapping tree, as in the validated runs."
)


def marker_unsupported_nodes(
    tree: TaxonomyTreeView,
    validation: LookupValidation,
    uncovered: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
) -> dict[str, Any]:
    """Return the mapping-tree nodes whose labels no marker supports (§3.2, R15).

    Two kinds: nodes below an auto-collapsed parent (the parent had no
    marker in the panel, so its children cannot be told apart) and, for the
    mouse bundle, nodes without any marker-training cell (``uncovered``:
    Multiome-only WMB subclasses and clusters). ``cell_type_mapper`` can
    still emit both, with inherited markers; RESOLVE (M3) must report them
    as unresolved at their level.

    Args:
        tree: The mapping tree.
        validation: ``validate_lookup`` result on that tree.
        uncovered: ``uncovered_nodes`` output, per level.

    Returns:
        The policy, the unsupported nodes as ``"<level>/<node>"`` keys, and
        their reasons.
    """
    entries: dict[str, dict[str, Any]] = {}
    for parent in validation.collapsed:
        for level, node in tree.descendants(parent.level, parent.node):
            entries.setdefault(
                f"{level}/{node}",
                {
                    "level": level,
                    "node": node,
                    "name": tree.name(level, node),
                    "reason": "below_collapsed_parent",
                    "collapsed_parent": parent.key,
                },
            )
    for level, nodes in (uncovered or {}).items():
        for item in nodes:
            key = f"{level}/{item['node']}"
            entry = entries.setdefault(
                key,
                {
                    "level": level,
                    "node": item["node"],
                    "name": item.get("name", ""),
                    "reason": "no_marker_training_cell",
                },
            )
            if entry["reason"] != "no_marker_training_cell":
                entry["also"] = "no_marker_training_cell"
    ordered = [entries[key] for key in sorted(entries)]
    return {
        "policy": MARKER_UNSUPPORTED_POLICY,
        "n_nodes": len(ordered),
        "n_nodes_per_level": {
            level: sum(1 for entry in ordered if entry["level"] == level)
            for level in tree.hierarchy
        },
        "nodes": [f"{entry['level']}/{entry['node']}" for entry in ordered],
        "details": ordered,
    }


def root_children_with_markers(
    lookup: Mapping[str, Any],
    tree: TaxonomyTreeView,
    min_markers: int = ROOT_BROAD_MIN_MARKERS,
) -> int:
    """Return how many children of the root have at least ``min_markers`` markers.

    A root child without children of its own counts when the root has at
    least ``min_markers`` markers (it is separated at the root). Plan §8.2:
    "how many broad classes the root separates".

    Args:
        lookup: A validated lookup.
        tree: The mapping tree.
        min_markers: Markers a child needs.

    Returns:
        The count.
    """
    first_level = tree.hierarchy[0]
    root_ok = len(lookup.get(ROOT_KEY, [])) >= min_markers
    count = 0
    for child in tree.children_of(None, None):
        if len(tree.children_of(first_level, child)) <= 1:
            count += int(root_ok)
        elif len(lookup.get(lookup_key(first_level, child), [])) >= min_markers:
            count += 1
    return count


# --------------------------------------------------------------------------
# Precomputed stats, profiles and negative genes


@dataclass
class PrecomputedStats:
    """Per-leaf statistics of a precompute, restricted to some genes.

    All arrays hold log2(CPM + 1) statistics as ``cell_type_mapper`` writes
    them (``normalization raw``): row ``i`` is leaf ``leaves[i]``.

    Attributes:
        path: The precompute.
        leaves: Leaf labels in row order.
        genes: Gene IDs in column order.
        n_cells: Cells per leaf.
        sum: Sum of log2(CPM + 1).
        sumsq: Sum of squares, or ``None`` when the file has none.
        gt0: Cells with a count above zero, or ``None``.
        tree: The precompute's taxonomy tree.
    """

    path: Path
    leaves: list[str]
    genes: list[str]
    n_cells: np.ndarray
    sum: np.ndarray
    sumsq: np.ndarray | None
    gt0: np.ndarray | None
    tree: TaxonomyTreeView

    @property
    def has_detection(self) -> bool:
        """Whether ``sumsq`` and ``gt0`` are present (self-built precomputes)."""
        return self.sumsq is not None and self.gt0 is not None


def precompute_genes(path: Path | str) -> list[str]:
    """Return the gene IDs (``col_names``) of a precompute.

    Args:
        path: The precompute.

    Returns:
        Gene IDs in column order.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        raw = handle["col_names"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return [str(gene) for gene in json.loads(text)]


def read_precomputed_stats(
    path: Path | str, genes: Sequence[str] | None = None
) -> PrecomputedStats:
    """Read a precompute's per-leaf statistics for some genes.

    Args:
        path: The precompute (HDF5).
        genes: Genes to read (those absent are skipped); ``None`` reads all.

    Returns:
        The statistics.
    """
    import h5py

    precompute = Path(path)
    with h5py.File(precompute, "r") as handle:
        columns = json.loads(_decode(handle["col_names"][()]))
        cluster_to_row = json.loads(_decode(handle["cluster_to_row"][()]))
        tree_dict = json.loads(_decode(handle["taxonomy_tree"][()]))
        index = {str(gene): position for position, gene in enumerate(columns)}
        selected = (
            [str(gene) for gene in columns]
            if genes is None
            else [str(gene) for gene in genes if str(gene) in index]
        )
        positions = np.array([index[gene] for gene in selected], dtype=np.int64)
        order = np.argsort(positions, kind="stable")
        sorted_positions = positions[order]
        inverse = np.argsort(order, kind="stable")

        def columns_of(name: str) -> np.ndarray | None:
            if name not in handle:
                return None
            if len(sorted_positions) == 0:
                return np.zeros((handle[name].shape[0], 0), dtype=np.float64)
            values = handle[name][:, sorted_positions]
            return np.asarray(values, dtype=np.float64)[:, inverse]

        n_cells = np.asarray(handle["n_cells"][()], dtype=np.float64)
        total = columns_of("sum")
        sumsq = columns_of("sumsq")
        gt0 = columns_of("gt0")
    if total is None:
        raise ReferenceBuildError(f"{precompute} has no 'sum' statistics")
    leaves = [
        str(label)
        for label, _row in sorted(cluster_to_row.items(), key=lambda item: item[1])
    ]
    return PrecomputedStats(
        path=precompute,
        leaves=leaves,
        genes=selected,
        n_cells=n_cells,
        sum=total,
        sumsq=sumsq,
        gt0=gt0,
        tree=TaxonomyTreeView.from_tree_dict(tree_dict),
    )


def _decode(raw: Any) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def lognormal_mean_cpm(stats: PrecomputedStats) -> np.ndarray:
    """Return each leaf's expected CPM per gene (conditional log-normal).

    The ``B_condLN`` reconstruction validated in
    ``exp/E1/01_stats_profile_validation.csv`` (median log-correlation
    0.99998 with the true mean CPM over 130 frontal clusters): with ``g`` the
    detecting cells, ``m = sum / g`` and ``v = sumsq / g - m**2`` are the mean
    and variance of log2(CPM + 1) among them, and
    ``E[CPM] = (g / n) * (2 ** (m + v * ln 2 / 2) - 1)``.

    Args:
        stats: Statistics with ``sumsq`` and ``gt0``.

    Returns:
        ``leaves x genes`` expected CPM.

    Raises:
        ReferenceBuildError: If the precompute lacks ``sumsq`` or ``gt0``.
    """
    if stats.sumsq is None or stats.gt0 is None:
        raise ReferenceBuildError(
            f"{stats.path} has no sumsq / gt0; profiles need a self-built precompute"
        )
    n_cells = np.maximum(stats.n_cells[:, None], 1.0)
    detecting = np.maximum(stats.gt0, 1.0)
    mean = stats.sum / detecting
    variance = np.maximum(stats.sumsq / detecting - mean**2, 0.0)
    expected: np.ndarray = (stats.gt0 / n_cells) * (
        np.power(2.0, mean + variance * math.log(2.0) / 2.0) - 1.0
    )
    return expected


def _leaf_membership(
    stats: PrecomputedStats, levels: Sequence[str]
) -> dict[str, dict[str, list[int]]]:
    """Return, per level, each node's leaf rows in ``stats``."""
    ancestors = stats.tree.ancestors_of_leaves()
    row_of = {leaf: row for row, leaf in enumerate(stats.leaves)}
    membership: dict[str, dict[str, list[int]]] = {}
    for level in levels:
        nodes: dict[str, list[int]] = {}
        if level == stats.tree.leaf_level:
            for leaf, row in row_of.items():
                nodes[leaf] = [row]
        elif level in ancestors:
            for leaf, node in ancestors[level].items():
                if leaf in row_of:
                    nodes.setdefault(node, []).append(row_of[leaf])
        else:
            raise ReferenceBuildError(
                f"level {level!r} is not in the precompute tree {stats.tree.hierarchy}"
            )
        membership[level] = nodes
    return membership


def reference_profiles_from_stats(
    stats: PrecomputedStats,
    *,
    levels: Sequence[str],
    keep_nodes: Mapping[str, Iterable[str]] | None = None,
    names: TaxonomyTreeView | None = None,
) -> pd.DataFrame:
    """Return expected detection and expression per node x gene (``profiles.parquet``).

    Leaves carry the conditional log-normal expected CPM
    (``lognormal_mean_cpm``); a node's profile is the cell-weighted mean over
    its leaves (as ``exp/E1/01_validate_stats_profiles.py`` aggregates
    subclusters to clusters), its detection fraction is ``sum(gt0) /
    sum(n_cells)`` and ``expected_fraction`` is its expected CPM normalised
    over the genes of ``stats`` (the panel genes present in the reference).

    Args:
        stats: Leaf statistics restricted to the panel genes.
        levels: Levels to profile (levels of ``stats``' tree).
        keep_nodes: Per level, the nodes to keep (the mapping tree's nodes);
            ``None`` keeps all.
        names: Tree whose names label the nodes (default: ``stats.tree``).

    Returns:
        Long table with columns ``level, node, node_name, n_cells, gene_id,
        detection_fraction, mean_log2cpm, mean_cpm, expected_fraction``.
        Genes are Ensembl IDs only: a bundle is shared by every panel with
        the same IDs, whatever symbols they declare, so symbols come from the
        run's own panel file.
    """
    expected = lognormal_mean_cpm(stats)
    assert stats.gt0 is not None
    membership = _leaf_membership(stats, levels)
    name_tree = names or stats.tree
    frames: list[pd.DataFrame] = []
    n_genes = len(stats.genes)
    for level in levels:
        allowed = None if keep_nodes is None else set(keep_nodes.get(level, ()))
        for node, rows in sorted(membership[level].items()):
            if allowed is not None and node not in allowed:
                continue
            weights = stats.n_cells[rows]
            n_total = float(weights.sum())
            if n_total <= 0:
                continue
            mean_cpm = (expected[rows] * weights[:, None]).sum(axis=0) / n_total
            detection = stats.gt0[rows].sum(axis=0) / n_total
            mean_log = stats.sum[rows].sum(axis=0) / n_total
            total_cpm = float(mean_cpm.sum())
            fraction = mean_cpm / total_cpm if total_cpm > 0 else np.zeros(n_genes)
            frames.append(
                pd.DataFrame(
                    {
                        "level": level,
                        "node": node,
                        "node_name": name_tree.name(level, node),
                        "n_cells": int(round(n_total)),
                        "gene_id": stats.genes,
                        "detection_fraction": detection,
                        "mean_log2cpm": mean_log,
                        "mean_cpm": mean_cpm,
                        "expected_fraction": fraction,
                    }
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                "level",
                "node",
                "node_name",
                "n_cells",
                "gene_id",
                "detection_fraction",
                "mean_log2cpm",
                "mean_cpm",
                "expected_fraction",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def broad_class_detection(
    stats: PrecomputedStats, leaf_broad_class: Mapping[str, str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the detection fraction per broad class x gene.

    Args:
        stats: Leaf statistics with ``gt0``.
        leaf_broad_class: Broad class per leaf; leaves mapped to
            ``UNASSIGNED_LABEL`` (sinks, unmapped nodes) are left out.

    Returns:
        ``(detection, n_cells)``: ``classes x genes`` fraction of cells with
        a count above zero, and cells per class.
    """
    if stats.gt0 is None:
        raise ReferenceBuildError(f"{stats.path} has no gt0; cannot find negatives")
    rows_by_class: dict[str, list[int]] = {}
    for row, leaf in enumerate(stats.leaves):
        broad = leaf_broad_class.get(leaf, UNASSIGNED_LABEL)
        if broad == UNASSIGNED_LABEL:
            continue
        rows_by_class.setdefault(broad, []).append(row)
    classes = sorted(rows_by_class)
    detection = np.zeros((len(classes), len(stats.genes)))
    n_cells = np.zeros(len(classes))
    for index, broad in enumerate(classes):
        rows = rows_by_class[broad]
        total = float(stats.n_cells[rows].sum())
        n_cells[index] = total
        if total > 0:
            detection[index] = stats.gt0[rows].sum(axis=0) / total
    return (
        pd.DataFrame(detection, index=classes, columns=stats.genes),
        pd.Series(n_cells, index=classes, name="n_cells"),
    )


def negative_gene_table(
    detection_by_reference: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    *,
    genes: Sequence[str],
    state_gene_ids: Iterable[str],
    max_fraction: float = 0.01,
) -> pd.DataFrame:
    """Return the negative genes per broad class (``negative_genes.parquet``, §5.6).

    A panel gene is negative for a broad class when fewer than
    ``max_fraction`` of that class's reference cells detect it in **every**
    reference (human: WHB frontal and SEA-AD Multiregion; mouse: the WMB
    panel precompute), and it is not a curated state gene. A gene missing
    from a reference, or a class missing from one, is never negative. State
    genes are matched by Ensembl ID (``load_state_gene_ids``), so neither an
    alias symbol nor an ID-only panel lets one through.

    Args:
        detection_by_reference: Per reference, ``broad_class_detection``
            output.
        genes: Panel gene IDs.
        state_gene_ids: Ensembl IDs of the curated state genes.
        max_fraction: Detection fraction below which a gene is negative.

    Returns:
        One row per (broad class, gene): ``broad_class, gene_id,
        detection_<reference>, n_cells_<reference>, is_state_gene,
        negative``.
    """
    states = {str(gene_id) for gene_id in state_gene_ids}
    references = sorted(detection_by_reference)
    classes = sorted(
        {
            broad
            for frame, _n in detection_by_reference.values()
            for broad in frame.index
        }
    )
    records: list[dict[str, Any]] = []
    for broad in classes:
        for gene in genes:
            record: dict[str, Any] = {"broad_class": broad, "gene_id": gene}
            is_negative = True
            for reference in references:
                frame, n_cells = detection_by_reference[reference]
                value = (
                    float(frame.at[broad, gene])
                    if broad in frame.index and gene in frame.columns
                    else float("nan")
                )
                record[f"detection_{reference}"] = value
                record[f"n_cells_{reference}"] = (
                    float(n_cells[broad]) if broad in n_cells.index else 0.0
                )
                if not (value == value and value < max_fraction):
                    is_negative = False
            is_state = gene in states
            record["is_state_gene"] = is_state
            record["negative"] = bool(is_negative and not is_state)
            records.append(record)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Vocab snapshot


@dataclass(frozen=True)
class _VocabBinding:
    """How a reference's tree nodes map to a packaged vocab table."""

    table_id: str
    key_level: str
    supertype_level: str | None = None


_VOCAB_BINDINGS: Final[dict[str, _VocabBinding]] = {
    WHB_TAXONOMY_ID: _VocabBinding("whb_supercluster", WHB_SUPC),
    SEAAD_TAXONOMY_ID: _VocabBinding(
        "seaad_mr_subclass", SEAAD_SUBCLASS, supertype_level=SEAAD_SUPERTYPE
    ),
    WMB_TAXONOMY_ID: _VocabBinding("wmb_class", WMB_CLAS),
}


VOCAB_SNAPSHOT_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "node",
    "node_name",
    "key_label",
    "key_name",
    "broad_class",
    "nt",
    "lineage",
    "sink",
    "region_plausible_frontal_cortex",
    "never_drop",
)


def node_vocab_table(tree: TaxonomyTreeView, taxonomy_id: str) -> pd.DataFrame:
    """Return every tree node's vocabulary classes (``vocab_snapshot.csv``).

    Nodes at and below the vocab's key level (WHB supercluster, SEA-AD
    subclass, WMB class) take the key node's classes; SEA-AD supertype leaves
    apply the supertype overrides ("VLMC & Perivascular"). Nodes above the key
    level take the class their leaves share, or ``UNASSIGNED_LABEL`` when the
    leaves differ.

    Args:
        tree: The mapping tree.
        taxonomy_id: ``CCN202210140``, ``CCN20260630`` or ``CCN20230722``.

    Returns:
        One row per node with the ``VOCAB_SNAPSHOT_COLUMNS``; columns the
        vocab lacks are empty. Empty for taxonomies without a vocab.
    """
    binding = _VOCAB_BINDINGS.get(taxonomy_id)
    if binding is None or binding.key_level not in tree.hierarchy:
        return pd.DataFrame(columns=list(VOCAB_SNAPSHOT_COLUMNS))
    vocab = load_vocab(binding.table_id)  # type: ignore[arg-type]
    label_to_name = vocab.label_to_name()
    key_index = tree.hierarchy.index(binding.key_level)
    key_of_leaf = (
        {leaf: leaf for leaf in tree.nodes(tree.leaf_level)}
        if tree.leaf_level == binding.key_level
        else tree.ancestors_of_leaves()[binding.key_level]
    )

    def classes(key_label: str, supertype: str | None) -> dict[str, Any]:
        name = label_to_name.get(key_label) or tree.name(binding.key_level, key_label)
        known = name in vocab
        record: dict[str, Any] = {
            "key_label": key_label,
            "key_name": name if known else "",
            "broad_class": UNASSIGNED_LABEL,
            "nt": "",
            "lineage": "",
            "sink": "",
            "region_plausible_frontal_cortex": "",
            "never_drop": "",
        }
        if not known:
            return record
        record["broad_class"] = vocab.broad_class(name, supertype)
        record["nt"] = vocab.nt(name, supertype) or ""
        columns = vocab.frame.columns
        if "lineage" in columns:
            record["lineage"] = vocab.lineage(name, supertype)
        if "sink" in columns:
            record["sink"] = bool(vocab.is_sink(name))
        if "region_plausible_frontal_cortex" in columns:
            record["region_plausible_frontal_cortex"] = bool(
                vocab.is_region_plausible(name, "frontal_cortex")
            )
        if "never_drop" in columns:
            record["never_drop"] = bool(vocab.is_never_drop(name))
        return record

    leaf_records: dict[str, dict[str, Any]] = {}
    for leaf in tree.nodes(tree.leaf_level):
        supertype = (
            tree.name(tree.leaf_level, leaf)
            if binding.supertype_level == tree.leaf_level
            else None
        )
        leaf_records[leaf] = {
            "level": tree.leaf_level,
            "node": leaf,
            "node_name": tree.name(tree.leaf_level, leaf),
            **classes(key_of_leaf.get(leaf, ""), supertype),
        }
    records: list[dict[str, Any]] = []
    for index, level in enumerate(tree.hierarchy[:-1]):
        for node in tree.nodes(level):
            leaves = tree.leaves_under(level, node)
            base = {"level": level, "node": node, "node_name": tree.name(level, node)}
            if index >= key_index and leaves:
                key_label = (
                    node if level == binding.key_level else key_of_leaf[leaves[0]]
                )
                records.append({**base, **classes(key_label, None)})
                continue
            broads = {leaf_records[leaf]["broad_class"] for leaf in leaves}
            nts = {leaf_records[leaf]["nt"] for leaf in leaves}
            records.append(
                {
                    **base,
                    **classes("", None),
                    "broad_class": broads.pop()
                    if len(broads) == 1
                    else UNASSIGNED_LABEL,
                    "nt": nts.pop() if len(nts) == 1 else "",
                }
            )
    records.extend(leaf_records.values())
    return pd.DataFrame.from_records(records, columns=list(VOCAB_SNAPSHOT_COLUMNS))


def leaf_broad_classes(stats: PrecomputedStats, taxonomy_id: str) -> dict[str, str]:
    """Return the vocab broad class of every leaf of a precompute.

    Args:
        stats: Statistics (their tree names the nodes).
        taxonomy_id: The taxonomy id selecting the vocab table.

    Returns:
        Broad class per leaf label.
    """
    table = node_vocab_table(stats.tree, taxonomy_id)
    leaves = table[table["level"] == stats.tree.leaf_level]
    return dict(zip(leaves["node"], leaves["broad_class"], strict=True))


def vocab_asset_sha256(taxonomy_id: str) -> dict[str, str]:
    """Return the sha256 of the packaged vocab table a taxonomy uses.

    Args:
        taxonomy_id: The taxonomy id.

    Returns:
        ``{file name: sha256}``; empty for taxonomies without a vocab.
    """
    binding = _VOCAB_BINDINGS.get(taxonomy_id)
    if binding is None:
        return {}
    name = VOCAB_FILES[binding.table_id]
    return {name: file_sha256(ASSET_DIR / name)}


# --------------------------------------------------------------------------
# Panel stub and cell_type_mapper runners


def write_panel_stub_h5ad(
    gene_ids: Sequence[str],
    path: Path | str,
    *,
    gene_symbols: Sequence[str] | None = None,
) -> Path:
    """Write a zero-cell h5ad whose ``var`` is the panel (ctm ``--query_path``).

    ``reference_markers`` and ``query_markers`` read only the query's gene
    list, so a stub restricts marker discovery to the panel without any
    query cells (plan §3.2).

    Args:
        gene_ids: Panel gene IDs (the ``var`` index, in this order).
        path: Output file.
        gene_symbols: Optional symbol per gene (``var["gene_symbol"]``).

    Returns:
        The written path.

    Raises:
        ValueError: If the IDs are empty or repeated.
    """
    import anndata as ad
    import scipy.sparse as sp

    ids = [str(gene) for gene in gene_ids]
    if not ids:
        raise ValueError("a panel stub needs at least one gene")
    if len(set(ids)) != len(ids):
        raise ValueError("panel stub gene IDs must be distinct")
    var = pd.DataFrame(index=pd.Index(ids, name="gene_id"))
    if gene_symbols is not None:
        var["gene_symbol"] = [str(symbol) for symbol in gene_symbols]
    stub = ad.AnnData(
        X=sp.csr_matrix((0, len(ids)), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        var=var,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stub.write_h5ad(output)
    return output


# cell_type_mapper steps run in a fresh interpreter each: forking ctm's
# worker pool from a multi-threaded parent (pyarrow, zarr, BLAS pools) can
# crash the workers, and a subprocess gives each step its own peak RSS.
CTM_CLI_MODULES: Final[dict[str, str]] = {
    "truncate": "cell_type_mapper.cli.truncate_precomputed_taxonomy",
    "precompute_abc": "cell_type_mapper.cli.precompute_stats_abc",
    "reference_markers": "cell_type_mapper.cli.reference_markers",
    "query_markers": "cell_type_mapper.cli.query_markers",
}
# One BLAS / numba thread per ctm worker (``n_processors`` workers already
# fill the task's CPUs; plan §3.3), and never a GPU (CPU-only processes).
CTM_SINGLE_THREAD_ENV: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_DROP_NODES_SCRIPT: Final = (
    "import json, sys\n"
    "from cell_type_mapper.diff_exp.precompute_utils import "
    "drop_nodes_from_precomputed_stats\n"
    "args = json.load(open(sys.argv[1]))\n"
    "drop_nodes_from_precomputed_stats(src_path=args['src_path'], "
    "dst_path=args['dst_path'], node_list=[tuple(node) for node in "
    "args['node_list']], clobber=True)\n"
)


def ctm_environment() -> dict[str, str]:
    """Return the environment of a ctm step: one BLAS thread, no GPU."""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in CTM_SINGLE_THREAD_ENV:
        environment[name] = "1"
    return environment


def _tail(path: Path, n_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


GNU_TIME: Final = Path("/usr/bin/time")


def _gnu_time_available() -> bool:
    return GNU_TIME.is_file() and os.access(GNU_TIME, os.X_OK)


def run_python_step(step: str, argv: Sequence[str], *, log_dir: Path) -> dict[str, Any]:
    """Run ``python <argv>`` in a fresh interpreter and return its metrics.

    Peak RSS is the largest single process of the step (ctm's workers
    included), as ``/usr/bin/time`` reports it, which is how the evidence
    logs measured it (e.g. WMB query markers 20-27 GB). Without GNU time it
    comes from ``wait4``, which also counts the parent's memory at fork
    (``peak_rss_source = "wait4"``). The step's whole process tree is also
    sampled (``ProcessTreeSampler``): ``peak_tree_rss_gb`` and
    ``peak_tree_pss_gb`` are the peaks of the summed RSS and PSS of all its
    processes, which a memory reserve must cover (plan §8.7).

    Args:
        step: Step name (log file stem).
        argv: Interpreter arguments, e.g. ``["-m", module, "--input_json", f]``.
        log_dir: Directory of the step's stdout / stderr logs.

    Returns:
        ``{"step", "returncode", "wall_s", "peak_rss_gb", "peak_rss_source",
        "peak_tree_rss_gb", "peak_tree_pss_gb", "peak_tree_processes",
        "tree_samples", "tree_sample_interval_s", "stdout_log", "stderr_log"}``.

    Raises:
        ReferenceBuildError: If the step exits non-zero.
    """
    import subprocess
    import sys

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{step}.stdout.log"
    stderr_path = log_dir / f"{step}.stderr.log"
    rusage_path = log_dir / f"{step}.rusage.txt"
    command = [sys.executable, *argv]
    use_time = _gnu_time_available()
    if use_time:
        command = [str(GNU_TIME), "-f", "%M %e", "-o", str(rusage_path), *command]
    from merxen.annotation.memory import ProcessTreeSampler

    start = time.monotonic()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=ctm_environment()
        )
        sampler = ProcessTreeSampler(process.pid).start()
        try:
            _pid, status, usage = os.wait4(process.pid, 0)
        finally:
            tree = sampler.stop()
        process.returncode = os.waitstatus_to_exitcode(status)
    peak_kb = float(usage.ru_maxrss)
    source = "wait4"
    if use_time:
        try:
            fields = rusage_path.read_text(encoding="utf-8").split()
            peak_kb = float(fields[-2])
            source = "gnu_time"
        except (OSError, IndexError, ValueError):
            source = "wait4"
    metrics = {
        "step": step,
        "returncode": process.returncode,
        "wall_s": round(time.monotonic() - start, 3),
        "peak_rss_gb": round(peak_kb / 1024**2, 3),
        "peak_rss_source": source,
        # Summed over the step's processes (ctm workers included), sampled.
        **tree.to_json(),
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
    }
    if process.returncode != 0:
        raise ReferenceBuildError(
            f"cell_type_mapper step {step!r} failed with exit code "
            f"{process.returncode}; stderr tail:\n{_tail(stderr_path)}"
        )
    return metrics


def _run_ctm_cli(
    step: str, module: str, config: Mapping[str, Any], *, log_dir: Path
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = log_dir / f"{step}.input.json"
    config_path.write_text(json.dumps(dict(config), indent=2) + "\n")
    return run_python_step(
        step, ["-m", module, "--input_json", str(config_path)], log_dir=log_dir
    )


def _run_ctm_truncate(config: dict[str, Any], *, log_dir: Path) -> dict[str, Any]:
    return _run_ctm_cli(
        "truncate", CTM_CLI_MODULES["truncate"], config, log_dir=log_dir
    )


def _run_ctm_precompute_abc(config: dict[str, Any], *, log_dir: Path) -> dict[str, Any]:
    return _run_ctm_cli(
        "precompute_abc", CTM_CLI_MODULES["precompute_abc"], config, log_dir=log_dir
    )


def _run_ctm_reference_markers(
    config: dict[str, Any], *, log_dir: Path
) -> dict[str, Any]:
    return _run_ctm_cli(
        "reference_markers",
        CTM_CLI_MODULES["reference_markers"],
        config,
        log_dir=log_dir,
    )


def _run_ctm_query_markers(config: dict[str, Any], *, log_dir: Path) -> dict[str, Any]:
    return _run_ctm_cli(
        "query_markers", CTM_CLI_MODULES["query_markers"], config, log_dir=log_dir
    )


def _run_ctm_drop_nodes(
    src_path: Path, dst_path: Path, nodes: list[tuple[str, str]], *, log_dir: Path
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = log_dir / "drop_nodes.input.json"
    config_path.write_text(
        json.dumps(
            {
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "node_list": [list(node) for node in nodes],
            }
        )
    )
    return run_python_step(
        "drop_nodes", ["-c", _DROP_NODES_SCRIPT, str(config_path)], log_dir=log_dir
    )


class _StepTimer:
    """Wall time per build step and metrics of each ctm step, for ``bundle.json``.

    Args:
        log_dir: Where ctm steps write their logs (inside the bundle).
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.seconds: dict[str, float] = {}
        self.ctm_steps: dict[str, dict[str, Any]] = {}

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        logger.info("Reference build step %s ...", name)
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.seconds[name] = round(self.seconds.get(name, 0.0) + elapsed, 3)
            logger.info("Reference build step %s took %.1f s", name, elapsed)

    def ctm(self, name: str, runner: Any, *args: Any) -> None:
        """Run a ctm step, timing it and recording its metrics."""
        with self.step(name):
            metrics = runner(*args, log_dir=self.log_dir)
        if isinstance(metrics, Mapping):
            self.ctm_steps[name] = dict(metrics)

    def to_json(self) -> dict[str, Any]:
        """Return the timings and ctm-step metrics."""
        peaks = [
            float(step.get("peak_rss_gb", 0.0)) for step in self.ctm_steps.values()
        ]
        tree_peaks = [
            float(step["peak_tree_pss_gb"])
            for step in self.ctm_steps.values()
            if step.get("peak_tree_pss_gb") is not None
        ]
        return {
            "timings_s": dict(self.seconds),
            "ctm_steps": dict(self.ctm_steps),
            "ctm_peak_rss_gb": max(peaks) if peaks else None,
            "ctm_peak_tree_pss_gb": max(tree_peaks) if tree_peaks else None,
        }


# --------------------------------------------------------------------------
# Pinned downloads (SEA-AD Multiregion)


@dataclass(frozen=True)
class PinnedFile:
    """A reference file pinned by URL, size, Allen md5 and sha256.

    Attributes:
        key: File key (``ensure_*`` result key, ``--download-seed`` key).
        url: Download URL.
        relative_path: Path under the download directory (Allen layout).
        size: Size in bytes.
        md5: md5 from the Allen release manifest.
        sha256: sha256 measured on a verified copy (md5 matched).
    """

    key: str
    url: str
    relative_path: str
    size: int
    md5: str
    sha256: str


ABC_BUCKET_URL: Final = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
# SEA-AD Multiregion taxonomy CCN20260630, Allen release manifest 20260711
# (research/allen_references_mapmycells.md:199; the plan's "sha256
# 9b1d5f50a412..." is the manifest md5). The sha256 values were measured on
# the archived copies whose md5 matches the manifest.
SEAAD_MULTIREGION_FILES: Final[tuple[PinnedFile, ...]] = (
    PinnedFile(
        key="precomputed_stats",
        url=f"{ABC_BUCKET_URL}/mapmycells/SEA-AD-Multiregion-taxonomy/20260711/"
        "precomputed_stats.SEA-AD-Multiregion.2026-06-29.h5",
        relative_path="mapmycells/SEA-AD-Multiregion-taxonomy/20260711/"
        "precomputed_stats.SEA-AD-Multiregion.2026-06-29.h5",
        size=443_095_208,
        md5="9b1d5f50a412cd69311fa1175d4836a7",
        sha256="abbd4984bc453171aab2882b55c343f15ad73ea415b88f0e960019ea71713f6a",
    ),
    PinnedFile(
        key="cluster",
        url=f"{ABC_BUCKET_URL}/metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster.csv",
        relative_path="metadata/SEA-AD-Multiregion-taxonomy/20260711/cluster.csv",
        size=6_034,
        md5="63104e79c4aa87edc98aeb2d61560d16",
        sha256="7b8b346b781f13a334a46e8546981b533675b4385a29ff9b6fcd664b888c1e41",
    ),
    PinnedFile(
        key="cluster_annotation_term",
        url=f"{ABC_BUCKET_URL}/metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_annotation_term.csv",
        relative_path="metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_annotation_term.csv",
        size=32_259,
        md5="6053cd36cb4ca06388063f9e83710dd2",
        sha256="3ccc7c52285eaf64e716e6dc5b71dce1ddfddb4b069f7e48ed036350f0c42c04",
    ),
    PinnedFile(
        key="cluster_annotation_term_set",
        url=f"{ABC_BUCKET_URL}/metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_annotation_term_set.csv",
        relative_path="metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_annotation_term_set.csv",
        size=208,
        md5="bc61cb45aa1579ae80cb68a2ac5dae8b",
        sha256="5053ad5e21b2c90a4259dda5bfdf48d539582075b1d27aedcd0d085a93c4f7ab",
    ),
    PinnedFile(
        key="cluster_to_cluster_annotation_membership",
        url=f"{ABC_BUCKET_URL}/metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_to_cluster_annotation_membership.csv",
        relative_path="metadata/SEA-AD-Multiregion-taxonomy/20260711/"
        "cluster_to_cluster_annotation_membership.csv",
        size=40_589,
        md5="c815f7e24b8eee7ab2ae8a7510edd76e",
        sha256="b6ab2cec0f897e5fc333503970650a25a3eb6d44c02d62fd2ebeaa86cf275979",
    ),
)
SEAAD_STATS_PIN: Final = SEAAD_MULTIREGION_FILES[0]
SEAAD_TAXONOMY_KEYS: Final[tuple[str, ...]] = tuple(
    pinned.key for pinned in SEAAD_MULTIREGION_FILES[1:]
)
# Source name of each SEA-AD taxonomy file in a reference spec.
SEAAD_TAXONOMY_SOURCES: Final[dict[str, str]] = {
    key: f"seaad_{key}" for key in SEAAD_TAXONOMY_KEYS
}


# MERFISH-C57BL6J-638850-CCF cell metadata with the CCF parcellation (ABC
# metadata release 20231215; research/allen_references_mapmycells.md). The
# sha256 was measured on the archived copy whose md5 matches the manifest.
MERFISH_CCF_METADATA_PIN: Final = PinnedFile(
    key="merfish_ccf_metadata",
    url=f"{ABC_BUCKET_URL}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/"
    "cell_metadata_with_parcellation_annotation.csv",
    relative_path="metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/"
    "cell_metadata_with_parcellation_annotation.csv",
    size=MERFISH_CCF_METADATA_SIZE,
    md5=MERFISH_CCF_METADATA_MD5,
    sha256="1e3ae23cc3f8d3d5839cc8a222798ff812fc66f4bd3c0fc8fca0c0279329874e",
)


# The 11,913 WMB self-map test cells (<= 10 per supertype, 10Xv3;
# research/selfmap/truth.csv). They are the resolvability test set (M3b) and
# are kept out of the wmb_panel marker training cells. Local only (no URL):
# a matching copy is seeded into <store>/.downloads, so bundles hash a
# stable path, never the dated evidence archive.
WMB_SELFMAP_TEST_CELLS_PIN: Final = PinnedFile(
    key="wmb_selfmap_test_cells",
    url="",
    relative_path="local/wmb_selfmap/truth_20260925.csv",
    size=1_856_009,
    md5="c87c95a568e64d43c32d6d39dc4ebe98",
    sha256="1b921d93cd0e17843221a3fbb74b94af1b0edefc7cf7971d76943ebc570b72b1",
)


class PinnedFileError(ReferenceBuildError):
    """A pinned reference file is missing or does not match its checksum."""


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, target: Path, pinned: PinnedFile) -> None:
    """Copy a seed file into the download cache when its sha256 matches."""
    staging = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.seeding")
    make_store_dir(target.parent)
    shutil.copyfile(source, staging)
    digest = file_sha256(staging)
    if digest != pinned.sha256:
        raise PinnedFileError(
            f"copy of seed {source} has sha256 {digest}, expected {pinned.sha256}; "
            f"the partial copy is kept at {staging}"
        )
    os.replace(staging, target)
    os.chmod(target, 0o444)


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on a lock file (created if needed, never removed)."""
    import fcntl

    make_store_dir(lock_path.parent)
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
    except PermissionError:
        # Another store user created the lock without write access for us.
        descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _verified_cached(target: Path, pinned: PinnedFile) -> bool:
    """Return whether the cached copy exists; raise if it is not the pinned file."""
    if not target.is_file():
        return False
    size = target.stat().st_size
    digest = file_sha256(target) if size == pinned.size else ""
    if digest != pinned.sha256:
        raise PinnedFileError(
            f"cached {target} does not match the pinned {pinned.key} "
            f"(size {size}, sha256 {digest or 'not computed'}; expected "
            f"{pinned.size} bytes, sha256 {pinned.sha256}). It is left in "
            "place; move it aside by hand to re-download."
        )
    return True


def ensure_pinned_file(
    pinned: PinnedFile,
    download_dir: Path | str,
    *,
    auto_download: bool = False,
    seed: Path | str | None = None,
) -> Path:
    """Return a pinned file in the download cache, verified by size and sha256.

    Order: an existing cached copy (verified; a mismatch raises and nothing
    is deleted), a seed file whose sha256 matches (copied, never linked), a
    download (resumable, to a staging name, renamed only once verified).
    Concurrent callers (parallel PREP tasks) serialise on a lock file next to
    the target, so the file is fetched once.

    Args:
        pinned: The pinned file.
        download_dir: Cache root (``<store>/.downloads`` by default in PREP).
        auto_download: Allow the download (``annotation_auto_download``).
        seed: A local copy to seed the cache from (e.g. the evidence
            archive), used only when its sha256 matches.

    Returns:
        The verified cached file.

    Raises:
        PinnedFileError: If the cached copy or the download does not match,
            or the file is missing and cannot be obtained.
    """
    target = Path(download_dir) / pinned.relative_path
    if _verified_cached(target, pinned):
        return target
    with _file_lock(target.with_name(f"{target.name}.ensure.lock")):
        if _verified_cached(target, pinned):
            return target
        if seed is not None:
            seed_path = Path(seed)
            if (
                seed_path.is_file()
                and seed_path.stat().st_size == pinned.size
                and file_sha256(seed_path) == pinned.sha256
            ):
                logger.info("Seeding %s from %s (sha256 verified)", target, seed_path)
                _copy_verified(seed_path, target, pinned)
                return target
            logger.warning(
                "Ignoring seed %s for %s: missing or not the pinned file",
                seed,
                pinned.key,
            )
        if not pinned.url:
            raise PinnedFileError(
                f"{pinned.key} is not in {download_dir} and has no download URL; "
                "pass a local copy whose sha256 matches the pin"
            )
        if not auto_download:
            raise PinnedFileError(
                f"{pinned.key} is not in {download_dir} and downloads are off; pass "
                f"the file as a source, a matching seed, or enable auto-download "
                f"({pinned.url})"
            )
        from merxen.analysis.mapmycells import _ensure_url_file

        staging = target.with_name(f"{target.name}.download")
        _ensure_url_file(pinned.url, staging, expected_size=pinned.size)
        digest = file_sha256(staging)
        if digest != pinned.sha256:
            raise PinnedFileError(
                f"download of {pinned.url} has sha256 {digest}, expected "
                f"{pinned.sha256}; it is kept at {staging} for review"
            )
        os.replace(staging, target)
        os.chmod(target, 0o444)
    return target


def ensure_seaad_multiregion_inputs(
    download_dir: Path | str,
    *,
    auto_download: bool = False,
    seeds: Mapping[str, Path | str] | None = None,
    include_taxonomy: bool = True,
) -> dict[str, Path]:
    """Return the pinned SEA-AD Multiregion precompute and taxonomy tables.

    The precompute (443 MB, 207 supertypes x 36,601 genes) and the four small
    taxonomy CSVs of release 20260711 are verified by size and sha256
    (``SEAAD_MULTIREGION_FILES``). The 0.54 GB cell-level taxonomy tables
    (embedding, cell-to-cluster membership) are not needed by the bundle and
    are not fetched.

    Args:
        download_dir: Cache root.
        auto_download: Allow downloads.
        seeds: Local copies by file key (``precomputed_stats``, ``cluster``,
            ...), used when their sha256 matches.
        include_taxonomy: Also ensure the taxonomy CSVs.

    Returns:
        Path by file key.
    """
    seeds = seeds or {}
    files = SEAAD_MULTIREGION_FILES if include_taxonomy else (SEAAD_STATS_PIN,)
    return {
        pinned.key: ensure_pinned_file(
            pinned,
            download_dir,
            auto_download=auto_download,
            seed=seeds.get(pinned.key),
        )
        for pinned in files
    }


# --------------------------------------------------------------------------
# Source preparation (before build_hash is computed)


@dataclass(frozen=True)
class SourceOptions:
    """How ``prepare_reference_spec`` may complete a spec's sources.

    Attributes:
        download_dir: Download cache (pinned SEA-AD and MERFISH files);
            ``None`` disables the cache.
        auto_download: Allow downloads into the cache.
        seeds: Local copies of pinned files by key.
        resolvability: Keep (and expand) the resolvability test-set sources
            of the primary and secondary references
            (``AnnotationResolvabilityConfig.enabled``).
    """

    download_dir: Path | None = None
    auto_download: bool = False
    seeds: Mapping[str, Path] = field(default_factory=dict)
    resolvability: bool = True


def _find_one(root: Path, patterns: Sequence[str], what: str) -> Path:
    """Return the single file under ``root`` matching the first fruitful pattern."""
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ReferenceBuildError(
                f"{what}: several files under {root} match {pattern!r}: "
                f"{[str(match) for match in matches]}; pass the file explicitly"
            )
    raise ReferenceBuildError(
        f"{what}: no file under {root} matches any of {list(patterns)}"
    )


def _expand_directory(
    sources: dict[str, Path],
    directory_source: str,
    files: Mapping[str, Sequence[str]],
) -> None:
    """Replace a directory source by the files a builder reads from it."""
    root = sources.pop(directory_source, None)
    if root is None:
        return
    if not root.is_dir():
        raise ReferenceBuildError(
            f"source {directory_source}={root} is not a directory"
        )
    for name, patterns in files.items():
        if name not in sources:
            sources[name] = _find_one(root, patterns, f"{directory_source} -> {name}")


def _require(sources: Mapping[str, Path], names: Iterable[str], reference: str) -> None:
    missing = [name for name in names if name not in sources]
    if missing:
        raise ReferenceBuildError(
            f"reference {reference!r} needs the source(s) {missing}; pass them as "
            "--source NAME=PATH (see merxen.annotation.reference SOURCE_*)"
        )


_WHB_METADATA_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_WHB_CELL_METADATA: (
        "cell_metadata.csv",
        "WHB-10Xv3/*/cell_metadata.csv",
        "metadata/WHB-10Xv3/*/cell_metadata.csv",
    ),
    SOURCE_WHB_ROI_MAP: (
        "region_of_interest_structure_map.csv",
        "WHB-10Xv3/*/region_of_interest_structure_map.csv",
        "metadata/WHB-10Xv3/*/region_of_interest_structure_map.csv",
    ),
    SOURCE_WHB_CLUSTER_ANNOTATION: (
        "cluster_annotation_term.csv",
        "WHB-taxonomy/*/cluster_annotation_term.csv",
        "metadata/WHB-taxonomy/*/cluster_annotation_term.csv",
    ),
    SOURCE_WHB_CLUSTER_MEMBERSHIP: (
        "cluster_to_cluster_annotation_membership.csv",
        "WHB-taxonomy/*/cluster_to_cluster_annotation_membership.csv",
        "metadata/WHB-taxonomy/*/cluster_to_cluster_annotation_membership.csv",
    ),
}
_WHB_H5AD_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_WHB_NEURONS_H5AD: (
        "WHB-10Xv3-Neurons-raw.h5ad",
        "*/WHB-10Xv3-Neurons-raw.h5ad",
        "WHB-10Xv3/*/WHB-10Xv3-Neurons-raw.h5ad",
    ),
    SOURCE_WHB_NONNEURONS_H5AD: (
        "WHB-10Xv3-Nonneurons-raw.h5ad",
        "*/WHB-10Xv3-Nonneurons-raw.h5ad",
        "WHB-10Xv3/*/WHB-10Xv3-Nonneurons-raw.h5ad",
    ),
}
_WMB_METADATA_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    SOURCE_WMB_CELL_METADATA: (
        "cell_metadata.csv",
        "WMB-10X/*/cell_metadata.csv",
        "metadata/WMB-10X/*/cell_metadata.csv",
    ),
    SOURCE_WMB_CLUSTER_ANNOTATION: (
        "cluster_annotation_term.csv",
        "WMB-taxonomy/*/cluster_annotation_term.csv",
        "metadata/WMB-taxonomy/*/cluster_annotation_term.csv",
    ),
    SOURCE_WMB_CLUSTER_MEMBERSHIP: (
        "cluster_to_cluster_annotation_membership.csv",
        "WMB-taxonomy/*/cluster_to_cluster_annotation_membership.csv",
        "metadata/WMB-taxonomy/*/cluster_to_cluster_annotation_membership.csv",
    ),
}
_SEAAD_METADATA_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    SEAAD_TAXONOMY_SOURCES[key]: (
        f"{key}.csv",
        f"SEA-AD-Multiregion-taxonomy/*/{key}.csv",
        f"metadata/SEA-AD-Multiregion-taxonomy/*/{key}.csv",
        f"SEA-AD-Multiregion-taxonomy__*__{key}.csv",
    )
    for key in SEAAD_TAXONOMY_KEYS
}


def _complete_seaad_sources(
    sources: dict[str, Path],
    options: SourceOptions,
    *,
    include_taxonomy: bool,
    reference_id: str,
) -> None:
    """Fill the SEA-AD precompute (and taxonomy) from the pinned download cache."""
    _expand_directory(
        sources,
        SOURCE_SEAAD_METADATA_DIR,
        _SEAAD_METADATA_PATTERNS if include_taxonomy else {},
    )
    wanted = [SOURCE_SEAAD_STATS]
    if include_taxonomy:
        wanted.extend(SEAAD_TAXONOMY_SOURCES.values())
    if all(name in sources for name in wanted):
        return
    if options.download_dir is None:
        _require(sources, [SOURCE_SEAAD_STATS], reference_id)
        return
    ensured = ensure_seaad_multiregion_inputs(
        options.download_dir,
        auto_download=options.auto_download,
        seeds=options.seeds,
        include_taxonomy=include_taxonomy
        and not all(name in sources for name in SEAAD_TAXONOMY_SOURCES.values()),
    )
    sources.setdefault(SOURCE_SEAAD_STATS, ensured["precomputed_stats"])
    for key, name in SEAAD_TAXONOMY_SOURCES.items():
        if key in ensured:
            sources.setdefault(name, ensured[key])


# The WHB cell metadata is always read: it holds the other-region
# non-neuronal test cells (and, without the region cell metadata, rebuilds
# the frontal cells with the ROI map).
_HO_METADATA_PATTERNS: Final[tuple[str, ...]] = (
    SOURCE_WHB_CELL_METADATA,
    SOURCE_WHB_CLUSTER_ANNOTATION,
    SOURCE_WHB_CLUSTER_MEMBERSHIP,
)
_HO_REBUILD_METADATA_PATTERNS: Final[tuple[str, ...]] = (SOURCE_WHB_ROI_MAP,)


def _region_cell_metadata_of(directory: Path | None) -> Path | None:
    """Return a region reference directory's ``region_cell_metadata.csv``."""
    if directory is None:
        return None
    if directory.is_file():
        return directory if directory.name == REGION_CELL_METADATA_FILE else None
    candidate = directory / REGION_CELL_METADATA_FILE
    return candidate if candidate.is_file() else None


def _complete_holdout_sources(sources: dict[str, Path]) -> None:
    """Expand the WHB directories into the held-out test set's source files.

    The test set needs the frontal cells' donors and clusters (the region
    cell metadata, else the WHB cell metadata and ROI map to rebuild it),
    the WHB cell metadata (other-region non-neuronal cells), the taxonomy
    tables and the two raw WHB h5ads (plan §3.2, §8.3).
    """
    metadata_names = list(_HO_METADATA_PATTERNS)
    if SOURCE_WHB_REGION_CELL_METADATA not in sources:
        metadata_names.extend(_HO_REBUILD_METADATA_PATTERNS)
    _expand_directory(
        sources,
        SOURCE_WHB_METADATA_DIR,
        {name: _WHB_METADATA_PATTERNS[name] for name in metadata_names},
    )
    _expand_directory(sources, SOURCE_WHB_H5AD_DIR, _WHB_H5AD_PATTERNS)


def _complete_wmb_test_cells(sources: dict[str, Path], options: SourceOptions) -> None:
    """Point the self-map test-cell source at its stable pinned copy.

    A given list whose sha256 matches ``WMB_SELFMAP_TEST_CELLS_PIN`` seeds
    ``<download_dir>/local/wmb_selfmap/`` and the bundle then reads (and
    hashes) that copy, so moving the evidence archive never changes
    ``build_hash``. Without a given list, a copy already in the cache is
    used. Another list is used as given (``bundle.json`` then records that
    it is not the validated test set).
    """
    pinned = WMB_SELFMAP_TEST_CELLS_PIN
    given = sources.get(SOURCE_WMB_TEST_CELLS)
    if options.download_dir is None:
        return
    cached = Path(options.download_dir) / pinned.relative_path
    if given is None:
        if _verified_cached(cached, pinned):
            sources[SOURCE_WMB_TEST_CELLS] = cached
        return
    if not given.is_file():
        raise ReferenceBuildError(
            f"source {SOURCE_WMB_TEST_CELLS}={given} is not a file"
        )
    if given.resolve() == cached.resolve():
        return
    if given.stat().st_size == pinned.size and file_sha256(given) == pinned.sha256:
        sources[SOURCE_WMB_TEST_CELLS] = ensure_pinned_file(
            pinned, options.download_dir, seed=given
        )
        return
    logger.warning(
        "%s=%s is not the validated self-map test set (sha256 %s); it is used as given",
        SOURCE_WMB_TEST_CELLS,
        given,
        pinned.sha256,
    )


def prepare_reference_spec(
    spec: AnnotationReferenceSpec, options: SourceOptions | None = None
) -> AnnotationReferenceSpec:
    """Return the spec with exactly the source files its builder reads.

    Runs before ``build_hash`` is computed, so the hash covers the files the
    build uses: directory sources (``whb_metadata_dir``, ``wmb_metadata_dir``,
    ``seaad_metadata_dir``, ``whb_h5ad_dir``) are expanded to files, a region
    reference directory given as ``region_precompute`` becomes its
    ``precompute/precomputed_stats.h5`` (+ ``region_reference_manifest.json``),
    and missing SEA-AD and MERFISH CCF files come from the pinned download
    cache.

    Args:
        spec: The reference spec.
        options: Download cache and seeds.

    Returns:
        A copy of the spec with its final sources.

    Raises:
        ReferenceBuildError: If a required source is missing.
    """
    options = options or SourceOptions()
    sources = {name: Path(path) for name, path in spec.sources.items()}
    reference_id = spec.reference_id
    if reference_id == "whb_frontal_supc_clus":
        region = sources.get(SOURCE_REGION_PRECOMPUTE)
        if region is not None and region.is_dir():
            sources[SOURCE_REGION_PRECOMPUTE] = _find_one(
                region,
                ("precompute/precomputed_stats.h5", "precomputed_stats.h5"),
                SOURCE_REGION_PRECOMPUTE,
            )
            manifest = region / "region_reference_manifest.json"
            if manifest.is_file():
                sources.setdefault(SOURCE_REGION_MANIFEST, manifest)
            region_metadata = _region_cell_metadata_of(region)
            if options.resolvability and region_metadata is not None:
                sources.setdefault(SOURCE_WHB_REGION_CELL_METADATA, region_metadata)
        if SOURCE_REGION_PRECOMPUTE in sources:
            if options.resolvability:
                # The held-out test set reads the raw h5ads and taxonomy.
                _complete_holdout_sources(sources)
            else:
                # Rebuild inputs are ignored when the precompute is given.
                for name in (SOURCE_WHB_METADATA_DIR, SOURCE_WHB_H5AD_DIR):
                    sources.pop(name, None)
        else:
            _expand_directory(sources, SOURCE_WHB_METADATA_DIR, _WHB_METADATA_PATTERNS)
            _expand_directory(sources, SOURCE_WHB_H5AD_DIR, _WHB_H5AD_PATTERNS)
            _require(
                sources,
                [*_WHB_METADATA_PATTERNS, *_WHB_H5AD_PATTERNS],
                f"{reference_id} (rebuild: no {SOURCE_REGION_PRECOMPUTE} given)",
            )
        _complete_seaad_sources(
            sources, options, include_taxonomy=False, reference_id=reference_id
        )
    elif reference_id == "seaad_mr_panel":
        _complete_seaad_sources(
            sources, options, include_taxonomy=True, reference_id=reference_id
        )
        region_dir = sources.pop(SOURCE_WHB_REGION_DIR, None)
        if options.resolvability:
            region_metadata = _region_cell_metadata_of(region_dir)
            if region_metadata is not None:
                sources.setdefault(SOURCE_WHB_REGION_CELL_METADATA, region_metadata)
            _complete_holdout_sources(sources)
        else:
            for name in (
                SOURCE_WHB_H5AD_DIR,
                SOURCE_WHB_METADATA_DIR,
                SOURCE_WHB_REGION_CELL_METADATA,
            ):
                sources.pop(name, None)
    elif reference_id == HO_REFERENCE_ID:
        region_dir = sources.pop(SOURCE_WHB_REGION_DIR, None) or sources.pop(
            SOURCE_REGION_PRECOMPUTE, None
        )
        region_metadata = _region_cell_metadata_of(region_dir)
        if region_metadata is not None:
            sources.setdefault(SOURCE_WHB_REGION_CELL_METADATA, region_metadata)
        _complete_holdout_sources(sources)
    elif reference_id == WMB_TESTSET_REFERENCE_ID:
        _expand_directory(sources, SOURCE_WMB_METADATA_DIR, _WMB_METADATA_PATTERNS)
        _require(sources, [SOURCE_WMB_H5AD_DIR, *_WMB_METADATA_PATTERNS], reference_id)
        _complete_wmb_test_cells(sources, options)
        _require(sources, [SOURCE_WMB_TEST_CELLS], reference_id)
    elif reference_id == "wmb_panel":
        _expand_directory(sources, SOURCE_WMB_METADATA_DIR, _WMB_METADATA_PATTERNS)
        _require(
            sources,
            [SOURCE_WMB_H5AD_DIR, *_WMB_METADATA_PATTERNS, SOURCE_WMB_MAPPING_STATS],
            reference_id,
        )
        _complete_wmb_test_cells(sources, options)
    elif reference_id == "wmb_region_share":
        if SOURCE_MERFISH_CCF_METADATA not in sources and options.download_dir:
            sources[SOURCE_MERFISH_CCF_METADATA] = ensure_pinned_file(
                MERFISH_CCF_METADATA_PIN,
                options.download_dir,
                auto_download=options.auto_download,
                seed=options.seeds.get(MERFISH_CCF_METADATA_PIN.key),
            )
        _require(sources, [SOURCE_MERFISH_CCF_METADATA], reference_id)
    elif reference_id == "whb_whole_ctx_panel":
        _require(sources, [SOURCE_WHB_WHOLE_PRECOMPUTE], reference_id)
    for name, path in sources.items():
        if not path.exists():
            raise ReferenceBuildError(f"source {name}={path} does not exist")
    return spec.model_copy(update={"sources": dict(sorted(sources.items()))})


# --------------------------------------------------------------------------
# Shared builder steps


def _panel_of(context: BuildContext) -> AnnotationPanel:
    if context.panel is None:
        raise ReferenceBuildError(
            f"reference {context.spec.reference_id!r} needs a panel to build"
        )
    return context.panel


def _config_of(context: BuildContext) -> AnnotationConfig:
    return context.config or AnnotationConfig(species=context.spec.species)


def _source_path(context: BuildContext, name: str) -> Path:
    record = context.sources.get(name)
    if record is None:
        raise ReferenceBuildError(
            f"reference {context.spec.reference_id!r} has no source {name!r}"
        )
    return Path(record.path)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


@dataclass(frozen=True)
class PanelMarkers:
    """Panel markers of one bundle.

    Attributes:
        query_genes: Panel genes present in the marker precompute (the stub).
        validation: ``validate_lookup`` result on the mapping tree.
        raw_lookup_sha256: Digest of ``query_markers.json`` markers.
        filtered_lookup_sha256: Digest of ``query_markers.filtered.json``.
        reference_markers: Files of the reference-marker step: sha256, size,
            and whether they are kept in the bundle.
        prefilter: The large-panel prefilter record (``PrefilterResult.summary``)
            when it ran, else ``None``.
        n_candidate_genes: Genes marker discovery chose from (the panel stub).
    """

    query_genes: list[str]
    validation: LookupValidation
    raw_lookup_sha256: str
    filtered_lookup_sha256: str
    reference_markers: list[dict[str, Any]]
    prefilter: dict[str, Any] | None = None
    n_candidate_genes: int = 0


def sibling_sets(tree: TaxonomyTreeView, leaves: Sequence[str]) -> list[SiblingSet]:
    """Return every parent of a marker tree with its children's leaf rows.

    Args:
        tree: The tree marker discovery runs on (``--drop_level`` applied).
        leaves: Precompute leaf labels in row order.

    Returns:
        One ``SiblingSet`` per parent, top-down; leaves absent from the
        precompute are skipped.
    """
    row_of = {str(leaf): row for row, leaf in enumerate(leaves)}
    ancestors = tree.ancestors_of_leaves()
    rows_by_node: dict[str, dict[str, list[int]]] = {}
    for ancestor_level, mapping in ancestors.items():
        nodes: dict[str, list[int]] = {}
        for leaf, ancestor in mapping.items():
            if leaf in row_of:
                nodes.setdefault(ancestor, []).append(row_of[leaf])
        rows_by_node[ancestor_level] = nodes
    result: list[SiblingSet] = []
    for level, node in tree.parents():
        child_level = tree.child_level(level)
        children = tuple(str(child) for child in tree.children_of(level, node))
        by_child = rows_by_node.get(child_level, {})
        kept: list[tuple[str, tuple[int, ...]]] = []
        for child in children:
            if child_level == tree.leaf_level:
                rows = [row_of[child]] if child in row_of else []
            else:
                rows = by_child.get(child, [])
            if rows:
                kept.append((child, tuple(sorted(rows))))
        result.append(
            SiblingSet(
                key=lookup_key(level, node),
                children=tuple(child for child, _rows in kept),
                leaf_rows=tuple(r for _child, r in kept),
            )
        )
    return result


def compute_marker_prefilter(
    marker_precompute: Path | str,
    genes: Sequence[str],
    *,
    drop_level: str | None,
    settings: PrefilterSettings,
) -> PrefilterResult:
    """Run the per-parent prefilter on a marker precompute (plan §8.7).

    Args:
        marker_precompute: The precompute marker discovery runs on.
        genes: Panel genes present in it (the unfiltered candidates).
        drop_level: ``--drop_level`` of the marker steps.
        settings: Cap and markers per pair.

    Returns:
        The candidate genes.
    """
    stats = read_precomputed_stats(marker_precompute, genes)
    tree = stats.tree
    if drop_level is not None and drop_level in tree.hierarchy:
        tree = tree.drop_level(drop_level)
    return per_parent_topk_union(
        stats.genes,
        stats.n_cells,
        stats.sum,
        stats.gt0,
        sibling_sets(tree, stats.leaves),
        settings,
    )


def marker_prefilter_settings(
    panel: AnnotationPanel, config: AnnotationConfig, n_per_utility: int
) -> PrefilterSettings | None:
    """Return the prefilter settings of a marker build, or ``None`` when off.

    The same condition as ``build_hash_payload``: panels above
    ``large_panel_genes`` unless ``large_panel_marker_prefilter`` is
    ``"none"``.
    """
    if (
        panel.n_genes <= config.panel.large_panel_genes
        or config.panel.large_panel_marker_prefilter == "none"
    ):
        return None
    return settings_from_config(config.panel.large_panel_prefilter_cap, n_per_utility)


def run_marker_steps(
    *,
    marker_precompute: Path,
    candidates: Sequence[str],
    drop_level: str | None,
    n_per_utility: int,
    scratch_dir: Path,
    reference_dir: Path,
    raw_lookup_path: Path,
    timer: _StepTimer,
    step_prefix: str = "",
) -> list[Path]:
    """Run ``reference_markers`` and ``query_markers`` on candidate genes.

    Both steps read the candidates from a zero-cell panel stub, with
    ``--n_processors`` and ``--max_gb`` explicit (``prep_resources``) and the
    same ``--drop_level``; ``annotation-panel-simulate`` reuses this for the
    unfiltered lookup of a prefiltered panel (plan §8.7).

    Args:
        marker_precompute: Precompute the markers are found on.
        candidates: Candidate genes (the panel, or the prefiltered set).
        drop_level: ``--drop_level`` of both steps.
        n_per_utility: ``--n_per_utility`` of the query markers.
        scratch_dir: Scratch for the stub and ctm's temporary files.
        reference_dir: Where the reference markers go.
        raw_lookup_path: The query-marker lookup to write.
        timer: Step timer (records both ctm steps).
        step_prefix: Prefix of the step names (e.g. ``"unfiltered_"``).

    Returns:
        The reference-marker files.

    Raises:
        ReferenceBuildError: If a step writes nothing.
    """
    resources = prep_resources()
    stub = write_panel_stub_h5ad(
        candidates, scratch_dir / f"{step_prefix}{PANEL_STUB_FILE}"
    )
    tmp_dir = scratch_dir / f"{step_prefix}ctm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_config: dict[str, Any] = {
        "precomputed_path_list": [str(marker_precompute)],
        "output_dir": str(reference_dir),
        "query_path": str(stub),
        "n_processors": resources.n_processors,
        "max_gb": resources.max_gb,
        "tmp_dir": str(tmp_dir),
        "clobber": True,
    }
    if drop_level is not None:
        reference_config["drop_level"] = drop_level
    timer.ctm(
        f"{step_prefix}reference_markers", _run_ctm_reference_markers, reference_config
    )
    reference_paths = sorted(
        path
        for path in reference_dir.iterdir()
        if path.is_file() and path.suffix == ".h5"
    )
    if not reference_paths:
        raise ReferenceBuildError(f"reference_markers wrote nothing in {reference_dir}")
    query_config: dict[str, Any] = {
        "output_path": str(raw_lookup_path),
        "reference_marker_path_list": [str(path) for path in reference_paths],
        "query_path": str(stub),
        "n_per_utility": n_per_utility,
        "n_processors": resources.n_processors,
        "tmp_dir": str(tmp_dir),
        "search_for_stats_file": False,
    }
    if drop_level is not None:
        query_config["drop_level"] = drop_level
    timer.ctm(f"{step_prefix}query_markers", _run_ctm_query_markers, query_config)
    if not raw_lookup_path.is_file():
        raise ReferenceBuildError(f"query_markers wrote no lookup at {raw_lookup_path}")
    return reference_paths


def find_panel_markers(
    context: BuildContext,
    *,
    marker_precompute: Path,
    tree: TaxonomyTreeView,
    query_genes: Sequence[str],
    drop_level: str | None,
    timer: _StepTimer,
    keep_reference_markers: bool = False,
) -> PanelMarkers:
    """Find the panel markers of a precompute and validate them on the mapping tree.

    ``reference_markers --query_path <panel stub>`` then ``query_markers
    --n_per_utility <spec.n_per_utility>`` (plan §3.2 recipes), with
    ``--n_processors`` and ``--max_gb`` explicit (``run_marker_steps``).
    Panels above ``large_panel_genes`` first run the per-parent prefilter
    (``marker_prefilter.json``; plan §8.7) unless it is ``"none"``: the stub
    then holds its candidate genes. Reference markers live in scratch and are
    deleted once the query markers exist (their sha256 is recorded) unless
    ``keep_reference_markers`` (panels above 1,000 genes keep them in the
    bundle, §8.7).

    Args:
        context: The build context.
        marker_precompute: Precompute the markers are found on.
        tree: The mapping tree (for ``validate_lookup``).
        query_genes: Panel genes present in the marker precompute.
        drop_level: ``--drop_level`` of both marker steps.
        timer: Step timer.
        keep_reference_markers: Keep the reference markers in the bundle.

    Returns:
        The markers and their diagnostics.
    """
    genes = sorted(set(query_genes))
    if not genes:
        raise ReferenceBuildError(
            f"no panel gene of {context.spec.reference_id!r} is in the reference"
        )
    candidates = genes
    prefilter_record: dict[str, Any] | None = None
    settings = marker_prefilter_settings(
        _panel_of(context), _config_of(context), context.spec.n_per_utility
    )
    if settings is not None:
        with timer.step("marker_prefilter"):
            prefilter = compute_marker_prefilter(
                marker_precompute, genes, drop_level=drop_level, settings=settings
            )
        _write_json(context.work_dir / MARKER_PREFILTER_FILE, prefilter.to_json())
        prefilter_record = prefilter.summary()
        candidates = prefilter.genes
    reference_dir = (
        context.work_dir / REFERENCE_MARKERS_DIR
        if keep_reference_markers
        else context.scratch_dir / REFERENCE_MARKERS_DIR
    )
    raw_path = context.work_dir / QUERY_MARKERS_FILE
    reference_paths = run_marker_steps(
        marker_precompute=marker_precompute,
        candidates=candidates,
        drop_level=drop_level,
        n_per_utility=context.spec.n_per_utility,
        scratch_dir=context.scratch_dir,
        reference_dir=reference_dir,
        raw_lookup_path=raw_path,
        timer=timer,
    )
    raw = read_lookup(raw_path)
    validation = validate_lookup(raw, tree, genes)
    _write_json(context.work_dir / QUERY_MARKERS_FILTERED_FILE, validation.lookup)
    records = [
        {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
            "kept": keep_reference_markers,
        }
        for path in reference_paths
    ]
    if not keep_reference_markers:
        # Panels up to large_panel_genes: the reference markers are deleted
        # once the query markers exist (sha256 kept; rebuilt in < 1 h if
        # needed, plan §3.2), before the long self-map.
        for path in reference_paths:
            path.unlink(missing_ok=True)
    return PanelMarkers(
        query_genes=genes,
        validation=validation,
        raw_lookup_sha256=lookup_sha256(raw),
        filtered_lookup_sha256=lookup_sha256(validation.lookup),
        reference_markers=records,
        prefilter=prefilter_record,
        n_candidate_genes=len(candidates),
    )


def panel_coverage(
    panel: AnnotationPanel,
    reference_genes: Iterable[str],
    markers: PanelMarkers,
    tree: TaxonomyTreeView,
    *,
    weak_parent_markers: int = 5,
    root_child_min_markers: int = ROOT_BROAD_MIN_MARKERS,
) -> dict[str, Any]:
    """Return the panel coverage diagnostics of a bundle (plan §8.2).

    Args:
        panel: The declared panel.
        reference_genes: Genes of the reference the markers were found on.
        markers: The panel markers.
        tree: The mapping tree.
        weak_parent_markers: Parents with fewer markers are weak.
        root_child_min_markers: Markers a root child needs to count as
            separated.

    Returns:
        Panel genes in / absent from the reference, marker and root-marker
        counts, the root children separated, markers per parent.
    """
    reference = set(reference_genes)
    absent = [gene for gene in panel.ensembl_ids if gene not in reference]
    summary = markers.validation.summary(weak_parent_markers)
    return {
        "n_panel_genes": panel.n_genes,
        "n_query_genes_used": len(markers.query_genes),
        "share_in_reference": round(len(markers.query_genes) / panel.n_genes, 6)
        if panel.n_genes
        else 0.0,
        "absent_from_reference": absent,
        "n_marker_genes": markers.validation.n_marker_genes,
        "root_markers": markers.validation.root_markers,
        "root_children": len(tree.children_of(None, None)),
        "root_children_separated": root_children_with_markers(
            markers.validation.lookup, tree, root_child_min_markers
        ),
        "root_child_min_markers": root_child_min_markers,
        "markers_per_parent_min": summary["markers_per_parent_min"],
        "markers_per_parent_median": summary["markers_per_parent_median"],
        "n_weak_parents": len(summary["weak_parents"]),
        "n_collapsed_parents": len(summary["collapsed_parents"]),
    }


def _markers_output(
    context: BuildContext,
    markers: PanelMarkers,
    tree: TaxonomyTreeView,
    config: AnnotationConfig,
) -> dict[str, Any]:
    summary = markers.validation.summary(config.panel.weak_parent_markers)
    return {
        "n_per_utility": context.spec.n_per_utility,
        "lookup_file": QUERY_MARKERS_FILTERED_FILE,
        "raw_lookup_file": QUERY_MARKERS_FILE,
        "lookup_sha256": markers.filtered_lookup_sha256,
        "raw_lookup_sha256": markers.raw_lookup_sha256,
        "n_panel_genes": len(markers.query_genes),
        "reference_markers": markers.reference_markers,
        "prefilter": markers.prefilter,
        "n_candidate_genes": markers.n_candidate_genes,
        **{key: value for key, value in summary.items() if key != "collapsed"},
        "collapsed": summary["collapsed"],
        "tree_node_counts": tree.node_counts(),
    }


def _bundle_depth_grid(context: BuildContext, n_genes: int | None) -> list[int]:
    """Return a bundle's depth grid: version 7's for a version-7 self-map."""
    from merxen.annotation import resolvability as res

    if context.panel is not None and _resolvability_enabled(context):
        plan = resolvability_plan(context.spec, context.panel, _config_of(context))
        if plan.version == res.RESOLVABILITY_VERSION_V7:
            return list(plan.grid)
    return context.spec.resolved_depth_grid(n_genes)


def _write_common_files(
    context: BuildContext,
    tree: TaxonomyTreeView,
    taxonomy_id: str,
) -> dict[str, Any]:
    """Write ``mapping_tree.json``, ``vocab_snapshot.csv`` and ``depth_grid.json``."""
    panel = context.panel
    n_genes = None if panel is None else panel.n_genes
    _write_json(context.work_dir / MAPPING_TREE_FILE, tree.to_json())
    vocab = node_vocab_table(tree, taxonomy_id)
    vocab.to_csv(context.work_dir / VOCAB_SNAPSHOT_FILE, index=False)
    grid = _bundle_depth_grid(context, n_genes)
    _write_json(
        context.work_dir / DEPTH_GRID_FILE,
        {
            "depth_grid": grid,
            "source": "spec" if context.spec.depth_grid is not None else "default",
            "species": context.spec.species,
            "n_panel_genes": n_genes,
        },
    )
    return {
        "mapping_tree_file": MAPPING_TREE_FILE,
        "vocab_snapshot_file": VOCAB_SNAPSHOT_FILE,
        "vocab_assets": vocab_asset_sha256(taxonomy_id),
        "depth_grid": grid,
    }


def _tree_output(
    tree: TaxonomyTreeView, spec: AnnotationReferenceSpec
) -> dict[str, Any]:
    return {
        "levels": list(tree.hierarchy),
        "node_counts": tree.node_counts(),
        "n_leaves": len(tree.nodes(tree.leaf_level)),
        "drop_level": spec.drop_level,
        "nodes_dropped": list(spec.nodes_to_drop),
    }


def _profiles_and_negatives(
    context: BuildContext,
    *,
    stats: PrecomputedStats,
    tree: TaxonomyTreeView,
    taxonomy_id: str,
    extra_detection: Mapping[str, tuple[pd.DataFrame, pd.Series]] | None,
    reference_label: str,
    write_negatives: bool,
) -> dict[str, Any]:
    """Write ``profiles.parquet`` (and ``negative_genes.parquet``)."""
    config = _config_of(context)
    output: dict[str, Any] = {}
    levels = [level for level in tree.hierarchy if level in stats.tree.hierarchy]
    if stats.has_detection:
        profiles = reference_profiles_from_stats(
            stats,
            levels=levels,
            keep_nodes={level: tree.nodes(level) for level in levels},
            names=tree,
        )
        _write_parquet(profiles, context.work_dir / PROFILES_FILE)
        output["profiles"] = {
            "file": PROFILES_FILE,
            "method": "B_condLN",
            "levels": levels,
            "n_rows": int(len(profiles)),
            "n_nodes": int(profiles[["level", "node"]].drop_duplicates().shape[0])
            if len(profiles)
            else 0,
            "n_genes": len(stats.genes),
            "source": stats.path.name,
        }
    else:
        output["profiles"] = {"file": None, "reason": "precompute has no sumsq / gt0"}
    if write_negatives:
        detection = {
            reference_label: broad_class_detection(
                stats, leaf_broad_classes(stats, taxonomy_id)
            )
        }
        detection.update(extra_detection or {})
        negatives = negative_gene_table(
            detection,
            genes=stats.genes,
            state_gene_ids=load_state_gene_ids(context.spec.species),
            max_fraction=config.flags.negative_gene_max_fraction,
        )
        _write_parquet(negatives, context.work_dir / NEGATIVE_GENES_FILE)
        per_class = (
            negatives[negatives["negative"]].groupby("broad_class").size().to_dict()
            if len(negatives)
            else {}
        )
        output["negative_genes"] = {
            "file": NEGATIVE_GENES_FILE,
            "references": sorted(detection),
            "max_fraction": config.flags.negative_gene_max_fraction,
            "n_state_genes_excluded": int(
                negatives.loc[negatives["is_state_gene"], "gene_id"].nunique()
            )
            if len(negatives)
            else 0,
            "n_negative_per_class": {
                str(key): int(value) for key, value in per_class.items()
            },
        }
    return output


def _state_and_vocab_params(
    species: Species, taxonomy_ids: Iterable[str]
) -> dict[str, Any]:
    from merxen.annotation.vocab import STATE_GENE_FILES

    state_file = STATE_GENE_FILES[species]
    params: dict[str, Any] = {
        "state_genes": {state_file: file_sha256(ASSET_DIR / state_file)},
        "vocab_assets": {},
        "profile_method": "B_condLN",
    }
    for taxonomy_id in taxonomy_ids:
        params["vocab_assets"].update(vocab_asset_sha256(taxonomy_id))
    return params


# --------------------------------------------------------------------------
# WHB frontal (human primary; set c)


def verify_whb_region_precompute(
    path: Path,
    *,
    manifest_path: Path | None = None,
    hierarchy: Sequence[str] = WHB_FRONTAL_HIERARCHY,
) -> dict[str, Any]:
    """Check a WHB region precompute before a bundle uses it.

    Args:
        path: The precompute (the bundle's copy).
        manifest_path: The legacy ``region_reference_manifest.json``, if any;
            its region labels, leaf filter and counts must agree.
        hierarchy: Levels the bundle maps (must be in the precompute).

    Returns:
        Counts and checks; ``matches_validated_frontal`` tells whether the
        counts equal the validated frontal precompute (653 leaves, 125,481
        cells, 59,357 genes).

    Raises:
        ReferenceBuildError: If the file is not a usable precompute or
            disagrees with its manifest.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        missing = [
            name
            for name in (
                "sum",
                "sumsq",
                "gt0",
                "n_cells",
                "col_names",
                "cluster_to_row",
            )
            if name not in handle
        ]
        if missing:
            raise ReferenceBuildError(f"{path} lacks the datasets {missing}")
        n_cells = np.asarray(handle["n_cells"][()])
        n_genes = int(handle["sum"].shape[1])
        cluster_to_row = json.loads(_decode(handle["cluster_to_row"][()]))
    tree = TaxonomyTreeView.from_precompute(path)
    absent_levels = [level for level in hierarchy if level not in tree.hierarchy]
    if absent_levels:
        raise ReferenceBuildError(
            f"{path} has levels {tree.hierarchy}, not {absent_levels}"
        )
    leaves = set(tree.nodes(tree.leaf_level))
    if leaves != set(cluster_to_row):
        raise ReferenceBuildError(f"{path}: tree leaves differ from its stats rows")
    if int((n_cells <= 0).sum()):
        raise ReferenceBuildError(f"{path} has leaves without cells")
    result: dict[str, Any] = {
        "hierarchy": list(tree.hierarchy),
        "node_counts": tree.node_counts(),
        "n_leaves": len(leaves),
        "n_cells": int(n_cells.sum()),
        "n_genes": n_genes,
        "manifest": None,
    }
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        summary = manifest.get("filtering_summary", {})
        problems = []
        if sorted(config.get("region_labels", [])) != sorted(WHB_FRONTAL_ROI_LABELS):
            problems.append(f"region labels {config.get('region_labels')}")
        if config.get("region_min_cells_per_leaf") != WHB_FRONTAL_MIN_CELLS_PER_LEAF:
            problems.append(
                f"min_cells_per_leaf {config.get('region_min_cells_per_leaf')}"
            )
        if summary.get("n_cells_after_min_leaf_filter") != result["n_cells"]:
            problems.append(
                f"{summary.get('n_cells_after_min_leaf_filter')} cells in the manifest"
            )
        if summary.get("n_leaf_aliases_after_min_leaf_filter") != result["n_leaves"]:
            problems.append(
                f"{summary.get('n_leaf_aliases_after_min_leaf_filter')} leaves in "
                "the manifest"
            )
        if problems:
            raise ReferenceBuildError(
                f"{path} disagrees with {manifest_path}: {'; '.join(problems)}"
            )
        result["manifest"] = {
            "path": str(manifest_path),
            "region_labels": list(config.get("region_labels", [])),
            "min_cells_per_leaf": config.get("region_min_cells_per_leaf"),
            "agrees": True,
        }
    result["matches_validated_frontal"] = all(
        result[key] == value for key, value in WHB_FRONTAL_VALIDATED.items()
    )
    if not result["matches_validated_frontal"]:
        logger.warning(
            "WHB region precompute %s differs from the validated frontal build "
            "(%s leaves, %s cells, %s genes; validated %s)",
            path,
            result["n_leaves"],
            result["n_cells"],
            result["n_genes"],
            WHB_FRONTAL_VALIDATED,
        )
    return result


def rebuild_whb_region_precompute(
    context: BuildContext, output_path: Path, timer: _StepTimer
) -> dict[str, Any]:
    """Rebuild the frontal WHB region precompute from the WHB h5ads.

    ``write_region_cell_metadata`` (ROIs Human A44-A45 / A46 / A32 / ACC,
    ``min_cells_per_leaf`` 10) then ``PrecomputationABCRunner`` over the
    neuron and non-neuron h5ads with the legacy region hierarchy
    (supercluster, cluster, subcluster), raw normalisation and pruning, as
    ``prepare_region_mapmycells_reference`` builds it.

    Args:
        context: The build context (``whb_*`` sources).
        output_path: Where the precompute goes (inside the bundle).
        timer: Step timer.

    Returns:
        The region filtering summary.
    """
    from merxen.analysis.mapmycells import _write_region_cell_metadata

    region_metadata = context.scratch_dir / "region_cell_metadata.csv"
    with timer.step("region_cell_metadata"):
        summary = _write_region_cell_metadata(
            cell_metadata_path=_source_path(context, SOURCE_WHB_CELL_METADATA),
            output_path=region_metadata,
            region_labels=list(WHB_FRONTAL_ROI_LABELS),
            min_cells_per_leaf=WHB_FRONTAL_MIN_CELLS_PER_LEAF,
            roi_map_path=_source_path(context, SOURCE_WHB_ROI_MAP),
            region_column="region_of_interest_label",
        )
    tmp_dir = context.scratch_dir / "ctm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    timer.ctm(
        "precompute_region",
        _run_ctm_precompute_abc,
        {
            "output_path": str(output_path),
            "hierarchy": list(WHB_SOURCE_HIERARCHY),
            "h5ad_path_list": [
                str(_source_path(context, SOURCE_WHB_NEURONS_H5AD)),
                str(_source_path(context, SOURCE_WHB_NONNEURONS_H5AD)),
            ],
            "cell_metadata_path": str(region_metadata),
            "cluster_annotation_path": str(
                _source_path(context, SOURCE_WHB_CLUSTER_ANNOTATION)
            ),
            "cluster_membership_path": str(
                _source_path(context, SOURCE_WHB_CLUSTER_MEMBERSHIP)
            ),
            "n_processors": prep_resources().n_processors,
            "split_by_dataset": False,
            "do_pruning": True,
            "tmp_dir": str(tmp_dir),
            "clobber": True,
            "normalization": "raw",
        },
    )
    summary["region_cell_metadata_sha256"] = file_sha256(region_metadata)
    return summary


def build_whb_frontal(context: BuildContext) -> dict[str, Any]:
    """Build the ``whb_frontal_supc_clus`` bundle (human primary; set c).

    Recipe (plan §3.2, ``research/pilot/run_whb_region.sh``): copy and verify
    the frontal region precompute (or rebuild it), truncate it to
    supercluster -> cluster (the mapping precompute), panel reference and
    query markers (``n_per_utility`` 30), the lookup validated on the
    truncated tree, profiles from the copied precompute (subcluster leaves
    aggregated to supercluster and cluster), and negative genes from WHB
    frontal and SEA-AD Multiregion. The set-c bundle is this builder on the
    set-c panel.

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    _check_self_map_sources(context, HO_REFERENCE_ID, HO_SOURCES)
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    source_path = context.work_dir / SOURCE_PRECOMPUTE_FILE
    output: dict[str, Any] = {
        "reference": "WHB frontal region (A44-A45, A46, A32, ACC)"
    }
    if SOURCE_REGION_PRECOMPUTE in context.sources:
        with timer.step("copy_region_precompute"):
            copied = context.copy_source(
                SOURCE_REGION_PRECOMPUTE, SOURCE_PRECOMPUTE_FILE
            )
        manifest_record = context.sources.get(SOURCE_REGION_MANIFEST)
        verification = verify_whb_region_precompute(
            source_path,
            manifest_path=None
            if manifest_record is None
            else Path(manifest_record.path),
        )
        output["source_precompute"] = {
            "mode": "copied",
            "file": SOURCE_PRECOMPUTE_FILE,
            "source_path": copied.source_path,
            "sha256": copied.sha256,
            "verification": verification,
        }
    else:
        summary = rebuild_whb_region_precompute(context, source_path, timer)
        output["source_precompute"] = {
            "mode": "rebuilt",
            "file": SOURCE_PRECOMPUTE_FILE,
            "sha256": file_sha256(source_path),
            "filtering_summary": summary,
            "verification": verify_whb_region_precompute(source_path),
        }
    hierarchy = list(spec.hierarchy or WHB_FRONTAL_HIERARCHY)
    mapping_path = context.work_dir / MAPPING_PRECOMPUTE_FILE
    timer.ctm(
        "truncate_taxonomy",
        _run_ctm_truncate,
        {
            "input_path": str(source_path),
            "output_path": str(mapping_path),
            "new_hierarchy": hierarchy,
        },
    )
    tree = mapping_tree(
        TaxonomyTreeView.from_precompute(mapping_path),
        drop_level=spec.drop_level,
        nodes_to_drop=spec.nodes_to_drop,
    )
    reference_genes = precompute_genes(mapping_path)
    reference_set = set(reference_genes)
    query_genes = [gene for gene in panel.ensembl_ids if gene in reference_set]
    markers = find_panel_markers(
        context,
        marker_precompute=mapping_path,
        tree=tree,
        query_genes=query_genes,
        drop_level=spec.drop_level,
        timer=timer,
        keep_reference_markers=panel.n_genes > config.panel.large_panel_genes,
    )
    with timer.step("profiles_and_negative_genes"):
        stats = read_precomputed_stats(source_path, markers.query_genes)
        seaad_stats = read_precomputed_stats(
            _source_path(context, SOURCE_SEAAD_STATS), markers.query_genes
        )
        output.update(
            _profiles_and_negatives(
                context,
                stats=stats,
                tree=tree,
                taxonomy_id=WHB_TAXONOMY_ID,
                extra_detection={
                    "seaad_mr": broad_class_detection(
                        seaad_stats, leaf_broad_classes(seaad_stats, SEAAD_TAXONOMY_ID)
                    )
                },
                reference_label="whb_frontal",
                write_negatives=True,
            )
        )
    output.update(_write_common_files(context, tree, WHB_TAXONOMY_ID))
    output.update(_tree_output(tree, spec))
    output["mapping_precompute"] = {
        "file": MAPPING_PRECOMPUTE_FILE,
        "hierarchy": hierarchy,
        "sha256": file_sha256(mapping_path),
    }
    output["markers"] = _markers_output(context, markers, tree, config)
    output["collapsed_parents"] = output["markers"]["collapsed_parents"]
    output["panel_coverage"] = panel_coverage(
        panel,
        reference_genes,
        markers,
        tree,
        weak_parent_markers=config.panel.weak_parent_markers,
    )
    output["marker_unsupported_nodes"] = marker_unsupported_nodes(
        tree, markers.validation
    )
    if _resolvability_enabled(context):
        # The self-map maps held-out cells onto the held-out bundle, never
        # onto this one, whose precompute contains the test donor (§5.7).
        output.update(
            _run_self_map(
                context,
                test_reference_id=HO_REFERENCE_ID,
                test_sources=HO_SOURCES,
                engine=None,
                specs_for=lambda engine: _whb_specs(engine, config),
                timer=timer,
                cells_rules=_whb_cells_rules(config),
            )
        )
    output.update(timer.to_json())
    return output


# --------------------------------------------------------------------------
# SEA-AD Multiregion (human secondary)


def build_seaad_mr(context: BuildContext) -> dict[str, Any]:
    """Build the ``seaad_mr_panel`` bundle (human secondary).

    Recipe (plan §3.2, ``research/pilot/run_seaad.sh``): copy the pinned
    SEA-AD Multiregion precompute (checked against the pinned sha256) as the
    mapping precompute, copy the small taxonomy tables, panel reference and
    query markers (``n_per_utility`` 30), the lookup validated on its tree,
    and supertype profiles aggregated to every level.

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    _check_self_map_sources(context, HO_REFERENCE_ID, HO_SOURCES)
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    output: dict[str, Any] = {"reference": "SEA-AD Multiregion (CCN20260630)"}
    with timer.step("copy_precompute"):
        copied = context.copy_source(SOURCE_SEAAD_STATS, MAPPING_PRECOMPUTE_FILE)
    mapping_path = context.work_dir / MAPPING_PRECOMPUTE_FILE
    taxonomy_files: dict[str, str] = {}
    for key, name in SEAAD_TAXONOMY_SOURCES.items():
        if name in context.sources:
            record = context.copy_source(name, f"{TAXONOMY_DIR}/{key}.csv")
            taxonomy_files[key] = record.bundle_path
    output["mapping_precompute"] = {
        "file": MAPPING_PRECOMPUTE_FILE,
        "source_path": copied.source_path,
        "sha256": copied.sha256,
        "pinned_sha256": SEAAD_STATS_PIN.sha256,
        "pinned_md5": SEAAD_STATS_PIN.md5,
        "pinned_url": SEAAD_STATS_PIN.url,
        "matches_pinned_release": copied.sha256 == SEAAD_STATS_PIN.sha256,
    }
    if copied.sha256 != SEAAD_STATS_PIN.sha256:
        logger.warning(
            "SEA-AD precompute %s is not the pinned release (sha256 %s)",
            copied.source_path,
            copied.sha256,
        )
    output["taxonomy_files"] = taxonomy_files
    tree = mapping_tree(
        TaxonomyTreeView.from_precompute(mapping_path),
        drop_level=spec.drop_level,
        nodes_to_drop=spec.nodes_to_drop,
    )
    reference_genes = precompute_genes(mapping_path)
    reference_set = set(reference_genes)
    query_genes = [gene for gene in panel.ensembl_ids if gene in reference_set]
    markers = find_panel_markers(
        context,
        marker_precompute=mapping_path,
        tree=tree,
        query_genes=query_genes,
        drop_level=spec.drop_level,
        timer=timer,
        keep_reference_markers=panel.n_genes > config.panel.large_panel_genes,
    )
    with timer.step("profiles"):
        stats = read_precomputed_stats(mapping_path, markers.query_genes)
        output.update(
            _profiles_and_negatives(
                context,
                stats=stats,
                tree=tree,
                taxonomy_id=SEAAD_TAXONOMY_ID,
                extra_detection=None,
                reference_label="seaad_mr",
                write_negatives=False,
            )
        )
    output.update(_write_common_files(context, tree, SEAAD_TAXONOMY_ID))
    output.update(_tree_output(tree, spec))
    output["markers"] = _markers_output(context, markers, tree, config)
    output["collapsed_parents"] = output["markers"]["collapsed_parents"]
    output["panel_coverage"] = panel_coverage(
        panel,
        reference_genes,
        markers,
        tree,
        weak_parent_markers=config.panel.weak_parent_markers,
    )
    if _resolvability_enabled(context):
        from merxen.annotation import resolvability as res

        # SEA-AD's 7-class precision on the WHB held-out cells (§8.3 step 1),
        # mapped onto this bundle; SEA-AD holds no WHB donor.
        output.update(
            _run_self_map(
                context,
                test_reference_id=HO_REFERENCE_ID,
                test_sources=HO_SOURCES,
                engine=_work_bundle(
                    context,
                    taxonomy_id=SEAAD_TAXONOMY_ID,
                    tree=tree,
                    lookup_sha256_value=markers.filtered_lookup_sha256,
                ),
                specs_for=lambda engine: res.seaad_level_specs(config.thresholds),
                timer=timer,
            )
        )
    output.update(timer.to_json())
    return output


# --------------------------------------------------------------------------
# WMB panel (mouse primary)


def read_cell_label_list(path: Path | str) -> set[str]:
    """Read a list of cell labels (``cell_label`` column, else the first column).

    Args:
        path: CSV (e.g. the self-map ``truth.csv``) or plain text, one label
            per line.

    Returns:
        The labels.
    """
    file_path = Path(path)
    if file_path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(file_path, sep=separator, dtype=str)
        column = "cell_label" if "cell_label" in frame.columns else frame.columns[0]
        return set(frame[column].dropna().astype(str))
    return {
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def read_gene_list(path: Path | str) -> list[str]:
    """Read gene IDs from a panel file, a JSON list or plain text.

    Args:
        path: ``panel_genes*.json`` (``ensembl_ids``), a JSON list, a CSV with
            an ID column (``gene_id``, ``ensembl_id``, ``gene_identifier``,
            else the first column) or one ID per line.

    Returns:
        Gene IDs (order kept, duplicates removed).
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    values: list[str]
    if file_path.suffix.lower() == ".json":
        payload: object = json.loads(text)
        if isinstance(payload, dict) and "ensembl_ids" in payload:
            values = [str(gene) for gene in payload["ensembl_ids"]]
        elif isinstance(payload, list):
            values = [str(gene) for gene in payload]
        else:
            raise ReferenceBuildError(f"{path} holds no gene list")
    elif file_path.suffix.lower() == ".csv":
        frame = pd.read_csv(file_path, dtype=str)
        column = next(
            (
                name
                for name in ("gene_id", "ensembl_id", "gene_identifier")
                if name in frame.columns
            ),
            frame.columns[0],
        )
        values = frame[column].dropna().astype(str).tolist()
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    return list(dict.fromkeys(values))


def sample_wmb_training_cells(
    cell_metadata_path: Path,
    matrices: Iterable[str],
    *,
    max_cells_per_cluster: int,
    exclude_cells: Iterable[str] = (),
    library_method: str = WMB_LIBRARY_METHOD,
    seed: int = WMB_SAMPLING_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample at most N cells per WMB cluster for the marker precompute.

    Reproduces ``research/selfmap/build_marker_ref.py``: cells of
    ``library_method`` 10Xv3 with a cluster, in the available matrices, minus
    the excluded (self-map test) cells, then ``min(n, N)`` per cluster with
    ``DataFrame.sample(random_state=seed)``, clusters in ascending alias
    order.

    Args:
        cell_metadata_path: WMB-10X ``cell_metadata.csv``.
        matrices: Feature-matrix labels available locally.
        max_cells_per_cluster: N (``annotation_wmb_max_cells_per_cluster``).
        exclude_cells: Cell labels never used (resolvability test cells).
        library_method: Library method kept.
        seed: Sampling seed.

    Returns:
        The sampled cells (``cell_label``, ``feature_matrix_label``,
        ``library_method``, ``cluster_alias``) and a summary.
    """
    meta = pd.read_csv(
        cell_metadata_path,
        usecols=[
            "cell_label",
            "feature_matrix_label",
            "library_method",
            "cluster_alias",
        ],
        dtype={"cluster_alias": "Int64"},
    )
    n_total = len(meta)
    meta = meta[meta["library_method"] == library_method].dropna(
        subset=["cluster_alias"]
    )
    n_library = len(meta)
    available = set(matrices)
    meta = meta[meta["feature_matrix_label"].isin(available)]
    n_available = len(meta)
    excluded = set(exclude_cells)
    is_excluded = meta["cell_label"].isin(excluded)
    meta = meta[~is_excluded]
    parts = [
        group.sample(n=min(len(group), max_cells_per_cluster), random_state=seed)
        for _alias, group in meta.groupby("cluster_alias", sort=True)
    ]
    sampled = (
        pd.concat(parts).reset_index(drop=True)
        if parts
        else meta.iloc[0:0].reset_index(drop=True)
    )
    per_cluster = sampled.groupby("cluster_alias").size()
    summary = {
        "n_cells_in_metadata": int(n_total),
        f"n_cells_{library_method}": int(n_library),
        "n_cells_in_available_matrices": int(n_available),
        "n_excluded_test_cells": int(is_excluded.sum()),
        "n_exclude_list": len(excluded),
        "n_sampled_cells": int(len(sampled)),
        "n_clusters": int(sampled["cluster_alias"].nunique()),
        "n_clusters_at_cap": int((per_cluster >= max_cells_per_cluster).sum()),
        "max_cells_per_cluster": int(max_cells_per_cluster),
        "seed": int(seed),
        "matrices": sorted(available),
        "cells_per_matrix": {
            str(key): int(value)
            for key, value in sampled.groupby("feature_matrix_label").size().items()
        },
    }
    return sampled, summary


def extract_panel_training_h5ads(
    sampled: pd.DataFrame,
    matrices: Mapping[str, Path],
    genes: Sequence[str],
    output_dir: Path,
    *,
    block_rows: int = 5000,
) -> tuple[list[Path], pd.DataFrame, dict[str, Any]]:
    """Write panel-restricted h5ads of the sampled cells (one per matrix).

    Args:
        sampled: ``sample_wmb_training_cells`` output.
        matrices: Path per feature-matrix label.
        genes: Gene IDs kept (must be in every matrix), in this order.
        output_dir: Scratch directory for the h5ads.
        block_rows: Rows read per block from the backed matrices.

    Returns:
        The written h5ads, the cells found (``cell_label``,
        ``cluster_alias``, ``feature_matrix_label``) and a summary.
    """
    import anndata as ad
    import scipy.sparse as sp

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    kept_parts: list[pd.DataFrame] = []
    missing: dict[str, int] = {}
    for label, group in sampled.groupby("feature_matrix_label", sort=True):
        source = ad.read_h5ad(matrices[str(label)], backed="r")
        try:
            positions = source.obs_names.get_indexer(group["cell_label"].astype(str))
            found = positions >= 0
            missing[str(label)] = int((~found).sum())
            group = group[found]
            positions = positions[found]
            order = np.argsort(positions, kind="stable")
            positions = positions[order]
            group = group.iloc[order]
            gene_index = source.var_names.get_indexer(list(genes))
            if (gene_index < 0).any():
                absent = [
                    gene for gene, idx in zip(genes, gene_index, strict=True) if idx < 0
                ]
                raise ReferenceBuildError(
                    f"{matrices[str(label)]} lacks panel genes {absent[:10]}"
                )
            blocks = [
                sp.csr_matrix(source.X[positions[start : start + block_rows]])[
                    :, gene_index
                ]
                for start in range(0, len(positions), block_rows)
            ]
        finally:
            source.file.close()
        matrix = (
            sp.vstack(blocks).tocsr().astype(np.float32)
            if blocks
            else sp.csr_matrix((0, len(genes)), dtype=np.float32)
        )
        restricted = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=group["cell_label"].astype(str).to_numpy()),
            var=pd.DataFrame(index=list(genes)),
        )
        path = output_dir / f"{label}-panel-raw.h5ad"
        restricted.write_h5ad(path)
        written.append(path)
        kept_parts.append(
            group[["cell_label", "cluster_alias", "feature_matrix_label"]]
        )
        logger.info("Extracted %d training cells from %s", len(group), label)
    kept = (
        pd.concat(kept_parts, ignore_index=True)
        if kept_parts
        else pd.DataFrame(
            columns=["cell_label", "cluster_alias", "feature_matrix_label"]
        )
    )
    return written, kept, {"n_cells_missing_from_matrices": missing}


def validated_wmb_marker_universe() -> list[str]:
    """Return the validated WMB marker-precompute universe (899 genes).

    Returns:
        The ag7 | VZG2 union of the validated self-map queries, sorted.
    """
    from merxen.annotation.vocab import load_asset_table

    return sorted(load_asset_table(WMB_MARKER_UNIVERSE_FILE)["ensembl_id"])


def _wmb_gene_universe(
    context: BuildContext, panel: AnnotationPanel, matrices: Mapping[str, Path]
) -> tuple[list[str], dict[str, Any]]:
    """Return the marker-precompute genes and how they were chosen.

    An explicit ``wmb_marker_gene_universe`` source adds its genes to the
    panel's. Otherwise a panel whose genes (those in the WMB matrices) all
    lie in the validated ag7 | VZG2 union uses that union, which reproduces
    the validated lookups exactly (plan §7.1, R28); any other panel uses its
    own genes (a new family, validated by M3's shadow checks).
    """
    import anndata as ad

    wanted = list(panel.ensembl_ids)
    first = next(iter(matrices.values()))
    source = ad.read_h5ad(first, backed="r")
    try:
        reference_genes = set(source.var_names.astype(str))
    finally:
        source.file.close()
    in_reference = {gene for gene in wanted if gene in reference_genes}
    extra: list[str] = []
    reason: str | None = None
    if SOURCE_WMB_GENE_UNIVERSE in context.sources:
        extra = read_gene_list(_source_path(context, SOURCE_WMB_GENE_UNIVERSE))
        universe_source = SOURCE_WMB_GENE_UNIVERSE
    else:
        validated = validated_wmb_marker_universe()
        if in_reference <= set(validated):
            extra = validated
            universe_source = WMB_UNIVERSE_VALIDATED
        else:
            universe_source = "panel"
            reason = (
                f"{len(in_reference - set(validated))} panel gene(s) lie outside the "
                "validated ag7 | VZG2 universe"
            )
            logger.warning(
                "wmb_panel: %s; the marker precompute uses the panel's own genes "
                "(not the validated configuration)",
                reason,
            )
    universe = sorted({gene for gene in [*wanted, *extra] if gene in reference_genes})
    return universe, {
        "n_panel_genes": panel.n_genes,
        "n_extra_universe_genes": len(set(extra) - set(wanted)),
        "universe_source": universe_source,
        "validated_universe": universe_source == WMB_UNIVERSE_VALIDATED,
        "validated_universe_asset": WMB_MARKER_UNIVERSE_FILE,
        "not_validated_reason": reason,
        "n_genes": len(universe),
        "universe_sha256": hashlib.sha256("\n".join(universe).encode()).hexdigest(),
        "panel_genes_absent": sorted(set(wanted) - reference_genes),
        "reference_genes": reference_genes,
    }


def _wmb_matrices(context: BuildContext) -> dict[str, Path]:
    """Return the WMB matrices the source record lists (never a new glob).

    The builder reads exactly the files whose identities ``build_hash``
    covers, so a matrix added to the directory during a build is not used.
    """
    record = context.sources.get(SOURCE_WMB_H5AD_DIR)
    if record is None:
        raise ReferenceBuildError(f"wmb_panel has no source {SOURCE_WMB_H5AD_DIR!r}")
    if record.kind == "file":
        paths = [Path(identity.path) for identity in record.files]
    else:
        paths = [
            Path(identity.path)
            for identity in record.files
            if Path(identity.path).name.endswith(WMB_H5AD_SUFFIX)
        ]
    return {
        path.name[: -len(WMB_H5AD_SUFFIX)]: path
        for path in sorted(paths)
        if path.name.endswith(WMB_H5AD_SUFFIX)
    }


def uncovered_nodes(
    mapping: TaxonomyTreeView, covered_leaves: Iterable[str]
) -> dict[str, list[dict[str, str]]]:
    """Return the mapping-tree nodes without any covered leaf, per level.

    Args:
        mapping: The mapping tree (Allen means).
        covered_leaves: Leaves present in the marker precompute.

    Returns:
        ``{level: [{"node": label, "name": name}, ...]}`` for every level,
        the leaf level included.
    """
    covered = set(covered_leaves)
    result: dict[str, list[dict[str, str]]] = {}
    for level in mapping.hierarchy:
        entries = []
        for node in mapping.nodes(level):
            if not covered.intersection(mapping.leaves_under(level, node)):
                entries.append({"node": node, "name": mapping.name(level, node)})
        result[level] = entries
    return result


def build_wmb_panel(context: BuildContext) -> dict[str, Any]:
    """Build the ``wmb_panel`` bundle (mouse primary; the validated configuration).

    Recipe (plan §3.2, §7.1; ``research/selfmap/build_marker_ref.py``,
    ``run_panel_markers.sh``, ``run_map2.sh``): (1) at most
    ``max_cells_per_cluster`` (50) cells per cluster from the local
    WMB-10Xv3 h5ads, excluding the self-map test cells (required while
    resolvability is enabled), restricted to the marker universe (the
    validated ag7 | VZG2 union for panels inside it, else the panel's own
    genes); (2) ``precompute_stats_abc`` -> the marker precompute; (3)
    reference markers and (4) query markers on the panel with the supertype
    level dropped and ``--n_processors`` / ``--max_gb`` explicit; (5) the
    Allen ``precomputed_stats_ABC_revision_230821.h5`` copied as the mapping
    precompute (sha256; md5 checked against the release); (6) the lookup
    filtered to the mapping tree (SUPT absent). Clusters and subclasses
    without any 10Xv3 marker-training cell are recorded
    (``uncovered_clusters``, ``uncovered_subclasses``,
    ``marker_unsupported_nodes``), never filled in.

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    output: dict[str, Any] = {"reference": "WMB (CCN20230722), 10Xv3 marker cells"}
    hierarchy = list(spec.hierarchy or WMB_HIERARCHY)
    drop_level = spec.drop_level
    matrices = _wmb_matrices(context)
    if not matrices:
        raise ReferenceBuildError(
            f"no *{WMB_H5AD_SUFFIX} under {_source_path(context, SOURCE_WMB_H5AD_DIR)}"
        )
    exclude: set[str] = set()
    test_cells: dict[str, Any] = {"excluded": False, "source": None}
    if SOURCE_WMB_TEST_CELLS in context.sources:
        test_record = context.sources[SOURCE_WMB_TEST_CELLS]
        exclude = read_cell_label_list(Path(test_record.path))
        digest = file_sha256(test_record.path)
        test_cells = {
            "excluded": True,
            "source": test_record.path,
            "sha256": digest,
            "n_cells": len(exclude),
            "matches_validated_test_set": digest == WMB_SELFMAP_TEST_CELLS_PIN.sha256,
        }
    elif config.resolvability.enabled:
        # The resolvability self-map (M3b) tests on these cells; training on
        # them would leak (plan §3.2, §8.3).
        raise ReferenceBuildError(
            "wmb_panel needs the self-map test cells (source "
            f"{SOURCE_WMB_TEST_CELLS!r}, annotation_wmb_selfmap_test_cells_path) "
            "while resolvability is enabled: the marker training cells must "
            "exclude them"
        )
    else:
        logger.warning(
            "wmb_panel is built without excluding the self-map test cells "
            "(resolvability disabled); the bundle records it"
        )
    with timer.step("sample_training_cells"):
        sampled, sampling = sample_wmb_training_cells(
            _source_path(context, SOURCE_WMB_CELL_METADATA),
            matrices,
            max_cells_per_cluster=spec.max_cells_per_cluster,
            exclude_cells=exclude,
        )
    universe, universe_summary = _wmb_gene_universe(context, panel, matrices)
    reference_genes = universe_summary.pop("reference_genes")
    if not set(universe) & set(panel.ensembl_ids):
        raise ReferenceBuildError(
            "no panel gene of wmb_panel is in the WMB expression matrices "
            f"({panel.n_genes} panel genes; species or gene-ID mismatch?)"
        )
    with timer.step("extract_panel_h5ads"):
        h5ads, kept, extraction = extract_panel_training_h5ads(
            sampled,
            matrices,
            universe,
            context.scratch_dir / "training_h5ad",
        )
    kept.to_csv(context.work_dir / MARKER_TRAINING_CELLS_FILE, index=False)
    cell_metadata = context.scratch_dir / "training_cell_metadata.csv"
    kept[["cell_label", "cluster_alias"]].to_csv(cell_metadata, index=False)
    marker_path = context.work_dir / MARKER_PRECOMPUTE_FILE
    tmp_dir = context.scratch_dir / "ctm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    timer.ctm(
        "marker_precompute",
        _run_ctm_precompute_abc,
        {
            "output_path": str(marker_path),
            "hierarchy": hierarchy,
            "h5ad_path_list": [str(path) for path in h5ads],
            "cell_metadata_path": str(cell_metadata),
            "cluster_annotation_path": str(
                _source_path(context, SOURCE_WMB_CLUSTER_ANNOTATION)
            ),
            "cluster_membership_path": str(
                _source_path(context, SOURCE_WMB_CLUSTER_MEMBERSHIP)
            ),
            "n_processors": prep_resources().n_processors,
            "tmp_dir": str(tmp_dir),
            "clobber": True,
            "normalization": "raw",
            "do_pruning": True,
        },
    )
    with timer.step("copy_mapping_precompute"):
        copied = context.copy_source(SOURCE_WMB_MAPPING_STATS, MAPPING_PRECOMPUTE_FILE)
    mapping_path = context.work_dir / MAPPING_PRECOMPUTE_FILE
    with timer.step("mapping_precompute_md5"):
        md5 = _file_md5(mapping_path)
    output["mapping_precompute"] = {
        "file": MAPPING_PRECOMPUTE_FILE,
        "source_path": copied.source_path,
        "sha256": copied.sha256,
        "md5": md5,
        "validated_release_md5": WMB_MAPPING_STATS_MD5,
        "matches_validated_release": md5 == WMB_MAPPING_STATS_MD5,
    }
    if md5 != WMB_MAPPING_STATS_MD5:
        logger.warning(
            "WMB mapping precompute %s is not the validated Allen release "
            "(md5 %s, expected %s; R28)",
            copied.source_path,
            md5,
            WMB_MAPPING_STATS_MD5,
        )
    universe_set = set(universe)
    marker_tree = TaxonomyTreeView.from_precompute(marker_path)
    mapping_full = TaxonomyTreeView.from_precompute(mapping_path)
    tree = mapping_tree(
        mapping_full, drop_level=drop_level, nodes_to_drop=spec.nodes_to_drop
    )
    markers = find_panel_markers(
        context,
        marker_precompute=marker_path,
        tree=tree,
        query_genes=[gene for gene in panel.ensembl_ids if gene in universe_set],
        drop_level=drop_level,
        timer=timer,
        keep_reference_markers=panel.n_genes > config.panel.large_panel_genes,
    )
    uncovered = uncovered_nodes(tree, marker_tree.nodes(marker_tree.leaf_level))
    output["uncovered_clusters"] = uncovered.get(tree.leaf_level, [])
    output["uncovered_subclasses"] = uncovered.get(WMB_SUBC, [])
    output["uncovered_nodes_per_level"] = {
        level: len(entries) for level, entries in uncovered.items()
    }
    output["marker_unsupported_nodes"] = marker_unsupported_nodes(
        tree, markers.validation, uncovered
    )
    output["known_limitations"] = [WMB_KNOWN_LIMITATION]
    output["selfmap_test_cells_excluded"] = bool(test_cells["excluded"])
    output["selfmap_test_cells"] = test_cells
    output["validated_configuration"] = {
        "mapping_release": md5 == WMB_MAPPING_STATS_MD5,
        "marker_universe": bool(universe_summary["validated_universe"]),
        "selfmap_test_cells": bool(test_cells.get("matches_validated_test_set")),
        "max_cells_per_cluster": spec.max_cells_per_cluster == 50,
    }
    output["validated_configuration"]["all"] = all(
        output["validated_configuration"].values()
    )
    with timer.step("profiles_and_negative_genes"):
        stats = read_precomputed_stats(marker_path, markers.query_genes)
        output.update(
            _profiles_and_negatives(
                context,
                stats=stats,
                tree=tree,
                taxonomy_id=WMB_TAXONOMY_ID,
                extra_detection=None,
                reference_label="wmb_panel",
                write_negatives=True,
            )
        )
    output.update(_write_common_files(context, tree, WMB_TAXONOMY_ID))
    output.update(_tree_output(tree, spec))
    output["marker_precompute"] = {
        "file": MARKER_PRECOMPUTE_FILE,
        "hierarchy": hierarchy,
        "sha256": file_sha256(marker_path),
        "node_counts": marker_tree.node_counts(),
        "training_cells_file": MARKER_TRAINING_CELLS_FILE,
        "sampling": sampling,
        "extraction": extraction,
        "gene_universe": universe_summary,
    }
    output["markers"] = _markers_output(context, markers, tree, config)
    output["collapsed_parents"] = output["markers"]["collapsed_parents"]
    output["panel_coverage"] = panel_coverage(
        panel,
        reference_genes,
        markers,
        tree,
        weak_parent_markers=config.panel.weak_parent_markers,
    )
    if _resolvability_enabled(context):
        # The test cells are outside the marker build and inside the Allen
        # means (~12k of 4M cells; negligible leakage, §8.3 step 1).
        output.update(
            _run_self_map(
                context,
                test_reference_id=WMB_TESTSET_REFERENCE_ID,
                test_sources=WMB_TESTSET_SOURCES,
                engine=_work_bundle(
                    context,
                    taxonomy_id=WMB_TAXONOMY_ID,
                    tree=tree,
                    lookup_sha256_value=markers.filtered_lookup_sha256,
                ),
                specs_for=lambda engine: _wmb_specs(engine, config),
                timer=timer,
            )
        )
    output.update(timer.to_json())
    return output


# --------------------------------------------------------------------------
# WMB region shares (mouse, panel-independent)


def merfish_region(
    division: pd.Series, structure: pd.Series, ap_mm: pd.Series
) -> pd.Series:
    """Return the E7 v2 region of MERFISH cells from their CCF parcellation.

    Region = CCF ``parcellation_division``, with fibre tracts -> ``fibre``,
    ventricles -> ``VS``, unassigned -> ``unassigned``, and OB split from
    OLF: MOB, AOB, the olfactory nerve layer (``In``) and OLF-unassigned
    anterior of AP 2.5 mm (``exp/E7/04_merfish_intrinsic_autoregion.py``;
    plan §7.8).

    Args:
        division: ``parcellation_division`` per cell.
        structure: ``parcellation_structure`` per cell.
        ap_mm: ``x_ccf`` (AP, mm) per cell.

    Returns:
        Region per cell.
    """
    division_text = division.astype(str)
    structure_text = structure.astype(str)
    region = division_text.copy()
    region[division_text.isin(FIBRE_DIVISIONS)] = "fibre"
    region[division_text.isin(VENTRICLE_DIVISIONS)] = "VS"
    region[division_text.isin(UNASSIGNED_DIVISIONS) | division.isna()] = "unassigned"
    is_ob = (
        ((division_text == "OLF") & structure_text.isin(OB_STRUCTURES))
        | (structure_text == OB_NERVE_STRUCTURE)
        | ((structure_text == OLF_UNASSIGNED_STRUCTURE) & (ap_mm < OB_ANTERIOR_AP_MM))
    )
    region[is_ob] = "OB"
    return region


def region_share_tables(
    cells: pd.DataFrame, *, window_half_width: int = AP_WINDOW_HALF_WIDTH
) -> dict[str, pd.DataFrame]:
    """Return the region-share and composition tables of MERFISH cells.

    Args:
        cells: One row per cell with ``brain_section_label``, ``class``,
            ``subclass``, ``x_ccf`` (AP, mm) and ``region``
            (``merfish_region``).
        window_half_width: Sections on each side of a window's centre (AP
            order); 1 gives three-section windows like E7's matched sections.

    Returns:
        ``region_share`` (level, node_name, region, n, share, n_grey, n_all;
        grey-matter cells), ``region_home`` (level, node_name, home,
        home_share, n_grey), ``section_composition`` (section, ap_ccf_mm,
        n_cells, level, node_name, n, freq; all cells), ``section_regions``
        (section, ap_ccf_mm, region, n, share; grey cells) and ``ap_windows``
        (window, ap_ccf_mm, sections, n_sections, ap_min_mm, ap_max_mm,
        n_cells, level, node_name, n, freq).
    """
    grey = cells[cells["region"].isin(GREY_REGIONS)]
    share_frames: list[pd.DataFrame] = []
    home_frames: list[pd.DataFrame] = []
    for level in ("class", "subclass"):
        counts = pd.crosstab(grey[level], grey["region"]).reindex(
            columns=list(GREY_REGIONS), fill_value=0
        )
        n_grey = counts.sum(axis=1)
        n_all = cells[level].value_counts()
        shares = counts.div(n_grey.replace(0, np.nan), axis=0).fillna(0.0)
        long = (
            counts.stack()
            .rename("n")
            .reset_index()
            .rename(columns={level: "node_name", "region": "region"})
        )
        long["share"] = shares.stack().to_numpy()
        long["n_grey"] = long["node_name"].map(n_grey).astype(int)
        long["n_all"] = long["node_name"].map(n_all).fillna(0).astype(int)
        long.insert(0, "level", level)
        share_frames.append(long)
        home = shares.idxmax(axis=1)
        home_frames.append(
            pd.DataFrame(
                {
                    "level": level,
                    "node_name": shares.index.astype(str),
                    "home": home.to_numpy(),
                    "home_share": shares.max(axis=1).to_numpy(),
                    "n_grey": n_grey.to_numpy(),
                }
            )
        )
    section_ap = cells.groupby("brain_section_label")["x_ccf"].median()
    section_n = cells["brain_section_label"].value_counts()
    composition_frames: list[pd.DataFrame] = []
    for level in ("class", "subclass"):
        counts = (
            cells.groupby(["brain_section_label", level])
            .size()
            .rename("n")
            .reset_index()
        )
        counts = counts.rename(
            columns={"brain_section_label": "section", level: "node_name"}
        )
        counts["n_cells"] = counts["section"].map(section_n).astype(int)
        counts["freq"] = counts["n"] / counts["n_cells"]
        counts["ap_ccf_mm"] = counts["section"].map(section_ap)
        counts.insert(1, "level", level)
        composition_frames.append(counts)
    composition = pd.concat(composition_frames, ignore_index=True)[
        ["section", "ap_ccf_mm", "n_cells", "level", "node_name", "n", "freq"]
    ]
    section_regions = (
        grey.groupby(["brain_section_label", "region"]).size().rename("n").reset_index()
    ).rename(columns={"brain_section_label": "section"})
    section_grey = section_regions.groupby("section")["n"].transform("sum")
    section_regions["share"] = section_regions["n"] / section_grey
    section_regions["ap_ccf_mm"] = section_regions["section"].map(section_ap)
    section_regions = section_regions[["section", "ap_ccf_mm", "region", "n", "share"]]
    ordered = section_ap.sort_values()
    sections = list(ordered.index)
    window_frames: list[pd.DataFrame] = []
    for index, centre in enumerate(sections):
        members = sections[
            max(0, index - window_half_width) : index + window_half_width + 1
        ]
        member_cells = cells[cells["brain_section_label"].isin(members)]
        for level in ("class", "subclass"):
            counts = member_cells[level].value_counts()
            frame = pd.DataFrame(
                {
                    "window": centre,
                    "ap_ccf_mm": float(ordered[centre]),
                    "sections": ";".join(members),
                    "n_sections": len(members),
                    "ap_min_mm": float(ordered[members].min()),
                    "ap_max_mm": float(ordered[members].max()),
                    "n_cells": len(member_cells),
                    "level": level,
                    "node_name": counts.index.astype(str),
                    "n": counts.to_numpy(),
                    "freq": (counts / len(member_cells)).to_numpy(),
                }
            )
            window_frames.append(frame)
    return {
        "region_share": pd.concat(share_frames, ignore_index=True),
        "region_home": pd.concat(home_frames, ignore_index=True),
        "section_composition": composition,
        "section_regions": section_regions,
        "ap_windows": pd.concat(window_frames, ignore_index=True)
        if window_frames
        else pd.DataFrame(),
    }


def build_wmb_region_share(context: BuildContext) -> dict[str, Any]:
    """Build the ``wmb_region_share`` bundle (mouse, panel-independent).

    Recipe (plan §3.2, §7.2, §7.8; ``exp/E7/01_merfish_tables.py`` with the
    v2 region scheme of ``04_merfish_intrinsic_autoregion.py``): subclass and
    class x CCF-division shares of grey-matter cells with OB split from OLF,
    each node's home division, per-section compositions and region shares,
    and AP-indexed composition windows (each section with its neighbours)
    for every MERFISH-C57BL6J-638850 section.

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    source = _source_path(context, SOURCE_MERFISH_CCF_METADATA)
    with timer.step("read_merfish_metadata"):
        cells = pd.read_csv(source, usecols=list(MERFISH_COLUMNS), engine="pyarrow")
        cells["region"] = merfish_region(
            cells["parcellation_division"],
            cells["parcellation_structure"],
            cells["x_ccf"],
        )
    with timer.step("region_tables"):
        tables = region_share_tables(cells)
    files = {
        "region_share": REGION_SHARE_FILE,
        "region_home": REGION_HOME_FILE,
        "section_composition": SECTION_COMPOSITION_FILE,
        "section_regions": SECTION_REGIONS_FILE,
        "ap_windows": AP_WINDOWS_FILE,
    }
    for key, name in files.items():
        _write_parquet(tables[key], context.work_dir / name)
    with timer.step("source_md5"):
        md5 = _file_md5(source)
    section_ap = cells.groupby("brain_section_label")["x_ccf"].median().sort_values()
    region_counts = cells["region"].value_counts()
    windows = tables["ap_windows"]
    return {
        "reference": "ABC MERFISH-C57BL6J-638850-CCF cell metadata",
        "region_scheme": REGION_SCHEME,
        "grey_regions": list(GREY_REGIONS),
        "files": files,
        "n_cells": int(len(cells)),
        "n_grey_cells": int(cells["region"].isin(GREY_REGIONS).sum()),
        "cells_per_region": {
            str(key): int(value) for key, value in region_counts.items()
        },
        "n_sections": int(len(section_ap)),
        "ap_range_mm": [float(section_ap.min()), float(section_ap.max())],
        "sections": [
            {"section": str(section), "ap_ccf_mm": float(ap)}
            for section, ap in section_ap.items()
        ],
        "n_windows": int(windows["window"].nunique()) if len(windows) else 0,
        "window_half_width": AP_WINDOW_HALF_WIDTH,
        "n_classes": int(cells["class"].nunique()),
        "n_subclasses": int(cells["subclass"].nunique()),
        "source_md5": md5,
        "source_matches_allen_release": md5 == MERFISH_CCF_METADATA_MD5,
        **timer.to_json(),
    }


# --------------------------------------------------------------------------
# Whole-WHB cortex panel (human, optional)


def cortex_implausible_superclusters() -> list[str]:
    """Return the WHB superclusters implausible in frontal cortex (16 in E1 (ii)).

    Returns:
        ``"<CCN202210140_SUPC>/<label>"`` entries from the packaged vocab.
    """
    vocab = load_vocab("whb_supercluster")
    frame = vocab.frame
    column = f"region_plausible_{WHB_WHOLE_CORTEX_REGION}"
    return sorted(
        f"{WHB_SUPC}/{label}"
        for label, plausible in zip(frame["label"], frame[column], strict=True)
        if not bool(plausible)
    )


def build_whb_whole_ctx(context: BuildContext) -> dict[str, Any]:
    """Build the optional ``whb_whole_ctx_panel`` bundle.

    Recipe (plan §3.2; ``research/insilico/06_wb_panel_markers.py``,
    ``08_make_ctx_precompute.py``, ``exp/E1/06_filter_ctx_lookup.py``): drop
    the cortex-implausible superclusters from the whole-WHB precompute (the
    mapping precompute), find panel markers on the unpruned precompute, and
    filter the lookup to the pruned tree. Refused above 1,000 genes (~75 h).

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    if panel.n_genes > WHB_WHOLE_MAX_GENES:
        raise ReferenceBuildError(
            f"whb_whole_ctx_panel is refused above {WHB_WHOLE_MAX_GENES} genes "
            f"(panel has {panel.n_genes}; plan §3.2, §8.7)"
        )
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    source = _source_path(context, SOURCE_WHB_WHOLE_PRECOMPUTE)
    nodes = list(spec.nodes_to_drop) or cortex_implausible_superclusters()
    mapping_path = context.work_dir / MAPPING_PRECOMPUTE_FILE
    timer.ctm(
        "drop_nodes",
        _run_ctm_drop_nodes,
        source,
        mapping_path,
        parse_nodes_to_drop(nodes),
    )
    tree = mapping_tree(
        TaxonomyTreeView.from_precompute(mapping_path), drop_level=spec.drop_level
    )
    reference_genes = precompute_genes(source)
    reference_set = set(reference_genes)
    markers = find_panel_markers(
        context,
        marker_precompute=source,
        tree=tree,
        query_genes=[gene for gene in panel.ensembl_ids if gene in reference_set],
        drop_level=spec.drop_level,
        timer=timer,
        keep_reference_markers=False,
    )
    output: dict[str, Any] = {"reference": "whole WHB, cortex-implausible dropped"}
    with timer.step("profiles"):
        stats = read_precomputed_stats(mapping_path, markers.query_genes)
        output.update(
            _profiles_and_negatives(
                context,
                stats=stats,
                tree=tree,
                taxonomy_id=WHB_TAXONOMY_ID,
                extra_detection=None,
                reference_label="whb_whole_ctx",
                write_negatives=False,
            )
        )
    output.update(_write_common_files(context, tree, WHB_TAXONOMY_ID))
    output.update(_tree_output(tree, spec))
    output["nodes_dropped"] = nodes
    output["mapping_precompute"] = {
        "file": MAPPING_PRECOMPUTE_FILE,
        "sha256": file_sha256(mapping_path),
    }
    output["markers"] = _markers_output(context, markers, tree, config)
    output["collapsed_parents"] = output["markers"]["collapsed_parents"]
    output["panel_coverage"] = panel_coverage(
        panel,
        reference_genes,
        markers,
        tree,
        weak_parent_markers=config.panel.weak_parent_markers,
    )
    output.update(timer.to_json())
    return output


# --------------------------------------------------------------------------
# Resolvability test sets and the self-map (plan §3.2, §8.3; M3b)


def membership_labels(
    membership_path: Path | str, term_sets: Sequence[str]
) -> pd.DataFrame:
    """Return each cluster alias's term label per taxonomy level.

    Args:
        membership_path: Allen ``cluster_to_cluster_annotation_membership.csv``.
        term_sets: Term-set labels (levels), e.g. ``CCN202210140_SUPC``.

    Returns:
        Indexed by ``cluster_alias`` (int), one column per term set.
    """
    frame = pd.read_csv(
        membership_path,
        usecols=[
            "cluster_alias",
            "cluster_annotation_term_set_label",
            "cluster_annotation_term_label",
        ],
    )
    frame = frame[frame["cluster_annotation_term_set_label"].isin(list(term_sets))]
    pivot = frame.pivot_table(
        index="cluster_alias",
        columns="cluster_annotation_term_set_label",
        values="cluster_annotation_term_label",
        aggfunc="first",
    )
    missing = [level for level in term_sets if level not in pivot.columns]
    if missing:
        raise ReferenceBuildError(f"{membership_path} has no terms at {missing}")
    pivot.index = pivot.index.astype(int)
    return pivot[list(term_sets)]


def holdout_donor(donors: pd.Series, requested: str) -> str:
    """Return the held-out donor (``auto``: the donor with the fewest cells).

    Rev3 correction (§3.2): the default H19.30.002 has the *fewest* frontal
    WHB cells (32,606 of 125,481), as in the E2 pilot.

    Args:
        donors: Donor label per candidate cell.
        requested: ``annotation_calibration_holdout_donor``.

    Returns:
        The donor.

    Raises:
        ReferenceBuildError: If a named donor has no cell.
    """
    counts = donors.astype(str).value_counts()
    if counts.empty:
        raise ReferenceBuildError("no donor among the frontal WHB cells")
    if requested == "auto":
        smallest = int(counts.min())
        return sorted(str(donor) for donor, n in counts.items() if n == smallest)[0]
    if requested not in counts.index:
        raise ReferenceBuildError(
            f"held-out donor {requested!r} has no frontal WHB cell "
            f"(donors: {sorted(counts.index)})"
        )
    return requested


def read_panel_counts(
    matrices: Mapping[str, Path],
    cells: pd.DataFrame,
    genes: Sequence[str],
    *,
    block_rows: int = 5000,
) -> sparse.csr_matrix:
    """Read the panel-restricted raw counts of some cells, in their order.

    Args:
        matrices: h5ad path per feature-matrix label.
        cells: ``cell_label`` and ``feature_matrix_label`` per cell.
        genes: Gene IDs kept (must be in every matrix), in this order.
        block_rows: Rows read per block from the backed matrices.

    Returns:
        Cells x genes counts (CSR, float64), in ``cells`` order.

    Raises:
        ReferenceBuildError: If a matrix is missing, lacks a gene or a cell.
    """
    import anndata as ad
    import scipy.sparse as sp

    labels = cells["cell_label"].astype(str).to_numpy()
    blocks: list[sp.csr_matrix] = []
    order: list[np.ndarray] = []
    for matrix_label, group in cells.groupby("feature_matrix_label", sort=True):
        path = matrices.get(str(matrix_label))
        if path is None:
            raise ReferenceBuildError(f"no expression matrix for {matrix_label!r}")
        source = ad.read_h5ad(path, backed="r")
        try:
            positions = source.obs_names.get_indexer(group["cell_label"].astype(str))
            if (positions < 0).any():
                absent = group["cell_label"].astype(str).to_numpy()[positions < 0]
                raise ReferenceBuildError(
                    f"{path} lacks {int((positions < 0).sum())} cells, e.g. "
                    f"{list(absent[:5])}"
                )
            gene_index = source.var_names.get_indexer(list(genes))
            if (gene_index < 0).any():
                absent_genes = [
                    gene for gene, idx in zip(genes, gene_index, strict=True) if idx < 0
                ]
                raise ReferenceBuildError(
                    f"{path} lacks panel genes {absent_genes[:10]}"
                )
            sort = np.argsort(positions, kind="stable")
            sorted_positions = positions[sort]
            parts = [
                sp.csr_matrix(source.X[sorted_positions[start : start + block_rows]])[
                    :, gene_index
                ]
                for start in range(0, len(sorted_positions), block_rows)
            ]
        finally:
            source.file.close()
        blocks.append(sp.vstack(parts).tocsr().astype(np.float64))
        rows = np.flatnonzero(
            cells["feature_matrix_label"].astype(str).to_numpy() == str(matrix_label)
        )
        order.append(rows[sort])
        logger.info("Read %d cells x %d genes from %s", len(group), len(genes), path)
    if not blocks:
        return sp.csr_matrix((0, len(genes)), dtype=np.float64)
    stacked = sp.vstack(blocks).tocsr()
    positions_in_input = np.concatenate(order)
    inverse = np.empty_like(positions_in_input)
    inverse[positions_in_input] = np.arange(len(positions_in_input))
    result = stacked[inverse]
    if len(labels) != result.shape[0]:  # pragma: no cover - defensive
        raise ReferenceBuildError("panel count extraction lost cells")
    return result.tocsr()


def _ho_region_metadata(context: BuildContext) -> pd.DataFrame:
    """Return the frontal WHB cells (donor, cluster alias) of the test set."""
    columns = ["cell_label", "feature_matrix_label", "donor_label", "cluster_alias"]
    if SOURCE_WHB_REGION_CELL_METADATA in context.sources:
        path = _source_path(context, SOURCE_WHB_REGION_CELL_METADATA)
        return pd.read_csv(path, usecols=columns + ["region_of_interest_label"])
    if (
        SOURCE_WHB_CELL_METADATA in context.sources
        and SOURCE_WHB_ROI_MAP in context.sources
    ):
        from merxen.analysis.mapmycells import _write_region_cell_metadata

        path = context.scratch_dir / REGION_CELL_METADATA_FILE
        _write_region_cell_metadata(
            cell_metadata_path=_source_path(context, SOURCE_WHB_CELL_METADATA),
            output_path=path,
            region_labels=list(WHB_FRONTAL_ROI_LABELS),
            min_cells_per_leaf=WHB_FRONTAL_MIN_CELLS_PER_LEAF,
            roi_map_path=_source_path(context, SOURCE_WHB_ROI_MAP),
            region_column="region_of_interest_label",
        )
        return pd.read_csv(path, usecols=columns + ["region_of_interest_label"])
    raise ReferenceBuildError(
        f"{HO_REFERENCE_ID} needs {SOURCE_WHB_REGION_CELL_METADATA!r} (the region "
        f"reference directory's {REGION_CELL_METADATA_FILE}) or "
        f"{SOURCE_WHB_CELL_METADATA!r} + {SOURCE_WHB_ROI_MAP!r}"
    )


def _other_region_metadata(context: BuildContext, labels: pd.DataFrame) -> pd.DataFrame:
    """Return the WHB cells of the other-region dissections, with WHB levels.

    Reads the WHB cell metadata (only the columns the draw needs) and keeps
    the non-neuronal nuclei of ``HO_OTHER_REGION_ROI_LABELS``.

    Args:
        context: The build context (``whb_cell_metadata`` source).
        labels: WHB level labels per cluster alias (``membership_labels``).

    Returns:
        ``cell_label``, ``feature_matrix_label``, ``donor_label``,
        ``cluster_alias``, ``region_of_interest_label`` and the WHB levels.
    """
    path = _source_path(context, SOURCE_WHB_CELL_METADATA)
    frame = pd.read_csv(
        path,
        usecols=[
            "cell_label",
            "feature_matrix_label",
            "donor_label",
            "cluster_alias",
            "region_of_interest_label",
        ],
        dtype={
            "cell_label": str,
            "feature_matrix_label": str,
            "donor_label": str,
            "region_of_interest_label": str,
        },
    )
    frame = frame[
        frame["region_of_interest_label"].isin(list(HO_OTHER_REGION_ROI_LABELS))
        & (frame["feature_matrix_label"] == HO_OTHER_REGION_MATRIX)
    ]
    return frame.join(labels, on="cluster_alias")


def _matrix_genes(path: Path) -> list[str]:
    import anndata as ad

    source = ad.read_h5ad(path, backed="r")
    try:
        return [str(gene) for gene in source.var_names]
    finally:
        source.file.close()


def _write_training_h5ads(
    counts: sparse.csr_matrix,
    cells: pd.DataFrame,
    genes: Sequence[str],
    directory: Path,
) -> list[Path]:
    """Write one panel-restricted training h5ad per feature matrix."""
    import anndata as ad

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    matrix_labels = cells["feature_matrix_label"].astype(str).to_numpy()
    for label in sorted(set(matrix_labels)):
        rows = np.flatnonzero(matrix_labels == label)
        adata = ad.AnnData(
            X=counts[rows].astype(np.float32),
            obs=pd.DataFrame(index=cells["cell_label"].astype(str).to_numpy()[rows]),
            var=pd.DataFrame(index=list(genes)),
        )
        path = directory / f"{label}-panel-raw.h5ad"
        adata.write_h5ad(path)
        written.append(path)
    return written


def _spill_groups(truth_labels: Iterable[str], taxonomy_id: str) -> list[str]:
    """Return the vocab broad class of truth nodes (spill groups)."""
    from merxen.annotation.vocab import load_vocab

    table = load_vocab(
        "whb_supercluster" if taxonomy_id == WHB_TAXONOMY_ID else "wmb_class"
    )
    label_to_name = table.label_to_name()
    groups = []
    for label in truth_labels:
        name = label_to_name.get(str(label))
        groups.append(
            table.broad_class(name)
            if name is not None and name in table
            else UNASSIGNED_LABEL
        )
    return groups


def nonneuronal_superclusters(labels: Iterable[str]) -> list[str]:
    """Return the WHB superclusters of a non-neuronal vocab broad class.

    Neurons and the ``Mixed/Unknown`` superclusters (Miscellaneous,
    Splatter, Ependymal, Bergmann glia, Choroid plexus) are left out.

    Args:
        labels: WHB supercluster labels (``CS202210140_*``).

    Returns:
        The non-neuronal labels, sorted.
    """
    table = load_vocab("whb_supercluster")
    label_to_name = table.label_to_name()
    kept = []
    for label in {str(value) for value in labels}:
        name = label_to_name.get(label)
        if name is None or name not in table:
            continue
        if table.broad_class(name) not in {NEURONS, UNASSIGNED_LABEL}:
            kept.append(label)
    return sorted(kept)


def ho_truth_exclusions(labels: Iterable[str], region: str) -> dict[str, str]:
    """Return the WHB superclusters that cannot be held-out test truths.

    A test cell's truth must be nameable by a production call at every
    level it is scored at (``resolvability.whb_level_specs``): superclusters
    that are sinks, have no floor class (the ``Mixed/Unknown`` broad class:
    no broad or supercluster truth) or are implausible in ``region`` (calls
    to them are excluded) are left out (``HO_TRUTH_EXCLUSION_VERSION``).

    Args:
        labels: WHB supercluster labels (``CS202210140_*``) of the pool.
        region: The anatomical region token (``frontal_cortex``).

    Returns:
        Excluded label -> reason (``sink``, ``no_floor_class`` or
        ``region_implausible``, the first that applies).
    """
    table = load_vocab("whb_supercluster")
    label_to_name = table.label_to_name()
    excluded: dict[str, str] = {}
    for label in sorted({str(value) for value in labels}):
        name = label_to_name.get(label)
        if name is None or name not in table:
            excluded[label] = HO_EXCLUDED_NO_FLOOR_CLASS
            continue
        broad = table.broad_class(name)
        group = broad if broad not in (None, UNASSIGNED_LABEL) else None
        if table.is_sink(name):
            excluded[label] = HO_EXCLUDED_SINK
        elif human_floor_class(group, table.nt(name)) is None:
            excluded[label] = HO_EXCLUDED_NO_FLOOR_CLASS
        elif not table.is_region_plausible(name, region):
            excluded[label] = HO_EXCLUDED_REGION_IMPLAUSIBLE
    return excluded


def other_region_candidates(
    cell_metadata: pd.DataFrame,
    *,
    reference_cells: set[str],
    training_superclusters: Iterable[str],
    training_clusters: set[str],
    roi_labels: Sequence[str] = HO_OTHER_REGION_ROI_LABELS,
    feature_matrix: str = HO_OTHER_REGION_MATRIX,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], int, int]:
    """Return the admissible other-region candidates (``other_region_test_cells``).

    Non-neuronal nuclei (``feature_matrix``) of the dissections
    ``roi_labels``, of non-neuronal superclusters the training reference
    holds, not reference cells, of training clusters only
    (``HO_OTHER_REGION_VERSION`` 2), one row per cell label.

    Returns:
        ``(candidates indexed by cell label, candidates of clusters the
        training reference lacks, eligible superclusters, cells in the ROIs,
        reference cells dropped)``.
    """
    frame = cell_metadata[
        cell_metadata["region_of_interest_label"].astype(str).isin(list(roi_labels))
        & (cell_metadata["feature_matrix_label"].astype(str) == feature_matrix)
    ]
    frame = frame.dropna(subset=list(WHB_SOURCE_HIERARCHY))
    mappable = {str(label) for label in training_superclusters}
    eligible = set(nonneuronal_superclusters(frame[WHB_SUPC].astype(str))) & mappable
    frame = frame[frame[WHB_SUPC].astype(str).isin(eligible)]
    n_in_rois = int(len(frame))
    is_reference = frame["cell_label"].astype(str).isin(reference_cells)
    n_reference = int(is_reference.sum())
    frame = frame[~is_reference]
    is_unseen = ~frame[WHB_CLUS].astype(str).isin(training_clusters)
    unseen_candidates = frame[is_unseen]
    frame = frame[~is_unseen]
    frame = frame.drop_duplicates("cell_label").set_index("cell_label", drop=False)
    frame.index = frame.index.astype(str)
    return frame, unseen_candidates, eligible, n_in_rois, n_reference


def whb_supercluster_classes(labels: Iterable[str]) -> dict[str, str | None]:
    """Return the leaf-level class of each WHB supercluster label (COP apart)."""
    from merxen.annotation.vocab import floor_class_for

    label_to_name = load_vocab("whb_supercluster").label_to_name()
    result: dict[str, str | None] = {}
    for label in {str(value) for value in labels}:
        name = label_to_name.get(label)
        try:
            result[label] = (
                None
                if name is None
                else floor_class_for(name, species="human", level="supercluster")
            )
        except (KeyError, ValueError):
            result[label] = None
    return result


def whb_class_top_up(
    test_rows: pd.DataFrame,
    *,
    donor_pool: pd.DataFrame,
    other_candidates: pd.DataFrame,
    rule: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top the human test set's thin classes up (plan §8.3 v7.6; version 7).

    Classes are the supercluster level's classes (the E2 floor classes, COP
    apart). Pools in order: the held-out donor's cells the test set does not
    hold (``donor_pool``: its admissible cells), then, for non-neuronal
    classes only, the other-region candidates not yet drawn
    (``other_region_candidates``: training clusters only, never a frontal
    reference or other donor's cell); stratified by supercluster.

    Args:
        test_rows: The test cells so far (``cell_label``, WHB levels,
            ``TEST_SOURCE_COLUMN``).
        donor_pool: The held-out donor's admissible cells.
        other_candidates: The admissible other-region cells.
        rule: ``top_up_rule("human", config)``.

    Returns:
        ``(test rows with the top-up appended, record)``.
    """
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si

    held = set(test_rows["cell_label"].astype(str))
    donor = donor_pool[~donor_pool["cell_label"].astype(str).isin(held)].copy()
    donor["_pool"] = "holdout_donor"
    other = other_candidates[
        ~other_candidates["cell_label"].astype(str).isin(held)
    ].copy()
    other["_pool"] = "other_region"
    columns = list(test_rows.columns.drop(TEST_SOURCE_COLUMN, errors="ignore"))
    candidates = pd.concat(
        [donor.reset_index(drop=True), other.reset_index(drop=True)],
        ignore_index=True,
    )
    classes = whb_supercluster_classes(
        list(candidates[WHB_SUPC].astype(str)) + list(test_rows[WHB_SUPC].astype(str))
    )
    candidates["_class"] = candidates[WHB_SUPC].astype(str).map(classes)
    candidates = candidates.dropna(subset=["_class"])
    candidates = candidates.set_index(candidates["cell_label"].astype(str), drop=False)
    have_classes = test_rows[WHB_SUPC].astype(str).map(classes).dropna()
    have = {str(k): int(v) for k, v in have_classes.value_counts().items()}
    non_neuronal = sorted(
        str(cls)
        for cls in classes.values()
        if cls is not None and si.is_neuronal_class(str(cls), "human") is False
    )
    chosen, record = res.class_top_up(
        candidates,
        class_column="_class",
        leaf_column=WHB_SUPC,
        have=have,
        target=int(rule["target"]),
        seed=int(rule["seed"]),
        pool_column="_pool",
        pool_order=["holdout_donor", "other_region"],
        pool_classes={"other_region": non_neuronal},
    )
    added = candidates.loc[list(chosen)].reset_index(drop=True)
    added[TEST_SOURCE_COLUMN] = np.where(
        added["_pool"].astype(str) == "holdout_donor",
        TOP_UP_SOURCE,
        TOP_UP_SOURCE_OTHER_REGION,
    )
    record["rule"] = {**dict(rule), "summary": record["rule"]}
    record["disjoint_from_test_cells"] = not (
        set(added["cell_label"].astype(str)) & held
    )
    record["per_pool"] = {
        str(k): int(v) for k, v in added["_pool"].astype(str).value_counts().items()
    }
    rows = pd.concat(
        [test_rows, added[[*columns, TEST_SOURCE_COLUMN]]], ignore_index=True
    )
    return rows, record


def other_region_test_cells(
    cell_metadata: pd.DataFrame,
    *,
    reference_cells: Iterable[str],
    training_superclusters: Iterable[str],
    training_clusters: Iterable[str],
    have: Mapping[str, int],
    cap: int,
    room: int | None,
    roi_labels: Sequence[str] = HO_OTHER_REGION_ROI_LABELS,
    feature_matrix: str = HO_OTHER_REGION_MATRIX,
    seed: int = HO_OTHER_REGION_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Draw other-region non-neuronal test cells for the human test set.

    User decision 2026-09-27 (H18 follow-up; E2
    ``research/insilico/01b_extract_nonneurons.py``): WHB non-neuronal
    nuclei (``feature_matrix``) from neocortical dissections outside the
    frontal reference (``roi_labels``), of the non-neuronal superclusters
    (``nonneuronal_superclusters``) the held-out training reference holds,
    top every non-neuronal supercluster of the test set up to the
    per-supercluster cap (``res.top_up_test_cells``). Only candidates of
    clusters the held-out training reference holds (``training_clusters``:
    the kept training clusters, at least ``HO_MIN_TRAINING_CELLS_PER_CLUSTER``
    cells) are drawn (``HO_OTHER_REGION_VERSION`` 2, user decision
    2026-09-27): a cell of a cluster the reference lacks has a cluster truth
    no call can name. The candidates left out are counted per supercluster
    and the draw is checked to hold none. Candidates among
    ``reference_cells`` (every frontal region cell: the training, marker and
    held-out donor cells) are dropped and counted; the draw is checked
    disjoint from them.

    Args:
        cell_metadata: WHB cell metadata joined with the WHB levels
            (``cell_label``, ``feature_matrix_label``, ``donor_label``,
            ``cluster_alias``, ``region_of_interest_label`` and the
            supercluster, cluster and subcluster labels).
        reference_cells: Cell labels of the frontal reference cells.
        training_superclusters: Supercluster labels of the held-out training
            reference (a drawn cell's supercluster must be one).
        training_clusters: Cluster labels of the held-out training reference
            (a drawn cell's cluster must be one).
        have: Held-out donor test cells per supercluster.
        cap: The test set's per-supercluster cap.
        room: Test cells the set may still take (``n_test_cells`` - donor).
        roi_labels: Dissections to draw from.
        feature_matrix: WHB feature matrix of the candidates.
        seed: Sampling seed.

    Returns:
        ``(rows, record)``: the drawn cells (``cell_metadata`` columns) and
        what ``bundle.json`` records (dissections, counts per supercluster,
        broad class, dissection and donor, the cluster rule and its
        exclusions, the reference-cell exclusions, disjointness).

    Raises:
        ReferenceBuildError: If a drawn cell is a reference cell or of a
            cluster the training reference lacks.
    """
    from merxen.annotation import resolvability as res

    reference = {str(label) for label in reference_cells}
    kept_clusters = {str(label) for label in training_clusters}
    frame, unseen_candidates, eligible, n_in_rois, n_reference = (
        other_region_candidates(
            cell_metadata,
            reference_cells=reference,
            training_superclusters=training_superclusters,
            training_clusters=kept_clusters,
            roi_labels=roi_labels,
            feature_matrix=feature_matrix,
        )
    )
    chosen = res.top_up_test_cells(
        frame, stratum=WHB_SUPC, have=have, cap=cap, room=room, seed=seed
    )
    rows = frame.loc[chosen].reset_index(drop=True)
    overlap = sorted(set(rows["cell_label"].astype(str)) & reference)
    if overlap:  # pragma: no cover - guarded by the filter above
        raise ReferenceBuildError(
            f"other-region test cells overlap the reference cells: {overlap[:5]}"
        )
    unseen = ~rows[WHB_CLUS].astype(str).isin(kept_clusters)
    if unseen.any():  # pragma: no cover - guarded by the filter above
        raise ReferenceBuildError(
            "other-region test cells of clusters the held-out training reference "
            f"lacks: {sorted(set(rows[WHB_CLUS][unseen].astype(str)))[:5]}"
        )
    supc = rows[WHB_SUPC].astype(str)
    groups = _spill_groups(supc, WHB_TAXONOMY_ID)

    def counts(values: Iterable[str]) -> dict[str, int]:
        series = pd.Series(list(values), dtype=object)
        return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}

    available = frame[WHB_SUPC].astype(str).value_counts()
    record = {
        "version": HO_OTHER_REGION_VERSION,
        "source": "WHB cell metadata (other dissections, non-neuronal nuclei)",
        "roi_labels": list(roi_labels),
        "feature_matrix": feature_matrix,
        "seed": int(seed),
        "cap_per_supercluster": int(cap),
        "eligible_superclusters": sorted(eligible),
        "n_candidates_in_rois": n_in_rois,
        "n_excluded_reference_cells": n_reference,
        "cluster_rule": HO_OTHER_REGION_RULE,
        "min_training_cells_per_cluster": HO_MIN_TRAINING_CELLS_PER_CLUSTER,
        "n_training_clusters": len(kept_clusters),
        "n_excluded_cluster_not_in_training": int(len(unseen_candidates)),
        "n_excluded_clusters": int(unseen_candidates[WHB_CLUS].astype(str).nunique()),
        "excluded_cluster_not_in_training_per_supercluster": counts(
            unseen_candidates[WHB_SUPC].astype(str)
        ),
        "n_candidates": int(len(frame)),
        "candidates_per_supercluster": {
            str(k): int(v) for k, v in available.sort_index().items()
        },
        "n_cells": int(len(rows)),
        "per_supercluster": counts(supc),
        "per_broad_class": counts(groups),
        "per_region": counts(rows["region_of_interest_label"].astype(str)),
        "per_donor": counts(rows["donor_label"].astype(str)),
        "n_cluster_not_in_training": int(unseen.sum()),
        "disjoint_from_reference_cells": not overlap,
        "n_reference_cells_checked": len(reference),
    }
    return rows, record


def build_whb_frontal_ho(context: BuildContext) -> dict[str, Any]:
    """Build the ``whb_frontal_supc_clus_ho`` bundle (human, resolvability).

    Recipe (plan §3.2, §8.3; ``exp/E2/make_ho_ref.py``, ``run_all.sh``,
    ``research/insilico/01_extract.py``): the frontal WHB cells (A44-A45,
    A46, A32, ACC) restricted to the panel; the held-out donor (``auto``:
    the one with the fewest frontal cells, H19.30.002) is removed from the
    training cells, which keep clusters with at least 5 cells; a
    supercluster -> cluster -> subcluster precompute of the training cells
    (raw normalisation, as the pilot) truncated to supercluster -> cluster
    is the mapping precompute, with panel markers (``n_per_utility``); the
    held-out donor's cells of superclusters a production call can name
    (``ho_truth_exclusions``: no sinks, no ``Mixed/Unknown``, no
    region-implausible ones; E2; recorded in ``test_set.truth_exclusion``),
    at most 1,000 per supercluster (all cells of rare ones) and at most
    ``n_test_cells``, are the test set
    (``test_cells.h5ad`` with native panel counts and their truth), and
    every non-neuronal supercluster is topped up to that cap with WHB
    non-neuronal nuclei of neocortical dissections outside the frontal ROIs,
    only of clusters the training reference keeps
    (``other_region_test_cells``; user decisions 2026-09-27), recorded in
    ``bundle.json`` (``test_set.other_region``: the rule, what it left out)
    and per cell (``test_source``).

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    from merxen.annotation import resolvability as res

    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    output: dict[str, Any] = {
        "reference": "WHB frontal region without the held-out donor (E2 HO recipe)"
    }
    with timer.step("region_cells"):
        metadata = _ho_region_metadata(context)
        frontal_cells = set(metadata["cell_label"].astype(str))
        labels = membership_labels(
            _source_path(context, SOURCE_WHB_CLUSTER_MEMBERSHIP), WHB_SOURCE_HIERARCHY
        )
        metadata = metadata.join(labels, on="cluster_alias")
        metadata = metadata.dropna(subset=list(WHB_SOURCE_HIERARCHY))
    donor = holdout_donor(metadata["donor_label"], config.resolvability.holdout_donor)
    donor_counts = {
        str(key): int(value)
        for key, value in metadata["donor_label"].astype(str).value_counts().items()
    }
    is_test_donor = metadata["donor_label"].astype(str) == donor
    training = metadata[~is_test_donor]
    per_cluster = training[WHB_CLUS].value_counts()
    kept_clusters = per_cluster[per_cluster >= HO_MIN_TRAINING_CELLS_PER_CLUSTER].index
    dropped_clusters = sorted(set(per_cluster.index) - set(kept_clusters))
    training = training[training[WHB_CLUS].isin(kept_clusters)]
    pool = metadata[is_test_donor].set_index("cell_label", drop=False)
    region = config.anatomical_region or WHB_WHOLE_CORTEX_REGION
    exclusions = ho_truth_exclusions(pool[WHB_SUPC].astype(str), region)
    is_excluded = pool[WHB_SUPC].astype(str).isin(list(exclusions)).to_numpy()
    excluded_counts = pool[is_excluded][WHB_SUPC].astype(str).value_counts()
    n_pool_before = int(len(pool))
    pool = pool[~is_excluded]
    label_to_name = load_vocab("whb_supercluster").label_to_name()
    truth_exclusion = {
        "version": HO_TRUTH_EXCLUSION_VERSION,
        "rule": "held-out donor cells whose truth supercluster is a sink, has no "
        "floor class (Mixed/Unknown) or is implausible in the region are not "
        "test cells",
        "region": region,
        "n_pool_cells_before": n_pool_before,
        "n_excluded_pool_cells": int(is_excluded.sum()),
        "excluded_superclusters": {
            label: {
                "name": label_to_name.get(label),
                "reason": exclusions[label],
                "n_pool_cells": int(excluded_counts.get(label, 0)),
            }
            for label in sorted(exclusions)
        },
    }
    logger.info(
        "%s: %d of %d held-out donor cells left out (truth superclusters no call "
        "can name: %s)",
        HO_REFERENCE_ID,
        int(is_excluded.sum()),
        n_pool_before,
        ", ".join(
            f"{label_to_name.get(label, label)} {int(excluded_counts.get(label, 0))} "
            f"({reason})"
            for label, reason in sorted(exclusions.items())
        )
        or "none",
    )
    n_test = config.resolvability.n_test_cells or HO_DEFAULT_TEST_CELLS
    chosen = res.select_test_cells(
        pool,
        stratum=WHB_SUPC,
        max_per_stratum=HO_MAX_TEST_CELLS_PER_SUPERCLUSTER,
        n_max=n_test,
        seed=TEST_SET_SEED,
    )
    test_rows = pool.loc[chosen].reset_index(drop=True)
    test_rows[TEST_SOURCE_COLUMN] = TEST_SOURCE_DONOR
    cap = res.stratum_cap(
        pool.groupby(WHB_SUPC, observed=True).size(),
        HO_MAX_TEST_CELLS_PER_SUPERCLUSTER,
        n_test,
    )
    with timer.step("other_region_cells"):
        other_rows, other_record = other_region_test_cells(
            _other_region_metadata(context, labels),
            reference_cells=frontal_cells | set(training["cell_label"].astype(str)),
            training_superclusters=training[WHB_SUPC].astype(str).unique(),
            training_clusters=[str(value) for value in kept_clusters],
            have={
                str(key): int(value)
                for key, value in test_rows[WHB_SUPC].astype(str).value_counts().items()
            },
            cap=cap,
            room=max(0, int(n_test) - len(test_rows)),
        )
    logger.info(
        "%s: %d held-out donor test cells + %d other-region non-neuronal cells (%s)",
        HO_REFERENCE_ID,
        len(test_rows),
        len(other_rows),
        ", ".join(f"{k}: {v}" for k, v in other_record["per_broad_class"].items()),
    )
    other_rows[TEST_SOURCE_COLUMN] = TEST_SOURCE_OTHER_REGION
    test_rows = pd.concat(
        [test_rows, other_rows[list(test_rows.columns)]], ignore_index=True
    )
    plan = (
        resolvability_plan(spec, panel, config)
        if config.resolvability.enabled
        else None
    )
    top_up_record: dict[str, Any] | None = None
    if (
        plan is not None
        and plan.version == res.RESOLVABILITY_VERSION_V7
        and plan.top_up is not None
    ):
        with timer.step("class_top_up"):
            candidates, _unseen, _eligible, _n_rois, _n_reference = (
                other_region_candidates(
                    _other_region_metadata(context, labels),
                    reference_cells=frontal_cells
                    | set(training["cell_label"].astype(str)),
                    training_superclusters=training[WHB_SUPC].astype(str).unique(),
                    training_clusters={str(value) for value in kept_clusters},
                )
            )
            test_rows, top_up_record = whb_class_top_up(
                test_rows,
                donor_pool=pool.reset_index(drop=True),
                other_candidates=candidates.reset_index(drop=True),
                rule=plan.top_up,
            )
        logger.info(
            "%s: class top-up added %d test cells",
            HO_REFERENCE_ID,
            top_up_record["n_cells"],
        )
    matrices = {
        "WHB-10Xv3-Neurons": _source_path(context, SOURCE_WHB_NEURONS_H5AD),
        "WHB-10Xv3-Nonneurons": _source_path(context, SOURCE_WHB_NONNEURONS_H5AD),
    }
    reference_genes = set(_matrix_genes(matrices["WHB-10Xv3-Neurons"]))
    genes = sorted(gene for gene in panel.ensembl_ids if gene in reference_genes)
    if not genes:
        raise ReferenceBuildError(
            f"no panel gene of {spec.reference_id!r} is in the WHB matrices "
            f"({panel.n_genes} panel genes; species or gene-ID mismatch?)"
        )
    cells = pd.concat([training.reset_index(drop=True), test_rows], ignore_index=True)
    with timer.step("extract_panel_counts"):
        counts = read_panel_counts(matrices, cells, genes)
    n_training = len(training)
    training_counts = counts[:n_training]
    test_counts = counts[n_training:]
    h5ads = _write_training_h5ads(
        training_counts,
        cells.iloc[:n_training],
        genes,
        context.scratch_dir / "training_h5ad",
    )
    cell_metadata = context.scratch_dir / "training_cell_metadata.csv"
    cells.iloc[:n_training][["cell_label", "cluster_alias"]].to_csv(
        cell_metadata, index=False
    )
    source_path = context.scratch_dir / SOURCE_PRECOMPUTE_FILE
    tmp_dir = context.scratch_dir / "ctm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    timer.ctm(
        "precompute_training",
        _run_ctm_precompute_abc,
        {
            "output_path": str(source_path),
            "hierarchy": list(WHB_SOURCE_HIERARCHY),
            "h5ad_path_list": [str(path) for path in h5ads],
            "cell_metadata_path": str(cell_metadata),
            "cluster_annotation_path": str(
                _source_path(context, SOURCE_WHB_CLUSTER_ANNOTATION)
            ),
            "cluster_membership_path": str(
                _source_path(context, SOURCE_WHB_CLUSTER_MEMBERSHIP)
            ),
            "n_processors": prep_resources().n_processors,
            "split_by_dataset": False,
            "do_pruning": True,
            "tmp_dir": str(tmp_dir),
            "clobber": True,
            "normalization": "raw",
        },
    )
    hierarchy = list(spec.hierarchy or WHB_FRONTAL_HIERARCHY)
    mapping_path = context.work_dir / MAPPING_PRECOMPUTE_FILE
    timer.ctm(
        "truncate_taxonomy",
        _run_ctm_truncate,
        {
            "input_path": str(source_path),
            "output_path": str(mapping_path),
            "new_hierarchy": hierarchy,
        },
    )
    tree = mapping_tree(
        TaxonomyTreeView.from_precompute(mapping_path),
        drop_level=spec.drop_level,
        nodes_to_drop=spec.nodes_to_drop,
    )
    markers = find_panel_markers(
        context,
        marker_precompute=mapping_path,
        tree=tree,
        query_genes=genes,
        drop_level=spec.drop_level,
        timer=timer,
        keep_reference_markers=panel.n_genes > config.panel.large_panel_genes,
    )
    output.update(_write_common_files(context, tree, WHB_TAXONOMY_ID))
    output.update(_tree_output(tree, spec))
    output["mapping_precompute"] = {
        "file": MAPPING_PRECOMPUTE_FILE,
        "hierarchy": hierarchy,
        "sha256": file_sha256(mapping_path),
        "n_training_cells": int(n_training),
    }
    output["markers"] = _markers_output(context, markers, tree, config)
    output["collapsed_parents"] = output["markers"]["collapsed_parents"]
    output["panel_coverage"] = panel_coverage(
        panel,
        reference_genes,
        markers,
        tree,
        weak_parent_markers=config.panel.weak_parent_markers,
    )
    truth_supc = test_rows[WHB_SUPC].astype(str).to_numpy()
    obs = pd.DataFrame(
        {
            f"{res.TRUTH_PREFIX}{WHB_SUPC}": truth_supc,
            f"{res.TRUTH_PREFIX}{WHB_CLUS}": test_rows[WHB_CLUS].astype(str).to_numpy(),
            f"{res.TRUTH_PREFIX}{WHB_SUBC}": test_rows[WHB_SUBC].astype(str).to_numpy(),
            res.TRUTH_LEAF_COLUMN: truth_supc,
            res.SPILL_GROUP_COLUMN: _spill_groups(truth_supc, WHB_TAXONOMY_ID),
            "donor_label": test_rows["donor_label"].astype(str).to_numpy(),
            "feature_matrix_label": test_rows["feature_matrix_label"]
            .astype(str)
            .to_numpy(),
            "region_of_interest_label": test_rows["region_of_interest_label"]
            .astype(str)
            .to_numpy(),
            TEST_SOURCE_COLUMN: test_rows[TEST_SOURCE_COLUMN].astype(str).to_numpy(),
        },
        index=pd.Index(test_rows["cell_label"].astype(str).to_numpy(), name="cell_id"),
    )
    test = res.HeldOutCells(counts=test_counts, genes=list(genes), obs=obs)
    written = res.write_test_cells(test, context.work_dir)
    native = test.native_counts
    output["test_set"] = {
        **written,
        "holdout_donor": donor,
        "holdout_donor_requested": config.resolvability.holdout_donor,
        "donor_cells": donor_counts,
        "n_pool_cells": int(len(pool)),
        "n_test_cells_requested": int(n_test),
        "max_per_supercluster": HO_MAX_TEST_CELLS_PER_SUPERCLUSTER,
        "cap_per_supercluster": int(cap),
        "per_source": {
            str(key): int(value)
            for key, value in test_rows[TEST_SOURCE_COLUMN].value_counts().items()
        },
        "other_region": other_record,
        **(
            {}
            if plan is None or plan.version != res.RESOLVABILITY_VERSION_V7
            else {
                "resolvability_version": plan.version,
                "class_top_up": top_up_record,
            }
        ),
        "truth_exclusion": truth_exclusion,
        "seed": TEST_SET_SEED,
        "per_supercluster": {
            str(key): int(value)
            for key, value in pd.Series(truth_supc).value_counts().sort_index().items()
        },
        "n_training_cells": int(n_training),
        "n_training_clusters": int(len(kept_clusters)),
        "training_clusters_dropped": [str(value) for value in dropped_clusters],
        "min_training_cells_per_cluster": HO_MIN_TRAINING_CELLS_PER_CLUSTER,
        "native_counts_median": float(np.median(native)) if len(native) else None,
    }
    output.update(timer.to_json())
    return output


def wmb_class_top_up(
    cells: pd.DataFrame,
    meta: pd.DataFrame,
    sampled: pd.DataFrame,
    used: Iterable[str],
    rule: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top the mouse test set's thin WMB classes up (plan §8.3 v7.6; version 7).

    Candidates: 10Xv3 cells of the available matrices that neither the
    marker build trained on (``sampled``, the ``wmb_panel`` training sample)
    nor the test set holds (``used`` and ``cells``), only of clusters with
    at least ``min_training_cells_per_cluster`` marker-training cells (the
    training-cluster rule), at most ``max_cluster_frac`` of a cluster's
    10Xv3 cells each (the gate-P bound); ``resolvability.class_top_up``
    draws them per class, stratified by subclass.

    Args:
        cells: The test cells so far (``cell_label``, WMB levels,
            ``cluster_alias``).
        meta: Every candidate 10Xv3 cell with its WMB levels.
        sampled: The marker-training sample.
        used: Cell labels already excluded (self-map list and training).
        rule: ``top_up_rule("mouse", config)``.

    Returns:
        ``(cells with the top-up appended, record)``.

    Raises:
        ReferenceBuildError: If a drawn cell is a training or test cell or of
            a cluster outside the training-cluster rule.
    """
    from merxen.annotation import resolvability as res

    training = set(sampled["cell_label"].astype(str))
    test_ids = set(cells["cell_label"].astype(str))
    per_cluster = sampled["cluster_alias"].astype(int).value_counts()
    minimum = int(rule["min_training_cells_per_cluster"])
    kept = {int(alias) for alias, count in per_cluster.items() if count >= minimum}
    excluded = set(used) | training | test_ids
    is_candidate = ~meta["cell_label"].astype(str).isin(excluded)
    n_before_rule = int(is_candidate.sum())
    in_training_cluster = meta["cluster_alias"].astype(int).isin(kept)
    candidates = meta[is_candidate & in_training_cluster]
    candidates = candidates.set_index(candidates["cell_label"].astype(str), drop=False)
    sizes = meta["cluster_alias"].astype(int).value_counts()
    fraction = float(rule["max_cluster_frac"])
    caps = {
        str(alias): int(math.floor(fraction * size)) for alias, size in sizes.items()
    }
    have = {
        str(key): int(value)
        for key, value in cells[WMB_CLAS].astype(str).value_counts().items()
    }
    chosen, record = res.class_top_up(
        candidates.assign(_cluster=candidates["cluster_alias"].astype(int).astype(str)),
        class_column=WMB_CLAS,
        leaf_column=WMB_SUBC,
        have=have,
        target=int(rule["target"]),
        seed=int(rule["seed"]),
        cluster_column="_cluster",
        cluster_cap=caps,
    )
    added = candidates.loc[list(chosen)].reset_index(drop=True)
    leaked = sorted(set(added["cell_label"].astype(str)) & (training | test_ids))
    outside = sorted(set(added["cluster_alias"].astype(int)) - kept)
    if leaked or outside:  # pragma: no cover - guarded by the filters above
        raise ReferenceBuildError(
            f"class top-up drew training / test cells {leaked[:5]} or cells of "
            f"clusters outside the training-cluster rule {outside[:5]}"
        )
    taken = added["cluster_alias"].astype(int).value_counts()
    record.update(
        {
            "rule": {**dict(rule), "summary": record["rule"]},
            "n_candidates_before_cluster_rule": n_before_rule,
            "n_excluded_cluster_not_in_training": int(
                (is_candidate & ~in_training_cluster).sum()
            ),
            "max_cluster_share_taken": float(
                max((taken[alias] / sizes[alias] for alias in taken.index), default=0.0)
            ),
            "disjoint_from_training": not (
                set(added["cell_label"].astype(str)) & training
            ),
            "disjoint_from_test_cells": not (
                set(added["cell_label"].astype(str)) & test_ids
            ),
        }
    )
    added = added.assign(test_source=TOP_UP_SOURCE)
    return pd.concat([cells, added[list(cells.columns)]], ignore_index=True), record


def build_wmb_selfmap_testset(context: BuildContext) -> dict[str, Any]:
    """Build the ``wmb_selfmap_testset`` bundle (mouse, resolvability).

    Recipe (plan §3.2, §8.3; ``research/selfmap/build_queries.py``): the
    11,913 self-map test cells (<= 10 WMB-10Xv3 cells per supertype;
    ``truth.csv``), plus up to ``mouse_nonneuronal_extra_per_supertype``
    (30) cells per supertype of the non-neuronal classes (Astro-Epen,
    OPC-Oligo, Vascular, Immune) drawn from 10Xv3 cells the ``wmb_panel``
    marker build did not use (its training sample is recomputed with the
    same rule, so that build is unchanged), restricted to the panel with
    native counts and truth at class, subclass, supertype and cluster.

    Args:
        context: The build context.

    Returns:
        ``bundle.json`` builder output.
    """
    from merxen.annotation import resolvability as res

    spec = context.spec
    panel = _panel_of(context)
    config = _config_of(context)
    timer = _StepTimer(context.work_dir / CTM_LOG_DIR)
    matrices = _wmb_matrices(context)
    if not matrices:
        raise ReferenceBuildError(
            f"no *{WMB_H5AD_SUFFIX} under {_source_path(context, SOURCE_WMB_H5AD_DIR)}"
        )
    base = read_cell_label_list(_source_path(context, SOURCE_WMB_TEST_CELLS))
    base_digest = file_sha256(_source_path(context, SOURCE_WMB_TEST_CELLS))
    with timer.step("training_sample"):
        sampled, sampling = sample_wmb_training_cells(
            _source_path(context, SOURCE_WMB_CELL_METADATA),
            matrices,
            max_cells_per_cluster=spec.max_cells_per_cluster,
            exclude_cells=base,
        )
    with timer.step("candidate_cells"):
        meta = pd.read_csv(
            _source_path(context, SOURCE_WMB_CELL_METADATA),
            usecols=[
                "cell_label",
                "feature_matrix_label",
                "library_method",
                "cluster_alias",
            ],
            dtype={"cluster_alias": "Int64"},
        )
        meta = meta[meta["library_method"] == WMB_LIBRARY_METHOD].dropna(
            subset=["cluster_alias"]
        )
        meta = meta[meta["feature_matrix_label"].isin(set(matrices))]
        meta["cluster_alias"] = meta["cluster_alias"].astype(int)
        labels = membership_labels(
            _source_path(context, SOURCE_WMB_CLUSTER_MEMBERSHIP), WMB_HIERARCHY
        )
        meta = meta.join(labels, on="cluster_alias").dropna(subset=list(WMB_HIERARCHY))
    base_rows = meta[meta["cell_label"].isin(base)].copy()
    if (
        config.resolvability.n_test_cells
        and len(base_rows) > config.resolvability.n_test_cells
    ):
        chosen = res.select_test_cells(
            base_rows.set_index("cell_label", drop=False),
            stratum=WMB_SUPT,
            max_per_stratum=len(base_rows),
            n_max=config.resolvability.n_test_cells,
            seed=TEST_SET_SEED,
        )
        base_rows = base_rows[base_rows["cell_label"].isin(set(chosen))]
    base_rows["test_source"] = "selfmap_truth"
    used = set(base) | set(sampled["cell_label"].astype(str))
    pool = meta[
        ~meta["cell_label"].isin(used)
        & meta[WMB_CLAS].isin(list(WMB_NONNEURONAL_TEST_CLASSES))
    ]
    extra_per_supertype = config.resolvability.mouse_nonneuronal_extra_per_supertype
    parts = [
        group.sample(
            n=min(len(group), extra_per_supertype), random_state=WMB_TESTSET_EXTRA_SEED
        )
        for _, group in pool.groupby(WMB_SUPT, sort=True)
        if extra_per_supertype > 0
    ]
    extra = pd.concat(parts) if parts else pool.iloc[0:0]
    extra = extra.assign(test_source="nonneuronal_extra")
    cells = pd.concat([base_rows, extra], ignore_index=True)
    plan = (
        resolvability_plan(spec, panel, config)
        if config.resolvability.enabled
        else None
    )
    top_up_record: dict[str, Any] | None = None
    if (
        plan is not None
        and plan.version == res.RESOLVABILITY_VERSION_V7
        and plan.top_up is not None
    ):
        with timer.step("class_top_up"):
            cells, top_up_record = wmb_class_top_up(
                cells, meta, sampled, used, plan.top_up
            )
        logger.info(
            "%s: class top-up added %d test cells (%s)",
            WMB_TESTSET_REFERENCE_ID,
            top_up_record["n_cells"],
            ", ".join(
                f"{cls} {item['before']} -> {item['after']}"
                for cls, item in top_up_record["per_class"].items()
            ),
        )
    reference_genes = set(_matrix_genes(next(iter(matrices.values()))))
    genes = sorted(gene for gene in panel.ensembl_ids if gene in reference_genes)
    if not genes:
        raise ReferenceBuildError(
            f"no panel gene of {spec.reference_id!r} is in the WMB matrices "
            f"({panel.n_genes} panel genes; species or gene-ID mismatch?)"
        )
    with timer.step("extract_panel_counts"):
        counts = read_panel_counts(matrices, cells, genes)
    truth_class = cells[WMB_CLAS].astype(str).to_numpy()
    obs = pd.DataFrame(
        {
            **{
                f"{res.TRUTH_PREFIX}{level}": cells[level].astype(str).to_numpy()
                for level in WMB_HIERARCHY
            },
            res.TRUTH_LEAF_COLUMN: cells[WMB_SUBC].astype(str).to_numpy(),
            res.SPILL_GROUP_COLUMN: _spill_groups(truth_class, WMB_TAXONOMY_ID),
            "test_source": cells["test_source"].astype(str).to_numpy(),
            "feature_matrix_label": cells["feature_matrix_label"]
            .astype(str)
            .to_numpy(),
        },
        index=pd.Index(cells["cell_label"].astype(str).to_numpy(), name="cell_id"),
    )
    test = res.HeldOutCells(counts=counts, genes=list(genes), obs=obs)
    written = res.write_test_cells(test, context.work_dir)
    grid = (
        list(plan.grid)
        if plan is not None and plan.version == res.RESOLVABILITY_VERSION_V7
        else spec.resolved_depth_grid(panel.n_genes)
    )
    _write_json(
        context.work_dir / DEPTH_GRID_FILE,
        {
            "depth_grid": grid,
            "source": "spec" if spec.depth_grid is not None else "default",
            "species": spec.species,
            "n_panel_genes": panel.n_genes,
        },
    )
    native = test.native_counts
    per_class = pd.Series(truth_class).value_counts().sort_index()
    version_7: dict[str, Any] = {}
    if plan is not None and plan.version == res.RESOLVABILITY_VERSION_V7:
        version_7 = {
            "resolvability_version": plan.version,
            "class_top_up": top_up_record,
            "n_class_top_up": int((obs["test_source"] == TOP_UP_SOURCE).sum()),
        }
    return {
        "reference": "WMB self-map test cells (10Xv3), panel-restricted",
        "test_set": {
            **written,
            "n_selfmap_truth_cells": int((obs["test_source"] == "selfmap_truth").sum()),
            "n_selfmap_list": len(base),
            "selfmap_list_sha256": base_digest,
            "matches_validated_test_set": base_digest
            == WMB_SELFMAP_TEST_CELLS_PIN.sha256,
            "n_nonneuronal_extra": int(len(extra)),
            "nonneuronal_classes": list(WMB_NONNEURONAL_TEST_CLASSES),
            "extra_per_supertype": int(extra_per_supertype),
            "extra_seed": WMB_TESTSET_EXTRA_SEED,
            "marker_training_sample": {
                "n_cells": int(sampling["n_sampled_cells"]),
                "disjoint": bool(
                    not set(obs.index) & set(sampled["cell_label"].astype(str))
                ),
            },
            "per_class": {str(key): int(value) for key, value in per_class.items()},
            "native_counts_median": float(np.median(native)) if len(native) else None,
            **version_7,
        },
        "depth_grid": grid,
        **timer.to_json(),
    }


def _test_set_sources(
    context: BuildContext, reference_id: str, names: Sequence[str]
) -> dict[str, Path]:
    """Return the test-set source paths of a self-map, or fail if one is missing.

    Builders call it before their marker steps, so a missing held-out source
    fails the build in seconds rather than after them (M3b review).

    Raises:
        ReferenceBuildError: If a test-set source is missing.
    """
    available = {
        name: Path(context.sources[name].path)
        for name in names
        if name in context.sources
    }
    if reference_id == HO_REFERENCE_ID:
        has_metadata = SOURCE_WHB_REGION_CELL_METADATA in available or (
            SOURCE_WHB_CELL_METADATA in context.sources
            and SOURCE_WHB_ROI_MAP in context.sources
        )
        missing = [] if has_metadata else [SOURCE_WHB_REGION_CELL_METADATA]
        if SOURCE_WHB_REGION_CELL_METADATA not in available:
            for name in (SOURCE_WHB_CELL_METADATA, SOURCE_WHB_ROI_MAP):
                if name in context.sources:
                    available[name] = Path(context.sources[name].path)
        missing += [
            name
            for name in names
            if name != SOURCE_WHB_REGION_CELL_METADATA and name not in available
        ]
    else:
        missing = [name for name in names if name not in available]
    if missing:
        raise ReferenceBuildError(
            f"{context.spec.reference_id}: the resolvability self-map (enabled) "
            f"needs the test-set source(s) {missing} for {reference_id}; pass the "
            "WHB h5ad and metadata directories (annotation_whb_h5ad_dir, "
            "annotation_whb_metadata_dir, annotation_whb_region_precompute_source) "
            "or the WMB sources, or disable annotation resolvability"
        )
    return available


def _check_self_map_sources(
    context: BuildContext, reference_id: str, names: Sequence[str]
) -> None:
    """Fail fast when resolvability is on and a test-set source is missing."""
    if _resolvability_enabled(context):
        _test_set_sources(context, reference_id, names)


def _test_set_spec(
    context: BuildContext, reference_id: str, names: Sequence[str]
) -> AnnotationReferenceSpec:
    """Return the spec of a primary bundle's resolvability test set.

    The test set is built from the primary spec's own source files (so the
    primary ``build_hash`` covers it) and the WHB (human) or WMB (mouse)
    reference settings, so the WHB and SEA-AD self-maps share one held-out
    bundle.
    """
    config = _config_of(context)
    available = _test_set_sources(context, reference_id, names)
    template_id = (
        "whb_frontal_supc_clus" if reference_id == HO_REFERENCE_ID else "wmb_panel"
    )
    template = next(
        (item for item in config.references if item.reference_id == template_id),
        None,
    )
    if template is None or context.spec.reference_id == template_id:
        template = context.spec
    known = KNOWN_REFERENCES[reference_id]
    return AnnotationReferenceSpec(
        reference_id=reference_id,
        species=known["species"],
        role=known["role"],
        sources=dict(sorted(available.items())),
        hierarchy=list(known.get("hierarchy", [])),
        n_per_utility=template.n_per_utility,
        max_cells_per_cluster=template.max_cells_per_cluster,
        bootstrap_factor=template.bootstrap_factor,
        bootstrap_iteration=template.bootstrap_iteration,
        rng_seed=template.rng_seed,
        depth_grid=template.depth_grid,
    )


def _work_bundle(
    context: BuildContext,
    *,
    taxonomy_id: str,
    tree: TaxonomyTreeView,
    lookup_sha256_value: str,
) -> Any:
    """Return an ``MmcBundle`` view of the bundle being built (self-mapping)."""
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.store import ANNOTATION_BUILDER_VERSION

    panel = _panel_of(context)
    return MmcBundle(
        reference_id=context.spec.reference_id,
        role=context.spec.role,
        species=context.spec.species,
        build_hash=context.build_hash,
        path=context.work_dir,
        panel_hash=panel.panel_hash,
        n_panel_genes=panel.n_genes,
        taxonomy_id=taxonomy_id,
        levels=tuple(tree.hierarchy),
        drop_level=context.spec.drop_level,
        mapping_precompute=context.work_dir / MAPPING_PRECOMPUTE_FILE,
        lookup=context.work_dir / QUERY_MARKERS_FILTERED_FILE,
        lookup_sha256=lookup_sha256_value,
        mapping_tree=context.work_dir / MAPPING_TREE_FILE,
        vocab_snapshot=context.work_dir / VOCAB_SNAPSHOT_FILE,
        builder_version=ANNOTATION_BUILDER_VERSION,
        manifest={},
    )


def mmc_map_function(
    engine: Any,
    *,
    spec: AnnotationReferenceSpec,
    config: AnnotationConfig,
    scratch_dir: Path,
    log_dir: Path,
    runs: list[dict[str, Any]],
) -> Any:
    """Return a self-map ``map_fn``: MapMyCells with the production settings.

    Args:
        engine: The ``MmcBundle`` the simulated cells are mapped onto.
        spec: The reference spec (bootstrap settings, seed).
        config: The annotation config (ctm version).
        scratch_dir: Scratch for queries, restricted lookups and MMC output.
        log_dir: Where MMC's logs go.
        runs: Receives one record per mapping run (wall time, peak RSS).

    Returns:
        ``(query, tag, seed) -> MMC tidy table``.
    """
    from merxen.annotation.mapmycells_engine import (
        MmcEngineParams,
        read_tidy_parquet,
        restrict_lookup,
        run_mmc,
        write_query_h5ad,
    )

    scratch = scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)

    def map_query(query: Any, tag: str, seed: int) -> pd.DataFrame:
        name = f"{engine.reference_id}__{tag}"
        query_path = write_query_h5ad(
            query.counts, list(query.obs.index), query.genes, scratch / f"{name}.h5ad"
        )
        restricted = restrict_lookup(
            engine, query.genes, scratch / f"{name}.lookup.json"
        )
        params = MmcEngineParams(
            bootstrap_factor=spec.bootstrap_factor,
            bootstrap_iteration=spec.bootstrap_iteration,
            rng_seed=spec.rng_seed + int(seed),
            n_processors=prep_resources().n_processors,
        )
        result = run_mmc(
            query_path,
            engine,
            params,
            output_parquet=scratch / f"{name}.parquet",
            work_dir=scratch / "mmc",
            expected_ctm_version=config.ctm_version,
            lookup_path=restricted.path,
            log_dir=log_dir,
        )
        tidy, _ = read_tidy_parquet(result.parquet)
        runs.append(
            {
                "tag": tag,
                "engine": engine.reference_id,
                "engine_build_hash": engine.build_hash,
                "n_cells": result.n_cells,
                "n_query_genes": result.n_query_genes,
                "wall_s": result.wall_s,
                "peak_rss_gb": result.peak_rss_gb,
                "n_processors": params.n_processors,
                "rng_seed": params.rng_seed,
            }
        )
        query_path.unlink(missing_ok=True)
        result.parquet.unlink(missing_ok=True)
        return tidy

    return map_query


def _mmc_map_function(
    context: BuildContext, engine: Any, runs: list[dict[str, Any]]
) -> Any:
    """Return the PREP self-map's ``map_fn`` (``mmc_map_function``)."""
    return mmc_map_function(
        engine,
        spec=context.spec,
        config=_config_of(context),
        scratch_dir=context.scratch_dir / "resolvability",
        log_dir=context.work_dir / CTM_LOG_DIR / "resolvability",
        runs=runs,
    )


def self_map_rule_settings(config: AnnotationConfig) -> Any:
    """Return the resolvability rule settings of a config (PREP's self-map)."""
    from merxen.annotation import resolvability as res

    return res.RuleSettings.from_config(
        config.resolvability,
        config.thresholds,
        trust_max_depth=config.panel.trust_max_depth,
        broad_only_min_class_share=config.panel.broad_only_min_class_share,
        hard_floor=config.thresholds.hard_min_counts or config.min_counts or 10,
    )


def level_specs_for(
    reference_id: str, engine: Any, config: AnnotationConfig
) -> list[Any]:
    """Return the self-map level specs of a primary or secondary reference.

    Args:
        reference_id: ``whb_frontal_supc_clus``, ``seaad_mr_panel`` or
            ``wmb_panel``.
        engine: The ``MmcBundle`` mapped onto.
        config: The annotation config.

    Returns:
        The level specs.

    Raises:
        ReferenceBuildError: For a reference without a self-map.
    """
    from merxen.annotation import resolvability as res

    if reference_id == "whb_frontal_supc_clus":
        return _whb_specs(engine, config)
    if reference_id == "seaad_mr_panel":
        return res.seaad_level_specs(config.thresholds)
    if reference_id == "wmb_panel":
        return _wmb_specs(engine, config)
    raise ReferenceBuildError(f"{reference_id} has no resolvability self-map")


def cells_rules_for(reference_id: str, config: AnnotationConfig) -> list[Any]:
    """Return the production rules the self-map applies to its cells table."""
    return _whb_cells_rules(config) if reference_id == "whb_frontal_supc_clus" else []


def _resolvability_enabled(context: BuildContext) -> bool:
    config = _config_of(context)
    return bool(config.resolvability.enabled) and context.spec.role in (
        "primary",
        "secondary",
    )


def _run_self_map(
    context: BuildContext,
    *,
    test_reference_id: str,
    test_sources: Sequence[str],
    engine: Any | None,
    specs_for: Any,
    timer: _StepTimer,
    cells_rules: Sequence[Any] = (),
) -> dict[str, Any]:
    """Run the resolvability self-map of a bundle and write its tables (§8.3).

    Args:
        context: The build context of the primary or secondary bundle.
        test_reference_id: The test-set reference (held-out WHB, WMB set).
        test_sources: Source names the test set is built from.
        engine: The bundle to map onto (``None``: the test-set bundle, i.e.
            the held-out WHB reference).
        specs_for: ``engine -> level specs``.
        timer: Step timer.
        cells_rules: Production rules applied to the cells table before the
            decisions (``resolvability.whb_cells_rules``).

    Returns:
        Builder output: ``resolvability`` (compact record) and, when
        resolvability forces one, ``panel_trust``.
    """
    from merxen.annotation import resolvability as res
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.vocab import load_floor_table

    if context.store is None:
        raise ReferenceBuildError(
            f"{context.spec.reference_id}: the resolvability self-map needs the "
            "reference store (build through ReferenceStore.get_or_build)"
        )
    config = _config_of(context)
    panel = _panel_of(context)
    test_spec = _test_set_spec(context, test_reference_id, test_sources)
    with timer.step("resolvability_test_set"):
        test_ref = context.store.get_or_build(test_spec, panel, config=config)
    test = res.load_test_cells(Path(test_ref.path))
    mapped_engine = engine if engine is not None else MmcBundle.from_dir(test_ref.path)
    runs: list[dict[str, Any]] = []
    plan = resolvability_plan(context.spec, panel, config)
    if plan.version == res.RESOLVABILITY_VERSION_V7:
        return _run_self_map_v7(
            context,
            plan=plan,
            test=test,
            test_ref=test_ref,
            engine=engine,
            mapped_engine=mapped_engine,
            specs_for=specs_for,
            timer=timer,
            cells_rules=cells_rules,
        )
    grid = context.spec.resolved_depth_grid(panel.n_genes)
    settings = self_map_rule_settings(config)
    with timer.step("resolvability_self_map"):
        result = res.run_resolvability(
            test,
            specs=specs_for(mapped_engine),
            depths=grid,
            recipes=res.simulation_recipes(config.resolvability, seed=TEST_SET_SEED),
            map_fn=_mmc_map_function(context, mapped_engine, runs),
            settings=settings,
            species=context.spec.species,
            floor_table=load_floor_table(context.spec.species),
            fine_seed_check=bool(config.thresholds.allow_fine_levels),
            cells_rules=cells_rules,
            provenance={
                "reference_id": context.spec.reference_id,
                "engine": {
                    "reference_id": mapped_engine.reference_id,
                    "build_hash": mapped_engine.build_hash,
                    "self": engine is not None,
                },
                "test_set_bundle": {
                    "reference_id": test_ref.reference_id,
                    "build_hash": test_ref.build_hash,
                    "path": test_ref.path,
                },
                "mapping_runs": runs,
                "decisions_note": (
                    "decision settings are recorded, not hashed; RESOLVE "
                    "re-derives the decisions from resolvability_cells.parquet "
                    "reweighted to each dataset's composition"
                ),
            },
        )
        result.write(context.work_dir)
    output: dict[str, Any] = {"resolvability": result.bundle_output()}
    output["resolvability"]["test_set_bundle"] = {
        "reference_id": test_ref.reference_id,
        "build_hash": test_ref.build_hash,
    }
    output["resolvability"]["mapping_runs"] = runs
    if result.trust.state is not None:
        output["panel_trust"] = result.trust.state
    return output


def _run_self_map_v7(
    context: BuildContext,
    *,
    plan: ResolvabilityPlan,
    test: Any,
    test_ref: Any,
    engine: Any | None,
    mapped_engine: Any,
    specs_for: Any,
    timer: _StepTimer,
    cells_rules: Sequence[Any] = (),
) -> dict[str, Any]:
    """Run a version-7 self-map (plan §8.3 v7) and write its tables.

    The plan's members are simulated on the (topped-up) test set and mapped
    one at a time; ``resolvability.run_resolvability_v7`` decides the
    ensemble, applies the saturated-bp rule and the monotone fill, and
    writes the four version-7 files (``resolvability_class_depth.parquet``
    included). Floors and the trust constraint come from the decisions before
    the fill.
    """
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.store import BUNDLE_MANIFEST_NAME
    from merxen.annotation.vocab import load_floor_table

    config = _config_of(context)
    runs: list[dict[str, Any]] = []
    manifest_path = Path(test_ref.path) / BUNDLE_MANIFEST_NAME
    test_output = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("builder_output")
        or {}
        if manifest_path.is_file()
        else {}
    )
    top_up = (test_output.get("test_set") or {}).get("class_top_up")
    profile = (
        si.profile_from_asset(plan.profile_asset)
        if plan.profile_asset is not None
        else None
    )
    ensemble = res.EnsembleSettings.from_config(config.resolvability)
    with timer.step("resolvability_self_map"):
        result = res.run_resolvability_v7(
            test,
            specs=specs_for(mapped_engine),
            depths=list(plan.grid),
            members=list(plan.members),
            map_fn=_mmc_map_function(context, mapped_engine, runs),
            settings=self_map_rule_settings(config),
            ensemble=ensemble,
            species=context.spec.species,
            floor_table=load_floor_table(context.spec.species),
            fine_seed_check=bool(config.thresholds.allow_fine_levels),
            cells_rules=cells_rules,
            profile=profile,
            provenance={
                "reference_id": context.spec.reference_id,
                "engine": {
                    "reference_id": mapped_engine.reference_id,
                    "build_hash": mapped_engine.build_hash,
                    "self": engine is not None,
                },
                "test_set_bundle": {
                    "reference_id": test_ref.reference_id,
                    "build_hash": test_ref.build_hash,
                    "path": test_ref.path,
                },
                "mapping_runs": runs,
                "v7_inputs": plan.summary_record(),
                "top_up": top_up,
                "decisions_note": (
                    "decision settings are recorded, not hashed; version-7 "
                    "decisions are re-derived from resolvability_cells.parquet "
                    "with ensemble_decide (RESOLVE: M3c follow-up, plan §12)"
                ),
            },
        )
        result.write(context.work_dir)
    output: dict[str, Any] = {"resolvability": result.bundle_output()}
    output["resolvability"]["test_set_bundle"] = {
        "reference_id": test_ref.reference_id,
        "build_hash": test_ref.build_hash,
    }
    output["resolvability"]["mapping_runs"] = runs
    output["resolvability"]["top_up"] = top_up
    if result.trust.state is not None:
        output["panel_trust"] = result.trust.state
    return output


def _whb_specs(engine: Any, config: AnnotationConfig) -> list[Any]:
    from merxen.annotation import resolvability as res

    vocab = engine.vocab()
    if vocab is None:
        raise ReferenceBuildError(f"{engine.path} has no vocab snapshot")
    return res.whb_level_specs(
        vocab,
        config.thresholds,
        region=config.anatomical_region or WHB_WHOLE_CORTEX_REGION,
    )


def _whb_cells_rules(config: AnnotationConfig) -> list[Any]:
    from merxen.annotation import resolvability as res

    return res.whb_cells_rules(config.thresholds)


def _wmb_specs(engine: Any, config: AnnotationConfig) -> list[Any]:
    from merxen.annotation import resolvability as res

    vocab = engine.vocab()
    if vocab is None:
        raise ReferenceBuildError(f"{engine.path} has no vocab snapshot")
    full = TaxonomyTreeView.from_precompute(engine.mapping_precompute)
    supertype_of: dict[str, str] = {}
    if WMB_SUPT in full.hierarchy:
        for supertype in full.nodes(WMB_SUPT):
            for cluster in full.children_of(WMB_SUPT, supertype):
                supertype_of[str(cluster)] = str(supertype)
    return res.wmb_level_specs(
        vocab, config.thresholds, supertype_of_cluster=supertype_of or None
    )


# --------------------------------------------------------------------------
# Resolvability version 7 (M3c; plan §8.3 v7.1): per-family selection and the
# version-7 inputs of a self-map. A version-6 family's payload never changes
# (``BundleBuilder.panel_params`` returns nothing for it).

# The reference a measured factor table is measured against, per self-map.
V7_MEMBER_TABLE_REFERENCE: Final[dict[str, str]] = {
    "wmb_panel": "wmb_10xv3",
    WMB_TESTSET_REFERENCE_ID: "wmb_10xv3",
}
# Test-set references and the primary references whose self-map they serve.
V7_TEST_SET_OF: Final[dict[str, str]] = {
    "whb_frontal_supc_clus": HO_REFERENCE_ID,
    "seaad_mr_panel": HO_REFERENCE_ID,
    "wmb_panel": WMB_TESTSET_REFERENCE_ID,
}
TOP_UP_SOURCE: Final = "class_top_up"
TOP_UP_SOURCE_OTHER_REGION: Final = "class_top_up_other_region"


@dataclass(frozen=True)
class ResolvabilityPlan:
    """What a panel's self-map runs (plan §8.3 v7.1-v7.6).

    Attributes:
        version: 6 or 7.
        family_id: The panel's family (after trust inheritance), if known.
        chemistry: ``sim_inputs.ChemistryResolution`` (version 7).
        members: The ensemble members (version 7).
        grid: The depth grid.
        assets: The simulation-input assets the self-map uses.
        profile_asset: The ``profile`` asset of the family's species x
            chemistry (class-depth predictions), if any.
        top_up: The test-set top-up rule (version 7; ``None`` when off).
    """

    version: int
    family_id: str | None
    chemistry: Any | None = None
    members: tuple[Any, ...] = ()
    grid: tuple[int, ...] = ()
    assets: tuple[Any, ...] = ()
    profile_asset: Any | None = None
    top_up: dict[str, Any] | None = None

    def simulation_payload(self) -> dict[str, Any]:
        """Return ``resolvability.v7_simulation_payload`` of the plan."""
        from merxen.annotation import resolvability as res

        return res.v7_simulation_payload(
            members=self.members,
            assets=self.assets,
            chemistry=self.chemistry.to_json()
            if self.chemistry is not None
            else "unknown",
            depth_grid=self.grid,
            top_up=self.top_up,
        )

    def summary_record(self) -> dict[str, Any]:
        """Return what the version-7 summary records of the plan."""
        from merxen.annotation import sim_inputs as si

        return {
            "family_id": self.family_id,
            "chemistry": None if self.chemistry is None else self.chemistry.to_json(),
            "assets": si.asset_hashes(self.assets),
            "profile_asset": None
            if self.profile_asset is None
            else self.profile_asset.asset_id,
            "top_up_rule": self.top_up,
            "depth_grid_v7": list(self.grid),
        }


def panel_resolvability_version(panel: Any, config: AnnotationConfig) -> int:
    """Return the resolvability version of a panel (plan §8.3 v7.1).

    ``auto``: 6 for the families of ``validated_panels.csv`` (by the panel's
    family after trust inheritance, or its listed hash) and the pins of
    ``resolvability_v6_pins.csv``; 7 for every other family.

    Args:
        panel: The declared panel (``AnnotationPanel``; any object with a
            ``panel_hash`` and optionally a ``panel_family``).
        config: The annotation config (``resolvability.version``).

    Returns:
        6 or 7.

    Raises:
        ReferenceBuildError: When version 7 is forced on a version-6 family
            (refused outside the diagnostic, which writes to an output
            directory, never to the store).
    """
    from merxen.annotation import resolvability as res
    from merxen.annotation.diagnostics import load_validated_panels

    family = getattr(getattr(panel, "panel_family", None), "family_id", None)
    automatic = res.resolvability_version_for(
        family,
        getattr(panel, "panel_hash", None),
        validated=load_validated_panels(config.panel.validated_panels_path),
    )
    requested = config.resolvability.version
    if requested == "auto":
        return automatic
    if int(requested) == res.RESOLVABILITY_VERSION_V7 and automatic == (
        res.RESOLVABILITY_VERSION_V6
    ):
        raise ReferenceBuildError(
            f"resolvability version 7 is refused for the version-6 family "
            f"{family or getattr(panel, 'panel_hash', '?')!s} (its decisions are "
            "pre-registered for gates H and M, plan §8.3 v7.1); compute it as a "
            "diagnostic with annotation-panel-simulate --resolvability-version 7"
        )
    return int(requested)


def _panel_platform(panel: Any) -> str | None:
    platforms = [str(item).upper() for item in getattr(panel, "platforms", []) or []]
    if not platforms:
        return None
    return platforms[0] if len(set(platforms)) == 1 else "MIXED"


def top_up_rule(species: str, config: AnnotationConfig) -> dict[str, Any] | None:
    """Return the test-set top-up rule a version-7 self-map hashes (v7.6).

    Args:
        species: ``human`` or ``mouse``.
        config: The annotation config.

    Returns:
        The rule, or ``None`` when ``topup_min_class_test_cells`` is 0.
    """
    from merxen.annotation import resolvability as res

    target = int(config.resolvability.topup_min_class_test_cells)
    if target <= 0:
        return None
    common = {
        "version": res.TOP_UP_VERSION,
        "target": target,
        "seed": TEST_SET_SEED,
        "stratified_by": "leaf type (water filling)",
        "draw": "lowest draw_key(seed, class_top_up, cell id) first",
    }
    if species == "mouse":
        return {
            **common,
            "class_level": WMB_CLAS,
            "leaf_level": WMB_SUBC,
            "pool": "10Xv3 cells neither marker-training nor test cells",
            "min_training_cells_per_cluster": HO_MIN_TRAINING_CELLS_PER_CLUSTER,
            "max_cluster_frac": float(
                config.resolvability.gate_p_topup_max_cluster_frac
            ),
        }
    return {
        **common,
        "class_level": "supercluster floor class (COP separate)",
        "leaf_level": WHB_SUPC,
        "pools": ["holdout_donor", "other_region"],
        "other_region": {
            "version": HO_OTHER_REGION_VERSION,
            "classes": "non-neuronal",
            "training_clusters_only": True,
        },
    }


def resolvability_plan(
    spec: AnnotationReferenceSpec, panel: Any, config: AnnotationConfig
) -> ResolvabilityPlan:
    """Return the self-map plan of a reference on a panel (plan §8.3 v7).

    Version 6: nothing beyond the version-6 grid. Version 7: the panel's
    chemistry (``sim_inputs.resolve_chemistry`` with ``panel_chemistry``),
    the members (R1 x ``ensemble_r1_seeds``, R3 where a measured table of
    the species x chemistry x reference exists, ``clean``, the human Prime
    lung stress member), the grid (13 values above 1,000 genes), the assets
    (member and stress tables, the class-depth profile) and the top-up rule.

    Args:
        spec: The reference spec (primary, secondary or test set).
        panel: The declared panel.
        config: The annotation config.

    Returns:
        The plan.
    """
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si

    version = panel_resolvability_version(panel, config)
    family = getattr(getattr(panel, "panel_family", None), "family_id", None)
    n_genes = int(getattr(panel, "n_genes", 0) or 0)
    if version == res.RESOLVABILITY_VERSION_V6:
        return ResolvabilityPlan(
            version=version,
            family_id=family,
            grid=tuple(spec.resolved_depth_grid(n_genes)),
        )
    species = str(spec.species)
    chemistry = si.resolve_chemistry(
        getattr(panel, "ensembl_ids", []) or [],
        species=species,
        platform=_panel_platform(panel),
        declared=str(config.panel.panel_chemistry),
    )
    reference = V7_MEMBER_TABLE_REFERENCE.get(
        spec.reference_id
    ) or V7_MEMBER_TABLE_REFERENCE.get(V7_TEST_SET_OF.get(spec.reference_id, ""))
    member_table = (
        None
        if reference is None
        else si.member_table_for(species, chemistry.chemistry, reference)
    )
    stress_table = (
        si.get_asset(si.STRESS_HUMAN_LUNG)
        if species == "human" and chemistry.chemistry == "xenium_prime"
        else None
    )
    members = res.ensemble_members(
        config.resolvability,
        species=species,
        chemistry=chemistry.chemistry,
        member_table=member_table,
        stress_table=stress_table,
        r1_seeds=tuple(config.resolvability.ensemble_r1_seeds),
        table_rule=config.resolvability.r3_table_rule,
    )
    if member_table is not None:
        members = [
            res.EnsembleMember(
                res.member_recipe(
                    member.recipe.name,
                    int(member.recipe.seed),
                    config.resolvability,
                    table=member_table,
                    table_rule=config.resolvability.r3_table_rule,
                    residual_sd_log2=config.resolvability.r3_residual_sd_log2,
                ),
                member.role,
            )
            if member.recipe.name == res.R3_RECIPE
            else member
            for member in members
        ]
    profiles = si.find_assets(
        role="profile", species=species, chemistry=chemistry.chemistry
    )
    profile_asset = (
        sorted(profiles, key=lambda item: item.asset_id)[0] if profiles else None
    )
    assets = [
        item for item in (member_table, stress_table, profile_asset) if item is not None
    ]
    return ResolvabilityPlan(
        version=version,
        family_id=family,
        chemistry=chemistry,
        members=tuple(members),
        grid=tuple(res.v7_depth_grid(species, n_genes, spec.depth_grid)),
        assets=tuple(assets),
        profile_asset=profile_asset,
        top_up=top_up_rule(species, config),
    )


def _v7_primary_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig, test_reference_id: str
) -> Any:
    """Return ``panel -> params`` of a primary or secondary version-7 self-map."""

    def params(panel: Any) -> dict[str, Any] | None:
        if not config.resolvability.enabled or spec.role not in (
            "primary",
            "secondary",
        ):
            return None
        from merxen.annotation import resolvability as res

        plan = resolvability_plan(spec, panel, config)
        if plan.version != res.RESOLVABILITY_VERSION_V7:
            return None
        base = _resolvability_params(spec, config, test_reference_id)
        base.pop("recipes", None)
        base["resolvability_version"] = plan.version
        base["test_set"] = {**base["test_set"], "top_up": plan.top_up}
        base["v7"] = plan.simulation_payload()
        return {"resolvability": base}

    return params


def _v7_test_set_params(spec: AnnotationReferenceSpec, config: AnnotationConfig) -> Any:
    """Return ``panel -> params`` of a version-7 test set (the top-up rule)."""

    def params(panel: Any) -> dict[str, Any] | None:
        if not config.resolvability.enabled:
            return None
        from merxen.annotation import resolvability as res

        plan = resolvability_plan(spec, panel, config)
        if plan.version != res.RESOLVABILITY_VERSION_V7:
            return None
        return {"resolvability_version": plan.version, "top_up": plan.top_up}

    return params


def _resolvability_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig, test_reference_id: str
) -> dict[str, Any]:
    """Return what the self-map output depends on (hashed; plan §3.2)."""
    from merxen.annotation import resolvability as res

    if not config.resolvability.enabled or spec.role not in ("primary", "secondary"):
        return {"enabled": False}
    test_params: dict[str, Any]
    if test_reference_id == HO_REFERENCE_ID:
        test_params = _ho_params(spec, config)
    else:
        test_params = _wmb_testset_params(spec, config)
    return {
        "enabled": True,
        "resolvability_version": res.RESOLVABILITY_VERSION,
        "recipes": [
            recipe.to_json()
            for recipe in res.simulation_recipes(
                config.resolvability, seed=TEST_SET_SEED
            )
        ],
        "test_set": {"reference_id": test_reference_id, **test_params},
        "self_map_mapping": {
            "bootstrap_factor": spec.bootstrap_factor,
            "bootstrap_iteration": spec.bootstrap_iteration,
            "rng_seed": spec.rng_seed,
        },
        "fine_seed_check": bool(config.thresholds.allow_fine_levels),
        "anatomical_region": config.anatomical_region,
    }


def _ho_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        **_panel_marker_params(config),
        "holdout_donor": config.resolvability.holdout_donor,
        "n_test_cells": config.resolvability.n_test_cells or HO_DEFAULT_TEST_CELLS,
        "max_test_cells_per_supercluster": HO_MAX_TEST_CELLS_PER_SUPERCLUSTER,
        "min_training_cells_per_cluster": HO_MIN_TRAINING_CELLS_PER_CLUSTER,
        "source_hierarchy": list(WHB_SOURCE_HIERARCHY),
        "roi_labels": list(WHB_FRONTAL_ROI_LABELS),
        "min_cells_per_leaf": WHB_FRONTAL_MIN_CELLS_PER_LEAF,
        "seed": TEST_SET_SEED,
        "vocab_assets": vocab_asset_sha256(WHB_TAXONOMY_ID),
        "truth_exclusion": {
            "version": HO_TRUTH_EXCLUSION_VERSION,
            "region": config.anatomical_region or WHB_WHOLE_CORTEX_REGION,
        },
        "other_region": {
            "version": HO_OTHER_REGION_VERSION,
            "roi_labels": list(HO_OTHER_REGION_ROI_LABELS),
            "feature_matrix": HO_OTHER_REGION_MATRIX,
            "seed": HO_OTHER_REGION_SEED,
        },
    }


def _wmb_testset_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        "library_method": WMB_LIBRARY_METHOD,
        "sampling_seed": WMB_SAMPLING_SEED,
        "max_cells_per_cluster": spec.max_cells_per_cluster,
        "n_test_cells": config.resolvability.n_test_cells,
        "nonneuronal_classes": list(WMB_NONNEURONAL_TEST_CLASSES),
        "extra_per_supertype": (
            config.resolvability.mouse_nonneuronal_extra_per_supertype
        ),
        "extra_seed": WMB_TESTSET_EXTRA_SEED,
        "seed": TEST_SET_SEED,
        "matrix_pattern": f"*{WMB_H5AD_SUFFIX}",
        "vocab_assets": vocab_asset_sha256(WMB_TAXONOMY_ID),
    }


# --------------------------------------------------------------------------
# Registration


def _panel_marker_params(config: AnnotationConfig) -> dict[str, Any]:
    """Config values every panel builder reads (weak parents, kept markers)."""
    return {
        "weak_parent_markers": config.panel.weak_parent_markers,
        "large_panel_genes": config.panel.large_panel_genes,
    }


def _whb_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        **_panel_marker_params(config),
        "source_hierarchy": list(WHB_SOURCE_HIERARCHY),
        "roi_labels": list(WHB_FRONTAL_ROI_LABELS),
        "min_cells_per_leaf": WHB_FRONTAL_MIN_CELLS_PER_LEAF,
        "negative_gene_max_fraction": config.flags.negative_gene_max_fraction,
        "negative_gene_references": ["whb_frontal", "seaad_mr"],
        **_state_and_vocab_params("human", (WHB_TAXONOMY_ID, SEAAD_TAXONOMY_ID)),
        "resolvability": _resolvability_params(spec, config, HO_REFERENCE_ID),
    }


def _seaad_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        **_panel_marker_params(config),
        "pinned_sha256": SEAAD_STATS_PIN.sha256,
        "pinned_md5": SEAAD_STATS_PIN.md5,
        "profile_method": "B_condLN",
        "vocab_assets": vocab_asset_sha256(SEAAD_TAXONOMY_ID),
        "resolvability": _resolvability_params(spec, config, HO_REFERENCE_ID),
    }


def _wmb_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    from merxen.annotation.vocab import asset_path

    return {
        **_panel_marker_params(config),
        "library_method": WMB_LIBRARY_METHOD,
        "sampling_seed": WMB_SAMPLING_SEED,
        "marker_hierarchy": list(spec.hierarchy or WMB_HIERARCHY),
        "validated_mapping_md5": WMB_MAPPING_STATS_MD5,
        "matrix_pattern": f"*{WMB_H5AD_SUFFIX}",
        "marker_universe": {
            WMB_MARKER_UNIVERSE_FILE: file_sha256(asset_path(WMB_MARKER_UNIVERSE_FILE))
        },
        "validated_test_cells_sha256": WMB_SELFMAP_TEST_CELLS_PIN.sha256,
        "negative_gene_max_fraction": config.flags.negative_gene_max_fraction,
        **_state_and_vocab_params("mouse", (WMB_TAXONOMY_ID,)),
        "resolvability": _resolvability_params(spec, config, WMB_TESTSET_REFERENCE_ID),
    }


def _region_share_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        "region_scheme": REGION_SCHEME,
        "grey_regions": list(GREY_REGIONS),
        "ob_anterior_ap_mm": OB_ANTERIOR_AP_MM,
        "ap_window_half_width": AP_WINDOW_HALF_WIDTH,
    }


def _whole_ctx_params(
    spec: AnnotationReferenceSpec, config: AnnotationConfig
) -> dict[str, Any]:
    return {
        **_panel_marker_params(config),
        "nodes_to_drop": list(spec.nodes_to_drop) or cortex_implausible_superclusters(),
        "max_genes": WHB_WHOLE_MAX_GENES,
        "profile_method": "B_condLN",
        "vocab_assets": vocab_asset_sha256(WHB_TAXONOMY_ID),
    }


def predicted_wmb_query_marker_peak_gb(n_candidate_genes: int) -> float:
    """Return the predicted WMB query-marker peak (GB) for some candidate genes.

    Args:
        n_candidate_genes: Genes marker discovery chooses from.

    Returns:
        The largest measured peak up to ``WMB_QUERY_MARKER_MEASURED_MAX_GENES``
        candidates, scaled with the candidates beyond (plan §8.7).
    """
    scale = max(1.0, n_candidate_genes / WMB_QUERY_MARKER_MEASURED_MAX_GENES)
    return WMB_QUERY_MARKER_PEAK_GB_MEASURED * scale


def prep_memory_reserve_gb() -> float:
    """Return the PREP task's memory reserve (``max_gb / PREP_MAX_GB_FRACTION``)."""
    return prep_resources().max_gb / PREP_MAX_GB_FRACTION


def _wmb_large_panel_refusal(
    panel: AnnotationPanel, config: AnnotationConfig
) -> str | None:
    """Refuse a large WMB panel without the prefilter above the memory reserve."""
    if (
        panel.n_genes <= config.panel.large_panel_genes
        or config.panel.large_panel_marker_prefilter != "none"
    ):
        return None
    predicted = predicted_wmb_query_marker_peak_gb(panel.n_genes)
    reserve = prep_memory_reserve_gb()
    needed = predicted * PREP_MEMORY_MARGIN
    if needed <= reserve:
        return None
    return (
        f"panel {panel.name} has {panel.n_genes} genes and "
        "large_panel_marker_prefilter is 'none': the predicted WMB query-marker "
        f"peak ({predicted:.0f} GB) + {PREP_MEMORY_MARGIN - 1:.0%} "
        f"({needed:.0f} GB) exceeds the PREP memory reserve "
        f"({reserve:.0f} GB = --max-gb {prep_resources().max_gb} / "
        f"{PREP_MAX_GB_FRACTION}); the prefilter is mandatory above the reserve "
        "(plan §8.7, OD-E8): in the pipeline raise annotation_prep_large_memory; "
        "the prefilter (large_panel_marker_prefilter = per_parent_topk_union) "
        "can be set only through --annotation-config of the standalone "
        "annotation-reference-prep / annotation-panel-simulate commands"
    )


def _whole_ctx_refusal(panel: AnnotationPanel, config: AnnotationConfig) -> str | None:
    """Refuse the whole-WHB bundle above 1,000 genes (plan §3.2, §8.7)."""
    if panel.n_genes <= WHB_WHOLE_MAX_GENES:
        return None
    return (
        f"whb_whole_ctx_panel is refused above {WHB_WHOLE_MAX_GENES} genes (panel "
        f"{panel.name} has {panel.n_genes}; about 75 h extrapolated, plan §8.7)"
    )


@dataclass(frozen=True)
class _BuilderRecipe:
    name: str
    build: Any
    taxonomy_id: str | None
    params: Any
    uses_panel: bool = True
    source_patterns: Mapping[str, str] = field(default_factory=dict)
    finds_markers: bool = True
    refuse: Any = None
    # (spec, config) -> (panel -> params | None): the version-7 inputs.
    panel_params: Any = None


def _ho_primary_v7(spec: AnnotationReferenceSpec, config: AnnotationConfig) -> Any:
    return _v7_primary_params(spec, config, HO_REFERENCE_ID)


def _wmb_primary_v7(spec: AnnotationReferenceSpec, config: AnnotationConfig) -> Any:
    return _v7_primary_params(spec, config, WMB_TESTSET_REFERENCE_ID)


BUILDER_RECIPES: Final[dict[str, _BuilderRecipe]] = {
    "whb_frontal_supc_clus": _BuilderRecipe(
        "whb_frontal_supc_clus",
        build_whb_frontal,
        WHB_TAXONOMY_ID,
        _whb_params,
        panel_params=_ho_primary_v7,
    ),
    "seaad_mr_panel": _BuilderRecipe(
        "seaad_mr_panel",
        build_seaad_mr,
        SEAAD_TAXONOMY_ID,
        _seaad_params,
        panel_params=_ho_primary_v7,
    ),
    "wmb_panel": _BuilderRecipe(
        "wmb_panel",
        build_wmb_panel,
        WMB_TAXONOMY_ID,
        _wmb_params,
        # Only the matrices enter build_hash: the shared ABC cache also holds
        # the legacy downloader's .lock and .tmp files (mapmycells.py).
        source_patterns={SOURCE_WMB_H5AD_DIR: f"*{WMB_H5AD_SUFFIX}"},
        refuse=_wmb_large_panel_refusal,
        panel_params=_wmb_primary_v7,
    ),
    "wmb_region_share": _BuilderRecipe(
        "wmb_region_share",
        build_wmb_region_share,
        WMB_TAXONOMY_ID,
        _region_share_params,
        uses_panel=False,
        finds_markers=False,
    ),
    "whb_whole_ctx_panel": _BuilderRecipe(
        "whb_whole_ctx_panel",
        build_whb_whole_ctx,
        WHB_TAXONOMY_ID,
        _whole_ctx_params,
        refuse=_whole_ctx_refusal,
    ),
    HO_REFERENCE_ID: _BuilderRecipe(
        HO_REFERENCE_ID,
        build_whb_frontal_ho,
        WHB_TAXONOMY_ID,
        _ho_params,
        panel_params=_v7_test_set_params,
    ),
    WMB_TESTSET_REFERENCE_ID: _BuilderRecipe(
        WMB_TESTSET_REFERENCE_ID,
        build_wmb_selfmap_testset,
        WMB_TAXONOMY_ID,
        _wmb_testset_params,
        source_patterns={SOURCE_WMB_H5AD_DIR: f"*{WMB_H5AD_SUFFIX}"},
        finds_markers=False,
        panel_params=_v7_test_set_params,
    ),
}


def builder_for(
    spec: AnnotationReferenceSpec, config: AnnotationConfig | None = None
) -> BundleBuilder:
    """Return the bundle builder of a reference spec.

    Args:
        spec: The reference spec.
        config: The annotation config; ``None`` uses the species defaults.

    Returns:
        The builder (name, build function, taxonomy, hashed parameters,
        ``prepare_spec``).

    Raises:
        ReferenceBuildError: If no builder exists for the reference.
    """
    recipe = BUILDER_RECIPES.get(spec.reference_id)
    if recipe is None:
        raise ReferenceBuildError(f"no builder for reference {spec.reference_id!r}")
    config = config or AnnotationConfig(species=spec.species)
    return BundleBuilder(
        name=recipe.name,
        build=recipe.build,
        taxonomy_id=recipe.taxonomy_id,
        params=recipe.params(spec, config),
        uses_panel=recipe.uses_panel,
        prepare_spec=prepare_reference_spec,
        source_patterns=dict(recipe.source_patterns),
        finds_markers=recipe.finds_markers,
        refuse=recipe.refuse,
        panel_params=None
        if recipe.panel_params is None
        else recipe.panel_params(spec, config),
    )


def build_reference_bundle(
    store: ReferenceStore,
    spec: AnnotationReferenceSpec,
    panel: AnnotationPanel | None,
    *,
    config: AnnotationConfig | None = None,
    options: SourceOptions | None = None,
) -> BundleRef:
    """Prepare a spec's sources and get or build its bundle.

    Args:
        store: The reference store.
        spec: The reference spec.
        panel: The declared panel (``None`` for panel-independent references).
        config: The annotation config.
        options: Download cache and seeds for ``prepare_reference_spec``.

    Returns:
        The ``BundleRef``.
    """
    builder = builder_for(spec, config)
    prepared = prepare_reference_spec(spec, options)
    return store.get_or_build(prepared, panel, builder=builder, config=config)


for _reference_id in BUILDER_RECIPES:
    if _reference_id not in KNOWN_REFERENCES:  # pragma: no cover - config drift
        raise RuntimeError(f"builder for unknown reference {_reference_id!r}")
    register_builder(_reference_id, builder_for)
