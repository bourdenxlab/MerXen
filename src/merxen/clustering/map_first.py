"""Map-first hierarchy: branches and leaves from the reference mapping (plan §6).

In ``map_first`` mode the hierarchy of a section comes from its label table
(``<sid>_celltype_labels.parquet``, written by RESOLVE), not from Leiden
resolutions (D-H1, OD-B1):

- **branch** = ``ct_branch`` (human: broad class with neurons split by NT,
  ``Neurons/unresolved`` and ``Oligodendrocyte lineage/unresolved`` for
  cells confident only at lineage, ``Mixed/Unknown``; mouse: WMB class);
- **leaf** = ``ct_leaf`` (the confident WHB supercluster / WMB subclass,
  else ``unresolved``). Human branches under ``min_branch_cells`` keep their
  leaves; mouse classes under it get ``unresolved`` leaves. Broad-only
  datasets (gate ``broad_only`` / ``failed``) and ``broad_only`` /
  ``refused`` panels have ``unresolved`` leaves only;
- ``hierarchical_cluster`` = ``"<branch>:<leaf>"``: categorical, never null,
  one value per (branch, leaf) pair.

``run_map_first_hierarchy`` checks the label table against the section
(``validate_label_table`` with the H5AD index: the H5AD holds exactly the
``in_table`` cells), copies the §4.5 label columns into ``obs``, fills the
legacy columns (``broad_class``, ``broad_atlas_label``,
``neuron_split_label``, ``subcluster_label``, ...), computes the
whole-section QC embedding and Leiden (``representation``) and the
subsample-stability diagnostic (``stability``), and records the provenance
in ``uns`` as scalars and JSON strings (§4.6), so the H5AD and the zarr
table written by FINALIZE keep it intact.

Imports: numpy, pandas, ``merxen.annotation.{schema,vocab,provenance}``
(pydantic) and ``merxen.clustering.*`` at module level. scanpy and anndata
are imported where they are used, and the plots reuse the legacy plot
helpers of ``merxen.analysis.clustering_squidpy`` lazily, so this module
imports in the GPU env and without ``cell_type_mapper`` or SpatialData
(plan §3.5, §11.2; test-enforced).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.provenance import (
    AnnotationProvenance,
    annotation_manifest_filename,
)
from merxen.annotation.schema import (
    LABEL_TABLE_VERSION,
    LABELS_METADATA_KEY,
    SUBCLUSTER_STATUSES,
    CellStatus,
    Columns,
    branch_categories,
    column_specs,
    label_table_filename,
    safe_token,
    validate_label_table,
)
from merxen.annotation.vocab import (
    MAP_FIRST_BROAD_LABELS,
    NEURONS,
    NT_CLASSES,
    SPECIES,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
    Species,
    broad_classes_for_map_first,
    load_vocab,
)
from merxen.clustering.cellset import restrict_to_table_cells, select_table_cells
from merxen.clustering.representation import (
    QC_BROAD_LEIDEN_KEY,
    QC_LEIDEN_KEY,
    EmbeddingParams,
    QcLeidenProvenance,
    branch_embedding,
    normalize_table_cells,
    whole_section_qc,
)
from merxen.clustering.stability import (
    STABILITY_FRACTION,
    STABILITY_N_SUBSAMPLES,
    SubsampleStability,
    adjusted_rand_index,
    subsample_ari,
)

if TYPE_CHECKING:
    import anndata as ad

    from merxen.config import ClusteringSquidpyConfig

logger = logging.getLogger(__name__)

MAP_FIRST_MODE: Final = "map_first"
# ``obs`` keys shared with legacy clustering (clustering_squidpy.*_KEY), so
# MENDER, cortical depth, the viewer and GASTON read map_first tables as
# they read legacy ones.
BROAD_CLUSTER_KEY: Final = QC_BROAD_LEIDEN_KEY
BROAD_ATLAS_LABEL_KEY: Final = "broad_atlas_label"
BROAD_CLASS_KEY: Final = "broad_class"
BROAD_SCORE_KEY: Final = "broad_annotation_score"
BROAD_SCORE_MARGIN_KEY: Final = "broad_annotation_score_margin"
BROAD_N_MARKERS_KEY: Final = "broad_annotation_n_markers"
NEURON_SPLIT_KEY: Final = "neuron_split_label"
SUBCLUSTER_LABEL_KEY: Final = "subcluster_label"
SUBCLUSTER_STATUS_KEY: Final = "subcluster_status"
HIERARCHICAL_CLUSTER_KEY: Final = "hierarchical_cluster"
HIERARCHICAL_UNS_KEY: Final = "merxen_hierarchical_clustering"
LEIDEN_PROVENANCE_UNS_KEY: Final = "merxen_leiden_provenance"
CLUSTERING_PARAMS_UNS_KEY: Final = "merxen_clustering_params"
HIERARCHY_SEPARATOR: Final = ":"
BRANCH_LEVEL: Final = "ct_branch"
LEAF_SOURCE_MAPPED: Final = "mapped"
# The leaf level per species (§6.1: WHB supercluster / WMB subclass).
LEAF_LEVEL: Final[dict[str, str]] = {"human": "supercluster", "mouse": "subclass"}
# The coarse level whose confident name is ``broad_atlas_label`` (§4.5).
ATLAS_LABEL_LEVEL: Final[dict[str, str]] = {"human": "supercluster", "mouse": "class"}
MAPPED_LEAF_STATUS: Final[dict[str, str]] = {
    "human": "mapped_supercluster",
    "mouse": "mapped_subclass",
}
STATUS_UNRESOLVED: Final = "unresolved"
STATUS_NOT_RESOLVABLE_PANEL: Final = "not_resolvable_panel"
STATUS_DATASET_BROAD_ONLY: Final = "dataset_broad_only"
STATUS_UNASSIGNED: Final = "unassigned"
NOT_NEURON: Final = "not_neuron"
NEURON_SPLIT_LABELS: Final[tuple[str, ...]] = (
    *NT_CLASSES,
    UNRESOLVED_LABEL,
    NOT_NEURON,
)
# A gate level or panel trust state under which no leaf may be emitted.
LEAF_SUPPRESSING_GATE_LEVELS: Final[frozenset[str]] = frozenset(
    {"broad_only", "failed"}
)
LEAF_SUPPRESSING_PANEL_TRUST: Final[frozenset[str]] = frozenset(
    {"broad_only", "refused"}
)
UNASSIGNED_BRANCH_SUFFIX: Final = f"/{UNRESOLVED_LABEL}"
# Label-table columns copied into ``obs`` besides ``ct_*`` and ``flag_*``
# (§4.5: flags and their continuous companions, exclude_hard,
# discovery_caution, depth_bin). ``soft_*``, ``mmc_*`` and ``ll_*`` stay in
# the parquet.
OBS_COMPANION_COLUMNS: Final[tuple[str, ...]] = (
    Columns.CONTAMINATION_SCORE,
    Columns.NEG_COUNTS,
    Columns.EXPECTED_GENES_Q95,
    Columns.OOD_Z,
    Columns.REGION_COHERENCE,
    Columns.MICROGLIA_STAT,
    Columns.MICROGLIA_WEIGHT,
    Columns.EXCLUDE_HARD,
    Columns.DISCOVERY_CAUTION,
    Columns.DEPTH_BIN,
)
PARQUET_ONLY_PREFIXES: Final[tuple[str, ...]] = (
    Columns.SOFT_BROAD_PREFIX,
    Columns.SOFT_CLASS_PREFIX,
    Columns.MMC_PREFIX,
    Columns.LL_PREFIX,
)
# Missing numbers in ``uns`` scalars (h5ad cannot store ``None``).
MISSING_INT: Final = -1
MISSING_TEXT: Final = "unknown"
# Panels above this many genes get dotplots of the most variable genes only.
DOTPLOT_MAX_GENES: Final = 300


# --------------------------------------------------------------------------
# Unassigned states (MENDER ``unassigned_state_policy``, plan §4.9)


def branch_of_state(state: str) -> str:
    """Return the branch part of a hierarchy state.

    Args:
        state: A ``hierarchical_cluster`` value (``"<branch>:<leaf>"``) or a
            plain branch-level state (``ct_mender_state``, ``ct_branch``,
            ``broad_class``).

    Returns:
        The text before the first ``":"``, or the whole state.
    """
    return str(state).split(HIERARCHY_SEPARATOR, 1)[0]


def is_unassigned_state(state: str) -> bool:
    """Return whether a MENDER cell state carries no assigned branch.

    Unassigned cells are ``Mixed/Unknown`` and the ``*/unresolved`` branches
    (``Neurons/unresolved``, ``Oligodendrocyte lineage/unresolved``): 28-61%
    of table cells on the current datasets, which as a state would make
    niches track depth and quality (§4.9). The rule reads the branch part,
    so it applies to ``hierarchical_cluster`` (whatever the leaf) and to the
    branch-level states alike. A confident branch with an ``unresolved``
    leaf (``Astrocytes:unresolved``) is assigned: its branch is known.

    Args:
        state: A cell state.

    Returns:
        ``True`` for ``Mixed/Unknown`` and ``*/unresolved`` branches.
    """
    branch = branch_of_state(state).strip()
    return branch == UNASSIGNED_LABEL or branch.endswith(UNASSIGNED_BRANCH_SUFFIX)


def unassigned_states(states: Iterable[str]) -> tuple[str, ...]:
    """Return the unassigned states among ``states``, sorted and unique.

    Args:
        states: Cell-state values or categories.

    Returns:
        The states for which ``is_unassigned_state`` holds.
    """
    return tuple(sorted({str(state) for state in states if is_unassigned_state(state)}))


# --------------------------------------------------------------------------
# Inputs


@dataclass(frozen=True)
class LabelInputs:
    """A section's label table and its provenance.

    Attributes:
        labels: The label table (§4.1), one row per segmented object.
        provenance: Its ``AnnotationProvenance``, or ``None`` if absent.
        labels_path: Where the table was read from.
        manifest_path: The annotation manifest read, if any.
    """

    labels: pd.DataFrame
    provenance: AnnotationProvenance | None
    labels_path: Path
    manifest_path: Path | None = None


def find_label_table(
    labels_dir: Path | str, sample_id: str, platform: str | None = None
) -> Path:
    """Return the label table of a sample under a RESOLVE output directory.

    RESOLVE writes ``<labels_dir>/<platform>/<sid>_celltype_labels.parquet``;
    a flat ``<labels_dir>/<sid>_celltype_labels.parquet`` is accepted too.

    Args:
        labels_dir: ``annotation_resolve_out`` (or a platform folder in it).
        sample_id: Sample id.
        platform: ``MERSCOPE`` / ``XENIUM`` (lower-cased for the folder).

    Returns:
        The existing parquet path.

    Raises:
        FileNotFoundError: If neither location holds the table.
    """
    root = Path(labels_dir)
    name = label_table_filename(sample_id)
    candidates = []
    if platform:
        candidates.append(root / str(platform).lower() / name)
    candidates.append(root / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no label table {name} under {root} "
        f"(looked in {[str(path) for path in candidates]})"
    )


def load_label_inputs(
    labels_dir: Path | str, sample_id: str, platform: str | None = None
) -> LabelInputs:
    """Read a sample's label table and provenance written by RESOLVE.

    The provenance comes from ``<sid>_annotation_manifest.json`` next to the
    table, else from the parquet schema metadata RESOLVE writes too.

    Args:
        labels_dir: ``annotation_resolve_out`` (or a platform folder in it).
        sample_id: Sample id.
        platform: ``MERSCOPE`` / ``XENIUM``.

    Returns:
        The label table and its provenance.
    """
    import pyarrow.parquet as pq

    path = find_label_table(labels_dir, sample_id, platform)
    table = pq.read_table(path)
    provenance: AnnotationProvenance | None = None
    manifest_path: Path | None = path.with_name(annotation_manifest_filename(sample_id))
    if manifest_path is not None and manifest_path.is_file():
        provenance = AnnotationProvenance.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest_path = None
        raw = (table.schema.metadata or {}).get(LABELS_METADATA_KEY)
        if raw is not None:
            provenance = AnnotationProvenance.from_uns_json(raw.decode("utf-8"))
    return LabelInputs(table.to_pandas(), provenance, path, manifest_path)


@dataclass(frozen=True)
class LeafGate:
    """Dataset-level reasons that leave a section without leaves.

    Attributes:
        gate_level: Dataset gate level (``full`` / ``broad_only`` /
            ``failed``), ``None`` when unknown.
        gate_warning: Gate warning flag.
        panel_trust: Panel trust state, ``None`` when unknown.
        reasons: Gate reasons.
    """

    gate_level: str | None = None
    gate_warning: bool = False
    panel_trust: str | None = None
    reasons: tuple[str, ...] = ()

    @property
    def suppresses_leaves(self) -> bool:
        """Return whether the gate or the panel trust forbids leaves."""
        return (self.gate_level in LEAF_SUPPRESSING_GATE_LEVELS) or (
            self.panel_trust in LEAF_SUPPRESSING_PANEL_TRUST
        )

    @classmethod
    def from_provenance(
        cls: type[LeafGate], provenance: AnnotationProvenance | None
    ) -> LeafGate:
        """Read the gate and the panel trust from annotation provenance.

        Args:
            provenance: The section's provenance, or ``None``.

        Returns:
            The gate (human ``gate``, mouse ``mouse_gate``; the stricter
            level when both exist) and the panel trust state.
        """
        if provenance is None:
            return cls()
        gates = [
            gate
            for gate in (provenance.gate, provenance.mouse_gate)
            if gate is not None and gate.level is not None
        ]
        order = {"full": 0, "broad_only": 1, "failed": 2}
        level = None
        warning = False
        reasons: list[str] = []
        for gate in gates:
            if level is None or order[str(gate.level)] > order[level]:
                level = str(gate.level)
            warning = warning or bool(gate.warning)
            reasons += [str(reason) for reason in gate.reasons]
        trust = provenance.panel.panel_trust if provenance.panel is not None else None
        return cls(
            gate_level=level,
            gate_warning=warning,
            panel_trust=None if trust is None else str(trust),
            reasons=tuple(reasons),
        )


def primary_root_markers(provenance: AnnotationProvenance | None) -> int | None:
    """Return the root markers of the primary reference's lookup.

    Args:
        provenance: The section's provenance, or ``None``.

    Returns:
        ``markers.root_markers`` of the ``primary`` reference, or ``None``.
    """
    if provenance is None:
        return None
    for reference in provenance.references.values():
        if reference.role == "primary" and reference.markers is not None:
            return reference.markers.root_markers
    return None


# --------------------------------------------------------------------------
# Hierarchy columns (pure pandas)


def _species_of(labels: pd.DataFrame) -> Species:
    values = pd.unique(labels[Columns.SPECIES].astype(str))
    if len(values) != 1 or values[0] not in SPECIES:
        raise ValueError(
            f"the label table must hold one species of {SPECIES}, got {list(values)}"
        )
    species: Species = values[0]
    return species


def _unique_tokens(names: Sequence[str]) -> dict[str, str]:
    """Map names to distinct safe tokens (``_2``, ``_3``, ... on collisions)."""
    tokens: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        base = safe_token(name)
        token = base
        suffix = 2
        while token in used:
            token = f"{base}_{suffix}"
            suffix += 1
        used.add(token)
        tokens[name] = token
    return tokens


def _leaf_order(species: Species, leaves: Iterable[str]) -> list[str]:
    present = {str(leaf) for leaf in leaves}
    if species == "human":
        vocab_order = list(load_vocab("whb_supercluster").names)
    else:
        vocab_order = []
    ordered = [leaf for leaf in vocab_order if leaf in present]
    ordered += sorted(present - set(ordered) - {UNRESOLVED_LABEL})
    if UNRESOLVED_LABEL in present:
        ordered.append(UNRESOLVED_LABEL)
    return ordered


def _branch_order(species: Species, branches: Iterable[str]) -> list[str]:
    present = {str(branch) for branch in branches}
    ordered = [branch for branch in branch_categories(species) if branch in present]
    return ordered + sorted(present - set(ordered))


def _branch_broad(branch: str) -> str:
    """Return the broad label a human branch implies (``Neurons/x`` -> Neurons)."""
    if branch.endswith(UNASSIGNED_BRANCH_SUFFIX):
        return branch[: -len(UNASSIGNED_BRANCH_SUFFIX)]
    if branch.startswith(f"{NEURONS}/"):
        return NEURONS
    return branch


@dataclass(frozen=True)
class MapFirstHierarchy:
    """The map_first hierarchy of one section's table cells.

    Attributes:
        obs: Hierarchy and legacy columns, indexed like the labels.
        species: Species of the labels.
        leaf_level: The mapped leaf level.
        branch_manifest: Per branch (safe-token keys): cell and leaf counts.
        gate: The leaf gate applied.
        n_leaves_suppressed_gate: Confident leaves set to ``unresolved``
            because the dataset gate or the panel trust forbids leaves.
        n_leaves_unassigned_branch: Confident leaves set to ``unresolved``
            on ``Mixed/Unknown`` cells (a contract inconsistency; 0 expected).
        n_leaves_small_class: Mouse leaves set to ``unresolved`` because
            their class has fewer than ``min_branch_cells`` table cells.
        n_branch_broad_mismatch: Human cells whose ``broad_class`` differs
            from the broad label their branch implies (0 expected).
    """

    obs: pd.DataFrame
    species: Species
    leaf_level: str
    branch_manifest: dict[str, dict[str, Any]]
    gate: LeafGate
    n_leaves_suppressed_gate: int = 0
    n_leaves_unassigned_branch: int = 0
    n_leaves_small_class: int = 0
    n_branch_broad_mismatch: int = 0

    def summary(self) -> dict[str, Any]:
        """Return counts for the manifest and ``uns`` (JSON-ready)."""
        leaves = self.obs[SUBCLUSTER_LABEL_KEY].astype(str)
        return {
            "species": self.species,
            "leaf_level": self.leaf_level,
            "n_cells": int(len(self.obs)),
            "n_branches": int(len(self.branch_manifest)),
            "n_leaves": int(leaves[leaves != UNRESOLVED_LABEL].nunique()),
            "n_hierarchical_clusters": int(
                self.obs[HIERARCHICAL_CLUSTER_KEY].nunique()
            ),
            "n_unresolved_leaf_cells": int((leaves == UNRESOLVED_LABEL).sum()),
            "n_leaves_suppressed_gate": int(self.n_leaves_suppressed_gate),
            "n_leaves_unassigned_branch": int(self.n_leaves_unassigned_branch),
            "n_leaves_small_class": int(self.n_leaves_small_class),
            "n_branch_broad_mismatch": int(self.n_branch_broad_mismatch),
            "subcluster_status_counts": {
                str(key): int(value)
                for key, value in self.obs[SUBCLUSTER_STATUS_KEY]
                .value_counts(sort=False)
                .items()
                if int(value) > 0
            },
        }


def _neuron_split(
    broad_class: np.ndarray, nt_status: np.ndarray, nt_name: np.ndarray
) -> np.ndarray:
    labels = np.full(broad_class.shape, NOT_NEURON, dtype=object)
    neurons = broad_class == NEURONS
    nt_confident = nt_status == CellStatus.CONFIDENT.value
    labels[neurons] = UNRESOLVED_LABEL
    for nt in NT_CLASSES:
        labels[neurons & nt_confident & (nt_name == nt)] = nt
    return labels


def build_hierarchy_columns(
    labels: pd.DataFrame,
    *,
    gate: LeafGate | None = None,
    min_branch_cells: int = 50,
    root_markers: int | None = None,
) -> MapFirstHierarchy:
    """Build the map_first hierarchy and legacy columns of table cells.

    Args:
        labels: Label-table rows of the table cells (``in_table``), in the
            order of the section's ``obs``; the index is kept.
        gate: Dataset gate and panel trust; ``broad_only`` / ``failed``
            gates and ``broad_only`` / ``refused`` panels force every leaf
            to ``unresolved``.
        min_branch_cells: Branch size below which mouse classes lose their
            leaves (human branches keep them; §6.2).
        root_markers: Root markers of the primary lookup
            (``broad_annotation_n_markers``; -1 when unknown).

    Returns:
        The hierarchy columns and branch manifest.

    Raises:
        ValueError: On several species, missing branches or a branch that
            contains the ``":"`` separator.
    """
    species = _species_of(labels)
    leaf_gate = gate or LeafGate()
    leaf_level = LEAF_LEVEL[species]
    n_cells = len(labels)
    branch = labels[Columns.CT_BRANCH].astype(object)
    if bool(branch.isna().any()):
        raise ValueError(f"{int(branch.isna().sum())} table cells have no ct_branch")
    branch_values = branch.astype(str).to_numpy(dtype=object)
    with_separator = sorted(
        {value for value in branch_values if HIERARCHY_SEPARATOR in value}
    )
    if with_separator:
        raise ValueError(
            f"branches must not contain {HIERARCHY_SEPARATOR!r}: {with_separator}"
        )
    leaf_raw = labels[Columns.CT_LEAF].astype(object)
    leaf = np.where(
        leaf_raw.isna().to_numpy(), UNRESOLVED_LABEL, leaf_raw.astype(str).to_numpy()
    ).astype(object)
    leaf_status = (
        labels[Columns.level(leaf_level, "status")].astype(str).to_numpy(dtype=object)
    )
    resolved = leaf != UNRESOLVED_LABEL

    # Dataset-level suppression: the gate or the panel trust, or RESOLVE's
    # per-cell record of it (not_attempted_gate at the leaf level).
    gate_rows = leaf_status == CellStatus.NOT_ATTEMPTED_GATE.value
    if leaf_gate.suppresses_leaves:
        gate_rows = np.ones(n_cells, dtype=bool)
    n_suppressed = int((gate_rows & resolved).sum())
    if n_suppressed:
        logger.warning(
            "%d confident leaves set to %r: gate %s, panel trust %s",
            n_suppressed,
            UNRESOLVED_LABEL,
            leaf_gate.gate_level,
            leaf_gate.panel_trust,
        )
    leaf[gate_rows] = UNRESOLVED_LABEL

    unassigned_branch = branch_values == UNASSIGNED_LABEL
    n_unassigned_leaves = int((unassigned_branch & (leaf != UNRESOLVED_LABEL)).sum())
    if n_unassigned_leaves:
        logger.warning(
            "%d %s cells carry a confident leaf; set to %r",
            n_unassigned_leaves,
            UNASSIGNED_LABEL,
            UNRESOLVED_LABEL,
        )
    leaf[unassigned_branch] = UNRESOLVED_LABEL

    branch_sizes = pd.Series(branch_values).value_counts()
    small_branches = {
        str(name) for name, size in branch_sizes.items() if int(size) < min_branch_cells
    }
    small_rows = np.isin(branch_values, list(small_branches))
    n_small = 0
    if species == "mouse":
        small_resolved = small_rows & (leaf != UNRESOLVED_LABEL)
        n_small = int(small_resolved.sum())
        leaf[small_rows] = UNRESOLVED_LABEL

    # subcluster_status: mapped, else why not (first matching reason).
    status = np.full(n_cells, STATUS_UNRESOLVED, dtype=object)
    mapped = leaf != UNRESOLVED_LABEL
    status[mapped] = MAPPED_LEAF_STATUS[species]
    not_resolvable = leaf_status == CellStatus.NOT_RESOLVABLE.value
    status[~mapped & not_resolvable] = STATUS_NOT_RESOLVABLE_PANEL
    status[~mapped & gate_rows] = STATUS_DATASET_BROAD_ONLY
    status[~mapped & unassigned_branch] = STATUS_UNASSIGNED

    final_level = labels[Columns.CT_FINAL_LEVEL].astype(str)
    broad_name = labels[Columns.level("broad", "name")]
    lineage_name = (
        labels[Columns.level("lineage", "name")] if species == "human" else None
    )
    broad_class = broad_classes_for_map_first(
        final_level, broad_name, lineage_name, species=species
    )
    broad_values = broad_class.astype(str).to_numpy(dtype=object)
    n_mismatch = 0
    if species == "human":
        implied = np.array([_branch_broad(value) for value in branch_values])
        n_mismatch = int((implied != broad_values).sum())
        if n_mismatch:
            logger.warning(
                "%d cells: broad_class differs from the broad label of ct_branch",
                n_mismatch,
            )

    atlas_level = ATLAS_LABEL_LEVEL[species]
    atlas_confident = (
        labels[Columns.level(atlas_level, "status")].astype(str).to_numpy()
        == CellStatus.CONFIDENT.value
    )
    atlas_name = labels[Columns.level(atlas_level, "name")].astype(object).to_numpy()
    atlas_label = np.where(atlas_confident, atlas_name, broad_values).astype(object)
    atlas_label = np.where(pd.isna(atlas_label), broad_values, atlas_label)

    neuron_split = _neuron_split(
        broad_values,
        labels[Columns.level("nt", "status")].astype(str).to_numpy(dtype=object),
        labels[Columns.level("nt", "name")].astype(object).to_numpy(),
    )

    hierarchical = np.array(
        [
            f"{b}{HIERARCHY_SEPARATOR}{c}"
            for b, c in zip(branch_values, leaf, strict=True)
        ],
        dtype=object,
    )
    branch_order = _branch_order(species, branch_values)
    leaf_order = _leaf_order(species, leaf)
    branch_rank = {name: rank for rank, name in enumerate(branch_order)}
    leaf_rank = {name: rank for rank, name in enumerate(leaf_order)}
    pairs = sorted(
        {(str(b), str(c)) for b, c in zip(branch_values, leaf, strict=True)},
        key=lambda pair: (branch_rank[pair[0]], leaf_rank[pair[1]]),
    )
    hierarchical_categories = [f"{b}{HIERARCHY_SEPARATOR}{c}" for b, c in pairs]

    broad_score = labels[Columns.level("broad", "conf")].to_numpy(dtype=np.float32)
    broad_margin = labels[Columns.level("broad", "margin")].to_numpy(dtype=np.float32)
    n_markers = MISSING_INT if root_markers is None else int(root_markers)
    obs = pd.DataFrame(
        {
            BROAD_CLASS_KEY: pd.Categorical(
                broad_values, categories=list(MAP_FIRST_BROAD_LABELS[species])
            ),
            BROAD_ATLAS_LABEL_KEY: pd.Categorical(atlas_label.astype(str)),
            BROAD_SCORE_KEY: broad_score,
            BROAD_SCORE_MARGIN_KEY: broad_margin,
            BROAD_N_MARKERS_KEY: np.full(n_cells, n_markers, dtype=np.int32),
            NEURON_SPLIT_KEY: pd.Categorical(
                neuron_split, categories=list(NEURON_SPLIT_LABELS)
            ),
            SUBCLUSTER_LABEL_KEY: pd.Categorical(leaf, categories=leaf_order),
            SUBCLUSTER_STATUS_KEY: pd.Categorical(
                status, categories=list(SUBCLUSTER_STATUSES)
            ),
            HIERARCHICAL_CLUSTER_KEY: pd.Categorical(
                hierarchical, categories=hierarchical_categories
            ),
        },
        index=labels.index,
    )

    tokens = _unique_tokens(branch_order)
    manifest: dict[str, dict[str, Any]] = {}
    for name in branch_order:
        rows = branch_values == name
        branch_leaves = pd.Series(leaf[rows]).value_counts()
        ordered = [value for value in leaf_order if value in branch_leaves.index]
        leaf_tokens = _unique_tokens(ordered)
        manifest[tokens[name]] = {
            "branch": name,
            "n_cells": int(rows.sum()),
            "small_branch": name in small_branches,
            "leaves_kept": not (species == "mouse" and name in small_branches),
            "n_leaves": int(sum(value != UNRESOLVED_LABEL for value in ordered)),
            "n_unresolved": int(branch_leaves.get(UNRESOLVED_LABEL, 0)),
            "leaves": {
                leaf_tokens[value]: {
                    "leaf": value,
                    "n_cells": int(branch_leaves[value]),
                }
                for value in ordered
            },
        }
    return MapFirstHierarchy(
        obs=obs,
        species=species,
        leaf_level=leaf_level,
        branch_manifest=manifest,
        gate=leaf_gate,
        n_leaves_suppressed_gate=n_suppressed,
        n_leaves_unassigned_branch=n_unassigned_leaves,
        n_leaves_small_class=n_small,
        n_branch_broad_mismatch=n_mismatch,
    )


def label_obs_columns(columns: Iterable[str]) -> list[str]:
    """Return the label-table columns that go into the clustered ``obs``.

    §4.5: all ``ct_*`` (never ``soft_*``), the ``flag_*`` columns and their
    continuous companions, ``exclude_hard``, ``discovery_caution`` and
    ``depth_bin``. ``mmc_*`` and ``ll_*`` stay in the parquet.

    Args:
        columns: Label-table columns, in table order.

    Returns:
        The selected columns, in table order.
    """
    selected = []
    for column in columns:
        name = str(column)
        if name.startswith(PARQUET_ONLY_PREFIXES):
            continue
        if (
            name.startswith("ct_")
            or name.startswith("flag_")
            or name in OBS_COMPANION_COLUMNS
        ):
            selected.append(name)
    return selected


def label_obs_frame(labels: pd.DataFrame, species: Species) -> pd.DataFrame:
    """Return the §4.5 label columns of table cells, typed for h5ad and zarr.

    Categorical columns stay categorical (a column read back from parquet as
    all-null objects becomes an empty categorical), so ``obs`` holds no
    object columns with missing values, which h5ad cannot write.

    Args:
        labels: Label-table rows of the table cells, aligned with ``obs``.
        species: Species of the labels.

    Returns:
        The columns, with ``labels``' index.
    """
    # The fine-level specs cover the report-only columns when present.
    specs = column_specs(species, include_fine_levels=True)
    frame = pd.DataFrame(index=labels.index)
    for column in label_obs_columns(labels.columns):
        values = labels[column]
        spec = specs.get(column)
        kind: str | None = spec.kind if spec is not None else None
        if kind == "category" or (kind is None and values.dtype == object):
            frame[column] = _as_categorical(values)
        elif isinstance(values.dtype, pd.api.extensions.ExtensionDtype):
            # Nullable Int32 / boolean (depth_bin, neg_counts, the nullable
            # flags) keep their missing values; h5ad and zarr store them as
            # nullable arrays.
            frame[column] = values.array
        else:
            frame[column] = values.to_numpy()
    return frame


def _as_categorical(values: pd.Series) -> pd.Categorical:
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = [str(value) for value in values.cat.categories]
        text = values.astype(object)
        return pd.Categorical(
            [None if pd.isna(value) else str(value) for value in text],
            categories=categories,
        )
    text = [None if pd.isna(value) else str(value) for value in values.astype(object)]
    present = sorted({value for value in text if value is not None})
    return pd.Categorical(text, categories=present)


# --------------------------------------------------------------------------
# The hierarchy of one section


@dataclass(frozen=True)
class MapFirstResult:
    """What ``run_map_first_hierarchy`` produced besides the AnnData.

    Attributes:
        hierarchy: The hierarchy columns and manifest.
        qc: Provenance of the QC Leiden.
        stability: The stability diagnostic, or ``None`` if skipped.
        qc_leiden_vs_leaf_ari: ARI of the QC Leiden with the mapped leaves
            on cells with a resolved leaf (NaN if fewer than 2 such cells).
        wall_time_s: Seconds for the whole section.
        artifacts: Written files by name.
    """

    hierarchy: MapFirstHierarchy
    qc: QcLeidenProvenance
    stability: SubsampleStability | None
    qc_leiden_vs_leaf_ari: float
    wall_time_s: float
    artifacts: dict[str, Path] = field(default_factory=dict)


def _check_config(config: ClusteringSquidpyConfig) -> None:
    if config.mode != MAP_FIRST_MODE:
        raise ValueError(
            f"run_map_first_hierarchy needs mode {MAP_FIRST_MODE!r}, got "
            f"{config.mode!r}; legacy clustering keeps its own path"
        )
    if config.leaf_source != LEAF_SOURCE_MAPPED:
        raise NotImplementedError(
            f"leaf_source {config.leaf_source!r} arrives with v1.1 (M11, plan "
            "§6.4); v1 leaves are mapped reference nodes"
        )


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _clean_scalars(values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values (h5ad ``uns`` cannot store them)."""
    return {key: value for key, value in values.items() if value is not None}


def leaf_agreement_ari(obs: pd.DataFrame) -> float:
    """Return the ARI of the QC Leiden with the mapped leaves.

    Args:
        obs: ``obs`` with ``leiden`` and ``hierarchical_cluster``.

    Returns:
        The ARI on cells with a resolved leaf, NaN with fewer than 2.
    """
    leaves = obs[SUBCLUSTER_LABEL_KEY].astype(str).to_numpy()
    rows = leaves != UNRESOLVED_LABEL
    if int(rows.sum()) < 2:
        return float("nan")
    return adjusted_rand_index(
        obs[QC_LEIDEN_KEY].astype(str).to_numpy()[rows],
        obs[HIERARCHICAL_CLUSTER_KEY].astype(str).to_numpy()[rows],
    )


def run_map_first_hierarchy(
    adata: ad.AnnData,
    labels: pd.DataFrame,
    config: ClusteringSquidpyConfig,
    output_dir: Path | str,
    sample_id: str,
    *,
    provenance: AnnotationProvenance | None = None,
    plots: bool = True,
    stability: bool = True,
    stability_subsamples: int = STABILITY_N_SUBSAMPLES,
) -> tuple[ad.AnnData, MapFirstResult]:
    """Build the map_first hierarchy and QC embedding of one section (§6.2).

    1. Select the table cells (``select_table_cells``, as legacy) and check
       the label table against them: ``validate_label_table`` with the H5AD
       index, so the H5AD ids equal the parquet ``in_table`` ids.
    2. Branch = ``ct_branch``, leaf = ``ct_leaf``, with the §6.2 rules
       (``build_hierarchy_columns``).
    3. Copy the §4.5 label columns and fill the legacy columns.
    4. Whole-section QC embedding and Leiden (``whole_section_qc``) and the
       stability diagnostic (``subsample_ari``), report-only.
    5. ``uns`` provenance (scalars and JSON strings, safe tokens), the
       manifest and, with ``plots``, UMAP / spatial plots by
       ``broad_class``, ``ct_leaf`` and ``ct_final_level`` and per-branch
       dotplots and UMAPs.

    Args:
        adata: The section's prepared objects with control features already
            removed (as for legacy ``run_scanpy_clustering``); raw counts in
            ``X``. Not modified.
        labels: Its label table (all segmented objects; §4.1).
        config: The run's ``ClusteringSquidpyConfig`` (``mode="map_first"``).
        output_dir: Directory for the manifest, tables and plots.
        sample_id: Sample id.
        provenance: The label table's ``AnnotationProvenance`` (gate, panel
            trust, root markers); stored in ``uns["merxen_annotation_json"]``.
        plots: Write the QC plots.
        stability: Run the subsample-stability diagnostic.
        stability_subsamples: Number of stability subsamples.

    Returns:
        ``(clustered, result)``: the table cells with the hierarchy, legacy
        columns, QC embedding and provenance, and the run record.

    Raises:
        ValueError: If the config is not map_first or the label table does
            not match the section (``LabelTableError``).
        NotImplementedError: For ``leaf_source="denovo"`` (v1.1).
    """
    started = time.perf_counter()
    _check_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    species = _species_of(labels)

    clustered = adata.copy()
    selection = select_table_cells(clustered, config.min_counts)
    restrict_to_table_cells(clustered, selection)
    table_ids = pd.Index(clustered.obs_names.astype(str))
    validate_label_table(labels, species, h5ad_index=table_ids)
    aligned = labels.set_index(labels[Columns.CELL_ID].astype(str), drop=False).loc[
        table_ids
    ]

    hierarchy = build_hierarchy_columns(
        aligned,
        gate=LeafGate.from_provenance(provenance),
        min_branch_cells=int(config.min_branch_cells),
        root_markers=primary_root_markers(provenance),
    )
    params = EmbeddingParams.from_config(config)
    # Normalise before the label columns go in: the min_cells gene filter
    # subsets the AnnData, and anndata drops unused categories on every
    # subset, which would lose the fixed vocabularies of broad_class,
    # neuron_split_label and subcluster_status.
    normalize_table_cells(clustered, params)
    label_frame = label_obs_frame(aligned, species)
    for frame in (label_frame, hierarchy.obs):
        for column in frame.columns:
            values = frame[column]
            clustered.obs[column] = (
                values.array
                if isinstance(values.dtype, pd.api.extensions.ExtensionDtype)
                else values.to_numpy()
            )
    qc = whole_section_qc(
        clustered,
        resolution=float(config.qc_leiden_resolution),
        seed=int(config.random_seed),
        params=params,
    )
    stability_record: SubsampleStability | None = None
    if stability and stability_subsamples > 0:
        stability_record = subsample_ari(
            clustered,
            QC_LEIDEN_KEY,
            resolution=float(config.qc_leiden_resolution),
            n_subsamples=int(stability_subsamples),
            fraction=STABILITY_FRACTION,
            n_neighbors=qc.n_neighbors_used,
            seed=int(config.random_seed),
        )
    agreement = leaf_agreement_ari(clustered.obs)

    artifacts: dict[str, Path] = {}
    if plots:
        artifacts.update(
            write_map_first_plots(
                clustered,
                output_dir=output,
                sample_id=sample_id,
                hierarchy=hierarchy,
                config=config,
            )
        )
    artifacts.update(
        _write_tables(
            clustered, output, sample_id=sample_id, stability=stability_record
        )
    )
    manifest_path = output / f"{sample_id}_hierarchical_manifest.json"
    artifacts["hierarchical_manifest"] = manifest_path
    _record_uns(
        clustered,
        config=config,
        hierarchy=hierarchy,
        qc=qc,
        stability=stability_record,
        agreement=agreement,
        provenance=provenance,
        artifacts=artifacts,
    )
    manifest = _manifest_payload(
        clustered.uns[HIERARCHICAL_UNS_KEY], hierarchy, qc, stability_record
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = MapFirstResult(
        hierarchy=hierarchy,
        qc=qc,
        stability=stability_record,
        qc_leiden_vs_leaf_ari=agreement,
        wall_time_s=float(time.perf_counter() - started),
        artifacts=artifacts,
    )
    summary = hierarchy.summary()
    logger.info(
        "%s: map_first hierarchy of %d table cells: %d branches, %d leaves, "
        "%d hierarchical clusters; gate %s, panel trust %s; QC Leiden %d "
        "clusters%s (%.1f s)",
        sample_id,
        summary["n_cells"],
        summary["n_branches"],
        summary["n_leaves"],
        summary["n_hierarchical_clusters"],
        hierarchy.gate.gate_level,
        hierarchy.gate.panel_trust,
        qc.n_clusters,
        ""
        if stability_record is None
        else f", stability mean ARI {stability_record.mean_ari:.3f}",
        result.wall_time_s,
    )
    return clustered, result


def _record_uns(
    clustered: ad.AnnData,
    *,
    config: ClusteringSquidpyConfig,
    hierarchy: MapFirstHierarchy,
    qc: QcLeidenProvenance,
    stability: SubsampleStability | None,
    agreement: float,
    provenance: AnnotationProvenance | None,
    artifacts: Mapping[str, Path],
) -> None:
    """Write the §4.6 ``uns`` records: scalars and JSON strings only."""
    if provenance is not None:
        provenance.write_to_uns(clustered.uns)
    summary = hierarchy.summary()
    gate = hierarchy.gate
    clustered.uns[HIERARCHICAL_UNS_KEY] = _clean_scalars(
        {
            "enabled": True,
            "mode": MAP_FIRST_MODE,
            "branch_level": BRANCH_LEVEL,
            "leaf_level": hierarchy.leaf_level,
            "leaf_source": str(config.leaf_source),
            "species": hierarchy.species,
            "engine": qc.engine,
            "engine_n_iterations": int(qc.n_iterations),
            "engine_seed": int(qc.random_state),
            "engine_resolution": float(qc.resolution),
            "broad_cluster_key": BROAD_CLUSTER_KEY,
            "broad_atlas_label_key": BROAD_ATLAS_LABEL_KEY,
            "broad_class_key": BROAD_CLASS_KEY,
            "subcluster_label_key": SUBCLUSTER_LABEL_KEY,
            "subcluster_status_key": SUBCLUSTER_STATUS_KEY,
            "hierarchical_cluster_key": HIERARCHICAL_CLUSTER_KEY,
            "neuron_split_key": NEURON_SPLIT_KEY,
            "min_branch_cells": int(config.min_branch_cells),
            # FINALIZE refuses to write this table under another suffix
            # (plan §4.8: never overwrite the legacy clustered table).
            "table_key_suffix": str(config.table_key_suffix),
            "label_table_version": int(LABEL_TABLE_VERSION),
            "gate_level": gate.gate_level or MISSING_TEXT,
            "gate_warning": bool(gate.gate_warning),
            "panel_trust": gate.panel_trust or MISSING_TEXT,
            "leaves_suppressed": bool(gate.suppresses_leaves),
            "n_branches": int(summary["n_branches"]),
            "n_leaves": int(summary["n_leaves"]),
            "n_hierarchical_clusters": int(summary["n_hierarchical_clusters"]),
            "n_leaves_suppressed_gate": int(hierarchy.n_leaves_suppressed_gate),
            "n_leaves_unassigned_branch": int(hierarchy.n_leaves_unassigned_branch),
            "n_leaves_small_class": int(hierarchy.n_leaves_small_class),
            "n_branch_broad_mismatch": int(hierarchy.n_branch_broad_mismatch),
            "branch_manifest_json": _json_text(hierarchy.branch_manifest),
            "subcluster_status_counts_json": _json_text(
                summary["subcluster_status_counts"]
            ),
            "qc_leiden_vs_leaf_ari": float(agreement),
            "qc_stability_json": _json_text(
                None if stability is None else stability.to_dict()
            ),
            "qc_stability_mean_ari": (
                float("nan") if stability is None else float(stability.mean_ari)
            ),
            "artifacts_json": _json_text(
                {key: str(value) for key, value in sorted(artifacts.items())}
            ),
        }
    )
    params = _clean_scalars(
        {
            "mode": MAP_FIRST_MODE,
            "role": "qc_only",
            "drop_control_features": bool(config.drop_control_features),
            "min_counts": int(config.min_counts),
            "min_cells": int(config.min_cells),
            "key_added": QC_LEIDEN_KEY,
            "normalize_target_sum": config.normalize_target_sum,
            "normalize_exclude_highly_expressed": bool(
                config.normalize_exclude_highly_expressed
            ),
            "normalize_max_fraction": float(config.normalize_max_fraction),
            "n_pcs": int(config.n_pcs),
            "n_neighbors": int(config.n_neighbors),
            "effective_neighbors": int(qc.n_neighbors_used),
            "leiden_resolution": float(qc.resolution),
            "umap_min_dist": float(config.umap_min_dist),
            "umap_spread": float(config.umap_spread),
            "random_seed": int(qc.random_state),
            "gpu_used": False,
            **qc.as_uns_params(),
        }
    )
    clustered.uns[f"{CLUSTERING_PARAMS_UNS_KEY}_{QC_LEIDEN_KEY}"] = params
    clustered.uns[CLUSTERING_PARAMS_UNS_KEY] = dict(params)
    clustered.uns[LEIDEN_PROVENANCE_UNS_KEY] = {
        QC_LEIDEN_KEY: qc.to_json(key_added=QC_LEIDEN_KEY, role="qc_only"),
        QC_BROAD_LEIDEN_KEY: qc.to_json(
            key_added=QC_BROAD_LEIDEN_KEY, role="qc_only", copy_of=QC_LEIDEN_KEY
        ),
    }


def _manifest_payload(
    uns_record: Mapping[str, Any],
    hierarchy: MapFirstHierarchy,
    qc: QcLeidenProvenance,
    stability: SubsampleStability | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        key: value
        for key, value in uns_record.items()
        if not str(key).endswith("_json")
    }
    for key in ("qc_leiden_vs_leaf_ari", "qc_stability_mean_ari"):
        payload[key] = _finite_or_none(float(payload[key]))
    payload["branch_manifest"] = hierarchy.branch_manifest
    payload["summary"] = hierarchy.summary()
    payload["gate"] = asdict(hierarchy.gate)
    payload["gate"]["reasons"] = list(hierarchy.gate.reasons)
    payload["qc_leiden"] = asdict(qc)
    payload["qc_stability"] = None if stability is None else stability.to_dict()
    payload["artifacts"] = json.loads(str(uns_record["artifacts_json"]))
    return payload


def _write_tables(
    clustered: ad.AnnData,
    output: Path,
    *,
    sample_id: str,
    stability: SubsampleStability | None,
) -> dict[str, Path]:
    tables = output / "tables" / "map_first"
    tables.mkdir(parents=True, exist_ok=True)
    obs = clustered.obs
    counts = (
        obs.groupby(
            [Columns.CT_BRANCH, SUBCLUSTER_LABEL_KEY, SUBCLUSTER_STATUS_KEY],
            observed=True,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    leaves_path = tables / f"{sample_id}_map_first_leaves.csv"
    counts.to_csv(leaves_path, index=False)
    crosstab_path = tables / f"{sample_id}_qc_leiden_by_hierarchical_cluster.csv"
    pd.crosstab(obs[HIERARCHICAL_CLUSTER_KEY], obs[QC_LEIDEN_KEY]).to_csv(crosstab_path)
    paths = {"map_first_leaves": leaves_path, "qc_leiden_crosstab": crosstab_path}
    if stability is not None:
        stability_path = tables / f"{sample_id}_qc_leiden_stability.csv"
        pd.DataFrame(
            {
                "subsample": range(1, len(stability.aris) + 1),
                "ari": stability.aris,
                "n_clusters": stability.n_clusters_subsample,
            }
        ).to_csv(stability_path, index=False)
        paths["qc_leiden_stability"] = stability_path
    return paths


# --------------------------------------------------------------------------
# Plots


def _dotplot_genes(branch: ad.AnnData, max_genes: int) -> list[str]:
    genes = [str(name) for name in branch.var_names]
    if len(genes) <= max_genes:
        return genes
    matrix = branch.X
    means = np.asarray(matrix.mean(axis=0)).ravel()
    squares = (
        np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        if hasattr(matrix, "multiply")
        else np.asarray((np.asarray(matrix) ** 2).mean(axis=0)).ravel()
    )
    variance = squares - means**2
    order = np.argsort(-variance, kind="stable")[:max_genes]
    return [genes[index] for index in sorted(order)]


def write_map_first_plots(
    clustered: ad.AnnData,
    *,
    output_dir: Path,
    sample_id: str,
    hierarchy: MapFirstHierarchy,
    config: ClusteringSquidpyConfig,
) -> dict[str, Path]:
    """Write the map_first QC plots of one section (§6.2 step 5).

    UMAP and spatial maps by ``broad_class``, ``ct_leaf`` and
    ``ct_final_level`` (with the QC Leiden for comparison), and per branch a
    dotplot of the panel genes by leaf and, for branches of at least
    ``min_branch_cells`` cells, a branch UMAP by leaf. The plot helpers are
    legacy ``clustering_squidpy``'s, imported here so the module itself
    stays light.

    Args:
        clustered: Output of the hierarchy with the QC embedding.
        output_dir: Base directory (``plots/`` and ``tables/`` below it).
        sample_id: Sample id (file prefix).
        hierarchy: The section's hierarchy.
        config: The run's config (point sizes, dpi, ``min_branch_cells``).

    Returns:
        The written files by name.
    """
    from merxen.analysis import clustering_squidpy as legacy

    colors = [
        BROAD_CLASS_KEY,
        Columns.CT_LEAF,
        Columns.CT_FINAL_LEVEL,
        QC_LEIDEN_KEY,
        "total_counts",
        "n_genes_by_counts",
    ]
    prefix = f"{sample_id}_map_first"
    plots = output_dir / "plots"
    paths: dict[str, Path] = {
        f"{prefix}_umap": legacy.plot_umap(
            clustered,
            plots / "umap" / f"{prefix}_umap.png",
            color=colors,
            dpi=config.figure_dpi,
        )
    }
    if "spatial" in clustered.obsm:
        for color in (BROAD_CLASS_KEY, Columns.CT_LEAF, Columns.CT_FINAL_LEVEL):
            name = f"{prefix}_spatial_{color}"
            paths[name] = legacy.plot_spatial_scatter(
                clustered,
                plots / "spatial" / f"{name}.png",
                color=color,
                point_size=config.spatial_scatter_point_size,
                dpi=config.figure_dpi,
            )
        grid_name = f"{prefix}_spatial_grid_{BROAD_CLASS_KEY}"
        paths[grid_name] = legacy.plot_spatial_cluster_grid(
            clustered,
            plots / "spatial_grid" / f"{grid_name}.png",
            color=BROAD_CLASS_KEY,
            point_size_highlight=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
    branches = clustered.obs[Columns.CT_BRANCH].astype(str).to_numpy()
    for token, record in hierarchy.branch_manifest.items():
        rows = branches == record["branch"]
        if int(rows.sum()) < 3:
            continue
        branch = clustered[rows].copy()
        branch.obs[SUBCLUSTER_LABEL_KEY] = branch.obs[
            SUBCLUSTER_LABEL_KEY
        ].cat.remove_unused_categories()
        genes = _dotplot_genes(branch, DOTPLOT_MAX_GENES)
        mean, fraction = legacy.compute_group_gene_summary(
            branch, genes, groupby=SUBCLUSTER_LABEL_KEY
        )
        branch_prefix = f"{prefix}_branch_{token}"
        tables = output_dir / "tables" / "dotplot"
        tables.mkdir(parents=True, exist_ok=True)
        mean_path = tables / f"{branch_prefix}_gene_mean_expression.csv"
        fraction_path = tables / f"{branch_prefix}_gene_fraction_expressing.csv"
        mean.to_csv(mean_path)
        fraction.to_csv(fraction_path)
        paths[f"{branch_prefix}_gene_mean_expression"] = mean_path
        paths[f"{branch_prefix}_gene_fraction_expressing"] = fraction_path
        paths[f"{branch_prefix}_gene_dotplot"] = legacy.plot_group_gene_dotplot(
            mean,
            fraction,
            plots / "dotplot" / f"{branch_prefix}_gene_dotplot.png",
            cluster_genes=True,
            x_axis_label="Leaf",
            y_axis_label="Panel genes",
            title=f"{sample_id} {record['branch']}: panel genes by mapped leaf",
            dpi=max(int(config.figure_dpi), 220),
        )
        if int(rows.sum()) >= int(config.min_branch_cells):
            embedded = branch_embedding(
                clustered,
                rows,
                n_neighbors=int(config.n_neighbors),
                seed=int(config.random_seed),
                umap_min_dist=float(config.umap_min_dist),
                umap_spread=float(config.umap_spread),
            )
            if embedded is not None:
                paths[f"{branch_prefix}_umap"] = legacy.plot_umap(
                    embedded,
                    plots / "umap" / f"{branch_prefix}_umap.png",
                    color=[
                        SUBCLUSTER_LABEL_KEY,
                        Columns.CT_FINAL_LEVEL,
                        "total_counts",
                    ],
                    dpi=config.figure_dpi,
                )
    return paths
