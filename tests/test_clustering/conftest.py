"""Synthetic sections with label tables for the map_first clustering tests.

A section is an AnnData of raw counts (control features already removed, as
``run_map_first_hierarchy`` expects) plus its label table (plan §4.1) built
the way RESOLVE writes it, and optionally its ``AnnotationProvenance``.
Each cell type has its own gene block, so the QC Leiden finds structure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.provenance import (
    AnnotationProvenance,
    DatasetGateProvenance,
    MarkerProvenance,
    MouseGateProvenance,
    PanelProvenance,
    ReferenceProvenance,
)
from merxen.annotation.schema import (
    LEVELS,
    CellStatus,
    Columns,
    coerce_label_table_dtypes,
    soft_columns,
)
from merxen.annotation.vocab import FINAL_LEVELS, UNASSIGNED_LABEL, UNRESOLVED_LABEL
from merxen.config import ClusteringSquidpyConfig, ClusteringSquidpySampleConfig

C = CellStatus.CONFIDENT
LOW = CellStatus.LOW_CONFIDENCE
FLOOR = CellStatus.BELOW_FLOOR
NA = CellStatus.NOT_APPLICABLE
PARENT = CellStatus.PARENT_UNRESOLVED
GATE = CellStatus.NOT_ATTEMPTED_GATE
NOT_RES = CellStatus.NOT_RESOLVABLE
LC = CellStatus.LOW_COUNTS

MIN_COUNTS = 10
N_GENES = 60


@dataclass(frozen=True)
class CellType:
    """One synthetic cell population and the labels RESOLVE gives it.

    Attributes:
        key: Short name.
        n_cells: Number of objects.
        levels: Per annotation level, ``(status, name)``.
        final: ``(ct_final_level, ct_final_name)``.
        branch: ``ct_branch``.
        leaf: ``ct_leaf``.
        block: Index of its marker gene block (``None``: diffuse profile).
        depth: Mean counts per object.
    """

    key: str
    n_cells: int
    levels: dict[str, tuple[str, str | None]]
    final: tuple[str, str]
    branch: str
    leaf: str
    block: int | None
    depth: float = 120.0


def _human(
    lineage: tuple[str, str | None],
    broad: tuple[str, str | None],
    nt: tuple[str, str | None],
    supercluster: tuple[str, str | None],
    seaad: tuple[str, str | None],
) -> dict[str, tuple[str, str | None]]:
    return {
        "lineage": lineage,
        "broad": broad,
        "nt": nt,
        "supercluster": supercluster,
        "seaad_subclass": seaad,
    }


UL_IT = "Upper-layer intratelencephalic"
HUMAN_TYPES: tuple[CellType, ...] = (
    CellType(
        "ul_it",
        150,
        _human(
            (C, "Neurons"),
            (C, "Neurons"),
            (C, "Excitatory"),
            (C, UL_IT),
            (C, "L2/3 IT"),
        ),
        ("supercluster", UL_IT),
        "Neurons/Excitatory",
        UL_IT,
        0,
    ),
    CellType(
        "mge",
        110,
        _human(
            (C, "Neurons"),
            (C, "Neurons"),
            (C, "Inhibitory"),
            (C, "MGE interneuron"),
            (C, "Sst"),
        ),
        ("supercluster", "MGE interneuron"),
        "Neurons/Inhibitory",
        "MGE interneuron",
        1,
    ),
    CellType(
        "neuron_nt_unresolved",
        60,
        _human(
            (C, "Neurons"),
            (C, "Neurons"),
            (LOW, "Excitatory"),
            (PARENT, UL_IT),
            (LOW, "L2/3 IT"),
        ),
        ("broad", "Neurons"),
        "Neurons/unresolved",
        UNRESOLVED_LABEL,
        0,
        40.0,
    ),
    CellType(
        "astro",
        110,
        _human(
            (C, "Astrocytes"),
            (C, "Astrocytes"),
            (NA, None),
            (C, "Astrocyte"),
            (C, "Astrocyte"),
        ),
        ("supercluster", "Astrocyte"),
        "Astrocytes",
        "Astrocyte",
        2,
    ),
    CellType(
        "astro_not_resolvable",
        30,
        _human(
            (C, "Astrocytes"),
            (C, "Astrocytes"),
            (NA, None),
            (NOT_RES, "Astrocyte"),
            (LOW, "Astrocyte"),
        ),
        ("broad", "Astrocytes"),
        "Astrocytes",
        UNRESOLVED_LABEL,
        2,
        25.0,
    ),
    CellType(
        "oligo",
        120,
        _human(
            (C, "Oligodendrocyte lineage"),
            (C, "Oligodendrocytes"),
            (NA, None),
            (C, "Oligodendrocyte"),
            (C, "Oligodendrocyte"),
        ),
        ("supercluster", "Oligodendrocyte"),
        "Oligodendrocytes",
        "Oligodendrocyte",
        3,
    ),
    CellType(
        "oligo_lineage",
        50,
        _human(
            (C, "Oligodendrocyte lineage"),
            (FLOOR, "Oligodendrocyte precursors"),
            (NA, None),
            (PARENT, "Oligodendrocyte precursor"),
            (PARENT, "OPC"),
        ),
        ("lineage", "Oligodendrocyte lineage"),
        "Oligodendrocyte lineage/unresolved",
        UNRESOLVED_LABEL,
        3,
        14.0,
    ),
    CellType(
        "microglia_small",
        20,
        _human(
            (C, "Microglia"),
            (C, "Microglia"),
            (NA, None),
            (C, "Microglia"),
            (C, "Microglia-PVM"),
        ),
        ("supercluster", "Microglia"),
        "Microglia",
        "Microglia",
        4,
    ),
    CellType(
        "mixed",
        45,
        _human(
            (LOW, "Neurons"),
            (PARENT, "Neurons"),
            (PARENT, "Inhibitory"),
            (PARENT, "MGE interneuron"),
            (PARENT, "Sst"),
        ),
        ("none", UNASSIGNED_LABEL),
        UNASSIGNED_LABEL,
        UNRESOLVED_LABEL,
        None,
        14.0,
    ),
)

MOUSE_TYPES: tuple[CellType, ...] = (
    CellType(
        "it",
        140,
        {
            "broad": (C, "Neurons"),
            "class": (C, "01 IT-ET Glut"),
            "nt": (C, "Excitatory"),
            "subclass": (C, "007 L2/3 IT CTX Glut"),
        },
        ("subclass", "007 L2/3 IT CTX Glut"),
        "01 IT-ET Glut",
        "007 L2/3 IT CTX Glut",
        0,
    ),
    CellType(
        "cge",
        60,
        {
            "broad": (C, "Neurons"),
            "class": (C, "06 CTX-CGE GABA"),
            "nt": (C, "Inhibitory"),
            "subclass": (LOW, "046 Vip Gaba"),
        },
        ("nt", "Inhibitory"),
        "06 CTX-CGE GABA",
        UNRESOLVED_LABEL,
        1,
        60.0,
    ),
    CellType(
        "astro",
        90,
        {
            "broad": (C, "Astrocytes/Ependymal"),
            "class": (C, "30 Astro-Epen"),
            "nt": (NA, None),
            "subclass": (C, "319 Astro-TE NN"),
        },
        ("subclass", "319 Astro-TE NN"),
        "30 Astro-Epen",
        "319 Astro-TE NN",
        2,
    ),
    CellType(
        "oligo_small",
        30,
        {
            "broad": (C, "Oligodendrocyte lineage"),
            "class": (C, "31 OPC-Oligo"),
            "nt": (NA, None),
            "subclass": (C, "327 Oligo NN"),
        },
        ("subclass", "327 Oligo NN"),
        "31 OPC-Oligo",
        "327 Oligo NN",
        3,
    ),
    CellType(
        "mixed",
        40,
        {
            "broad": (LOW, "Neurons"),
            "class": (PARENT, "01 IT-ET Glut"),
            "nt": (PARENT, "Excitatory"),
            "subclass": (PARENT, "007 L2/3 IT CTX Glut"),
        },
        ("none", UNASSIGNED_LABEL),
        UNASSIGNED_LABEL,
        UNRESOLVED_LABEL,
        None,
        14.0,
    ),
)


def broad_only(cell_type: CellType, *, leaf_level: str = "supercluster") -> CellType:
    """Return ``cell_type`` as RESOLVE labels it on a broad-only dataset.

    Levels below broad/NT become ``not_attempted_gate`` without a name, the
    leaf is ``unresolved`` and the final level is capped at ``nt``.
    """
    levels = dict(cell_type.levels)
    fine = [leaf_level] + (["seaad_subclass"] if leaf_level == "supercluster" else [])
    for level in fine:
        if level in levels:
            levels[level] = (GATE, None)
    final = cell_type.final
    if final[0] == leaf_level:
        # The deepest confident level left, in the species chain order.
        chain = FINAL_LEVELS["human" if leaf_level == "supercluster" else "mouse"]
        final = ("none", UNASSIGNED_LABEL)
        for level in chain[1:]:
            status, name = levels.get(level, (NA, None))
            if status == C:
                final = (level, str(name))
    return CellType(
        cell_type.key,
        cell_type.n_cells,
        levels,
        final,
        cell_type.branch,
        UNRESOLVED_LABEL,
        cell_type.block,
        cell_type.depth,
    )


@dataclass
class Section:
    """A synthetic section: counts, labels and provenance."""

    adata: ad.AnnData
    labels: pd.DataFrame
    provenance: AnnotationProvenance
    species: str
    type_of_cell: pd.Series = field(default_factory=pd.Series)


def _counts(
    types: Sequence[CellType], n_low: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    rows = []
    keys = []
    block_size = 8
    for cell_type in types:
        for _ in range(cell_type.n_cells):
            lam = np.full(N_GENES, 0.15)
            if cell_type.block is not None:
                start = cell_type.block * block_size
                lam[start : start + block_size] = 3.0
            else:
                lam[:] = 0.6
            lam *= cell_type.depth / lam.sum()
            counts = rng.poisson(lam)
            while counts.sum() < MIN_COUNTS:
                counts[rng.integers(N_GENES)] += 1
            rows.append(counts)
            keys.append(cell_type.key)
    for _ in range(n_low):
        counts = np.zeros(N_GENES, dtype=int)
        counts[rng.choice(N_GENES, size=int(rng.integers(0, MIN_COUNTS)))] = 1
        rows.append(counts)
        keys.append("low_counts")
    return np.asarray(rows, dtype=np.float32), keys


def build_label_table(
    species: str,
    types: Sequence[CellType],
    keys: Sequence[str],
    counts: np.ndarray,
    cell_ids: Sequence[str],
) -> pd.DataFrame:
    """Return a valid label table for the objects of a synthetic section."""
    by_key = {cell_type.key: cell_type for cell_type in types}
    totals = counts.sum(axis=1).astype(np.int32)
    n_genes = (counts > 0).sum(axis=1).astype(np.int32)
    in_table = totals >= MIN_COUNTS
    n_objects = len(keys)
    data: dict[str, Any] = {
        Columns.CELL_ID: list(cell_ids),
        Columns.INSTANCE_ID: np.arange(1, n_objects + 1, dtype=np.int64),
        Columns.PAIR_ID: ["P0001"] * n_objects,
        Columns.SAMPLE_ID: ["P0001_MERSCOPE"] * n_objects,
        Columns.PLATFORM: ["MERSCOPE"] * n_objects,
        Columns.SEGMENTATION: ["proseg_hybrid"] * n_objects,
        Columns.SPECIES: [species] * n_objects,
        Columns.ANATOMICAL_REGION: (
            ["frontal_cortex"] * n_objects if species == "human" else [None] * n_objects
        ),
        Columns.PANEL_HASH: ["a" * 64] * n_objects,
        Columns.TOTAL_COUNTS: totals,
        Columns.N_GENES: n_genes,
        Columns.GENES_PER_COUNT: np.where(
            totals > 0, n_genes / np.maximum(totals, 1), np.nan
        ).astype(np.float32),
        Columns.IN_TABLE: in_table,
        Columns.DEPTH_BIN: pd.array(
            [int(min(t, 250)) if t >= MIN_COUNTS else None for t in totals],
            dtype="Int32",
        ),
        Columns.RESOLVABILITY_EXTRAPOLATED: totals > 250,
        Columns.N_MISSING_PANEL_GENES: np.zeros(n_objects, dtype=np.int32),
    }
    for level in LEVELS[species]:
        statuses = []
        names: list[str | None] = []
        for key, table in zip(keys, in_table, strict=True):
            if not table or key == "low_counts":
                statuses.append(str(LC))
                names.append(None)
                continue
            status, name = by_key[key].levels[level]
            statuses.append(str(status))
            names.append(name)
        attempted = np.array([name is not None for name in names])
        data[Columns.level(level, "name")] = names
        data[Columns.level(level, "raw")] = np.where(attempted, 0.8, np.nan)
        data[Columns.level(level, "conf")] = np.where(attempted, 0.8, np.nan)
        data[Columns.level(level, "corr")] = np.where(attempted, 0.5, np.nan)
        data[Columns.level(level, "runner_up")] = [None] * n_objects
        data[Columns.level(level, "margin")] = np.where(attempted, 0.3, np.nan)
        data[Columns.level(level, "status")] = statuses
        data[Columns.level(level, "validated")] = [
            status == CellStatus.CONFIDENT for status in statuses
        ]

    def per_cell(getter: Callable[[CellType], Any], default: Any) -> list[Any]:
        return [
            getter(by_key[key]) if table and key != "low_counts" else default
            for key, table in zip(keys, in_table, strict=True)
        ]

    data[Columns.CT_FINAL_LEVEL] = per_cell(lambda t: t.final[0], "none")
    data[Columns.CT_FINAL_NAME] = per_cell(lambda t: t.final[1], UNASSIGNED_LABEL)
    data[Columns.CT_CONSENSUS_TIER] = np.where(in_table, 2, -1).astype(np.int8)
    data[Columns.CT_BRANCH] = per_cell(lambda t: t.branch, UNASSIGNED_LABEL)
    data[Columns.CT_LEAF] = per_cell(lambda t: t.leaf, UNRESOLVED_LABEL)
    data[Columns.CT_MENDER_STATE] = data[Columns.CT_BRANCH]
    false = np.zeros(n_objects, dtype=bool)
    contaminated = pd.array(
        [bool(i % 7 == 0) if t else None for i, t in enumerate(in_table)],
        dtype="boolean",
    )
    data.update(
        {
            Columns.FLAG_LOW_COUNTS: ~in_table,
            Columns.FLAG_BELOW_FLOOR: false.copy(),
            Columns.FLAG_METHOD_DISAGREE: false.copy(),
            Columns.FLAG_IMPLAUSIBLE: false.copy(),
            Columns.CONTAMINATION_SCORE: np.where(in_table, 0.01, np.nan),
            Columns.NEG_COUNTS: pd.array(
                [int(i % 3) if t else None for i, t in enumerate(in_table)],
                dtype="Int32",
            ),
            Columns.FLAG_CONTAMINATED: contaminated,
            Columns.EXPECTED_GENES_Q95: np.where(in_table, 30.0, np.nan),
            Columns.FLAG_DIFFUSE_PROFILE: pd.array(
                [False if t else None for t in in_table], dtype="boolean"
            ),
            Columns.OOD_Z: np.where(in_table, 0.0, np.nan),
            Columns.FLAG_OOD: pd.array(
                [False if t else None for t in in_table], dtype="boolean"
            ),
            Columns.MICROGLIA_STAT: np.full(n_objects, np.nan),
            Columns.MICROGLIA_WEIGHT: np.full(n_objects, np.nan),
            Columns.FLAG_MICROGLIAL_SPILLOVER: pd.array(
                [None] * n_objects, dtype="boolean"
            ),
            Columns.EXCLUDE_HARD: ~in_table,
            Columns.DISCOVERY_CAUTION: np.asarray(contaminated.fillna(False)),
        }
    )
    if species == "human":
        data[Columns.FLAG_COP_SUPPRESSED] = false.copy()
    else:
        data[Columns.FLAG_REGION_INCOHERENT] = pd.array(
            [False] * n_objects, dtype="boolean"
        )
        data[Columns.REGION_COHERENCE] = np.full(n_objects, 0.9)
        data[Columns.FLAG_ASTRO_LOWCOUNT] = false.copy()
    for position, column in enumerate(soft_columns(species)):  # type: ignore[arg-type]
        data[column] = np.full(n_objects, 1.0 if position == 0 else 0.0)
    data["mmc_whb_supercluster_bp"] = np.full(n_objects, 0.5)
    frame = pd.DataFrame(data)
    return coerce_label_table_dtypes(frame, species)  # type: ignore[arg-type]


def make_provenance(
    species: str,
    *,
    gate_level: str = "full",
    panel_trust: str = "validated",
    root_markers: int = 42,
) -> AnnotationProvenance:
    """Return provenance with a gate, a panel trust and a primary reference."""
    reference_id = "whb_frontal_supc_clus" if species == "human" else "wmb_panel"
    reference = ReferenceProvenance(
        reference_id=reference_id,
        role="primary",
        markers=MarkerProvenance(root_markers=root_markers, n_per_utility=30),
    )
    kwargs: dict[str, Any] = {
        "species": species,
        "panel": PanelProvenance(panel_hash="a" * 64, panel_trust=panel_trust),
        "references": {reference_id: reference},
    }
    if species == "human":
        kwargs["anatomical_region"] = "frontal_cortex"
        kwargs["gate"] = DatasetGateProvenance(
            level=gate_level,
            warning=gate_level != "full",
            reasons=[] if gate_level == "full" else ["frac_ge30: A < 0.3"],
        )
    else:
        kwargs["mouse_gate"] = MouseGateProvenance(level=gate_level)
    return AnnotationProvenance(**kwargs)


def make_section(
    species: str = "human",
    *,
    types: Sequence[CellType] | None = None,
    n_low: int = 25,
    seed: int = 0,
    gate_level: str = "full",
    panel_trust: str = "validated",
) -> Section:
    """Return a synthetic section of one species."""
    rng = np.random.default_rng(seed)
    cell_types = tuple(
        types
        if types is not None
        else (HUMAN_TYPES if species == "human" else MOUSE_TYPES)
    )
    counts, keys = _counts(cell_types, n_low, rng)
    order = rng.permutation(len(keys))
    counts = counts[order]
    keys = [keys[index] for index in order]
    cell_ids = [f"cell_{index:05d}" for index in range(len(keys))]
    labels = build_label_table(species, cell_types, keys, counts, cell_ids)
    genes = [f"GENE{index:03d}" for index in range(N_GENES)]
    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "instance_id": np.arange(1, len(keys) + 1, dtype=np.int64),
            "region": pd.Categorical(["cell_boundaries"] * len(keys)),
            "total_counts": counts.sum(axis=1),
            "n_genes_by_counts": (counts > 0).sum(axis=1),
        },
        index=pd.Index(cell_ids),
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    adata.obsm["spatial"] = rng.uniform(0.0, 2000.0, size=(len(keys), 2))
    adata.uns["spatialdata_attrs"] = {
        "region": "cell_boundaries",
        "region_key": "region",
        "instance_key": "instance_id",
    }
    adata.uns["merxen_clustering_squidpy"] = {
        "table_key": "table",
        "shape_key": "cell_boundaries",
    }
    return Section(
        adata=adata,
        labels=labels,
        provenance=make_provenance(
            species, gate_level=gate_level, panel_trust=panel_trust
        ),
        species=species,
        type_of_cell=pd.Series(keys, index=pd.Index(cell_ids)),
    )


def make_config(tmp_path: Path, **overrides: Any) -> ClusteringSquidpyConfig:
    """Return a map_first ``ClusteringSquidpyConfig`` for a synthetic section."""
    settings: dict[str, Any] = {
        "pair_id": "P0001",
        "output_dir": tmp_path / "out",
        "samples": [
            ClusteringSquidpySampleConfig(
                sample_id="P0001_MERSCOPE",
                platform="MERSCOPE",
                zarr_path=tmp_path / "missing.zarr",
                segmentation="proseg_hybrid",
            )
        ],
        "mode": "map_first",
        "table_key_suffix": "mapfirst",
        "use_gpu": False,
        "n_pcs": 10,
        "n_neighbors": 15,
        "figure_dpi": 72,
    }
    settings.update(overrides)
    return ClusteringSquidpyConfig(**settings)


@pytest.fixture
def human_section() -> Section:
    """A human section with every branch kind (full gate, validated panel)."""
    return make_section("human")


@pytest.fixture
def mouse_section() -> Section:
    """A mouse section with a class under ``min_branch_cells``."""
    return make_section("mouse")
