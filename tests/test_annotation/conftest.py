"""Shared fixtures for the annotation contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.schema import (
    LEVELS,
    CellStatus,
    Columns,
    coerce_label_table_dtypes,
    soft_columns,
)
from merxen.annotation.vocab import UNASSIGNED_LABEL, UNRESOLVED_LABEL

C = CellStatus.CONFIDENT
LOW = CellStatus.LOW_CONFIDENCE
FLOOR = CellStatus.BELOW_FLOOR
NA = CellStatus.NOT_APPLICABLE
PARENT = CellStatus.PARENT_UNRESOLVED
LC = CellStatus.LOW_COUNTS

# One row per synthetic cell: counts, per-level (status, name), final level,
# final name, branch, leaf. The first object is below min_counts.
HUMAN_CELLS: list[dict[str, Any]] = [
    {
        "counts": 5,
        "levels": {level: (LC, None) for level in LEVELS["human"]},
        "final": ("none", UNASSIGNED_LABEL),
        "branch": UNASSIGNED_LABEL,
        "leaf": UNRESOLVED_LABEL,
    },
    {
        "counts": 240,
        "levels": {
            "lineage": (C, "Neurons"),
            "broad": (C, "Neurons"),
            "nt": (C, "Excitatory"),
            "supercluster": (C, "Upper-layer intratelencephalic"),
            "seaad_subclass": (C, "L2/3 IT"),
        },
        "final": ("supercluster", "Upper-layer intratelencephalic"),
        "branch": "Neurons/Excitatory",
        "leaf": "Upper-layer intratelencephalic",
    },
    {
        "counts": 80,
        "levels": {
            "lineage": (C, "Astrocytes"),
            "broad": (C, "Astrocytes"),
            "nt": (NA, None),
            "supercluster": (LOW, "Astrocyte"),
            "seaad_subclass": (LOW, "Astrocyte"),
        },
        "final": ("broad", "Astrocytes"),
        "branch": "Astrocytes",
        "leaf": UNRESOLVED_LABEL,
    },
    {
        "counts": 14,
        "levels": {
            "lineage": (C, "Oligodendrocyte lineage"),
            "broad": (FLOOR, "Oligodendrocyte precursors"),
            "nt": (NA, None),
            "supercluster": (PARENT, "Oligodendrocyte precursor"),
            "seaad_subclass": (PARENT, "OPC"),
        },
        "final": ("lineage", "Oligodendrocyte lineage"),
        "branch": "Oligodendrocyte lineage/unresolved",
        "leaf": UNRESOLVED_LABEL,
    },
    {
        "counts": 11,
        "levels": {
            "lineage": (LOW, "Neurons"),
            "broad": (PARENT, "Neurons"),
            "nt": (PARENT, "Inhibitory"),
            "supercluster": (PARENT, "MGE interneuron"),
            "seaad_subclass": (PARENT, "Sst"),
        },
        "final": ("none", UNASSIGNED_LABEL),
        "branch": UNASSIGNED_LABEL,
        "leaf": UNRESOLVED_LABEL,
    },
]

MOUSE_CELLS: list[dict[str, Any]] = [
    {
        "counts": 3,
        "levels": {level: (LC, None) for level in LEVELS["mouse"]},
        "final": ("none", UNASSIGNED_LABEL),
        "branch": UNASSIGNED_LABEL,
        "leaf": UNRESOLVED_LABEL,
    },
    {
        "counts": 600,
        "levels": {
            "broad": (C, "Neurons"),
            "class": (C, "01 IT-ET Glut"),
            "nt": (C, "Excitatory"),
            "subclass": (C, "007 L2/3 IT CTX Glut"),
        },
        "final": ("subclass", "007 L2/3 IT CTX Glut"),
        "branch": "01 IT-ET Glut",
        "leaf": "007 L2/3 IT CTX Glut",
    },
    {
        "counts": 35,
        "levels": {
            "broad": (C, "Astrocytes/Ependymal"),
            "class": (C, "30 Astro-Epen"),
            "nt": (NA, None),
            "subclass": (FLOOR, "319 Astro-TE NN"),
        },
        "final": ("class", "30 Astro-Epen"),
        "branch": "30 Astro-Epen",
        "leaf": UNRESOLVED_LABEL,
    },
]


def _build_table(species: str, cells: list[dict[str, Any]]) -> pd.DataFrame:
    n_cells = len(cells)
    counts = np.array([cell["counts"] for cell in cells], dtype=np.int32)
    in_table = counts >= 10
    data: dict[str, Any] = {
        Columns.CELL_ID: [f"cell_{index}" for index in range(n_cells)],
        Columns.INSTANCE_ID: np.arange(1, n_cells + 1, dtype=np.int64),
        Columns.PAIR_ID: ["P0001"] * n_cells,
        Columns.SAMPLE_ID: ["P0001_MERSCOPE"] * n_cells,
        Columns.PLATFORM: ["MERSCOPE"] * n_cells,
        Columns.SEGMENTATION: ["proseg_hybrid"] * n_cells,
        Columns.SPECIES: [species] * n_cells,
        Columns.ANATOMICAL_REGION: (
            ["frontal_cortex"] * n_cells if species == "human" else [None] * n_cells
        ),
        Columns.PANEL_HASH: ["a" * 64] * n_cells,
        Columns.TOTAL_COUNTS: counts,
        Columns.N_GENES: np.minimum(counts, 40).astype(np.int32),
        Columns.GENES_PER_COUNT: (np.minimum(counts, 40) / counts).astype(np.float32),
        Columns.IN_TABLE: in_table,
        Columns.DEPTH_BIN: pd.array(
            [int(min(count, 250)) if count >= 10 else None for count in counts],
            dtype="Int32",
        ),
        Columns.RESOLVABILITY_EXTRAPOLATED: counts > 250,
        Columns.N_MISSING_PANEL_GENES: np.zeros(n_cells, dtype=np.int32),
    }
    for level in LEVELS[species]:
        statuses = [str(cell["levels"][level][0]) for cell in cells]
        names = [cell["levels"][level][1] for cell in cells]
        attempted = [name is not None for name in names]
        data[Columns.level(level, "name")] = names
        data[Columns.level(level, "raw")] = np.where(attempted, 0.8, np.nan)
        data[Columns.level(level, "conf")] = np.where(attempted, 0.8, np.nan)
        data[Columns.level(level, "corr")] = np.where(attempted, 0.5, np.nan)
        data[Columns.level(level, "runner_up")] = [None] * n_cells
        data[Columns.level(level, "margin")] = np.where(attempted, 0.3, np.nan)
        data[Columns.level(level, "status")] = statuses
        data[Columns.level(level, "validated")] = [
            status == CellStatus.CONFIDENT for status in statuses
        ]
    data[Columns.CT_FINAL_LEVEL] = [cell["final"][0] for cell in cells]
    data[Columns.CT_FINAL_NAME] = [cell["final"][1] for cell in cells]
    data[Columns.CT_CONSENSUS_TIER] = np.full(
        n_cells, 2 if species == "human" else 1, dtype=np.int8
    )
    data[Columns.CT_BRANCH] = [cell["branch"] for cell in cells]
    data[Columns.CT_LEAF] = [cell["leaf"] for cell in cells]
    data[Columns.CT_MENDER_STATE] = [cell["branch"] for cell in cells]
    false = np.zeros(n_cells, dtype=bool)
    nullable_false = pd.array([False] * n_cells, dtype="boolean")
    data.update(
        {
            Columns.FLAG_LOW_COUNTS: ~in_table,
            Columns.FLAG_BELOW_FLOOR: false.copy(),
            Columns.FLAG_METHOD_DISAGREE: false.copy(),
            Columns.FLAG_IMPLAUSIBLE: false.copy(),
            Columns.CONTAMINATION_SCORE: np.where(in_table, 0.01, np.nan),
            Columns.NEG_COUNTS: pd.array([0] * n_cells, dtype="Int32"),
            Columns.FLAG_CONTAMINATED: nullable_false.copy(),
            Columns.EXPECTED_GENES_Q95: np.full(n_cells, 30.0),
            Columns.FLAG_DIFFUSE_PROFILE: nullable_false.copy(),
            Columns.OOD_Z: np.zeros(n_cells),
            Columns.FLAG_OOD: nullable_false.copy(),
            Columns.MICROGLIA_STAT: np.full(n_cells, np.nan),
            Columns.MICROGLIA_WEIGHT: np.full(n_cells, np.nan),
            Columns.FLAG_MICROGLIAL_SPILLOVER: pd.array(
                [None] * n_cells, dtype="boolean"
            ),
            Columns.EXCLUDE_HARD: ~in_table,
            Columns.DISCOVERY_CAUTION: false.copy(),
        }
    )
    if species == "human":
        data[Columns.FLAG_COP_SUPPRESSED] = false.copy()
    else:
        data[Columns.FLAG_REGION_INCOHERENT] = nullable_false.copy()
        data[Columns.REGION_COHERENCE] = np.full(n_cells, 0.9)
        data[Columns.FLAG_ASTRO_LOWCOUNT] = false.copy()
    soft = soft_columns(species)  # type: ignore[arg-type]
    for position, column in enumerate(soft):
        data[column] = np.full(n_cells, 1.0 if position == 0 else 0.0)
    frame = pd.DataFrame(data)
    return coerce_label_table_dtypes(frame, species)  # type: ignore[arg-type]


@pytest.fixture
def make_label_table() -> Callable[[str], pd.DataFrame]:
    """Return a factory for small valid label tables of either species."""

    def factory(species: str) -> pd.DataFrame:
        cells = HUMAN_CELLS if species == "human" else MOUSE_CELLS
        return _build_table(species, cells)

    return factory
