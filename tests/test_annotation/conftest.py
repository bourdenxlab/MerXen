"""Shared fixtures for the annotation contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

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


# --------------------------------------------------------------------------
# Fake MapMyCells: tiny bundles and a fake mapper subprocess (MAP tests)


# Display names of the fake taxonomy levels, as ctm's hierarchy_mapper has them.
FAKE_LEVEL_NAMES: dict[str, str] = {
    "CCN202210140_SUPC": "supercluster",
    "CCN202210140_CLUS": "cluster",
    "SEA_LEVEL_0": "Class",
    "SEA_LEVEL_1": "Subclass",
    "SEA_LEVEL_2": "Supertype",
    "CCN20230722_CLAS": "class",
    "CCN20230722_SUBC": "subclass",
    "CCN20230722_SUPT": "supertype",
    "CCN20230722_CLUS": "cluster",
    "FAKE_SUPC": "supercluster",
    "FAKE_CLUS": "cluster",
}


@dataclass(frozen=True)
class FakeNode:
    """A first-level node of a fake taxonomy, with its marker gene."""

    label: str
    name: str
    marker: str
    vocab: dict[str, str]


@dataclass
class FakeMmc:
    """Build fake bundles and stand in for the mapper subprocess.

    The fake mapper assigns each query cell to the first-level node whose
    marker gene has the most counts; its bootstrap probability is that
    marker's share of the marker counts (rounded to 0.01), the other nodes
    are the runner-ups, and every deeper level repeats the node's single
    child with the same probability.
    """

    root: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_with: int | None = None
    record_seed: int | None = None

    def bundle(
        self,
        reference_id: str,
        *,
        role: str,
        species: str,
        panel_hash: str,
        n_genes: int,
        levels: list[str],
        nodes: list[FakeNode],
        drop_level: str | None = None,
        builder_version: int | None = None,
        build_hash: str | None = None,
        store_root: Path | None = None,
    ) -> Path:
        from merxen.annotation.reference import lookup_sha256
        from merxen.annotation.store import (
            ANNOTATION_BUILDER_VERSION,
            STORE_SCHEMA_VERSION,
        )

        build_hash = (
            build_hash
            or hashlib.sha256(
                f"{reference_id}:{panel_hash}:{builder_version}".encode()
            ).hexdigest()
        )
        bundle_dir = (store_root or self.root) / reference_id / build_hash
        bundle_dir.mkdir(parents=True)
        tree: dict[str, Any] = {"hierarchy": levels}
        names: dict[str, dict[str, dict[str, str]]] = {level: {} for level in levels}
        for depth, level in enumerate(levels):
            entries: dict[str, list[str]] = {}
            for node in nodes:
                label = node.label if depth == 0 else f"{node.label}_{depth}"
                child = [f"{node.label}_{depth + 1}"] if depth + 1 < len(levels) else []
                entries[label] = child
                names[level][label] = {
                    "name": node.name if depth == 0 else f"{node.name} {depth}"
                }
            tree[level] = entries
        tree["name_mapper"] = names
        (bundle_dir / "mapping_tree.json").write_text(json.dumps(tree))
        (bundle_dir / "mapping_precompute.h5").write_bytes(b"fake precompute")
        lookup: dict[str, Any] = {"None": [node.marker for node in nodes]}
        for depth, level in enumerate(levels[:-1]):
            for node in nodes:
                label = node.label if depth == 0 else f"{node.label}_{depth}"
                lookup[f"{level}/{label}"] = []
        lookup_path = bundle_dir / "query_markers.filtered.json"
        lookup_path.write_text(json.dumps(lookup))
        (bundle_dir / "fake_markers.json").write_text(
            json.dumps({node.label: node.marker for node in nodes})
        )
        rows = []
        for node in nodes:
            rows.append(
                {
                    "level": levels[0],
                    "node": node.label,
                    "node_name": node.name,
                    "broad_class": "",
                    "nt": "",
                    "lineage": "",
                    "sink": "False",
                    "region_plausible_frontal_cortex": "True",
                    "never_drop": "",
                    **node.vocab,
                }
            )
        pd.DataFrame(rows).to_csv(bundle_dir / "vocab_snapshot.csv", index=False)
        files = [
            {"path": path.name, "size": path.stat().st_size}
            for path in sorted(bundle_dir.iterdir())
        ]
        manifest = {
            "schema_version": STORE_SCHEMA_VERSION,
            "status": "complete",
            "build_hash": build_hash,
            "reference_id": reference_id,
            "species": species,
            "role": role,
            "builder_version": builder_version or ANNOTATION_BUILDER_VERSION,
            "taxonomy_id": "FAKE",
            "panel": {"panel_hash": panel_hash, "n_genes": n_genes},
            "builder_output": {
                "levels": [level for level in levels if level != drop_level],
                "drop_level": drop_level,
                "mapping_precompute": {"file": "mapping_precompute.h5"},
                "mapping_tree_file": "mapping_tree.json",
                "vocab_snapshot_file": "vocab_snapshot.csv",
                "markers": {
                    "lookup_file": lookup_path.name,
                    "lookup_sha256": lookup_sha256(lookup),
                },
            },
            "files": files,
            "created_at": "2026-09-26T00:00:00+00:00",
        }
        (bundle_dir / "bundle.json").write_text(json.dumps(manifest))
        return bundle_dir

    def install(self, monkeypatch: pytest.MonkeyPatch, ctm_version: str) -> None:
        """Replace the mapper subprocess and the installed ctm version."""
        from merxen.annotation import mapmycells_engine

        monkeypatch.setattr(
            mapmycells_engine, "installed_ctm_version", lambda: ctm_version
        )
        monkeypatch.setattr(mapmycells_engine.subprocess, "run", self.run)

    def run(
        self,
        command: list[str],
        check: bool,
        stdout: TextIO,
        stderr: TextIO,
        env: dict[str, str],
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """The fake ``subprocess.run`` of the mapper."""
        import anndata as ad

        assert check is False and text is True

        def option(name: str) -> str | None:
            return command[command.index(name) + 1] if name in command else None

        self.calls.append({"command": list(command), "env": dict(env)})
        if self.fail_with is not None:
            stderr.write("fake mapper failed\n")
            return subprocess.CompletedProcess(command, self.fail_with)
        query = ad.read_h5ad(str(option("--query_path")))
        precompute = Path(str(option("--precomputed_stats.path")))
        bundle_dir = precompute.parent
        tree = json.loads((bundle_dir / "mapping_tree.json").read_text())
        markers = json.loads((bundle_dir / "fake_markers.json").read_text())
        lookup = json.loads(
            Path(str(option("--query_markers.serialized_lookup"))).read_text()
        )
        levels = list(tree["hierarchy"])
        drop_level = option("--drop_level")
        counts = np.asarray(query.X.toarray())
        genes = list(query.var_names)
        usable = [
            (label, marker)
            for label, marker in markers.items()
            if marker in genes and marker in lookup.get("None", [])
        ]
        results = []
        for row, cell_id in enumerate(query.obs_names):
            values = {
                label: float(counts[row, genes.index(marker)])
                for label, marker in usable
            }
            total = sum(values.values())
            ranked = sorted(values, key=lambda label: (-values[label], label))
            probability = {
                label: (round(values[label] / total, 2) if total else 0.0)
                for label in ranked
            }
            best = ranked[0]
            result: dict[str, Any] = {"cell_id": str(cell_id)}
            for depth, level in enumerate(levels):
                suffix = "" if depth == 0 else f"_{depth}"
                runners = [label for label in ranked[1:] if probability[label] > 0]
                result[level] = {
                    "assignment": best + suffix,
                    "bootstrapping_probability": probability[best],
                    "aggregate_probability": probability[best],
                    "avg_correlation": 0.5,
                    "runner_up_assignment": [label + suffix for label in runners],
                    "runner_up_correlation": [0.1 for _ in runners],
                    "runner_up_probability": [probability[label] for label in runners],
                    "directly_assigned": level != drop_level,
                }
            results.append(result)
        payload = {
            "taxonomy_tree": {
                **tree,
                "hierarchy_mapper": {
                    level: FAKE_LEVEL_NAMES.get(level, level) for level in levels
                },
            },
            "results": results,
            "config": {
                "type_assignment": {
                    "rng_seed": (
                        self.record_seed
                        if self.record_seed is not None
                        else int(str(option("--type_assignment.rng_seed")))
                    ),
                    "bootstrap_factor": float(
                        str(option("--type_assignment.bootstrap_factor"))
                    ),
                    "bootstrap_iteration": int(
                        str(option("--type_assignment.bootstrap_iteration"))
                    ),
                    "n_processors": int(str(option("--type_assignment.n_processors"))),
                    "n_runners_up": int(str(option("--type_assignment.n_runners_up"))),
                    "normalization": option("--type_assignment.normalization"),
                    "chunk_size": 10000,
                },
                "drop_level": drop_level,
            },
            "metadata": {"version": "1.7.2"},
            "n_unmapped_genes": 0,
        }
        Path(str(option("--extended_result_path"))).write_text(json.dumps(payload))
        Path(str(option("--log_path"))).write_text("fake log\n")
        stdout.write("fake mapper ok\n")
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def fake_mmc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeMmc:
    """A fake mapper installed for ctm 1.7.2, with a bundle root."""
    fake = FakeMmc(root=tmp_path / "store")
    fake.root.mkdir()
    fake.install(monkeypatch, "1.7.2")
    return fake
