"""Shared fixtures for the annotation contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
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
    marker gene has the most counts; its bootstrap probability ``p`` is that
    marker's share of the marker counts (rounded to 0.01), the other nodes
    are the runner-ups, and every deeper level repeats the node's single
    child (named ``"<name> <depth>"``) with bootstrap probability
    ``p ** (depth + 1)`` and the product of those as its aggregate
    probability. ``--nodes_to_drop`` removes a first-level node (given at any
    level of its chain) from the candidates, as ctm drops it from the tree;
    the recorded config lists the dropped nodes.
    """

    root: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail_with: int | None = None
    record_seed: int | None = None
    record_drop_level: str | None = None
    record_nodes_to_drop: list[list[str]] | None = None

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
        monkeypatch.setattr(mapmycells_engine, "_launch_mapper", self.run)

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool,
        stdout: TextIO,
        stderr: TextIO,
        env: Mapping[str, str],
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        """The fake ``mapmycells_engine._launch_mapper``."""
        import anndata as ad

        assert check is False and text is True
        command = list(command)

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
        self.calls[-1]["lookup"] = lookup
        levels = list(tree["hierarchy"])
        drop_level = option("--drop_level")
        nodes_text = option("--nodes_to_drop")
        nodes_to_drop = json.loads(nodes_text) if nodes_text is not None else None
        dropped = {
            label
            for label in markers
            for level, node in nodes_to_drop or []
            if node == label or str(node).startswith(f"{label}_")
        }
        self.calls[-1]["nodes_to_drop"] = nodes_to_drop
        counts = np.asarray(query.X.toarray())
        genes = list(query.var_names)
        usable = [
            (label, marker)
            for label, marker in markers.items()
            if marker in genes
            and marker in lookup.get("None", [])
            and label not in dropped
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
            aggregate = 1.0
            for depth, level in enumerate(levels):
                suffix = "" if depth == 0 else f"_{depth}"
                runners = [label for label in ranked[1:] if probability[label] > 0]
                # Deeper levels are less certain (bp ** (depth + 1)) and the
                # aggregate probability is the product down the hierarchy, as
                # ctm's, so bp and aggregate_probability differ below the root.
                bp = round(probability[best] ** (depth + 1), 2)
                aggregate = round(aggregate * bp, 4)
                result[level] = {
                    "assignment": best + suffix,
                    "bootstrapping_probability": bp,
                    "aggregate_probability": aggregate,
                    "avg_correlation": 0.5,
                    "runner_up_assignment": [label + suffix for label in runners],
                    "runner_up_correlation": [0.1 for _ in runners],
                    "runner_up_probability": [
                        round(probability[label] ** (depth + 1), 2) for label in runners
                    ],
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
                "drop_level": (
                    self.record_drop_level
                    if self.record_drop_level is not None
                    else drop_level
                ),
                "nodes_to_drop": (
                    self.record_nodes_to_drop
                    if self.record_nodes_to_drop is not None
                    else nodes_to_drop
                ),
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


# --------------------------------------------------------------------------
# RESOLVE: trust decisions and resolvability decision tables (M4 tests)

HUMAN_TABLE_LEVELS: tuple[tuple[str, str, float, float, str | None], ...] = (
    ("lineage", "lineage", 0.73, 0.90, None),
    ("broad", "broad", 0.73, 0.90, "broad"),
    ("nt", "nt", 0.73, 0.90, "broad"),
    ("supercluster", "leaf", 0.69, 0.85, "supercluster"),
    ("cluster", "fine", 0.69, 0.85, "supercluster"),
)
HUMAN_GRID: tuple[int, ...] = (10, 15, 30, 60, 120, 250)
HUMAN_TABLE_CLASSES: tuple[str, ...] = (
    "Exc",
    "Inh",
    "OtherNeuron",
    "Astro",
    "Oligo",
    "OPC",
    "COP",
    "Immune",
    "Vascular",
    "Fibroblast",
)


@pytest.fixture
def make_trust() -> Callable[..., Any]:
    """Return a factory of primary / secondary ``TrustDecision`` objects.

    ``state`` is ``validated_real`` (set a, real data, up to supercluster),
    ``validated_simulation`` (with ``level_records``), ``provisional``,
    ``broad_only`` or ``refused``.
    """
    from merxen.annotation.diagnostics import TrustDecision, ValidatedLevelRecord

    def factory(
        state: str = "validated_real",
        *,
        role: str = "primary",
        species: str = "human",
        level_records: Sequence[Mapping[str, Any]] = (),
        validated_max_level: str | None = None,
    ) -> TrustDecision:
        common: dict[str, Any] = {
            "reference_id": "whb_frontal_supc_clus" if role == "primary" else "seaad",
            "role": role,
            "species": species,
            "panel_hash": "a" * 64,
            "family_id": "human_set_a" if species == "human" else "mouse_ag7",
            "family_basis": "listed",
        }
        leaf = "supercluster" if species == "human" else "subclass"
        if state == "validated_real":
            return TrustDecision(
                **common,
                state="validated",
                validation_basis="real_data",
                validated_max_level=validated_max_level or leaf,
            )
        if state == "validated_simulation":
            records = tuple(
                ValidatedLevelRecord(
                    family_id=common["family_id"],
                    panel_hash="a" * 64,
                    in_class_set=True,
                    **dict(record),
                )
                for record in level_records
            )
            return TrustDecision(
                **{**common, "family_id": "sim_family"},
                state="validated",
                validation_basis="simulation",
                validated_max_level=validated_max_level or "broad",
                level_records=records,
            )
        if state in ("provisional", "broad_only", "refused"):
            return TrustDecision(
                **{**common, "family_id": None, "family_basis": None}, state=state
            )
        raise ValueError(state)

    return factory


def decision_rows(
    *,
    regimes: Sequence[str] = ("validated", "provisional"),
    levels: Sequence[tuple[str, str, float, float, str | None]] = HUMAN_TABLE_LEVELS,
    classes: Sequence[str] = HUMAN_TABLE_CLASSES,
    grid: Sequence[int] = HUMAN_GRID,
    overrides: Mapping[tuple[str, str, str, int], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return a ``resolvability.decide``-like table: everything emitted.

    ``overrides`` maps ``(regime, level, class, depth)`` to column values
    (e.g. ``{"status": "not_resolvable", "reason": "..."}`` or a raised
    ``threshold``). The provisional threshold defaults to the level default.
    """
    records = []
    for regime in regimes:
        for level, _role, default, _target, _floor in levels:
            for cls in classes:
                for depth in grid:
                    record: dict[str, Any] = {
                        "regime": regime,
                        "level": level,
                        "class": cls,
                        "depth": depth,
                        "status": "emitted",
                        "threshold": default,
                        "t_star": default,
                        "extrapolated": False,
                        "reason": None,
                    }
                    record.update(
                        (overrides or {}).get((regime, level, cls, depth), {})
                    )
                    records.append(record)
    return pd.DataFrame.from_records(records)


@pytest.fixture
def human_level_meta() -> list[Any]:
    """Return the ``LevelMeta`` of the synthetic human decision tables."""
    from merxen.annotation.resolvability import LevelMeta

    return [
        LevelMeta(level, "SUPC", role, default, target, floor)  # type: ignore[arg-type]
        for level, role, default, target, floor in HUMAN_TABLE_LEVELS
    ]


@pytest.fixture
def make_decisions() -> Callable[..., pd.DataFrame]:
    """Return ``decision_rows``: a synthetic decisions table factory."""
    return decision_rows


# --------------------------------------------------------------------------
# Promotion by simulation (M13 C10): trust decided by ``trust_state``

# The bundle's resolvability constraint before and after a gate-P PR:
# ``resolvable`` (a self-map without a constraint), ``no_self_map``, or the
# self-map's ``broad_only`` / ``refused`` verdict.
PROMOTION_CONSTRAINTS: tuple[str, ...] = (
    "resolvable",
    "no_self_map",
    "broad_only",
    "refused",
)
PROMOTION_REFERENCE: dict[str, str] = {
    "human": "whb_frontal_supc_clus",
    "mouse": "wmb_panel",
}
# (level, class, status, validated_min_depth, in_class_set) of the
# simulation family's ``validated_panel_levels.csv``; classes are RESOLVE's
# class keys (human E2 floor classes, mouse WMB classes).
PROMOTION_LEVEL_RECORDS: dict[
    str, tuple[tuple[str, str, str, int | None, bool], ...]
] = {
    "human": (
        ("lineage", "Exc", "validated", 10, True),
        ("lineage", "Astro", "validated", 30, True),
        ("broad", "Exc", "validated", 15, True),
        ("broad", "Astro", "validated", 60, True),
        ("broad", "Oligo", "validated", 30, True),
        ("broad", "Immune", "not_evaluable", None, False),
        ("nt", "Exc", "validated", 60, True),
        ("supercluster", "Exc", "validated", 120, True),
        ("supercluster", "Astro", "failed:NP4", None, True),
    ),
    "mouse": (
        ("broad", "01 IT-ET Glut", "validated", 20, True),
        ("broad", "30 Astro-Epen", "validated", 30, True),
        ("class", "01 IT-ET Glut", "validated", 30, True),
        ("class", "30 Astro-Epen", "validated", 60, True),
        ("class", "24 MY Glut", "not_evaluable", None, False),
        ("subclass", "01 IT-ET Glut", "validated", 120, True),
        ("subclass", "19 MB Glut", "failed:NP3", None, True),
    ),
}
PROMOTION_MAX_LEVEL: dict[str, str] = {"human": "broad", "mouse": "class"}


def promotion_gene_ids(species: str, n: int = 100) -> tuple[str, ...]:
    """Return the synthetic panel of the promotion tests."""
    prefix = "ENSG" if species == "human" else "ENSMUSG"
    return tuple(f"{prefix}{index:011d}" for index in range(9000, 9000 + n))


def promotion_table(
    species: str, gene_ids: Sequence[str], *, platform: str = "MERSCOPE"
) -> Any:
    """Return the validated tables after a gate-P PR promoted the panel.

    One ``validated_panels.csv`` row with ``validation_basis = simulation``
    for the panel (validated up to broad / class) and its
    ``PROMOTION_LEVEL_RECORDS``: validated classes with their
    ``validated_min_depth``, a failed and an unevaluable class.
    """
    from merxen.annotation.diagnostics import (
        ValidatedLevelRecord,
        ValidatedPanelRecord,
        ValidatedPanelTable,
    )
    from merxen.annotation.panel import compute_panel_hash

    panel_hash = compute_panel_hash(sorted(gene_ids))
    family_id = f"{species}_sim_family"
    record = ValidatedPanelRecord(
        panel_id=f"{species}_sim_panel",
        family_id=family_id,
        panel_hash=panel_hash,
        panel_role="sample_panel",
        species=species,  # type: ignore[arg-type]
        platforms=(platform,),
        n_genes=len(set(gene_ids)),
        validated_max_level=PROMOTION_MAX_LEVEL[species],
        validation_basis="simulation",
        evidence="gate_p/test",
        date="2026-10-06",
        approving_pr="#0",
    )
    levels = tuple(
        ValidatedLevelRecord.model_validate(
            {
                "family_id": family_id,
                "panel_hash": panel_hash,
                "level": level,
                "class": cls,
                "in_class_set": in_class_set,
                "status": status,
                "validated_min_depth": min_depth,
                "tested_max_depth": 120 if status == "validated" else None,
                "evidence": "gate_p/test",
            }
        )
        for level, cls, status, min_depth, in_class_set in PROMOTION_LEVEL_RECORDS[
            species
        ]
    )
    return ValidatedPanelTable(records=(record,), levels=levels, source="test")


@pytest.fixture
def promotion_trust() -> Callable[..., tuple[Any, Any]]:
    """Return a factory of trust decisions before and after a promotion.

    ``factory(constraint, *, species="human", role="primary", gene_ids=None,
    rules=None, platform="MERSCOPE")`` returns ``(before, after)``:
    ``diagnostics.trust_state`` of
    one panel and bundle (coverage passing every refusal rule, the
    resolvability ``constraint`` of ``PROMOTION_CONSTRAINTS``), decided
    against no validated family (``before``: the panel's own family) and
    against ``promotion_table`` (``after``: the family listed with
    ``validation_basis = simulation``). Only the validated tables differ.
    """
    from merxen.annotation.diagnostics import (
        CoverageDiagnostics,
        ResolvabilityTrust,
        TrustDecision,
        TrustRules,
        ValidatedPanelTable,
        trust_state,
    )
    from merxen.annotation.panel import compute_panel_hash, panel_family

    def factory(
        constraint: str,
        *,
        species: str = "human",
        role: str = "primary",
        gene_ids: Sequence[str] | None = None,
        rules: TrustRules | None = None,
        platform: str = "MERSCOPE",
    ) -> tuple[TrustDecision, TrustDecision]:
        if constraint not in PROMOTION_CONSTRAINTS:
            raise ValueError(constraint)
        panel = tuple(gene_ids) if gene_ids is not None else promotion_gene_ids(species)
        reference_id = (
            PROMOTION_REFERENCE[species] if role == "primary" else "seaad_mr_panel"
        )
        resolvability = (
            None
            if constraint == "no_self_map"
            else ResolvabilityTrust(
                state=None if constraint == "resolvable" else constraint,  # type: ignore[arg-type]
                reasons=() if constraint == "resolvable" else (f"{constraint}: x",),
            )
        )
        coverage = CoverageDiagnostics(
            reference_id=reference_id,
            n_panel_genes=len(panel),
            n_query_genes_used=max(len(panel), 100),
            root_markers=50,
        )
        decisions = []
        for table in (
            ValidatedPanelTable(records=()),
            promotion_table(species, panel, platform=platform),
        ):
            decisions.append(
                trust_state(
                    reference_id=reference_id,
                    role=role,  # type: ignore[arg-type]
                    species=species,  # type: ignore[arg-type]
                    panel_hash=compute_panel_hash(sorted(panel)),
                    n_panel_genes=len(panel),
                    family=panel_family(
                        panel,
                        species=species,
                        platforms=[platform],
                        known_families=table.known_families(),
                    ),
                    validated=table,
                    rules=rules,
                    coverage=coverage,
                    resolvability=resolvability,
                )
            )
        before, after = decisions
        assert before.family_basis == "own" and after.family_basis == "listed"
        return before, after

    return factory


LEVEL_RESULT_FIELDS: tuple[str, ...] = (
    "name",
    "raw",
    "conf",
    "corr",
    "runner_up",
    "margin",
    "status",
    "threshold",
    "floor",
    "class_key",
    "extrapolated",
)
EMISSION_FIELDS: tuple[str, ...] = (
    "emitted",
    "threshold",
    "default_threshold",
    "local_threshold",
    "extrapolated",
    "depth_bin",
    "reason",
)


def _same_values(first: Any, second: Any) -> bool:
    return bool(pd.Series(np.asarray(first)).equals(pd.Series(np.asarray(second))))


def assert_same_emission(before: Any, after: Any) -> None:
    """Assert two RESOLVE results emit the same (human or mouse; plan §8.2, §14).

    Names, scores, statuses, thresholds, floors, class keys, the final
    label, the tier, the flags, the gate level and its reasons, the depth
    bins and every level's emission must be identical. Only
    ``ct_<L>_validated``, the gate's warning flag and reasons, its trust
    state and the floor warnings may differ, and ``ct_<L>_validated`` only
    ever marks confident labels.

    Args:
        before: ``HumanResolution`` or ``MouseResolution`` before promotion.
        after: The same after promotion.
    """
    assert list(before.levels) == list(after.levels)
    for level, first in before.levels.items():
        second = after.levels[level]
        for name in LEVEL_RESULT_FIELDS:
            assert _same_values(getattr(first, name), getattr(second, name)), (
                level,
                name,
            )
        for result in (first, second):
            assert not (result.validated & ~result.confident).any(), level
    for name in (
        "final_level",
        "final_name",
        "consensus_tier",
        "depth_bin",
        "resolvability_extrapolated",
        "in_table",
    ):
        assert _same_values(getattr(before, name), getattr(after, name)), name
    assert list(before.flags) == list(after.flags)
    for name, values in before.flags.items():
        assert _same_values(values, after.flags[name]), name
    assert before.gate.level == after.gate.level
    assert before.gate.level_reasons == after.gate.level_reasons
    assert before.gate.attempts_leaf == after.gate.attempts_leaf
    assert list(before.emissions) == list(after.emissions)
    for level, first_emission in before.emissions.items():
        second_emission = after.emissions[level]
        assert first_emission.regime == second_emission.regime, level
        assert first_emission.threshold_source == second_emission.threshold_source
        for name in EMISSION_FIELDS:
            assert _same_values(
                getattr(first_emission, name), getattr(second_emission, name)
            ), (level, name)
    validated_columns = {
        name for name in before.to_columns() if name.endswith("_validated")
    }
    first_columns, second_columns = before.to_columns(), after.to_columns()
    assert list(first_columns) == list(second_columns)
    for name, values in first_columns.items():
        if name not in validated_columns:
            assert _same_values(values, second_columns[name]), name


# --------------------------------------------------------------------------
# A mouse section for the region step and mouse RESOLVE (M6 tests)

MOUSE_LEVELS: tuple[str, ...] = (
    "CCN20230722_CLAS",
    "CCN20230722_SUBC",
    "CCN20230722_SUPT",
    "CCN20230722_CLUS",
)
MOUSE_CLAS, MOUSE_SUBC = MOUSE_LEVELS[0], MOUSE_LEVELS[1]
MOUSE_IDS: tuple[str, ...] = tuple(f"ENSMUSG{index:011d}" for index in range(1, 6))
MOUSE_SYMBOLS: tuple[str, ...] = ("Slc17a7", "Slc17a6", "Hoxb5", "Aqp4", "Other")
MOUSE_NODES: tuple[FakeNode, ...] = (
    FakeNode("CL_01", "01 IT-ET Glut", MOUSE_IDS[0], {"broad_class": "Neurons"}),
    FakeNode("CL_19", "19 MB Glut", MOUSE_IDS[1], {"broad_class": "Neurons"}),
    FakeNode("CL_24", "24 MY Glut", MOUSE_IDS[2], {"broad_class": "Neurons"}),
    FakeNode(
        "CL_30", "30 Astro-Epen", MOUSE_IDS[3], {"broad_class": "Astrocytes/Ependymal"}
    ),
)
# MERFISH grey cells per division of each class; the fake tree gives every
# class one subclass, "<class> 1".
MOUSE_MERFISH: dict[str, dict[str, int]] = {
    "01 IT-ET Glut": {"Isocortex": 1000},
    "19 MB Glut": {"MB": 900, "MY": 100},
    "24 MY Glut": {"MY": 1000},
    "30 Astro-Epen": {"MY": 500, "Isocortex": 500},
}
MOUSE_SID = "AG_MERSCOPE"
MOUSE_TILE = 150.0
MOUSE_REGION_SHARE_HASH = "e" * 64


def mouse_panel() -> Any:
    """Return the fake mouse section's single-sample panel."""
    from merxen.annotation.panel import AnnotationPanel, compute_panel_hash

    return AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="mouse",
        platforms=["MERSCOPE"],
        sample_ids=[MOUSE_SID],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(list(MOUSE_IDS)),
        n_genes=len(MOUSE_IDS),
        ensembl_ids=list(MOUSE_IDS),
        symbols=list(MOUSE_SYMBOLS),
    )


def mouse_required(panel: Any) -> Any:
    """Return PREP's required bundles for the fake mouse panel."""
    from merxen.annotation.panel import (
        PANEL_GENES_FILE,
        RequiredBundle,
        RequiredBundles,
    )

    return RequiredBundles(
        pair_id="AG",
        segmentation="proseg_hybrid",
        species="mouse",
        panel_mode="single_sample",
        status="ok",
        bundles=[
            RequiredBundle(
                reference_id="wmb_panel",
                role="primary",
                species="mouse",
                purpose="annotation",
                panel_name="sample",
                panel_hash=panel.panel_hash,
                panel_file=PANEL_GENES_FILE,
                n_panel_genes=panel.n_genes,
            ),
            RequiredBundle(
                reference_id="wmb_region_share",
                role="region_share",
                species="mouse",
                purpose="panel_independent",
            ),
        ],
        n_required=2,
    )


def mouse_region_share_bundle(
    root: Path,
    *,
    build_hash: str = MOUSE_REGION_SHARE_HASH,
    composition: pd.DataFrame | None = None,
) -> Path:
    """Write a fake ``wmb_region_share`` bundle of ``MOUSE_MERFISH``.

    Args:
        root: The store root.
        build_hash: The bundle's build hash (its directory name).
        composition: An optional ``section_composition.parquet`` table (G4).

    Returns:
        The bundle directory.
    """
    from merxen.annotation.config import MOUSE_CCF_REGIONS

    share_rows, home_rows = [], []
    for class_name, counts in MOUSE_MERFISH.items():
        total = sum(counts.values())
        for level, name in (("class", class_name), ("subclass", f"{class_name} 1")):
            for region in MOUSE_CCF_REGIONS:
                share_rows.append(
                    {
                        "level": level,
                        "node_name": name,
                        "region": region,
                        "n": counts.get(region, 0),
                        "share": counts.get(region, 0) / total,
                        "n_grey": total,
                        "n_all": total,
                    }
                )
        home = max(counts, key=lambda region: counts[region])
        home_rows.append(
            {
                "level": "subclass",
                "node_name": f"{class_name} 1",
                "home": home,
                "home_share": counts[home] / total,
                "n_grey": total,
            }
        )
    bundle_dir = root / "wmb_region_share" / build_hash
    bundle_dir.mkdir(parents=True)
    pd.DataFrame(share_rows).to_parquet(bundle_dir / "region_share.parquet")
    pd.DataFrame(home_rows).to_parquet(bundle_dir / "region_home.parquet")
    if composition is not None:
        composition.to_parquet(bundle_dir / "section_composition.parquet")
    files = [
        {"path": path.name, "size": path.stat().st_size}
        for path in sorted(bundle_dir.iterdir())
    ]
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "complete",
                "build_hash": build_hash,
                "reference_id": "wmb_region_share",
                "species": "mouse",
                "role": "region_share",
                "builder_version": 3,
                "panel": None,
                "files": files,
            }
        )
    )
    return bundle_dir


def mouse_section() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return counts, µm coordinates and kinds of the fake mouse section.

    200 tiles: Isocortex (01) left, MB (19) right, 5 MY sink calls in MB.
    The MY cells carry some MB marker counts, so without class 24 they map
    to 19 MB Glut; their own bootstrap probability (.71) is too low for them
    to vote. One more MY call in MB (``my_lost``) splits evenly between the
    IT-ET and MB markers once class 24 is dropped: a confident neuron that
    no class re-maps confidently (implausible). Two astrocytes (not
    neurons) never vote. The below-min_counts object comes first, so a
    selection of table rows by position rather than by cell id shifts every
    table cell's coordinates.
    """
    counts: list[list[int]] = [[1, 0, 0, 0, 0, 0]]  # below min_counts
    xy: list[tuple[float, float]] = [(0.5 * MOUSE_TILE, 0.5 * MOUSE_TILE)]
    kinds = ["low"]
    for i in range(20):
        for j in range(10):
            marker = 0 if i < 10 else 1
            for k in range(3):
                row = [0] * 6
                row[marker] = 30
                counts.append(row)
                xy.append(((i + 0.2 + 0.3 * k) * MOUSE_TILE, (j + 0.5) * MOUSE_TILE))
                kinds.append("ctx" if marker == 0 else "mb")
    for index in range(5):
        counts.append([0, 8, 20, 0, 0, 0])
        xy.append(((12 + index + 0.5) * MOUSE_TILE, 3.5 * MOUSE_TILE))
        kinds.append("my")
    counts.append([3, 3, 20, 0, 0, 0])
    xy.append((15.5 * MOUSE_TILE, 6.5 * MOUSE_TILE))
    kinds.append("my_lost")
    for index in range(2):
        counts.append([0, 0, 0, 60, 0, 0])  # above the subclass floor (50)
        xy.append(((2 + index + 0.5) * MOUSE_TILE, 2.5 * MOUSE_TILE))
        kinds.append("astro")
    return np.array(counts), np.array(xy, dtype=np.float64), kinds


def write_mouse_h5ad(path: Path, *, spatial: bool = True) -> tuple[Path, list[str]]:
    """Write the fake mouse section as a prepared h5ad.

    Args:
        path: The h5ad path.
        spatial: Write ``obsm['spatial']``.

    Returns:
        The path and the object kinds (``mouse_section``).
    """
    import anndata as ad
    from scipy import sparse

    counts, xy, kinds = mouse_section()
    var_names = [*MOUSE_SYMBOLS, "Blank-0001"]
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    var["gene"] = var_names
    var["ensembl_id"] = [*MOUSE_IDS, "Blank-0001"]
    obs = pd.DataFrame(index=[f"M{index}" for index in range(len(counts))])
    adata = ad.AnnData(X=sparse.csr_matrix(counts.astype(np.int64)), obs=obs, var=var)
    if spatial:
        adata.obsm["spatial"] = xy
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path, kinds


@pytest.fixture
def mouse_setup(tmp_path: Path, fake_mmc: FakeMmc) -> dict[str, Any]:
    """The fake mouse section, its panel, WMB bundle and region-share bundle."""
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mapmycells_engine import MmcBundle
    from merxen.annotation.mouse_regions import RegionShareBundle
    from merxen.annotation.panel import PANEL_GENES_FILE, REQUIRED_BUNDLES_FILE
    from merxen.annotation.pipeline import MapSample, map_bundles

    panel = mouse_panel()
    panel_dir = tmp_path / "panel"
    panel.write(panel_dir / PANEL_GENES_FILE)
    required = mouse_required(panel)
    (panel_dir / REQUIRED_BUNDLES_FILE).write_text(
        json.dumps(required.model_dump(mode="json"))
    )
    bundle_dir = fake_mmc.bundle(
        "wmb_panel",
        role="primary",
        species="mouse",
        panel_hash=panel.panel_hash,
        n_genes=panel.n_genes,
        levels=list(MOUSE_LEVELS),
        nodes=list(MOUSE_NODES),
        drop_level="CCN20230722_SUPT",
    )
    bundle = MmcBundle.from_dir(bundle_dir)
    config = AnnotationConfig(species="mouse").coupled_to_clustering(10)
    runs = map_bundles(
        required, panel_dir, {("wmb_panel", panel.panel_hash): bundle}, config
    )
    h5ad, kinds = write_mouse_h5ad(
        tmp_path / "prepared" / "merscope" / f"{MOUSE_SID}.h5ad"
    )
    region_dir = mouse_region_share_bundle(tmp_path / "store")
    return {
        "panel": panel,
        "panel_dir": panel_dir,
        "bundle": bundle,
        "config": config,
        "runs": runs,
        "sample": MapSample(MOUSE_SID, "MERSCOPE", h5ad, "prepared"),
        "kinds": kinds,
        "region_dir": region_dir,
        "regions": RegionShareBundle.from_dir(region_dir),
    }


def map_mouse(setup: Mapping[str, Any], output: Path, **kwargs: Any) -> Any:
    """Run MAP on the ``mouse_setup`` section (keyword arguments override)."""
    from merxen.annotation.pipeline import annotate_map

    return annotate_map(
        [kwargs.pop("sample", setup["sample"])],
        setup["runs"],
        kwargs.pop("config", setup["config"]),
        output_dir=output,
        pair_id="AG",
        segmentation="proseg_hybrid",
        region_shares=kwargs.pop("region_shares", setup["regions"]),
        **kwargs,
    )
