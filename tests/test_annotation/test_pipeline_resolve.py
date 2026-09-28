"""Tests for the RESOLVE step and ``merxen annotate-resolve`` (plan §3.4, M4).

The MAP outputs are synthetic: prepared H5ADs, tidy MMC parquets written
directly (real WHB supercluster and SEA-AD subclass names, so the vocab rules
apply) and a ``map_manifest.json`` whose sample fingerprints come from the
same loader RESOLVE uses. Bundles are the fake MAP bundles
(``conftest.FakeMmc.bundle``) with negative genes and profiles added.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import sparse

from merxen.annotation import pipeline as pl
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mapmycells_engine import (
    TIDY_COLUMNS,
    MmcBundle,
    write_tidy_parquet,
)
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    MapManifest,
    MapRunRecord,
    MapSample,
    MapSampleRecord,
    ResolveError,
    annotate_resolve,
    load_samples,
    read_label_table,
)
from merxen.annotation.provenance import (
    PROVENANCE_UNS_KEY,
    AnnotationProvenance,
    unsafe_structure_problems,
)
from merxen.annotation.schema import (
    CellStatus,
    Columns,
    soft_columns,
    validate_label_table,
)
from merxen.annotation.store import file_sha256
from merxen.cli import main as cli_main

from .conftest import FakeMmc
from .test_pipeline import GENE_IDS, SYMBOLS, _human_bundles, _panel_dir

MakeTrust = Callable[..., Any]

# Fake WHB labels (conftest nodes of test_pipeline) and their real names.
WHB = {
    "CS_EXC": "Upper-layer intratelencephalic",
    "CS_AST": "Astrocyte",
    "CS_OLI": "Oligodendrocyte",
    "CS_SPL": "Splatter",
    "CS_HIP": "Hippocampal CA1-3",
}
SEA_OF = {
    "CS_EXC": ("SC_L23", "L2/3 IT"),
    "CS_AST": ("SC_AST", "Astrocyte"),
    "CS_OLI": ("SC_OLI", "Oligodendrocyte"),
    "CS_SPL": ("SC_L23", "L2/3 IT"),
    "CS_HIP": ("SC_L23", "L2/3 IT"),
}
LABELS = list(WHB)
WEIGHTS = np.array([0.45, 0.25, 0.2, 0.05, 0.05])
MARKER = {label: index for index, label in enumerate(LABELS)}
WHB_LEVELS = ("CCN202210140_SUPC", "CCN202210140_CLUS")
SEA_LEVELS = ("SEA_LEVEL_0", "SEA_LEVEL_1", "SEA_LEVEL_2")
PLATFORM_CELLS = {"MERSCOPE": 260, "XENIUM": 220}


@dataclass
class Setup:
    """A synthetic MAP output of one pair."""

    root: Path
    map_dir: Path
    panel_dir: Path
    panel: AnnotationPanel
    bundles: dict[str, MmcBundle]
    samples: list[MapSample]
    calls: dict[str, pd.DataFrame]
    config: AnnotationConfig


def _config(**updates: Any) -> AnnotationConfig:
    return AnnotationConfig(species="human", **updates).coupled_to_clustering(10)


def _cells(platform: str, seed: int) -> pd.DataFrame:
    """Per-object calls and counts of one synthetic section."""
    rng = np.random.default_rng(seed)
    n = PLATFORM_CELLS[platform]
    labels = rng.choice(LABELS, size=n, p=WEIGHTS)
    depth = rng.integers(3, 400, n)
    bp = np.round(rng.uniform(0.55, 1.0, n), 2)
    return pd.DataFrame(
        {
            "cell_id": [f"{platform[0]}{index}" for index in range(n)],
            "label": labels,
            "depth": depth,
            "bp": bp,
            "corr": rng.normal(0.6, 0.05, n),
            "sea_bp": np.where(rng.uniform(size=n) < 0.9, 0.95, 0.4),
            "x": rng.uniform(0, 3000, n),
            "y": rng.uniform(0, 3000, n),
        }
    )


def _counts(cells: pd.DataFrame, seed: int) -> np.ndarray:
    """Counts on the six panel genes plus a control: mostly the marker."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((len(cells), 7), dtype=np.int64)
    for row, (label, depth) in enumerate(
        zip(cells["label"], cells["depth"], strict=True)
    ):
        profile = np.full(6, 0.04)
        profile[MARKER[label]] = 0.8
        profile /= profile.sum()
        matrix[row, :6] = rng.multinomial(int(depth), profile)
    matrix[:, 6] = 1  # a control probe, removed before counting
    return matrix


def _write_prepared(
    path: Path, cells: pd.DataFrame, counts: np.ndarray, platform: str
) -> Path:
    control = "Blank-0001" if platform == "MERSCOPE" else "NegControlProbe_00001"
    var = pd.DataFrame(index=pd.Index([*SYMBOLS, control], dtype=str))
    var["gene"] = list(var.index)
    if platform == "XENIUM":
        var["ensembl_id"] = [*GENE_IDS, control]
    obs = pd.DataFrame(index=pd.Index(cells["cell_id"].astype(str)))
    obs["instance_id"] = np.arange(len(cells), dtype=np.int64) + 1000
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.obsm["spatial"] = cells[["x", "y"]].to_numpy(np.float64)
    adata.uns["spatialdata_attrs"] = {
        "instance_key": "instance_id",
        "region": "cells",
        "region_key": "region",
    }
    adata.uns["merxen_clustering_squidpy"] = {
        "platform": platform,
        "shape_key": (
            "MOSAIK_proseg_hybrid_aligned_nonrigid"
            if platform == "MERSCOPE"
            else "MOSAIK_proseg_hybrid"
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def _runner_ups(
    row: dict[str, Any], alternatives: list[tuple[str, str, float]]
) -> None:
    for rank in range(1, 6):
        if rank <= len(alternatives):
            label, name, probability = alternatives[rank - 1]
            row[f"runner_up_{rank}_assignment"] = label
            row[f"runner_up_{rank}_name"] = name
            row[f"runner_up_{rank}_probability"] = probability
            row[f"runner_up_{rank}_correlation"] = 0.1
        else:
            row[f"runner_up_{rank}_assignment"] = None
            row[f"runner_up_{rank}_name"] = None
            row[f"runner_up_{rank}_probability"] = np.nan
            row[f"runner_up_{rank}_correlation"] = np.nan


def _tidy_row(
    cell_id: str,
    level: str,
    level_name: str,
    label: str,
    name: str,
    bp: float,
    corr: float,
    alternatives: list[tuple[str, str, float]],
    aggregate: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cell_id": cell_id,
        "level": level,
        "level_name": level_name,
        "assignment": label,
        "name": name,
        "bp": bp,
        "aggregate_probability": bp if aggregate is None else aggregate,
        "avg_correlation": corr,
        "directly_assigned": True,
        "n_runners_up": len(alternatives),
    }
    _runner_ups(row, alternatives)
    return row


def _whb_tidy(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in cells.itertuples(index=False):
        label = str(record.label)
        other = next(item for item in LABELS if item != label)
        rest = round(1.0 - float(record.bp), 2)
        rows.append(
            _tidy_row(
                record.cell_id,
                WHB_LEVELS[0],
                "supercluster",
                label,
                WHB[label],
                float(record.bp),
                float(record.corr),
                [(other, WHB[other], rest)] if rest > 0 else [],
            )
        )
        rows.append(
            _tidy_row(
                record.cell_id,
                WHB_LEVELS[1],
                "cluster",
                f"{label}_1",
                f"{WHB[label]} 1",
                round(float(record.bp) ** 2, 4),
                float(record.corr),
                [],
            )
        )
    return pd.DataFrame(rows, columns=list(TIDY_COLUMNS))


def _sea_tidy(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in cells.itertuples(index=False):
        label, name = SEA_OF[str(record.label)]
        bp = float(record.sea_bp)
        rows.append(
            _tidy_row(
                record.cell_id, SEA_LEVELS[0], "Class", "CL", "Class", bp, 0.5, []
            )
        )
        rows.append(
            _tidy_row(
                record.cell_id,
                SEA_LEVELS[1],
                "Subclass",
                label,
                name,
                bp,
                0.5,
                [],
                aggregate=bp * bp,
            )
        )
        rows.append(
            _tidy_row(
                record.cell_id,
                SEA_LEVELS[2],
                "Supertype",
                f"{label}_1",
                f"{name}_1",
                bp,
                0.5,
                [],
            )
        )
    return pd.DataFrame(rows, columns=list(TIDY_COLUMNS))


def _bundle_tables(bundle: MmcBundle) -> None:
    """Add negative genes and supercluster profiles to a fake WHB bundle."""
    classes = ["Neurons", "Astrocytes", "Oligodendrocytes"]
    negative_gene = {"Neurons": 2, "Astrocytes": 0, "Oligodendrocytes": 0}
    records = []
    for cls in classes:
        for index, gene in enumerate(GENE_IDS):
            is_negative = index == negative_gene[cls]
            records.append(
                {
                    "broad_class": cls,
                    "gene_id": gene,
                    "detection_seaad_mr": 0.0 if is_negative else 0.5,
                    "n_cells_seaad_mr": 100.0,
                    "detection_whb_frontal": 0.0 if is_negative else 0.5,
                    "n_cells_whb_frontal": 100.0,
                    "is_state_gene": False,
                    "negative": is_negative,
                }
            )
    pd.DataFrame(records).to_parquet(bundle.path / "negative_genes.parquet")
    profile_rows = []
    for label, name in WHB.items():
        for index, gene in enumerate(GENE_IDS):
            profile_rows.append(
                {
                    "level": WHB_LEVELS[0],
                    "node": label,
                    "node_name": name,
                    "n_cells": 100,
                    "gene_id": gene,
                    "detection_fraction": 0.5,
                    "mean_log2cpm": 1.0,
                    "mean_cpm": 800.0 if index == MARKER[label] else 40.0,
                    "expected_fraction": 0.1,
                }
            )
    pd.DataFrame(profile_rows).to_parquet(bundle.path / "profiles.parquet")


def _add_panel_coverage(bundle: MmcBundle) -> None:
    """Give a fake bundle the coverage records PREP writes (trust diagnostics)."""
    path = bundle.path / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest["builder_output"]["panel_coverage"] = {
        "n_panel_genes": 6,
        "n_query_genes_used": 60,
        "root_markers": 20,
        "root_children": 5,
        "root_children_separated": 5,
    }
    path.write_text(json.dumps(manifest))


def _run_record(
    run_id: str,
    bundle: MmcBundle,
    panel: AnnotationPanel,
    parquet: Path,
    map_dir: Path,
    n_cells: int,
) -> MapRunRecord:
    return MapRunRecord(
        run_id=run_id,
        reference_id=bundle.reference_id,
        role=bundle.role,
        purposes=["annotation"],
        panel_name="intersection",
        panel_hash=panel.panel_hash,
        n_panel_genes=panel.n_genes,
        build_hash=bundle.build_hash,
        bundle_path=str(bundle.path),
        builder_version=bundle.builder_version,
        bundle_lookup_sha256=bundle.lookup_sha256,
        lookup_sha256=bundle.lookup_sha256,
        lookup_restricted=False,
        query_fingerprint="synthetic",
        n_cells=n_cells,
        n_query_genes=panel.n_genes,
        n_missing_panel_genes=0,
        engine_params={
            "bootstrap_factor": 0.5,
            "bootstrap_iteration": 100,
            "rng_seed": 0,
            "n_processors": 6,
        },
        ctm_version="1.7.2",
        wall_s=1.0,
        parquet=str(parquet.relative_to(map_dir)),
        parquet_sha256=file_sha256(parquet),
        tidy_schema_version=1,
    )


def _setup(root: Path, fake_mmc: FakeMmc, *, coverage: bool = False) -> Setup:
    config = _config()
    panel_dir, panel = _panel_dir(root / "map" / "panel")
    bundles = {
        key[0]: bundle for key, bundle in _human_bundles(fake_mmc, panel).items()
    }
    _bundle_tables(bundles["whb_frontal_supc_clus"])
    if coverage:
        for bundle in bundles.values():
            _add_panel_coverage(bundle)
        bundles = {key: MmcBundle.from_dir(item.path) for key, item in bundles.items()}
    map_dir = root / "map"
    samples: list[MapSample] = []
    calls: dict[str, pd.DataFrame] = {}
    for seed, platform in enumerate(PLATFORM_CELLS):
        cells = _cells(platform, seed)
        counts = _counts(cells, seed + 10)
        sample_id = f"PX_{platform}"
        path = _write_prepared(
            root / "prepared" / platform.lower() / f"{sample_id}_prepared.h5ad",
            cells,
            counts,
            platform,
        )
        samples.append(MapSample(sample_id, platform, path, "prepared"))
        calls[platform] = cells
    loaded = load_samples(samples, config, min_counts=10)
    records: dict[str, MapSampleRecord] = {}
    for item in loaded:
        platform = item.sample.platform
        table_ids = set(item.obs_names[item.in_table].astype(str))
        frame = calls[platform]
        cells = frame[frame["cell_id"].isin(table_ids)].reset_index(drop=True)
        runs = {}
        for run_id, tidy in (
            ("whb_frontal_supc_clus", _whb_tidy(cells)),
            ("seaad_mr_panel", _sea_tidy(cells)),
        ):
            parquet = (
                map_dir
                / platform.lower()
                / f"{item.sample.sample_id}_mmc_{run_id}.parquet"
            )
            write_tidy_parquet(tidy, parquet, {"synthetic": True})
            runs[run_id] = _run_record(
                run_id, bundles[run_id], panel, parquet, map_dir, len(cells)
            )
        records[item.sample.sample_id] = MapSampleRecord(
            sample_id=item.sample.sample_id,
            platform=platform,
            source="prepared",
            h5ad_path=str(item.sample.h5ad_path),
            sample_fingerprint=item.sample_fingerprint(),
            n_objects=item.n_objects,
            n_table_cells=item.n_table_cells,
            min_counts=10,
            declared_panel_hash=item.declared.panel_hash,
            n_features=len(item.feature_names),
            n_resolved_features=sum(1 for gene in item.feature_ids if gene),
            runs=runs,
        )
    manifest = MapManifest(
        pair_id="PX",
        segmentation="proseg_hybrid",
        species="human",
        anatomical_region="frontal_cortex",
        created_at="2026-09-28T00:00:00+00:00",
        ctm_version="1.7.2",
        ctm_commit=None,
        n_processors=6,
        min_counts=10,
        samples=records,
    )
    manifest.write(map_dir / MAP_MANIFEST_NAME)
    return Setup(root, map_dir, panel_dir, panel, bundles, samples, calls, config)


def _trusts(make_trust: MakeTrust, state: str = "validated_real") -> dict[str, Any]:
    return {
        "whb_frontal_supc_clus": make_trust(state),
        "seaad_mr_panel": make_trust(state, role="secondary"),
    }


def _resolve(
    setup: Setup,
    make_trust: MakeTrust,
    output: str = "resolve_out",
    *,
    state: str = "validated_real",
    **kwargs: Any,
) -> pl.ResolveResult:
    return annotate_resolve(
        setup.map_dir,
        kwargs.pop("config", setup.config),
        output_dir=setup.root / output,
        trust_overrides=kwargs.pop("trust_overrides", _trusts(make_trust, state)),
        n_bootstrap=kwargs.pop("n_bootstrap", 30),
        **kwargs,
    )


def test_annotate_resolve_writes_valid_labels_provenance_and_summary(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    result = _resolve(setup, make_trust)

    assert set(result.samples) == {"PX_MERSCOPE", "PX_XENIUM"}
    for sample_id, sample in result.samples.items():
        platform = sample.platform
        labels_path = (
            tmp_path
            / "resolve_out"
            / platform.lower()
            / (f"{sample_id}_celltype_labels.parquet")
        )
        assert sample.labels_path == labels_path and labels_path.is_file()
        labels, stored = read_label_table(labels_path)
        # The table survives parquet with its contract dtypes.
        validate_label_table(labels, "human")
        cells = setup.calls[platform]
        assert len(labels) == len(cells)
        assert labels[Columns.CELL_ID].tolist() == cells["cell_id"].tolist()
        assert labels[Columns.IN_TABLE].tolist() == (cells["depth"] >= 10).tolist()
        assert (labels[Columns.INSTANCE_ID] >= 1000).all()
        assert set(labels[Columns.PANEL_HASH].astype(str)) == {setup.panel.panel_hash}
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        low = labels.loc[~table, Columns.level("broad", "status")].astype(str)
        assert set(low) == {CellStatus.LOW_COUNTS.value}
        soft = labels[list(soft_columns("human"))].to_numpy(np.float64)
        np.testing.assert_allclose(soft[table].sum(axis=1), 1.0, atol=1e-5)
        assert np.isnan(soft[~table]).all()
        for column in (
            "mmc_whb_supercluster_name",
            "mmc_whb_supercluster_runner_up_1_bp",
            "mmc_seaad_Subclass_name",
        ):
            assert column in labels.columns
        confident = labels[Columns.level("broad", "status")].astype(str) == "confident"
        assert confident.any()
        sinks = labels["mmc_whb_supercluster_name"].astype(str) == "Splatter"
        assert set(labels.loc[sinks, Columns.CT_BRANCH].astype(str)) <= {
            "Mixed/Unknown",
            "Neurons/unresolved",
        }
        exc = labels[Columns.CT_FINAL_NAME].astype(str) == (
            "Upper-layer intratelencephalic"
        )
        assert set(labels.loc[exc, Columns.CT_BRANCH].astype(str)) == {
            "Neurons/Excitatory"
        }
        assert (labels[Columns.CT_MENDER_STATE] == labels[Columns.CT_BRANCH]).all()
        # Provenance: the manifest JSON, the parquet schema and uns agree.
        manifest_path = sample.manifest_path
        assert manifest_path is not None
        from_file = AnnotationProvenance.model_validate_json(manifest_path.read_text())
        assert stored == from_file == sample.provenance
        payload = json.loads(from_file.to_uns_json())
        assert unsafe_structure_problems(payload) == []
        assert payload["consensus"]["degraded_mode"] == "whb_sea"
        assert payload["gate"]["level"] in {"full", "broad_only"}
        assert payload["panel"]["panel_trust"] == "validated"
        assert payload["panel"]["banner"] is False
        assert "contaminated" in payload["flags"]["informative"]
        assert payload["thresholds"]["threshold_source"] == "validated_default"
        assert set(payload["composition"]) == {
            "soft",
            "soft_ge30",
            "confident",
            "argmax",
        }
        assert payload["confident_fraction_table"]["broad"] > 0
    summary_path = tmp_path / "resolve_out" / "PX_resolve_summary.json"
    assert result.summary_path == summary_path
    summary = json.loads(summary_path.read_text())
    assert summary["pair_id"] == "PX" and summary["step"] == "annotate_resolve"
    merscope = summary["samples"]["PX_MERSCOPE"]
    assert merscope["resolution"]["gate"]["level"] in {"full", "broad_only"}
    assert {item["flag"] for item in merscope["flags"]["strata"]} == {
        "contaminated",
        "diffuse_profile",
        "ood",
    }
    assert merscope["reweighted_to_composition"] is False  # no resolvability tables
    jsd = summary["pair"]["jsd"]
    assert {item["kind"] for item in jsd} == {
        "soft",
        "soft_ge30",
        "confident",
        "argmax",
    }
    for item in jsd:
        assert item["resampling"] == "joint"
        assert item["ci_low"] <= item["ci_high"]
        assert item["n_reps"] == 30


def test_provenance_round_trips_through_h5ad_uns(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    result = _resolve(setup, make_trust)
    provenance = result.samples["PX_XENIUM"].provenance
    adata = ad.AnnData(X=np.zeros((2, 1)))
    provenance.write_to_uns(adata.uns)
    path = tmp_path / "roundtrip.h5ad"
    adata.write_h5ad(path)
    restored = AnnotationProvenance.read_from_uns(ad.read_h5ad(path).uns)
    assert restored == provenance
    assert isinstance(ad.read_h5ad(path).uns[PROVENANCE_UNS_KEY], str)


def test_provisional_trust_sets_the_banner_and_warning_without_lowering_the_gate(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    validated = _resolve(setup, make_trust, "validated")
    provisional = _resolve(setup, make_trust, "provisional", state="provisional")
    for sample_id in validated.samples:
        first = validated.samples[sample_id].provenance
        second = provisional.samples[sample_id].provenance
        assert first.panel is not None and second.panel is not None
        assert first.panel.banner is False and second.panel.banner is True
        assert first.gate is not None and second.gate is not None
        assert second.gate.level == first.gate.level
        assert second.gate.warning
        assert any("panel_provisional" in reason for reason in second.gate.reasons)
        assert not any("panel_provisional" in reason for reason in first.gate.reasons)
        summary = provisional.samples[sample_id].summary
        assert summary["banner"] is True and summary["trust"]["state"] == "provisional"
        # Without a resolvability table the provisional regime cannot raise a
        # threshold: emission and statuses stay the validated ones.
        labels_a = validated.samples[sample_id].labels
        labels_b = provisional.samples[sample_id].labels
        for level in ("lineage", "broad", "nt"):
            column = Columns.level(level, "status")
            assert labels_a[column].astype(str).tolist() == (
                labels_b[column].astype(str).tolist()
            )
        validated_flags = labels_a[Columns.level("broad", "validated")].to_numpy(bool)
        assert validated_flags.any()
        assert not labels_b[Columns.level("broad", "validated")].any()


class _FakeTables:
    """A stand-in for ``ResolvabilityTables`` (decisions from a fixture)."""

    def __init__(self, decisions: pd.DataFrame, levels: list[Any]) -> None:
        self._decisions = decisions
        self.levels = levels
        self.summary: dict[str, Any] = {
            "decision_recipe": "R1_contam_HO",
            "recipes": [{"name": "R1_contam_HO", "version": 1}],
            "d_max": {"broad": {"Exc": 250, "Astro": 250}},
            "fine_level_seed_stability": {},
        }
        self.calls: list[Any] = []

    @property
    def depth_grid(self) -> list[int]:
        return [10, 15, 30, 60, 120, 250]

    def decisions(
        self, *, composition: Any = None, settings: Any = None
    ) -> pd.DataFrame:
        self.calls.append(composition)
        return self._decisions


def test_resolvability_tables_are_reweighted_and_trust_sets_the_regime(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    make_decisions: Callable[..., pd.DataFrame],
    human_level_meta: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from merxen.annotation import resolvability
    from merxen.annotation.resolvability import DatasetComposition

    setup = _setup(tmp_path, fake_mmc)
    # The provisional regime raises broad Exc to 0.99 everywhere; the validated
    # regime is emitted at the defaults.
    overrides = {
        ("provisional", "broad", "Exc", depth): {"threshold": 0.99, "t_star": 0.99}
        for depth in (10, 15, 30, 60, 120, 250)
    }
    tables = _FakeTables(make_decisions(overrides=overrides), human_level_meta)
    monkeypatch.setattr(resolvability, "load_resolvability", lambda path: tables)
    validated = _resolve(setup, make_trust, "validated")
    assert all(isinstance(call, DatasetComposition) for call in tables.calls)
    composition = tables.calls[0]
    assert sum(composition.overall.values()) > 0
    assert set(composition.overall) <= set(WHB) | {None}
    provisional = _resolve(setup, make_trust, "provisional", state="provisional")
    plain = _resolve(
        setup,
        make_trust,
        "plain",
        config=_config().model_copy(
            update={
                "resolvability": _config().resolvability.model_copy(
                    update={"reweight_to_composition": False}
                )
            }
        ),
    )
    assert tables.calls[-1] is None
    # A real-data-validated family keeps its pre-registered emission: the
    # provisional raises in the table change nothing, so the statuses equal
    # a run without resolvability tables.
    monkeypatch.setattr(resolvability, "load_resolvability", lambda path: None)
    untabled = _resolve(setup, make_trust, "untabled")
    for sample_id, sample in validated.samples.items():
        for level in ("lineage", "broad", "nt", "supercluster"):
            column = Columns.level(level, "status")
            assert sample.labels[column].astype(str).tolist() == (
                untabled.samples[sample_id].labels[column].astype(str).tolist()
            )
    for sample_id, sample in validated.samples.items():
        assert sample.summary["reweighted_to_composition"] is True
        assert plain.samples[sample_id].summary["reweighted_to_composition"] is False
        broad = Columns.level("broad", "status")
        exc = sample.labels["mmc_whb_supercluster_name"].astype(str) == (
            "Upper-layer intratelencephalic"
        )
        confident_validated = (sample.labels[broad].astype(str) == "confident") & exc
        confident_provisional = (
            provisional.samples[sample_id].labels[broad].astype(str) == "confident"
        ) & exc
        assert confident_validated.sum() > confident_provisional.sum()
        prov = sample.provenance.resolvability["whb_frontal_supc_clus"]
        assert prov.reweighted_to_composition is True
        assert prov.recipe == "R1_contam_HO"
        assert "broad" in prov.resolvable_share
        assert prov.emitted_depth_bins["broad"]["exc"] == [10, 15, 30, 60, 120, 250]
        second = provisional.samples[sample_id].provenance
        assert second.thresholds is not None
        assert "resolvability_local" in str(second.thresholds.threshold_source)


def test_a_refused_panel_writes_statuses_only(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    manifest = pl.load_map_manifest(setup.map_dir / MAP_MANIFEST_NAME)
    refused = manifest.model_copy(
        update={
            "samples": {},
            "panel_status": "refused",
            "panel_reasons": ["intersection: species_mismatch"],
        }
    )
    refused.write(setup.map_dir / MAP_MANIFEST_NAME)
    with pytest.raises(ResolveError, match="prepared inputs"):
        annotate_resolve(setup.map_dir, setup.config, output_dir=tmp_path / "x")
    result = annotate_resolve(
        setup.map_dir,
        setup.config,
        output_dir=tmp_path / "refused_out",
        samples=setup.samples,
        n_bootstrap=10,
    )
    for sample in result.samples.values():
        labels = sample.labels
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        for level in ("lineage", "broad", "nt", "supercluster", "seaad_subclass"):
            status = labels[Columns.level(level, "status")].astype(str)
            assert set(status[table]) == {CellStatus.NOT_ATTEMPTED_GATE.value}
        assert labels[Columns.EXCLUDE_HARD].all()
        assert (labels[Columns.CT_FINAL_LEVEL].astype(str) == "none").all()
        assert sample.summary["resolution"]["gate"]["level"] == "failed"
        assert sample.provenance.panel is not None
        assert sample.provenance.panel.panel_trust == "refused"
        assert sample.provenance.consensus is not None
        assert sample.provenance.consensus.degraded_mode == "primary_missing"


def test_resolve_refuses_inputs_that_do_not_fit(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    with pytest.raises(ResolveError, match="min_counts"):
        annotate_resolve(
            setup.map_dir,
            AnnotationConfig(species="human").coupled_to_clustering(20),
            output_dir=tmp_path / "a",
        )
    with pytest.raises(ResolveError, match="human run"):
        annotate_resolve(
            setup.map_dir,
            AnnotationConfig(species="mouse").coupled_to_clustering(10),
            output_dir=tmp_path / "b",
        )
    # A changed parquet.
    parquet = setup.map_dir / "xenium" / "PX_XENIUM_mmc_seaad_mr_panel.parquet"
    original = parquet.read_bytes()
    parquet.write_bytes(original + b"\0")
    with pytest.raises(ResolveError, match="sha256"):
        _resolve(setup, make_trust, "c")
    parquet.write_bytes(original)
    # Changed counts.
    merscope = setup.samples[0]
    adata = ad.read_h5ad(merscope.h5ad_path)
    adata.X = adata.X * 2
    adata.write_h5ad(merscope.h5ad_path)
    with pytest.raises(ResolveError, match="fingerprint"):
        _resolve(setup, make_trust, "d")


def test_mouse_resolve_is_m6(tmp_path: Path, fake_mmc: FakeMmc) -> None:
    setup = _setup(tmp_path, fake_mmc)
    manifest = pl.load_map_manifest(setup.map_dir / MAP_MANIFEST_NAME)
    manifest.model_copy(update={"species": "mouse"}).write(
        setup.map_dir / MAP_MANIFEST_NAME
    )
    with pytest.raises(NotImplementedError, match="M6"):
        annotate_resolve(
            setup.map_dir,
            AnnotationConfig(species="mouse").coupled_to_clustering(10),
            output_dir=tmp_path / "mouse",
        )


def test_a_bundle_override_must_describe_the_same_mapping(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    from .conftest import FakeNode

    setup = _setup(tmp_path, fake_mmc)
    other_panel = fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash="b" * 64,
        n_genes=6,
        levels=list(WHB_LEVELS),
        nodes=[FakeNode("CS_EXC", WHB["CS_EXC"], GENE_IDS[0], {})],
    )
    with pytest.raises(ResolveError, match="built on panel"):
        _resolve(
            setup,
            make_trust,
            "e",
            bundle_overrides={"whb_frontal_supc_clus": other_panel},
        )
    other_lookup = fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=setup.panel.panel_hash,
        n_genes=6,
        levels=list(WHB_LEVELS),
        nodes=[FakeNode("CS_EXC", WHB["CS_EXC"], GENE_IDS[0], {})],
        build_hash="c" * 64,
    )
    with pytest.raises(ResolveError, match="lookup"):
        _resolve(
            setup,
            make_trust,
            "f",
            bundle_overrides={"whb_frontal_supc_clus": other_lookup},
        )
    sea = fake_mmc.bundle(
        "seaad_mr_panel",
        role="secondary",
        species="human",
        panel_hash=setup.panel.panel_hash,
        n_genes=6,
        levels=list(SEA_LEVELS),
        nodes=[FakeNode("SC_L23", "L2/3 IT", GENE_IDS[0], {})],
        build_hash="d" * 64,
    )
    with pytest.raises(ResolveError, match="not whb_frontal_supc_clus"):
        _resolve(
            setup, make_trust, "g", bundle_overrides={"whb_frontal_supc_clus": sea}
        )


def test_cli_annotate_resolve_runs_on_map_outputs(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    setup = _setup(tmp_path, fake_mmc, coverage=True)
    output = tmp_path / "cli_out"
    result = CliRunner().invoke(
        cli_main,
        [
            "annotate-resolve",
            "--map-dir",
            str(setup.map_dir),
            "--out",
            str(output),
            "--n-segmented",
            "PX_MERSCOPE=400",
            "--n-bootstrap",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "annotate-resolve: 2 sample(s)" in result.output
    summary = json.loads((output / "PX_resolve_summary.json").read_text())
    merscope = summary["samples"]["PX_MERSCOPE"]
    assert merscope["n_segmented"] == 400
    # The fake panel is outside the validated families and its bundle has no
    # self-map: the fail-safe broad_only trust state caps the gate.
    assert merscope["trust"]["state"] == "broad_only"
    assert merscope["resolution"]["gate"]["level"] == "broad_only"
    assert (output / "merscope" / "PX_MERSCOPE_celltype_labels.parquet").is_file()
    inside = CliRunner().invoke(
        cli_main,
        [
            "annotate-resolve",
            "--map-dir",
            str(setup.map_dir),
            "--out",
            str(setup.map_dir / "r"),
        ],
    )
    assert inside.exit_code != 0
    assert "lies inside" in inside.output
    bad = CliRunner().invoke(
        cli_main,
        [
            "annotate-resolve",
            "--map-dir",
            str(setup.map_dir),
            "--out",
            str(tmp_path / "bad"),
            "--n-segmented",
            "PX_MERSCOPE",
        ],
    )
    assert bad.exit_code != 0


def test_human_branch_columns_follow_the_final_chain(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    labels = _resolve(setup, make_trust).samples["PX_MERSCOPE"].labels
    final = labels[Columns.CT_FINAL_LEVEL].astype(str)
    branch = labels[Columns.CT_BRANCH].astype(str)
    leaf = labels[Columns.CT_LEAF].astype(str)
    assert (branch[final == "none"] == "Mixed/Unknown").all()
    supercluster = final == "supercluster"
    assert supercluster.any()
    assert (
        leaf[supercluster]
        == labels.loc[supercluster, Columns.CT_FINAL_NAME].astype(str)
    ).all()
    assert (leaf[~supercluster] == "unresolved").all()
    lineage_neurons = (final == "lineage") & (
        labels[Columns.level("lineage", "name")].astype(str) == "Neurons"
    )
    assert (branch[lineage_neurons] == "Neurons/unresolved").all()
