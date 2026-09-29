"""Tests for the annotation MAP step and ``merxen annotate`` (plan §3.3, M3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import sparse

from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mapmycells_engine import (
    MMC_SINGLE_THREAD_ENV,
    MmcBundle,
    MmcEngineError,
    level_frame,
    query_fingerprint,
    read_tidy_parquet,
)
from merxen.annotation.panel import (
    PANEL_GENES_FILE,
    PANEL_GENES_SETC_FILE,
    REQUIRED_BUNDLES_FILE,
    AnnotationPanel,
    RequiredBundle,
    RequiredBundles,
    compute_panel_hash,
    load_annotation_panel,
)
from merxen.annotation.pipeline import (
    MAP_MANIFEST_NAME,
    MOUSE_REGION_STEP,
    PROVISIONAL_METADATA_KEY,
    PROVISIONAL_SUFFIX,
    MapError,
    MapSample,
    annotate_map,
    build_sample_query,
    check_output_outside_inputs,
    load_map_manifest,
    load_samples,
    locate_bundle,
    map_bundles,
    published_layout,
    store_subset_bundle_finder,
)
from merxen.annotation.store import ReferenceStore
from merxen.cli import main as cli_main

from .conftest import FakeMmc, FakeNode

GENE_IDS = [f"ENSG{index:011d}" for index in range(1, 7)]
SYMBOLS = ["GEXC", "GAST", "GOLI", "GSPL", "GHIP", "GOTHER"]
WHB_LEVELS = ["CCN202210140_SUPC", "CCN202210140_CLUS"]
SEA_LEVELS = ["SEA_LEVEL_0", "SEA_LEVEL_1", "SEA_LEVEL_2"]

WHB_NODES = [
    FakeNode(
        "CS_EXC",
        "Upper-layer intratelencephalic",
        GENE_IDS[0],
        {"broad_class": "Neurons", "lineage": "Neurons", "nt": "Excitatory"},
    ),
    FakeNode(
        "CS_AST",
        "Astrocyte",
        GENE_IDS[1],
        {"broad_class": "Astrocytes", "lineage": "Astrocytes"},
    ),
    FakeNode(
        "CS_OLI",
        "Oligodendrocyte",
        GENE_IDS[2],
        {"broad_class": "Oligodendrocytes", "lineage": "Oligodendrocyte lineage"},
    ),
    FakeNode(
        "CS_SPL",
        "Splatter",
        GENE_IDS[3],
        {
            "broad_class": "Mixed/Unknown",
            "lineage": "Neurons",
            "nt": "Excitatory",
            "sink": "True",
        },
    ),
    FakeNode(
        "CS_HIP",
        "Hippocampal CA1-3",
        GENE_IDS[4],
        {
            "broad_class": "Neurons",
            "lineage": "Neurons",
            "nt": "Excitatory",
            "region_plausible_frontal_cortex": "False",
        },
    ),
]
SEA_NODES = [
    FakeNode("SC_L23", "L2/3 IT", GENE_IDS[0], {"broad_class": "Neurons"}),
    FakeNode("SC_AST", "Astrocyte", GENE_IDS[1], {"broad_class": "Astrocytes"}),
    FakeNode(
        "SC_OLI", "Oligodendrocyte", GENE_IDS[2], {"broad_class": "Oligodendrocytes"}
    ),
]

# MERSCOPE cells (counts on GEXC, GAST, GOLI, GSPL, GHIP, GOTHER, Blank-0001).
MERSCOPE_COUNTS = np.array(
    [
        [20, 0, 0, 0, 0, 1, 0],  # c0: excitatory, confident
        [2, 10, 0, 0, 0, 0, 0],  # c1: astrocyte, bp .83
        [6, 5, 0, 0, 0, 0, 0],  # c2: lineage .55 -> low confidence
        [0, 0, 0, 15, 0, 0, 0],  # c3: Splatter sink -> implausible
        [0, 0, 0, 0, 12, 0, 0],  # c4: hippocampal -> region-implausible
        [3, 0, 0, 0, 0, 0, 0],  # c5: below min_counts
        [7, 0, 0, 0, 5, 0, 0],  # c6: lineage 1.0, supercluster .58
        [0, 0, 30, 0, 0, 0, 100],  # c7: oligodendrocyte; the Blank is a control
    ]
)
XENIUM_COUNTS = np.array(
    [
        [0, 12, 0, 0, 0, 0, 0],
        [15, 0, 0, 0, 0, 0, 2],
        [0, 0, 11, 0, 0, 0, 0],
    ]
)


def _write_h5ad(
    path: Path,
    counts: np.ndarray,
    *,
    platform: str,
    var_names: list[str],
    ensembl_ids: list[str] | None,
    source: str,
) -> Path:
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    var["gene"] = var_names
    if ensembl_ids is not None:
        var["ensembl_id"] = ensembl_ids
    obs = pd.DataFrame(index=[f"{platform[0]}{index}" for index in range(len(counts))])
    obs["instance_id"] = np.arange(len(counts), dtype=np.int64) + 100
    matrix = sparse.csr_matrix(counts.astype(np.int64))
    if source == "clustered":
        adata = ad.AnnData(X=matrix.astype(np.float32), obs=obs, var=var)
        adata.layers["counts"] = matrix
    else:
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["spatialdata_attrs"] = {
        "instance_key": "instance_id",
        "region": "cells",
        "region_key": "region",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def _pair_samples(root: Path, *, source: str = "prepared") -> list[MapSample]:
    """A MERSCOPE (symbols only) + Xenium (IDs) pair with one control each."""
    merscope = _write_h5ad(
        root / "merscope" / f"PX_MERSCOPE_{source}.h5ad",
        MERSCOPE_COUNTS,
        platform="MERSCOPE",
        var_names=[*SYMBOLS, "Blank-0001"],
        ensembl_ids=None,
        source=source,
    )
    xenium = _write_h5ad(
        root / "xenium" / f"PX_XENIUM_{source}.h5ad",
        XENIUM_COUNTS,
        platform="XENIUM",
        var_names=[*SYMBOLS, "NegControlProbe_00001"],
        ensembl_ids=[*GENE_IDS, "NegControlProbe_00001"],
        source=source,
    )
    kind = "prepared" if source == "prepared" else "clustered"
    return [
        MapSample("PX_MERSCOPE", "MERSCOPE", merscope, kind),  # type: ignore[arg-type]
        MapSample("PX_XENIUM", "XENIUM", xenium, kind),  # type: ignore[arg-type]
    ]


def _panel(ids: list[str], name: str = "intersection") -> AnnotationPanel:
    ids = sorted(ids)
    return AnnotationPanel(
        name=name,
        kind="intersection" if name == "intersection" else "setc",
        species="human",
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=["PX_MERSCOPE", "PX_XENIUM"],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[SYMBOLS[GENE_IDS.index(gene_id)] for gene_id in ids],
    )


def _panel_dir(root: Path, *, setc: bool = False) -> tuple[Path, AnnotationPanel]:
    root.mkdir(parents=True, exist_ok=True)
    panel = _panel(GENE_IDS)
    panel.write(root / PANEL_GENES_FILE)
    bundles = [
        RequiredBundle(
            reference_id="whb_frontal_supc_clus",
            role="primary",
            species="human",
            purpose="annotation",
            panel_name="intersection",
            panel_hash=panel.panel_hash,
            panel_file=PANEL_GENES_FILE,
            n_panel_genes=panel.n_genes,
        ),
        RequiredBundle(
            reference_id="seaad_mr_panel",
            role="secondary",
            species="human",
            purpose="annotation",
            panel_name="intersection",
            panel_hash=panel.panel_hash,
            panel_file=PANEL_GENES_FILE,
            n_panel_genes=panel.n_genes,
        ),
    ]
    if setc:
        setc_panel = _panel(GENE_IDS[:5], name="setc")
        setc_panel.write(root / PANEL_GENES_SETC_FILE)
        bundles.insert(
            1,
            RequiredBundle(
                reference_id="whb_frontal_supc_clus",
                role="primary",
                species="human",
                purpose="setc_sensitivity",
                panel_name="setc",
                panel_hash=setc_panel.panel_hash,
                panel_file=PANEL_GENES_SETC_FILE,
                n_panel_genes=setc_panel.n_genes,
            ),
        )
    required = RequiredBundles(
        pair_id="PX",
        segmentation="proseg_hybrid",
        species="human",
        panel_mode="intersection",
        status="ok",
        bundles=bundles,
        n_required=len(bundles),
    )
    (root / REQUIRED_BUNDLES_FILE).write_text(
        json.dumps(required.model_dump(mode="json"))
    )
    return root, panel


def _human_bundles(
    fake_mmc: FakeMmc, panel: AnnotationPanel, *, store_root: Path | None = None
) -> dict[tuple[str, str | None], MmcBundle]:
    whb = fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=panel.panel_hash,
        n_genes=panel.n_genes,
        levels=WHB_LEVELS,
        nodes=WHB_NODES,
        store_root=store_root,
    )
    sea = fake_mmc.bundle(
        "seaad_mr_panel",
        role="secondary",
        species="human",
        panel_hash=panel.panel_hash,
        n_genes=panel.n_genes,
        levels=SEA_LEVELS,
        nodes=SEA_NODES,
        store_root=store_root,
    )
    return {
        ("whb_frontal_supc_clus", panel.panel_hash): MmcBundle.from_dir(whb),
        ("seaad_mr_panel", panel.panel_hash): MmcBundle.from_dir(sea),
    }


def _config(**updates: Any) -> AnnotationConfig:
    config = AnnotationConfig(species="human", **updates)
    return config.coupled_to_clustering(10)


def _setup(
    tmp_path: Path, fake_mmc: FakeMmc, *, setc: bool = False
) -> tuple[list[MapSample], list[Any], AnnotationConfig]:
    samples = _pair_samples(tmp_path / "prepared")
    panel_dir, panel = _panel_dir(tmp_path / "panel", setc=setc)
    bundles = _human_bundles(fake_mmc, panel)
    if setc:
        setc_panel = _panel(GENE_IDS[:5], name="setc")
        setc_dir = fake_mmc.bundle(
            "whb_frontal_supc_clus",
            role="primary",
            species="human",
            panel_hash=setc_panel.panel_hash,
            n_genes=setc_panel.n_genes,
            levels=WHB_LEVELS,
            nodes=WHB_NODES,
        )
        bundles[("whb_frontal_supc_clus", setc_panel.panel_hash)] = MmcBundle.from_dir(
            setc_dir
        )
    config = _config()
    required = RequiredBundles.model_validate_json(
        (panel_dir / REQUIRED_BUNDLES_FILE).read_text()
    )
    runs = map_bundles(required, panel_dir, bundles, config)
    return samples, runs, config


def test_annotate_map_end_to_end_with_a_fake_mapper(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    output = tmp_path / "annotation_out"

    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=output,
        pair_id="PX",
        segmentation="proseg_hybrid",
        n_processors=6,
    )

    assert [run.run_id for run in runs] == ["whb_frontal_supc_clus", "seaad_mr_panel"]
    assert (output / MAP_MANIFEST_NAME).is_file()
    written = load_map_manifest(output / MAP_MANIFEST_NAME)
    assert written.model_dump() == manifest.model_dump()
    assert written.ctm_version == "1.7.2"
    assert written.n_processors == 6
    assert written.min_counts == 10
    merscope = written.samples["PX_MERSCOPE"]
    assert merscope.n_objects == 8
    assert merscope.n_table_cells == 7
    assert merscope.controls_removed == {"name_pattern": 1}
    whb = merscope.runs["whb_frontal_supc_clus"]
    assert whb.n_cells == 7
    assert whb.n_query_genes == 6
    assert whb.n_missing_panel_genes == 0
    assert not whb.lookup_restricted
    assert whb.engine_params["rng_seed"] == 0
    assert whb.engine_params["bootstrap_factor"] == 0.5
    assert whb.engine_params["bootstrap_iteration"] == 100
    assert whb.build_hash == runs[0].bundle.build_hash
    assert not whb.reused
    table_ids = [f"M{index}" for index in range(8) if index != 5]
    totals = MERSCOPE_COUNTS[:, :6].sum(axis=1)[[0, 1, 2, 3, 4, 6, 7]]
    assert whb.query_fingerprint == query_fingerprint(table_ids, totals, GENE_IDS)
    parquet = output / whb.parquet
    assert parquet.name == "PX_MERSCOPE_mmc_whb_frontal_supc_clus.parquet"
    tidy, metadata = read_tidy_parquet(parquet)
    assert metadata["query_fingerprint"] == whb.query_fingerprint
    supc = level_frame(tidy, "CCN202210140_SUPC")
    assert list(supc.index) == table_ids
    assert supc.loc["M7", "assignment"] == "CS_OLI"
    assert not list(output.rglob("*.json.gz"))
    assert not list(output.rglob("*extended.json"))
    assert not (output / ".work").exists()
    assert len(fake_mmc.calls) == 4
    for call in fake_mmc.calls:
        for name in MMC_SINGLE_THREAD_ENV:
            assert call["env"][name] == "1"
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    xenium = written.samples["PX_XENIUM"]
    assert xenium.n_table_cells == 3
    assert set(xenium.runs) == {"whb_frontal_supc_clus", "seaad_mr_panel"}


def test_provisional_labels_apply_raw_thresholds_only(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    import pyarrow.parquet as pq

    samples, runs, config = _setup(tmp_path, fake_mmc)
    output = tmp_path / "annotation_out"
    manifest = annotate_map(
        samples, runs, config, output_dir=output, pair_id="PX", segmentation="seg"
    )

    record = manifest.samples["PX_MERSCOPE"]
    assert record.provisional_labels is not None
    path = output / record.provisional_labels
    assert path.name == f"PX_MERSCOPE{PROVISIONAL_SUFFIX}"
    metadata = json.loads(pq.read_schema(path).metadata[PROVISIONAL_METADATA_KEY])
    assert metadata["provisional"] is True
    assert "RESOLVE" in metadata["rules"]
    labels = pd.read_parquet(path).set_index("cell_id")
    assert list(labels.index) == [f"M{index}" for index in range(8)]
    assert labels.loc["M7", "total_counts"] == 30
    assert labels.loc["M7", "instance_id"] == 107
    status = labels[
        [
            "ct_lineage_status",
            "ct_broad_status",
            "ct_nt_status",
            "ct_supercluster_status",
            "ct_seaad_subclass_status",
        ]
    ].astype(str)
    assert list(status.loc["M0"]) == ["confident"] * 5
    assert labels.loc["M0", "ct_nt_name"] == "Excitatory"
    assert labels.loc["M0", "ct_final_level"] == "supercluster"
    assert labels.loc["M0", "ct_final_name"] == "Upper-layer intratelencephalic"
    assert list(status.loc["M1"]) == [
        "confident",
        "confident",
        "not_applicable",
        "confident",
        "confident",
    ]
    assert labels.loc["M1", "ct_broad_name"] == "Astrocytes"
    assert labels.loc["M1", "ct_broad_raw"] == pytest.approx(0.83)
    assert labels.loc["M1", "ct_broad_runner_up"] == "Neurons"
    assert labels.loc["M1", "ct_broad_margin"] == pytest.approx(0.66)
    assert status.loc["M2", "ct_lineage_status"] == "low_confidence"
    assert status.loc["M2", "ct_broad_status"] == "parent_unresolved"
    assert labels.loc["M2", "ct_final_level"] == "none"
    assert labels.loc["M2", "ct_final_name"] == "Mixed/Unknown"
    assert status.loc["M3", "ct_lineage_status"] == "implausible"
    assert status.loc["M4", "ct_lineage_status"] == "implausible"
    assert status.loc["M4", "ct_supercluster_status"] == "parent_unresolved"
    assert list(status.loc["M5"]) == ["low_counts"] * 5
    assert pd.isna(labels.loc["M5", "ct_broad_name"])
    assert not labels.loc["M5", "in_table"]
    # Same-class runner-ups add up (Excitatory .58 + hippocampal .42).
    assert labels.loc["M6", "ct_lineage_raw"] == pytest.approx(1.0)
    assert labels.loc["M6", "ct_nt_raw"] == pytest.approx(1.0)
    assert status.loc["M6", "ct_supercluster_status"] == "low_confidence"
    assert labels.loc["M6", "ct_final_level"] == "nt"
    assert labels.loc["M7", "ct_final_name"] == "Oligodendrocyte"
    assert labels.loc["M7", "mmc_whb_supercluster_name"] == "Oligodendrocyte"
    assert labels.loc["M7", "mmc_seaad_subclass_name"] == "Oligodendrocyte 1"
    assert "mmc_whb_supercluster_runner_up_5_bp" in labels.columns
    assert record.provisional_summary["broad"] == pytest.approx(4 / 7, abs=1e-6)


def test_setc_runs_where_required_and_share_the_sample_query_rules(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc, setc=True)

    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="proseg_hybrid",
    )

    runs_by_id = manifest.samples["PX_MERSCOPE"].runs
    assert set(runs_by_id) == {
        "whb_frontal_supc_clus",
        "whb_frontal_supc_clus_setc",
        "seaad_mr_panel",
    }
    setc = runs_by_id["whb_frontal_supc_clus_setc"]
    assert setc.purposes == ["setc_sensitivity"]
    assert setc.n_query_genes == 5
    assert setc.build_hash != runs_by_id["whb_frontal_supc_clus"].build_hash
    labels = pd.read_parquet(
        tmp_path / "out" / manifest.samples["PX_MERSCOPE"].provisional_labels  # type: ignore[operator]
    )
    assert "mmc_whb_setc_supercluster_bp" in labels.columns


def test_annotate_map_reuses_identical_published_runs(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    published = tmp_path / "published"
    first = annotate_map(
        samples, runs, config, output_dir=published, pair_id="PX", segmentation="s"
    )
    n_calls = len(fake_mmc.calls)

    second = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "rerun",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )

    assert len(fake_mmc.calls) == n_calls
    for sample_id, record in second.samples.items():
        for run_id, run in record.runs.items():
            assert run.reused
            original = first.samples[sample_id].runs[run_id]
            assert run.query_fingerprint == original.query_fingerprint
            assert run.parquet_sha256 == original.parquet_sha256
            assert (tmp_path / "rerun" / run.parquet).is_file()


def test_reuse_ignores_the_resolve_only_settings(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    """A threshold, gate, flag or degraded-mode change re-runs RESOLVE only.

    The MAP reuse key (query fingerprint, build_hash, engine parameters, ctm
    version, tidy schema, lookup) holds none of RESOLVE's settings, so a
    re-run MAP with them changed copies every published run (plan §3.1).
    """
    samples, runs, config = _setup(tmp_path, fake_mmc)
    published = tmp_path / "published"
    annotate_map(
        samples, runs, config, output_dir=published, pair_id="PX", segmentation="s"
    )
    n_calls = len(fake_mmc.calls)
    changed = config.model_copy(
        update={
            "allow_single_method": True,
            "thresholds": config.thresholds.model_copy(
                update={"whb_broad": 0.8, "whb_supercluster": 0.75}
            ),
            "gate": config.gate.model_copy(update={"min_frac_ge30": 0.5}),
            "flags": config.flags.model_copy(update={"contamination_alpha": 0.05}),
        }
    )

    manifest = annotate_map(
        samples,
        runs,
        changed,
        output_dir=tmp_path / "rerun",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )

    assert len(fake_mmc.calls) == n_calls
    assert all(
        run.reused
        for record in manifest.samples.values()
        for run in record.runs.values()
    )


def test_reuse_needs_the_same_query_and_engine_parameters(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    output = tmp_path / "out"
    annotate_map(
        samples, runs, config, output_dir=output, pair_id="PX", segmentation="s"
    )
    n_calls = len(fake_mmc.calls)

    # One more count in one MERSCOPE cell changes that sample's fingerprint.
    changed = MERSCOPE_COUNTS.copy()
    changed[0, 0] += 1
    _write_h5ad(
        samples[0].h5ad_path,
        changed,
        platform="MERSCOPE",
        var_names=[*SYMBOLS, "Blank-0001"],
        ensembl_ids=None,
        source="prepared",
    )
    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=output,
        pair_id="PX",
        segmentation="s",
        reuse_from=output,
    )
    assert len(fake_mmc.calls) == n_calls + 2
    assert not manifest.samples["PX_MERSCOPE"].runs["seaad_mr_panel"].reused
    assert manifest.samples["PX_XENIUM"].runs["seaad_mr_panel"].reused

    # Another process count is another mapping (R12).
    annotate_map(
        samples,
        runs,
        config,
        output_dir=output,
        pair_id="PX",
        segmentation="s",
        reuse_from=output,
        n_processors=3,
    )
    assert len(fake_mmc.calls) == n_calls + 6

    # reuse_published = false maps again.
    annotate_map(
        samples,
        runs,
        _config(reuse_published=False),
        output_dir=output,
        pair_id="PX",
        segmentation="s",
        reuse_from=output,
        n_processors=3,
    )
    assert len(fake_mmc.calls) == n_calls + 10


def test_reuse_refuses_a_changed_published_parquet(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    output = tmp_path / "out"
    manifest = annotate_map(
        samples, runs, config, output_dir=output, pair_id="PX", segmentation="s"
    )
    n_calls = len(fake_mmc.calls)
    parquet = output / manifest.samples["PX_XENIUM"].runs["seaad_mr_panel"].parquet
    parquet.write_bytes(parquet.read_bytes() + b"x")

    rerun = annotate_map(
        samples,
        runs,
        config,
        output_dir=output,
        pair_id="PX",
        segmentation="s",
        reuse_from=output,
    )

    assert len(fake_mmc.calls) == n_calls + 1
    assert not rerun.samples["PX_XENIUM"].runs["seaad_mr_panel"].reused


def test_load_samples_removes_controls_like_the_legacy_filter(tmp_path: Path) -> None:
    from merxen.analysis.clustering_squidpy import remove_control_features

    names = [
        *SYMBOLS,
        "Blank-0001",
        "NegControlProbe_00042",
        "UnassignedCodeword_0003",
        "NegControlCodeword_0500",
    ]
    counts = np.ones((3, len(names)), dtype=np.int64) * 3
    path = _write_h5ad(
        tmp_path / "x.h5ad",
        counts,
        platform="XENIUM",
        var_names=names,
        ensembl_ids=[*GENE_IDS, "", "", "", ""],
        source="prepared",
    )
    sample = MapSample("PX_XENIUM", "XENIUM", path, "prepared")

    (loaded,) = load_samples([sample], _config(), min_counts=10)

    legacy = remove_control_features(ad.read_h5ad(path))
    assert loaded.feature_names == list(legacy.var_names)
    assert loaded.feature_ids == GENE_IDS
    np.testing.assert_array_equal(loaded.total_counts, [18, 18, 18])
    np.testing.assert_array_equal(loaded.n_genes, [6, 6, 6])
    assert loaded.in_table.all()


def test_load_samples_keeps_every_published_table_cell(tmp_path: Path) -> None:
    samples = _pair_samples(tmp_path / "clustered", source="clustered")

    loaded = load_samples(samples, _config(), min_counts=10)

    merscope = loaded[0]
    assert merscope.sample.source == "clustered"
    assert merscope.in_table.all()
    assert merscope.n_below_min_in_clustered == 1
    # MERSCOPE symbols resolve through the Xenium IDs of the pair.
    assert merscope.feature_ids == GENE_IDS


def test_build_sample_query_restricts_to_the_panel_and_lists_missing_genes(
    tmp_path: Path,
) -> None:
    samples = _pair_samples(tmp_path)
    loaded = load_samples(samples, _config(), min_counts=10)
    extra = "ENSG00000000099"
    panel = _panel(GENE_IDS[:3]).model_copy()
    panel = AnnotationPanel(
        **{
            **panel.model_dump(),
            "ensembl_ids": sorted([*GENE_IDS[:3], extra]),
            "symbols": ["GEXC", "GAST", "GOLI", "X"],
            "n_genes": 4,
            "panel_hash": compute_panel_hash([*GENE_IDS[:3], extra]),
        }
    )

    query = build_sample_query(loaded[0], panel)

    assert query.gene_ids == GENE_IDS[:3]
    assert query.missing_gene_ids == [extra]
    assert list(query.cell_ids) == [f"M{index}" for index in range(8) if index != 5]
    np.testing.assert_array_equal(
        query.counts.toarray(), MERSCOPE_COUNTS[[0, 1, 2, 3, 4, 6, 7], :3]
    )
    np.testing.assert_array_equal(
        query.total_counts, MERSCOPE_COUNTS[[0, 1, 2, 3, 4, 6, 7], :6].sum(axis=1)
    )


def test_declared_genes_min_cells_dropped_are_mapped_with_zero_counts(
    tmp_path: Path,
) -> None:
    # A published clustered MERSCOPE table: its control-filter record lists
    # GOTHER, which min_cells dropped from var. Counting it missing made the
    # clustered path request an own-family subset bundle where the prepared
    # path (which still has the gene) requests none (M3b review 2).
    from types import SimpleNamespace

    from merxen.annotation.pipeline import _subset_bundle_for

    xenium, merscope = (
        _write_h5ad(
            tmp_path / platform.lower() / f"PX_{platform}_clustered.h5ad",
            counts,
            platform=platform,
            var_names=names,
            ensembl_ids=ids,
            source="clustered",
        )
        for platform, counts, names, ids in (
            (
                "XENIUM",
                XENIUM_COUNTS,
                [*SYMBOLS, "NegControlProbe_00001"],
                [*GENE_IDS, ""],
            ),
            ("MERSCOPE", MERSCOPE_COUNTS[:, :5], SYMBOLS[:5], None),
        )
    )
    adata = ad.read_h5ad(merscope)
    adata.uns["merxen_clustering_squidpy"] = {
        "control_feature_filter": {
            "retained_features": list(SYMBOLS),
            "removed_control_features": ["Blank-0001"],
        }
    }
    adata.write_h5ad(merscope)
    samples = [
        MapSample("PX_MERSCOPE", "MERSCOPE", merscope, "clustered"),  # type: ignore[arg-type]
        MapSample("PX_XENIUM", "XENIUM", xenium, "clustered"),  # type: ignore[arg-type]
    ]
    loaded, _ = load_samples(samples, _config(), min_counts=10)
    assert loaded.declared.ensembl_ids == sorted(GENE_IDS)
    assert "GOTHER" not in loaded.feature_names
    assert loaded.undetected_declared_ids == (GENE_IDS[5],)

    query = build_sample_query(loaded, _panel(GENE_IDS))

    assert query.gene_ids == GENE_IDS
    assert query.missing_gene_ids == []
    assert query.undetected_gene_ids == [GENE_IDS[5]]
    assert query.counts.toarray()[:, 5].sum() == 0
    # The panel is complete: no subset bundle is requested.
    run = SimpleNamespace(panel=_panel(GENE_IDS))
    assert _subset_bundle_for(
        loaded,
        run,  # type: ignore[arg-type]
        _config(),
        output=tmp_path / "out",
        find_subset_bundle=None,
    ) == (run, None)


def test_build_sample_query_sums_features_resolving_to_one_panel_gene(
    tmp_path: Path,
) -> None:
    # A second feature (an alias probe, like H2AX / H2AFX) resolves to the
    # first panel gene: its counts are added to that gene's column.
    duplicate = np.array([[4], [3], [0]])
    counts = np.hstack([XENIUM_COUNTS[:, :6], duplicate, XENIUM_COUNTS[:, 6:]])
    path = _write_h5ad(
        tmp_path / "xenium" / "PX_XENIUM_prepared.h5ad",
        counts,
        platform="XENIUM",
        var_names=[*SYMBOLS, "GEXC_ALT", "NegControlProbe_00001"],
        ensembl_ids=[*GENE_IDS, GENE_IDS[0], "NegControlProbe_00001"],
        source="prepared",
    )
    sample = MapSample("PX_XENIUM", "XENIUM", path, "prepared")  # type: ignore[arg-type]
    (loaded,) = load_samples([sample], _config(), min_counts=10)
    assert loaded.feature_ids.count(GENE_IDS[0]) == 2

    query = build_sample_query(loaded, _panel(GENE_IDS))

    assert query.gene_ids == GENE_IDS
    dense = query.counts.toarray()
    np.testing.assert_array_equal(dense[:, 0], XENIUM_COUNTS[:, 0] + duplicate[:, 0])
    np.testing.assert_array_equal(dense[:, 1:], XENIUM_COUNTS[:, 1:6])


def test_a_missing_marker_gene_restricts_the_lookup(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    # The Xenium panel lost the hippocampal marker (GHIP).
    keep = [index for index in range(7) if index != 4]
    _write_h5ad(
        samples[1].h5ad_path,
        XENIUM_COUNTS[:, keep],
        platform="XENIUM",
        var_names=[[*SYMBOLS, "NegControlProbe_00001"][index] for index in keep],
        ensembl_ids=[[*GENE_IDS, "NegControlProbe_00001"][index] for index in keep],
        source="prepared",
    )

    manifest = annotate_map(
        samples[1:],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
    )

    run = manifest.samples["PX_XENIUM"].runs["whb_frontal_supc_clus"]
    assert run.n_missing_panel_genes == 1
    assert run.missing_panel_genes == [GENE_IDS[4]]
    assert run.lookup_restricted
    assert run.lookup_sha256 != run.bundle_lookup_sha256
    assert run.n_query_genes == 5
    # GHIP is a root marker and 1 of 6 genes is 17% > 5%: a subset bundle of
    # its own family is requested; without it the restricted lookup is used.
    subset = run.subset_bundle
    assert subset is not None and subset.status == "requested"
    assert subset.trigger.action == "own_family"
    assert subset.trigger.reasons == [
        "root_marker_missing",
        "weak_parent",
        "missing_frac",
    ]
    assert subset.trigger.weak_parents == {"None": 4}  # 5 root markers - 1
    assert subset.subset_panel_hash == compute_panel_hash(
        [g for g in GENE_IDS if g != GENE_IDS[4]]
    )
    assert subset.parent_build_hash == runs[0].bundle.build_hash
    panel = load_annotation_panel(tmp_path / "out" / subset.subset_panel_file)
    assert panel.kind == "subset" and panel.excluded_ids == [GENE_IDS[4]]
    assert panel.panel_hash == subset.subset_panel_hash


def test_a_subset_bundle_in_the_store_is_mapped_with(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    keep = [index for index in range(7) if index != 4]
    _write_h5ad(
        samples[1].h5ad_path,
        XENIUM_COUNTS[:, keep],
        platform="XENIUM",
        var_names=[[*SYMBOLS, "NegControlProbe_00001"][index] for index in keep],
        ensembl_ids=[[*GENE_IDS, "NegControlProbe_00001"][index] for index in keep],
        source="prepared",
    )
    present = [g for g in GENE_IDS if g != GENE_IDS[4]]
    subset_dir = fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=compute_panel_hash(present),
        n_genes=len(present),
        levels=WHB_LEVELS,
        nodes=[node for node in WHB_NODES if node.marker != GENE_IDS[4]],
    )
    subset_bundle = MmcBundle.from_dir(subset_dir)
    asked: list[tuple[str, str]] = []

    def finder(reference_id: str, panel_hash: str) -> MmcBundle | None:
        asked.append((reference_id, panel_hash))
        return subset_bundle if reference_id == "whb_frontal_supc_clus" else None

    manifest = annotate_map(
        samples[1:],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
        find_subset_bundle=finder,
    )

    records = manifest.samples["PX_XENIUM"].runs
    whb = records["whb_frontal_supc_clus"]
    assert whb.subset_bundle is not None and whb.subset_bundle.status == "used"
    assert whb.build_hash == subset_bundle.build_hash
    assert whb.subset_bundle.subset_build_hash == subset_bundle.build_hash
    assert whb.subset_bundle.parent_build_hash == runs[0].bundle.build_hash
    assert whb.panel_hash == compute_panel_hash(present)
    assert whb.panel_name == "intersection_subset"
    assert not whb.lookup_restricted and whb.n_missing_panel_genes == 0
    # SEA-AD has no subset bundle: requested, mapped with the parent bundle
    # (GHIP is none of its markers, so its lookup is not even restricted).
    seaad = records["seaad_mr_panel"]
    assert seaad.subset_bundle is not None
    assert seaad.subset_bundle.status == "requested"
    assert seaad.subset_bundle.trigger.reasons == ["missing_frac"]
    assert seaad.n_missing_panel_genes == 1 and not seaad.lookup_restricted
    assert {reference for reference, _ in asked} == {
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    }


def _drop_ghip_from_xenium(h5ad_path: Path) -> list[str]:
    """Rewrite the Xenium sample without GHIP; return the genes present."""
    keep = [index for index in range(7) if index != 4]
    _write_h5ad(
        h5ad_path,
        XENIUM_COUNTS[:, keep],
        platform="XENIUM",
        var_names=[[*SYMBOLS, "NegControlProbe_00001"][index] for index in keep],
        ensembl_ids=[[*GENE_IDS, "NegControlProbe_00001"][index] for index in keep],
        source="prepared",
    )
    return [g for g in GENE_IDS if g != GENE_IDS[4]]


def _whb_subset_bundle(
    fake_mmc: FakeMmc, present: list[str], *, build_hash: str | None = None
) -> Path:
    return fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=compute_panel_hash(present),
        n_genes=len(present),
        levels=WHB_LEVELS,
        nodes=[node for node in WHB_NODES if node.marker != GENE_IDS[4]],
        build_hash=build_hash,
    )


def test_several_subset_bundles_in_the_store_are_recorded_as_ambiguous(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    present = _drop_ghip_from_xenium(samples[1].h5ad_path)
    for build_hash in ("a" * 64, "b" * 64):
        _whb_subset_bundle(fake_mmc, present, build_hash=build_hash)

    manifest = annotate_map(
        samples[1:],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
        find_subset_bundle=store_subset_bundle_finder(ReferenceStore(fake_mmc.root)),
    )

    whb = manifest.samples["PX_XENIUM"].runs["whb_frontal_supc_clus"]
    assert whb.subset_bundle is not None
    assert whb.subset_bundle.status == "ambiguous"
    assert len(whb.subset_bundle.candidates) == 2
    assert whb.subset_bundle.subset_build_hash is None
    # Mapped with the parent bundle and the restricted lookup, as requested.
    assert whb.build_hash == runs[0].bundle.build_hash
    assert whb.lookup_restricted


def test_the_store_subset_finder_opens_a_bundle_like_a_bundle_ref(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    present = [g for g in GENE_IDS if g != GENE_IDS[4]]
    bundle_dir = _whb_subset_bundle(fake_mmc, present)
    finder = store_subset_bundle_finder(ReferenceStore(fake_mmc.root))
    found = finder("whb_frontal_supc_clus", compute_panel_hash(present))
    assert found is not None and found.path == bundle_dir
    assert finder("whb_frontal_supc_clus", compute_panel_hash(GENE_IDS[:2])) is None
    # A changed marker lookup fails the bundle's integrity check.
    lookup = bundle_dir / "query_markers.filtered.json"
    lookup.write_text(lookup.read_text().replace("]", ', "ENSG00000000042"]', 1))
    with pytest.raises(MmcEngineError, match="not intact"):
        finder("whb_frontal_supc_clus", compute_panel_hash(present))
    # A bundle.json that disagrees with its content address is never used.
    manifest_path = bundle_dir / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build_hash"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert finder("whb_frontal_supc_clus", compute_panel_hash(present)) is None


def test_map_bundles_needs_every_required_bundle(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    bundles = _human_bundles(fake_mmc, panel)
    required = RequiredBundles.model_validate_json(
        (panel_dir / REQUIRED_BUNDLES_FILE).read_text()
    )
    del bundles[("seaad_mr_panel", panel.panel_hash)]

    with pytest.raises(MapError, match="no bundle for seaad_mr_panel"):
        map_bundles(required, panel_dir, bundles, _config())
    runs = map_bundles(
        required, panel_dir, bundles, _config(), references=["whb_frontal_supc_clus"]
    )
    assert [run.run_id for run in runs] == ["whb_frontal_supc_clus"]


def test_mouse_maps_with_the_drop_level_and_no_pruning_when_none(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    levels = [
        "CCN20230722_CLAS",
        "CCN20230722_SUBC",
        "CCN20230722_SUPT",
        "CCN20230722_CLUS",
    ]
    ids = [f"ENSMUSG{index:011d}" for index in range(1, 4)]
    nodes = [
        FakeNode(
            "CL_01",
            "01 IT-ET Glut",
            ids[0],
            {"broad_class": "Neurons", "nt": "Excitatory"},
        ),
        FakeNode(
            "CL_30", "30 Astro-Epen", ids[1], {"broad_class": "Astrocytes/Ependymal"}
        ),
    ]
    panel = AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="mouse",
        platforms=["MERSCOPE"],
        sample_ids=["AG_MERSCOPE"],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(ids),
        n_genes=3,
        ensembl_ids=ids,
        symbols=["Slc17a7", "Aqp4", "Other"],
    )
    bundle = MmcBundle.from_dir(
        fake_mmc.bundle(
            "wmb_panel",
            role="primary",
            species="mouse",
            panel_hash=panel.panel_hash,
            n_genes=3,
            levels=levels,
            nodes=nodes,
            drop_level="CCN20230722_SUPT",
        )
    )
    panel_dir = tmp_path / "panel"
    panel.write(panel_dir / PANEL_GENES_FILE)
    required = RequiredBundles(
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
                n_panel_genes=3,
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
    config = AnnotationConfig(species="mouse").coupled_to_clustering(10)
    runs = map_bundles(
        required, panel_dir, {("wmb_panel", panel.panel_hash): bundle}, config
    )
    path = _write_h5ad(
        tmp_path / "AG_MERSCOPE.h5ad",
        np.array([[30, 0, 1], [0, 12, 0], [2, 1, 0]]),
        platform="MERSCOPE",
        var_names=["Slc17a7", "Aqp4", "Other"],
        ensembl_ids=ids,
        source="prepared",
    )

    sample = MapSample("AG_MERSCOPE", "MERSCOPE", path, "prepared")
    with pytest.raises(MapError, match="needs the wmb_region_share bundle"):
        annotate_map(
            [sample],
            runs,
            config,
            output_dir=tmp_path / "x",
            pair_id="AG",
            segmentation="proseg_hybrid",
        )
    assert not fake_mmc.calls  # refused before anything is mapped

    manifest = annotate_map(
        [sample],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="AG",
        segmentation="proseg_hybrid",
        section_regions={"AG_MERSCOPE": "none"},
    )

    command = fake_mmc.calls[-1]["command"]
    assert command[command.index("--drop_level") + 1] == "CCN20230722_SUPT"
    assert "--nodes_to_drop" not in command
    assert manifest.mouse_region_step == MOUSE_REGION_STEP.format(variant="v1")
    record = manifest.samples["AG_MERSCOPE"]
    assert set(record.runs) == {"wmb_panel"}
    assert record.mouse_regions is not None
    assert record.mouse_regions.status == "disabled"
    assert record.mouse_regions.source == "none"
    assert record.mouse_regions.remap is None
    assert record.mouse_regions.region_share is None
    labels = pd.read_parquet(tmp_path / "out" / str(record.provisional_labels))
    labels = labels.set_index("cell_id")
    assert labels.loc["M0", "ct_class_name"] == "01 IT-ET Glut"
    assert labels.loc["M0", "ct_subclass_status"] == "confident"
    assert labels.loc["M0", "ct_nt_name"] == "Excitatory"
    assert labels.loc["M1", "ct_broad_name"] == "Astrocytes/Ependymal"
    assert labels.loc["M1", "ct_nt_status"] == "not_applicable"
    assert labels.loc["M2", "ct_class_status"] == "low_counts"
    assert labels.loc["M0", "ct_final_level"] == "subclass"
    assert "mmc_wmb_supertype_name" in labels.columns


def test_published_layout_reads_the_results_path(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "results/P7513/reseg/clustering_squidpy/clustering_squidpy_out/xenium"
        / "P7513_XENIUM_clustered.h5ad"
    )

    layout = published_layout(path)

    assert layout.results_root == (tmp_path / "results").resolve()
    assert layout.pair_id == "P7513"
    assert layout.segmentation == "reseg"
    assert layout.platform == "XENIUM"
    assert layout.sample_id == "P7513_XENIUM"
    other = published_layout(tmp_path / "loose" / "S1_MERSCOPE_clustered.h5ad")
    assert other.results_root is None
    assert other.platform == "MERSCOPE"


def test_output_must_stay_outside_the_results_tree(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "results/P1/proseg_hybrid/clustering_squidpy/clustering_squidpy_out"
        / "merscope/P1_MERSCOPE_clustered.h5ad"
    )

    for bad in (
        tmp_path / "results" / "P1" / "annot",
        path.parent,
        tmp_path / "results",
    ):
        with pytest.raises(MapError, match="never into the results tree"):
            check_output_outside_inputs(bad, [path])
    check_output_outside_inputs(tmp_path / "shadow", [path])


def test_locate_bundle_picks_the_current_builder_bundle(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    panel = _panel(GENE_IDS)
    common: dict[str, Any] = {
        "role": "primary",
        "species": "human",
        "panel_hash": panel.panel_hash,
        "n_genes": panel.n_genes,
        "levels": WHB_LEVELS,
        "nodes": WHB_NODES,
    }
    fake_mmc.bundle("whb_frontal_supc_clus", builder_version=1, **common)
    current = fake_mmc.bundle("whb_frontal_supc_clus", **common)
    store = ReferenceStore(fake_mmc.root)

    assert (
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash).path == current
    )
    with pytest.raises(MapError, match="no builder-v"):
        locate_bundle(store, "seaad_mr_panel", panel.panel_hash)
    fake_mmc.bundle("whb_frontal_supc_clus", build_hash="f" * 64, **common)
    with pytest.raises(MapError, match="2 bundles"):
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash)


def test_locate_bundle_prefers_the_current_resolvability_version(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    from merxen.annotation.resolvability import RESOLVABILITY_VERSION

    panel = _panel(GENE_IDS)
    common: dict[str, Any] = {
        "role": "primary",
        "species": "human",
        "panel_hash": panel.panel_hash,
        "n_genes": panel.n_genes,
        "levels": WHB_LEVELS,
        "nodes": WHB_NODES,
    }

    def with_version(path: Path, version: int) -> Path:
        manifest_path = path / "bundle.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["builder_output"]["resolvability"] = {"resolvability_version": version}
        manifest_path.write_text(json.dumps(manifest))
        return path

    with_version(
        fake_mmc.bundle("whb_frontal_supc_clus", build_hash="a" * 64, **common),
        RESOLVABILITY_VERSION - 1,
    )
    current = with_version(
        fake_mmc.bundle("whb_frontal_supc_clus", build_hash="b" * 64, **common),
        RESOLVABILITY_VERSION,
    )
    store = ReferenceStore(fake_mmc.root)
    # A rebuild after a RESOLVABILITY_VERSION bump sits next to the old
    # bundle on the same panel: standalone runs take the current tables.
    assert (
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash).path == current
    )
    with_version(
        fake_mmc.bundle("whb_frontal_supc_clus", build_hash="c" * 64, **common),
        RESOLVABILITY_VERSION,
    )
    with pytest.raises(MapError, match="2 bundles"):
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash)


def test_locate_bundle_warns_when_only_stale_self_map_tables_exist(
    tmp_path: Path, fake_mmc: FakeMmc, caplog: pytest.LogCaptureFixture
) -> None:
    # A panel not rebuilt since a RESOLVABILITY_VERSION bump (the M3b large
    # store) keeps its older bundle: standalone runs still find it, and the
    # log names its stale version (M3b review 3).
    from merxen.annotation.resolvability import RESOLVABILITY_VERSION

    panel = _panel(GENE_IDS)
    common: dict[str, Any] = {
        "role": "primary",
        "species": "human",
        "panel_hash": panel.panel_hash,
        "n_genes": panel.n_genes,
        "levels": WHB_LEVELS,
        "nodes": WHB_NODES,
    }
    path = fake_mmc.bundle("whb_frontal_supc_clus", build_hash="a" * 64, **common)
    manifest = json.loads((path / "bundle.json").read_text())
    manifest["builder_output"]["resolvability"] = {
        "resolvability_version": RESOLVABILITY_VERSION - 2
    }
    (path / "bundle.json").write_text(json.dumps(manifest))
    store = ReferenceStore(fake_mmc.root)
    with caplog.at_level(logging.WARNING, logger="merxen.annotation.pipeline"):
        found = locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash)
        finder = store_subset_bundle_finder(store)
        subset = finder("whb_frontal_supc_clus", panel.panel_hash)
    assert found.path == path
    assert subset is not None and subset.path == path
    stale = [record for record in caplog.records if "stale" in record.getMessage()]
    assert len(stale) == 2
    assert f"has version {RESOLVABILITY_VERSION - 2}" in stale[0].getMessage()
    assert f"resolvability version {RESOLVABILITY_VERSION})" in stale[0].getMessage()
    # A current bundle, or one without a self-map, logs nothing.
    caplog.clear()
    manifest["builder_output"]["resolvability"] = {}
    (path / "bundle.json").write_text(json.dumps(manifest))
    with caplog.at_level(logging.WARNING, logger="merxen.annotation.pipeline"):
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash)
    manifest["builder_output"]["resolvability"] = {
        "resolvability_version": RESOLVABILITY_VERSION
    }
    (path / "bundle.json").write_text(json.dumps(manifest))
    with caplog.at_level(logging.WARNING, logger="merxen.annotation.pipeline"):
        locate_bundle(store, "whb_frontal_supc_clus", panel.panel_hash)
    assert not [record for record in caplog.records if "stale" in record.getMessage()]


def test_locate_bundle_takes_the_bundle_of_the_configured_prefilter(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    # A 5K panel's store holds a bundle built with the per-parent prefilter
    # and one without it: the run config decides which one a standalone run
    # maps with (M3b review 2); without a config both stay ambiguous.
    from merxen.annotation.prefilter import prefilter_payload

    panel = _panel(GENE_IDS)
    common: dict[str, Any] = {
        "role": "primary",
        "species": "mouse",
        "panel_hash": panel.panel_hash,
        "n_genes": 5006,
        "levels": WHB_LEVELS,
        "nodes": WHB_NODES,
    }

    def with_prefilter(path: Path, prefilter: dict[str, Any] | None) -> Path:
        manifest_path = path / "bundle.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["build_hash_payload"] = {
            "panel": {"panel_hash": panel.panel_hash, "n_genes": 5006},
            "large_panel_prefilter": prefilter,
        }
        manifest_path.write_text(json.dumps(manifest))
        return path

    plain = with_prefilter(
        fake_mmc.bundle("wmb_panel", build_hash="a" * 64, **common), None
    )
    filtered = with_prefilter(
        fake_mmc.bundle("wmb_panel", build_hash="b" * 64, **common),
        prefilter_payload(2000, 30),
    )
    store = ReferenceStore(fake_mmc.root)
    default = AnnotationConfig(species="mouse")
    assert (
        locate_bundle(store, "wmb_panel", panel.panel_hash, config=default).path
        == plain
    )
    opted_in = AnnotationConfig(
        species="mouse",
        panel={"large_panel_marker_prefilter": "per_parent_topk_union"},
    )
    assert (
        locate_bundle(store, "wmb_panel", panel.panel_hash, config=opted_in).path
        == filtered
    )
    other_cap = AnnotationConfig(
        species="mouse",
        panel={
            "large_panel_marker_prefilter": "per_parent_topk_union",
            "large_panel_prefilter_cap": 1500,
        },
    )
    with pytest.raises(MapError, match="no builder-v"):
        locate_bundle(store, "wmb_panel", panel.panel_hash, config=other_cap)
    with pytest.raises(MapError, match="2 bundles"):
        locate_bundle(store, "wmb_panel", panel.panel_hash)
    finder = store_subset_bundle_finder(store, default)
    found = finder("wmb_panel", panel.panel_hash)
    assert found is not None and found.path == plain


def _published_pair(root: Path) -> list[Path]:
    paths = []
    for platform, counts, names, ids in (
        ("MERSCOPE", MERSCOPE_COUNTS, [*SYMBOLS, "Blank-0001"], None),
        ("XENIUM", XENIUM_COUNTS, [*SYMBOLS, "NegControlProbe_00001"], [*GENE_IDS, ""]),
    ):
        path = (
            root
            / "PX/proseg_hybrid/clustering_squidpy/clustering_squidpy_out"
            / platform.lower()
            / f"PX_{platform}_clustered.h5ad"
        )
        paths.append(
            _write_h5ad(
                path,
                counts,
                platform=platform,
                var_names=names,
                ensembl_ids=ids,
                source="clustered",
            )
        )
    return paths


def test_cli_annotate_runs_on_published_clustered_h5ads(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    inputs = _published_pair(tmp_path / "results")
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    _human_bundles(fake_mmc, panel)
    output = tmp_path / "shadow" / "PX"
    arguments = [
        "annotate",
        "--species",
        "human",
        *(item for path in inputs for item in ("--from-clustered-h5ad", str(path))),
        "--store",
        str(fake_mmc.root),
        "--panel-dir",
        str(panel_dir),
        "--out",
        str(output),
        "--n-processors",
        "2",
    ]

    result = CliRunner().invoke(cli_main, arguments)

    assert result.exit_code == 0, result.output
    manifest = load_map_manifest(output / MAP_MANIFEST_NAME)
    assert manifest.pair_id == "PX"
    assert manifest.segmentation == "proseg_hybrid"
    assert manifest.n_processors == 2
    assert manifest.samples["PX_MERSCOPE"].source == "clustered"
    assert manifest.samples["PX_MERSCOPE"].n_table_cells == 8
    assert (output / "merscope" / "PX_MERSCOPE_mmc_seaad_mr_panel.parquet").is_file()
    n_calls = len(fake_mmc.calls)

    again = CliRunner().invoke(cli_main, arguments)

    assert again.exit_code == 0, again.output
    assert len(fake_mmc.calls) == n_calls
    assert "reused" in again.output


def test_cli_annotate_computes_the_panel_when_none_is_given(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    inputs = _published_pair(tmp_path / "results")
    output = tmp_path / "shadow"

    result = CliRunner().invoke(
        cli_main,
        [
            "annotate",
            "--species",
            "human",
            *(item for path in inputs for item in ("--from-clustered-h5ad", str(path))),
            "--out",
            str(output),
            "--bundle",
            f"whb_frontal_supc_clus={tmp_path / 'missing'}",
        ],
    )

    # The panel is computed from the inputs; the missing bundle then fails
    # with a one-line error.
    assert result.exit_code != 0
    assert (output / "panel" / REQUIRED_BUNDLES_FILE).is_file()
    assert "Error:" in result.output


def test_cli_annotate_refuses_an_output_in_the_results_tree(tmp_path: Path) -> None:
    inputs = _published_pair(tmp_path / "results")

    result = CliRunner().invoke(
        cli_main,
        [
            "annotate",
            "--species",
            "human",
            "--from-clustered-h5ad",
            str(inputs[0]),
            "--out",
            str(tmp_path / "results" / "PX" / "annotation"),
        ],
    )

    assert result.exit_code != 0
    assert "never into the results tree" in result.output
    assert not (tmp_path / "results" / "PX" / "annotation").exists()


def _prepared_dir(root: Path) -> tuple[Path, Path]:
    """A CLUSTERING_SQUIDPY_PREPARE output and its clustering config."""
    samples = _pair_samples(root / "clustering_prepare_out")
    prepared = root / "clustering_prepare_out"
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "samples": {
                    sample.sample_id: str(sample.h5ad_path.relative_to(prepared))
                    for sample in samples
                }
            }
        )
    )
    config = root / "clustering_squidpy_config.json"
    config.write_text(
        json.dumps(
            {
                "pair_id": "PX",
                "min_counts": 10,
                "samples": [
                    {"sample_id": sample.sample_id, "platform": sample.platform}
                    for sample in samples
                ],
            }
        )
    )
    return prepared, config


def _bundle_refs(
    root: Path, bundles: dict[tuple[str, str | None], MmcBundle]
) -> list[Path]:
    from merxen.annotation.store import BundleRef

    paths = []
    for index, ((reference_id, panel_hash), bundle) in enumerate(bundles.items()):
        paths.append(
            BundleRef(
                reference_id=reference_id,
                species="human",
                role="primary" if reference_id.startswith("whb") else "secondary",
                panel_hash=panel_hash,
                build_hash=bundle.build_hash,
                path=str(bundle.path),
                store_root=str(bundle.path.parents[1]),
            ).write(root / f"bundle_ref_{index + 1}.json")
        )
    return paths


def _pipeline_arguments(
    prepared: Path, config: Path, panel_dir: Path, refs: list[Path], output: Path
) -> list[str]:
    """The arguments CLUSTERING_SQUIDPY_ANNOTATE_MAP passes (annotation.nf)."""
    return [
        "annotate",
        "--species",
        "human",
        "--prepared-dir",
        str(prepared),
        "--clustering-config",
        str(config),
        "--segmentation",
        "proseg_hybrid",
        "--panel-dir",
        str(panel_dir),
        *(item for ref in refs for item in ("--bundle-ref", str(ref))),
        "--require-bundle-refs",
        "--allow-refused-panel",
        "--out",
        str(output),
        "--n-processors",
        "2",
    ]


def test_cli_annotate_runs_the_pipeline_arguments(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    """Prepared directory, clustering config and PREP's bundle refs only."""
    from merxen.annotation.store import BundleRef

    prepared, config = _prepared_dir(tmp_path / "inputs")
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    refs = _bundle_refs(tmp_path / "refs", _human_bundles(fake_mmc, panel))
    # A ref of an unmapped role (the mouse region shares, M6) is staged too;
    # it has no mapping precompute and must never be opened.
    refs.append(
        BundleRef(
            reference_id="wmb_region_share",
            species="human",
            role="region_share",
            panel_hash=None,
            build_hash="0" * 64,
            path=str(tmp_path / "missing"),
            store_root=str(tmp_path),
        ).write(tmp_path / "refs" / "bundle_ref_9.json")
    )
    output = tmp_path / "annotation_map_out"

    result = CliRunner().invoke(
        cli_main, _pipeline_arguments(prepared, config, panel_dir, refs, output)
    )

    assert result.exit_code == 0, result.output
    manifest = load_map_manifest(output / MAP_MANIFEST_NAME)
    assert manifest.pair_id == "PX"
    assert manifest.min_counts == 10
    assert manifest.panel_status == "ok"
    assert manifest.samples["PX_MERSCOPE"].source == "prepared"
    assert set(manifest.samples["PX_XENIUM"].runs) == {
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    }
    assert (output / "xenium" / "PX_XENIUM_mmc_whb_frontal_supc_clus.parquet").is_file()


def test_cli_annotate_require_bundle_refs_never_reads_the_store(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    prepared, config = _prepared_dir(tmp_path / "inputs")
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    bundles = _human_bundles(fake_mmc, panel)
    whb_only = {
        key: value for key, value in bundles.items() if key[0].startswith("whb")
    }
    refs = _bundle_refs(tmp_path / "refs", whb_only)
    arguments = _pipeline_arguments(
        prepared, config, panel_dir, refs, tmp_path / "out"
    ) + ["--store", str(fake_mmc.root)]

    result = CliRunner().invoke(cli_main, arguments)

    # The store holds the SEA-AD bundle, but a pipeline task maps only what
    # PREP resolved.
    assert result.exit_code != 0
    assert "no --bundle-ref for seaad_mr_panel" in result.output
    assert fake_mmc.calls == []


def test_cli_annotate_require_bundle_refs_only_requests_a_subset_bundle(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    prepared, config = _prepared_dir(tmp_path / "inputs")
    xenium = json.loads((prepared / "manifest.json").read_text())["samples"][
        "PX_XENIUM"
    ]
    present = _drop_ghip_from_xenium(prepared / xenium)
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    bundles = _human_bundles(fake_mmc, panel)
    refs = _bundle_refs(tmp_path / "refs", bundles)
    subset_dir = _whb_subset_bundle(fake_mmc, present)
    pipeline = _pipeline_arguments(
        prepared, config, panel_dir, refs, tmp_path / "pipeline"
    ) + ["--store", str(fake_mmc.root)]

    result = CliRunner().invoke(cli_main, pipeline)

    # The store holds the subset bundle, but a pipeline task never looks it
    # up: PREP did not stage it and -resume would not track it.
    assert result.exit_code == 0, result.output
    manifest = load_map_manifest(tmp_path / "pipeline" / MAP_MANIFEST_NAME)
    whb = manifest.samples["PX_XENIUM"].runs["whb_frontal_supc_clus"]
    assert whb.subset_bundle is not None
    assert whb.subset_bundle.status == "requested"
    assert (
        whb.build_hash
        == bundles[("whb_frontal_supc_clus", panel.panel_hash)].build_hash
    )
    # A standalone run with the same store maps with the subset bundle.
    standalone = [item for item in pipeline if item != "--require-bundle-refs"]
    standalone[standalone.index(str(tmp_path / "pipeline"))] = str(
        tmp_path / "standalone"
    )
    result = CliRunner().invoke(cli_main, standalone)
    assert result.exit_code == 0, result.output
    manifest = load_map_manifest(tmp_path / "standalone" / MAP_MANIFEST_NAME)
    whb = manifest.samples["PX_XENIUM"].runs["whb_frontal_supc_clus"]
    assert whb.subset_bundle is not None and whb.subset_bundle.status == "used"
    assert whb.build_hash == MmcBundle.from_dir(subset_dir).build_hash


def test_cli_annotate_takes_min_counts_from_the_clustering_config(
    tmp_path: Path,
) -> None:
    prepared, config = _prepared_dir(tmp_path / "inputs")
    panel_dir, _ = _panel_dir(tmp_path / "panel")
    arguments = _pipeline_arguments(prepared, config, panel_dir, [], tmp_path / "out")

    result = CliRunner().invoke(cli_main, [*arguments, "--min-counts", "5"])

    assert result.exit_code != 0
    assert "differs from the clustering config's min_counts 10" in result.output


def _refuse(panel_dir: Path) -> None:
    path = panel_dir / REQUIRED_BUNDLES_FILE
    required = RequiredBundles.model_validate_json(path.read_text())
    refused = required.model_copy(
        update={
            "status": "refused",
            "reasons": ["gene-ID resolution 0.40 < 0.95"],
            "bundles": [],
            "n_required": 0,
        }
    )
    path.write_text(json.dumps(refused.model_dump(mode="json")))


def test_cli_annotate_records_a_refused_panel_without_mapping(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    prepared, config = _prepared_dir(tmp_path / "inputs")
    panel_dir, _ = _panel_dir(tmp_path / "panel")
    _refuse(panel_dir)
    output = tmp_path / "annotation_map_out"

    result = CliRunner().invoke(
        cli_main, _pipeline_arguments(prepared, config, panel_dir, [], output)
    )

    assert result.exit_code == 0, result.output
    manifest = load_map_manifest(output / MAP_MANIFEST_NAME)
    assert manifest.panel_status == "refused"
    assert manifest.panel_reasons == ["gene-ID resolution 0.40 < 0.95"]
    assert manifest.samples == {}
    assert manifest.min_counts == 10
    assert fake_mmc.calls == []
    # Without --allow-refused-panel (standalone use) a refused panel fails.
    arguments = _pipeline_arguments(prepared, config, panel_dir, [], tmp_path / "x")
    arguments.remove("--allow-refused-panel")
    refused = CliRunner().invoke(cli_main, arguments)
    assert refused.exit_code != 0
    assert "was refused" in refused.output


# --------------------------------------------------------------------------
# Runs per bundle use (per_platform pairs, merged uses)


def _platform_panel(ids: list[str], name: str, mode: str = "per_platform") -> Any:
    from merxen.annotation.panel import PanelFile

    ids = sorted(ids)
    platform = name.upper()
    panel = AnnotationPanel(
        name=name,
        kind="platform" if name != "intersection" else "intersection",
        species="human",
        platforms=[platform] if name != "intersection" else ["MERSCOPE", "XENIUM"],
        sample_ids=[f"PX_{platform}"] if name != "intersection" else ["PX_MERSCOPE"],
        panel_mode=mode,  # type: ignore[arg-type]
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[SYMBOLS[GENE_IDS.index(gene_id)] for gene_id in ids],
    )
    file_name = (
        f"panel_genes_{name}.json" if mode == "per_platform" else PANEL_GENES_FILE
    )
    return PanelFile(panel=panel, file_name=file_name)


def _per_platform_setup(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    merscope_ids: list[str],
    xenium_ids: list[str],
) -> tuple[list[MapSample], list[Any], AnnotationConfig, RequiredBundles]:
    """A per_platform pair through the real required_bundles and map_bundles."""
    from merxen.annotation.panel import required_bundles

    shared = sorted(set(merscope_ids) & set(xenium_ids))
    panels = {
        "merscope": _platform_panel(merscope_ids, "merscope"),
        "xenium": _platform_panel(xenium_ids, "xenium"),
        "intersection": _platform_panel(shared, "intersection"),
    }
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    for item in panels.values():
        item.panel.write(panel_dir / item.file_name)
    config = _config()
    bundle_list = required_bundles(
        panels,
        references=config.references,
        species="human",
        panel_mode="per_platform",
        segmentation="proseg_hybrid",
    )
    required = RequiredBundles(
        pair_id="PX",
        segmentation="proseg_hybrid",
        species="human",
        panel_mode="per_platform",
        status="ok",
        bundles=bundle_list,
        n_required=len(bundle_list),
    )
    bundles: dict[tuple[str, str | None], MmcBundle] = {}
    for item in required.bundles:
        nodes = WHB_NODES if item.reference_id.startswith("whb") else SEA_NODES
        levels = WHB_LEVELS if item.reference_id.startswith("whb") else SEA_LEVELS
        bundles[(item.reference_id, item.panel_hash)] = MmcBundle.from_dir(
            fake_mmc.bundle(
                item.reference_id,
                role=item.role,
                species="human",
                panel_hash=str(item.panel_hash),
                n_genes=int(item.n_panel_genes or 0),
                levels=levels,
                nodes=nodes,
            )
        )
    runs = map_bundles(required, panel_dir, bundles, config)
    return _pair_samples(tmp_path / "prepared"), runs, config, required


def _runs_by_platform(runs: list[Any]) -> dict[str, set[str]]:
    return {
        platform: {run.run_id for run in runs if run.applies_to(platform)}
        for platform in ("MERSCOPE", "XENIUM")
    }


ALL_RUNS = {
    "whb_frontal_supc_clus",
    "whb_frontal_supc_clus_xpanel",
    "seaad_mr_panel",
}


def test_per_platform_pair_maps_each_platform_panel_and_the_intersection(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    merscope_ids = [GENE_IDS[index] for index in (0, 1, 2, 3, 5)]
    xenium_ids = [GENE_IDS[index] for index in (0, 1, 2, 4, 5)]
    samples, runs, config, _ = _per_platform_setup(
        tmp_path, fake_mmc, merscope_ids, xenium_ids
    )

    assert _runs_by_platform(runs) == {"MERSCOPE": ALL_RUNS, "XENIUM": ALL_RUNS}
    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
    )

    for sample_id, own_ids in (
        ("PX_MERSCOPE", merscope_ids),
        ("PX_XENIUM", xenium_ids),
    ):
        record = manifest.samples[sample_id]
        assert set(record.runs) == ALL_RUNS
        own = record.runs["whb_frontal_supc_clus"]
        xpanel = record.runs["whb_frontal_supc_clus_xpanel"]
        assert own.purposes == ["annotation"]
        assert own.panel_hash == compute_panel_hash(sorted(own_ids))
        assert xpanel.purposes == ["intersection_xpanel"]
        assert xpanel.n_query_genes == 4
        assert xpanel.same_mapping_as is None
        labels = pd.read_parquet(tmp_path / "out" / str(record.provisional_labels))
        assert "mmc_whb_xpanel_supercluster_bp" in labels.columns
    # Six distinct mappings: two samples x (own WHB, intersection WHB, SEA-AD).
    assert len(fake_mmc.calls) == 6


def test_intersection_equal_to_the_merscope_panel_still_maps_xenium_on_it(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    # A small MERSCOPE panel inside a larger Xenium panel (§8.5): the
    # intersection has the MERSCOPE panel's hash, so required_bundles merges
    # the two uses into one bundle.
    merscope_ids = GENE_IDS[:5]
    samples, runs, config, required = _per_platform_setup(
        tmp_path, fake_mmc, merscope_ids, GENE_IDS
    )
    whb_items = [
        item for item in required.bundles if item.reference_id.startswith("whb")
    ]
    merged = next(item for item in whb_items if len(item.uses) == 2)
    assert [use.purpose for use in merged.uses] == [
        "annotation",
        "intersection_xpanel",
    ]

    assert _runs_by_platform(runs) == {"MERSCOPE": ALL_RUNS, "XENIUM": ALL_RUNS}
    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
    )

    xenium = manifest.samples["PX_XENIUM"].runs
    assert set(xenium) == ALL_RUNS
    assert xenium["whb_frontal_supc_clus_xpanel"].n_query_genes == 5
    assert xenium["whb_frontal_supc_clus"].n_query_genes == 6
    merscope = manifest.samples["PX_MERSCOPE"].runs
    assert set(merscope) == ALL_RUNS
    # The MERSCOPE sample's own panel is the intersection: one mapping,
    # recorded under both run ids.
    alias = merscope["whb_frontal_supc_clus_xpanel"]
    assert alias.same_mapping_as == "whb_frontal_supc_clus"
    assert alias.purposes == ["intersection_xpanel"]
    assert (
        alias.query_fingerprint == merscope["whb_frontal_supc_clus"].query_fingerprint
    )
    tidy, metadata = read_tidy_parquet(tmp_path / "out" / alias.parquet)
    assert metadata["run_id"] == "whb_frontal_supc_clus_xpanel"
    assert metadata["purposes"] == ["intersection_xpanel"]
    assert len(fake_mmc.calls) == 5


def test_identical_platform_panels_map_both_samples(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config, required = _per_platform_setup(
        tmp_path, fake_mmc, GENE_IDS, GENE_IDS
    )
    assert len(required.bundles) == 2

    assert _runs_by_platform(runs) == {"MERSCOPE": ALL_RUNS, "XENIUM": ALL_RUNS}
    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="s",
    )

    for record in manifest.samples.values():
        assert set(record.runs) == ALL_RUNS
        assert record.provisional_summary["broad"] > 0
    assert len(fake_mmc.calls) == 4


def test_setc_equal_to_set_a_records_the_setc_run(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    from merxen.annotation.panel import PanelFile, required_bundles

    panel = _panel(GENE_IDS)
    setc = _panel(GENE_IDS, name="setc")
    panels = {
        "intersection": PanelFile(panel=panel, file_name=PANEL_GENES_FILE),
        "setc": PanelFile(panel=setc, file_name=PANEL_GENES_SETC_FILE),
    }
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    panel.write(panel_dir / PANEL_GENES_FILE)
    setc.write(panel_dir / PANEL_GENES_SETC_FILE)
    config = _config()
    bundle_list = required_bundles(
        panels,
        references=config.references,
        species="human",
        panel_mode="intersection",
        segmentation="proseg_hybrid",
    )
    required = RequiredBundles(
        pair_id="PX",
        segmentation="proseg_hybrid",
        species="human",
        panel_mode="intersection",
        status="ok",
        bundles=bundle_list,
        n_required=len(bundle_list),
    )
    assert len(required.bundles) == 2
    runs = map_bundles(required, panel_dir, _human_bundles(fake_mmc, panel), config)

    manifest = annotate_map(
        _pair_samples(tmp_path / "prepared"),
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PX",
        segmentation="proseg_hybrid",
    )

    record = manifest.samples["PX_MERSCOPE"]
    assert set(record.runs) == {
        "whb_frontal_supc_clus",
        "whb_frontal_supc_clus_setc",
        "seaad_mr_panel",
    }
    assert record.runs["whb_frontal_supc_clus_setc"].purposes == ["setc_sensitivity"]
    labels = pd.read_parquet(tmp_path / "out" / str(record.provisional_labels))
    assert "mmc_whb_setc_supercluster_bp" in labels.columns
    assert len(fake_mmc.calls) == 4


def test_a_sample_without_any_run_is_an_error_unless_its_panel_was_refused(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config, _ = _per_platform_setup(
        tmp_path,
        fake_mmc,
        [GENE_IDS[index] for index in (0, 1, 2, 3, 5)],
        [GENE_IDS[index] for index in (0, 1, 2, 4, 5)],
    )
    merscope_only = [run for run in runs if run.panel.platforms == ["MERSCOPE"]]

    with pytest.raises(MapError, match="no MMC run applies to platform XENIUM"):
        annotate_map(
            samples,
            merscope_only,
            config,
            output_dir=tmp_path / "out",
            pair_id="PX",
            segmentation="s",
        )
    manifest = annotate_map(
        samples,
        merscope_only,
        config,
        output_dir=tmp_path / "out2",
        pair_id="PX",
        segmentation="s",
        refused_platforms=["XENIUM"],
    )
    assert manifest.samples["PX_XENIUM"].panel_status == "refused"
    assert manifest.samples["PX_XENIUM"].runs == {}
    assert set(manifest.samples["PX_MERSCOPE"].runs) == {
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    }


def test_refused_platforms_reads_the_panel_reasons() -> None:
    from merxen.annotation.pipeline import refused_platforms

    required = RequiredBundles(
        pair_id="PX",
        segmentation="s",
        species="human",
        panel_mode="per_platform",
        status="ok",
        reasons=["xenium: 12 resolved genes < min_mapped_genes 50"],
        bundles=[],
        n_required=0,
    )
    assert refused_platforms(required) == ["XENIUM"]
    assert (
        refused_platforms(required.model_copy(update={"panel_mode": "intersection"}))
        == []
    )


def test_map_bundles_refuses_a_bundle_built_for_another_panel(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    other = _panel(GENE_IDS[:4])
    bundles = _human_bundles(fake_mmc, other)
    required = RequiredBundles.model_validate_json(
        (panel_dir / REQUIRED_BUNDLES_FILE).read_text()
    )
    keyed = {
        (reference_id, panel.panel_hash): bundle
        for (reference_id, _), bundle in bundles.items()
    }

    with pytest.raises(MapError, match="is for panel"):
        map_bundles(required, panel_dir, keyed, _config())


def test_map_bundles_refuses_a_panel_file_with_another_hash(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    panel_dir, panel = _panel_dir(tmp_path / "panel")
    bundles = _human_bundles(fake_mmc, panel)
    required = RequiredBundles.model_validate_json(
        (panel_dir / REQUIRED_BUNDLES_FILE).read_text()
    )
    _panel(GENE_IDS[:4]).write(panel_dir / PANEL_GENES_FILE)

    with pytest.raises(MapError, match="has panel hash"):
        map_bundles(required, panel_dir, bundles, _config())


# --------------------------------------------------------------------------
# Published-output reuse: the key and a manifest that cannot serve it


def _published_once(
    tmp_path: Path, fake_mmc: FakeMmc, **config_updates: Any
) -> tuple[list[MapSample], list[Any], AnnotationConfig, Path, int]:
    samples, runs, config = _setup(tmp_path, fake_mmc)
    if config_updates:
        config = config.model_copy(update=config_updates)
    output = tmp_path / "published"
    annotate_map(
        samples, runs, config, output_dir=output, pair_id="PX", segmentation="s"
    )
    return samples, runs, config, output, len(fake_mmc.calls)


def test_reuse_needs_the_same_bundle_build(tmp_path: Path, fake_mmc: FakeMmc) -> None:
    import dataclasses

    samples, runs, config, published, n_calls = _published_once(tmp_path, fake_mmc)
    rebuilt_dir = fake_mmc.bundle(
        "whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=runs[0].panel.panel_hash,
        n_genes=runs[0].panel.n_genes,
        levels=WHB_LEVELS,
        nodes=WHB_NODES,
        build_hash="b" * 64,
    )
    rebuilt = [
        dataclasses.replace(run, bundle=MmcBundle.from_dir(rebuilt_dir))
        if run.reference_id == "whb_frontal_supc_clus"
        else run
        for run in runs
    ]

    manifest = annotate_map(
        samples,
        rebuilt,
        config,
        output_dir=tmp_path / "rerun",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )

    assert len(fake_mmc.calls) == n_calls + 2
    for record in manifest.samples.values():
        assert not record.runs["whb_frontal_supc_clus"].reused
        assert record.runs["whb_frontal_supc_clus"].build_hash == "b" * 64
        assert record.runs["seaad_mr_panel"].reused


def test_reuse_needs_the_same_ctm_version(
    tmp_path: Path, fake_mmc: FakeMmc, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, runs, _, published, n_calls = _published_once(tmp_path, fake_mmc)
    fake_mmc.install(monkeypatch, "1.8.0")

    manifest = annotate_map(
        samples,
        runs,
        _config(ctm_version="1.8.0"),
        output_dir=tmp_path / "rerun",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )

    assert len(fake_mmc.calls) == n_calls + 4
    assert not any(
        run.reused
        for record in manifest.samples.values()
        for run in record.runs.values()
    )


def _edit_manifest(path: Path, edit: Any) -> None:
    payload = json.loads(path.read_text())
    edit(payload)
    path.write_text(json.dumps(payload))


def test_reuse_needs_the_tidy_schema_version_and_the_same_lookup(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config, published, n_calls = _published_once(tmp_path, fake_mmc)

    def edit(payload: dict[str, Any]) -> None:
        runs_m = payload["samples"]["PX_MERSCOPE"]["runs"]
        # A manifest written before the tidy schema version was recorded.
        runs_m["whb_frontal_supc_clus"].pop("tidy_schema_version")
        # A mapping made with another (restricted) lookup.
        runs_m["seaad_mr_panel"]["lookup_sha256"] = "0" * 64

    _edit_manifest(published / MAP_MANIFEST_NAME, edit)

    manifest = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "rerun",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )

    assert len(fake_mmc.calls) == n_calls + 2
    merscope = manifest.samples["PX_MERSCOPE"].runs
    assert not merscope["whb_frontal_supc_clus"].reused
    assert not merscope["seaad_mr_panel"].reused
    assert merscope["whb_frontal_supc_clus"].tidy_schema_version == 1
    assert all(run.reused for run in manifest.samples["PX_XENIUM"].runs.values())


def test_reuse_copies_a_kept_extended_json_or_maps_again(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples, runs, config, published, n_calls = _published_once(tmp_path, fake_mmc)
    keep = config.model_copy(update={"keep_extended_json": True})

    # The published runs kept no JSON: keeping it now means mapping again.
    kept = annotate_map(
        samples,
        runs,
        keep,
        output_dir=tmp_path / "kept",
        pair_id="PX",
        segmentation="s",
        reuse_from=published,
    )
    assert len(fake_mmc.calls) == n_calls + 4
    n_calls = len(fake_mmc.calls)

    again = annotate_map(
        samples,
        runs,
        keep,
        output_dir=tmp_path / "again",
        pair_id="PX",
        segmentation="s",
        reuse_from=tmp_path / "kept",
    )
    assert len(fake_mmc.calls) == n_calls
    for record in again.samples.values():
        for run in record.runs.values():
            assert run.reused
            assert run.extended_json is not None
            assert (tmp_path / "again" / run.extended_json).is_file()
    # Without keep_extended_json a reused record names no JSON it lacks.
    plain = annotate_map(
        samples,
        runs,
        config,
        output_dir=tmp_path / "plain",
        pair_id="PX",
        segmentation="s",
        reuse_from=tmp_path / "kept",
    )
    assert all(
        run.reused and run.extended_json is None
        for record in plain.samples.values()
        for run in record.runs.values()
    )
    assert kept.samples["PX_XENIUM"].runs["seaad_mr_panel"].extended_json


@pytest.mark.parametrize("kind", ["stub", "schema", "garbage"])
def test_an_unusable_published_manifest_only_disables_reuse(
    tmp_path: Path, fake_mmc: FakeMmc, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    samples, runs, config, published, n_calls = _published_once(tmp_path, fake_mmc)
    path = published / MAP_MANIFEST_NAME
    if kind == "stub":
        # What stubMapManifestJson writes on -stub-run (AnnotationReferences).
        path.write_text(
            json.dumps(
                {
                    "stub": True,
                    "pair_id": "PX",
                    "segmentation": "s",
                    "species": "human",
                    "panel_status": "ok",
                    "panel_reasons": [],
                    "n_required": 2,
                    "bundle_refs": ["map_inputs/bundle_refs/bundle_ref_1.json"],
                    "samples": {},
                }
            )
        )
    elif kind == "schema":
        _edit_manifest(path, lambda payload: payload.update(schema_version=99))
    else:
        path.write_text("{not json")

    with caplog.at_level("WARNING"):
        manifest = annotate_map(
            samples,
            runs,
            config,
            output_dir=tmp_path / "rerun",
            pair_id="PX",
            segmentation="s",
            reuse_from=published,
        )

    assert len(fake_mmc.calls) == n_calls + 4
    assert set(manifest.samples) == {"PX_MERSCOPE", "PX_XENIUM"}
    assert "reuse disabled" in caplog.text


def test_cli_annotate_refuses_a_work_dir_in_the_results_tree(tmp_path: Path) -> None:
    inputs = _published_pair(tmp_path / "results")
    work = tmp_path / "results" / "PX" / "scratch"

    result = CliRunner().invoke(
        cli_main,
        [
            "annotate",
            "--species",
            "human",
            *(item for path in inputs for item in ("--from-clustered-h5ad", str(path))),
            "--out",
            str(tmp_path / "shadow"),
            "--work-dir",
            str(work),
        ],
    )

    assert result.exit_code != 0
    assert "work dir" in result.output
    assert "never into the results tree" in result.output
    assert not work.exists()


def test_output_guard_covers_prepared_inputs_and_named_results_roots(
    tmp_path: Path,
) -> None:
    prepared = (
        tmp_path
        / "results/P1/proseg_hybrid/clustering_squidpy/clustering_prepare_out"
        / "merscope/P1_MERSCOPE_prepared.h5ad"
    )
    with pytest.raises(MapError, match="results tree"):
        check_output_outside_inputs(tmp_path / "results" / "P2" / "x", [prepared])
    loose = tmp_path / "staging" / "P1_MERSCOPE_prepared.h5ad"
    check_output_outside_inputs(tmp_path / "results" / "x", [loose])
    with pytest.raises(MapError, match="protected results root"):
        check_output_outside_inputs(
            tmp_path / "results" / "x",
            [loose],
            protected_roots=[tmp_path / "results"],
        )


# --------------------------------------------------------------------------
# Provisional labels: which threshold, which parent, which probability


T_IDS = [f"ENSG{index:011d}" for index in range(101, 111)]
T_SYMBOLS = [f"T{index}" for index in range(10)]
T_WHB_NODES = [
    FakeNode(
        "TW_EXC",
        "Upper-layer intratelencephalic",
        T_IDS[0],
        {"broad_class": "Neurons", "lineage": "Neurons", "nt": "Excitatory"},
    ),
    FakeNode(
        "TW_INH",
        "MGE interneuron",
        T_IDS[1],
        {"broad_class": "Neurons", "lineage": "Neurons", "nt": "Inhibitory"},
    ),
    FakeNode(
        "TW_OLI",
        "Oligodendrocyte",
        T_IDS[2],
        {"broad_class": "Oligodendrocytes", "lineage": "Oligodendrocyte lineage"},
    ),
    FakeNode(
        "TW_OPC",
        "Oligodendrocyte precursor",
        T_IDS[3],
        {
            "broad_class": "Oligodendrocyte precursors",
            "lineage": "Oligodendrocyte lineage",
        },
    ),
    FakeNode(
        "TW_AST",
        "Astrocyte",
        T_IDS[4],
        {"broad_class": "Astrocytes", "lineage": "Astrocytes"},
    ),
    FakeNode(
        "TW_SPL",
        "Splatter",
        T_IDS[9],
        {"broad_class": "Mixed/Unknown", "lineage": "Neurons", "sink": "True"},
    ),
]
# SEA-AD reads its own markers, so its probabilities are set apart from WHB's.
T_SEA_NODES = [
    FakeNode("TS_L23", "L2/3 IT", T_IDS[5], {"broad_class": "Neurons"}),
    FakeNode("TS_AST", "Astrocyte", T_IDS[6], {"broad_class": "Astrocytes"}),
    FakeNode(
        "TS_OLI", "Oligodendrocyte", T_IDS[7], {"broad_class": "Oligodendrocytes"}
    ),
]
# Counts on T0..T9 (T8 is filler: no node's marker). SEA-AD's subclass level
# (depth 1 in the fake) has bp = p ** 2 and aggregate_probability = p ** 3.
T_CELLS: dict[str, list[int]] = {
    # WHB EXC .70 / INH .30: lineage and broad 1.0; NT .70 < .73; the
    # supercluster .70 passes its own .69 (not the broad .73).
    "super70": [70, 30, 0, 0, 0, 10, 0, 0, 0, 0],
    # OLI .71 / OPC .29: lineage 1.0 but broad .71 < .73 (not .69).
    "broad71": [0, 0, 71, 29, 0, 0, 0, 10, 0, 0],
    # A supercluster bp of exactly .69, stored as float32, still passes.
    "super69": [69, 31, 0, 0, 0, 10, 0, 0, 0, 0],
    # SEA-AD p .89: subclass bp .79, aggregate .7031 (E2: aggregate).
    # Below 60 counts the .55 threshold applies; the aggregate passes.
    "sea_below60": [20, 0, 0, 0, 0, 25, 3, 0, 0, 0],
    # SEA-AD p .80: subclass bp .64, aggregate .512: below 60 counts it fails
    # .55 on the aggregate (the bp .64 would pass).
    "sea_agg_fails": [20, 0, 0, 0, 0, 24, 6, 0, 0, 0],
    # The same SEA-AD call from 60 counts passes .45.
    "sea_from60": [40, 0, 0, 0, 0, 24, 6, 0, 0, 0],
    # Exactly 60 counts is "from 60".
    "sea_exactly60": [30, 0, 0, 0, 0, 24, 6, 0, 0, 0],
    # WHB lineage .60: broad parent_unresolved, so SEA-AD (p 1.0) is too.
    "sea_parent": [60, 0, 0, 0, 40, 10, 0, 0, 0, 0],
    # A neuron with a confident lineage (EXC + the Splatter sink's lineage)
    # but broad .70: NT and supercluster are parent_unresolved.
    "nt_parent": [70, 0, 0, 0, 0, 10, 0, 0, 0, 30],
}
T_STATUS = {
    "super70": ["confident", "confident", "low_confidence", "confident", "confident"],
    "broad71": [
        "confident",
        "low_confidence",
        "not_applicable",
        "parent_unresolved",
        "parent_unresolved",
    ],
    "super69": ["confident", "confident", "low_confidence", "confident", "confident"],
    "sea_below60": ["confident"] * 5,
    "sea_agg_fails": ["confident"] * 4 + ["low_confidence"],
    "sea_from60": ["confident"] * 5,
    "sea_exactly60": ["confident"] * 5,
    "sea_parent": [
        "low_confidence",
        "parent_unresolved",
        "parent_unresolved",
        "parent_unresolved",
        "parent_unresolved",
    ],
    "nt_parent": [
        "confident",
        "low_confidence",
        "parent_unresolved",
        "parent_unresolved",
        "parent_unresolved",
    ],
}
HUMAN_STATUS_COLUMNS = [
    "ct_lineage_status",
    "ct_broad_status",
    "ct_nt_status",
    "ct_supercluster_status",
    "ct_seaad_subclass_status",
]


def _threshold_matrix_labels(
    tmp_path: Path, fake_mmc: FakeMmc, config: AnnotationConfig
) -> pd.DataFrame:
    counts = np.array(list(T_CELLS.values()))
    path = _write_h5ad(
        tmp_path / "PT_XENIUM.h5ad",
        counts,
        platform="XENIUM",
        var_names=T_SYMBOLS,
        ensembl_ids=T_IDS,
        source="prepared",
    )
    ids = sorted(T_IDS)
    panel = AnnotationPanel(
        name="intersection",
        kind="intersection",
        species="human",
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=["PT_XENIUM"],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[T_SYMBOLS[T_IDS.index(gene_id)] for gene_id in ids],
    )
    panel_dir = tmp_path / "panel"
    panel.write(panel_dir / PANEL_GENES_FILE)
    required = RequiredBundles(
        pair_id="PT",
        segmentation="s",
        species="human",
        panel_mode="intersection",
        status="ok",
        bundles=[
            RequiredBundle(
                reference_id=reference_id,
                role=role,
                species="human",
                purpose="annotation",
                panel_name="intersection",
                panel_hash=panel.panel_hash,
                panel_file=PANEL_GENES_FILE,
                n_panel_genes=panel.n_genes,
            )
            for reference_id, role in (
                ("whb_frontal_supc_clus", "primary"),
                ("seaad_mr_panel", "secondary"),
            )
        ],
        n_required=2,
    )
    bundles = {
        (reference_id, panel.panel_hash): MmcBundle.from_dir(
            fake_mmc.bundle(
                reference_id,
                role=role,
                species="human",
                panel_hash=panel.panel_hash,
                n_genes=panel.n_genes,
                levels=levels,
                nodes=nodes,
            )
        )
        for reference_id, role, levels, nodes in (
            ("whb_frontal_supc_clus", "primary", WHB_LEVELS, T_WHB_NODES),
            ("seaad_mr_panel", "secondary", SEA_LEVELS, T_SEA_NODES),
        )
    }
    runs = map_bundles(required, panel_dir, bundles, config)
    manifest = annotate_map(
        [MapSample("PT_XENIUM", "XENIUM", path, "prepared")],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="PT",
        segmentation="s",
    )
    labels = pd.read_parquet(
        tmp_path / "out" / str(manifest.samples["PT_XENIUM"].provisional_labels)
    )
    labels.index = pd.Index(list(T_CELLS))
    return labels


@pytest.mark.parametrize("cell", list(T_CELLS))
def test_provisional_statuses_pin_each_threshold_and_parent(
    tmp_path: Path, fake_mmc: FakeMmc, cell: str
) -> None:
    labels = _threshold_matrix_labels(tmp_path, fake_mmc, _config())

    statuses = labels.loc[cell, HUMAN_STATUS_COLUMNS].astype(str).tolist()

    assert statuses == T_STATUS[cell]


def test_provisional_seaad_subclass_reads_the_subclass_level_aggregate(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    labels = _threshold_matrix_labels(tmp_path, fake_mmc, _config())

    # The subclass level, not the class level (names differ by level).
    assert labels.loc["sea_below60", "ct_seaad_subclass_name"] == "L2/3 IT 1"
    assert labels.loc["sea_below60", "ct_seaad_subclass_raw"] == pytest.approx(
        0.89 * 0.79, abs=1e-4
    )
    assert labels.loc["super69", "ct_supercluster_raw"] == pytest.approx(0.69)
    assert labels.loc["super70", "ct_final_level"] == "supercluster"
    assert labels.loc["nt_parent", "ct_final_level"] == "lineage"


def test_provisional_seaad_split_follows_second_vote_below_counts(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    from merxen.annotation.config import AnnotationThresholds

    config = _config(thresholds=AnnotationThresholds(second_vote_below_counts=40))

    labels = _threshold_matrix_labels(tmp_path, fake_mmc, config)

    # 50 counts is "from" a split at 40: the .45 threshold passes .512.
    assert labels.loc["sea_agg_fails", "total_counts"] == 50
    assert labels.loc["sea_agg_fails", "ct_seaad_subclass_status"] == "confident"


M_IDS = [f"ENSMUSG{index:011d}" for index in range(1, 5)]
M_NODES = [
    FakeNode(
        "CL_01",
        "01 IT-ET Glut",
        M_IDS[0],
        {"broad_class": "Neurons", "nt": "Excitatory"},
    ),
    FakeNode(
        "CL_06",
        "06 CTX-CGE GABA",
        M_IDS[1],
        {"broad_class": "Neurons", "nt": "Inhibitory"},
    ),
    FakeNode(
        "CL_30", "30 Astro-Epen", M_IDS[2], {"broad_class": "Astrocytes/Ependymal"}
    ),
]
# Class bp p, subclass bp p ** 2 (the fake's depth 1).
M_CELLS: dict[str, list[int]] = {
    # Class .92 passes .90; subclass .85 passes .80 (not .90).
    "subclass85": [92, 0, 8, 0],
    # A class bp of exactly .90, stored as float32, passes.
    "class90": [90, 0, 10, 0],
    # Class .85 < .90 with broad 1.0 (both neurons): NT and subclass are
    # parent_unresolved (their parent is the class).
    "class85": [85, 15, 0, 0],
    # Broad .60: the class is parent_unresolved, not low_confidence.
    "broad60": [60, 0, 40, 0],
}
M_STATUS = {
    "subclass85": ["confident", "confident", "confident", "confident"],
    "class90": ["confident", "confident", "confident", "confident"],
    "class85": [
        "confident",
        "low_confidence",
        "parent_unresolved",
        "parent_unresolved",
    ],
    "broad60": [
        "low_confidence",
        "parent_unresolved",
        "parent_unresolved",
        "parent_unresolved",
    ],
}


@pytest.mark.parametrize("cell", list(M_CELLS))
def test_mouse_provisional_statuses_pin_each_threshold_and_parent(
    tmp_path: Path, fake_mmc: FakeMmc, cell: str
) -> None:
    levels = [
        "CCN20230722_CLAS",
        "CCN20230722_SUBC",
        "CCN20230722_SUPT",
        "CCN20230722_CLUS",
    ]
    ids = sorted(M_IDS)
    panel = AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="mouse",
        platforms=["MERSCOPE"],
        sample_ids=["MT_MERSCOPE"],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=["Slc17a7", "Gad1", "Aqp4", "Other"],
    )
    panel_dir = tmp_path / "panel"
    panel.write(panel_dir / PANEL_GENES_FILE)
    required = RequiredBundles(
        pair_id="MT",
        segmentation="s",
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
            )
        ],
        n_required=1,
    )
    bundle = MmcBundle.from_dir(
        fake_mmc.bundle(
            "wmb_panel",
            role="primary",
            species="mouse",
            panel_hash=panel.panel_hash,
            n_genes=panel.n_genes,
            levels=levels,
            nodes=M_NODES,
            drop_level="CCN20230722_SUPT",
        )
    )
    # The thresholds only: no region pruning (the region step is tested in
    # test_mouse_region_step_*).
    config = AnnotationConfig(
        species="mouse", mouse_section_regions="none"
    ).coupled_to_clustering(10)
    runs = map_bundles(
        required, panel_dir, {("wmb_panel", panel.panel_hash): bundle}, config
    )
    path = _write_h5ad(
        tmp_path / "MT_MERSCOPE.h5ad",
        np.array(list(M_CELLS.values())),
        platform="MERSCOPE",
        var_names=["Slc17a7", "Gad1", "Aqp4", "Other"],
        ensembl_ids=M_IDS,
        source="prepared",
    )
    manifest = annotate_map(
        [MapSample("MT_MERSCOPE", "MERSCOPE", path, "prepared")],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="MT",
        segmentation="s",
    )
    labels = pd.read_parquet(
        tmp_path / "out" / str(manifest.samples["MT_MERSCOPE"].provisional_labels)
    )
    labels.index = pd.Index(list(M_CELLS))

    statuses = labels.loc[
        cell,
        ["ct_broad_status", "ct_class_status", "ct_nt_status", "ct_subclass_status"],
    ].astype(str)

    assert statuses.tolist() == M_STATUS[cell]
