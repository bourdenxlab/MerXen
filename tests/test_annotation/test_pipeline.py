"""Tests for the annotation MAP step and ``merxen annotate`` (plan §3.3, M3)."""

from __future__ import annotations

import json
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


def test_mouse_maps_unpruned_with_the_drop_level(
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

    manifest = annotate_map(
        [MapSample("AG_MERSCOPE", "MERSCOPE", path, "prepared")],
        runs,
        config,
        output_dir=tmp_path / "out",
        pair_id="AG",
        segmentation="proseg_hybrid",
    )

    command = fake_mmc.calls[-1]["command"]
    assert command[command.index("--drop_level") + 1] == "CCN20230722_SUPT"
    assert manifest.mouse_region_step == MOUSE_REGION_STEP
    record = manifest.samples["AG_MERSCOPE"]
    assert set(record.runs) == {"wmb_panel"}
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
