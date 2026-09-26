"""Tests for declared panels, set a / set c and required bundles (plan §3.2, §8)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from scipy import sparse

from merxen.annotation.config import AnnotationConfig, AnnotationPanelConfig
from merxen.annotation.panel import (
    PANEL_GENES_FILE,
    PANEL_GENES_SETC_FILE,
    PANEL_REPORT_FILE,
    REQUIRED_BUNDLES_FILE,
    AnnotationPanel,
    ControlRegistry,
    KnownPanelFamily,
    PanelFile,
    PanelSource,
    PlatformPseudobulk,
    RequiredBundles,
    SharedTissueMask,
    compute_panel,
    compute_panel_hash,
    declared_panel,
    intersection_panel,
    jaccard,
    load_annotation_panel,
    load_fallback_table,
    load_shared_tissue_mask,
    pair_symbol_lookup,
    panel_family,
    panel_from_gene_list,
    platform_pseudobulk,
    raw_panel_from_var,
    read_panel_file,
    required_bundles,
    resolve_panel_mode,
    setc_panel,
)
from merxen.cli import main as cli_main

N_SHARED = 60
H2AFX_ID = "ENSG77700000001"
MERSCOPE_ONLY = {"MONLY1": "ENSG88800000001", "MONLY2": "ENSG88800000002"}
XENIUM_ONLY = {"XONLY1": "ENSG99900000001", "XONLY2": "ENSG99900000002"}


def shared_ids(n: int = N_SHARED, offset: int = 0) -> list[str]:
    return [f"ENSG{index:011d}" for index in range(offset, offset + n)]


def shared_symbols(n: int = N_SHARED, offset: int = 0) -> list[str]:
    return [f"GENE{index}" for index in range(offset, offset + n)]


def write_h5ad(
    path: Path,
    *,
    counts: np.ndarray,
    var_names: list[str],
    platform: str,
    ensembl_ids: list[str] | None = None,
    spatial: np.ndarray | None = None,
    shape_key: str = "MOSAIK_proseg_hybrid",
) -> Path:
    """Write a prepared-style H5AD (counts in X, symbols in var)."""
    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    var["gene"] = var_names
    if ensembl_ids is not None:
        var["ensembl_id"] = ensembl_ids
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.astype(np.float32)),
        obs=pd.DataFrame(index=[f"{platform}_{i}" for i in range(counts.shape[0])]),
        var=var,
    )
    adata.obsm["spatial"] = (
        spatial if spatial is not None else np.zeros((counts.shape[0], 2))
    )
    adata.uns["merxen_clustering_squidpy"] = {
        "platform": platform,
        "shape_key": shape_key,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def fallback_csv(tmp_path: Path) -> Path:
    """A local WHB-style gene.csv for the M0e fallback."""
    rows = [("H2AX", H2AFX_ID)] + [(s, i) for s, i in MERSCOPE_ONLY.items()]
    rows += [("AMBIG", "ENSG55500000001"), ("AMBIG", "ENSG55500000002")]
    path = tmp_path / "gene.csv"
    pd.DataFrame(rows, columns=["gene_symbol", "gene_identifier"]).to_csv(
        path, index=False
    )
    return path


def make_pair(
    tmp_path: Path,
    *,
    xenium_factor: dict[int, float] | None = None,
    xenium_low_count_cells: int = 0,
    merscope_extra_genes: dict[str, str] | None = None,
    merscope_native_ids: bool = False,
    merscope_shape_key: str = "MOSAIK_proseg_hybrid_aligned_nonrigid",
    xenium_outside_cells: int = 0,
    n_cells: int = 200,
    n_shared: int = N_SHARED,
    shuffle_xenium_var: bool = False,
    xenium_cells: slice | None = None,
    name: str = "prepared",
) -> Path:
    """Write a two-platform prepared directory with planted platform effects.

    MERSCOPE declares symbols only (plus blanks, H2AX and MERSCOPE-only
    genes); Xenium declares native IDs (plus H2AFX and Xenium-only genes).
    Every table cell holds one count per gene unless ``xenium_factor`` scales
    a shared gene on Xenium.
    """
    root = tmp_path / name
    ids, symbols = shared_ids(n_shared), shared_symbols(n_shared)
    extra = merscope_extra_genes if merscope_extra_genes is not None else MERSCOPE_ONLY
    merscope_genes = symbols + ["H2AX", *extra] + [f"Blank-{i}" for i in range(5)]
    merscope_counts = np.ones((n_cells, len(merscope_genes)))
    merscope_ids = None
    if merscope_native_ids:
        merscope_ids = ids + [H2AFX_ID, *extra.values()] + [""] * 5
    write_h5ad(
        root / "merscope" / "S_M_prepared.h5ad",
        counts=merscope_counts,
        var_names=merscope_genes,
        ensembl_ids=merscope_ids,
        platform="MERSCOPE",
        spatial=np.column_stack([np.full(n_cells, 2.0), np.full(n_cells, 2.0)]),
        shape_key=merscope_shape_key,
    )
    xenium_genes = symbols + ["H2AFX", *XENIUM_ONLY]
    xenium_ids = ids + [H2AFX_ID, *XENIUM_ONLY.values()]
    counts = np.ones((n_cells, len(xenium_genes)))
    for gene, factor in (xenium_factor or {}).items():
        counts[:, gene] = factor
    spatial = np.column_stack([np.full(n_cells, 2.0), np.full(n_cells, 2.0)])
    if xenium_low_count_cells:
        low = np.zeros((xenium_low_count_cells, len(xenium_genes)))
        low[:, 3] = 19
        counts = np.vstack([counts, low])
        spatial = np.vstack([spatial, np.full((xenium_low_count_cells, 2), 2.0)])
    if xenium_outside_cells:
        outside = np.ones((xenium_outside_cells, len(xenium_genes)))
        outside[:, 4] = 200
        counts = np.vstack([counts, outside])
        spatial = np.vstack([spatial, np.full((xenium_outside_cells, 2), 7.0)])
    if xenium_cells is not None:
        counts, spatial = counts[xenium_cells], spatial[xenium_cells]
    order = np.arange(len(xenium_genes))
    if shuffle_xenium_var:
        order = np.random.default_rng(0).permutation(len(xenium_genes))
    write_h5ad(
        root / "xenium" / "S_X_prepared.h5ad",
        counts=counts[:, order],
        var_names=[xenium_genes[i] for i in order],
        ensembl_ids=[xenium_ids[i] for i in order],
        platform="XENIUM",
        spatial=spatial,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "samples": {
                    "S_M": "merscope/S_M_prepared.h5ad",
                    "S_X": "xenium/S_X_prepared.h5ad",
                }
            }
        )
    )
    return root


def human_config(tmp_path: Path, **panel: Any) -> AnnotationConfig:
    return AnnotationConfig(
        species="human",
        panel=AnnotationPanelConfig(
            gene_id_fallback_csv=fallback_csv(tmp_path), **panel
        ),
    )


def clustering_config(
    segmentation: str = "proseg_hybrid", min_counts: int = 20
) -> dict[str, Any]:
    return {
        "pair_id": "P0001",
        "min_counts": min_counts,
        "samples": [
            {"sample_id": "S_M", "platform": "MERSCOPE", "segmentation": segmentation},
            {"sample_id": "S_X", "platform": "XENIUM", "segmentation": segmentation},
        ],
    }


def make_mask(tmp_path: Path) -> SharedTissueMask:
    """A 20 x 20 px mask whose left half is tissue; 2 px per dataset unit."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :10] = 1
    np.save(tmp_path / "shared_tissue_mask.npy", mask)
    summary = {
        "coordinate_frames": {
            "fixed_platform": "XENIUM",
            "fixed_dataset_to_image_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        }
    }
    (tmp_path / "registration_summary.json").write_text(json.dumps(summary))
    return load_shared_tissue_mask(
        tmp_path / "shared_tissue_mask.npy", tmp_path / "registration_summary.json"
    )


# --------------------------------------------------------------------------
# panel_hash


def test_panel_hash_is_the_sha256_of_the_sorted_ids() -> None:
    ids = ["ENSG00000000002", "ENSG00000000001"]
    expected = hashlib.sha256(b"ENSG00000000001\nENSG00000000002").hexdigest()
    assert compute_panel_hash(ids) == expected
    assert compute_panel_hash(ids + ids[:1]) == expected
    assert compute_panel_hash([*ids, "ENSG00000000003"]) != expected


def test_annotation_panel_rejects_a_tampered_hash(tmp_path: Path) -> None:
    ids = shared_ids(3)
    panel = AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="human",
        platforms=["XENIUM"],
        sample_ids=["S_X"],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(ids),
        n_genes=3,
        ensembl_ids=ids,
        symbols=shared_symbols(3),
    )
    path = panel.write(tmp_path / "panel.json")
    assert load_annotation_panel(path) == panel
    payload = json.loads(path.read_text())
    payload["ensembl_ids"] = shared_ids(3, offset=1)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="panel_hash"):
        load_annotation_panel(path)


def test_panel_hash_ignores_cells_zero_counts_and_var_order(tmp_path: Path) -> None:
    config = human_config(tmp_path)
    reference = compute_panel(
        make_pair(tmp_path),
        "human",
        output_dir=tmp_path / "out_a",
        config=config,
        clustering_config=clustering_config(),
    )
    # Fewer cells, a zero-count column and another var order: the declared
    # panel is the same, so every hash is the same.
    varied = compute_panel(
        make_pair(
            tmp_path,
            name="varied",
            shuffle_xenium_var=True,
            xenium_factor={10: 0.0},
            xenium_cells=slice(0, 50),
        ),
        "human",
        output_dir=tmp_path / "out_b",
        config=config,
        clustering_config=clustering_config(),
    )
    for key in ("S_M", "S_X"):
        assert (
            reference.report["declared_panels"][key]["panel_hash"]
            == varied.report["declared_panels"][key]["panel_hash"]
        )
    assert (
        reference.panels["intersection"].panel.panel_hash
        == varied.panels["intersection"].panel.panel_hash
    )
    grown = compute_panel(
        make_pair(tmp_path, name="grown", n_shared=N_SHARED + 1),
        "human",
        output_dir=tmp_path / "out_c",
        config=config,
        clustering_config=clustering_config(),
    )
    assert (
        grown.panels["intersection"].panel.panel_hash
        != reference.panels["intersection"].panel.panel_hash
    )


# --------------------------------------------------------------------------
# Controls and ID resolution


def test_control_registry_keeps_only_gene_expression_feature_types() -> None:
    registry = ControlRegistry()
    types = {
        "GENE1": "Gene Expression",
        "NegControlProbe_00001": "Negative Control Probe",
        "NegControlCodeword_0500": "Negative Control Codeword",
        "Intergenic_Region_1": "Genomic Control",
        "UnassignedCodeword_0001": "Unassigned Codeword",
        "DeprecatedCodeword_0001": "Deprecated Codeword",
        "antisense_GENE2": "Gene Expression",
    }
    reasons = {
        name: registry.control_reason(name, platform="XENIUM", feature_type=kind)
        for name, kind in types.items()
    }
    assert reasons["GENE1"] is None
    # The feature type decides over the name rules (shared registry).
    assert reasons["antisense_GENE2"] is None
    assert reasons["NegControlProbe_00001"] == "feature_type_negative_control_probe"
    assert reasons["Intergenic_Region_1"] == "feature_type_genomic_control"
    assert all(
        reason is not None
        for name, reason in reasons.items()
        if name not in {"GENE1", "antisense_GENE2"}
    )


def test_control_registry_name_rules() -> None:
    registry = ControlRegistry(extra_patterns=(re.compile("^Reporter"),))
    assert registry.control_reason("Blank-12", platform="MERSCOPE") == "name_pattern"
    assert registry.control_reason("NegControlProbe_1", platform=None) == "name_pattern"
    assert registry.control_reason("ReporterGFP", platform="XENIUM") == "extra_pattern"
    # A real gene containing a control token is kept when it has an ID.
    assert registry.control_reason("BLANKET1", platform="MERSCOPE") == "control_token"
    assert (
        registry.control_reason("BLANKET1", platform="MERSCOPE", has_native_id=True)
        is None
    )
    assert registry.control_reason("GFAP", platform="MERSCOPE") is None


def test_declared_panel_resolution_order(tmp_path: Path) -> None:
    var = pd.DataFrame(
        {
            "gene": ["GENE0", "GENE1", "H2AX", "MONLY1", "AMBIG", "NOTHING", "Blank-1"],
            "ensembl_id": ["ENSG00000000000.5", "", "", "", "", "", ""],
        },
        index=["GENE0", "GENE1", "H2AX", "MONLY1", "AMBIG", "NOTHING", "Blank-1"],
    )
    raw = raw_panel_from_var(var, source=PanelSource(kind="h5ad_var"))
    panel = declared_panel(
        raw,
        species="human",
        platform="MERSCOPE",
        sample_id="S_M",
        pair_lookup={"GENE1": "ENSG00000000001"},
        fallback=load_fallback_table(fallback_csv(tmp_path), "human"),
    )
    assert panel.id_sources == {
        "ENSG00000000000": "native",
        "ENSG00000000001": "pair_lookup",
        H2AFX_ID: "fallback_table",
        MERSCOPE_ONLY["MONLY1"]: "fallback_table",
    }
    assert panel.unresolved == {
        "AMBIG": "ambiguous_in_fallback_table",
        "NOTHING": "not_in_fallback_table",
    }
    assert panel.controls_removed == {"name_pattern": ["Blank-1"]}
    assert panel.n_non_control == 6
    assert panel.resolution_share == pytest.approx(4 / 6)
    no_table = declared_panel(raw, species="human", platform="MERSCOPE")
    assert no_table.unresolved["H2AX"] == "no_fallback_table"


def test_declared_panel_merges_duplicates_and_flags_other_species() -> None:
    var = pd.DataFrame(
        {
            "gene": ["A", "A_dup", "Gfap"],
            "gene_ids": ["ENSG00000000001", "ENSG00000000001", "ENSMUSG00000020932"],
        },
        index=["A", "A_dup", "Gfap"],
    )
    panel = declared_panel(
        raw_panel_from_var(var, source=PanelSource(kind="h5ad_var")),
        species="human",
        platform="XENIUM",
    )
    assert panel.ensembl_ids == ["ENSG00000000001"]
    assert panel.merged_duplicates == {"ENSG00000000001": ["A", "A_dup"]}
    assert panel.unresolved == {"Gfap": "other_species_id"}
    assert panel.other_species_ids == ["ENSMUSG00000020932"]


def test_read_xenium_gene_panel_json(tmp_path: Path) -> None:
    targets = [
        {
            "type": {
                "data": {"id": f"ENSG{i:011d}", "name": f"GENE{i}"},
                "descriptor": "gene",
            }
        }
        for i in range(3)
    ] + [
        {
            "type": {
                "data": {"name": "NegControlProbe_00002"},
                "descriptor": "negative_control",
            }
        }
    ]
    path = tmp_path / "gene_panel.json"
    path.write_text(json.dumps({"metadata": {}, "payload": {"targets": targets}}))
    raw = read_panel_file(path)
    assert raw.source.kind == "xenium_gene_panel_json"
    assert raw.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    panel = declared_panel(raw, species="human", platform="XENIUM")
    assert panel.ensembl_ids == shared_ids(3)
    assert panel.controls_removed == {
        "feature_type_negative_control_probe": ["NegControlProbe_00002"]
    }


def test_read_merscope_codebook(tmp_path: Path) -> None:
    path = tmp_path / "codebook_0_panel.csv"
    path.write_text(
        "# chemistryVersion: Merfish 2.0\n"
        "name,id,barcodeType,V0001\n"
        "GENE0,ENST00000000001,merfish,1\n"
        "GENE1,ENST00000000002,merfish,0\n"
        "Blank-0,-1,merfish,1\n"
    )
    raw = read_panel_file(path)
    assert raw.source.kind == "merscope_codebook"
    # Transcript IDs are not gene IDs: the pair lookup resolves the symbols.
    assert list(raw.features["native_id"]) == ["", "", ""]
    panel = declared_panel(
        raw,
        species="human",
        platform="MERSCOPE",
        pair_lookup={"GENE0": "ENSG00000000000", "GENE1": "ENSG00000000001"},
    )
    assert panel.n_genes == 2
    assert panel.controls_removed == {"name_pattern": ["Blank-0"]}


def test_read_gene_table(tmp_path: Path) -> None:
    path = tmp_path / "genes_a.csv"
    pd.DataFrame({"gene": shared_symbols(3), "ensembl_id": shared_ids(3)}).to_csv(
        path, index=False
    )
    panel = declared_panel(read_panel_file(path), species="human", platform=None)
    assert panel.ensembl_ids == shared_ids(3)
    unsupported = tmp_path / "panel.bin"
    unsupported.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported"):
        read_panel_file(unsupported)


def test_pair_lookup_skips_symbols_with_conflicting_ids() -> None:
    first = raw_panel_from_var(
        pd.DataFrame(
            {"gene": ["A", "B"], "gene_ids": ["ENSG00000000001", "ENSG00000000002"]},
            index=["A", "B"],
        ),
        source=PanelSource(kind="h5ad_var"),
    )
    second = raw_panel_from_var(
        pd.DataFrame({"gene": ["B"], "gene_ids": ["ENSG00000000009"]}, index=["B"]),
        source=PanelSource(kind="h5ad_var"),
    )
    assert pair_symbol_lookup([first, second], "human") == {"A": "ENSG00000000001"}
    assert pair_symbol_lookup([first], "mouse") == {}


# --------------------------------------------------------------------------
# Panel modes and families


def _declared(ids: list[str], platform: str) -> Any:
    var = pd.DataFrame({"gene": ids, "gene_ids": ids}, index=ids)
    return declared_panel(
        raw_panel_from_var(var, source=PanelSource(kind="h5ad_var")),
        species="human",
        platform=platform,
        sample_id=f"S_{platform[0]}",
    )


def test_resolve_panel_mode() -> None:
    same = shared_ids(100)
    close = shared_ids(95)
    far = shared_ids(50) + shared_ids(50, offset=500)
    assert resolve_panel_mode([_declared(same, "XENIUM")]) == ("single_sample", None)
    mode, score = resolve_panel_mode(
        [_declared(same, "MERSCOPE"), _declared(close, "XENIUM")]
    )
    assert (mode, score) == ("intersection", pytest.approx(0.95))
    mode, _ = resolve_panel_mode(
        [_declared(same, "MERSCOPE"), _declared(far, "XENIUM")]
    )
    assert mode == "per_platform"
    mode, _ = resolve_panel_mode(
        [_declared(same, "MERSCOPE"), _declared(far, "XENIUM")],
        requested="intersection",
    )
    assert mode == "intersection"
    with pytest.raises(ValueError):
        resolve_panel_mode([])


def test_panel_family_inheritance() -> None:
    ids = shared_ids(100)
    own = panel_family(ids, species="human", platforms=["XENIUM"])
    assert own.basis == "own"
    assert own.family_id == f"human_xenium_{compute_panel_hash(ids)[:12]}"
    family = KnownPanelFamily(
        family_id="human_set_a",
        species="human",
        platforms=frozenset({"XENIUM"}),
        panel_hash=compute_panel_hash(shared_ids(98)),
        ensembl_ids=frozenset(shared_ids(98)),
        root_markers=frozenset(shared_ids(5)),
    )
    inherited = panel_family(
        ids, species="human", platforms=["XENIUM"], known_families=[family]
    )
    assert inherited.basis == "inherited" and inherited.family_id == "human_set_a"
    # Another platform, a missing root marker or a low Jaccard never inherits.
    assert (
        panel_family(
            ids, species="human", platforms=["MERSCOPE"], known_families=[family]
        ).basis
        == "own"
    )
    assert (
        panel_family(
            ids[5:], species="human", platforms=["XENIUM"], known_families=[family]
        ).basis
        == "own"
    )
    assert (
        panel_family(
            ids[:80], species="human", platforms=["XENIUM"], known_families=[family]
        ).basis
        == "own"
    )
    assert jaccard([], []) == 0.0


# --------------------------------------------------------------------------
# Set a and set c


def test_intersection_is_set_a_including_h2ax(tmp_path: Path) -> None:
    result = compute_panel(
        make_pair(tmp_path),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    set_a = result.panels["intersection"].panel
    # 60 shared genes + MERSCOPE H2AX = Xenium H2AFX; platform-only genes out.
    assert set_a.n_genes == N_SHARED + 1
    assert H2AFX_ID in set_a.ensembl_ids
    assert not set(MERSCOPE_ONLY.values()) & set(set_a.ensembl_ids)
    assert not set(XENIUM_ONLY.values()) & set(set_a.ensembl_ids)
    position = set_a.ensembl_ids.index(H2AFX_ID)
    assert set_a.symbols[position] == "H2AFX"
    assert set_a.symbols_by_platform["MERSCOPE"][position] == "H2AX"
    assert result.report["cross_platform_id_matches"] == [
        {
            "ensembl_id": H2AFX_ID,
            "merscope": "H2AX",
            "xenium": "H2AFX",
            "merscope_source": "fallback_table",
            "xenium_source": "native",
        }
    ]
    merscope = result.report["declared_panels"]["S_M"]
    assert merscope["n_controls_removed"] == 5
    assert merscope["gene_id_resolution"] == {"fallback_table": 3, "pair_lookup": 60}
    assert result.report["intersection_only_in"] == {
        "merscope": ["MONLY1", "MONLY2"],
        "xenium": ["XONLY1", "XONLY2"],
    }


def test_setc_drops_platform_deviant_genes(tmp_path: Path) -> None:
    result = compute_panel(
        make_pair(tmp_path, xenium_factor={0: 16.0, 1: 16.0, 2: 0.0}),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    set_a = result.panels["intersection"].panel
    setc = result.panels["setc"].panel
    dropped = {"ENSG00000000000", "ENSG00000000001", "ENSG00000000002"}
    assert set(setc.excluded_ids) == dropped
    assert set(setc.ensembl_ids) == set(set_a.ensembl_ids) - dropped
    assert setc.parent_panel_hash == set_a.panel_hash
    assert setc.panel_hash == compute_panel_hash(setc.ensembl_ids)
    report = result.report["setc"]
    assert report["n_set_a"] == N_SHARED + 1 and report["n_setc"] == N_SHARED - 2
    assert report["excluded"]["ENSG00000000000"] == pytest.approx(
        np.log2(16.001 / 1.001), rel=1e-6
    )
    assert report["mask_applied"] is False
    assert report["mask_reason"] == "no_shared_tissue_mask"
    assert (tmp_path / "out" / PANEL_GENES_SETC_FILE).is_file()


def test_setc_centres_on_the_pair_median(tmp_path: Path) -> None:
    # Xenium reads every gene 2x deeper; only the planted 32x gene deviates
    # from the pair median by more than 2 log2 units.
    factors = {gene: 2.0 for gene in range(N_SHARED)}
    factors[5] = 32.0
    result = compute_panel(
        make_pair(tmp_path, xenium_factor=factors),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    assert result.panels["setc"].panel.excluded_ids == ["ENSG00000000005"]
    assert result.report["setc"]["pair_median_log2_ratio"] == pytest.approx(
        np.log2(2.001 / 1.001), rel=1e-6
    )


def test_setc_uses_table_cells_only(tmp_path: Path) -> None:
    prepared = make_pair(tmp_path, xenium_low_count_cells=400)
    config = human_config(tmp_path)
    with_threshold = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "a",
        config=config,
        clustering_config=clustering_config(min_counts=20),
    )
    assert with_threshold.panels["setc"].panel.excluded_ids == []
    pseudobulk = with_threshold.report["setc"]["pseudobulk"]["XENIUM"]
    assert pseudobulk["n_objects"] == 600 and pseudobulk["n_table_cells"] == 200
    without_threshold = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "b",
        config=config,
        clustering_config=clustering_config(min_counts=0),
    )
    assert without_threshold.panels["setc"].panel.excluded_ids == ["ENSG00000000003"]


def test_setc_restricts_to_the_shared_tissue_mask(tmp_path: Path) -> None:
    mask = make_mask(tmp_path)
    prepared = make_pair(tmp_path, xenium_outside_cells=50)
    config = human_config(tmp_path)
    masked = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "masked",
        config=config,
        clustering_config=clustering_config(),
        shared_mask=mask,
    )
    assert masked.report["setc"]["mask_applied"] is True
    assert masked.report["setc"]["pseudobulk"]["XENIUM"]["n_cells_used"] == 200
    assert masked.panels["setc"].panel.excluded_ids == []
    whole = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "whole",
        config=config,
        clustering_config=clustering_config(),
    )
    assert whole.panels["setc"].panel.excluded_ids == ["ENSG00000000004"]


def test_setc_skips_the_mask_when_merscope_is_not_aligned(tmp_path: Path) -> None:
    mask = make_mask(tmp_path)
    prepared = make_pair(
        tmp_path, xenium_outside_cells=50, merscope_shape_key="MOSAIK_proseg_hybrid"
    )
    result = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
        shared_mask=mask,
    )
    assert result.report["setc"]["mask_applied"] is False
    assert "not in the XENIUM frame" in result.report["setc"]["mask_reason"]


def test_shared_tissue_mask_lookup(tmp_path: Path) -> None:
    mask = make_mask(tmp_path)
    points = np.array([[2.0, 2.0], [7.0, 2.0], [-1.0, 2.0], [2.0, 30.0], [4.9, 9.9]])
    assert mask.contains(points, block_rows=3).tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError, match=".npy"):
        load_shared_tissue_mask(
            tmp_path / "mask.tif", tmp_path / "registration_summary.json"
        )


def test_platform_pseudobulk_excludes_controls_from_the_totals(
    tmp_path: Path,
) -> None:
    prepared = make_pair(tmp_path)
    h5ad = prepared / "merscope" / "S_M_prepared.h5ad"
    merscope_panel = declared_panel(
        raw_panel_from_var(ad.read_h5ad(h5ad).var, source=PanelSource(kind="h5ad_var")),
        species="human",
        platform="MERSCOPE",
        pair_lookup=dict(zip(shared_symbols(), shared_ids(), strict=True)),
    )
    # 63 gene counts + 5 blank counts per cell: blanks never count towards
    # the table-cell threshold.
    kept = platform_pseudobulk(
        h5ad, declared=merscope_panel, min_counts=63, block_rows=7
    )
    assert kept.n_objects == kept.n_table_cells == kept.n_cells_used == 200
    assert kept.mean_counts["ENSG00000000000"] == pytest.approx(1.0)
    assert "" not in kept.mean_counts
    none = platform_pseudobulk(h5ad, declared=merscope_panel, min_counts=64)
    assert none.n_table_cells == 0


@pytest.mark.parametrize("layout", ["csc", "dense", "counts_layer"])
def test_platform_pseudobulk_reads_other_count_layouts(
    tmp_path: Path, layout: str
) -> None:
    counts = np.ones((30, 4))
    counts[:, 1] = 3.0
    names = ["GENE0", "GENE1", "GENE2", "Blank-1"]
    var = pd.DataFrame({"gene": names, "ensembl_id": [*shared_ids(3), ""]}, index=names)
    matrix: Any = sparse.csc_matrix(counts) if layout == "csc" else counts
    adata = ad.AnnData(X=matrix, var=var)
    if layout == "counts_layer":
        adata.layers["counts"] = sparse.csr_matrix(counts)
        adata.X = np.zeros_like(counts)
    path = tmp_path / "sample.h5ad"
    adata.write_h5ad(path)
    declared = declared_panel(
        raw_panel_from_var(var, source=PanelSource(kind="h5ad_var")),
        species="human",
        platform="XENIUM",
    )
    result = platform_pseudobulk(path, declared=declared, min_counts=5, block_rows=8)
    assert result.n_table_cells == 30
    assert result.mean_counts == {
        "ENSG00000000000": 1.0,
        "ENSG00000000001": 3.0,
        "ENSG00000000002": 1.0,
    }


def test_setc_panel_requires_both_platforms() -> None:
    ids = shared_ids(3)
    set_a = AnnotationPanel(
        name="intersection",
        kind="intersection",
        species="human",
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=[],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ids),
        n_genes=3,
        ensembl_ids=ids,
        symbols=shared_symbols(3),
    )
    empty = PlatformPseudobulk(
        sample_id="S_X",
        platform="XENIUM",
        n_objects=0,
        n_table_cells=0,
        n_cells_used=0,
        mask_applied=False,
        mean_counts={},
    )
    with pytest.raises(ValueError, match="MERSCOPE"):
        setc_panel(set_a, {"XENIUM": empty})
    with pytest.raises(ValueError, match="no MERSCOPE table cells"):
        setc_panel(set_a, {"XENIUM": empty, "MERSCOPE": empty})


# --------------------------------------------------------------------------
# Required bundles


def test_same_panel_human_pair_needs_set_a_and_set_c_bundles(tmp_path: Path) -> None:
    result = compute_panel(
        make_pair(tmp_path, xenium_factor={0: 16.0}),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config("proseg_hybrid"),
    )
    required = RequiredBundles.model_validate_json(
        (tmp_path / "out" / REQUIRED_BUNDLES_FILE).read_text()
    )
    assert required.status == "ok" and required.panel_mode == "intersection"
    assert [(b.reference_id, b.purpose, b.panel_file) for b in required.bundles] == [
        ("whb_frontal_supc_clus", "annotation", PANEL_GENES_FILE),
        ("whb_frontal_supc_clus", "setc_sensitivity", PANEL_GENES_SETC_FILE),
        ("seaad_mr_panel", "annotation", PANEL_GENES_FILE),
    ]
    assert required.n_required == 3
    assert required == result.required
    written = load_annotation_panel(tmp_path / "out" / PANEL_GENES_FILE)
    assert required.bundles[0].panel_hash == written.panel_hash
    setc = load_annotation_panel(tmp_path / "out" / PANEL_GENES_SETC_FILE)
    assert [b.n_panel_genes for b in required.bundles] == [
        written.n_genes,
        setc.n_genes,
        written.n_genes,
    ]
    report = json.loads((tmp_path / "out" / PANEL_REPORT_FILE).read_text())
    assert report["panel_mode"] == "intersection" and report["n_required_bundles"] == 3


def test_set_c_bundle_only_for_its_segmentations(tmp_path: Path) -> None:
    prepared = make_pair(tmp_path, xenium_factor={0: 16.0})
    config = human_config(tmp_path)
    other = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "reseg",
        config=config,
        clustering_config=clustering_config("reseg"),
    )
    assert [b.purpose for b in other.required.bundles] == ["annotation", "annotation"]
    off = config.model_copy(update={"xplat_sensitivity": "off"})
    disabled = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "off",
        config=off,
        clustering_config=clustering_config("proseg_hybrid"),
    )
    assert disabled.required.n_required == 2
    # The set-c panel file is still written for the report.
    assert (tmp_path / "off" / PANEL_GENES_SETC_FILE).is_file()


def test_set_c_equal_to_set_a_shares_its_bundle(tmp_path: Path) -> None:
    result = compute_panel(
        make_pair(tmp_path),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    assert result.panels["setc"].panel.panel_hash == (
        result.panels["intersection"].panel.panel_hash
    )
    assert result.required.n_required == 2


def test_per_platform_pair_needs_five_bundles(tmp_path: Path) -> None:
    extra = {f"MEXTRA{i}": f"ENSG66600{i:06d}" for i in range(40)}
    result = compute_panel(
        make_pair(tmp_path, merscope_extra_genes=extra, merscope_native_ids=True),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    assert result.report["panel_mode"] == "per_platform"
    assert result.report["pair_jaccard"] < 0.9
    assert "setc" not in result.panels
    keys = {(b.reference_id, b.panel_name, b.purpose) for b in result.required.bundles}
    assert keys == {
        ("whb_frontal_supc_clus", "merscope", "annotation"),
        ("whb_frontal_supc_clus", "xenium", "annotation"),
        ("seaad_mr_panel", "merscope", "annotation"),
        ("seaad_mr_panel", "xenium", "annotation"),
        ("whb_frontal_supc_clus", "intersection", "intersection_xpanel"),
    }
    assert result.required.n_required == 5
    for name in ("merscope", "xenium", "intersection"):
        assert (tmp_path / "out" / f"panel_genes_{name}.json").is_file()
    assert result.report["xplat_broad_only"] is True  # 61 genes < 100


def test_single_platform_mouse_sample(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    ids = [f"ENSMUSG{i:011d}" for i in range(60)]
    write_h5ad(
        root / "merscope" / "VZG2_prepared.h5ad",
        counts=np.ones((30, 61)),
        var_names=[f"Gene{i}" for i in range(60)] + ["Blank-1"],
        ensembl_ids=[*ids, ""],
        platform="MERSCOPE",
    )
    (root / "manifest.json").write_text(
        json.dumps({"samples": {"VZG2": "merscope/VZG2_prepared.h5ad"}})
    )
    result = compute_panel(
        root, "mouse", output_dir=tmp_path / "out", segmentation="original_seg"
    )
    assert result.report["panel_mode"] == "single_sample"
    assert [
        (b.reference_id, b.panel_hash is None) for b in result.required.bundles
    ] == [
        ("wmb_panel", False),
        ("wmb_region_share", True),
    ]
    panel = load_annotation_panel(tmp_path / "out" / PANEL_GENES_FILE)
    assert panel.kind == "single_sample" and panel.n_genes == 60
    assert panel.panel_family is not None
    assert panel.panel_family.family_id.startswith("mouse_merscope_")


def test_too_small_panels_are_refused_without_bundles(tmp_path: Path) -> None:
    result = compute_panel(
        make_pair(tmp_path, n_shared=20),
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
    )
    assert result.required.status == "refused"
    assert result.required.bundles == [] and result.required.n_required == 0
    assert result.report["annotation_panels"]["intersection"]["refused"]


def test_required_bundles_ignore_resolvability_references(tmp_path: Path) -> None:
    config = AnnotationConfig(
        species="human",
        references=[
            {
                "reference_id": "whb_frontal_supc_clus",
                "species": "human",
                "role": "primary",
            },
            {
                "reference_id": "whb_frontal_supc_clus_ho",
                "species": "human",
                "role": "resolvability",
            },
        ],
    )
    ids = shared_ids(60)
    panel = AnnotationPanel(
        name="sample",
        kind="single_sample",
        species="human",
        platforms=["XENIUM"],
        sample_ids=["S_X"],
        panel_mode="single_sample",
        panel_hash=compute_panel_hash(ids),
        n_genes=60,
        ensembl_ids=ids,
        symbols=shared_symbols(),
    )
    bundles = required_bundles(
        {"sample": PanelFile(panel=panel, file_name=PANEL_GENES_FILE)},
        references=config.references,
        species="human",
        panel_mode="single_sample",
    )
    assert [bundle.reference_id for bundle in bundles] == ["whb_frontal_supc_clus"]


def test_declared_panel_from_a_vendor_file_wins_over_var(tmp_path: Path) -> None:
    prepared = make_pair(tmp_path)
    ids = [*shared_ids(), H2AFX_ID, *XENIUM_ONLY.values(), "ENSG12300000001"]
    names = [*shared_symbols(), "H2AFX", *XENIUM_ONLY, "ZERODETECT"]
    targets = [
        {"type": {"data": {"id": i, "name": n}, "descriptor": "gene"}}
        for i, n in zip(ids, names, strict=True)
    ]
    panel_json = tmp_path / "gene_panel.json"
    panel_json.write_text(json.dumps({"payload": {"targets": targets}}))
    result = compute_panel(
        prepared,
        "human",
        output_dir=tmp_path / "out",
        config=human_config(tmp_path),
        clustering_config=clustering_config(),
        panel_files={"XENIUM": panel_json},
    )
    xenium = result.report["declared_panels"]["S_X"]
    assert xenium["source"]["kind"] == "xenium_gene_panel_json"
    # The zero-detection probe absent from var is part of the declared panel.
    assert xenium["n_genes"] == N_SHARED + 1 + len(XENIUM_ONLY) + 1


def test_panel_from_gene_list(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"gene": shared_symbols(), "ensembl_id": shared_ids()}).to_csv(
        path, index=False
    )
    result = panel_from_gene_list(path, "human", output_dir=tmp_path / "out")
    assert result.required.status == "ok"
    assert [b.reference_id for b in result.required.bundles] == [
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    ]
    assert (
        load_annotation_panel(tmp_path / "out" / PANEL_GENES_FILE).kind == "gene_list"
    )


def test_cli_annotation_panel(tmp_path: Path) -> None:
    prepared = make_pair(tmp_path, xenium_factor={0: 16.0})
    config_path = tmp_path / "annotation_config.json"
    config_path.write_text(human_config(tmp_path).model_dump_json())
    clustering_path = tmp_path / "clustering_squidpy_config.json"
    clustering_path.write_text(json.dumps(clustering_config()))
    make_mask(tmp_path)
    result = CliRunner().invoke(
        cli_main,
        [
            "annotation-panel",
            "--prepared-dir",
            str(prepared),
            "--species",
            "human",
            "--platforms",
            "MERSCOPE,XENIUM",
            "--output-dir",
            str(tmp_path / "out"),
            "--annotation-config",
            str(config_path),
            "--clustering-config",
            str(clustering_path),
            "--shared-tissue-mask",
            str(tmp_path / "shared_tissue_mask.npy"),
            "--registration-summary",
            str(tmp_path / "registration_summary.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3 required bundle(s)" in result.output
    required = json.loads((tmp_path / "out" / REQUIRED_BUNDLES_FILE).read_text())
    assert (
        required["pair_id"] == "P0001" and required["segmentation"] == "proseg_hybrid"
    )
    report = json.loads((tmp_path / "out" / PANEL_REPORT_FILE).read_text())
    assert report["setc"]["mask_applied"] is True
    assert report["min_counts_source"] == "clustering_config"


def test_cli_annotation_panel_needs_one_input(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_main,
        ["annotation-panel", "--species", "human", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "exactly one of" in result.output


def test_intersection_panel_prefers_xenium_symbols() -> None:
    merscope = _declared(shared_ids(3), "MERSCOPE")
    xenium = _declared(shared_ids(3), "XENIUM")
    panel = intersection_panel([merscope, xenium], panel_mode="intersection")
    assert panel.platforms == ["MERSCOPE", "XENIUM"]
    assert set(panel.declared_panel_hashes) == {"merscope", "xenium"}
