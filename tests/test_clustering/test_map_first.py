"""Tests for the map_first hierarchy (plan §6.2, §4.5, §4.6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from merxen.annotation.pipeline import write_label_table
from merxen.annotation.provenance import (
    PROVENANCE_UNS_KEY,
    AnnotationProvenance,
    annotation_manifest_filename,
    is_safe_key,
)
from merxen.annotation.schema import (
    SUBCLUSTER_STATUSES,
    CellStatus,
    Columns,
    LabelTableError,
    label_table_filename,
)
from merxen.annotation.vocab import (
    MAP_FIRST_BROAD_LABELS,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
)
from merxen.clustering.map_first import (
    BROAD_ATLAS_LABEL_KEY,
    BROAD_CLASS_KEY,
    BROAD_N_MARKERS_KEY,
    BROAD_SCORE_KEY,
    BROAD_SCORE_MARGIN_KEY,
    HIERARCHICAL_CLUSTER_KEY,
    HIERARCHICAL_UNS_KEY,
    LEIDEN_PROVENANCE_UNS_KEY,
    NEURON_SPLIT_KEY,
    NEURON_SPLIT_LABELS,
    SUBCLUSTER_LABEL_KEY,
    SUBCLUSTER_STATUS_KEY,
    LeafGate,
    build_hierarchy_columns,
    find_label_table,
    is_unassigned_state,
    label_obs_columns,
    load_label_inputs,
    run_map_first_hierarchy,
    unassigned_states,
)

from .conftest import (
    HUMAN_TYPES,
    MOUSE_TYPES,
    CellType,
    Section,
    broad_only,
    make_config,
    make_provenance,
    make_section,
)

SAMPLE = "P0001_MERSCOPE"


def _run(
    section: Section,
    tmp_path: Path,
    *,
    provenance: AnnotationProvenance | None | str = "section",
    plots: bool = False,
    stability: bool = False,
    **config: Any,
) -> tuple[ad.AnnData, Any]:
    chosen = section.provenance if provenance == "section" else provenance
    return run_map_first_hierarchy(
        section.adata,
        section.labels,
        make_config(tmp_path, **config),
        tmp_path / "hierarchy",
        SAMPLE,
        provenance=chosen,  # type: ignore[arg-type]
        plots=plots,
        stability=stability,
    )


def _type_of(section: Section, clustered: ad.AnnData) -> pd.Series:
    return section.type_of_cell.loc[clustered.obs_names]


def _first(clustered: ad.AnnData, section: Section, key: str, column: str) -> str:
    types = _type_of(section, clustered)
    values = clustered.obs.loc[types.to_numpy() == key, column].astype(str)
    assert values.nunique() == 1, (key, column, values.unique())
    return str(values.iloc[0])


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_hierarchical_cluster_is_categorical_non_null_and_unique(
    species: str, tmp_path: Path
) -> None:
    section = make_section(species)
    clustered, _ = _run(section, tmp_path)
    values = clustered.obs[HIERARCHICAL_CLUSTER_KEY]
    assert isinstance(values.dtype, pd.CategoricalDtype)
    assert not values.isna().any()
    assert not values.astype(str).str.strip().eq("").any()
    categories = [str(category) for category in values.cat.categories]
    assert len(categories) == len(set(categories))
    assert set(categories) == set(values.astype(str))
    pairs = clustered.obs[[Columns.CT_BRANCH, SUBCLUSTER_LABEL_KEY]].astype(str)
    expected = pairs[Columns.CT_BRANCH] + ":" + pairs[SUBCLUSTER_LABEL_KEY]
    assert values.astype(str).equals(expected)
    # One value per (branch, leaf) pair: the mapping is injective.
    per_value = pairs.groupby(values.astype(str).to_numpy()).nunique()
    assert (per_value == 1).all().all()
    branches = {category.split(":", 1)[0] for category in categories}
    assert branches == set(clustered.obs[Columns.CT_BRANCH].astype(str))


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_h5ad_ids_equal_parquet_in_table_ids(species: str, tmp_path: Path) -> None:
    section = make_section(species, n_low=30)
    clustered, _ = _run(section, tmp_path)
    labels = section.labels
    in_table = set(labels.loc[labels[Columns.IN_TABLE], Columns.CELL_ID])
    assert set(clustered.obs_names) == in_table
    assert clustered.n_obs == len(in_table) < section.adata.n_obs
    assert clustered.obs_names.is_unique
    # The input section is not modified.
    assert section.adata.n_obs == len(labels)


def test_label_table_from_another_cell_set_is_refused(tmp_path: Path) -> None:
    section = make_section("human")
    with pytest.raises(LabelTableError, match="in_table"):
        _run(section, tmp_path, min_counts=20)
    dropped = section.labels.iloc[1:].reset_index(drop=True)
    with pytest.raises(LabelTableError):
        run_map_first_hierarchy(
            section.adata,
            dropped,
            make_config(tmp_path),
            tmp_path / "h",
            SAMPLE,
            plots=False,
            stability=False,
        )


def test_neuron_split_label_follows_the_nt_vocabulary(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, _ = _run(section, tmp_path)
    split = clustered.obs[NEURON_SPLIT_KEY]
    assert isinstance(split.dtype, pd.CategoricalDtype)
    assert list(split.cat.categories) == list(NEURON_SPLIT_LABELS)
    expected = {
        "ul_it": "Excitatory",
        "mge": "Inhibitory",
        "neuron_nt_unresolved": UNRESOLVED_LABEL,
        "astro": "not_neuron",
        "oligo": "not_neuron",
        "oligo_lineage": "not_neuron",
        "microglia_small": "not_neuron",
        "mixed": "not_neuron",
    }
    for key, label in expected.items():
        assert _first(clustered, section, key, NEURON_SPLIT_KEY) == label
    # The viewer matches the split case-insensitively.
    lowered = {value.lower() for value in split.astype(str)}
    assert {"excitatory", "inhibitory"} <= lowered

    mouse = make_section("mouse")
    mouse_clustered, _ = _run(mouse, tmp_path / "mouse")
    assert _first(mouse_clustered, mouse, "it", NEURON_SPLIT_KEY) == "Excitatory"
    assert _first(mouse_clustered, mouse, "cge", NEURON_SPLIT_KEY) == "Inhibitory"
    assert _first(mouse_clustered, mouse, "astro", NEURON_SPLIT_KEY) == "not_neuron"


def test_human_hierarchy_columns_per_branch_kind(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, result = _run(section, tmp_path)
    expectations = {
        # key: (broad_class, hierarchical_cluster, subcluster_status, atlas)
        "ul_it": (
            "Neurons",
            "Neurons/Excitatory:Upper-layer intratelencephalic",
            "mapped_supercluster",
            "Upper-layer intratelencephalic",
        ),
        "neuron_nt_unresolved": (
            "Neurons",
            "Neurons/unresolved:unresolved",
            "unresolved",
            "Neurons",
        ),
        "astro_not_resolvable": (
            "Astrocytes",
            "Astrocytes:unresolved",
            "not_resolvable_panel",
            "Astrocytes",
        ),
        "oligo_lineage": (
            "Oligodendrocyte lineage",
            "Oligodendrocyte lineage/unresolved:unresolved",
            "unresolved",
            "Oligodendrocyte lineage",
        ),
        "mixed": (
            UNASSIGNED_LABEL,
            "Mixed/Unknown:unresolved",
            "unassigned",
            UNASSIGNED_LABEL,
        ),
        # A human branch under min_branch_cells keeps its leaves (§6.2).
        "microglia_small": (
            "Microglia",
            "Microglia:Microglia",
            "mapped_supercluster",
            "Microglia",
        ),
    }
    for key, (broad, cluster, status, atlas) in expectations.items():
        assert _first(clustered, section, key, BROAD_CLASS_KEY) == broad
        assert _first(clustered, section, key, HIERARCHICAL_CLUSTER_KEY) == cluster
        assert _first(clustered, section, key, SUBCLUSTER_STATUS_KEY) == status
        assert _first(clustered, section, key, BROAD_ATLAS_LABEL_KEY) == atlas
    obs = clustered.obs
    assert list(obs[BROAD_CLASS_KEY].cat.categories) == list(
        MAP_FIRST_BROAD_LABELS["human"]
    )
    assert list(obs[SUBCLUSTER_STATUS_KEY].cat.categories) == list(SUBCLUSTER_STATUSES)
    assert (
        obs[SUBCLUSTER_LABEL_KEY].astype(str).equals(obs[Columns.CT_LEAF].astype(str))
    )
    np.testing.assert_array_equal(
        obs[BROAD_SCORE_KEY].to_numpy(), obs[Columns.level("broad", "conf")].to_numpy()
    )
    np.testing.assert_array_equal(
        obs[BROAD_SCORE_MARGIN_KEY].to_numpy(),
        obs[Columns.level("broad", "margin")].to_numpy(),
    )
    assert (obs[BROAD_N_MARKERS_KEY] == 42).all()
    manifest = result.hierarchy.branch_manifest
    assert manifest["microglia"]["small_branch"] is True
    assert manifest["microglia"]["leaves_kept"] is True
    assert result.hierarchy.n_branch_broad_mismatch == 0
    assert result.hierarchy.n_leaves_suppressed_gate == 0


def test_mouse_classes_under_min_branch_cells_get_unresolved_leaves(
    tmp_path: Path,
) -> None:
    section = make_section("mouse")
    clustered, result = _run(section, tmp_path)
    assert _first(clustered, section, "oligo_small", SUBCLUSTER_LABEL_KEY) == (
        UNRESOLVED_LABEL
    )
    assert _first(clustered, section, "oligo_small", HIERARCHICAL_CLUSTER_KEY) == (
        "31 OPC-Oligo:unresolved"
    )
    assert _first(clustered, section, "oligo_small", SUBCLUSTER_STATUS_KEY) == (
        "unresolved"
    )
    # The parquet's own ct_leaf is kept as written by RESOLVE.
    assert _first(clustered, section, "oligo_small", Columns.CT_LEAF) == "327 Oligo NN"
    assert _first(clustered, section, "it", HIERARCHICAL_CLUSTER_KEY) == (
        "01 IT-ET Glut:007 L2/3 IT CTX Glut"
    )
    assert _first(clustered, section, "it", SUBCLUSTER_STATUS_KEY) == "mapped_subclass"
    assert _first(clustered, section, "astro", BROAD_ATLAS_LABEL_KEY) == "30 Astro-Epen"
    assert result.hierarchy.n_leaves_small_class == 30
    record = result.hierarchy.branch_manifest["31_opc_oligo"]
    assert record["small_branch"] is True
    assert record["leaves_kept"] is False
    # With a smaller threshold the class keeps its leaf.
    kept, _ = _run(section, tmp_path / "small", min_branch_cells=10)
    assert _first(kept, section, "oligo_small", SUBCLUSTER_LABEL_KEY) == "327 Oligo NN"


def _broad_only_section(species: str, *, gate_level: str = "broad_only") -> Section:
    leaf_level = "supercluster" if species == "human" else "subclass"
    base = HUMAN_TYPES if species == "human" else MOUSE_TYPES
    types = [broad_only(cell_type, leaf_level=leaf_level) for cell_type in base]
    return make_section(species, types=types, gate_level=gate_level)


@pytest.mark.parametrize("species", ["human", "mouse"])
@pytest.mark.parametrize("gate_level", ["broad_only", "failed"])
def test_broad_only_datasets_have_unresolved_leaves(
    species: str, gate_level: str, tmp_path: Path
) -> None:
    section = _broad_only_section(species, gate_level=gate_level)
    clustered, result = _run(section, tmp_path)
    obs = clustered.obs
    assert (obs[SUBCLUSTER_LABEL_KEY].astype(str) == UNRESOLVED_LABEL).all()
    assert obs[HIERARCHICAL_CLUSTER_KEY].astype(str).str.endswith(":unresolved").all()
    statuses = set(obs[SUBCLUSTER_STATUS_KEY].astype(str))
    assert statuses == {"dataset_broad_only", "unassigned"}
    assert result.hierarchy.gate.suppresses_leaves
    uns = clustered.uns[HIERARCHICAL_UNS_KEY]
    assert uns["gate_level"] == gate_level
    assert bool(uns["leaves_suppressed"]) is True
    assert uns["n_leaves"] == 0


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_broad_only_gate_suppresses_leaves_even_if_the_parquet_has_them(
    species: str, tmp_path: Path
) -> None:
    section = make_section(species)
    gate = make_provenance(species, gate_level="broad_only")
    clustered, result = _run(section, tmp_path, provenance=gate)
    assert (clustered.obs[SUBCLUSTER_LABEL_KEY].astype(str) == UNRESOLVED_LABEL).all()
    confident_leaves = int(
        (section.labels[Columns.CT_LEAF].astype(str) != UNRESOLVED_LABEL).sum()
    )
    assert result.hierarchy.n_leaves_suppressed_gate == confident_leaves > 0


@pytest.mark.parametrize("trust", ["broad_only", "refused"])
@pytest.mark.parametrize("species", ["human", "mouse"])
def test_broad_only_and_refused_panels_have_unresolved_leaves(
    species: str, trust: str, tmp_path: Path
) -> None:
    section = make_section(species)
    panel = make_provenance(species, panel_trust=trust)
    clustered, result = _run(section, tmp_path, provenance=panel)
    obs = clustered.obs
    assert (obs[SUBCLUSTER_LABEL_KEY].astype(str) == UNRESOLVED_LABEL).all()
    assert set(obs[SUBCLUSTER_STATUS_KEY].astype(str)) <= {
        "dataset_broad_only",
        "unassigned",
    }
    assert clustered.uns[HIERARCHICAL_UNS_KEY]["panel_trust"] == trust
    assert result.hierarchy.gate.panel_trust == trust


def test_provisional_panels_and_full_gates_keep_leaves(tmp_path: Path) -> None:
    section = make_section("human")
    provisional = make_provenance("human", panel_trust="provisional")
    clustered, result = _run(section, tmp_path, provenance=provisional)
    assert not result.hierarchy.gate.suppresses_leaves
    assert (clustered.obs[SUBCLUSTER_LABEL_KEY].astype(str) != UNRESOLVED_LABEL).any()


def test_without_provenance_the_per_cell_gate_status_still_applies(
    tmp_path: Path,
) -> None:
    section = _broad_only_section("human")
    clustered, result = _run(section, tmp_path, provenance=None)
    assert result.hierarchy.gate.gate_level is None
    assert (clustered.obs[SUBCLUSTER_LABEL_KEY].astype(str) == UNRESOLVED_LABEL).all()
    assert "dataset_broad_only" in set(clustered.obs[SUBCLUSTER_STATUS_KEY].astype(str))
    assert PROVENANCE_UNS_KEY not in clustered.uns
    uns = clustered.uns[HIERARCHICAL_UNS_KEY]
    assert uns["gate_level"] == "unknown"
    assert (clustered.obs[BROAD_N_MARKERS_KEY] == -1).all()


def test_label_columns_are_copied_as_in_the_parquet(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, _ = _run(section, tmp_path)
    labels = section.labels.set_index(Columns.CELL_ID).loc[clustered.obs_names]
    copied = label_obs_columns(section.labels.columns)
    assert copied
    assert not [column for column in copied if column.startswith(("soft_", "mmc_"))]
    for column in (
        Columns.DEPTH_BIN,
        Columns.EXCLUDE_HARD,
        Columns.DISCOVERY_CAUTION,
        Columns.CONTAMINATION_SCORE,
        Columns.NEG_COUNTS,
        Columns.FLAG_CONTAMINATED,
        Columns.CT_MENDER_STATE,
        Columns.CT_FINAL_LEVEL,
        Columns.level("supercluster", "status"),
        Columns.level("seaad_subclass", "runner_up"),
    ):
        assert column in copied
    for column in copied:
        got = clustered.obs[column]
        want = labels[column]
        assert got.astype(object).where(got.notna(), None).tolist() == (
            want.astype(object).where(want.notna(), None).tolist()
        ), column
        if isinstance(want.dtype, pd.CategoricalDtype) or want.dtype == object:
            assert isinstance(got.dtype, pd.CategoricalDtype), column
    assert "soft_broad_neurons" not in clustered.obs
    assert "mmc_whb_supercluster_bp" not in clustered.obs
    assert Columns.IN_TABLE not in clustered.obs


def test_uns_records_are_scalars_and_json_strings(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, result = _run(section, tmp_path, stability=True)
    record = clustered.uns[HIERARCHICAL_UNS_KEY]
    for key, value in record.items():
        assert is_safe_key(key), key
        assert isinstance(value, str | bool | int | float | np.generic), (key, value)
    assert record["mode"] == "map_first"
    assert record["branch_level"] == "ct_branch"
    assert record["leaf_level"] == "supercluster"
    assert record["leaf_source"] == "mapped"
    assert record["engine"] == "scanpy-igraph"
    assert record["engine_n_iterations"] == 2
    assert record["engine_seed"] == 0
    branch_manifest = json.loads(record["branch_manifest_json"])
    assert all(is_safe_key(key) for key in branch_manifest)
    for branch in branch_manifest.values():
        assert all(is_safe_key(key) for key in branch["leaves"])
    assert {entry["branch"] for entry in branch_manifest.values()} == set(
        clustered.obs[Columns.CT_BRANCH].astype(str)
    )
    stability = json.loads(record["qc_stability_json"])
    assert len(stability["aris"]) == 5
    assert stability["role"] == "report_only"
    assert record["qc_stability_mean_ari"] == pytest.approx(stability["mean_ari"])
    assert 0.0 <= record["qc_leiden_vs_leaf_ari"] <= 1.0
    provenance = AnnotationProvenance.read_from_uns(clustered.uns)
    assert provenance == section.provenance
    assert clustered.uns[PROVENANCE_UNS_KEY] == section.provenance.to_uns_json()
    leiden = json.loads(clustered.uns[LEIDEN_PROVENANCE_UNS_KEY]["leiden"])
    assert leiden["engine"] == "scanpy-igraph"
    assert leiden["role"] == "qc_only"
    params = clustered.uns["merxen_clustering_params_leiden"]
    assert params["role"] == "qc_only"
    assert params["leiden_resolution"] == 0.5
    assert "normalize_target_sum" not in params  # None is not h5ad-safe
    manifest = json.loads(
        (tmp_path / "hierarchy" / f"{SAMPLE}_hierarchical_manifest.json").read_text()
    )
    assert manifest["summary"]["n_cells"] == clustered.n_obs
    assert manifest["qc_stability"]["n_subsamples"] == 5
    assert manifest["branch_manifest"] == branch_manifest
    assert result.stability is not None
    assert result.qc.n_clusters == clustered.obs["leiden"].nunique()


def test_qc_embedding_and_counts_layer(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, result = _run(section, tmp_path)
    assert {"X_pca", "X_umap", "spatial"} <= set(clustered.obsm)
    assert (
        clustered.obs["leiden"]
        .astype(str)
        .equals(clustered.obs["leiden_broad"].astype(str))
    )
    raw = section.adata[clustered.obs_names, clustered.var_names].X.toarray()
    np.testing.assert_array_equal(clustered.layers["counts"].toarray(), raw)
    np.testing.assert_array_equal(clustered.obs["n_counts"].to_numpy(), raw.sum(axis=1))
    assert result.qc.random_state == 0
    assert result.qc_leiden_vs_leaf_ari > 0.3


def test_config_must_be_map_first_with_mapped_leaves(tmp_path: Path) -> None:
    section = make_section("human")
    legacy = make_config(tmp_path, mode="legacy", table_key_suffix="")
    with pytest.raises(ValueError, match="map_first"):
        run_map_first_hierarchy(
            section.adata, section.labels, legacy, tmp_path / "h", SAMPLE
        )
    denovo = make_config(tmp_path).model_copy(update={"leaf_source": "denovo"})
    with pytest.raises(NotImplementedError, match="v1.1"):
        run_map_first_hierarchy(
            section.adata, section.labels, denovo, tmp_path / "h", SAMPLE
        )


def test_plots_and_tables_are_written(tmp_path: Path) -> None:
    section = make_section("human")
    clustered, result = _run(section, tmp_path, plots=True)
    artifacts = result.artifacts
    for name in (
        f"{SAMPLE}_map_first_umap",
        f"{SAMPLE}_map_first_spatial_broad_class",
        f"{SAMPLE}_map_first_spatial_ct_leaf",
        f"{SAMPLE}_map_first_spatial_ct_final_level",
        f"{SAMPLE}_map_first_spatial_grid_broad_class",
        f"{SAMPLE}_map_first_branch_neurons_excitatory_gene_dotplot",
        f"{SAMPLE}_map_first_branch_neurons_excitatory_umap",
        f"{SAMPLE}_map_first_branch_mixed_unknown_gene_dotplot",
        "map_first_leaves",
        "qc_leiden_crosstab",
        "hierarchical_manifest",
    ):
        assert name in artifacts, name
        assert Path(artifacts[name]).exists(), name
    # Branches below min_branch_cells get a dotplot but no branch UMAP.
    assert f"{SAMPLE}_map_first_branch_microglia_gene_dotplot" in artifacts
    assert f"{SAMPLE}_map_first_branch_microglia_umap" not in artifacts
    leaves = pd.read_csv(artifacts["map_first_leaves"])
    assert int(leaves["n_cells"].sum()) == clustered.n_obs
    listed = json.loads(clustered.uns[HIERARCHICAL_UNS_KEY]["artifacts_json"])
    assert set(listed) == set(artifacts)


def test_build_hierarchy_rejects_bad_branches() -> None:
    section = make_section("human")
    table = section.labels.loc[section.labels[Columns.IN_TABLE]].copy()
    bad = table.copy()
    bad[Columns.CT_BRANCH] = bad[Columns.CT_BRANCH].cat.add_categories(["A:B"])
    bad.iloc[0, bad.columns.get_loc(Columns.CT_BRANCH)] = "A:B"
    with pytest.raises(ValueError, match="must not contain"):
        build_hierarchy_columns(bad)
    missing = table.copy()
    missing.iloc[0, missing.columns.get_loc(Columns.CT_BRANCH)] = None
    with pytest.raises(ValueError, match="no ct_branch"):
        build_hierarchy_columns(missing)
    two = table.copy()
    two[Columns.SPECIES] = pd.Categorical(
        ["human"] * (len(two) - 1) + ["mouse"], categories=["human", "mouse"]
    )
    with pytest.raises(ValueError, match="one species"):
        build_hierarchy_columns(two)


def test_leaf_on_an_unassigned_branch_is_dropped() -> None:
    section = make_section("human")
    table = section.labels.loc[section.labels[Columns.IN_TABLE]].copy()
    rows = np.flatnonzero(table[Columns.CT_BRANCH].astype(str) == UNASSIGNED_LABEL)
    table.iloc[rows[:3], table.columns.get_loc(Columns.CT_LEAF)] = "MGE interneuron"
    hierarchy = build_hierarchy_columns(table)
    assert hierarchy.n_leaves_unassigned_branch == 3
    assert set(hierarchy.obs[SUBCLUSTER_LABEL_KEY].iloc[rows].astype(str)) == {
        UNRESOLVED_LABEL
    }


def test_gate_from_provenance_takes_the_stricter_level() -> None:
    human = make_provenance("human", gate_level="broad_only", panel_trust="validated")
    gate = LeafGate.from_provenance(human)
    assert gate.gate_level == "broad_only"
    assert gate.gate_warning is True
    assert gate.reasons == ("frac_ge30: A < 0.3",)
    assert gate.suppresses_leaves
    assert not LeafGate.from_provenance(None).suppresses_leaves
    assert not LeafGate(gate_level="full", panel_trust="provisional").suppresses_leaves
    assert LeafGate(gate_level="full", panel_trust="refused").suppresses_leaves


@pytest.mark.parametrize(
    ("state", "unassigned"),
    [
        ("Mixed/Unknown", True),
        ("Mixed/Unknown:unresolved", True),
        ("Neurons/unresolved", True),
        ("Neurons/unresolved:unresolved", True),
        ("Oligodendrocyte lineage/unresolved:unresolved", True),
        ("Neurons/Excitatory:Upper-layer intratelencephalic", False),
        ("Neurons/Excitatory:unresolved", False),
        ("Astrocytes:unresolved", False),
        ("Oligodendrocyte lineage", False),
        ("01 IT-ET Glut:unresolved", False),
        ("30 Astro-Epen", False),
        ("Mixed/Unknown:not_subclustered", True),
    ],
)
def test_unassigned_states(state: str, unassigned: bool) -> None:
    assert is_unassigned_state(state) is unassigned


def test_unassigned_states_are_sorted_and_unique() -> None:
    states = ["Neurons/unresolved:unresolved", "Mixed/Unknown:unresolved", "x"] * 2
    assert unassigned_states(states) == (
        "Mixed/Unknown:unresolved",
        "Neurons/unresolved:unresolved",
    )


def test_load_label_inputs_reads_the_resolve_outputs(tmp_path: Path) -> None:
    section = make_section("human")
    folder = tmp_path / "annotation_resolve_out" / "merscope"
    provenance_json = section.provenance.to_uns_json()
    write_label_table(
        section.labels, folder / label_table_filename(SAMPLE), provenance_json
    )
    root = tmp_path / "annotation_resolve_out"
    # Without a manifest the provenance comes from the parquet metadata.
    inputs = load_label_inputs(root, SAMPLE, "MERSCOPE")
    assert inputs.manifest_path is None
    assert inputs.provenance == section.provenance
    assert len(inputs.labels) == len(section.labels)
    manifest = folder / annotation_manifest_filename(SAMPLE)
    other = make_provenance("human", gate_level="broad_only")
    manifest.write_text(other.to_uns_json())
    inputs = load_label_inputs(root, SAMPLE, "MERSCOPE")
    assert inputs.manifest_path == manifest
    assert inputs.provenance == other
    assert find_label_table(folder, SAMPLE) == inputs.labels_path
    with pytest.raises(FileNotFoundError, match="no label table"):
        find_label_table(root, "P9999_XENIUM", "XENIUM")
    clustered, _ = run_map_first_hierarchy(
        section.adata,
        inputs.labels,
        make_config(tmp_path),
        tmp_path / "h",
        SAMPLE,
        provenance=inputs.provenance,
        plots=False,
        stability=False,
    )
    assert (clustered.obs[SUBCLUSTER_LABEL_KEY].astype(str) == UNRESOLVED_LABEL).all()


def test_mouse_hierarchy_uses_class_branches() -> None:
    section = make_section("mouse")
    table = section.labels.loc[section.labels[Columns.IN_TABLE]]
    hierarchy = build_hierarchy_columns(table, min_branch_cells=50)
    assert hierarchy.leaf_level == "subclass"
    assert hierarchy.species == "mouse"
    categories = list(hierarchy.obs[HIERARCHICAL_CLUSTER_KEY].cat.categories)
    # Branches follow the WMB class order, Mixed/Unknown last.
    assert categories[0].startswith("01 IT-ET Glut:")
    assert categories[-1] == "Mixed/Unknown:unresolved"
    assert set(hierarchy.obs[SUBCLUSTER_STATUS_KEY].astype(str)) == {
        "mapped_subclass",
        "unresolved",
        "unassigned",
    }


def test_statuses_come_from_the_leaf_level_status() -> None:
    types = [
        CellType(
            "gated",
            60,
            {
                "lineage": (CellStatus.CONFIDENT, "Astrocytes"),
                "broad": (CellStatus.CONFIDENT, "Astrocytes"),
                "nt": (CellStatus.NOT_APPLICABLE, None),
                "supercluster": (CellStatus.NOT_ATTEMPTED_GATE, None),
                "seaad_subclass": (CellStatus.NOT_ATTEMPTED_GATE, None),
            },
            ("broad", "Astrocytes"),
            "Astrocytes",
            UNRESOLVED_LABEL,
            2,
        ),
        HUMAN_TYPES[0],
    ]
    section = make_section("human", types=types)
    table = section.labels.loc[section.labels[Columns.IN_TABLE]]
    hierarchy = build_hierarchy_columns(table)
    statuses = hierarchy.obs[SUBCLUSTER_STATUS_KEY].astype(str)
    gated = (table[Columns.CT_BRANCH].astype(str) == "Astrocytes").to_numpy()
    assert set(statuses[gated]) == {"dataset_broad_only"}
    assert set(statuses[~gated]) == {"mapped_supercluster"}


def test_report_only_fine_level_columns_are_copied_as_categoricals(
    tmp_path: Path,
) -> None:
    section = make_section("human")
    labels = section.labels.copy()
    for field in ("name", "raw", "conf", "corr", "runner_up", "margin", "status"):
        labels[Columns.level("cluster", field)] = labels[
            Columns.level("supercluster", field)
        ]
    labels[Columns.level("cluster", "validated")] = labels[
        Columns.level("supercluster", "validated")
    ]
    section.labels = labels
    clustered, _ = _run(section, tmp_path)
    for field in ("name", "runner_up", "status"):
        column = Columns.level("cluster", field)
        assert isinstance(clustered.obs[column].dtype, pd.CategoricalDtype), column
    assert clustered.obs[Columns.level("cluster", "conf")].dtype == np.float32
