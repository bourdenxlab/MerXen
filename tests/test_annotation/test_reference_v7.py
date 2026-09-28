"""PREP of resolvability version 7 (plan §8.3 v7.1, v7.6; M3c).

Family selection and the build hash (a version-6 family's payload never
changes; a version-7 family's holds every version-7 input), the version-7
self-map of a primary bundle, and the test-set top-up of both species: never
a marker-training cell, a test cell or another frontal donor's cell, only of
training clusters, within the cluster cap.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from merxen.annotation import reference
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.reference import (
    ReferenceBuildError,
    builder_for,
    prepare_reference_spec,
    sample_wmb_training_cells,
)
from merxen.annotation.store import BUNDLE_MANIFEST_NAME, ReferenceStore

from .test_reference import (  # noqa: F401
    CLAS,
    CLUS,
    GENES,
    HO_OTHER_REGION,
    MARKER_OF_SUPC,
    SUPC,
    FakeCtm,
    ho_lookup,
    ho_precompute,
    ho_spec_sources,
    install_self_map,
    make_panel,
    small_resources,
    whb_resolvability_setup,
    whb_spec,
    write_ho_sources,
    write_wmb_testset_sources,
)

VALIDATED_SET_A_297 = "6e5fd5fb86ef0c3eaccf2efa5b792c9515b087fdf9e9cb6a3978d53268253dc3"
PINNED_P5011 = "ed7bc6bed98953942f43e6e4ac5af46469faa8195e443ff8f2981ef560808774"


def listed_panel(panel_hash: str, n_genes: int = 297) -> Any:
    return SimpleNamespace(panel_hash=panel_hash, n_genes=n_genes, species="human")


# --------------------------------------------------------------------------
# Family selection and build hash


def test_version_follows_the_family() -> None:
    config = AnnotationConfig(species="human")
    assert reference.panel_resolvability_version(
        listed_panel(VALIDATED_SET_A_297), config
    ) == (6)
    assert reference.panel_resolvability_version(
        listed_panel(PINNED_P5011), config
    ) == (6)
    inherited = SimpleNamespace(
        panel_hash="0" * 64,
        n_genes=300,
        species="human",
        panel_family=SimpleNamespace(family_id="human_set_a"),
    )
    assert reference.panel_resolvability_version(inherited, config) == 6
    assert reference.panel_resolvability_version(make_panel(GENES), config) == 7


def test_forcing_version_7_on_a_version_6_family_is_refused() -> None:
    forced = AnnotationConfig(species="human", resolvability={"version": 7})
    with pytest.raises(ReferenceBuildError, match="diagnostic"):
        reference.panel_resolvability_version(listed_panel(VALIDATED_SET_A_297), forced)
    assert reference.panel_resolvability_version(make_panel(GENES), forced) == 7
    six = AnnotationConfig(species="human", resolvability={"version": 6})
    assert reference.panel_resolvability_version(make_panel(GENES), six) == 6


def whb_primary_spec(tmp_path: Path) -> AnnotationReferenceSpec:
    return whb_spec(region_precompute=tmp_path)


def test_a_version_6_payload_holds_no_version_7_input(tmp_path: Path) -> None:
    spec = whb_primary_spec(tmp_path)
    config = AnnotationConfig(species="human")
    builder = builder_for(spec, config)
    listed = builder.params_for(cast("Any", listed_panel(VALIDATED_SET_A_297)))
    assert listed == builder.params
    assert "v7" not in listed["resolvability"]
    ho = AnnotationReferenceSpec(
        reference_id=reference.HO_REFERENCE_ID,
        species="human",
        role="resolvability",
        hierarchy=[SUPC, CLUS],
    )
    ho_builder = builder_for(ho, config)
    assert ho_builder.params_for(cast("Any", listed_panel(VALIDATED_SET_A_297))) == (
        ho_builder.params
    )
    # A version-7 family's payload holds the members, grid and top-up.
    params = builder.params_for(make_panel(GENES))["resolvability"]
    assert params["resolvability_version"] == 7
    assert "recipes" not in params
    assert [item["member"] for item in params["v7"]["members"]] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
        "clean@0",
    ]
    assert params["test_set"]["top_up"]["target"] == 200
    ho_params = ho_builder.params_for(make_panel(GENES))
    assert ho_params["top_up"] == params["test_set"]["top_up"]


def test_version_7_inputs_enter_the_hash_and_decision_knobs_do_not(
    tmp_path: Path,
) -> None:
    spec = whb_primary_spec(tmp_path)
    panel = make_panel(GENES)

    def payload(**updates: Any) -> dict[str, Any]:
        config = AnnotationConfig(species="human", resolvability=updates)
        return builder_for(spec, config).params_for(panel)["resolvability"]

    base = payload()
    assert payload(ensemble_r1_seeds=[0, 1, 3]) != base
    assert payload(topup_min_class_test_cells=300) != base
    assert payload(ensemble_spread_floor=0.05) == base
    assert payload(saturated_bp_share=0.95) == base
    assert payload(monotone_depth=False) == base
    # A version-6 family ignores the version-7 inputs.
    listed = cast("Any", listed_panel(VALIDATED_SET_A_297))
    six = builder_for(spec, AnnotationConfig(species="human")).params_for(listed)
    other = builder_for(
        spec,
        AnnotationConfig(species="human", resolvability={"ensemble_r1_seeds": [5, 6]}),
    ).params_for(listed)
    assert six == other


def test_the_version_7_grid_above_1000_genes(tmp_path: Path) -> None:
    spec = AnnotationReferenceSpec(
        reference_id="wmb_panel", species="mouse", role="primary"
    )
    genes = [f"ENSMUSG{index:011d}" for index in range(1200)]
    plan = reference.resolvability_plan(
        spec, make_panel(genes, species="mouse"), AnnotationConfig(species="mouse")
    )
    assert plan.version == 7
    assert list(plan.grid) == list(res.V7_LARGE_PANEL_GRID)
    assert plan.chemistry is not None and plan.chemistry.chemistry == "unknown"
    assert [member.name for member in plan.members if member.role == "emission"] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
    ]


def test_neighbour_structured_spill_stays_off() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        AnnotationConfig(
            species="mouse", resolvability={"neighbour_structured_spill": True}
        )


# --------------------------------------------------------------------------
# The version-7 self-map of a primary bundle


def test_a_version_7_primary_maps_every_member_on_the_topped_up_test_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_resources: Any,  # noqa: F811
) -> None:
    sources, ho, _ = whb_resolvability_setup(tmp_path, monkeypatch)
    markers = {node: GENES[index] for node, index in MARKER_OF_SUPC.items()}
    calls = install_self_map(monkeypatch, markers)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
            whb_h5ad_dir=ho["h5ad_dir"],
            whb_metadata_dir=ho["metadata"],
            whb_region_cell_metadata=ho["region_dir"]
            / reference.REGION_CELL_METADATA_FILE,
        )
    )
    config = AnnotationConfig(
        species="human",
        resolvability={"n_test_cells": 20, "topup_min_class_test_cells": 12},
    )
    (tmp_path / "scratch").mkdir()
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    bundle = store.get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]["resolvability"]
    assert output["resolvability_version"] == 7
    assert [call["tag"] for call in calls] == [
        "R1_contam_HO_seed0",
        "R1_contam_HO_seed1",
        "R1_contam_HO_seed2",
        "clean_seed0",
    ]
    for name in (
        res.RESOLVABILITY_FILE,
        res.RESOLVABILITY_CELLS_FILE,
        res.RESOLVABILITY_SUMMARY_FILE,
        res.CLASS_DEPTH_FILE,
    ):
        assert (bundle_dir / name).is_file()
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert summary["resolvability_version"] == 7
    assert summary["v7_inputs"]["chemistry"]["chemistry"] == "unknown"
    assert summary["top_up"]["version"] == res.TOP_UP_VERSION
    payload = manifest["build_hash_payload"]["builder_params"]["resolvability"]
    assert payload["v7"]["resolvability_version"] == 7
    assert payload["test_set"]["top_up"]["target"] == 12
    with pytest.raises(res.ResolvabilityError, match="version-7"):
        res.load_resolvability(bundle_dir)
    tables = res.load_resolvability(bundle_dir, allow_version_7=True)
    assert tables is not None and tables.version == 7
    # The held-out test set was topped up (class top-up, version 7 only).
    (ho_dir,) = (tmp_path / "store" / reference.HO_REFERENCE_ID).glob("[0-9a-f]*")
    test_output = json.loads((ho_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]
    assert test_output["resolvability_version"] == 7
    assert test_output["class_top_up"]["n_cells"] > 0


# --------------------------------------------------------------------------
# Test-set top-up: human (donor hold-out) and mouse (training-cluster rule)


def test_the_human_top_up_never_takes_another_frontal_donor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_resources: Any,  # noqa: F811
) -> None:
    sources = write_ho_sources(tmp_path)
    fake = FakeCtm(ho_lookup).install(monkeypatch)
    fake.precompute = ho_precompute
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.HO_REFERENCE_ID,
            species="human",
            role="resolvability",
            hierarchy=[SUPC, CLUS],
            sources=ho_spec_sources(sources),
        )
    )
    config = AnnotationConfig(
        species="human",
        resolvability={"n_test_cells": 12, "topup_min_class_test_cells": 12},
    )
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    test = res.load_test_cells(bundle_dir)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]
    record = output["class_top_up"]
    added = test.obs[
        test.obs[reference.TEST_SOURCE_COLUMN].isin(
            [reference.TOP_UP_SOURCE, reference.TOP_UP_SOURCE_OTHER_REGION]
        )
    ]
    assert len(added) == record["n_cells"] > 0
    frontal = added["region_of_interest_label"].astype(str) == "Human A46"
    # Frontal top-up cells are the held-out donor's; the rest other-region.
    assert set(added.loc[frontal, "donor_label"]) <= {"H_small"}
    other_rois = {roi for *_rest, roi, _donor in HO_OTHER_REGION}
    assert set(added.loc[~frontal, "region_of_interest_label"]) <= other_rois
    # Only non-neuronal classes draw other-region cells, never the dropped c5.
    donor_rows = added[added[reference.TEST_SOURCE_COLUMN] == reference.TOP_UP_SOURCE]
    other = added[added[reference.TEST_SOURCE_COLUMN] != reference.TOP_UP_SOURCE]
    assert set(other[f"{res.TRUTH_PREFIX}{CLUS}"]) <= {"c3", "c4"}
    assert not set(donor_rows.index) & set(
        test.obs.index[test.obs[reference.TEST_SOURCE_COLUMN] == "holdout_donor"]
    )
    assert record["disjoint_from_test_cells"] is True
    for item in record["per_class"].values():
        assert item["after"] <= 12


def test_the_mouse_top_up_never_takes_training_or_test_cells(
    tmp_path: Path,
) -> None:
    sources = write_wmb_testset_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.WMB_TESTSET_REFERENCE_ID,
            species="mouse",
            role="resolvability",
            max_cells_per_cluster=20,
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )
    config = AnnotationConfig(
        species="mouse",
        resolvability={
            "mouse_nonneuronal_extra_per_supertype": 0,
            "topup_min_class_test_cells": 60,
            "gate_p_topup_max_cluster_frac": 0.12,
        },
    )
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec,
        make_panel(genes[:30], species="mouse"),
        builder=builder_for(spec, config),
        config=config,
    )
    bundle_dir = Path(bundle.path)
    test = res.load_test_cells(bundle_dir)
    record = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]["class_top_up"]
    added = test.obs[test.obs["test_source"] == reference.TOP_UP_SOURCE]
    assert len(added) == record["n_cells"] > 0
    base = set(pd.read_csv(sources["truth"])["cell_label"])
    sampled, _ = sample_wmb_training_cells(
        sources["metadata"] / "cell_metadata.csv",
        {"WMB-10Xv3-AAA": Path(), "WMB-10Xv3-BBB": Path()},
        max_cells_per_cluster=20,
        exclude_cells=base,
    )
    assert not set(added.index) & (set(sampled["cell_label"]) | base)
    meta = pd.read_csv(sources["metadata"] / "cell_metadata.csv").set_index(
        "cell_label"
    )
    clusters = meta.loc[list(added.index), "cluster_alias"]
    training = sampled.groupby("cluster_alias").size()
    assert all(training.get(alias, 0) >= 5 for alias in clusters)
    # At most 12% of a cluster's 25 cells: 3 per cluster.
    assert clusters.value_counts().max() <= 3
    assert record["disjoint_from_training"] and record["disjoint_from_test_cells"]
    per_class = test.obs[f"{res.TRUTH_PREFIX}{CLAS}"].value_counts()
    for cls, item in record["per_class"].items():
        assert per_class[cls] == item["after"] <= 60
