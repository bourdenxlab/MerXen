"""WHB region profiles: the hippocampal-formation map_first references.

The human map_first route maps on a WHB region reference. Frontal cortex is
the validated region; the hippocampal formation (20 WHB ROIs) is a
provisional second region with its own primary (``whb_hippocampal_supc_clus``)
and held-out test set (``whb_hippocampal_supc_clus_ho``), built by the frontal
builders from a region profile.

The first test pins the frontal build-hash payloads: the digests were taken
on the code before the region profiles existed (v1.0.2), so a frontal bundle
never rebuilds because of them. If a later change alters a frontal payload on
purpose (a new builder param, a vocab edit), update the digests in the same
commit and say why in its message.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import reference
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.panel import AnnotationPanel, compute_panel_hash
from merxen.annotation.reference import (
    HO_REFERENCE_ID,
    WHB_FRONTAL_ROI_LABELS,
    WHB_HIPPOCAMPAL_ROI_LABELS,
    ReferenceBuildError,
    builder_for,
    ho_reference_id,
    prepare_reference_spec,
    verify_whb_region_precompute,
    whb_profile,
)
from merxen.annotation.store import canonical_json

SUPC = "CCN202210140_SUPC"
CLUS = "CCN202210140_CLUS"
HIPPOCAMPAL_PRIMARY = "whb_hippocampal_supc_clus"
HIPPOCAMPAL_HO = "whb_hippocampal_supc_clus_ho"

# sha256 of canonical_json(builder.params_for(panel)) on a188207 (v1.0.2),
# keyed by (resolvability enabled, reference id, panel genes).
FRONTAL_PAYLOAD_DIGESTS = {
    (True, "whb_frontal_supc_clus", 10): (
        "9ea9307a52ffc7fcc37b575abb05f9bec81b4b3588661e9cad8bec36fa250e7d"
    ),
    (True, "whb_frontal_supc_clus", 2000): (
        "69a29c91b49fbf98cbac5da20cc427dc766fb6931496c09ee8840b9654a4b120"
    ),
    (True, "whb_frontal_supc_clus_ho", 10): (
        "e9774a388988960e5088900143744c9465eba163fd9fe1aaca66061ae71ae3f0"
    ),
    (True, "whb_frontal_supc_clus_ho", 2000): (
        "e9774a388988960e5088900143744c9465eba163fd9fe1aaca66061ae71ae3f0"
    ),
    (True, "seaad_mr_panel", 10): (
        "96cfcd80a55ae94dcedf5a2e3a5fe80060a971e64a3218600dfcc1bf395bca32"
    ),
    (True, "seaad_mr_panel", 2000): (
        "1ea2d60c0ad240eb2e896edc206592973681e0d11d2466721d420c890b1fd4bd"
    ),
    (False, "whb_frontal_supc_clus", 10): (
        "ab3ef7b732b300a6bdfb1c4f0245a3d07fca0cee943c653e5df77e442cf9673e"
    ),
    (False, "whb_frontal_supc_clus", 2000): (
        "ab3ef7b732b300a6bdfb1c4f0245a3d07fca0cee943c653e5df77e442cf9673e"
    ),
    (False, "whb_frontal_supc_clus_ho", 10): (
        "3ca55104e973c7564c1480798c5789a9885537f6145a84ab0ee76c57e9352a42"
    ),
    (False, "whb_frontal_supc_clus_ho", 2000): (
        "3ca55104e973c7564c1480798c5789a9885537f6145a84ab0ee76c57e9352a42"
    ),
    (False, "seaad_mr_panel", 10): (
        "8d50d2531ed031bdb0d7cbc7d86f221beb20da3745264e6a34e73c70a80c5900"
    ),
    (False, "seaad_mr_panel", 2000): (
        "8d50d2531ed031bdb0d7cbc7d86f221beb20da3745264e6a34e73c70a80c5900"
    ),
}
ROLES = {
    "whb_frontal_supc_clus": "primary",
    "whb_frontal_supc_clus_ho": "resolvability",
    "seaad_mr_panel": "secondary",
    HIPPOCAMPAL_PRIMARY: "primary",
    HIPPOCAMPAL_HO: "resolvability",
}


def make_panel(n_genes: int) -> AnnotationPanel:
    """Return a human declared panel of ``n_genes`` synthetic IDs."""
    ids = sorted(f"ENSG{index:011d}" for index in range(1, n_genes + 1))
    return AnnotationPanel(
        name="intersection",
        kind="intersection",
        species="human",
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=["S_M", "S_X"],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[f"SYM{gene[-4:]}" for gene in ids],
    )


def make_spec(reference_id: str, **sources: Path) -> AnnotationReferenceSpec:
    """Return a human reference spec with the plan's hierarchy."""
    return AnnotationReferenceSpec(
        reference_id=reference_id,
        species="human",
        role=ROLES[reference_id],  # type: ignore[arg-type]
        hierarchy=[SUPC, CLUS] if reference_id.startswith("whb") else [],
        sources=sources,
    )


def hippocampal_config(*, resolvability: bool = True) -> AnnotationConfig:
    """Return a human config of the hippocampal-formation region."""
    return AnnotationConfig(
        species="human",
        anatomical_region="hippocampus",
        references=[
            {
                "reference_id": HIPPOCAMPAL_PRIMARY,
                "species": "human",
                "role": "primary",
                "hierarchy": [SUPC, CLUS],
            },
            {"reference_id": "seaad_mr_panel", "species": "human", "role": "secondary"},
        ],
        resolvability={"enabled": resolvability},
    )


def payload_digest(
    reference_id: str, config: AnnotationConfig, panel: AnnotationPanel
) -> str:
    builder = builder_for(make_spec(reference_id), config)
    return hashlib.sha256(
        canonical_json(builder.params_for(panel)).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("enabled", "reference_id", "n_genes"), sorted(FRONTAL_PAYLOAD_DIGESTS)
)
def test_frontal_build_hash_payloads_unchanged(
    enabled: bool, reference_id: str, n_genes: int
) -> None:
    """The region profiles leave every frontal builder payload byte-identical."""
    config = AnnotationConfig(species="human", resolvability={"enabled": enabled})
    digest = payload_digest(reference_id, config, make_panel(n_genes))
    assert digest == FRONTAL_PAYLOAD_DIGESTS[(enabled, reference_id, n_genes)]


def test_hippocampal_profile_holds_the_20_whb_rois() -> None:
    profile = whb_profile(region="hippocampus")
    assert profile.primary_id == HIPPOCAMPAL_PRIMARY
    assert profile.ho_id == HIPPOCAMPAL_HO
    assert len(profile.roi_labels) == 20
    assert profile.roi_labels == WHB_HIPPOCAMPAL_ROI_LABELS
    assert {"Human CA1", "Human DG-CA4", "Human Sub", "Human MEC"} <= set(
        profile.roi_labels
    )
    assert profile.validated is None
    assert not set(profile.roi_labels) & set(reference.HO_OTHER_REGION_ROI_LABELS)
    assert whb_profile(reference_id=HIPPOCAMPAL_HO) is profile
    frontal = whb_profile(region="frontal_cortex")
    assert frontal.roi_labels == WHB_FRONTAL_ROI_LABELS
    assert frontal.ho_id == HO_REFERENCE_ID
    assert whb_profile(reference_id="whb_frontal_supc_clus") is frontal
    with pytest.raises(ReferenceBuildError, match="no WHB region profile"):
        whb_profile(region="striatum")


def test_ho_reference_id_follows_the_config_region() -> None:
    assert ho_reference_id(AnnotationConfig(species="human")) == HO_REFERENCE_ID
    assert ho_reference_id(hippocampal_config()) == HIPPOCAMPAL_HO


def test_hippocampal_payloads_name_the_hippocampal_rois_and_test_set() -> None:
    config = hippocampal_config()
    panel = make_panel(10)
    whb = builder_for(make_spec(HIPPOCAMPAL_PRIMARY), config)
    assert whb.name == HIPPOCAMPAL_PRIMARY
    params = whb.params_for(panel)
    assert params["roi_labels"] == list(WHB_HIPPOCAMPAL_ROI_LABELS)
    assert params["negative_gene_references"] == ["whb_hippocampal", "seaad_mr"]
    test_set = params["resolvability"]["test_set"]
    assert test_set["reference_id"] == HIPPOCAMPAL_HO
    assert test_set["roi_labels"] == list(WHB_HIPPOCAMPAL_ROI_LABELS)
    assert test_set["truth_exclusion"]["region"] == "hippocampus"
    assert params["resolvability"]["anatomical_region"] == "hippocampus"
    seaad = builder_for(make_spec("seaad_mr_panel"), config).params_for(panel)
    assert seaad["resolvability"]["test_set"]["reference_id"] == HIPPOCAMPAL_HO
    # SEA-AD keeps its id; the hippocampal test set gives it its own hash.
    frontal_digest = FRONTAL_PAYLOAD_DIGESTS[(True, "seaad_mr_panel", 10)]
    assert payload_digest("seaad_mr_panel", config, panel) != frontal_digest
    ho = builder_for(make_spec(HIPPOCAMPAL_HO), config)
    assert ho.name == HIPPOCAMPAL_HO
    assert ho.params_for(panel)["roi_labels"] == list(WHB_HIPPOCAMPAL_ROI_LABELS)


def test_the_held_out_test_set_follows_the_region() -> None:
    config = hippocampal_config()
    assert reference.V7_TEST_SET_OF[HIPPOCAMPAL_PRIMARY] == HIPPOCAMPAL_HO
    # The WHB cells-table rules (the COP rule) apply to every region primary.
    assert len(reference.cells_rules_for(HIPPOCAMPAL_PRIMARY, config)) == len(
        reference.cells_rules_for(
            "whb_frontal_supc_clus", AnnotationConfig(species="human")
        )
    )
    assert reference.cells_rules_for(HIPPOCAMPAL_PRIMARY, config)
    assert reference.cells_rules_for("seaad_mr_panel", config) == []
    test_spec = reference._spec_of_test_set(
        make_spec(HIPPOCAMPAL_PRIMARY), config, {}, HIPPOCAMPAL_HO
    )
    assert test_spec.reference_id == HIPPOCAMPAL_HO
    assert test_spec.role == "resolvability"
    assert test_spec.hierarchy == [SUPC, CLUS]


def test_ho_truth_exclusions_keep_hippocampal_superclusters() -> None:
    from merxen.annotation import vocab

    table = vocab.primary_vocab("human")
    name_to_label = {name: label for label, name in table.label_to_name().items()}
    names = [
        "Hippocampal CA1-3",
        "Hippocampal CA4",
        "Hippocampal dentate gyrus",
        "Splatter",
        "Miscellaneous",
    ]
    labels = [name_to_label[name] for name in names]
    hippocampal = reference.ho_truth_exclusions(labels, region="hippocampus")
    frontal = reference.ho_truth_exclusions(labels, region="frontal_cortex")
    for name in names[:3]:
        assert name_to_label[name] not in hippocampal
        assert frontal[name_to_label[name]] == reference.HO_EXCLUDED_REGION_IMPLAUSIBLE
    assert hippocampal[name_to_label["Splatter"]] == reference.HO_EXCLUDED_SINK
    assert hippocampal[name_to_label["Miscellaneous"]] == reference.HO_EXCLUDED_SINK


def test_self_map_test_cells_leave_out_other_region_cop_for_both_regions() -> None:
    from merxen.annotation import resolvability as res

    obs = pd.DataFrame(
        {
            f"{res.TRUTH_PREFIX}{SUPC}": reference.ho_self_map_excluded_labels()[:1]
            * 2,
            res.TRUTH_LEAF_COLUMN: ["c1", "c1"],
            res.SPILL_GROUP_COLUMN: ["Glia", "Glia"],
            reference.TEST_SOURCE_COLUMN: [
                reference.TEST_SOURCE_OTHER_REGION,
                reference.TEST_SOURCE_DONOR,
            ],
        },
        index=["a", "b"],
    )
    cells = res.HeldOutCells(
        counts=sparse.csr_matrix(np.ones((2, 1))), genes=["g"], obs=obs
    )
    for test_id in (HO_REFERENCE_ID, HIPPOCAMPAL_HO):
        kept, record = reference.self_map_test_cells(cells, test_id)
        assert record is not None and record["n_excluded"] == 1
        assert list(kept.obs.index) == ["b"]
    kept, record = reference.self_map_test_cells(cells, "wmb_selfmap_testset")
    assert record is None and kept is cells


# --------------------------------------------------------------------------
# Region precompute checks


def write_region_dir(
    root: Path, labels: Sequence[str], *, min_cells: int = 10
) -> tuple[Path, Path]:
    """Write a minimal two-leaf WHB region precompute and its manifest."""
    tree = {
        "hierarchy": [SUPC, CLUS],
        SUPC: {"Astrocyte": ["c1"], "Hippocampal CA4": ["c2"]},
        CLUS: {"c1": [], "c2": []},
    }
    precompute = root / "precompute" / "precomputed_stats.h5"
    precompute.parent.mkdir(parents=True)
    n_cells = np.array([12, 15])
    with h5py.File(precompute, "w") as handle:
        for name in ("sum", "sumsq", "gt0"):
            handle.create_dataset(name, data=np.ones((2, 3)))
        handle.create_dataset("n_cells", data=n_cells)
        handle.create_dataset("col_names", data=json.dumps(["g1", "g2", "g3"]))
        handle.create_dataset("cluster_to_row", data=json.dumps({"c1": 0, "c2": 1}))
        handle.create_dataset("taxonomy_tree", data=json.dumps(tree))
    manifest = root / "region_reference_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config": {
                    "region_labels": list(labels),
                    "region_min_cells_per_leaf": min_cells,
                },
                "filtering_summary": {
                    "n_cells_after_min_leaf_filter": int(n_cells.sum()),
                    "n_leaf_aliases_after_min_leaf_filter": 2,
                },
            }
        )
    )
    return precompute, manifest


def test_verify_checks_the_manifest_against_its_region(tmp_path: Path) -> None:
    hippo_pre, hippo_manifest = write_region_dir(
        tmp_path / "hpf", WHB_HIPPOCAMPAL_ROI_LABELS
    )
    frontal_pre, frontal_manifest = write_region_dir(
        tmp_path / "frontal", WHB_FRONTAL_ROI_LABELS
    )
    hippocampus = whb_profile(region="hippocampus")
    frontal = whb_profile(region="frontal_cortex")
    result = verify_whb_region_precompute(
        hippo_pre, manifest_path=hippo_manifest, profile=hippocampus
    )
    assert result["manifest"]["agrees"] is True
    assert result["matches_validated_frontal"] is False
    with pytest.raises(ReferenceBuildError, match="region labels"):
        verify_whb_region_precompute(
            hippo_pre, manifest_path=hippo_manifest, profile=frontal
        )
    with pytest.raises(ReferenceBuildError, match="region labels"):
        verify_whb_region_precompute(
            frontal_pre, manifest_path=frontal_manifest, profile=hippocampus
        )
    # The default stays the frontal profile.
    verify_whb_region_precompute(frontal_pre, manifest_path=frontal_manifest)


def test_prepare_expands_a_hippocampal_region_directory(tmp_path: Path) -> None:
    region = tmp_path / "hpf"
    precompute, manifest = write_region_dir(region, WHB_HIPPOCAMPAL_ROI_LABELS)
    seaad = tmp_path / "seaad.h5"
    seaad.write_bytes(b"s")
    spec = make_spec(
        HIPPOCAMPAL_PRIMARY, region_precompute=region, seaad_precomputed_stats=seaad
    )
    prepared = prepare_reference_spec(
        spec, reference.SourceOptions(resolvability=False, auto_download=False)
    )
    assert prepared.sources["region_precompute"] == precompute
    assert prepared.sources["region_manifest"] == manifest


def test_hippocampal_primary_needs_a_region_precompute(tmp_path: Path) -> None:
    spec = make_spec(HIPPOCAMPAL_PRIMARY, whb_metadata_dir=tmp_path)
    with pytest.raises(ReferenceBuildError, match="region_precompute"):
        prepare_reference_spec(
            spec, reference.SourceOptions(resolvability=False, auto_download=False)
        )
