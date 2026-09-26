"""Tests for annotation provenance (merxen.annotation.provenance)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest
from pydantic import ValidationError

from merxen.annotation.provenance import (
    PROVENANCE_UNS_KEY,
    AnnotationProvenance,
    DatasetGateProvenance,
    EngineProvenance,
    FlagProvenance,
    MarkerProvenance,
    MouseGateProvenance,
    PanelProvenance,
    RealQcProvenance,
    ReferenceProvenance,
    ResolvabilityProvenance,
    SourceIdentity,
    ThresholdProvenance,
    annotation_manifest_filename,
    is_safe_key,
    to_uns_json,
    unsafe_structure_problems,
)
from merxen.annotation.schema import safe_token


def _human_provenance() -> AnnotationProvenance:
    whb = ReferenceProvenance(
        reference_id="whb_frontal_supc_clus",
        role="primary",
        taxonomy_id="CCN202210140",
        levels=["CCN202210140_SUPC", "CCN202210140_CLUS"],
        n_leaves=653,
        bundle_path="/store/whb_frontal_supc_clus/abc123",
        build_hash="abc123",
        sources={
            "precompute": SourceIdentity(
                path="/store/src/precompute.h5",
                size=1024,
                mtime_ns=1,
                sha256="f" * 64,
                sha256_scope="head_tail_64mb",
            )
        },
        n_query_genes_used=296,
        markers=MarkerProvenance(
            lookup_sha256="e" * 64,
            n_per_utility=30,
            n_panel_genes=296,
            markers_per_parent_min=12,
            markers_per_parent_median=30.0,
        ),
        panel_trust="validated",
    )
    seaad = ReferenceProvenance(reference_id="seaad_mr_panel", role="secondary")
    return AnnotationProvenance(
        species="human",
        anatomical_region="frontal_cortex",
        merxen_version="0.1.0",
        panel=PanelProvenance(
            panel_hash="a" * 64,
            panel_family="human_set_a",
            panel_mode="intersection",
            panel_trust="validated",
            validation_basis="real_data",
            validated_max_level="supercluster",
            validated_share={"broad": 1.0, "supercluster": 0.98},
            real_data_qc=RealQcProvenance(
                outcomes={"marker_consistency": "pass", "paired_broad_jsd": "warn"},
                downgrades=[],
                warn_only=True,
            ),
            n_declared_genes=296,
            gene_id_resolution={"native": 290, "reference_gene_csv": 6},
            n_unmapped=0,
            controls_removed={safe_token("Blank"): 20},
            panel_report_sha256="b" * 64,
        ),
        references={whb.reference_id: whb, seaad.reference_id: seaad},
        engine=EngineProvenance(
            ctm_version="1.7.2",
            bootstrap_factor=0.5,
            bootstrap_iteration=100,
            rng_seed=0,
            n_processors=6,
            wall_time_s=812.5,
        ),
        resolvability={
            "whb_frontal_supc_clus": ResolvabilityProvenance(
                recipe="R1_contam_HO",
                recipe_version=1,
                emitted_depth_bins={
                    "supercluster": {
                        safe_token("Oligodendrocyte precursors"): [120, 250],
                        safe_token("Neurons"): [10, 15, 30, 60, 120, 250],
                    }
                },
                d_max={safe_token("Microglia"): 120},
                extrapolated_share={safe_token("Microglia"): 0.12},
                reweighted_to_composition=True,
                resolvable_share={"supercluster": 0.64},
            )
        },
        thresholds=ThresholdProvenance(
            values={"whb_broad": 0.73, "whb_supercluster": 0.69},
            threshold_source="validated_default",
            floors_sha256="c" * 64,
            floor_source="real_e2",
        ),
        flags=FlagProvenance(
            thresholds={"contamination_alpha": 0.01},
            gene_sets={"negative_neurons": ["AQP4", "GJA1"]},
            realised_rates={
                "flag_contaminated": {
                    safe_token("Astrocytes/MERSCOPE"): 0.04,
                }
            },
            informative={
                "flag_contaminated": {safe_token("Astrocytes/MERSCOPE"): True}
            },
            null_reasons={"flag_microglial_spillover": "disabled_for_human"},
        ),
        gate=DatasetGateProvenance(
            frac_ge30=0.62,
            table_broad_coverage=0.71,
            segmented_broad_coverage=0.4,
            level="full",
            warning=False,
        ),
        confident_fraction_table={"broad": 0.71, "supercluster": 0.55},
        confident_fraction_segmented={"broad": 0.4, "supercluster": 0.31},
    )


def _mouse_provenance() -> AnnotationProvenance:
    return AnnotationProvenance(
        species="mouse",
        references={
            "wmb_panel": ReferenceProvenance(
                reference_id="wmb_panel",
                role="primary",
                drop_level="CCN20230722_SUPT",
                nodes_dropped=["05 OB-IMN GABA"],
            )
        },
        mouse_gate=MouseGateProvenance(
            signals={"g1_density_ratio": 2.9, "g4_astro_epen_points": None},
            level="full",
            warning=True,
            reasons=["g4_astro_epen_outside_band"],
            section_regions=["Isocortex", "HPF", "TH"],
            region_source="auto",
            n_assigned_tiles=1200,
            drop_list_size=12,
            n_remapped=3000,
            rule_variant="v1",
        ),
    )


def _walk_lists(value: Any) -> list[Any]:
    found = []
    if isinstance(value, dict):
        for item in value.values():
            found += _walk_lists(item)
    elif isinstance(value, list):
        found.append(value)
        for item in value:
            found += _walk_lists(item)
    return found


@pytest.mark.parametrize("builder", [_human_provenance, _mouse_provenance])
def test_json_round_trip_is_lossless_and_canonical(builder: Any) -> None:
    provenance = builder()
    text = provenance.to_uns_json()
    assert to_uns_json(provenance) == text
    restored = AnnotationProvenance.from_uns_json(text)
    assert restored == provenance
    assert restored.to_uns_json() == text
    assert json.loads(text) == json.loads(builder().to_uns_json())
    assert unsafe_structure_problems(json.loads(text)) == []


def test_serialised_provenance_has_no_list_of_mappings() -> None:
    payload = json.loads(_human_provenance().to_uns_json())
    for values in _walk_lists(payload):
        assert not any(isinstance(item, dict) for item in values)
    assert payload["schema_version"] == 1
    assert payload["label_table_version"] == 1
    assert payload["panel"]["validation_basis"] == "real_data"


@pytest.mark.filterwarnings("ignore:Writing zarr v2 data:UserWarning")
@pytest.mark.parametrize("backend", ["h5ad", "zarr"])
def test_provenance_survives_anndata_round_trips(backend: str, tmp_path: Path) -> None:
    provenance = _human_provenance()
    adata = ad.AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.obs_names = ["cell_0", "cell_1"]
    adata.var_names = ["GENE1", "GENE2"]
    provenance.write_to_uns(adata.uns)
    if backend == "h5ad":
        path = tmp_path / "labels.h5ad"
        adata.write_h5ad(path)
        restored = ad.read_h5ad(path)
    else:
        path = tmp_path / "labels.zarr"
        adata.write_zarr(path)
        restored = ad.read_zarr(path)
    assert isinstance(restored.uns[PROVENANCE_UNS_KEY], str)
    assert AnnotationProvenance.read_from_uns(restored.uns) == provenance


def test_read_from_uns_handles_absent_and_bad_values() -> None:
    assert AnnotationProvenance.read_from_uns({}) is None
    text = _mouse_provenance().to_uns_json()
    assert AnnotationProvenance.read_from_uns({PROVENANCE_UNS_KEY: np.str_(text)}) == (
        _mouse_provenance()
    )
    with pytest.raises(TypeError, match="JSON string"):
        AnnotationProvenance.read_from_uns({PROVENANCE_UNS_KEY: {"species": "human"}})


@pytest.mark.parametrize("key", ["Astrocytes/Ependymal", "has space", "", "a:b"])
def test_unsafe_keys_are_rejected(key: str) -> None:
    assert not is_safe_key(key)
    with pytest.raises(ValidationError, match="unsafe key"):
        PanelProvenance(validated_share={key: 0.5})


def test_safe_keys() -> None:
    for key in ("broad", "whb_frontal_supc_clus", "g1_density_ratio", "a.b-c"):
        assert is_safe_key(key)
    assert not is_safe_key(3)


def test_unsafe_structures_are_listed() -> None:
    problems = unsafe_structure_problems(
        {"ok": [{"nested": 1}], "bad/key": 1.0, "nan": float("nan")}
    )
    assert "$.ok[0]: list holds a mapping" in problems
    assert "$: unsafe key 'bad/key'" in problems
    assert any("non-finite" in problem for problem in problems)


def test_non_finite_floats_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EngineProvenance(wall_time_s=float("nan"))
    with pytest.raises(ValidationError):
        DatasetGateProvenance(frac_ge30=float("inf"))
    with pytest.raises(ValidationError, match="Extra inputs"):
        EngineProvenance(gpu=True)  # type: ignore[call-arg]


def test_reference_keys_must_match_reference_ids() -> None:
    reference = ReferenceProvenance(reference_id="wmb_panel", role="primary")
    with pytest.raises(ValidationError, match="differs from its reference_id"):
        AnnotationProvenance(species="mouse", references={"wmb": reference})
    with pytest.raises(ValidationError, match="safe token"):
        ReferenceProvenance(reference_id="wmb/panel", role="primary")
    with pytest.raises(ValidationError, match="unknown references"):
        AnnotationProvenance(
            species="mouse",
            references={"wmb_panel": reference},
            resolvability={"whb_frontal_supc_clus": ResolvabilityProvenance()},
        )


def test_trust_basis_and_species_rules() -> None:
    with pytest.raises(ValidationError, match="validation_basis"):
        PanelProvenance(panel_trust="provisional", validation_basis="simulation")
    PanelProvenance(panel_trust="validated", validation_basis="simulation")
    PanelProvenance(panel_trust="provisional")
    with pytest.raises(ValidationError):
        PanelProvenance(panel_trust="trusted")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="mouse_gate"):
        AnnotationProvenance(species="human", mouse_gate=MouseGateProvenance())
    with pytest.raises(ValidationError):
        DatasetGateProvenance(level="partial")  # type: ignore[arg-type]


def test_minimal_provenance_defaults() -> None:
    provenance = AnnotationProvenance(species="human")
    assert provenance.mode == "map_first"
    assert provenance.panel is None
    assert provenance.references == {}
    payload = json.loads(provenance.to_uns_json())
    assert payload["species"] == "human"
    assert payload["panel"] is None


def test_manifest_filename() -> None:
    assert annotation_manifest_filename("P1212_XENIUM") == (
        "P1212_XENIUM_annotation_manifest.json"
    )
