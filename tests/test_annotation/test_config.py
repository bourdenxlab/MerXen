"""Tests for the annotation configuration models (merxen.annotation.config)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from merxen.annotation.config import (
    CLUSTERING_MODES,
    DEFAULT_CLUSTERING_MODE,
    FLIPPED_SPECIES,
    HUMAN_DEPTH_GRID,
    KNOWN_REFERENCES,
    MAP_FIRST_TABLE_KEY_SUFFIX,
    MOUSE_CCF_REGIONS,
    WIDE_DEPTH_GRID,
    AnnotationConfig,
    AnnotationFlagsConfig,
    AnnotationGate,
    AnnotationPanelConfig,
    AnnotationRealQcConfig,
    AnnotationReferenceSpec,
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
    GatePStressRecipe,
    MouseGateConfig,
    MouseRegionConfig,
    default_depth_grid,
    default_references,
    normalise_mouse_section_regions,
    resolve_clustering_mode,
    resolve_table_key_suffix,
    species_defaults,
)
from merxen.annotation.vocab import load_vocab


def test_defaults_are_inert_and_legacy() -> None:
    config = AnnotationConfig()
    assert config.enabled is False
    assert DEFAULT_CLUSTERING_MODE == {"human": "legacy", "mouse": "legacy"}
    assert frozenset() == FLIPPED_SPECIES
    assert resolve_clustering_mode("human") == "legacy"
    assert resolve_clustering_mode("mouse") == "legacy"
    assert resolve_table_key_suffix("human", "legacy") == ""


def test_human_defaults() -> None:
    config = AnnotationConfig()
    assert config.species == "human"
    assert config.anatomical_region == "frontal_cortex"
    assert [spec.reference_id for spec in config.references] == [
        "whb_frontal_supc_clus",
        "seaad_mr_panel",
    ]
    assert config.primary_reference().reference_id == "whb_frontal_supc_clus"
    assert config.thresholds.max_leaf_level == "supercluster"
    assert config.flags.microglial_spillover_enabled is False
    assert config.real_qc.marker_consistency_warn == 0.75
    assert config.resolvability.n_test_cells == 25_000
    assert config.thresholds.hard_min_counts == config.min_counts == 10
    assert config.ctm_version == "1.7.2"
    assert config.xplat_sensitivity == "geneset_c"
    assert config.mouse_section_regions == "auto"


def test_mouse_defaults() -> None:
    config = AnnotationConfig(species="mouse")
    assert config.anatomical_region is None
    assert [spec.reference_id for spec in config.references] == [
        "wmb_panel",
        "wmb_region_share",
    ]
    assert config.primary_reference().drop_level == "CCN20230722_SUPT"
    assert config.thresholds.max_leaf_level == "subclass"
    assert config.flags.microglial_spillover_enabled is True
    assert config.real_qc.marker_consistency_warn == 0.80
    assert config.resolvability.n_test_cells == 11_913


def test_species_defaults_agree_with_the_models() -> None:
    for species in ("human", "mouse"):
        defaults = species_defaults(species)  # type: ignore[arg-type]
        config = AnnotationConfig(species=species)  # type: ignore[arg-type]
        assert defaults["clustering_mode"] == DEFAULT_CLUSTERING_MODE[species]
        assert defaults["max_leaf_level"] == config.thresholds.max_leaf_level
        assert defaults["references"] == [
            spec.reference_id for spec in config.references
        ]
    with pytest.raises(ValueError, match="species"):
        species_defaults("rat")  # type: ignore[arg-type]


def test_explicit_values_override_species_defaults() -> None:
    config = AnnotationConfig(
        species="human",
        flags=AnnotationFlagsConfig(microglial_spillover_enabled=True),
        real_qc=AnnotationRealQcConfig(marker_consistency_warn=0.8),
        resolvability=AnnotationResolvabilityConfig(n_test_cells=1000),
    )
    assert config.flags.microglial_spillover_enabled is True
    assert config.real_qc.marker_consistency_warn == 0.8
    assert config.resolvability.n_test_cells == 1000


def test_only_frontal_cortex_is_validated_for_human() -> None:
    with pytest.raises(ValidationError, match="OD-C7"):
        AnnotationConfig(anatomical_region="temporal_cortex")
    assert AnnotationConfig(anatomical_region="  ").anatomical_region == (
        "frontal_cortex"
    )
    with pytest.raises(ValidationError, match="human-only"):
        AnnotationConfig(species="mouse", anatomical_region="frontal_cortex")


def test_hard_min_counts_is_coupled_to_min_counts() -> None:
    config = AnnotationConfig(min_counts=15)
    assert config.thresholds.hard_min_counts == 15
    with pytest.raises(ValidationError, match="must equal min_counts"):
        AnnotationConfig(
            min_counts=15, thresholds=AnnotationThresholds(hard_min_counts=10)
        )
    AnnotationConfig(min_counts=15, thresholds=AnnotationThresholds(hard_min_counts=15))
    coupled = config.with_min_counts(20)
    assert coupled.min_counts == coupled.thresholds.hard_min_counts == 20
    assert config.min_counts == 15


def test_max_leaf_level_stays_the_species_leaf() -> None:
    with pytest.raises(ValidationError, match="OD-E4"):
        AnnotationConfig(thresholds=AnnotationThresholds(max_leaf_level="subclass"))
    config = AnnotationConfig(
        species="mouse", thresholds=AnnotationThresholds(max_leaf_level="subclass")
    )
    assert config.thresholds.max_leaf_level == "subclass"


def test_reference_rules() -> None:
    whb = AnnotationReferenceSpec(
        reference_id="whb_frontal_supc_clus", species="human", role="primary"
    )
    with pytest.raises(ValidationError, match="exactly one primary"):
        AnnotationConfig(
            references=[
                whb,
                AnnotationReferenceSpec(
                    reference_id="custom_whb", species="human", role="primary"
                ),
            ]
        )
    with pytest.raises(ValidationError, match="exactly one primary"):
        AnnotationConfig(
            references=[
                AnnotationReferenceSpec(
                    reference_id="seaad_mr_panel", species="human", role="secondary"
                )
            ]
        )
    with pytest.raises(ValidationError, match="duplicated reference ids"):
        AnnotationConfig(references=[whb, whb])
    with pytest.raises(ValidationError, match="not mouse references"):
        AnnotationConfig(species="mouse", references=[whb])
    with pytest.raises(ValidationError, match="role 'primary'"):
        AnnotationReferenceSpec(
            reference_id="whb_frontal_supc_clus", species="human", role="secondary"
        )
    with pytest.raises(ValidationError, match="mouse reference"):
        AnnotationReferenceSpec(
            reference_id="wmb_panel", species="human", role="primary"
        )
    with pytest.raises(ValidationError, match="lower-case token"):
        AnnotationReferenceSpec(
            reference_id="WHB/frontal", species="human", role="primary"
        )


def test_known_references_are_consistent() -> None:
    for reference_id in KNOWN_REFERENCES:
        known = KNOWN_REFERENCES[reference_id]
        spec = AnnotationReferenceSpec(reference_id=reference_id, **known)
        assert spec.n_per_utility == 30
        assert spec.bootstrap_factor == 0.5
        assert spec.bootstrap_iteration == 100
        assert spec.rng_seed == 0
    for species in ("human", "mouse"):
        specs = default_references(species)  # type: ignore[arg-type]
        assert [spec.role for spec in specs].count("primary") == 1


def test_depth_grids() -> None:
    assert default_depth_grid("human") == list(HUMAN_DEPTH_GRID)
    assert default_depth_grid("human", 815) == list(HUMAN_DEPTH_GRID)
    assert default_depth_grid("human", 5001) == list(WIDE_DEPTH_GRID)
    assert default_depth_grid("mouse", 500) == list(WIDE_DEPTH_GRID)
    spec = AnnotationReferenceSpec(
        reference_id="wmb_panel", species="mouse", role="primary"
    )
    assert spec.resolved_depth_grid() == list(WIDE_DEPTH_GRID)
    custom = AnnotationReferenceSpec(
        reference_id="wmb_panel", species="mouse", role="primary", depth_grid=[10, 50]
    )
    assert custom.resolved_depth_grid(5000) == [10, 50]
    for bad in ([], [0, 10], [10, 10], [30, 15]):
        with pytest.raises(ValidationError, match="depth_grid"):
            AnnotationReferenceSpec(
                reference_id="wmb_panel",
                species="mouse",
                role="primary",
                depth_grid=bad,
            )


def test_threshold_defaults_and_validators() -> None:
    thresholds = AnnotationThresholds()
    assert (thresholds.whb_broad, thresholds.whb_supercluster) == (0.73, 0.69)
    assert (thresholds.seaad_broad, thresholds.seaad_subclass_below60) == (0.68, 0.55)
    assert thresholds.seaad_subclass_from60 == 0.45
    assert (thresholds.wmb_class, thresholds.wmb_subclass) == (0.90, 0.80)
    assert thresholds.allow_fine_levels is False
    assert thresholds.provisional_target(0.85, below60=False) == pytest.approx(0.90)
    assert thresholds.provisional_target(0.85, below60=True) == pytest.approx(0.95)
    assert thresholds.provisional_target(0.90, below60=True) == pytest.approx(0.97)
    with pytest.raises(ValidationError, match=r"whb_broad must lie in \[0, 1\]"):
        AnnotationThresholds(whb_broad=1.2)
    with pytest.raises(ValidationError, match="below60"):
        AnnotationThresholds(
            provisional_target_margin=0.1, provisional_target_margin_below60=0.05
        )


def test_panel_config_validators() -> None:
    panel = AnnotationPanelConfig()
    assert panel.panel_mode == "auto"
    assert panel.control_feature_types_keep == ["Gene Expression"]
    assert panel.large_panel_genes == 1000
    with pytest.raises(ValidationError, match="own_family_missing_frac"):
        AnnotationPanelConfig(own_family_missing_frac=0.01)
    with pytest.raises(ValidationError, match="invalid control pattern"):
        AnnotationPanelConfig(extra_control_patterns=["^(Blank"])
    with pytest.raises(ValidationError, match="min_gene_id_resolution"):
        AnnotationPanelConfig(min_gene_id_resolution=1.5)
    # Paths are typed, never checked on disk (the GPU env loads the config).
    missing = Path("/nonexistent/gene.csv")
    assert AnnotationPanelConfig(gene_id_fallback_csv=missing).gene_id_fallback_csv == (
        missing
    )


def test_resolvability_and_gate_p_settings() -> None:
    settings = AnnotationResolvabilityConfig()
    assert settings.recipe == "R1_contam_HO"
    assert settings.threshold_rule == "local_isotonic"
    assert settings.gate_p_seeds == [0, 1]
    assert settings.gate_p_human_donors == ["H19.30.002", "H19.30.001", "H18.30.002"]
    assert settings.gate_p_stress == GatePStressRecipe(
        spill_fraction=0.35, gene_efficiency_sigma=1.0, platform_factor_cap_log2=2.0
    )
    assert settings.gate_p_min_confident_n == 200
    assert settings.gate_p_class_min_test_cells == 700
    with pytest.raises(ValidationError, match="set_level"):
        AnnotationResolvabilityConfig(threshold_rule="set_level")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="gate_p_replicate_min_confident_n"):
        AnnotationResolvabilityConfig(gate_p_replicate_min_confident_n=300)
    with pytest.raises(ValidationError, match="gate_p_seeds"):
        AnnotationResolvabilityConfig(gate_p_seeds=[])
    with pytest.raises(ValidationError, match="spill_fraction"):
        GatePStressRecipe(spill_fraction=1.5)


def test_gates() -> None:
    gate = AnnotationGate()
    assert (gate.min_frac_ge30, gate.min_table_broad_coverage) == (0.30, 0.25)
    assert (gate.warn_segmented_broad_coverage, gate.warn_unvalidated_share) == (
        0.15,
        0.10,
    )
    mouse = MouseGateConfig()
    assert (mouse.g1_density_ratio_fail, mouse.g1_density_ratio_warn) == (1.5, 2.0)
    assert (mouse.g2_marker_consistency_fail, mouse.g5_spillover_warn) == (0.70, 0.15)
    with pytest.raises(ValidationError, match="g1_density_ratio_warn"):
        MouseGateConfig(g1_density_ratio_warn=1.2)
    with pytest.raises(ValidationError, match="g2_marker_consistency_warn"):
        MouseGateConfig(g2_marker_consistency_warn=0.6)
    with pytest.raises(ValidationError, match="min_frac_ge30"):
        AnnotationGate(min_frac_ge30=-0.1)


def test_flag_defaults() -> None:
    flags = AnnotationFlagsConfig()
    assert flags.contamination_null == "dataset_empirical"
    assert flags.contamination_alpha == 0.01
    assert flags.microglial_spillover_enabled is None
    assert flags.ood_robust_z == -3.0
    with pytest.raises(ValidationError):
        AnnotationFlagsConfig(contamination_null="reference")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AnnotationFlagsConfig(ood_robust_z=1.0)


def test_mouse_region_config() -> None:
    regions = MouseRegionConfig()
    wmb = load_vocab("wmb_class")
    assert set(regions.never_drop_classes) == {
        name for name in wmb.names if wmb.is_never_drop(name)
    }
    assert regions.coupled_regions == {}
    assert regions.tile_um == 150.0
    assert MouseRegionConfig(coupled_regions={"OB": ["OLF"]}).coupled_regions == {
        "OB": ["OLF"]
    }
    with pytest.raises(ValidationError, match="unknown regions"):
        MouseRegionConfig(coupled_regions={"OB": ["AON"]})
    with pytest.raises(ValidationError, match="ap_scope_min_mm"):
        MouseRegionConfig(ap_scope_min_mm=11.0)
    assert "OB" in MOUSE_CCF_REGIONS and len(MOUSE_CCF_REGIONS) == 13


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        (" AUTO ", "auto"),
        ("None", "none"),
        ("isocortex; HPF;hpf", "Isocortex;HPF"),
        ("TH;;HY", "TH;HY"),
    ],
)
def test_normalise_mouse_section_regions(value: str, expected: str) -> None:
    assert normalise_mouse_section_regions(value) == expected


@pytest.mark.parametrize("value", ["", " ", ";", "Isocortex;Cortex"])
def test_normalise_mouse_section_regions_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        normalise_mouse_section_regions(value)


def test_config_section_regions_are_normalised() -> None:
    config = AnnotationConfig(species="mouse", mouse_section_regions="hpf;TH")
    assert config.mouse_section_regions == "HPF;TH"
    with pytest.raises(ValidationError, match="unknown mouse section region"):
        AnnotationConfig(species="mouse", mouse_section_regions="Striatum")


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_mode_resolution(species: str) -> None:
    per_species = {f"mode_{species}": "map_first"}
    assert resolve_clustering_mode(species) == "legacy"  # type: ignore[arg-type]
    assert resolve_clustering_mode(species, **per_species) == "map_first"  # type: ignore[arg-type]
    assert (
        resolve_clustering_mode(species, mode="legacy", **per_species)  # type: ignore[arg-type]
        == "legacy"
    )
    assert resolve_clustering_mode(species, mode="map_first") == "map_first"  # type: ignore[arg-type]
    assert resolve_clustering_mode(species, mode="") == "legacy"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clustering mode"):
        resolve_clustering_mode(species, mode="denovo")  # type: ignore[arg-type]
    other = "mouse" if species == "human" else "human"
    assert (
        resolve_clustering_mode(species, **{f"mode_{other}": "map_first"})  # type: ignore[arg-type]
        == "legacy"
    )
    assert CLUSTERING_MODES == ("legacy", "map_first")


@pytest.mark.parametrize(
    ("mode", "flipped", "expected"),
    [
        ("legacy", False, ""),
        ("legacy", True, ""),
        ("map_first", False, MAP_FIRST_TABLE_KEY_SUFFIX),
        ("map_first", True, ""),
    ],
)
def test_table_key_suffix_in_all_mode_and_flip_combinations(
    mode: str, flipped: bool, expected: str
) -> None:
    flipped_species = frozenset({"human"}) if flipped else frozenset()
    assert (
        resolve_table_key_suffix("human", mode, flipped_species=flipped_species)
        == expected
    )


def test_explicit_table_key_suffix() -> None:
    assert resolve_table_key_suffix("mouse", "map_first", "trial2") == "trial2"
    assert resolve_table_key_suffix("mouse", "map_first", "") == ""
    assert resolve_table_key_suffix("mouse", "legacy", "trial2") == ""
    with pytest.raises(ValueError, match="lower-case token"):
        resolve_table_key_suffix("mouse", "map_first", "Trial/2")
    with pytest.raises(ValueError, match="unknown clustering mode"):
        resolve_table_key_suffix("mouse", "denovo")


def test_json_round_trip_and_unknown_fields() -> None:
    config = AnnotationConfig(
        species="mouse",
        enabled=True,
        reference_store=Path("/store"),
        mouse_section_regions="Isocortex;HPF",
    )
    restored = AnnotationConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnnotationConfig(enable=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnnotationThresholds(never_emit_levels=["cluster"])  # type: ignore[call-arg]
