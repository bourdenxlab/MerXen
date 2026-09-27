"""Configuration models for reference-based annotation (plan §3.7).

``merxen.config`` imports these models for the clustering configuration
(hook H8); RESOLVE and MAP read them from ``annotation_config.json``.

Validators never touch the filesystem: the GPU clustering environment loads
the configuration too, and the paths it names may not exist there. Paths are
only typed; the pipeline checks them in preflight and at use.

Defaults are inert: annotation is disabled and the clustering mode of both
species stays ``legacy`` until that species' flip (M8 human, M9 mouse), so
the legacy pipeline is unchanged. ``resolve_clustering_mode`` and
``resolve_table_key_suffix`` mirror ``workflows/lib/AnnotationDefaults.groovy``.
The clustering-mode types and ``AdaptiveSplitConfig`` are the fields that
``ClusteringSquidpyConfig`` and ``MenderConfig`` gain at hook H8.

This module imports only the standard library, numpy, pandas and pydantic
(numpy and pandas through ``merxen.annotation.schema`` and ``vocab``), so the
GPU clustering environment can load it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from merxen.annotation.schema import ReferenceRole
from merxen.annotation.vocab import SPECIES, Species
from merxen.table_keys import validate_table_key_suffix

ClusteringMode = Literal["legacy", "map_first"]
CLUSTERING_MODES: Final[tuple[str, ...]] = ("legacy", "map_first")
# Leaves of the map_first hierarchy: confident reference nodes (v1) or, from
# v1.1 (M11), the opt-in de novo split inside confident superclusters (§6.4).
LeafSource = Literal["mapped", "denovo"]
# MENDER treatment of unassigned cells (§4.9): legacy keeps them as a state;
# map_first runs exclude them from the neighbourhood features (OD-B4).
UnassignedStatePolicy = Literal["state", "exclude_from_features"]
LEGACY_UNASSIGNED_STATE_POLICY: Final = "state"
MAP_FIRST_UNASSIGNED_STATE_POLICY: Final = "exclude_from_features"
# A clustered table-key suffix: "" or one lower-case token (§4.8).
TableKeySuffix = Annotated[str, AfterValidator(validate_table_key_suffix)]
# Default clustering mode per species. Only the flip PRs (M8 human, M9 mouse)
# change these, together with ``FLIPPED_SPECIES``.
DEFAULT_CLUSTERING_MODE: Final[dict[str, str]] = {"human": "legacy", "mouse": "legacy"}
FLIPPED_SPECIES: Final[frozenset[str]] = frozenset()
# Table-key suffix of a map_first run of a species that has not flipped yet, so
# the legacy clustered table is never overwritten (OD-A3, plan §4.8).
MAP_FIRST_TABLE_KEY_SUFFIX: Final = "mapfirst"

# Human anatomical regions with validated references and vocab columns
# (OD-C7: frontal cortex only; a new region needs §8.9's work first).
VALIDATED_HUMAN_REGIONS: Final[tuple[str, ...]] = ("frontal_cortex",)
DEFAULT_HUMAN_REGION: Final = "frontal_cortex"
# Grey-matter CCF divisions of the mouse region model, with OB split from OLF
# (E7 ``GREYREG``); ``mouse_section_regions`` names a subset of these.
MOUSE_CCF_REGIONS: Final[tuple[str, ...]] = (
    "Isocortex",
    "OLF",
    "OB",
    "HPF",
    "CTXsp",
    "STR",
    "PAL",
    "TH",
    "HY",
    "MB",
    "P",
    "MY",
    "CB",
)
MOUSE_SECTION_REGION_KEYWORDS: Final[tuple[str, ...]] = ("auto", "none")

# Depth grids for resolvability and floors (plan §3.7, §8.3).
HUMAN_DEPTH_GRID: Final[tuple[int, ...]] = (10, 15, 30, 60, 120, 250)
WIDE_DEPTH_GRID: Final[tuple[int, ...]] = (10, 20, 50, 100, 250, 500, 1000, 2000)
LARGE_PANEL_GENES: Final = 1000

DEFAULT_REFERENCE_IDS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("whb_frontal_supc_clus", "seaad_mr_panel"),
    "mouse": ("wmb_panel", "wmb_region_share"),
}
# Species, role and taxonomy settings of the references the plan defines
# (§3.2). ``hierarchy`` is left empty where the bundle builder reads it from the
# precompute.
KNOWN_REFERENCES: Final[dict[str, dict[str, Any]]] = {
    "whb_frontal_supc_clus": {
        "species": "human",
        "role": "primary",
        "hierarchy": ["CCN202210140_SUPC", "CCN202210140_CLUS"],
    },
    "whb_frontal_supc_clus_ho": {
        "species": "human",
        "role": "resolvability",
        "hierarchy": ["CCN202210140_SUPC", "CCN202210140_CLUS"],
    },
    "seaad_mr_panel": {"species": "human", "role": "secondary"},
    "whb_whole_ctx_panel": {"species": "human", "role": "sensitivity"},
    "ll_whb_ctx_profiles": {"species": "human", "role": "likelihood"},
    "wmb_panel": {
        "species": "mouse",
        "role": "primary",
        "drop_level": "CCN20230722_SUPT",
    },
    "wmb_selfmap_testset": {"species": "mouse", "role": "resolvability"},
    "wmb_region_share": {"species": "mouse", "role": "region_share"},
}
_REFERENCE_ID_PATTERN: Final = re.compile(r"^[a-z0-9_]+$")


def _check_species(species: str) -> str:
    if species not in SPECIES:
        raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
    return species


def default_depth_grid(species: Species, n_panel_genes: int | None = None) -> list[int]:
    """Return the resolvability depth grid of a species and panel size.

    Args:
        species: ``"human"`` or ``"mouse"``.
        n_panel_genes: Declared panel size; panels above ``LARGE_PANEL_GENES``
            use the wide grid for either species.

    Returns:
        Human panels up to 1,000 genes: ``[10, 15, 30, 60, 120, 250]``; mouse
        and panels above 1,000 genes: ``[10, 20, 50, 100, 250, 500, 1000,
        2000]``.
    """
    _check_species(species)
    is_large = n_panel_genes is not None and n_panel_genes > LARGE_PANEL_GENES
    if species == "human" and not is_large:
        return list(HUMAN_DEPTH_GRID)
    return list(WIDE_DEPTH_GRID)


def _check_fraction(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


class _AnnotationModel(BaseModel):
    """Base model: unknown fields rejected, so typos in JSON fail loudly."""

    model_config = ConfigDict(extra="forbid")


class AnnotationReferenceSpec(_AnnotationModel):
    """One reference bundle to build or use (plan §3.2, §3.7).

    Attributes:
        reference_id: Store id (lower-case token), e.g.
            ``"whb_frontal_supc_clus"``.
        species: ``"human"`` or ``"mouse"``.
        role: How RESOLVE uses the reference.
        sources: Source files by name (checked in preflight, not here).
        hierarchy: Taxonomy levels mapped, coarse to fine; empty when the
            builder reads them from the precompute.
        nodes_to_drop: Nodes removed from the tree.
        drop_level: Level removed from the tree (mouse ``CCN20230722_SUPT``).
        n_per_utility: MapMyCells query markers per utility.
        max_cells_per_cluster: Cells per cluster in a self-built precompute.
        bootstrap_factor: MapMyCells bootstrap factor.
        bootstrap_iteration: MapMyCells bootstrap iterations.
        rng_seed: Mapping seed.
        depth_grid: Resolvability depth grid; ``None`` selects the species
            and panel-size default (``resolved_depth_grid``).
    """

    reference_id: str
    species: Species
    role: ReferenceRole
    sources: dict[str, Path] = Field(default_factory=dict)
    hierarchy: list[str] = Field(default_factory=list)
    nodes_to_drop: list[str] = Field(default_factory=list)
    drop_level: str | None = None
    n_per_utility: int = Field(default=30, ge=1)
    max_cells_per_cluster: int = Field(default=50, ge=1)
    bootstrap_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    bootstrap_iteration: int = Field(default=100, ge=1)
    rng_seed: int = 0
    depth_grid: list[int] | None = None

    @field_validator("reference_id")
    @classmethod
    def _check_reference_id(cls: type[AnnotationReferenceSpec], value: str) -> str:
        if not _REFERENCE_ID_PATTERN.fullmatch(value):
            raise ValueError(
                f"reference_id {value!r} must be a lower-case token "
                "(letters, digits, underscores)"
            )
        return value

    @field_validator("depth_grid")
    @classmethod
    def _check_depth_grid(
        cls: type[AnnotationReferenceSpec], value: list[int] | None
    ) -> list[int] | None:
        if value is None:
            return None
        if not value or any(depth < 1 for depth in value):
            raise ValueError("depth_grid must hold positive counts")
        if any(
            later <= earlier for earlier, later in zip(value, value[1:], strict=False)
        ):
            raise ValueError("depth_grid must be strictly increasing")
        return value

    @model_validator(mode="after")
    def _check_known_reference(
        self: AnnotationReferenceSpec,
    ) -> AnnotationReferenceSpec:
        known = KNOWN_REFERENCES.get(self.reference_id)
        if known is None:
            return self
        if known["species"] != self.species:
            raise ValueError(
                f"reference {self.reference_id!r} is a {known['species']} reference"
            )
        if known["role"] != self.role:
            raise ValueError(
                f"reference {self.reference_id!r} has role {known['role']!r}, "
                f"not {self.role!r}"
            )
        return self

    def resolved_depth_grid(self, n_panel_genes: int | None = None) -> list[int]:
        """Return the explicit depth grid or the species default.

        Args:
            n_panel_genes: Declared panel size (selects the wide grid above
                1,000 genes).

        Returns:
            The depth grid.
        """
        if self.depth_grid is not None:
            return list(self.depth_grid)
        return default_depth_grid(self.species, n_panel_genes)


def default_references(species: Species) -> list[AnnotationReferenceSpec]:
    """Return the default reference specs of a species (plan §3.7 params).

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Human: WHB frontal (primary) and SEA-AD Multiregion (secondary).
        Mouse: WMB panel (primary) and the WMB region shares.
    """
    specs = []
    for reference_id in DEFAULT_REFERENCE_IDS[_check_species(species)]:
        known = dict(KNOWN_REFERENCES[reference_id])
        specs.append(AnnotationReferenceSpec(reference_id=reference_id, **known))
    return specs


def expand_known_references(value: Any) -> Any:
    """Complete bare reference ids and partial specs of known references.

    The pipeline writes ``annotation_config.json`` from its params and names
    the references by id only (``annotation_<species>_references``), so the
    taxonomy settings of a known reference (species, role, hierarchy,
    ``drop_level``) come from ``KNOWN_REFERENCES`` in one place. A string
    becomes that reference's known spec; a mapping keeps its own keys and
    takes the missing ones from the known spec. Anything else, and every
    unknown id, is returned unchanged for normal validation.

    Args:
        value: The raw ``references`` value of an ``AnnotationConfig``.

    Returns:
        The value with known ids expanded to spec mappings.
    """
    if not isinstance(value, list | tuple):
        return value
    expanded: list[Any] = []
    for item in value:
        if isinstance(item, str) and item in KNOWN_REFERENCES:
            expanded.append({"reference_id": item, **KNOWN_REFERENCES[item]})
        elif isinstance(item, dict) and item.get("reference_id") in KNOWN_REFERENCES:
            expanded.append({**KNOWN_REFERENCES[item["reference_id"]], **item})
        else:
            expanded.append(item)
    return expanded


class AnnotationThresholds(_AnnotationModel):
    """Confidence thresholds, targets and emission limits (plan §3.7, §5.2).

    Raw thresholds are the validated defaults (validated panels) and the
    minimum elsewhere; a panel without real-data validation may raise them
    (``threshold_source = resolvability_local``), never lower them.

    Attributes:
        mode: ``"raw"`` (v1) or ``"simulation_calibrated"`` (v1.1).
        whb_broad: WHB broad / lineage bootstrap threshold.
        whb_supercluster: WHB supercluster threshold.
        seaad_broad: SEA-AD broad threshold ("confidently disagrees").
        seaad_subclass_below60: SEA-AD subclass threshold below 60 counts.
        seaad_subclass_from60: SEA-AD subclass threshold from 60 counts.
        wmb_class: WMB class threshold.
        wmb_subclass: WMB subclass threshold.
        target_lineage: Precision target at lineage.
        target_broad: Precision target at broad.
        target_nt: Precision target at NT.
        target_class: Precision target at WMB class.
        target_supercluster: Precision target at WHB supercluster.
        target_subclass: Precision target at WMB subclass.
        provisional_target_margin: Target margin for provisional and
            simulation-validated panels at 60 counts or more.
        provisional_target_margin_below60: The margin below 60 counts.
        provisional_target_cap: Cap on margin-raised targets.
        provisional_mouse_subclass_floor: Mouse subclass floor while
            provisional.
        second_vote_below_counts: Below this depth the second method must
            agree (human).
        floors_path: Floor table; ``None`` uses the packaged
            ``floors_<species>.csv``.
        max_leaf_level: Deepest leaf level (replaces ``never_emit_levels``);
            ``None`` selects the species default.
        allow_fine_levels: Emit the report-only fine level (OD-E4).
        hard_min_counts: Hard floor; always equal to
            ``AnnotationConfig.min_counts``, which is the clustering
            ``min_counts`` once coupled (``None`` until then; read it through
            ``AnnotationConfig.require_min_counts``).
    """

    mode: Literal["raw", "simulation_calibrated"] = "raw"
    whb_broad: float = 0.73
    whb_supercluster: float = 0.69
    seaad_broad: float = 0.68
    seaad_subclass_below60: float = 0.55
    seaad_subclass_from60: float = 0.45
    wmb_class: float = 0.90
    wmb_subclass: float = 0.80
    target_lineage: float = 0.90
    target_broad: float = 0.90
    target_nt: float = 0.90
    target_class: float = 0.90
    target_supercluster: float = 0.85
    target_subclass: float = 0.85
    provisional_target_margin: float = 0.05
    provisional_target_margin_below60: float = 0.10
    provisional_target_cap: float = 0.97
    provisional_mouse_subclass_floor: int = Field(default=60, ge=1)
    second_vote_below_counts: int = Field(default=60, ge=1)
    floors_path: Path | None = None
    max_leaf_level: Literal["supercluster", "subclass"] | None = None
    allow_fine_levels: bool = False
    hard_min_counts: int | None = Field(default=None, ge=0)

    @field_validator(
        "whb_broad",
        "whb_supercluster",
        "seaad_broad",
        "seaad_subclass_below60",
        "seaad_subclass_from60",
        "wmb_class",
        "wmb_subclass",
        "target_lineage",
        "target_broad",
        "target_nt",
        "target_class",
        "target_supercluster",
        "target_subclass",
        "provisional_target_margin",
        "provisional_target_margin_below60",
        "provisional_target_cap",
    )
    @classmethod
    def _check_probability(
        cls: type[AnnotationThresholds], value: float, info: Any
    ) -> float:
        return _check_fraction(value, info.field_name)

    @model_validator(mode="after")
    def _check_margins(self: AnnotationThresholds) -> AnnotationThresholds:
        if self.provisional_target_margin_below60 < self.provisional_target_margin:
            raise ValueError(
                "provisional_target_margin_below60 must be at least "
                "provisional_target_margin (simulation is more optimistic "
                "below 60 counts)"
            )
        return self

    def provisional_target(self, target: float, *, below60: bool) -> float:
        """Return a target raised by the provisional margin and capped.

        Args:
            target: Validated-panel precision target.
            below60: Whether the depth bin lies below 60 counts.

        Returns:
            ``min(target + margin, provisional_target_cap)``.
        """
        margin = (
            self.provisional_target_margin_below60
            if below60
            else self.provisional_target_margin
        )
        return min(target + margin, self.provisional_target_cap)


class AnnotationPanelConfig(_AnnotationModel):
    """Panel resolution, trust-state and large-panel settings (plan §8).

    Attributes:
        panel_mode: ``auto`` (intersection when the platform panels have
            Jaccard at least ``intersection_min_jaccard``), ``intersection``
            or ``per_platform``.
        intersection_min_jaccard: ``auto`` threshold for ``intersection``.
        min_gene_id_resolution: Share of non-control features that must
            resolve to IDs of the run species, else ``refused``.
        min_native_prefix_share: Share of native IDs that must carry the run
            species' prefix (ENSG / ENSMUSG).
        max_other_species_prefix_share: Share of IDs with the other species'
            prefix above which the panel is ``refused``.
        min_mapped_genes: Fewer mapped genes make the panel ``refused``.
        min_root_markers: Fewer root markers make the panel ``refused``.
        weak_parent_markers: Parents with fewer markers are weak.
        trust_max_depth: Deepest grid depth the trust rules consider.
        broad_only_min_class_share: Share of broad classes (with enough test
            cells) whose leaf level must be resolvable, else ``broad_only``.
        gene_id_fallback_csv: Local reference ``gene.csv`` for symbol
            fallback (M0e); the run species' table when ``gene_tables`` has
            none.
        gene_tables: Local gene tables by species (WHB / WMB ``gene.csv`` or
            a reference ``.h5ad`` ``var``; no network): the run species'
            table is the symbol fallback and release-drift reference, and
            both species' tables feed the exact-case species test (§8.4).
        gene_alias_table: Optional local alias table (no network).
        gene_id_overrides_csv: Curated symbol-to-ID overrides.
        control_feature_types_keep: Feature types kept as genes.
        extra_control_patterns: Extra control-name regular expressions.
        species_exact_case_min_ratio: Exact-case / case-insensitive match
            ratio the run species needs (species check, §8.4).
        family_min_jaccard: Jaccard for panel-family trust inheritance.
        subset_bundle_missing_frac: Missing-gene share that triggers a subset
            bundle.
        own_family_missing_frac: Missing-gene share that makes the subset its
            own family.
        large_panel_genes: Panels above this size use the large-panel guards.
        large_panel_marker_prefilter: Marker prefilter for large panels.
        large_panel_prefilter_cap: Gene cap of the prefilter.
        validated_panels_path: Validated families; ``None`` uses the packaged
            table (added in M3b).
        setc_max_abs_log2_deviation: Set c drops a set-a gene whose pseudobulk
            ``log2(mean X / mean M)`` lies further than this from the pair
            median (plan §3.2).
        setc_log2_pseudocount: Pseudocount added to both pseudobulk means
            (per-cell mean counts) before the log2 ratio.
        xplat_min_intersection_genes: A ``per_platform`` intersection panel
            with fewer genes supports broad-level cross-platform statistics
            only (plan §8.5).
    """

    panel_mode: Literal["auto", "intersection", "per_platform"] = "auto"
    intersection_min_jaccard: float = 0.9
    min_gene_id_resolution: float = 0.95
    min_native_prefix_share: float = 0.95
    max_other_species_prefix_share: float = 0.05
    min_mapped_genes: int = Field(default=50, ge=1)
    min_root_markers: int = Field(default=10, ge=1)
    weak_parent_markers: int = Field(default=5, ge=1)
    trust_max_depth: int = Field(default=250, ge=1)
    broad_only_min_class_share: float = 0.5
    gene_id_fallback_csv: Path | None = None
    gene_tables: dict[Species, Path] = Field(default_factory=dict)
    gene_alias_table: Path | None = None
    gene_id_overrides_csv: Path | None = None
    control_feature_types_keep: list[str] = Field(
        default_factory=lambda: ["Gene Expression"]
    )
    extra_control_patterns: list[str] = Field(default_factory=list)
    species_exact_case_min_ratio: float = 0.5
    family_min_jaccard: float = 0.95
    subset_bundle_missing_frac: float = 0.01
    own_family_missing_frac: float = 0.05
    large_panel_genes: int = Field(default=LARGE_PANEL_GENES, ge=1)
    large_panel_marker_prefilter: Literal["per_parent_topk_union", "none"] = (
        "per_parent_topk_union"
    )
    large_panel_prefilter_cap: int = Field(default=2000, ge=1)
    validated_panels_path: Path | None = None
    setc_max_abs_log2_deviation: float = Field(default=2.0, gt=0.0)
    setc_log2_pseudocount: float = Field(default=1e-3, gt=0.0)
    xplat_min_intersection_genes: int = Field(default=100, ge=1)

    @field_validator(
        "intersection_min_jaccard",
        "min_gene_id_resolution",
        "min_native_prefix_share",
        "max_other_species_prefix_share",
        "broad_only_min_class_share",
        "species_exact_case_min_ratio",
        "family_min_jaccard",
        "subset_bundle_missing_frac",
        "own_family_missing_frac",
    )
    @classmethod
    def _check_share(
        cls: type[AnnotationPanelConfig], value: float, info: Any
    ) -> float:
        return _check_fraction(value, info.field_name)

    @field_validator("extra_control_patterns")
    @classmethod
    def _check_patterns(
        cls: type[AnnotationPanelConfig], value: list[str]
    ) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"invalid control pattern {pattern!r}: {error}"
                ) from error
        return value

    @model_validator(mode="after")
    def _check_missing_fracs(self: AnnotationPanelConfig) -> AnnotationPanelConfig:
        if self.own_family_missing_frac <= self.subset_bundle_missing_frac:
            raise ValueError(
                "own_family_missing_frac must exceed subset_bundle_missing_frac"
            )
        return self


class GatePStressRecipe(_AnnotationModel):
    """Gate-P stress recipe (plan §3.7, §14; acceptance only).

    Attributes:
        spill_fraction: Foreign-class spill.
        gene_efficiency_sigma: Sigma of the LogNormal(0, sigma) efficiency.
        platform_factor_cap_log2: Cap on the drawn cross-platform gene
            factors (log2).
    """

    spill_fraction: float = 0.35
    gene_efficiency_sigma: float = Field(default=1.0, gt=0.0)
    platform_factor_cap_log2: float = Field(default=2.0, gt=0.0)

    @field_validator("spill_fraction")
    @classmethod
    def _check_spill(cls: type[GatePStressRecipe], value: float) -> float:
        return _check_fraction(value, "spill_fraction")


class AnnotationResolvabilityConfig(_AnnotationModel):
    """Resolvability self-map and gate-P settings (plan §3.7, §8.3, §8.8).

    Attributes:
        enabled: Run the self-map in reference prep.
        recipe: Simulation recipe (E2 ``R1_contam_HO``).
        gene_efficiency_sigma: Sigma of the per-gene LogNormal efficiency.
        spill_fraction: Foreign-broad-class spill.
        n_test_cells: Test cells; ``None`` selects the species default
            (human 25,000; mouse the 11,913 self-map cells).
        mouse_nonneuronal_extra_per_supertype: Extra mouse test cells per
            non-neuronal supertype.
        holdout_donor: Human held-out donor (``auto`` = the frontal donor
            with the fewest cells, H19.30.002).
        threshold_rule: Always ``local_isotonic`` (never the set-level rule,
            E2 verdict 3).
        threshold_cap: Cap on local thresholds.
        min_cells_per_bin: Test cells a (class, depth) bin needs.
        min_confident_n: Confident calls a tested set needs (a depth bin, else
            the pooled deep set of §8.3; user decision 2026-09-27).
        wilson_margin: Wilson lower bound may sit this far below the target.
        split_halves: Choose thresholds on one half, check on the other.
        reweight_to_composition: Reweight to the dataset composition in
            RESOLVE.
        weight_min_type_cells: Test cells a truth type needs in a depth bin
            to be reweighted on its own; rarer types take their broad
            class's weight (§8.3 composition reweighting, M3b review).
        weight_trim_factor: Composition weights are capped at this multiple
            of their depth bin's median weight (``0``: no cap).
        composition_min_bin_cells: Depth bins with fewer dataset cells are
            reweighted to the dataset's overall composition.
        min_coverage: Coverage an emitted bin needs.
        seed_stability_max_change: Seed-1 change allowed for fine levels.
        gate_p_seeds: Gate-P seeds.
        gate_p_human_donors: Gate-P leave-one-donor-out donors.
        gate_p_mouse_test_draws: Gate-P disjoint mouse test draws.
        gate_p_stress: Gate-P stress recipe.
        gate_p_min_confident_n: Confident calls per tested set (pooled).
        gate_p_replicate_min_confident_n: Confident calls per replicate.
        gate_p_class_min_test_cells: Test cells a class needs to enter C_P.
        gate_p_topup_max_cluster_frac: Top-up cap per cluster (mouse).
        gate_p_spread_se_multiplier: Replicate spread allowed, in standard
            errors.
        gate_p_min_coverage: Coverage gate P requires.
    """

    enabled: bool = True
    recipe: Literal["R1_contam_HO"] = "R1_contam_HO"
    gene_efficiency_sigma: float = Field(default=0.8, gt=0.0)
    spill_fraction: float = 0.25
    n_test_cells: int | None = Field(default=None, ge=1)
    mouse_nonneuronal_extra_per_supertype: int = Field(default=30, ge=0)
    holdout_donor: str = "auto"
    threshold_rule: Literal["local_isotonic"] = "local_isotonic"
    threshold_cap: float = 0.99
    min_cells_per_bin: int = Field(default=50, ge=1)
    min_confident_n: int = Field(default=50, ge=1)
    wilson_margin: float = 0.02
    split_halves: bool = True
    reweight_to_composition: bool = True
    weight_min_type_cells: int = Field(default=20, ge=1)
    weight_trim_factor: float = Field(default=10.0, ge=0.0)
    composition_min_bin_cells: int = Field(default=50, ge=0)
    min_coverage: float = 0.2
    seed_stability_max_change: float = 0.02
    gate_p_seeds: list[int] = Field(default_factory=lambda: [0, 1])
    gate_p_human_donors: list[str] = Field(
        default_factory=lambda: ["H19.30.002", "H19.30.001", "H18.30.002"]
    )
    gate_p_mouse_test_draws: int = Field(default=2, ge=1)
    gate_p_stress: GatePStressRecipe = Field(default_factory=GatePStressRecipe)
    gate_p_min_confident_n: int = Field(default=200, ge=1)
    gate_p_replicate_min_confident_n: int = Field(default=100, ge=1)
    gate_p_class_min_test_cells: int = Field(default=700, ge=1)
    gate_p_topup_max_cluster_frac: float = 0.05
    gate_p_spread_se_multiplier: float = Field(default=3.5, gt=0.0)
    gate_p_min_coverage: float = 0.30

    @field_validator("threshold_rule", mode="before")
    @classmethod
    def _reject_set_level(cls: type[AnnotationResolvabilityConfig], value: Any) -> Any:
        if value == "set_level":
            raise ValueError(
                "threshold_rule 'set_level' is not allowed: it missed the target "
                "in 37-50% of E2 combinations; use 'local_isotonic'"
            )
        return value

    @field_validator(
        "spill_fraction",
        "threshold_cap",
        "wilson_margin",
        "min_coverage",
        "seed_stability_max_change",
        "gate_p_topup_max_cluster_frac",
        "gate_p_min_coverage",
    )
    @classmethod
    def _check_share(
        cls: type[AnnotationResolvabilityConfig], value: float, info: Any
    ) -> float:
        return _check_fraction(value, info.field_name)

    @model_validator(mode="after")
    def _check_gate_p(
        self: AnnotationResolvabilityConfig,
    ) -> AnnotationResolvabilityConfig:
        if not self.gate_p_seeds:
            raise ValueError("gate_p_seeds must not be empty")
        if self.gate_p_replicate_min_confident_n > self.gate_p_min_confident_n:
            raise ValueError(
                "gate_p_replicate_min_confident_n cannot exceed gate_p_min_confident_n"
            )
        return self


class AnnotationGate(_AnnotationModel):
    """Human dataset gate (plan §5.4; pre-registered).

    Attributes:
        depth_counts: Counts defining A (share of table cells at this depth
            or more).
        min_frac_ge30: Below this A the level is ``broad_only``.
        min_table_broad_coverage: Below this confident broad coverage of
            table cells the level is ``failed``.
        warn_segmented_broad_coverage: Below this coverage of segmented
            objects the warning flag is set.
        warn_unvalidated_share: Simulation-validated families warn when more
            of a dataset's confident labels fall outside the validated region.
    """

    depth_counts: int = Field(default=30, ge=1)
    min_frac_ge30: float = 0.30
    min_table_broad_coverage: float = 0.25
    warn_segmented_broad_coverage: float = 0.15
    warn_unvalidated_share: float = 0.10

    @field_validator(
        "min_frac_ge30",
        "min_table_broad_coverage",
        "warn_segmented_broad_coverage",
        "warn_unvalidated_share",
    )
    @classmethod
    def _check_share(cls: type[AnnotationGate], value: float, info: Any) -> float:
        return _check_fraction(value, info.field_name)


class MouseGateConfig(_AnnotationModel):
    """Label-free mouse dataset gate G1–G5 (plan §7.6).

    Attributes:
        g1_radius_um: Radius for the transcript density around centroids.
        g1_density_ratio_fail: G1 fails below this density ratio.
        g1_density_ratio_warn: G1 warns below this density ratio.
        g1_shift_fail_um: G1 fails above this shift against the platform's
            own segmentation.
        g2_marker_consistency_fail: G2 fails below this consistency.
        g2_marker_consistency_warn: G2 warns below this consistency.
        g3_implausible_warn: G3 warns above this pre-pruning implausible share.
        g4_astro_epen_band_points: G4 warns when Astro-Epen is further from the
            AP-matched MERFISH window (percentage points).
        g4_immune_band_points: G4 band for Immune (percentage points).
        g5_spillover_warn: G5 warns above this spill-over flag rate.
    """

    g1_radius_um: float = Field(default=4.0, gt=0.0)
    g1_density_ratio_fail: float = Field(default=1.5, gt=0.0)
    g1_density_ratio_warn: float = Field(default=2.0, gt=0.0)
    g1_shift_fail_um: float = Field(default=5.0, gt=0.0)
    g2_marker_consistency_fail: float = 0.70
    g2_marker_consistency_warn: float = 0.80
    g3_implausible_warn: float = 0.03
    g4_astro_epen_band_points: float = Field(default=5.0, gt=0.0)
    g4_immune_band_points: float = Field(default=1.0, gt=0.0)
    g5_spillover_warn: float = 0.15

    @field_validator(
        "g2_marker_consistency_fail",
        "g2_marker_consistency_warn",
        "g3_implausible_warn",
        "g5_spillover_warn",
    )
    @classmethod
    def _check_share(cls: type[MouseGateConfig], value: float, info: Any) -> float:
        return _check_fraction(value, info.field_name)

    @model_validator(mode="after")
    def _check_order(self: MouseGateConfig) -> MouseGateConfig:
        if self.g1_density_ratio_warn < self.g1_density_ratio_fail:
            raise ValueError("g1_density_ratio_warn must be at least the fail ratio")
        if self.g2_marker_consistency_warn < self.g2_marker_consistency_fail:
            raise ValueError(
                "g2_marker_consistency_warn must be at least the fail value"
            )
        return self


class AnnotationRealQcConfig(_AnnotationModel):
    """Downgrade-only QC on real in-house datasets (plan §8.8; rev3).

    Attributes:
        marker_consistency_warn: Marker-referee warning threshold; ``None``
            selects the species default (human 0.75; mouse G2's 0.80).
        marker_consistency_broad_only: Human: below this the dataset becomes
            ``broad_only``.
        paired_broad_jsd_warn: Paired-platform soft broad JSD warning.
        uninformative_strata_warn_frac: Warn when more flag strata are
            uninformative.
        genes_per_count_gap_warn: Warn when native cells carry this much more
            genes than simulated cells.
        prefilter_spotcheck_min_agreement: 5K prefilter spot-check agreement.
        seeded_families_warn_only_until_gate: The seeded real-data families
            only warn until their species gate has merged.
    """

    marker_consistency_warn: float | None = None
    marker_consistency_broad_only: float = 0.70
    paired_broad_jsd_warn: float = 0.20
    uninformative_strata_warn_frac: float = 0.5
    genes_per_count_gap_warn: float = Field(default=0.45, ge=0.0)
    prefilter_spotcheck_min_agreement: float = 0.95
    seeded_families_warn_only_until_gate: bool = True

    @field_validator(
        "marker_consistency_broad_only",
        "paired_broad_jsd_warn",
        "uninformative_strata_warn_frac",
        "prefilter_spotcheck_min_agreement",
    )
    @classmethod
    def _check_share(
        cls: type[AnnotationRealQcConfig], value: float, info: Any
    ) -> float:
        return _check_fraction(value, info.field_name)

    @field_validator("marker_consistency_warn")
    @classmethod
    def _check_warn(
        cls: type[AnnotationRealQcConfig], value: float | None
    ) -> float | None:
        return (
            None if value is None else _check_fraction(value, "marker_consistency_warn")
        )


class AnnotationFlagsConfig(_AnnotationModel):
    """Report-only per-cell flags (plan §4.3, §5.6, §8.6).

    Attributes:
        contamination_null: Null model (``dataset_empirical`` only; the
            reference null over-fires, 17-75% of cells).
        contamination_alpha: Beta-binomial tail probability.
        contamination_min_neg_counts: Negative counts a flagged cell needs.
        contamination_null_depth_quantile: Cells above this depth quantile of
            their class fit the null.
        negative_gene_max_fraction: A gene is negative for a class when
            detected in fewer of its reference cells.
        flag_rate_uninformative_above: Contamination strata above this
            realised rate are uninformative (flag null).
        diffuse_quantile: Quantile of simulated distinct genes.
        diffuse_n_simulations: Simulations per grid point.
        diffuse_rate_uninformative_above: Diffuse strata above this rate are
            uninformative.
        ood_robust_z: Robust z below which a cell is out of distribution.
        microglial_spillover_enabled: ``None`` selects the species default
            (mouse on, human off; OD-C5).
        microglia_stat_min: Spill-over statistic threshold (E3 TAU).
        microglia_weight_min: Spill-over weight threshold.
        microglia_fpr_max: Flag rate allowed among marker astrocytes.
        specific_gene_ratio: Microglia / max non-target class mean ratio of a
            specific gene.
        specific_gene_min_share: Minimum microglial share of a specific gene.
        min_specific_genes: Fewer specific genes make the flag null.
        astro_lowcount_below: Mouse Astro-Epen calls below this depth are
            flagged.
    """

    contamination_null: Literal["dataset_empirical"] = "dataset_empirical"
    contamination_alpha: float = 0.01
    contamination_min_neg_counts: int = Field(default=3, ge=0)
    contamination_null_depth_quantile: float = 0.75
    negative_gene_max_fraction: float = 0.01
    flag_rate_uninformative_above: float = 0.15
    diffuse_quantile: float = 0.95
    diffuse_n_simulations: int = Field(default=200, ge=1)
    diffuse_rate_uninformative_above: float = 0.30
    ood_robust_z: float = Field(default=-3.0, lt=0.0)
    microglial_spillover_enabled: bool | None = None
    microglia_stat_min: float = 10.0
    microglia_weight_min: float = 0.05
    microglia_fpr_max: float = 0.005
    specific_gene_ratio: float = Field(default=20.0, gt=1.0)
    specific_gene_min_share: float = 0.001
    min_specific_genes: int = Field(default=3, ge=1)
    astro_lowcount_below: int = Field(default=100, ge=1)

    @field_validator(
        "contamination_alpha",
        "contamination_null_depth_quantile",
        "negative_gene_max_fraction",
        "flag_rate_uninformative_above",
        "diffuse_quantile",
        "diffuse_rate_uninformative_above",
        "microglia_weight_min",
        "microglia_fpr_max",
        "specific_gene_min_share",
    )
    @classmethod
    def _check_share(
        cls: type[AnnotationFlagsConfig], value: float, info: Any
    ) -> float:
        return _check_fraction(value, info.field_name)


class MouseRegionConfig(_AnnotationModel):
    """Mouse region inference and two-tier pruning (plan §3.7, §7.2; E7).

    M6b may change the tile parameters, ``never_drop_classes`` and
    ``coupled_regions`` (§7.8).

    Attributes:
        tile_um: Tile size.
        min_neurons_per_tile: Confident neurons a tile needs.
        confident_neuron_min_bp: Class and subclass bootstrap probability of a
            confident neuron.
        min_tile_fraction: Share of assigned tiles a present region needs.
        min_component_tiles: Connected tiles a present region needs.
        min_assigned_tiles: Fewer assigned tiles skip pruning (warning).
        class_low_share: Classes with less of their MERFISH cells in present
            regions use the strict subclass rule.
        subclass_share_in_low_class: Subclass share threshold in such classes.
        subclass_share_default: Subclass share threshold otherwise.
        min_merfish_cells_subclass: Subclasses with fewer MERFISH cells follow
            their class.
        never_drop_classes: WMB classes pruning never drops.
        coupled_regions: Region coupling (e.g. ``{"OB": ["OLF"]}``); off until
            M6b.
        coherence_k: Neighbours for region coherence.
        coherence_min: Coherence below which a cell is flagged.
        coherence_max_conf: Only cells below this confidence are flagged.
        ap_scope_min_mm: The report warns in front of this AP estimate.
        ap_scope_max_mm: The report warns behind this AP estimate.
        rule_variant: Region-rule variant (``v1`` until M6b selects one).
    """

    tile_um: float = Field(default=150.0, gt=0.0)
    min_neurons_per_tile: int = Field(default=3, ge=1)
    confident_neuron_min_bp: float = 0.9
    min_tile_fraction: float = 0.01
    min_component_tiles: int = Field(default=3, ge=1)
    min_assigned_tiles: int = Field(default=200, ge=1)
    class_low_share: float = 0.20
    subclass_share_in_low_class: float = 0.30
    subclass_share_default: float = 0.10
    min_merfish_cells_subclass: int = Field(default=20, ge=1)
    never_drop_classes: list[str] = Field(
        default_factory=lambda: [
            "25 Pineal Glut",
            "30 Astro-Epen",
            "31 OPC-Oligo",
            "33 Vascular",
            "34 Immune",
        ]
    )
    coupled_regions: dict[str, list[str]] = Field(default_factory=dict)
    coherence_k: int = Field(default=30, ge=1)
    coherence_min: float = 0.1
    coherence_max_conf: float = 0.8
    ap_scope_min_mm: float = 2.4
    ap_scope_max_mm: float = 10.4
    rule_variant: str = "v1"

    @field_validator(
        "confident_neuron_min_bp",
        "min_tile_fraction",
        "class_low_share",
        "subclass_share_in_low_class",
        "subclass_share_default",
        "coherence_min",
        "coherence_max_conf",
    )
    @classmethod
    def _check_share(cls: type[MouseRegionConfig], value: float, info: Any) -> float:
        return _check_fraction(value, info.field_name)

    @field_validator("coupled_regions")
    @classmethod
    def _check_regions(
        cls: type[MouseRegionConfig], value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        names = {region for key, regions in value.items() for region in [key, *regions]}
        unknown = sorted(names - set(MOUSE_CCF_REGIONS))
        if unknown:
            raise ValueError(
                f"coupled_regions names unknown regions {unknown}; "
                f"known: {MOUSE_CCF_REGIONS}"
            )
        return value

    @model_validator(mode="after")
    def _check_ap_scope(self: MouseRegionConfig) -> MouseRegionConfig:
        if self.ap_scope_min_mm >= self.ap_scope_max_mm:
            raise ValueError("ap_scope_min_mm must be below ap_scope_max_mm")
        return self


def normalise_mouse_section_regions(value: str) -> str:
    """Normalise a ``mouse_section_regions`` value.

    Args:
        value: ``auto``, ``none`` or ``;``-separated CCF divisions (case
            insensitive).

    Returns:
        ``"auto"``, ``"none"`` or the canonical divisions joined by ``;`` in
        input order, without duplicates.

    Raises:
        ValueError: On a blank value or an unknown division.
    """
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("mouse_section_regions must not be blank")
    if cleaned.lower() in MOUSE_SECTION_REGION_KEYWORDS:
        return cleaned.lower()
    canonical = {region.lower(): region for region in MOUSE_CCF_REGIONS}
    regions: list[str] = []
    for part in cleaned.split(";"):
        token = part.strip()
        if not token:
            continue
        if token.lower() not in canonical:
            raise ValueError(
                f"unknown mouse section region {token!r}; use auto, none or "
                f"';'-separated divisions from {MOUSE_CCF_REGIONS}"
            )
        region = canonical[token.lower()]
        if region not in regions:
            regions.append(region)
    if not regions:
        raise ValueError("mouse_section_regions names no region")
    return ";".join(regions)


def species_defaults(species: Species) -> dict[str, Any]:
    """Return the species-dependent annotation defaults.

    ``workflows/lib/AnnotationDefaults.groovy`` mirrors these values; a string
    test compares the two.

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Default clustering mode, max leaf level, spill-over flag switch,
        marker-consistency warning, resolvability test cells, anatomical
        region and reference ids.
    """
    if _check_species(species) == "human":
        return {
            "clustering_mode": DEFAULT_CLUSTERING_MODE["human"],
            "max_leaf_level": "supercluster",
            "microglial_spillover_enabled": False,
            "marker_consistency_warn": 0.75,
            "n_test_cells": 25_000,
            "anatomical_region": DEFAULT_HUMAN_REGION,
            "references": list(DEFAULT_REFERENCE_IDS["human"]),
        }
    return {
        "clustering_mode": DEFAULT_CLUSTERING_MODE["mouse"],
        "max_leaf_level": "subclass",
        "microglial_spillover_enabled": True,
        "marker_consistency_warn": 0.80,
        "n_test_cells": 11_913,
        "anatomical_region": None,
        "references": list(DEFAULT_REFERENCE_IDS["mouse"]),
    }


class AnnotationConfig(_AnnotationModel):
    """Configuration of reference-based annotation (``annotation_config.json``).

    Species defaults (``species_defaults``) fill every field left ``None``.

    Attributes:
        enabled: Run annotation (``False`` by default; legacy runs never set
            it).
        species: ``"human"`` or ``"mouse"``.
        anatomical_region: Human region token; only ``frontal_cortex`` is
            validated (OD-C7). Mouse uses ``mouse_section_regions`` instead.
        mouse_section_regions: Default mouse regions: ``auto``, ``none`` or
            ``;``-separated CCF divisions (a samplesheet column overrides it).
        min_counts: Table-cell threshold, equal to the clustering
            ``min_counts``. ``None`` until the config is coupled to the
            clustering run (``ClusteringSquidpyConfig.coupled_annotation_config``
            or ``coupled_to_clustering``); ``require_min_counts`` refuses an
            uncoupled config, so no step can fall back to a default silently.
        references: Reference specs; empty selects the species defaults.
        panel: Panel settings.
        resolvability: Resolvability and gate-P settings.
        thresholds: Thresholds, targets and emission limits.
        gate: Human dataset gate.
        mouse_gate: Mouse dataset gate.
        real_qc: Downgrade-only real-data QC.
        flags: Report-only flags.
        mouse_regions: Mouse region inference and pruning.
        allow_single_method: Human degraded mode: WHB alone below 60 counts.
        xplat_sensitivity: Cross-platform sensitivity runs.
        xplat_sensitivity_segmentations: Segmentations with sensitivity runs.
        reuse_published: Skip mapping when the published map manifest matches.
        keep_extended_json: Keep the gzipped MapMyCells extended JSON.
        ctm_version: Required ``cell_type_mapper`` version.
        reference_store: Reference store; ``None`` resolves to
            ``<outdir>/annotation_references`` in the pipeline.
        reference_store_large: Store for panels above 1,000 genes.
    """

    enabled: bool = False
    species: Species = "human"
    anatomical_region: str | None = None
    mouse_section_regions: str = "auto"
    min_counts: int | None = Field(default=None, ge=0)
    references: list[AnnotationReferenceSpec] = Field(default_factory=list)
    panel: AnnotationPanelConfig = Field(default_factory=AnnotationPanelConfig)
    resolvability: AnnotationResolvabilityConfig = Field(
        default_factory=AnnotationResolvabilityConfig
    )
    thresholds: AnnotationThresholds = Field(default_factory=AnnotationThresholds)
    gate: AnnotationGate = Field(default_factory=AnnotationGate)
    mouse_gate: MouseGateConfig = Field(default_factory=MouseGateConfig)
    real_qc: AnnotationRealQcConfig = Field(default_factory=AnnotationRealQcConfig)
    flags: AnnotationFlagsConfig = Field(default_factory=AnnotationFlagsConfig)
    mouse_regions: MouseRegionConfig = Field(default_factory=MouseRegionConfig)
    allow_single_method: bool = False
    xplat_sensitivity: Literal["off", "geneset_c", "geneset_c+rescale"] = "geneset_c"
    xplat_sensitivity_segmentations: list[str] = Field(
        default_factory=lambda: ["proseg_hybrid"]
    )
    reuse_published: bool = True
    keep_extended_json: bool = False
    ctm_version: str = "1.7.2"
    reference_store: Path | None = None
    reference_store_large: Path | None = None

    @field_validator("references", mode="before")
    @classmethod
    def _expand_known_references(cls: type[AnnotationConfig], value: Any) -> Any:
        return expand_known_references(value)

    @field_validator("mouse_section_regions")
    @classmethod
    def _check_section_regions(cls: type[AnnotationConfig], value: str) -> str:
        return normalise_mouse_section_regions(value)

    @field_validator("anatomical_region")
    @classmethod
    def _normalise_region(cls: type[AnnotationConfig], value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @model_validator(mode="after")
    def _apply_species_defaults(self: AnnotationConfig) -> AnnotationConfig:
        # Pydantic keeps the sub-model instances a caller passes in, so the
        # defaults go into copies: filling the caller's objects would leak one
        # config's species defaults into the next config built from them.
        self.thresholds = self.thresholds.model_copy(deep=True)
        self.flags = self.flags.model_copy(deep=True)
        self.real_qc = self.real_qc.model_copy(deep=True)
        self.resolvability = self.resolvability.model_copy(deep=True)
        defaults = species_defaults(self.species)
        self._check_region(defaults)
        if not self.references:
            self.references = default_references(self.species)
        self._check_references()
        if self.thresholds.max_leaf_level is None:
            self.thresholds.max_leaf_level = defaults["max_leaf_level"]
        elif self.thresholds.max_leaf_level != defaults["max_leaf_level"]:
            raise ValueError(
                f"max_leaf_level {self.thresholds.max_leaf_level!r} is not a "
                f"{self.species} leaf level; leaves stay "
                f"{defaults['max_leaf_level']!r} in v1 (OD-E4)"
            )
        if self.flags.microglial_spillover_enabled is None:
            self.flags.microglial_spillover_enabled = defaults[
                "microglial_spillover_enabled"
            ]
        if self.real_qc.marker_consistency_warn is None:
            self.real_qc.marker_consistency_warn = defaults["marker_consistency_warn"]
        if self.resolvability.n_test_cells is None:
            self.resolvability.n_test_cells = defaults["n_test_cells"]
        self._couple_min_counts()
        return self

    def _check_region(self: AnnotationConfig, defaults: dict[str, Any]) -> None:
        if self.species == "mouse":
            if self.anatomical_region is not None:
                raise ValueError(
                    "anatomical_region is human-only; mouse sections use "
                    "mouse_section_regions"
                )
            return
        if self.anatomical_region is None:
            self.anatomical_region = defaults["anatomical_region"]
        if self.anatomical_region not in VALIDATED_HUMAN_REGIONS:
            raise ValueError(
                f"anatomical_region {self.anatomical_region!r} is not validated; "
                f"human annotation supports {VALIDATED_HUMAN_REGIONS} only (OD-C7). "
                "A new region needs its WHB ROI set, HO bundle, vocab "
                "plausibility column, markers and acceptance first (plan §8.9)."
            )

    def _check_references(self: AnnotationConfig) -> None:
        ids = [spec.reference_id for spec in self.references]
        duplicated = sorted({ref for ref in ids if ids.count(ref) > 1})
        if duplicated:
            raise ValueError(f"duplicated reference ids {duplicated}")
        other = [
            spec.reference_id
            for spec in self.references
            if spec.species != self.species
        ]
        if other:
            raise ValueError(f"references {other} are not {self.species} references")
        primaries = [
            spec.reference_id for spec in self.references if spec.role == "primary"
        ]
        if len(primaries) != 1:
            raise ValueError(
                f"exactly one primary reference is required, got {primaries}"
            )

    def _couple_min_counts(self: AnnotationConfig) -> None:
        hard_floor = self.thresholds.hard_min_counts
        if hard_floor is not None and hard_floor != self.min_counts:
            raise ValueError(
                f"thresholds.hard_min_counts ({hard_floor}) must equal "
                f"min_counts ({self.min_counts}): the table cells and the hard "
                "floor share one threshold, the clustering min_counts (plan "
                "§3.7, §4.4)"
            )
        self.thresholds.hard_min_counts = self.min_counts

    @property
    def is_coupled(self: AnnotationConfig) -> bool:
        """Whether ``min_counts`` was taken from the clustering run."""
        return self.min_counts is not None

    def require_min_counts(self: AnnotationConfig) -> int:
        """Return the table-cell threshold and hard floor of a coupled config.

        Every step that selects table cells or applies the hard floor (MAP,
        RESOLVE, the map_first hierarchy) reads the threshold here, so a
        config that skipped the coupling fails instead of using a default.

        Returns:
            ``min_counts`` (equal to ``thresholds.hard_min_counts``).

        Raises:
            ValueError: If the config was not coupled to the clustering run.
        """
        if self.min_counts is None:
            raise ValueError(
                "the annotation config is not coupled to the clustering run's "
                "min_counts; build it with "
                "ClusteringSquidpyConfig.coupled_annotation_config() or "
                "AnnotationConfig.coupled_to_clustering() (plan §3.7, §4.4)"
            )
        return self.min_counts

    def primary_reference(self) -> AnnotationReferenceSpec:
        """Return the primary reference spec.

        Returns:
            The spec with role ``primary``.
        """
        return next(spec for spec in self.references if spec.role == "primary")

    def with_min_counts(self, min_counts: int) -> AnnotationConfig:
        """Return a copy coupled to a clustering ``min_counts``.

        Args:
            min_counts: ``ClusteringSquidpyConfig.min_counts`` of the run.

        Returns:
            A validated copy whose ``min_counts`` and hard floor equal it.
        """
        data = self.model_dump(mode="python")
        data["min_counts"] = min_counts
        data["thresholds"].pop("hard_min_counts", None)
        return AnnotationConfig.model_validate(data)

    def coupled_to_clustering(self, clustering_min_counts: int) -> AnnotationConfig:
        """Return a copy that uses the clustering run's table-cell threshold.

        The table cells and the hard floor share one threshold (plan §3.7,
        §4.4), so ``min_counts`` and ``thresholds.hard_min_counts`` both
        become ``clustering_min_counts``.

        Args:
            clustering_min_counts: ``ClusteringSquidpyConfig.min_counts``.

        Returns:
            The coupled, validated copy.

        Raises:
            ValueError: If ``min_counts`` is already set to another value.
        """
        if self.min_counts is not None and self.min_counts != clustering_min_counts:
            raise ValueError(
                f"annotation min_counts ({self.min_counts}) must equal the "
                f"clustering min_counts ({clustering_min_counts}): table cells "
                "and the hard floor share one threshold (plan §4.4)"
            )
        return self.with_min_counts(clustering_min_counts)


class AdaptiveSplitConfig(_AnnotationModel):
    """Within-supercluster de novo split of map_first leaves (plan §6.4).

    The opt-in rule ``count_split_merge`` arrives with v1.1 (M11); until then
    the only rule is ``none`` and the leaves are confident reference nodes.

    Attributes:
        rule: Split rule (``none``: no split).
    """

    rule: Literal["none"] = "none"


def check_clustering_mode_settings(
    *,
    mode: str,
    leaf_source: str,
    adaptive_split: AdaptiveSplitConfig,
    table_key_suffix: str,
) -> None:
    """Check the clustering-mode fields of ``ClusteringSquidpyConfig`` together.

    Args:
        mode: ``legacy`` or ``map_first``.
        leaf_source: ``mapped`` or ``denovo``.
        adaptive_split: The de novo split settings.
        table_key_suffix: Clustered table-key suffix (``""`` for none).

    Raises:
        ValueError: On an unknown mode, a legacy run with a table-key suffix
            (legacy runs always write the unsuffixed table; plan §4.8), or
            de novo leaves without a split rule.
    """
    if mode not in CLUSTERING_MODES:
        raise ValueError(
            f"clustering mode must be one of {CLUSTERING_MODES}, got {mode!r}"
        )
    if mode == "legacy" and table_key_suffix:
        raise ValueError(
            f"table_key_suffix {table_key_suffix!r} is for map_first runs only; "
            "legacy runs write the unsuffixed clustered table (plan §4.8)"
        )
    if leaf_source == "denovo" and adaptive_split.rule == "none":
        raise ValueError(
            "leaf_source 'denovo' needs an adaptive_split rule; the de novo "
            "split is opt-in from v1.1 (M11, plan §6.4)"
        )


def resolve_clustering_mode(
    species: Species,
    *,
    mode: str | None = None,
    mode_human: str | None = None,
    mode_mouse: str | None = None,
) -> str:
    """Resolve the clustering mode of a run (mirrors ``resolveMode``).

    As in Groovy, values are case-insensitive and stripped, and a blank value
    counts as unset: a blank override falls through to the species param,
    and a blank species param to the species default.

    Args:
        species: Run species.
        mode: ``clustering_squidpy_mode``, an override for both species.
        mode_human: ``clustering_squidpy_mode_human`` (``None`` = default).
        mode_mouse: ``clustering_squidpy_mode_mouse`` (``None`` = default).

    Returns:
        ``"legacy"`` or ``"map_first"``.

    Raises:
        ValueError: On an unknown species or mode.
    """
    _check_species(species)
    per_species = mode_human if species == "human" else mode_mouse
    chosen = (
        _mode_or_none(mode)
        or _mode_or_none(per_species)
        or DEFAULT_CLUSTERING_MODE[species]
    )
    if chosen not in CLUSTERING_MODES:
        raise ValueError(
            f"clustering mode must be one of {CLUSTERING_MODES}, got {chosen!r}"
        )
    return chosen


def _mode_or_none(value: str | None) -> str | None:
    """Normalise a mode param as Groovy ``blankToNull`` + ``checkedMode`` do."""
    if value is None:
        return None
    return str(value).strip().lower() or None


def resolve_table_key_suffix(
    species: Species,
    resolved_mode: str,
    explicit_suffix: str | None = None,
    *,
    flipped_species: frozenset[str] = FLIPPED_SPECIES,
) -> str:
    """Resolve the clustered table-key suffix (mirrors ``tableKeySuffix``).

    Legacy runs always write the unsuffixed key. A map_first run uses the
    explicit suffix when given, else ``"mapfirst"`` while the species has not
    flipped, else ``""`` (plan §3.7, §4.8; OD-A3). Before the flip an explicit
    empty suffix is refused: the run would write the unsuffixed key and
    overwrite the legacy clustered table.

    Args:
        species: Run species.
        resolved_mode: Output of ``resolve_clustering_mode``.
        explicit_suffix: ``clustering_squidpy_table_key_suffix`` (``None`` =
            automatic).
        flipped_species: Species whose default is already ``map_first``.

    Returns:
        The suffix token (``""`` for none).

    Raises:
        ValueError: On an unknown mode, a suffix that is not a lower-case
            token, or an empty suffix for a map_first run of a species that
            has not flipped.
    """
    _check_species(species)
    if resolved_mode not in CLUSTERING_MODES:
        raise ValueError(f"unknown clustering mode {resolved_mode!r}")
    if resolved_mode == "legacy":
        return ""
    if explicit_suffix is None:
        return "" if species in flipped_species else MAP_FIRST_TABLE_KEY_SUFFIX
    suffix = validate_table_key_suffix(explicit_suffix)
    if not suffix and species not in flipped_species:
        raise ValueError(
            "an empty clustering_squidpy_table_key_suffix would make this "
            "map_first run write the unsuffixed clustered table and overwrite "
            f"the legacy one: {species} has not flipped to map_first yet "
            "(OD-A3, plan §4.8). Leave the suffix unset "
            f"({MAP_FIRST_TABLE_KEY_SUFFIX!r}) or name another token"
        )
    return suffix
