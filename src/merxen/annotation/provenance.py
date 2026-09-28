"""Annotation provenance stored in ``uns`` and the annotation manifest.

``uns["merxen_annotation_json"]`` holds one JSON string of
``AnnotationProvenance`` (plan §4.6); RESOLVE writes the same JSON to
``<sid>_annotation_manifest.json``. A JSON string is safe for both h5ad and
zarr, which reject or silently nest the structures a nested dict would need.
The models still keep the §4.6 rules inside the JSON, so it can be flattened
into ``uns`` later without surprises:

- every mapping key is a safe token (letters, digits, ``_``, ``.``, ``-``):
  never ``/`` (h5ad nests silently, zarr raises) and never a space; per-class
  and per-branch keys go through ``safe_token``;
- no list holds a mapping (h5ad raises ``TypeError`` on lists of dicts);
- floats are finite (JSON has no NaN).

This module imports only the standard library, numpy, pandas and pydantic
(numpy and pandas through ``merxen.annotation.schema`` and ``vocab``).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, MutableMapping
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from merxen.annotation.schema import (
    LABEL_TABLE_VERSION,
    GateLevel,
    PanelMode,
    PanelTrust,
    ReferenceRole,
    ValidationBasis,
)
from merxen.annotation.vocab import Species

# 2: panel diagnostics and trust fields (M3b): family basis, trust reasons,
# banner, validated-table digests, root markers and weak / collapsed parents.
# 3: the consensus (degraded mode, tier counts) and the soft composition of
# the sample (RESOLVE, M4).
PROVENANCE_SCHEMA_VERSION: Final = 3
PROVENANCE_UNS_KEY: Final = "merxen_annotation_json"
ANNOTATION_MANIFEST_SUFFIX: Final = "_annotation_manifest.json"
SAFE_KEY_PATTERN: Final = re.compile(r"^[A-Za-z0-9_.\-]+$")

RealQcOutcome = Literal["pass", "warn", "fail", "not_evaluable"]


def is_safe_key(key: object) -> bool:
    """Return whether a mapping key is safe for h5ad and zarr ``uns``.

    Args:
        key: Candidate key.

    Returns:
        ``True`` for non-empty strings of letters, digits, ``_``, ``.`` and
        ``-`` (so never ``/`` or whitespace).
    """
    return isinstance(key, str) and bool(SAFE_KEY_PATTERN.fullmatch(key))


def unsafe_structure_problems(value: Any, path: str = "$") -> list[str]:
    """List the §4.6 violations in a JSON-like structure.

    Args:
        value: Nested dicts, lists and scalars (e.g. ``model_dump(mode="json")``).
        path: Location prefix used in the messages.

    Returns:
        One message per unsafe key, list-held mapping or non-finite float.
    """
    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not is_safe_key(key):
                problems.append(f"{path}: unsafe key {key!r}")
            problems += unsafe_structure_problems(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for position, item in enumerate(value):
            if isinstance(item, Mapping):
                problems.append(f"{path}[{position}]: list holds a mapping")
            else:
                problems += unsafe_structure_problems(item, f"{path}[{position}]")
    elif isinstance(value, float) and not math.isfinite(value):
        problems.append(f"{path}: non-finite float {value!r}")
    return problems


class _ProvenanceModel(BaseModel):
    """Base model: unknown fields rejected, NaN / inf rejected, safe keys."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_uns_safe(self: _ProvenanceModel) -> _ProvenanceModel:
        problems = unsafe_structure_problems(self.model_dump(mode="json"))
        if problems:
            raise ValueError(
                "provenance is not h5ad/zarr-safe: " + "; ".join(problems[:10])
            )
        return self


class SourceIdentity(_ProvenanceModel):
    """Identity of one reference source file (plan §3.2 ``build_hash``).

    Attributes:
        path: Path as recorded by the store.
        size: Size in bytes.
        mtime_ns: Modification time in nanoseconds.
        sha256: Hex digest.
        sha256_scope: ``"full"`` or ``"head_tail_64mb"`` (first + last 64 MB,
            as in ``build_hash``; the full digest lives in ``bundle.json``).
    """

    path: str
    size: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None
    sha256_scope: Literal["full", "head_tail_64mb"] = "full"


class MarkerProvenance(_ProvenanceModel):
    """Marker lookup used for one reference.

    Attributes:
        lookup_sha256: Digest of the query-marker lookup.
        n_per_utility: MapMyCells ``n_per_utility``.
        n_panel_genes: Panel genes the lookup was built on.
        markers_per_parent_min: Fewest markers of any parent.
        markers_per_parent_median: Median markers per parent.
        prefilter: Large-panel marker prefilter (``None`` when off).
        root_markers: Markers of the taxonomy root (§8.2).
        root_children_separated: Root children with at least 10 markers
            (how many broad classes the root separates).
        n_weak_parents: Parents with fewer than ``weak_parent_markers``.
        n_collapsed_parents: Parents auto-collapsed for lack of markers.
        n_hidden_leaves: Leaves hidden by the collapsed parents.
    """

    lookup_sha256: str | None = None
    n_per_utility: int | None = None
    n_panel_genes: int | None = None
    markers_per_parent_min: int | None = None
    markers_per_parent_median: float | None = None
    prefilter: str | None = None
    root_markers: int | None = None
    root_children_separated: int | None = None
    n_weak_parents: int | None = None
    n_collapsed_parents: int | None = None
    n_hidden_leaves: int | None = None


class ReferenceProvenance(_ProvenanceModel):
    """One reference bundle used by the annotation.

    Attributes:
        reference_id: Store id, e.g. ``"whb_frontal_supc_clus"``.
        role: Reference role.
        taxonomy_id: Allen taxonomy id, e.g. ``"CCN202210140"``.
        levels: Taxonomy levels mapped, coarse to fine.
        n_leaves: Leaves of the mapping tree.
        collapsed_parents: Parents auto-collapsed for lack of markers.
        nodes_dropped: Nodes dropped from the tree (``nodes_to_drop``).
        drop_level: Level dropped from the tree (mouse ``CCN20230722_SUPT``).
        bundle_path: Bundle directory in the reference store.
        build_hash: Bundle ``build_hash``.
        sources: Source file identities keyed by a safe source name.
        n_query_genes_used: Panel genes present in this reference.
        markers: Marker lookup provenance.
        panel_trust: Trust state of this reference on the panel (§8.2).
        trust_reasons: Reason tokens of that state
            (``diagnostics.TrustDecision.reason_codes``).
        n_panel_genes_absent: Panel genes the reference lacks.
    """

    reference_id: str
    role: ReferenceRole
    taxonomy_id: str | None = None
    levels: list[str] = []
    n_leaves: int | None = None
    collapsed_parents: list[str] = []
    nodes_dropped: list[str] = []
    drop_level: str | None = None
    bundle_path: str | None = None
    build_hash: str | None = None
    sources: dict[str, SourceIdentity] = {}
    n_query_genes_used: int | None = None
    markers: MarkerProvenance | None = None
    panel_trust: PanelTrust | None = None
    trust_reasons: list[str] = []
    n_panel_genes_absent: int | None = None

    @field_validator("reference_id")
    @classmethod
    def _check_reference_id(cls: type[ReferenceProvenance], value: str) -> str:
        if not is_safe_key(value):
            raise ValueError(f"reference_id {value!r} is not a safe token")
        return value


class RealQcProvenance(_ProvenanceModel):
    """Downgrade-only real-data QC outcomes (plan §8.8).

    Attributes:
        outcomes: Outcome per check, keyed by a safe check name.
        downgrades: Downgrades applied to this dataset (e.g.
            ``"broad_only:marker_consistency"``).
        warn_only: Whether the checks only warned (seeded families before
            their species gate, ``seeded_families_warn_only_until_gate``).
    """

    outcomes: dict[str, RealQcOutcome] = {}
    downgrades: list[str] = []
    warn_only: bool | None = None


class PanelProvenance(_ProvenanceModel):
    """Panel, family and trust of the annotated dataset (plan §4.6, §8.1–§8.2).

    Attributes:
        panel_hash: sha256 of the sorted resolved IDs of the declared panel.
        panel_family: Family id (``validated_panels.csv``), if any.
        family_basis: How the panel got its family: ``own``, ``listed``
            (its hash is a row of ``validated_panels.csv``), ``inherited``
            (same species and platforms, Jaccard >= 0.95, all root markers)
            or ``subset`` (a subset panel of a dataset missing genes).
        panel_mode: Resolved panel mode of the pair.
        panel_trust: Trust state (``refused`` … ``validated``).
        trust_reasons: Reason tokens of the trust state.
        banner: Whether the report shows a trust banner (refused, broad-only
            and provisional panels; never a validated family).
        validation_basis: ``real_data`` or ``simulation`` for validated
            families; ``None`` otherwise.
        validated_max_level: The family's headline validated level.
        validated_panels_sha256: Digest of ``validated_panels.csv``.
        validated_panel_levels_sha256: Digest of ``validated_panel_levels.csv``
            (per-(level, class) records of simulation-validated families).
        validated_share: Share of confident labels inside the validated
            region, per level.
        real_data_qc: Real-data QC outcomes and downgrades.
        n_declared_genes: Genes of the declared panel after control removal.
        gene_id_resolution: Genes resolved per ID source.
        n_unmapped: Features that resolved to no ID.
        controls_removed: Control features removed, per safe type token.
        n_missing_panel_genes: Declared genes absent from the dataset.
        panel_report_sha256: Digest of ``panel_report.json``.
    """

    panel_hash: str | None = None
    panel_family: str | None = None
    family_basis: Literal["own", "listed", "inherited", "subset"] | None = None
    panel_mode: PanelMode | None = None
    panel_trust: PanelTrust | None = None
    trust_reasons: list[str] = []
    banner: bool | None = None
    validation_basis: ValidationBasis | None = None
    validated_max_level: str | None = None
    validated_panels_sha256: str | None = None
    validated_panel_levels_sha256: str | None = None
    validated_share: dict[str, float] = {}
    real_data_qc: RealQcProvenance | None = None
    n_declared_genes: int | None = None
    gene_id_resolution: dict[str, int] = {}
    n_unmapped: int | None = None
    controls_removed: dict[str, int] = {}
    n_missing_panel_genes: int | None = None
    panel_report_sha256: str | None = None

    @model_validator(mode="after")
    def _check_basis(self: PanelProvenance) -> PanelProvenance:
        if self.validation_basis is not None and self.panel_trust != "validated":
            raise ValueError("validation_basis is set only for validated panels")
        if (
            self.banner is not None
            and self.panel_trust is not None
            and self.banner != (self.panel_trust != "validated")
        ):
            raise ValueError(
                "banner is shown exactly for refused, broad_only and provisional panels"
            )
        return self


class ResolvabilityProvenance(_ProvenanceModel):
    """Resolvability decisions applied to the dataset (plan §8.3).

    Attributes:
        recipe: Simulation recipe, e.g. ``"R1_contam_HO"``.
        recipe_version: Recipe version (part of ``build_hash``).
        sha256: Digest of ``resolvability.parquet``.
        emitted_depth_bins: Emitted depth bins per level and safe class token.
        d_max: Deepest grid depth with enough test cells, per class token.
        extrapolated_share: Share of the class's cells whose (class, depth) verdict
            comes from a pooled deep set (``resolvability_extrapolated``).
        reweighted_to_composition: Whether RESOLVE reweighted the tables to
            the dataset's soft composition.
        resolvability_inherited: Whether a subset bundle inherited its
            family's tables.
        resolvable_share: Share of table cells resolvable per level.
    """

    recipe: str | None = None
    recipe_version: int | None = None
    sha256: str | None = None
    emitted_depth_bins: dict[str, dict[str, list[int]]] = {}
    d_max: dict[str, int] = {}
    extrapolated_share: dict[str, float] = {}
    reweighted_to_composition: bool | None = None
    resolvability_inherited: bool = False
    resolvable_share: dict[str, float] = {}


class EngineProvenance(_ProvenanceModel):
    """Mapping engine settings (plan §3.3).

    Attributes:
        engine: Engine name.
        ctm_version: ``cell_type_mapper`` version.
        ctm_commit: ``cell_type_mapper`` commit, when known.
        bootstrap_factor: MapMyCells bootstrap factor.
        bootstrap_iteration: MapMyCells bootstrap iterations.
        rng_seed: Seed.
        n_processors: Worker processes.
        wall_time_s: Mapping wall time in seconds.
    """

    engine: str = "mapmycells"
    ctm_version: str | None = None
    ctm_commit: str | None = None
    bootstrap_factor: float | None = None
    bootstrap_iteration: int | None = None
    rng_seed: int | None = None
    n_processors: int | None = None
    wall_time_s: float | None = None


class ThresholdProvenance(_ProvenanceModel):
    """Thresholds and floors applied (plan §3.7, §5.4).

    Attributes:
        mode: ``"raw"`` (v1) or ``"simulation_calibrated"`` (v1.1).
        values: Threshold values keyed by safe names (e.g. ``whb_broad``).
        threshold_source: ``validated_default`` or ``resolvability_local``.
        floors_sha256: Digest of the floor table used.
        floor_source: ``real_e2``, ``unknown_panel``, ``simulation_validated``…
        calibration: ``"none"`` (v1) or ``"simulation_ho"`` (v1.1).
    """

    mode: Literal["raw", "simulation_calibrated"] = "raw"
    values: dict[str, float] = {}
    threshold_source: str | None = None
    floors_sha256: str | None = None
    floor_source: str | None = None
    calibration: Literal["none", "simulation_ho"] = "none"


class FlagProvenance(_ProvenanceModel):
    """Flag settings, gene sets and realised rates (plan §4.3, §5.6).

    Attributes:
        thresholds: Flag thresholds keyed by safe names.
        gene_sets: Gene set per flag (e.g. the derived spill-over genes).
        realised_rates: Rate per flag and safe ``<class>__<platform>`` token.
        informative: Whether each flag stratum is informative.
        null_reasons: Why a flag is null (e.g. ``insufficient_panel_genes``).
    """

    thresholds: dict[str, float] = {}
    gene_sets: dict[str, list[str]] = {}
    realised_rates: dict[str, dict[str, float]] = {}
    informative: dict[str, dict[str, bool]] = {}
    null_reasons: dict[str, str] = {}


class DatasetGateProvenance(_ProvenanceModel):
    """Human dataset gate verdict (plan §5.4).

    Attributes:
        frac_ge30: A, the share of table cells with at least 30 counts.
        table_broad_coverage: Confident broad coverage of table cells.
        segmented_broad_coverage: Confident broad coverage of all segmented
            objects.
        level: ``full``, ``broad_only`` or ``failed``.
        warning: Whether the warning flag is set.
        reasons: Reasons for the level and the warning.
    """

    frac_ge30: float | None = None
    table_broad_coverage: float | None = None
    segmented_broad_coverage: float | None = None
    level: GateLevel | None = None
    warning: bool = False
    reasons: list[str] = []


class ConsensusProvenance(_ProvenanceModel):
    """The degraded mode and consensus of one sample (plan §5.2, §5.3).

    Attributes:
        degraded_mode: ``consensus.DEGRADED_MODES`` name (``whb_sea`` in v1).
        methods: Methods available (``whb``, ``sea``; ``ll`` never in v1 /
            v1.1 production, OD-B8).
        max_tier: The largest tier the mode allows.
        single_method_override: WHB decided alone below 60 counts
            (``annotation_allow_single_method``).
        likelihood_vote: Whether the LL typer voted (always false, OD-B8).
        tier_counts: Table cells per ``ct_consensus_tier`` value.
        cop_suppressed: Cells whose COP call stayed at lineage.
    """

    degraded_mode: str
    methods: list[str] = []
    max_tier: int | None = None
    single_method_override: bool = False
    likelihood_vote: bool = False
    tier_counts: dict[str, int] = {}
    cop_suppressed: int | None = None


class MouseGateProvenance(_ProvenanceModel):
    """Mouse dataset gate and region handling (plan §7.2, §7.6).

    Attributes:
        signals: Gate signals keyed ``g1_density_ratio`` … ``g5_spillover``.
        level: ``full``, ``broad_only`` or ``failed``.
        warning: Whether the warning flag is set.
        reasons: Reasons for the level and the warning.
        section_regions: Regions used for pruning.
        region_source: ``auto`` (inferred), ``explicit`` or ``none``.
        n_assigned_tiles: Tiles with a majority region.
        drop_list_sha256: Digest of the drop list.
        drop_list_size: Nodes dropped.
        n_remapped: Cells re-mapped after pruning.
        rule_variant: Region-rule variant (M6b).
    """

    signals: dict[str, float | None] = {}
    level: GateLevel | None = None
    warning: bool = False
    reasons: list[str] = []
    section_regions: list[str] = []
    region_source: Literal["auto", "explicit", "none"] | None = None
    n_assigned_tiles: int | None = None
    drop_list_sha256: str | None = None
    drop_list_size: int | None = None
    n_remapped: int | None = None
    rule_variant: str | None = None


class AnnotationProvenance(_ProvenanceModel):
    """Provenance of one sample's annotation (``uns["merxen_annotation_json"]``).

    Attributes:
        schema_version: Version of this model.
        label_table_version: ``schema.LABEL_TABLE_VERSION`` of the labels.
        mode: Clustering mode that produced the labels.
        species: ``"human"`` or ``"mouse"``.
        anatomical_region: Human region token (``frontal_cortex``).
        merxen_version: Package version.
        panel: Panel, family and trust.
        references: Reference bundles keyed by ``reference_id``.
        engine: Mapping engine settings.
        resolvability: Resolvability decisions keyed by primary
            ``reference_id``.
        thresholds: Thresholds and floors.
        flags: Flag settings and realised rates.
        gate: Human dataset gate.
        mouse_gate: Mouse gate and region handling.
        consensus: Degraded mode and consensus tier counts.
        composition: The sample's composition over table cells, per safe
            ``<kind>`` token (``soft``, ``soft_ge30``, ``confident``,
            ``argmax``) and ``share_<class>`` / ``share7_<class>`` key (§5.5).
        confident_fraction_table: Confident share of table cells per level.
        confident_fraction_segmented: Confident share of segmented objects
            per level.
    """

    schema_version: int = PROVENANCE_SCHEMA_VERSION
    label_table_version: int = LABEL_TABLE_VERSION
    mode: Literal["legacy", "map_first"] = "map_first"
    species: Species
    anatomical_region: str | None = None
    merxen_version: str | None = None
    panel: PanelProvenance | None = None
    references: dict[str, ReferenceProvenance] = {}
    engine: EngineProvenance | None = None
    resolvability: dict[str, ResolvabilityProvenance] = {}
    thresholds: ThresholdProvenance | None = None
    flags: FlagProvenance | None = None
    gate: DatasetGateProvenance | None = None
    mouse_gate: MouseGateProvenance | None = None
    consensus: ConsensusProvenance | None = None
    composition: dict[str, dict[str, float]] = {}
    confident_fraction_table: dict[str, float] = {}
    confident_fraction_segmented: dict[str, float] = {}

    @model_validator(mode="after")
    def _check_references(self: AnnotationProvenance) -> AnnotationProvenance:
        for key, reference in self.references.items():
            if key != reference.reference_id:
                raise ValueError(
                    f"references key {key!r} differs from its reference_id "
                    f"{reference.reference_id!r}"
                )
        unknown = sorted(set(self.resolvability) - set(self.references))
        if self.references and unknown:
            raise ValueError(f"resolvability for unknown references {unknown}")
        if self.species == "human" and self.mouse_gate is not None:
            raise ValueError("mouse_gate is only valid for mouse provenance")
        return self

    def to_uns_json(self) -> str:
        """Serialise to the canonical JSON string stored in ``uns``.

        Returns:
            Compact JSON with sorted keys (identical for identical content).
        """
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_uns_json(
        cls: type[AnnotationProvenance], text: str
    ) -> AnnotationProvenance:
        """Parse the JSON string written by ``to_uns_json``.

        Args:
            text: JSON text.

        Returns:
            The validated provenance.
        """
        return cls.model_validate_json(text)

    def write_to_uns(self, uns: MutableMapping[str, Any]) -> None:
        """Store the provenance under ``PROVENANCE_UNS_KEY``.

        Args:
            uns: An AnnData ``uns`` mapping.
        """
        uns[PROVENANCE_UNS_KEY] = self.to_uns_json()

    @classmethod
    def read_from_uns(
        cls: type[AnnotationProvenance], uns: Mapping[str, Any]
    ) -> AnnotationProvenance | None:
        """Read the provenance stored by ``write_to_uns``.

        Args:
            uns: An AnnData ``uns`` mapping.

        Returns:
            The provenance, or ``None`` when the key is absent (legacy runs).

        Raises:
            TypeError: If the stored value is not a string.
        """
        if PROVENANCE_UNS_KEY not in uns:
            return None
        value = uns[PROVENANCE_UNS_KEY]
        if hasattr(value, "item") and not isinstance(value, str):
            value = value.item()
        if not isinstance(value, str):
            raise TypeError(
                f"uns[{PROVENANCE_UNS_KEY!r}] must be a JSON string, got "
                f"{type(value).__name__}"
            )
        return cls.from_uns_json(value)


def to_uns_json(provenance: AnnotationProvenance) -> str:
    """Serialise provenance to the canonical ``uns`` JSON string.

    Args:
        provenance: The provenance.

    Returns:
        ``provenance.to_uns_json()``.
    """
    return provenance.to_uns_json()


def annotation_manifest_filename(sample_id: str) -> str:
    """Return the annotation manifest file name of a sample.

    Args:
        sample_id: Sample id (``<sid>``).

    Returns:
        ``"<sid>_annotation_manifest.json"``.
    """
    return f"{sample_id}{ANNOTATION_MANIFEST_SUFFIX}"
