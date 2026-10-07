"""Panel diagnostics, validated panel families and trust states (plan §8.2, §4.7).

Three things live here:

* **diagnostics** (``panel_diagnostics``): what ``ANNOTATE_PANEL`` and
  ``ANNOTATE_REFERENCE_PREP`` recorded about a panel, gathered into one model
  for the trust state, the report's panel card and provenance: gene-ID
  resolution per declared panel (features in, controls removed by type,
  genes resolved by source, unmapped features, merged duplicates, species
  check) and, per reference bundle, the coverage of the panel (genes absent
  from the reference, root markers, root children the root separates,
  markers per parent, weak parents, auto-collapsed parents and the leaves
  they hide) and the resolvability trust constraint;
* **validated families** (``load_validated_panels``): the packaged tables
  ``validated_panels.csv`` (one row per validated panel hash, grouped into
  families, with ``validated_max_level`` and ``validation_basis``
  ``real_data`` | ``simulation``), ``validated_panel_levels.csv`` (per
  (level, class) records of ``simulation`` families: status,
  ``validated_min_depth``, ``tested_max_depth``; gate P, M13) and
  ``validated_panel_genes.csv`` (each row's resolved IDs and root markers,
  so a near-identical panel can inherit the family, OD-E7); a gate-P PR
  adds a ``simulation`` family to them with ``write_simulation_family``
  (M13);
* **the trust-state machine** (``trust_state``): ``refused`` /
  ``broad_only`` / ``provisional`` / ``validated`` per (reference, panel),
  with its effects (``TrustDecision``).

Trust is decided outside ``build_hash`` (plan §3.2): the bundle caches its
coverage and resolvability tables, and the state follows from them, the
panel's family and the validated tables at the time of use, so a gate-P PR
that promotes a family changes no bundle.

Rules (§8.2; thresholds from ``AnnotationPanelConfig``), first match wins:

1. ``refused``: a declared panel the annotation panel derives from was
   refused by the gene-ID resolver (< 95% resolved, species mismatch, other
   species' IDs, symbols as IDs), fewer than ``min_mapped_genes`` panel genes
   in the reference, fewer than ``min_root_markers`` root markers, or the
   broad level fails the local emission rule at every depth up to
   ``trust_max_depth`` (resolvability);
2. ``broad_only``: the leaf level is resolvable for fewer than half of the
   classes with enough test cells at every such depth (resolvability), or
   the bundle of a panel outside the validated families has no
   resolvability table (the fail-safe: nothing shows its leaves are
   resolvable);
3. ``validated``: the panel's family (listed hash, inherited by Jaccard or
   a subset panel of a listed family) is in ``validated_panels.csv``, with
   its ``validation_basis``;
4. ``provisional``: everything else.

Effects (``TrustDecision``): a refused reference is not mapped (primary: the
dataset gate is ``failed`` with ``panel_refused``; secondary: degraded mode
``single_method``); a broad-only panel leaves the leaf levels
``not_attempted_gate`` with ``subcluster_status = not_resolvable_panel``;
provisional panels get the provisional margins, a banner and the gate
warning flag (never a lower gate level). Real-data-validated families use
the validated rules up to ``validated_max_level``. Simulation-validated
families emit exactly as provisional panels (same regime, margins and
max-rule floors), so promotion never changes what is emitted: it only sets
``ct_<L>_validated`` where (level, class) is validated at the cell's depth,
removes the banner and replaces the provisional warning flag by the
``warn_unvalidated_share`` rule (rev3).

This module imports only the standard library, numpy, pandas and pydantic
(and ``merxen.annotation.panel``, which needs no more).
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from merxen.annotation.panel import (
    AnnotationPanel,
    DeclaredPanel,
    KnownPanelFamily,
    PanelFamily,
    compute_panel_hash,
)
from merxen.annotation.provenance import MarkerProvenance, PanelProvenance
from merxen.annotation.schema import (
    PLATFORMS,
    CellStatus,
    GateLevel,
    PanelMode,
    PanelTrust,
    ReferenceRole,
    ValidationBasis,
    meets_threshold,
    safe_token,
)
from merxen.annotation.vocab import (
    NEURONS,
    Species,
    asset_path,
    floor_class_for,
    primary_vocab,
)

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig

logger = logging.getLogger(__name__)

DIAGNOSTICS_SCHEMA_VERSION: Final = 1
VALIDATED_PANELS_FILE: Final = "validated_panels.csv"
VALIDATED_PANEL_LEVELS_FILE: Final = "validated_panel_levels.csv"
VALIDATED_PANEL_GENES_FILE: Final = "validated_panel_genes.csv"
VALIDATED_PANELS_COLUMNS: Final[tuple[str, ...]] = (
    "panel_id",
    "family_id",
    "panel_hash",
    "panel_role",
    "species",
    "platforms",
    "n_genes",
    "validated_max_level",
    "validation_basis",
    "root_marker_source",
    "evidence",
    "date",
    "approving_pr",
    "note",
)
VALIDATED_LEVELS_COLUMNS: Final[tuple[str, ...]] = (
    "family_id",
    "panel_hash",
    "level",
    "class",
    "in_class_set",
    "status",
    "validated_min_depth",
    "tested_max_depth",
    "evidence",
)
VALIDATED_GENES_COLUMNS: Final[tuple[str, ...]] = (
    "panel_id",
    "ensembl_id",
    "gene_symbol",
    "root_marker",
)
LEVEL_STATUS_PATTERN: Final = re.compile(r"^(validated|not_evaluable|failed:NP[1-9])$")
_TOKEN_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_HASH_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_DATE_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Depth of each level in the hierarchy, per species: ``validated_max_level``
# covers every level of equal or lower rank. SEA-AD's subclass (the second
# vote) ranks with the WHB leaf it votes on; the report-only fine levels
# (OD-E4) rank above every leaf, so they are never inside a real-data
# family's validated region.
LEVEL_RANKS: Final[dict[str, dict[str, int]]] = {
    "human": {
        "lineage": 0,
        "broad": 1,
        "nt": 2,
        "supercluster": 3,
        "seaad_subclass": 3,
        "cluster": 4,
    },
    "mouse": {"broad": 0, "class": 1, "nt": 2, "subclass": 3, "supertype": 4},
}
LEAF_RANK: Final = 3
# Gate P promotes a family only when it is validated at least at this level
# (plan §14, gate-P rule).
MIN_SIMULATION_LEVEL: Final[dict[str, str]] = {"human": "broad", "mouse": "class"}
# Families listed by these bases may inherit trust (``panel_family``).
FAMILY_BASES_WITH_TRUST: Final[frozenset[str]] = frozenset(
    {"listed", "inherited", "subset"}
)

Regime = Literal["validated", "provisional"]
FamilyBasis = Literal["own", "listed", "inherited", "subset"]
FloorSource = Literal["packaged", "simulation_validated", "unknown_panel"]
ThresholdSource = Literal["validated_default", "resolvability_local"]
DegradedMode = Literal["single_method"]


# Panel-card text of the Xenium Prime 5K families (user decision 4 of
# 2026-09-28 and the M3c trust rules, plan §8.10, D-G13). The notes state
# limits; they never change a prediction, an emission or a trust state. The
# ProSeg range was added on 2026-09-29 (orchestrator decision D6, pending the
# user's confirmation; 5k_real/phase1b/REPORT.txt §5.3: ProSeg 3.2.0 with the
# vendor XOA cells as its prior).
GLIAL_UPPER_BOUND_NOTE: Final = (
    "Simulated glial coverage is an upper bound: -.06 to -.18 on "
    "vendor-segmented 5K cells, -.04 to -.14 re-segmented with ProSeg"
)
PRECISION_UNMEASURED_NOTE: Final = "Precision is unmeasured on real data"
PRIME_PRECISION_DETAIL_NOTE: Final = (
    "The thinned-cell agreement (.996 class / .987 subclass at 250 counts) is a "
    "self-consistency upper bound, not a precision measurement"
)
PRIME_TRUST_NOTE: Final = (
    "Trust: provisional with the provisional margins; nothing promotes the "
    "family automatically and the public 5K section never enters a gate or a "
    "promotion. Gate P (M13) runs on the resolvability version-7 ensemble: "
    "thresholds and emission are frozen from the ensemble and NP3-NP7 must "
    "pass in every emission member"
)
PRIME_REAL_QC_NOTE: Final = (
    "Real datasets: a downgrade-only per-class warning when a class's real "
    "coverage is below its simulated class-depth prediction at the dataset's "
    "own per-class depth by more than 0.10 (it fires for glia and small "
    "hypothalamic classes alike, and on v1-type large-mask segmentation); no "
    "empirical offset is applied"
)
PRIME_IN_SAMPLE_NOTE: Final = (
    "The 5K numbers come from one public section (one hemisphere, vendor XOA "
    "3.0 segmentation) and are in-sample for depth, composition and factors: "
    "their residual optimism is a lower bound for a new section or segmentation"
)
PRIME_MOUSE_NEXT_NOTE: Final = (
    "Mouse 5K: before any gate-P PR the first in-house dataset (MerXen "
    "segmentation) measures per-class depth, per-gene factors and "
    "contamination; PREP is re-run with them as new simulation inputs, never "
    "as trust evidence"
)
PRIME_HUMAN_NOTES: Final[tuple[str, ...]] = (
    "Human 5K: glial coverage is expected below simulation by analogy with "
    "mouse; the lung 5K / v1 ratios are a cross-tissue stress member only",
    "Human 5K: no R3 member (no factor table against WHB exists) and no mouse "
    "factors, depth profile or depth prior; predictions are per depth scenario "
    "(per grid bin and the lung-FFPE scenario, median 245 counts)",
    "Human 5K: provisional until an in-house human 5K dataset measures, in "
    "order, per-class depth, gene complexity, factors against WHB and real vs "
    "simulated coverage; human 5K precision is unmeasured everywhere",
)


def panel_card_notes(species: str, chemistry: str) -> list[str]:
    """Return the panel-card notes of a family (user decision 4; plan §8.10).

    Xenium Prime 5K families (chemistry ``xenium_prime``) carry the glial
    upper bound and the unmeasured precision (verbatim, user decision 4),
    then the M3c trust rules: provisional, gate P on the version-7 ensemble,
    the downgrade-only per-class coverage warning without an offset, and the
    in-sample caveat; mouse adds the first in-house dataset's measurements,
    human the human-specific limits. Other chemistries have none.

    Args:
        species: ``human`` or ``mouse``.
        chemistry: The panel chemistry (``sim_inputs.resolve_chemistry``).

    Returns:
        The notes, in display order.
    """
    if chemistry != "xenium_prime":
        return []
    notes = [
        GLIAL_UPPER_BOUND_NOTE,
        PRECISION_UNMEASURED_NOTE,
        PRIME_PRECISION_DETAIL_NOTE,
        PRIME_TRUST_NOTE,
        PRIME_REAL_QC_NOTE,
        PRIME_IN_SAMPLE_NOTE,
    ]
    if species == "human":
        notes.extend(PRIME_HUMAN_NOTES)
    else:
        notes.append(PRIME_MOUSE_NEXT_NOTE)
    return notes


class ValidatedPanelsError(ValueError):
    """The validated-panel tables are inconsistent or malformed."""


class _DiagModel(BaseModel):
    """Base model: unknown fields rejected, frozen, NaN rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def level_rank(species: str, level: str) -> int | None:
    """Return a level's rank in the species hierarchy (``None`` if unknown).

    Args:
        species: ``"human"`` or ``"mouse"``.
        level: Label-table or resolvability level name.

    Returns:
        0 for the coarsest level, 3 for the leaf, 4 for report-only fine
        levels; ``None`` for a level the species does not have.
    """
    return LEVEL_RANKS.get(species, {}).get(level)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Diagnostics


class GeneIdDiagnostics(_DiagModel):
    """Gene-ID resolution of one declared panel (plan §8.2, §8.4).

    Attributes:
        name: Declared panel key (sample id or platform).
        platform: Platform, if known.
        status: Resolver verdict (``ok`` or ``refused``).
        refusal_reasons: Resolver refusal tokens (``species_mismatch``,
            ``gene_id_resolution``, ``native_id_prefix``,
            ``other_species_ids``, ...).
        n_features_in: Features in the declared list.
        controls_removed: Control features removed, per reason / type token.
        n_non_control: Non-control features.
        n_genes: Resolved distinct genes.
        resolution_share: Resolved share of the non-control features.
        resolved_by_source: Resolved genes per ID source.
        unmapped: Reason per non-control feature without an ID.
        n_merged_duplicates: IDs that several features resolved to.
        other_species_share: Share of native values with another species'
            prefix, when native IDs exist.
        species_check: Status of the exact-case species test, when run
            (``pass``, ``species_mismatch``, ``not_evaluable``).
    """

    name: str
    platform: str | None = None
    status: Literal["ok", "refused"] = "ok"
    refusal_reasons: tuple[str, ...] = ()
    n_features_in: int = Field(ge=0)
    controls_removed: dict[str, int] = Field(default_factory=dict)
    n_non_control: int = Field(ge=0)
    n_genes: int = Field(ge=0)
    resolution_share: float = Field(ge=0.0, le=1.0)
    resolved_by_source: dict[str, int] = Field(default_factory=dict)
    unmapped: dict[str, str] = Field(default_factory=dict)
    n_merged_duplicates: int = Field(default=0, ge=0)
    other_species_share: float | None = None
    species_check: str | None = None

    @property
    def n_controls_removed(self) -> int:
        """Return the number of control features removed."""
        return sum(self.controls_removed.values())

    @classmethod
    def from_declared(cls, panel: DeclaredPanel) -> GeneIdDiagnostics:
        """Summarise a ``DeclaredPanel`` (``ANNOTATE_PANEL`` in memory).

        Args:
            panel: The declared panel.

        Returns:
            Its diagnostics.
        """
        resolution = panel.resolution
        return cls(
            name=panel.sample_id or (panel.platform or "panel").lower(),
            platform=panel.platform,
            status=panel.status,
            refusal_reasons=tuple(panel.refusal_reasons),
            n_features_in=panel.n_features_in,
            controls_removed={
                reason: len(names)
                for reason, names in sorted(panel.controls_removed.items())
            },
            n_non_control=panel.n_non_control,
            n_genes=panel.n_genes,
            resolution_share=round(panel.resolution_share, 6),
            resolved_by_source=panel.resolution_counts(),
            unmapped=dict(sorted(panel.unresolved.items())),
            n_merged_duplicates=len(panel.merged_duplicates),
            other_species_share=(
                None if resolution is None else resolution.other_species_share
            ),
            species_check=(
                None if resolution is None else _species_verdict(resolution)
            ),
        )

    @classmethod
    def from_report_entry(
        cls, name: str, entry: Mapping[str, Any]
    ) -> GeneIdDiagnostics:
        """Read one ``panel_report.json`` ``declared_panels`` entry.

        Args:
            name: The entry's key.
            entry: The entry.

        Returns:
            Its diagnostics.
        """
        controls = entry.get("controls_removed") or {}
        check = entry.get("species_check")
        return cls(
            name=name,
            platform=entry.get("platform"),
            status=entry.get("status", "ok"),
            refusal_reasons=tuple(entry.get("refusal_reasons") or ()),
            n_features_in=int(entry.get("n_features_in", 0)),
            controls_removed={
                str(reason): len(names) if isinstance(names, list) else int(names)
                for reason, names in sorted(controls.items())
            },
            n_non_control=int(entry.get("n_non_control", 0)),
            n_genes=int(entry.get("n_genes", 0)),
            resolution_share=float(entry.get("gene_id_resolution_share", 0.0)),
            resolved_by_source={
                str(key): int(value)
                for key, value in sorted(
                    (entry.get("gene_id_resolution") or {}).items()
                )
            },
            unmapped={
                str(key): str(value)
                for key, value in sorted((entry.get("unresolved") or {}).items())
            },
            n_merged_duplicates=len(entry.get("merged_duplicates") or {}),
            other_species_share=entry.get("other_species_share"),
            species_check=(check.get("status") if isinstance(check, Mapping) else None),
        )


def _species_verdict(resolution: Any) -> str | None:
    check = getattr(resolution, "species_check", None)
    status = getattr(check, "status", None)
    return None if status is None else str(status)


class CollapsedParentRecord(_DiagModel):
    """A parent auto-collapsed for lack of markers (plan §3.2).

    Attributes:
        key: Lookup key (``<level>/<node>``).
        level: Taxonomy level.
        hidden_leaves: Leaves the collapse hides.
    """

    key: str
    level: str | None = None
    hidden_leaves: tuple[str, ...] = ()


class CoverageDiagnostics(_DiagModel):
    """Coverage of a panel by one reference bundle (plan §8.2).

    Attributes:
        reference_id: Store id.
        build_hash: Bundle ``build_hash``.
        n_panel_genes: Panel genes.
        n_query_genes_used: Panel genes present in the reference (the
            "mapped genes" of the refusal rule).
        absent_from_reference: Panel genes the reference lacks.
        n_marker_genes: Genes used as markers anywhere.
        root_markers: Markers of the taxonomy root.
        root_children: Children of the root.
        root_children_separated: Root children with at least
            ``root_child_min_markers`` markers (how many broad classes the
            root separates).
        root_child_min_markers: That minimum.
        markers_per_parent_min: Fewest markers of any parent.
        markers_per_parent_median: Median markers per parent.
        weak_parent_markers: Parents with fewer markers are weak.
        weak_parents: The weak parents.
        collapsed_parents: Auto-collapsed parents and the leaves they hide.
    """

    reference_id: str
    build_hash: str | None = None
    n_panel_genes: int = Field(ge=0)
    n_query_genes_used: int = Field(ge=0)
    absent_from_reference: tuple[str, ...] = ()
    n_marker_genes: int | None = None
    root_markers: int = Field(ge=0)
    root_children: int | None = None
    root_children_separated: int | None = None
    root_child_min_markers: int | None = None
    markers_per_parent_min: int | None = None
    markers_per_parent_median: float | None = None
    weak_parent_markers: int | None = None
    weak_parents: tuple[str, ...] = ()
    collapsed_parents: tuple[CollapsedParentRecord, ...] = ()

    @property
    def n_hidden_leaves(self) -> int:
        """Return the number of leaves hidden by collapsed parents."""
        return sum(len(parent.hidden_leaves) for parent in self.collapsed_parents)

    @classmethod
    def from_bundle_manifest(cls, manifest: Mapping[str, Any]) -> CoverageDiagnostics:
        """Read the coverage records of a ``bundle.json``.

        Args:
            manifest: The parsed ``bundle.json`` (``builder_output`` with
                ``panel_coverage`` and ``markers``, as the M2 builders write).

        Returns:
            The coverage diagnostics.

        Raises:
            ValueError: If the bundle has no panel coverage (a
                panel-independent reference).
        """
        output = manifest.get("builder_output") or {}
        coverage = output.get("panel_coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError(
                f"bundle {manifest.get('reference_id')!r} has no panel_coverage "
                "(a panel-independent reference)"
            )
        markers = output.get("markers") or {}
        collapsed = [
            CollapsedParentRecord(
                key=str(item.get("key")),
                level=item.get("level"),
                hidden_leaves=tuple(
                    str(leaf) for leaf in item.get("hidden_leaves", [])
                ),
            )
            for item in markers.get("collapsed", []) or []
            if isinstance(item, Mapping)
        ]
        if not collapsed:
            collapsed = [
                CollapsedParentRecord(key=str(key))
                for key in markers.get("collapsed_parents", []) or []
            ]
        median = coverage.get("markers_per_parent_median")
        return cls(
            reference_id=str(manifest.get("reference_id")),
            build_hash=manifest.get("build_hash"),
            n_panel_genes=int(coverage.get("n_panel_genes", 0)),
            n_query_genes_used=int(coverage.get("n_query_genes_used", 0)),
            absent_from_reference=tuple(coverage.get("absent_from_reference") or ()),
            n_marker_genes=coverage.get("n_marker_genes"),
            root_markers=int(coverage.get("root_markers", 0)),
            root_children=coverage.get("root_children"),
            root_children_separated=coverage.get("root_children_separated"),
            root_child_min_markers=coverage.get("root_child_min_markers"),
            markers_per_parent_min=coverage.get("markers_per_parent_min"),
            markers_per_parent_median=None if median is None else float(median),
            weak_parent_markers=markers.get("weak_parent_markers"),
            weak_parents=tuple(markers.get("weak_parents") or ()),
            collapsed_parents=tuple(collapsed),
        )

    def marker_provenance(self, **extra: Any) -> MarkerProvenance:
        """Return the ``MarkerProvenance`` fields these diagnostics fill.

        Args:
            **extra: Other ``MarkerProvenance`` fields (lookup sha256,
                ``n_per_utility``, prefilter).

        Returns:
            The marker provenance.
        """
        return MarkerProvenance(
            n_panel_genes=self.n_panel_genes,
            markers_per_parent_min=self.markers_per_parent_min,
            markers_per_parent_median=self.markers_per_parent_median,
            root_markers=self.root_markers,
            root_children_separated=self.root_children_separated,
            n_weak_parents=len(self.weak_parents),
            n_collapsed_parents=len(self.collapsed_parents),
            n_hidden_leaves=self.n_hidden_leaves,
            **extra,
        )


class ResolvabilityTrust(_DiagModel):
    """What a bundle's resolvability self-map implies for trust (§8.2, §8.3).

    Attributes:
        state: ``refused``, ``broad_only`` or ``None`` (no constraint).
        reasons: Why.
        broad_emitted_bins: Broad bins emitted up to ``trust_max_depth``.
        leaf_share_by_depth: Share of eligible classes whose leaf level is
            emitted, per depth.
        leaf_classes: Classes with enough test cells for the leaf test.
    """

    state: Literal["refused", "broad_only"] | None = None
    reasons: tuple[str, ...] = ()
    broad_emitted_bins: int | None = None
    leaf_share_by_depth: dict[str, float] = Field(default_factory=dict)
    leaf_classes: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ResolvabilityTrust:
        """Read ``TrustConstraint.to_json()`` (``bundle.json`` / summary).

        Args:
            record: The constraint record.

        Returns:
            The constraint.
        """
        return cls(
            state=record.get("state"),
            reasons=tuple(record.get("reasons") or ()),
            broad_emitted_bins=record.get("broad_emitted_bins"),
            leaf_share_by_depth={
                str(depth): float(share)
                for depth, share in (record.get("leaf_share_by_depth") or {}).items()
            },
            leaf_classes=tuple(record.get("leaf_classes") or ()),
        )

    @classmethod
    def from_bundle_manifest(
        cls, manifest: Mapping[str, Any]
    ) -> ResolvabilityTrust | None:
        """Read a bundle's resolvability constraint, if it ran.

        Args:
            manifest: The parsed ``bundle.json``.

        Returns:
            The constraint, or ``None`` when the bundle has no self-map.
        """
        output = manifest.get("builder_output") or {}
        resolvability = output.get("resolvability")
        if not isinstance(resolvability, Mapping):
            return None
        record = resolvability.get("trust")
        return cls.from_record(record) if isinstance(record, Mapping) else None


class PanelDiagnostics(_DiagModel):
    """Everything recorded about one annotation panel (report item 1).

    Attributes:
        schema_version: ``DIAGNOSTICS_SCHEMA_VERSION``.
        panel_name: Annotation panel name.
        panel_hash: Its hash.
        species: Species.
        platforms: Platforms it serves.
        n_genes: Its genes.
        family: Its family (``panel_genes.json``).
        gene_ids: Gene-ID resolution of each declared panel it derives from.
        coverage: Coverage per reference id.
        resolvability: Resolvability constraint per reference id (``None``
            where the bundle has no self-map).
    """

    schema_version: int = DIAGNOSTICS_SCHEMA_VERSION
    panel_name: str
    panel_hash: str
    species: Species
    platforms: tuple[str, ...] = ()
    n_genes: int = Field(ge=0)
    family: PanelFamily | None = None
    gene_ids: tuple[GeneIdDiagnostics, ...] = ()
    coverage: dict[str, CoverageDiagnostics] = Field(default_factory=dict)
    resolvability: dict[str, ResolvabilityTrust | None] = Field(default_factory=dict)

    @property
    def gene_ids_refused(self) -> bool:
        """Return whether a declared panel of this panel was refused."""
        return any(item.status == "refused" for item in self.gene_ids)

    def controls_removed(self) -> dict[str, int]:
        """Return control features removed per type, summed over platforms."""
        totals: dict[str, int] = {}
        for item in self.gene_ids:
            for reason, count in item.controls_removed.items():
                totals[reason] = totals.get(reason, 0) + count
        return dict(sorted(totals.items()))

    def resolved_by_source(self) -> dict[str, int]:
        """Return resolved genes per ID source, summed over platforms."""
        totals: dict[str, int] = {}
        for item in self.gene_ids:
            for source, count in item.resolved_by_source.items():
                totals[source] = totals.get(source, 0) + count
        return dict(sorted(totals.items()))

    def n_unmapped(self) -> int:
        """Return the unmapped non-control features, summed over platforms."""
        return sum(len(item.unmapped) for item in self.gene_ids)


def panel_diagnostics(
    panel: AnnotationPanel,
    *,
    declared: Sequence[DeclaredPanel] = (),
    panel_report: Mapping[str, Any] | None = None,
    bundles: Iterable[Mapping[str, Any]] = (),
) -> PanelDiagnostics:
    """Gather the diagnostics of one annotation panel (plan §8.2).

    Args:
        panel: The annotation panel (``panel_genes*.json``).
        declared: Its declared panels in memory (``ANNOTATE_PANEL``); when
            empty, ``panel_report`` supplies them.
        panel_report: ``panel_report.json`` content; its declared panels are
            used when ``declared`` is empty (only those the panel derives
            from: its samples, or its platforms for a gene list).
        bundles: Parsed ``bundle.json`` of each reference built on the panel.

    Returns:
        The diagnostics.
    """
    gene_ids: list[GeneIdDiagnostics]
    if declared:
        gene_ids = [GeneIdDiagnostics.from_declared(item) for item in declared]
    else:
        entries = (panel_report or {}).get("declared_panels") or {}
        gene_ids = [
            GeneIdDiagnostics.from_report_entry(str(name), entry)
            for name, entry in sorted(entries.items())
            if _derives_from(panel, str(name), entry)
        ]
    coverage: dict[str, CoverageDiagnostics] = {}
    resolvability: dict[str, ResolvabilityTrust | None] = {}
    for manifest in bundles:
        reference_id = str(manifest.get("reference_id"))
        bundle_panel = (manifest.get("panel") or {}).get("panel_hash")
        if bundle_panel is not None and bundle_panel != panel.panel_hash:
            raise ValueError(
                f"bundle {reference_id} {str(manifest.get('build_hash'))[:16]} was "
                f"built on panel {str(bundle_panel)[:16]}, not "
                f"{panel.panel_hash[:16]}"
            )
        coverage[reference_id] = CoverageDiagnostics.from_bundle_manifest(manifest)
        resolvability[reference_id] = ResolvabilityTrust.from_bundle_manifest(manifest)
    return PanelDiagnostics(
        panel_name=panel.name,
        panel_hash=panel.panel_hash,
        species=panel.species,
        platforms=tuple(panel.platforms),
        n_genes=panel.n_genes,
        family=panel.panel_family,
        gene_ids=tuple(gene_ids),
        coverage=coverage,
        resolvability=resolvability,
    )


def _derives_from(panel: AnnotationPanel, name: str, entry: Mapping[str, Any]) -> bool:
    """Whether a ``panel_report`` declared panel feeds an annotation panel."""
    if panel.sample_ids:
        return name in panel.sample_ids or entry.get("sample_id") in panel.sample_ids
    platform = str(entry.get("platform") or "").upper()
    return not panel.platforms or platform in panel.platforms


# --------------------------------------------------------------------------
# Validated families (validated_panels.csv and companions)


class ValidatedPanelRecord(_DiagModel):
    """One row of ``validated_panels.csv`` (plan §4.7).

    Attributes:
        panel_id: Row id (safe token), keying ``validated_panel_genes.csv``.
        family_id: Family id (as in ``floors_<species>.csv``).
        panel_hash: The validated panel's hash.
        panel_role: ``set_a``, ``set_c``, ``sample_panel``, ...
        species: Species.
        platforms: Platforms of the panel (sorted).
        n_genes: Its genes.
        validated_max_level: The family's headline validated level.
        validation_basis: ``real_data`` (seeded families) or ``simulation``
            (a gate-P PR, OD-E1).
        root_marker_source: Where the root markers in
            ``validated_panel_genes.csv`` come from.
        evidence: Evidence paths (relative to the evidence archive).
        date: Decision date (``YYYY-MM-DD``).
        approving_pr: The PR that approved the row.
        note: Free text.
    """

    panel_id: str
    family_id: str
    panel_hash: str
    panel_role: str
    species: Species
    platforms: tuple[str, ...]
    n_genes: int = Field(ge=1)
    validated_max_level: str
    validation_basis: ValidationBasis
    root_marker_source: str = ""
    evidence: str
    date: str
    approving_pr: str
    note: str = ""

    @field_validator("panel_id", "family_id", "panel_role")
    @classmethod
    def _check_token(cls, value: str) -> str:
        if not _TOKEN_PATTERN.fullmatch(value):
            raise ValueError(f"{value!r} is not a lower-case safe token")
        return value

    @field_validator("panel_hash")
    @classmethod
    def _check_hash(cls, value: str) -> str:
        if not _HASH_PATTERN.fullmatch(value):
            raise ValueError(f"{value!r} is not a sha256 hex digest")
        return value

    @field_validator("platforms")
    @classmethod
    def _check_platforms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(value) - set(PLATFORMS))
        if not value or unknown:
            raise ValueError(f"platforms must be a non-empty subset of {PLATFORMS}")
        return tuple(sorted(set(value)))

    @field_validator("date")
    @classmethod
    def _check_date(cls, value: str) -> str:
        if not _DATE_PATTERN.fullmatch(value):
            raise ValueError(f"date {value!r} is not YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def _check_level(self) -> ValidatedPanelRecord:
        rank = level_rank(self.species, self.validated_max_level)
        if rank is None or rank > LEAF_RANK:
            raise ValueError(
                f"{self.panel_id}: validated_max_level {self.validated_max_level!r} "
                f"is not a {self.species} level up to the leaf"
            )
        if self.validation_basis == "simulation":
            minimum = level_rank(self.species, MIN_SIMULATION_LEVEL[self.species])
            assert minimum is not None
            if rank < minimum:
                raise ValueError(
                    f"{self.panel_id}: a simulation-validated family needs "
                    f"validated_max_level >= {MIN_SIMULATION_LEVEL[self.species]}"
                )
        return self


class ValidatedLevelRecord(_DiagModel):
    """One row of ``validated_panel_levels.csv`` (simulation families; §4.7).

    Attributes:
        family_id: Family id.
        panel_hash: The gate-P-scored panel of the family.
        level: Level.
        class_name: Resolvability class (the ``class`` column).
        in_class_set: Whether the class is in C_P (§14).
        status: ``validated``, ``failed:NP<k>`` or ``not_evaluable``.
        validated_min_depth: Shallowest depth from which the (level, class)
            is validated (required when validated).
        tested_max_depth: D_P, the shallowest bin of the pooled deep set;
            deeper bins are inherited and marked extrapolated.
        evidence: Evidence path.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, populate_by_name=True
    )

    family_id: str
    panel_hash: str
    level: str
    class_name: str = Field(alias="class")
    in_class_set: bool
    status: str
    validated_min_depth: int | None = Field(default=None, ge=1)
    tested_max_depth: int | None = Field(default=None, ge=1)
    evidence: str = ""

    @field_validator("status")
    @classmethod
    def _check_status(cls, value: str) -> str:
        if not LEVEL_STATUS_PATTERN.fullmatch(value):
            raise ValueError(
                f"status {value!r} is not validated, not_evaluable or failed:NP<k>"
            )
        return value

    @model_validator(mode="after")
    def _check_depths(self) -> ValidatedLevelRecord:
        if self.status == "validated":
            if self.validated_min_depth is None or self.tested_max_depth is None:
                raise ValueError(
                    f"{self.family_id} {self.level} {self.class_name}: a validated "
                    "record needs validated_min_depth and tested_max_depth"
                )
            if self.tested_max_depth < self.validated_min_depth:
                raise ValueError(
                    f"{self.family_id} {self.level} {self.class_name}: "
                    "tested_max_depth < validated_min_depth"
                )
        return self

    @property
    def is_validated(self) -> bool:
        """Return whether this (level, class) is validated."""
        return self.status == "validated"


@dataclass(frozen=True)
class PanelGeneList:
    """The resolved IDs and root markers of one validated panel row.

    Attributes:
        panel_id: Row id.
        ensembl_ids: Sorted distinct IDs.
        symbols: Symbol per ID.
        root_markers: Root-marker IDs (a member must contain all of them).
    """

    panel_id: str
    ensembl_ids: tuple[str, ...]
    symbols: Mapping[str, str]
    root_markers: frozenset[str]


@dataclass(frozen=True)
class ValidatedPanelTable:
    """The validated families (``validated_panels.csv`` and companions).

    Attributes:
        records: One record per validated panel hash.
        levels: Per-(level, class) records of simulation families.
        genes: Gene list per ``panel_id`` (rows without one match by exact
            hash only).
        sha256: Digest per file name.
        source: ``"packaged"`` or the directory the tables came from.
    """

    records: tuple[ValidatedPanelRecord, ...]
    levels: tuple[ValidatedLevelRecord, ...] = ()
    genes: Mapping[str, PanelGeneList] = field(default_factory=dict)
    sha256: Mapping[str, str] = field(default_factory=dict)
    source: str = "packaged"

    def __post_init__(self) -> None:
        _check_table(self)

    def record_for_hash(self, panel_hash: str) -> ValidatedPanelRecord | None:
        """Return the record listing a panel hash, if any."""
        return next(
            (item for item in self.records if item.panel_hash == panel_hash), None
        )

    def family_records(self, family_id: str) -> tuple[ValidatedPanelRecord, ...]:
        """Return a family's records (empty when the family is not listed)."""
        return tuple(item for item in self.records if item.family_id == family_id)

    def family_ids(self) -> list[str]:
        """Return the listed family ids, sorted."""
        return sorted({item.family_id for item in self.records})

    def level_records(self, family_id: str) -> tuple[ValidatedLevelRecord, ...]:
        """Return a family's per-(level, class) records."""
        return tuple(item for item in self.levels if item.family_id == family_id)

    def known_families(self, species: str | None = None) -> list[KnownPanelFamily]:
        """Return the families ``panel.panel_family`` can inherit from.

        Args:
            species: Keep only this species' rows (``None``: all).

        Returns:
            One ``KnownPanelFamily`` per record; a record without a gene
            list matches by exact hash only.
        """
        families = []
        for record in self.records:
            if species is not None and record.species != species:
                continue
            genes = self.genes.get(record.panel_id)
            families.append(
                KnownPanelFamily(
                    family_id=record.family_id,
                    species=record.species,
                    platforms=frozenset(record.platforms),
                    panel_hash=record.panel_hash,
                    ensembl_ids=frozenset(() if genes is None else genes.ensembl_ids),
                    root_markers=frozenset() if genes is None else genes.root_markers,
                )
            )
        return families

    def describe(self) -> dict[str, Any]:
        """Return the tables' identity for reports and provenance."""
        return {
            "source": self.source,
            "sha256": dict(sorted(self.sha256.items())),
            "families": {
                family_id: {
                    "validation_basis": self.family_records(family_id)[
                        0
                    ].validation_basis,
                    "validated_max_level": self.family_records(family_id)[
                        0
                    ].validated_max_level,
                    "panel_ids": [
                        item.panel_id for item in self.family_records(family_id)
                    ],
                    "n_level_records": len(self.level_records(family_id)),
                }
                for family_id in self.family_ids()
            },
        }


def _check_table(table: ValidatedPanelTable) -> None:
    """Cross-row checks of the validated tables (raise ``ValidatedPanelsError``)."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    families: dict[str, ValidatedPanelRecord] = {}
    for record in table.records:
        if record.panel_id in seen_ids:
            problems.append(f"duplicated panel_id {record.panel_id}")
        if record.panel_hash in seen_hashes:
            problems.append(f"duplicated panel_hash {record.panel_hash[:16]}")
        seen_ids.add(record.panel_id)
        seen_hashes.add(record.panel_hash)
        first = families.setdefault(record.family_id, record)
        for name in ("species", "validation_basis", "validated_max_level"):
            if getattr(first, name) != getattr(record, name):
                problems.append(f"family {record.family_id}: rows disagree on {name}")
        genes = table.genes.get(record.panel_id)
        if genes is not None:
            if compute_panel_hash(list(genes.ensembl_ids)) != record.panel_hash:
                problems.append(
                    f"{record.panel_id}: the gene list does not hash to panel_hash"
                )
            if len(genes.ensembl_ids) != record.n_genes:
                problems.append(
                    f"{record.panel_id}: {len(genes.ensembl_ids)} listed genes, "
                    f"n_genes {record.n_genes}"
                )
            if not genes.root_markers <= set(genes.ensembl_ids):
                problems.append(f"{record.panel_id}: root markers outside the panel")
    unknown_genes = sorted(set(table.genes) - seen_ids)
    if unknown_genes:
        problems.append(f"gene lists for unknown panel_id {unknown_genes}")
    level_keys: set[tuple[str, str, str]] = set()
    for level in table.levels:
        family = families.get(level.family_id)
        if family is None:
            problems.append(f"level record for unknown family {level.family_id}")
            continue
        if family.validation_basis != "simulation":
            problems.append(
                f"level record for {level.family_id}, a {family.validation_basis} "
                "family (validated_panel_levels.csv holds simulation rows only)"
            )
        hashes = {item.panel_hash for item in table.family_records(level.family_id)}
        if level.panel_hash not in hashes:
            problems.append(
                f"level record {level.family_id} {level.level}: panel_hash "
                f"{level.panel_hash[:16]} is not a row of the family"
            )
        rank = level_rank(family.species, level.level)
        if rank is None or rank > LEAF_RANK:
            problems.append(
                f"level record {level.family_id}: {level.level!r} is not a "
                f"{family.species} level up to the leaf"
            )
        key = (level.family_id, level.level, level.class_name)
        if key in level_keys:
            problems.append(f"duplicated level record {key}")
        level_keys.add(key)
    for family_id, family in families.items():
        if family.validation_basis != "simulation":
            continue
        records = [item for item in table.levels if item.family_id == family_id]
        if not records:
            problems.append(
                f"simulation family {family_id} has no validated_panel_levels.csv rows"
            )
            continue
        max_rank = level_rank(family.species, family.validated_max_level)
        assert max_rank is not None
        for level_record in records:
            rank = level_rank(family.species, level_record.level)
            if (
                rank is not None
                and rank <= max_rank
                and level_record.in_class_set
                and not level_record.is_validated
            ):
                problems.append(
                    f"{family_id}: validated_max_level {family.validated_max_level} "
                    f"but C_P class {level_record.class_name} is "
                    f"{level_record.status} at {level_record.level} (gate-P rule, "
                    "§14)"
                )
    if problems:
        raise ValidatedPanelsError(
            f"validated panel tables ({table.source}): " + "; ".join(problems)
        )


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidatedPanelsError(f"{path.name}: missing columns {missing}")
    return frame


def _parse_bool(value: str, where: str) -> bool:
    text = value.strip().lower()
    if text not in {"true", "false"}:
        raise ValidatedPanelsError(f"{where}: {value!r} is not true / false")
    return text == "true"


def _optional_int(value: str) -> int | None:
    text = value.strip()
    return int(text) if text else None


def read_validated_panels(
    directory: Path | str, *, source: str | None = None
) -> ValidatedPanelTable:
    """Read the three validated-panel tables from a directory.

    ``validated_panel_levels.csv`` and ``validated_panel_genes.csv`` are
    optional (an absent file means no records).

    Args:
        directory: Directory holding ``validated_panels.csv``.
        source: Label recorded as the tables' source (default: the path).

    Returns:
        The validated table.

    Raises:
        ValidatedPanelsError: If a table is malformed or inconsistent.
        FileNotFoundError: If ``validated_panels.csv`` is missing.
    """
    base = Path(directory)
    panels_path = base / VALIDATED_PANELS_FILE
    if not panels_path.is_file():
        raise FileNotFoundError(f"{panels_path} not found")
    sha256 = {VALIDATED_PANELS_FILE: _file_sha256(panels_path)}
    records: list[ValidatedPanelRecord] = []
    for position, row in _read_csv(panels_path, VALIDATED_PANELS_COLUMNS).iterrows():
        try:
            records.append(
                ValidatedPanelRecord(
                    panel_id=row["panel_id"].strip(),
                    family_id=row["family_id"].strip(),
                    panel_hash=row["panel_hash"].strip(),
                    panel_role=row["panel_role"].strip(),
                    species=row["species"].strip(),
                    platforms=tuple(
                        item.strip().upper()
                        for item in row["platforms"].split(";")
                        if item.strip()
                    ),
                    n_genes=int(row["n_genes"]),
                    validated_max_level=row["validated_max_level"].strip(),
                    validation_basis=row["validation_basis"].strip(),
                    root_marker_source=row["root_marker_source"].strip(),
                    evidence=row["evidence"].strip(),
                    date=row["date"].strip(),
                    approving_pr=row["approving_pr"].strip(),
                    note=row["note"].strip(),
                )
            )
        except ValueError as error:
            raise ValidatedPanelsError(
                f"{VALIDATED_PANELS_FILE} row {int(str(position)) + 2}: {error}"
            ) from error
    levels: list[ValidatedLevelRecord] = []
    levels_path = base / VALIDATED_PANEL_LEVELS_FILE
    if levels_path.is_file():
        sha256[VALIDATED_PANEL_LEVELS_FILE] = _file_sha256(levels_path)
        frame = _read_csv(levels_path, VALIDATED_LEVELS_COLUMNS)
        for position, row in frame.iterrows():
            where = f"{VALIDATED_PANEL_LEVELS_FILE} row {int(str(position)) + 2}"
            try:
                levels.append(
                    ValidatedLevelRecord.model_validate(
                        {
                            "family_id": row["family_id"].strip(),
                            "panel_hash": row["panel_hash"].strip(),
                            "level": row["level"].strip(),
                            "class": row["class"].strip(),
                            "in_class_set": _parse_bool(row["in_class_set"], where),
                            "status": row["status"].strip(),
                            "validated_min_depth": _optional_int(
                                row["validated_min_depth"]
                            ),
                            "tested_max_depth": _optional_int(row["tested_max_depth"]),
                            "evidence": row["evidence"].strip(),
                        }
                    )
                )
            except ValueError as error:
                raise ValidatedPanelsError(f"{where}: {error}") from error
    genes: dict[str, PanelGeneList] = {}
    genes_path = base / VALIDATED_PANEL_GENES_FILE
    if genes_path.is_file():
        sha256[VALIDATED_PANEL_GENES_FILE] = _file_sha256(genes_path)
        frame = _read_csv(genes_path, VALIDATED_GENES_COLUMNS)
        for panel_id, group in frame.groupby("panel_id", sort=True):
            ids = [str(value).strip() for value in group["ensembl_id"]]
            if len(set(ids)) != len(ids):
                raise ValidatedPanelsError(
                    f"{VALIDATED_PANEL_GENES_FILE}: duplicated IDs in {panel_id}"
                )
            roots = {
                gene_id
                for gene_id, flag in zip(ids, group["root_marker"], strict=True)
                if _parse_bool(str(flag), f"{VALIDATED_PANEL_GENES_FILE} {panel_id}")
            }
            genes[str(panel_id)] = PanelGeneList(
                panel_id=str(panel_id),
                ensembl_ids=tuple(sorted(ids)),
                symbols=dict(
                    zip(ids, (str(s) for s in group["gene_symbol"]), strict=True)
                ),
                root_markers=frozenset(roots),
            )
    table = ValidatedPanelTable(
        records=tuple(records),
        levels=tuple(levels),
        genes=genes,
        sha256=sha256,
        source=source or str(base),
    )
    missing_genes = [item.panel_id for item in records if item.panel_id not in genes]
    if missing_genes:
        logger.info(
            "Validated panels without a gene list (exact-hash matching only): %s",
            ", ".join(missing_genes),
        )
    return table


@cache
def _packaged_validated_panels() -> ValidatedPanelTable:
    return read_validated_panels(
        asset_path(VALIDATED_PANELS_FILE).parent, source="packaged"
    )


def load_validated_panels(path: Path | str | None = None) -> ValidatedPanelTable:
    """Load the validated families (``AnnotationPanelConfig.validated_panels_path``).

    Args:
        path: ``validated_panels.csv`` or the directory holding it with its
            companions; ``None`` loads the packaged tables.

    Returns:
        The validated table.
    """
    if path is None:
        return _packaged_validated_panels()
    location = Path(path)
    directory = location.parent if location.suffix == ".csv" else location
    if location.suffix == ".csv" and location.name != VALIDATED_PANELS_FILE:
        raise ValidatedPanelsError(
            f"{location}: the validated table must be named {VALIDATED_PANELS_FILE} "
            "(its companions are read from the same directory)"
        )
    return read_validated_panels(directory)


# --------------------------------------------------------------------------
# Writing a gate-P family (M13; plan §14 gate-P rule, §4.7)


def level_class_keys(species: str, level: str) -> frozenset[str]:
    """Return the class keys a ``validated_panel_levels.csv`` row may name.

    The keys are those RESOLVE reads the per-(level, class) records with
    (``TrustDecision.validated_mask``): the consensus ``class_key`` of each
    level, derived from the primary vocab. Human: the E2 floor classes of
    the plausible WHB superclusters at lineage, broad and NT, with ``COP``
    apart at supercluster only (``vocab.floor_class_for``); mouse: the WMB
    class names. NT holds only the neuron classes, because NT does not apply
    to the others (M13 D14 (a): no row where NT does not apply). The SEA-AD
    subclass (the second vote; D14 (a): ``ct_seaad_subclass_validated`` is
    report-only) and the report-only fine levels have no keys: gate P scores
    the primary reference up to the leaf only.

    Args:
        species: ``"human"`` or ``"mouse"``.
        level: A level name.

    Returns:
        The class keys of the level (empty for a level gate P never
        records).
    """
    if species not in ("human", "mouse"):
        return frozenset()
    vocab = primary_vocab("human" if species == "human" else "mouse")
    names = vocab.names
    if species == "human":
        if level == "nt":
            chosen = [name for name in names if vocab.broad_class(name) == NEURONS]
        elif level in ("lineage", "broad", "supercluster"):
            chosen = list(names)
        else:
            return frozenset()
        keys = {
            floor_class_for(
                name,
                species="human",
                level="supercluster" if level == "supercluster" else "broad",
            )
            for name in chosen
        }
        return frozenset(key for key in keys if key is not None)
    if level == "nt":
        return frozenset(name for name in names if vocab.broad_class(name) == NEURONS)
    if level in ("broad", "class", "subclass"):
        return frozenset(names)
    return frozenset()


def own_family_id(species: str, platforms: Iterable[str], panel_hash: str) -> str:
    """Return a panel's own, hash-derived family id (``panel.panel_family``).

    Args:
        species: Species.
        platforms: The panel's platforms.
        panel_hash: The panel hash.

    Returns:
        ``<species>_<platforms>_<hash prefix>``, as ``panel_family`` names a
        panel that is its own family (M13 D16: mechanical).
    """
    token = "_".join(sorted({str(item).lower() for item in platforms})) or "any"
    return f"{species}_{token}_{panel_hash[:12]}"


def _csv_text(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Rows rendered as CSV lines (no header), ``\\n`` line ends."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(f"a row of {len(row)} fields for {len(columns)} columns")
        writer.writerow(row)
    return buffer.getvalue()


def _appended_text(path: Path, columns: Sequence[str], body: str) -> str:
    """A table's text with ``body`` appended; existing lines kept as they are.

    Raises:
        ValidatedPanelsError: If the existing header is not ``columns``.
    """
    header = ",".join(columns) + "\n"
    if not path.is_file():
        return header + body
    text = path.read_text(encoding="utf-8")
    first = text.splitlines()[0] if text else ""
    if first != ",".join(columns):
        raise ValidatedPanelsError(
            f"{path.name}: the header {first!r} is not {','.join(columns)!r}"
        )
    if not text.endswith("\n"):
        text += "\n"
    return text + body


def _replace_tables(base: Path, staged: Path, names: Sequence[str]) -> None:
    """Move staged tables over ``base``'s, ``validated_panels.csv`` last.

    The companions go first, so the family is listed only once its level and
    gene rows are in place. The current files are copied into the staging
    directory first; if a replacement fails, the files already replaced are
    restored (a file that did not exist is removed) and the error is raised,
    so the tables are left as they were.

    Args:
        base: The tables' directory.
        staged: The staging directory (on the same filesystem) holding the
            new tables under the same names.
        names: The tables to replace.
    """
    order = sorted(names, key=lambda name: name == VALIDATED_PANELS_FILE)
    kept: dict[str, Path | None] = {}
    for name in order:
        current = base / name
        if current.is_file():
            copy = staged / f"{name}.previous"
            copy.write_bytes(current.read_bytes())
            kept[name] = copy
        else:
            kept[name] = None
    replaced: list[str] = []
    try:
        for name in order:
            os.replace(staged / name, base / name)
            replaced.append(name)
    except BaseException:
        for name in reversed(replaced):
            backup = kept[name]
            try:
                if backup is None:
                    (base / name).unlink(missing_ok=True)
                else:
                    os.replace(backup, base / name)
            except OSError:
                logger.exception(
                    "Validated tables %s: could not restore %s after a failed "
                    "write; restore it from version control",
                    base,
                    name,
                )
        raise


def write_simulation_family(
    directory: Path | str,
    record: ValidatedPanelRecord,
    levels: Sequence[ValidatedLevelRecord],
    genes: PanelGeneList,
    *,
    self_map: ResolvabilityTrust | None,
) -> ValidatedPanelTable:
    """Add a gate-P family to the validated tables (plan §14, §4.7; M13).

    Appends the family's ``validated_panels.csv`` row (``validation_basis =
    simulation``), its ``validated_panel_levels.csv`` rows and its
    ``validated_panel_genes.csv`` rows to the tables in ``directory`` (the
    packaged tables in a gate-P PR, which the user approves; §14 gate-P
    rule). The existing lines are kept as they are. The combined tables are
    written to a temporary directory and read back with
    ``read_validated_panels`` (all its cross-row checks: the rank rule of
    ``validated_max_level``, the gene list hashing to ``panel_hash``)
    before they replace the files, so nothing is written when any check
    fails. ``validated_panels.csv`` is replaced last, and a replacement that
    fails restores the files already replaced.

    Refused:

    - a record whose basis is not ``simulation``;
    - a family whose bundle has no resolvability self-map (``self_map``
      ``None``; M13 D13 (a): such a family stays ``broad_only``, so its
      evidence cannot promote it);
    - a ``family_id`` other than the panel's own hash-derived id (D16), or
      one already in the tables;
    - level rows of another family or panel hash, none at all, or naming a
      class that is not a consensus class key of its level
      (``level_class_keys``: no NT row where NT does not apply, no SEA-AD
      subclass row; D14 (a));
    - a gene list of another ``panel_id``.

    Args:
        directory: The directory of ``validated_panels.csv`` (created with
            the three tables when it holds none).
        record: The family's panel row.
        levels: Its per-(level, class) rows.
        genes: Its resolved IDs, symbols and root markers.
        self_map: The family bundle's resolvability constraint
            (``ResolvabilityTrust.from_bundle_manifest``; ``None`` when the
            bundle has no self-map).

    Returns:
        The tables as read back from ``directory``.

    Raises:
        ValidatedPanelsError: For any refusal above, or when the combined
            tables fail ``read_validated_panels``.
    """
    if record.validation_basis != "simulation":
        raise ValidatedPanelsError(
            f"{record.panel_id}: write_simulation_family writes simulation rows "
            f"only, not {record.validation_basis}"
        )
    if self_map is None:
        raise ValidatedPanelsError(
            f"{record.family_id}: the family bundle has no resolvability self-map; "
            "a simulation family without one stays broad_only (M13 D13 (a)), so "
            "its gate-P evidence cannot promote it"
        )
    expected = own_family_id(record.species, record.platforms, record.panel_hash)
    if record.family_id != expected:
        raise ValidatedPanelsError(
            f"family_id {record.family_id!r} is not the panel's hash-derived id "
            f"{expected!r} (M13 D16)"
        )
    if not levels:
        raise ValidatedPanelsError(
            f"{record.family_id}: a simulation family needs its "
            f"{VALIDATED_PANEL_LEVELS_FILE} rows"
        )
    problems: list[str] = []
    for level in levels:
        if level.family_id != record.family_id or level.panel_hash != (
            record.panel_hash
        ):
            problems.append(
                f"the level row {level.level}/{level.class_name} is of "
                f"{level.family_id} {level.panel_hash[:16]}, not the record's"
            )
        if level.class_name not in level_class_keys(record.species, level.level):
            problems.append(
                f"{level.level}/{level.class_name} is not a consensus class key of "
                f"the {record.species} level {level.level!r} (D14 (a))"
            )
    if genes.panel_id != record.panel_id:
        problems.append(
            f"the gene list is of {genes.panel_id!r}, not {record.panel_id!r}"
        )
    if problems:
        raise ValidatedPanelsError(
            f"{record.family_id}: " + "; ".join(sorted(set(problems)))
        )
    base = Path(directory)
    panels_path = base / VALIDATED_PANELS_FILE
    if panels_path.is_file():
        existing = read_validated_panels(base)
        if record.family_id in existing.family_ids():
            raise ValidatedPanelsError(
                f"family {record.family_id} is already in {panels_path}"
            )
    panel_rows = [
        [
            record.panel_id,
            record.family_id,
            record.panel_hash,
            record.panel_role,
            record.species,
            ";".join(record.platforms),
            str(record.n_genes),
            record.validated_max_level,
            record.validation_basis,
            record.root_marker_source,
            record.evidence,
            record.date,
            record.approving_pr,
            record.note,
        ]
    ]
    level_rows = [
        [
            level.family_id,
            level.panel_hash,
            level.level,
            level.class_name,
            "true" if level.in_class_set else "false",
            level.status,
            "" if level.validated_min_depth is None else str(level.validated_min_depth),
            "" if level.tested_max_depth is None else str(level.tested_max_depth),
            level.evidence,
        ]
        for level in levels
    ]
    gene_rows = [
        [
            genes.panel_id,
            gene_id,
            str(genes.symbols.get(gene_id, gene_id)),
            "true" if gene_id in genes.root_markers else "false",
        ]
        for gene_id in sorted(genes.ensembl_ids)
    ]
    texts = {
        name: _appended_text(base / name, columns, _csv_text(columns, rows))
        for name, columns, rows in (
            (VALIDATED_PANELS_FILE, VALIDATED_PANELS_COLUMNS, panel_rows),
            (VALIDATED_PANEL_LEVELS_FILE, VALIDATED_LEVELS_COLUMNS, level_rows),
            (VALIDATED_PANEL_GENES_FILE, VALIDATED_GENES_COLUMNS, gene_rows),
        )
    }
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix=".validated_") as staging:
        staged = Path(staging)
        for name, text in texts.items():
            (staged / name).write_text(text, encoding="utf-8")
        table = read_validated_panels(staged, source=str(base))
        if record.family_id not in table.family_ids():
            raise ValidatedPanelsError(
                f"{record.family_id}: the written tables do not list the family"
            )
        _replace_tables(base, staged, list(texts))
    logger.info(
        "Validated tables %s: added the simulation family %s (%s, %d level rows)",
        base,
        record.family_id,
        record.validated_max_level,
        len(levels),
    )
    return read_validated_panels(base)


# --------------------------------------------------------------------------
# Trust states


class TrustRules(_DiagModel):
    """The §8.2 thresholds the trust state applies (``AnnotationPanelConfig``).

    Attributes:
        min_mapped_genes: Fewer panel genes in the reference refuse it.
        min_root_markers: Fewer root markers refuse it.
        resolvability_enabled: Whether PREP runs the self-map; a bundle
            without one then means it was not run.
    """

    min_mapped_genes: int = Field(default=50, ge=1)
    min_root_markers: int = Field(default=10, ge=1)
    resolvability_enabled: bool = True

    @classmethod
    def from_config(cls, config: AnnotationConfig) -> TrustRules:
        """Return the rules of an annotation config.

        Args:
            config: The annotation config.

        Returns:
            The rules.
        """
        return cls(
            min_mapped_genes=config.panel.min_mapped_genes,
            min_root_markers=config.panel.min_root_markers,
            resolvability_enabled=config.resolvability.enabled,
        )


class TrustReason(_DiagModel):
    """One reason behind a trust state (or a note that did not decide it).

    Attributes:
        code: Stable token (``gene_ids:<resolver reason>``,
            ``min_mapped_genes``, ``root_markers``,
            ``resolvability:broad_unresolvable``,
            ``resolvability:leaf_unresolvable``, ``resolvability_not_run``,
            ``family_not_validated``, ``family_validated``, ...).
        detail: One sentence.
    """

    code: str
    detail: str


class TrustDecision(_DiagModel):
    """The trust state of one (reference, panel) and its effects (§8.2).

    Attributes:
        reference_id: Store id.
        role: Reference role.
        species: Species.
        panel_hash: Panel hash.
        state: ``refused``, ``broad_only``, ``provisional`` or ``validated``.
        reasons: Why the state was reached (refusals, broad-only causes, the
            family verdict).
        notes: Checks that did not decide the state (e.g. a validated family
            whose bundle has no self-map, so H18 cannot be checked).
        complete: Whether coverage and resolvability were both available.
        family_id: The panel's family.
        family_basis: ``own``, ``listed``, ``inherited`` or ``subset``.
        validation_basis: For listed families: ``real_data`` or
            ``simulation``.
        validated_max_level: The family's headline validated level.
        validated_panel_ids: The family's rows in ``validated_panels.csv``.
        level_records: The family's per-(level, class) records.
        tables_sha256: Digests of the validated tables consulted.
    """

    reference_id: str
    role: ReferenceRole
    species: Species
    panel_hash: str
    state: PanelTrust
    reasons: tuple[TrustReason, ...] = ()
    notes: tuple[TrustReason, ...] = ()
    complete: bool = True
    family_id: str | None = None
    family_basis: FamilyBasis | None = None
    validation_basis: ValidationBasis | None = None
    validated_max_level: str | None = None
    validated_panel_ids: tuple[str, ...] = ()
    level_records: tuple[ValidatedLevelRecord, ...] = ()
    tables_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_basis(self) -> TrustDecision:
        if (self.validation_basis is None) != (self.state != "validated"):
            raise ValueError("validation_basis is set exactly for validated panels")
        if self.level_records and self.validation_basis != "simulation":
            raise ValueError("level records belong to simulation families only")
        return self

    # --- codes -------------------------------------------------------------

    @property
    def reason_codes(self) -> list[str]:
        """Return the reason tokens."""
        return [reason.code for reason in self.reasons]

    # --- mapping and gate --------------------------------------------------

    @property
    def map_skipped(self) -> bool:
        """Return whether MAP skips this reference (refused)."""
        return self.state == "refused"

    @property
    def gate_level_cap(self) -> GateLevel | None:
        """Return the gate level the trust state forces, if any.

        A refused primary reference fails the dataset gate (``panel_refused``;
        every table cell ``not_attempted_gate`` and ``exclude_hard``); a
        broad-only primary panel caps it at ``broad_only``. Other roles and
        states never change the gate level (a provisional panel only warns).
        """
        if self.role != "primary":
            return None
        if self.state == "refused":
            return "failed"
        if self.state == "broad_only":
            return "broad_only"
        return None

    @property
    def gate_reason(self) -> str | None:
        """Return the gate reason token of ``gate_level_cap``."""
        cap = self.gate_level_cap
        if cap == "failed":
            return "panel_refused"
        if cap == "broad_only":
            return "panel_broad_only"
        return None

    @property
    def degraded_mode(self) -> DegradedMode | None:
        """Return the degraded mode a refused secondary reference causes (§5.3)."""
        if self.role == "secondary" and self.state == "refused":
            return "single_method"
        return None

    @property
    def exclude_hard(self) -> bool:
        """Return whether every cell is ``exclude_hard`` (refused primary)."""
        return self.role == "primary" and self.state == "refused"

    @property
    def subcluster_status(self) -> str | None:
        """Return the forced ``subcluster_status`` (``not_resolvable_panel``)."""
        return "not_resolvable_panel" if self.state == "broad_only" else None

    def level_status_override(self, level: str) -> CellStatus | None:
        """Return the status the trust state forces on a level, if any.

        Args:
            level: Level name.

        Returns:
            ``not_attempted_gate`` for every level of a refused reference and
            for the leaf and finer levels of a broad-only panel; ``None``
            otherwise.
        """
        if self.state == "refused":
            return CellStatus.NOT_ATTEMPTED_GATE
        if self.state == "broad_only":
            rank = level_rank(self.species, level)
            if rank is None or rank >= LEAF_RANK:
                return CellStatus.NOT_ATTEMPTED_GATE
        return None

    # --- report --------------------------------------------------------------

    @property
    def banner(self) -> bool:
        """Return whether the report shows a trust banner (§9 item 1).

        Refused, broad-only and provisional panels get one; validated
        families, whatever their basis, do not (rev3: promotion removes it).
        """
        return self.state != "validated"

    @property
    def gate_warning(self) -> bool:
        """Return whether the trust state alone sets the gate warning flag.

        Provisional panels always warn. Simulation-validated families warn
        only through ``unvalidated_share_warning`` (their confident labels
        outside the validated region); real-data families never do.
        """
        return self.state == "provisional"

    # --- emission ------------------------------------------------------------

    def _real_data_covers(self, level: str) -> bool:
        if self.state != "validated" or self.validation_basis != "real_data":
            return False
        assert self.validated_max_level is not None
        rank = level_rank(self.species, level)
        max_rank = level_rank(self.species, self.validated_max_level)
        return rank is not None and max_rank is not None and rank <= max_rank

    def emission_regime(self, level: str) -> Regime:
        """Return the resolvability regime of a level (``level_emission``).

        Args:
            level: Level name.

        Returns:
            ``validated`` (base targets, pre-registered thresholds) for a
            real-data-validated family up to its ``validated_max_level``;
            ``provisional`` (targets + margins, local thresholds) for every
            other level and panel, simulation-validated families included,
            so promotion by simulation never changes what is emitted.
        """
        return "validated" if self._real_data_covers(level) else "provisional"

    def threshold_source(self, level: str) -> ThresholdSource:
        """Return ``threshold_source`` of a level (§8.3)."""
        if self.emission_regime(level) == "validated":
            return "validated_default"
        return "resolvability_local"

    def floor_source(self, level: str) -> FloorSource:
        """Return how a level's floors are chosen (§5.4).

        Args:
            level: Level name.

        Returns:
            ``packaged`` (the real-derived floors of the family,
            ``floors_<species>.csv``), ``simulation_validated`` (max of the
            known and simulated floors, no warning) or ``unknown_panel``
            (the same max rule, with a warning).
        """
        if self._real_data_covers(level):
            return "packaged"
        if self.state == "validated" and self.validation_basis == "simulation":
            return "simulation_validated"
        return "unknown_panel"

    def floor_warning(self, level: str) -> bool:
        """Return whether the floor choice warns (``unknown_panel``)."""
        return self.floor_source(level) == "unknown_panel"

    def applies_provisional_margins(self, level: str) -> bool:
        """Return whether the provisional target margins apply to a level."""
        return self.emission_regime(level) == "provisional"

    # --- validated region ----------------------------------------------------

    def level_record(self, level: str, cls: str) -> ValidatedLevelRecord | None:
        """Return the family's record for (level, class), if any."""
        return next(
            (
                item
                for item in self.level_records
                if item.level == level and item.class_name == cls
            ),
            None,
        )

    def is_validated(self, level: str, cls: str | None, depth: float | None) -> bool:
        """Return whether a confident label lies in the validated region.

        Args:
            level: Level name.
            cls: The cell's resolvability class at the level.
            depth: The cell's counts (or depth bin).

        Returns:
            Real-data families: the level is up to ``validated_max_level``.
            Simulation families: (level, class) is validated and ``depth >=
            validated_min_depth``. Otherwise false.
        """
        if self._real_data_covers(level):
            return True
        if self.state != "validated" or self.validation_basis != "simulation":
            return False
        if cls is None or depth is None:
            return False
        record = self.level_record(level, cls)
        if record is None or not record.is_validated:
            return False
        assert record.validated_min_depth is not None
        return bool(depth >= record.validated_min_depth)

    def validated_mask(
        self,
        level: str,
        classes: Sequence[str | None] | np.ndarray | pd.Series,
        depths: Sequence[float] | np.ndarray | pd.Series,
        confident: Sequence[bool] | np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Return ``ct_<L>_validated`` for many cells (§4.1).

        Args:
            level: Level name.
            classes: Each cell's resolvability class at the level.
            depths: Each cell's counts.
            confident: Whether each cell is ``confident`` at the level.

        Returns:
            Boolean array: confident cells inside the validated region.
        """
        confident_array = np.asarray(confident, dtype=bool)
        if self._real_data_covers(level):
            return confident_array.copy()
        result = np.zeros(confident_array.shape, dtype=bool)
        if self.state != "validated" or self.validation_basis != "simulation":
            return result
        class_array = np.asarray(classes, dtype=object)
        depth_array = np.asarray(depths, dtype=np.float64)
        for record in self.level_records:
            if record.level != level or not record.is_validated:
                continue
            assert record.validated_min_depth is not None
            in_class = class_array == record.class_name
            deep = meets_threshold(depth_array, float(record.validated_min_depth))
            result |= in_class & deep
        return np.asarray(result & confident_array, dtype=bool)

    def is_extrapolated(self, level: str, cls: str | None, depth: float | None) -> bool:
        """Return whether a validated label lies deeper than ``tested_max_depth``.

        Such labels are validated by inheritance from the pooled deep set
        (§14 evaluation rules) and reported as extrapolated.
        """
        if not self.is_validated(level, cls, depth) or cls is None or depth is None:
            return False
        record = self.level_record(level, cls)
        return bool(
            record is not None
            and record.tested_max_depth is not None
            and depth > record.tested_max_depth
        )

    def unvalidated_share_warning(
        self,
        validated_share: Mapping[str, float | None],
        *,
        max_share: float = 0.10,
    ) -> tuple[bool, list[str]]:
        """Return the simulation-family gate warning (``warn_unvalidated_share``).

        Args:
            validated_share: Share of confident labels inside the validated
                region, per emitted level (``None``: no confident label).
            max_share: ``AnnotationGate.warn_unvalidated_share``.

        Returns:
            Whether to warn, and one reason per level whose confident labels
            fall outside the validated region more than ``max_share``.
            Always ``(False, [])`` unless the family is simulation-validated.
        """
        if self.state != "validated" or self.validation_basis != "simulation":
            return False, []
        reasons = [
            f"unvalidated_share:{level}"
            for level, share in sorted(validated_share.items())
            if share is not None and 1.0 - share > max_share + 1e-12
        ]
        return bool(reasons), reasons

    # --- serialisation -------------------------------------------------------

    def effects(self) -> dict[str, Any]:
        """Return the level-independent effects as JSON (report, manifests)."""
        return {
            "map_skipped": self.map_skipped,
            "gate_level_cap": self.gate_level_cap,
            "gate_reason": self.gate_reason,
            "degraded_mode": self.degraded_mode,
            "exclude_hard": self.exclude_hard,
            "subcluster_status": self.subcluster_status,
            "banner": self.banner,
            "gate_warning": self.gate_warning,
        }

    def to_json(self) -> dict[str, Any]:
        """Return the decision and its effects as JSON."""
        record = self.model_dump(mode="json", by_alias=True)
        record["effects"] = self.effects()
        return record


def _family_verdict(
    family: PanelFamily | None,
    species: str,
    validated: ValidatedPanelTable,
) -> tuple[tuple[ValidatedPanelRecord, ...], TrustReason]:
    """The listed records a panel's family earns trust from, and why."""
    if family is None:
        return (), TrustReason(
            code="family_not_validated", detail="the panel has no family record"
        )
    records = validated.family_records(family.family_id)
    if family.basis not in FAMILY_BASES_WITH_TRUST or not records:
        return (), TrustReason(
            code="family_not_validated",
            detail=(
                f"family {family.family_id} ({family.basis}) is not in "
                f"{VALIDATED_PANELS_FILE}"
            ),
        )
    if records[0].species != species:
        return (), TrustReason(
            code="family_not_validated",
            detail=f"family {family.family_id} is a {records[0].species} family",
        )
    return records, TrustReason(
        code="family_validated",
        detail=(
            f"family {family.family_id} ({family.basis}, Jaccard "
            f"{family.jaccard:.3f}) is validated on "
            f"{records[0].validation_basis} up to {records[0].validated_max_level}"
        ),
    )


def trust_state(
    *,
    reference_id: str,
    role: ReferenceRole,
    species: Species,
    panel_hash: str,
    n_panel_genes: int,
    family: PanelFamily | None,
    validated: ValidatedPanelTable,
    rules: TrustRules | None = None,
    gene_ids: Sequence[GeneIdDiagnostics] = (),
    coverage: CoverageDiagnostics | None = None,
    resolvability: ResolvabilityTrust | None = None,
) -> TrustDecision:
    """Return the trust state of one (reference, panel) (plan §8.2).

    Args:
        reference_id: Store id.
        role: Reference role.
        species: Run species.
        panel_hash: The annotation panel's hash.
        n_panel_genes: Its genes.
        family: Its family (``AnnotationPanel.panel_family``).
        validated: The validated families.
        rules: The thresholds (default: the §3.7 defaults).
        gene_ids: Gene-ID diagnostics of its declared panels.
        coverage: The bundle's coverage (``None`` before PREP: the decision
            is then a preview, ``complete=False``, from the gene IDs and the
            family alone).
        resolvability: The bundle's resolvability constraint (``None`` when
            no self-map ran).

    Returns:
        The decision.
    """
    rules = rules or TrustRules()
    reasons: list[TrustReason] = []
    notes: list[TrustReason] = []
    for item in gene_ids:
        if item.status != "refused":
            continue
        for token in item.refusal_reasons or ("gene_id_resolution",):
            reasons.append(
                TrustReason(
                    code=f"gene_ids:{token}",
                    detail=(
                        f"declared {item.name} panel refused by the gene-ID "
                        f"resolver ({token}; {item.resolution_share:.3f} resolved)"
                    ),
                )
            )
    mapped = n_panel_genes if coverage is None else coverage.n_query_genes_used
    if mapped < rules.min_mapped_genes:
        reasons.append(
            TrustReason(
                code="min_mapped_genes",
                detail=(
                    f"{mapped} panel genes in the reference < min_mapped_genes "
                    f"{rules.min_mapped_genes}"
                ),
            )
        )
    if coverage is None:
        notes.append(
            TrustReason(
                code="coverage_unavailable",
                detail=(
                    "no bundle yet (before PREP): the root-marker and "
                    "resolvability rules were not applied"
                ),
            )
        )
    elif coverage.root_markers < rules.min_root_markers:
        reasons.append(
            TrustReason(
                code="root_markers",
                detail=(
                    f"{coverage.root_markers} root markers < min_root_markers "
                    f"{rules.min_root_markers}"
                ),
            )
        )
    records, family_reason = _family_verdict(family, species, validated)
    applies_resolvability = role in ("primary", "secondary")
    if resolvability is not None and resolvability.state == "refused":
        reasons.append(
            TrustReason(
                code="resolvability:broad_unresolvable",
                detail="; ".join(resolvability.reasons)
                or "the broad level fails the local emission rule at every depth",
            )
        )
    complete = coverage is not None and (
        resolvability is not None or not applies_resolvability
    )
    state: PanelTrust
    if reasons:
        state = "refused"
        if records:
            notes.append(family_reason)
    elif resolvability is not None and resolvability.state == "broad_only":
        state = "broad_only"
        reasons.append(
            TrustReason(
                code="resolvability:leaf_unresolvable",
                detail="; ".join(resolvability.reasons)
                or "the leaf level is resolvable for too few classes",
            )
        )
        if records:
            notes.append(family_reason)
    elif (
        resolvability is None
        and applies_resolvability
        and coverage is not None
        and not records
    ):
        state = "broad_only"
        why = (
            "resolvability is disabled"
            if not rules.resolvability_enabled
            else "the bundle has no resolvability self-map"
        )
        reasons.append(
            TrustReason(
                code="resolvability_not_run",
                detail=(
                    f"{why}: nothing shows the leaf level is resolvable on a "
                    "panel outside the validated families (fail-safe)"
                ),
            )
        )
    elif records:
        state = "validated"
        reasons.append(family_reason)
        if resolvability is None and applies_resolvability and coverage is not None:
            notes.append(
                TrustReason(
                    code="resolvability_not_run",
                    detail=(
                        "no resolvability self-map: the H18 regression check "
                        "cannot run for this validated family"
                    ),
                )
            )
    else:
        state = "provisional"
        reasons.append(family_reason)
    basis = records[0].validation_basis if state == "validated" else None
    decision = TrustDecision(
        reference_id=reference_id,
        role=role,
        species=species,
        panel_hash=panel_hash,
        state=state,
        reasons=tuple(reasons),
        notes=tuple(notes),
        complete=complete,
        family_id=None if family is None else family.family_id,
        family_basis=None if family is None else family.basis,
        validation_basis=basis,
        validated_max_level=records[0].validated_max_level if records else None,
        validated_panel_ids=tuple(item.panel_id for item in records),
        level_records=(
            validated.level_records(records[0].family_id)
            if basis == "simulation"
            else ()
        ),
        tables_sha256=dict(validated.sha256),
    )
    logger.info(
        "Trust %s / %s: %s (%s)",
        reference_id,
        panel_hash[:12],
        state,
        ", ".join(decision.reason_codes) or "no reason",
    )
    return decision


def trust_for_panel(
    diagnostics: PanelDiagnostics,
    *,
    reference_id: str,
    role: ReferenceRole,
    validated: ValidatedPanelTable,
    rules: TrustRules | None = None,
) -> TrustDecision:
    """Return ``trust_state`` from a panel's gathered diagnostics.

    Args:
        diagnostics: ``panel_diagnostics`` output.
        reference_id: The reference to decide for.
        role: Its role.
        validated: The validated families.
        rules: The thresholds.

    Returns:
        The decision.
    """
    return trust_state(
        reference_id=reference_id,
        role=role,
        species=diagnostics.species,
        panel_hash=diagnostics.panel_hash,
        n_panel_genes=diagnostics.n_genes,
        family=diagnostics.family,
        validated=validated,
        rules=rules,
        gene_ids=diagnostics.gene_ids,
        coverage=diagnostics.coverage.get(reference_id),
        resolvability=diagnostics.resolvability.get(reference_id),
    )


def family_validation_preview(
    family: PanelFamily | None,
    species: str,
    validated: ValidatedPanelTable,
) -> dict[str, Any]:
    """Return the family's validation as ``ANNOTATE_PANEL`` can see it.

    Before PREP there is no coverage or resolvability, so this is only the
    family part of the trust state: ``validated`` families stay validated
    unless PREP's refusal or broad-only checks fire.

    Args:
        family: The panel's family.
        species: Species.
        validated: The validated families.

    Returns:
        ``family_id``, ``family_basis``, ``listed``, ``validation_basis``,
        ``validated_max_level``, ``expected_trust`` (``validated`` or
        ``provisional``, before PREP's checks) and the reason.
    """
    records, reason = _family_verdict(family, species, validated)
    return {
        "family_id": None if family is None else family.family_id,
        "family_basis": None if family is None else family.basis,
        "listed": bool(records),
        "validation_basis": records[0].validation_basis if records else None,
        "validated_max_level": records[0].validated_max_level if records else None,
        "validated_panel_ids": [item.panel_id for item in records],
        "expected_trust": "validated" if records else "provisional",
        "reason": reason.detail,
    }


# --------------------------------------------------------------------------
# Provenance


def validated_share_by_class(
    decision: TrustDecision,
    level: str,
    classes: Sequence[str | None] | np.ndarray | pd.Series,
    depths: Sequence[float] | np.ndarray | pd.Series,
    confident: Sequence[bool] | np.ndarray | pd.Series,
) -> tuple[float | None, dict[str, float]]:
    """Return the validated share of confident labels, overall and per class.

    Args:
        decision: The trust decision.
        level: Level name.
        classes: Each cell's class at the level.
        depths: Each cell's counts.
        confident: Whether each cell is confident at the level.

    Returns:
        The share over all confident cells (``None`` without any) and per
        class with confident cells, keyed by ``safe_token(class)``.
    """
    confident_array = np.asarray(confident, dtype=bool)
    mask = decision.validated_mask(level, classes, depths, confident_array)
    n_confident = int(confident_array.sum())
    overall = None if n_confident == 0 else float(mask.sum()) / n_confident
    class_array = np.asarray(classes, dtype=object)
    per_class: dict[str, float] = {}
    for cls in sorted({str(item) for item in class_array[confident_array]}):
        in_class = confident_array & (class_array.astype(str) == cls)
        per_class[safe_token(cls)] = round(
            float(mask[in_class].sum()) / int(in_class.sum()), 6
        )
    return (None if overall is None else round(overall, 6)), per_class


def panel_provenance(
    decision: TrustDecision,
    diagnostics: PanelDiagnostics,
    *,
    panel_mode: PanelMode | None = None,
    validated_share: Mapping[str, float] | None = None,
    n_missing_panel_genes: int | None = None,
    panel_report_sha256: str | None = None,
) -> PanelProvenance:
    """Return ``AnnotationProvenance.panel`` for a dataset (plan §4.6).

    Args:
        decision: The primary reference's trust decision.
        diagnostics: The panel's diagnostics.
        panel_mode: The pair's resolved panel mode.
        validated_share: Validated share of confident labels per level.
        n_missing_panel_genes: Declared genes absent from the dataset.
        panel_report_sha256: Digest of ``panel_report.json``.

    Returns:
        The panel provenance.
    """
    gene_counts = diagnostics.resolved_by_source()
    return PanelProvenance(
        panel_hash=diagnostics.panel_hash,
        panel_family=decision.family_id,
        family_basis=decision.family_basis,
        panel_mode=panel_mode,
        panel_trust=decision.state,
        trust_reasons=decision.reason_codes,
        banner=decision.banner,
        validation_basis=decision.validation_basis,
        validated_max_level=(
            decision.validated_max_level if decision.state == "validated" else None
        ),
        validated_panels_sha256=decision.tables_sha256.get(VALIDATED_PANELS_FILE),
        validated_panel_levels_sha256=decision.tables_sha256.get(
            VALIDATED_PANEL_LEVELS_FILE
        ),
        validated_share=dict(validated_share or {}),
        n_declared_genes=diagnostics.n_genes,
        gene_id_resolution={safe_token(k): v for k, v in gene_counts.items()},
        n_unmapped=diagnostics.n_unmapped(),
        controls_removed={
            safe_token(k): v for k, v in diagnostics.controls_removed().items()
        },
        n_missing_panel_genes=n_missing_panel_genes,
        panel_report_sha256=panel_report_sha256,
    )
