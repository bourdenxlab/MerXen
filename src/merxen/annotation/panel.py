"""Declared panels, annotation panels and required bundles (plan §3.2, §8.1, §8.5).

``ANNOTATE_PANEL`` (``merxen annotation-panel``) runs this module once per
pair x segmentation and writes:

* ``panel_report.json``: gene-ID resolution by source, unresolved features,
  controls removed and why, merged duplicates, the panel mode, set c, the
  panel families;
* one ``panel_genes*.json`` per annotation panel (``AnnotationPanel``):
  sorted Ensembl IDs, symbols, platforms, ``panel_hash``, panel mode;
* ``required_bundles.json``: the (reference, panel) bundles this pair x
  segmentation needs from ``ANNOTATE_REFERENCE_PREP``.

A panel is the **declared** gene list of a platform (the Xenium
``gene_panel.json``, the MERSCOPE codebook, a source-table ``var`` or, by
default, the unfiltered ``var`` of the prepared H5AD) after control removal
and gene-ID resolution, never the observed or ``min_cells``-filtered genes;
``panel_hash`` is the sha256 of its sorted IDs (plan §8.1). Controls come from
the shared registry ``merxen.control_features``; IDs come from the gene-ID
resolver ``merxen.annotation.gene_ids`` (native ID, the pair's other platform,
the run species' local gene table, aliases, curated overrides), which also runs
the exact-case species test. A declared panel whose resolution is refused
(species mismatch, < 95% resolved, symbols in the ID column, another species'
IDs) refuses every annotation panel built from it: it gets no bundle. A
published clustered H5AD declares the features its control filter kept
(``uns["merxen_clustering_squidpy"]["control_feature_filter"]``), not the
``min_cells``-filtered ``var``, so its panel hash is the prepared panel's.

Panel modes (plan §3.2, §8.5): two platform panels with Jaccard >= 0.9 form an
``intersection`` panel (human: set a), and same-panel human pairs also get set
c. For the seeded set-a family (the 296-gene evidence panel and its post-M0e
297-gene form) set c is set a minus the curated E5 list of 32 platform-deviant
genes (``setc_exclusions_human.csv``), the set E5 validated; it is one
family-level panel shared by every pair. Other families fall back to the
label-free rule (a gene is dropped when its pseudobulk platform ratio deviates
by more than 2 log2 units from the pair median, over table cells inside the
shared tissue mask), flagged ``label_free_rule`` in the report; the rule is
also computed as a cross-check for the curated family whenever prepared data
exist. Other pairs are ``per_platform``: each platform's own panel plus their
intersection for cross-platform statistics. Unpaired samples use their own
panel (``single_sample``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from merxen.annotation.config import (
    AnnotationConfig,
    AnnotationPanelConfig,
    AnnotationReferenceSpec,
)
from merxen.annotation.gene_ids import (
    NATIVE_ID_COLUMNS,
    SPECIES_ID_PATTERNS,
    SYMBOL_COLUMNS,
    FeatureInput,
    GeneIdResolution,
    GeneIdSource,
    GeneIdSources,
    GeneTable,
    ResolutionRules,
    clean_native_value,
    clean_text,
    gene_id_sources,
    load_overrides,
    resolve_gene_ids,
    strip_version,
)
from merxen.annotation.schema import PanelMode, ReferenceRole
from merxen.annotation.vocab import Species
from merxen.control_features import (
    XENIUM_GENE_CODEWORD_CATEGORIES,
    has_control_token,
    matches_control_name_pattern,
)
from merxen.gene_ids import is_ensembl_gene_id

if TYPE_CHECKING:
    from merxen.analysis.mapmycells import GeneIdFallbackTable
    from merxen.annotation.diagnostics import ValidatedPanelTable

logger = logging.getLogger(__name__)

# 2: RequiredBundle.uses; set-c report basis and curated family.
# 3: gene-ID resolver fields (status, species check, resolution table sha256).
# 4: families from validated_panels.csv (basis "listed"), the report's
#    validated_panels record and per-panel family_validation.
PANEL_SCHEMA_VERSION: Final = 4
SETC_FAMILY_FILES: Final[dict[str, str]] = {"human": "setc_families_human.csv"}
SETC_EXCLUSION_FILES: Final[dict[str, str]] = {"human": "setc_exclusions_human.csv"}
PANEL_GENES_FILE: Final = "panel_genes.json"
PANEL_GENES_SETC_FILE: Final = "panel_genes_setc.json"
PANEL_GENES_INTERSECTION_FILE: Final = "panel_genes_intersection.json"
PANEL_REPORT_FILE: Final = "panel_report.json"
REQUIRED_BUNDLES_FILE: Final = "required_bundles.json"
PREPARED_MANIFEST_FILE: Final = "manifest.json"

FEATURE_TYPE_COLUMNS: Final[tuple[str, ...]] = ("feature_types", "feature_type")
# Xenium ``transcripts.parquet`` columns that also carry the feature type
# (XOA >= 3.0), used where a table has no feature-type column (plan §8.4).
CODEWORD_CATEGORY_COLUMN: Final = "codeword_category"
IS_GENE_COLUMN: Final = "is_gene"
GENE_FEATURE_TYPE: Final = "Gene Expression"
NOT_GENE_FEATURE_TYPE: Final = "not is_gene"
# Xenium ``gene_panel.json`` target descriptors and the feature types they
# stand for (the cell-feature-matrix names).
XENIUM_PANEL_DESCRIPTOR_TYPES: Final[dict[str, str]] = {
    "gene": GENE_FEATURE_TYPE,
    "negative_control": "Negative Control Probe",
}
PLATFORMS: Final[tuple[str, ...]] = ("MERSCOPE", "XENIUM")
# Roles that need a bundle on every annotation panel; the primary reference
# also gets the set-c and intersection panels. Region shares are
# panel-independent and resolvability references are built inside PREP.
PANEL_ROLES: Final[frozenset[str]] = frozenset(
    {"primary", "secondary", "sensitivity", "likelihood"}
)
PANEL_INDEPENDENT_ROLES: Final[frozenset[str]] = frozenset({"region_share"})
DEFAULT_MIN_COUNTS: Final = 10

# Kept for the M2 name: the resolver's sources (plan §8.4).
ResolutionSource = GeneIdSource
PanelKind = Literal[
    "intersection", "setc", "platform", "single_sample", "gene_list", "subset"
]
PanelSourceKind = Literal[
    "xenium_gene_panel_json",
    "merscope_codebook",
    "gene_table",
    "h5ad_var",
    "prepared_h5ad_var",
    "clustered_h5ad_declared",
]
BundlePurpose = Literal[
    "annotation", "setc_sensitivity", "intersection_xpanel", "panel_independent"
]


def compute_panel_hash(gene_ids: Sequence[str]) -> str:
    """Return ``panel_hash``: sha256 of the sorted, distinct resolved IDs.

    The IDs are joined with ``"\\n"`` (no trailing newline) and hashed as
    UTF-8, so the hash depends on the gene set only: not on feature order,
    symbols, duplicates, counts or cells (plan §8.1).

    Args:
        gene_ids: Unversioned Ensembl gene IDs.

    Returns:
        64 lower-case hex characters.
    """
    text = "\n".join(sorted(set(gene_ids)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_clean_text = clean_text


# --------------------------------------------------------------------------
# Models


class _PanelModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PanelSource(_PanelModel):
    """Where a declared panel was read from.

    Attributes:
        kind: The source type.
        path: The file read.
        sha256: sha256 of that file (``None`` for an in-memory table).
    """

    kind: PanelSourceKind
    path: str | None = None
    sha256: str | None = None


class PanelFamily(_PanelModel):
    """The family a panel belongs to (plan §8.1, OD-E7).

    Attributes:
        family_id: Family id; the panel's own id unless it has a listed one.
        basis: ``"own"``, ``"listed"`` (its hash is a row of
            ``validated_panels.csv`` for the same species and platforms),
            ``"inherited"`` (same species and platforms, Jaccard >=
            ``family_min_jaccard`` with a listed panel, all its root markers
            present) or ``"subset"`` (a subset panel of a dataset missing
            panel genes, which keeps its parent's family; ``subset_panel``).
        reference_panel_hash: Hash of the family's panel.
        jaccard: Jaccard of this panel with the family's panel (for a
            subset, with its parent panel).
        matched_platforms: The panel's platforms the family was validated
            on (``listed`` / ``inherited``; empty for its own family).
    """

    family_id: str
    basis: Literal["own", "listed", "inherited", "subset"]
    reference_panel_hash: str
    jaccard: float
    matched_platforms: list[str] = Field(default_factory=list)


class DeclaredResolution(_PanelModel):
    """Identity of one declared panel's gene-ID resolution (plan §8.4).

    Recorded in the annotation panel and in ``bundle.json``
    (``built_from_panel``) as provenance: bundles stay keyed by the
    resolved IDs (``panel_hash``), so two resolutions that give the same
    IDs share one bundle and the table checksum is not part of
    ``build_hash`` (a deviation from plan §8.4, M3b review).

    Attributes:
        resolution_table_sha256: ``GeneIdResolution.table_sha256`` (``None``
            for panels resolved before M3b).
        gene_tables: The local gene table the resolver consulted, per
            species (``None``: not available).
    """

    resolution_table_sha256: str | None = None
    gene_tables: dict[str, str | None] = Field(default_factory=dict)


class DeclaredPanel(_PanelModel):
    """One platform's declared panel after control removal and ID resolution.

    Attributes:
        sample_id: Sample the panel belongs to (``None`` for a gene list).
        platform: ``"MERSCOPE"``, ``"XENIUM"`` or ``None``.
        species: Run species.
        source: Where the declared list came from.
        panel_hash: ``compute_panel_hash(ensembl_ids)``.
        ensembl_ids: Sorted distinct resolved IDs.
        symbols: One symbol per ID (the first feature resolving to it).
        id_sources: Resolution source per ID (``native``, ``pair_lookup``,
            ``fallback_table``, ``symbol_fallback``, ``alias``,
            ``override``; the first feature resolving to it).
        symbol_to_id: Resolved ID per non-control feature symbol.
        feature_ids: Resolved ID per non-control feature name.
        n_features_in: Features in the declared list.
        controls_removed: Removed control features per reason token.
        kept_despite_control_token: Features kept although their name
            contains a control token, because they carry a native Ensembl ID
            or resolve to a reference gene.
        unresolved: Reason per non-control feature without an ID (by symbol).
        merged_duplicates: Features merged into one ID (ID -> symbols).
        other_species_ids: Native IDs with another species' prefix.
        resolution: The gene-ID resolver's full result (``None`` only for
            panels built before M3b).
    """

    sample_id: str | None
    platform: str | None
    species: Species
    source: PanelSource
    panel_hash: str
    ensembl_ids: list[str]
    symbols: list[str]
    id_sources: dict[str, ResolutionSource]
    symbol_to_id: dict[str, str]
    feature_ids: dict[str, str] = Field(default_factory=dict)
    n_features_in: int
    controls_removed: dict[str, list[str]]
    kept_despite_control_token: list[str] = Field(default_factory=list)
    unresolved: dict[str, str]
    merged_duplicates: dict[str, list[str]] = Field(default_factory=dict)
    other_species_ids: list[str] = Field(default_factory=list)
    resolution: GeneIdResolution | None = None

    @property
    def status(self) -> Literal["ok", "refused"]:
        """Return the gene-ID resolution verdict (``ok`` or ``refused``)."""
        return "ok" if self.resolution is None else self.resolution.status

    @property
    def refusal_reasons(self) -> list[str]:
        """Return why the resolution refused the panel (empty when ``ok``)."""
        return [] if self.resolution is None else list(self.resolution.refusal_reasons)

    def refusal_text(self) -> str:
        """Return one sentence per refusal reason, joined."""
        if self.resolution is None:
            return ""
        details = self.resolution.refusal_details
        return "; ".join(
            f"{reason}: {details.get(reason, reason)}"
            for reason in self.resolution.refusal_reasons
        )

    def control_reason_by_name(self) -> dict[str, str]:
        """Return the control reason per removed feature name."""
        return {
            name: reason
            for reason, names in self.controls_removed.items()
            for name in names
        }

    @property
    def n_genes(self) -> int:
        """Return the number of resolved genes."""
        return len(self.ensembl_ids)

    @property
    def n_non_control(self) -> int:
        """Return the number of non-control features."""
        return self.n_features_in - sum(
            len(names) for names in self.controls_removed.values()
        )

    @property
    def resolution_share(self) -> float:
        """Return the share of non-control features that resolved to an ID."""
        n_non_control = self.n_non_control
        if n_non_control == 0:
            return 0.0
        return (n_non_control - len(self.unresolved)) / n_non_control

    def resolution_counts(self) -> dict[str, int]:
        """Return the number of resolved genes per ID source."""
        counts: dict[str, int] = {}
        for source in self.id_sources.values():
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def symbol_for(self, gene_id: str) -> str:
        """Return the declared symbol of a resolved ID (``""`` if absent)."""
        mapping = dict(zip(self.ensembl_ids, self.symbols, strict=True))
        return mapping.get(gene_id, "")

    @model_validator(mode="after")
    def _check_ids(self: DeclaredPanel) -> DeclaredPanel:
        _check_sorted_ids(self.ensembl_ids, self.symbols, self.panel_hash)
        return self


def _check_sorted_ids(ids: list[str], symbols: list[str], panel_hash: str) -> None:
    if ids != sorted(set(ids)):
        raise ValueError("ensembl_ids must be sorted and distinct")
    if len(symbols) != len(ids):
        raise ValueError("symbols must hold one symbol per ID")
    if panel_hash != compute_panel_hash(ids):
        raise ValueError("panel_hash does not match ensembl_ids")


class AnnotationPanel(_PanelModel):
    """An annotation panel (``panel_genes*.json``; plan §3.2).

    Attributes:
        schema_version: ``PANEL_SCHEMA_VERSION``.
        name: ``"intersection"``, ``"setc"``, ``"merscope"``, ``"xenium"``,
            ``"sample"``, ``"gene_list"`` or ``<parent>_subset``.
        kind: Panel kind.
        species: Species.
        platforms: Platforms whose data the panel serves (sorted).
        sample_ids: Samples the panel was derived from.
        panel_mode: Resolved panel mode of the pair.
        panel_hash: sha256 of the sorted IDs (``compute_panel_hash``).
        n_genes: Number of genes.
        ensembl_ids: Sorted distinct Ensembl gene IDs.
        symbols: One symbol per ID (Xenium's where both platforms declare it).
        symbols_by_platform: Each platform's symbol per ID (``""`` if absent).
        declared_panel_hashes: Declared-panel hash per platform or sample.
        parent_panel_hash: For set c, the hash of set a; for a subset panel,
            the hash of the panel it was cut from.
        excluded_ids: For set c, the set-a IDs it drops; for a subset panel,
            the parent's genes the dataset lacks.
        panel_family: The panel's family.
        declared_resolutions: Gene-ID resolution identity of each declared
            panel behind it, keyed like ``declared_panel_hashes``
            (provenance, not part of ``panel_hash`` or ``build_hash``).
    """

    schema_version: int = PANEL_SCHEMA_VERSION
    name: str
    kind: PanelKind
    species: Species
    platforms: list[str]
    sample_ids: list[str]
    panel_mode: PanelMode
    panel_hash: str
    n_genes: int
    ensembl_ids: list[str]
    symbols: list[str]
    symbols_by_platform: dict[str, list[str]] = Field(default_factory=dict)
    declared_panel_hashes: dict[str, str] = Field(default_factory=dict)
    parent_panel_hash: str | None = None
    excluded_ids: list[str] = Field(default_factory=list)
    panel_family: PanelFamily | None = None
    declared_resolutions: dict[str, DeclaredResolution] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self: AnnotationPanel) -> AnnotationPanel:
        _check_sorted_ids(self.ensembl_ids, self.symbols, self.panel_hash)
        if self.n_genes != len(self.ensembl_ids):
            raise ValueError("n_genes must equal the number of ensembl_ids")
        for platform, symbols in self.symbols_by_platform.items():
            if len(symbols) != len(self.ensembl_ids):
                raise ValueError(f"symbols_by_platform[{platform!r}] has wrong length")
        return self

    def symbols_sha256(self) -> str:
        """Return the sha256 of the panel's ID-to-symbols table.

        Recorded in ``bundle.json`` (``built_from_panel``) and not part of
        ``build_hash``: bundles are keyed by the resolved IDs, whose
        resolution the panel hash already covers, and carry no symbols, so
        panels with the same IDs but other symbols (MERSCOPE H2AX, Xenium
        H2AFX; an ID-only gene list) share one bundle.

        Returns:
            sha256 of the canonical JSON ``{id: sorted distinct symbols}``.
        """
        table: dict[str, set[str]] = {
            gene_id: {symbol} if symbol else set()
            for gene_id, symbol in zip(self.ensembl_ids, self.symbols, strict=True)
        }
        for symbols in self.symbols_by_platform.values():
            for gene_id, symbol in zip(self.ensembl_ids, symbols, strict=True):
                if symbol:
                    table[gene_id].add(symbol)
        canonical = json.dumps(
            {gene_id: sorted(values) for gene_id, values in sorted(table.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write(self, path: Path | str) -> Path:
        """Write the panel as JSON.

        Args:
            path: Output file.

        Returns:
            The written path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return output


def load_annotation_panel(path: Path | str) -> AnnotationPanel:
    """Read and validate a ``panel_genes*.json`` file.

    Args:
        path: The file.

    Returns:
        The panel (its hash is re-checked against its IDs).
    """
    return AnnotationPanel.model_validate_json(Path(path).read_text(encoding="utf-8"))


class BundleUse(_PanelModel):
    """One use of a required bundle (a purpose on a named panel).

    Attributes:
        purpose: Why the bundle is needed.
        panel_name: Annotation panel name (``None`` if panel-independent).
        panel_file: ``panel_genes*.json`` file name (``None`` if
            panel-independent).
    """

    purpose: BundlePurpose
    panel_name: str | None = None
    panel_file: str | None = None


class RequiredBundle(_PanelModel):
    """One (reference, panel) bundle a pair x segmentation needs.

    Two uses can share one bundle, e.g. the intersection panel equal to a
    platform panel of a ``per_platform`` pair, or set c equal to set a; the
    bundle is listed once and ``uses`` names every purpose, so MAP selects
    bundles by use. ``purpose``, ``panel_name`` and ``panel_file`` repeat the
    first use.

    Attributes:
        reference_id: Store id.
        role: Reference role.
        species: Species.
        purpose: Why the bundle is needed (its first use).
        panel_name: Annotation panel name (``None`` if panel-independent).
        panel_hash: Panel hash (``None`` if panel-independent).
        panel_file: ``panel_genes*.json`` file name in the ANNOTATE_PANEL
            output directory (``None`` if panel-independent).
        n_panel_genes: Genes of that panel (``None`` if panel-independent);
            ``ANNOTATE_REFERENCE_PREP`` sizes its resources from it.
        uses: Every use of the bundle, the first one included.
    """

    reference_id: str
    role: ReferenceRole
    species: Species
    purpose: BundlePurpose
    panel_name: str | None = None
    panel_hash: str | None = None
    panel_file: str | None = None
    n_panel_genes: int | None = Field(default=None, ge=0)
    uses: list[BundleUse] = Field(default_factory=list)

    @model_validator(mode="after")
    def _first_use(self: RequiredBundle) -> RequiredBundle:
        first = BundleUse(
            purpose=self.purpose,
            panel_name=self.panel_name,
            panel_file=self.panel_file,
        )
        if not self.uses:
            self.uses = [first]
        elif self.uses[0] != first:
            raise ValueError("uses[0] must repeat purpose, panel_name and panel_file")
        return self

    def has_use(self, purpose: BundlePurpose) -> bool:
        """Return whether the bundle serves a purpose."""
        return any(use.purpose == purpose for use in self.uses)

    @property
    def key(self) -> tuple[str, str, str | None]:
        """Return the PREP task key ``(species, reference_id, panel_hash)``."""
        return (self.species, self.reference_id, self.panel_hash)


class RequiredBundles(_PanelModel):
    """``required_bundles.json`` of one pair x segmentation (plan §3.1).

    Attributes:
        schema_version: ``PANEL_SCHEMA_VERSION``.
        pair_id: Pair id.
        segmentation: Segmentation.
        species: Species.
        panel_mode: Resolved panel mode.
        status: ``"ok"``, or ``"refused"`` when no panel can be annotated.
        reasons: Why panels were refused.
        bundles: The required bundles, one per distinct key.
        n_required: ``len(bundles)``, the MAP ``groupKey`` size.
    """

    schema_version: int = PANEL_SCHEMA_VERSION
    pair_id: str | None
    segmentation: str | None
    species: Species
    panel_mode: PanelMode
    status: Literal["ok", "refused"]
    reasons: list[str] = Field(default_factory=list)
    bundles: list[RequiredBundle]
    n_required: int

    @model_validator(mode="after")
    def _check_count(self: RequiredBundles) -> RequiredBundles:
        if self.n_required != len(self.bundles):
            raise ValueError("n_required must equal the number of bundles")
        keys = [bundle.key for bundle in self.bundles]
        if len(keys) != len(set(keys)):
            raise ValueError("required bundles must be distinct")
        return self


# --------------------------------------------------------------------------
# Controls


@dataclass(frozen=True)
class ControlRegistry:
    """Control-feature rules for declared panels (plan §8.4).

    Wraps the shared registry ``merxen.control_features`` (the 10x / Vizgen
    documented control names, OD-D12). A feature type, when the source has
    one (``feature_types`` of the Xenium source table, the type a
    ``codeword_category`` or ``is_gene`` implies, a ``gene_panel.json``
    descriptor; ``feature_types_of``), decides alone, as for ProSeg
    transcripts: only ``keep_feature_types`` are genes. Otherwise a
    configured extra pattern or the platform's anchored control name pattern
    marks a control (MERSCOPE ``Blank-N``; Xenium ``NegControlProbe_``,
    ``NegControlCodeword_``, ``UnassignedCodeword_``, ``DeprecatedCodeword_``,
    ``Intergenic_Region_``, ``GenomicControl``, ``BLANK_``, ``antisense_``);
    last, the ``CONTROL_TOKENS`` substring rule, which never removes a
    feature that carries a native Ensembl ID or resolves to a reference gene.
    An anchored match is never kept.

    Attributes:
        keep_feature_types: Feature types that are genes.
        extra_patterns: Extra control-name patterns (``extra_control_patterns``).
    """

    keep_feature_types: frozenset[str] = frozenset({GENE_FEATURE_TYPE})
    extra_patterns: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def from_config(cls, config: AnnotationPanelConfig) -> ControlRegistry:
        """Build the registry from the panel config.

        Args:
            config: Panel settings.

        Returns:
            The registry.
        """
        return cls(
            keep_feature_types=frozenset(config.control_feature_types_keep),
            extra_patterns=tuple(
                re.compile(pattern) for pattern in config.extra_control_patterns
            ),
        )

    def control_reason(
        self,
        name: str,
        *,
        platform: str | None,
        feature_type: str = "",
        has_native_id: bool = False,
        is_reference_gene: bool = False,
    ) -> str | None:
        """Return why a feature is a control, or ``None`` for a gene.

        Args:
            name: Feature name.
            platform: ``"MERSCOPE"`` or ``"XENIUM"`` (selects the name
                pattern); ``None`` checks both.
            feature_type: The source's feature type, ``""`` if none.
            has_native_id: Whether the feature carries a native Ensembl ID.
            is_reference_gene: Whether its symbol is a gene of the run
                species' reference gene table; such a feature is never
                removed by the ``CONTROL_TOKENS`` substring rule (an
                anchored name match still removes it).

        Returns:
            A reason token (``feature_type_<type>``, ``extra_pattern``,
            ``name_pattern``, ``control_token``) or ``None``.
        """
        if feature_type:
            if feature_type in self.keep_feature_types:
                return None
            return "feature_type_" + _token(feature_type)
        if any(pattern.search(name) for pattern in self.extra_patterns):
            return "extra_pattern"
        if matches_control_name_pattern(name, platform):
            return "name_pattern"
        if has_control_token(name) and not (has_native_id or is_reference_gene):
            return "control_token"
        return None


def _token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def feature_type_from_codeword_category(value: Any) -> str:
    """Return the feature type a Xenium ``codeword_category`` value implies.

    The shared registry's gene categories (``predesigned_gene``,
    ``custom_gene``; ``merxen.control_features``) are ``Gene Expression``;
    every other category (negative control probe / codeword, genomic
    control, unassigned, deprecated) is kept as its own type, so the
    registry removes it as ``feature_type_<category>``.

    Args:
        value: A ``codeword_category`` cell.

    Returns:
        The feature type (``""`` for a missing value).
    """
    text = _clean_text(value)
    if not text:
        return ""
    return GENE_FEATURE_TYPE if text in XENIUM_GENE_CODEWORD_CATEGORIES else text


def feature_type_from_is_gene(value: Any) -> str:
    """Return the feature type a Xenium ``is_gene`` value implies.

    Args:
        value: An ``is_gene`` cell (bool, 0 / 1 or ``"true"`` / ``"false"``).

    Returns:
        ``Gene Expression``, ``NOT_GENE_FEATURE_TYPE`` or ``""`` (missing or
        unreadable).
    """
    if isinstance(value, bool | np.bool_):
        return GENE_FEATURE_TYPE if bool(value) else NOT_GENE_FEATURE_TYPE
    text = _clean_text(value).lower()
    if text in {"true", "t", "1", "yes", "1.0"}:
        return GENE_FEATURE_TYPE
    if text in {"false", "f", "0", "no", "0.0"}:
        return NOT_GENE_FEATURE_TYPE
    return ""


def feature_types_of(table: pd.DataFrame) -> list[str] | None:
    """Return each row's feature type from the columns that carry one.

    Per row, the first non-empty of: a feature-type column
    (``FEATURE_TYPE_COLUMNS``), the type ``codeword_category`` implies, the
    type ``is_gene`` implies (plan §8.4: the Xenium source table, the
    transcripts' category or the panel file).

    Args:
        table: A ``var`` or panel table.

    Returns:
        One type per row (``""`` where none is known), or ``None`` when the
        table has none of these columns.
    """
    columns: list[list[str]] = []
    type_column = next((c for c in FEATURE_TYPE_COLUMNS if c in table.columns), None)
    if type_column is not None:
        columns.append([_clean_text(value) for value in table[type_column]])
    if CODEWORD_CATEGORY_COLUMN in table.columns:
        columns.append(
            [
                feature_type_from_codeword_category(value)
                for value in table[CODEWORD_CATEGORY_COLUMN]
            ]
        )
    if IS_GENE_COLUMN in table.columns:
        columns.append(
            [feature_type_from_is_gene(value) for value in table[IS_GENE_COLUMN]]
        )
    if not columns:
        return None
    return [
        next((value for value in row if value), "")
        for row in zip(*columns, strict=True)
    ]


# --------------------------------------------------------------------------
# Reading declared panels


@dataclass(frozen=True)
class RawPanel:
    """A declared feature list before control removal and ID resolution.

    Attributes:
        features: One row per feature with columns ``name``, ``symbol``,
            ``native_value`` (the unversioned text of the native ID column,
            ``""`` if none), ``native_id`` (that value when it is an Ensembl
            gene ID, else ``""``), ``feature_type`` (``""`` if the source
            has none) and ``recorded_id`` (the source's own identifier when
            it is kept for the record only, e.g. a MERSCOPE codebook's
            transcript ID; ``""`` otherwise), in source order.
        source: Where it was read from.
        native_id_column: The column the native values came from.
    """

    features: pd.DataFrame
    source: PanelSource
    native_id_column: str | None = None


def _raw_from_columns(
    names: Sequence[Any],
    *,
    symbols: Sequence[Any] | None,
    native_ids: Sequence[Any] | None,
    feature_types: Sequence[Any] | None,
    source: PanelSource,
    native_id_column: str | None = None,
    recorded_ids: Sequence[Any] | None = None,
) -> RawPanel:
    cleaned_names = [_clean_text(name) for name in names]
    cleaned_symbols = (
        [_clean_text(symbol) for symbol in symbols]
        if symbols is not None
        else list(cleaned_names)
    )
    cleaned_symbols = [
        symbol or name
        for symbol, name in zip(cleaned_symbols, cleaned_names, strict=True)
    ]
    values = (
        [clean_native_value(value) for value in native_ids]
        if native_ids is not None
        else [""] * len(cleaned_names)
    )
    ids = [value if is_ensembl_gene_id(value) else "" for value in values]
    types = (
        [_clean_text(value) for value in feature_types]
        if feature_types is not None
        else [""] * len(cleaned_names)
    )
    recorded = (
        [_clean_text(value) for value in recorded_ids]
        if recorded_ids is not None
        else [""] * len(cleaned_names)
    )
    frame = pd.DataFrame(
        {
            "name": cleaned_names,
            "symbol": cleaned_symbols,
            "native_value": values,
            "native_id": ids,
            "feature_type": types,
            "recorded_id": recorded,
        }
    )
    frame = frame[frame["name"] != ""].reset_index(drop=True)
    return RawPanel(
        features=frame,
        source=source,
        native_id_column=native_id_column if native_ids is not None else None,
    )


def raw_panel_from_var(
    var: pd.DataFrame,
    *,
    source: PanelSource,
) -> RawPanel:
    """Return the declared features of an AnnData ``var`` table.

    Args:
        var: ``var`` with feature names as index; optional symbol, ID and
            feature-type columns (``SYMBOL_COLUMNS``, ``NATIVE_ID_COLUMNS``,
            ``FEATURE_TYPE_COLUMNS``). The native ID column is the first ID
            column with any value, whether or not the values are Ensembl
            IDs (symbols stored there make the resolver refuse the panel).
            An index of Ensembl IDs is used as the ID column when no ID
            column exists.
        source: Where the table came from.

    Returns:
        The raw panel.
    """
    index = [str(name) for name in var.index]
    symbol_column = next((c for c in SYMBOL_COLUMNS if c in var.columns), None)
    id_column = _first_id_column(var)
    symbols = var[symbol_column].tolist() if symbol_column is not None else None
    names: Sequence[Any] = index
    native_ids: Sequence[Any] | None = None
    used_column: str | None = id_column
    if id_column is not None:
        native_ids = var[id_column].tolist()
    elif any(is_ensembl_gene_id(strip_version(name)) for name in index):
        used_column = "index"
        # An ID index (reference-style var): the symbol column names the
        # feature, so the control name rules see symbols such as Blank-1.
        native_ids = index
        if symbols is not None:
            names = [
                _clean_text(symbol) or name
                for symbol, name in zip(symbols, index, strict=True)
            ]
    return _raw_from_columns(
        names,
        symbols=symbols,
        native_ids=native_ids,
        feature_types=feature_types_of(var),
        source=source,
        native_id_column=used_column,
    )


def _first_id_column(table: pd.DataFrame) -> str | None:
    """Return the first native ID column that holds any value.

    The values need not be Ensembl IDs: a column of symbols is still the
    declared ID column, and the resolver refuses it (``native_id_prefix``,
    the ag7 symbols-as-IDs failure) instead of silently skipping it.
    """
    for column in NATIVE_ID_COLUMNS:
        if column in table.columns and any(
            clean_native_value(value) for value in table[column]
        ):
            return column
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_panel_file(path: Path | str) -> RawPanel:
    """Read a declared panel from a vendor panel file or a gene table.

    Supported: a Xenium ``gene_panel.json`` (targets with their descriptor as
    feature type), a MERSCOPE codebook CSV (``name``, ``id``, ``barcodeType``;
    blanks recognised by name), a gene table (``.csv`` / ``.tsv`` /
    ``.parquet`` with a symbol and / or an Ensembl ID column, optional
    feature type) or the ``var`` of an ``.h5ad``.

    Args:
        path: The file.

    Returns:
        The raw panel.

    Raises:
        ValueError: If the file type or layout is not recognised.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"panel file {file_path} does not exist")
    suffix = file_path.suffix.lower()
    digest = _file_sha256(file_path)
    if suffix == ".json":
        return _read_xenium_gene_panel(file_path, digest)
    if suffix == ".h5ad":
        return raw_panel_from_var(
            read_h5ad_var(file_path),
            source=PanelSource(kind="h5ad_var", path=str(file_path), sha256=digest),
        )
    if suffix == ".parquet":
        table = pd.read_parquet(file_path)
    elif suffix in {".csv", ".tsv", ".txt"} or ".csv" in file_path.suffixes:
        is_tsv = suffix == ".tsv" or ".tsv" in file_path.suffixes
        # index_col=False: MERSCOPE codebook rows end with a trailing
        # separator (one field more than the header), which would otherwise
        # make the first column the index and shift every column left.
        table = pd.read_csv(
            file_path,
            sep="\t" if is_tsv else ",",
            comment="#",
            dtype=str,
            keep_default_na=False,
            index_col=False,
        )
    else:
        raise ValueError(f"unsupported panel file type: {file_path}")
    columns = {str(column).strip(): column for column in table.columns}
    if "barcodeType" in columns and "name" in columns:
        # Codebook ids are transcript identifiers (Ensembl transcript IDs,
        # RefSeq accessions, a UUID for a custom transgene; -1 or the blank
        # name for blanks): recorded, never used as gene IDs and never
        # counted by the native-ID prefix rules (plan §8.4; M3b review 2).
        # Only an id that is an Ensembl gene ID serves as the native ID.
        codebook_ids = table[columns["id"]].tolist() if "id" in columns else None
        gene_ids = (
            [
                value if is_ensembl_gene_id(clean_native_value(value)) else ""
                for value in codebook_ids
            ]
            if codebook_ids is not None
            else None
        )
        has_gene_ids = gene_ids is not None and any(gene_ids)
        return _raw_from_columns(
            table[columns["name"]].tolist(),
            symbols=None,
            native_ids=gene_ids if has_gene_ids else None,
            feature_types=None,
            source=PanelSource(
                kind="merscope_codebook", path=str(file_path), sha256=digest
            ),
            native_id_column="id" if has_gene_ids else None,
            recorded_ids=codebook_ids,
        )
    symbol_column = next((c for c in (*SYMBOL_COLUMNS, "name") if c in columns), None)
    id_column = _first_id_column(table.rename(columns=lambda c: str(c).strip()))
    if symbol_column is None and id_column is None:
        raise ValueError(
            f"panel table {file_path} needs a symbol column ({SYMBOL_COLUMNS}) or "
            f"an Ensembl ID column ({NATIVE_ID_COLUMNS}); found {list(columns)}"
        )
    names_column = symbol_column or id_column
    assert names_column is not None
    return _raw_from_columns(
        table[columns[names_column]].tolist(),
        symbols=(
            table[columns[symbol_column]].tolist()
            if symbol_column is not None
            else None
        ),
        native_ids=table[columns[id_column]].tolist()
        if id_column is not None
        else None,
        feature_types=feature_types_of(table.rename(columns=lambda c: str(c).strip())),
        source=PanelSource(kind="gene_table", path=str(file_path), sha256=digest),
        native_id_column=id_column,
    )


def _read_xenium_gene_panel(path: Path, digest: str) -> RawPanel:
    document: object = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload", document) if isinstance(document, dict) else None
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, list):
        raise ValueError(f"{path} is not a Xenium gene_panel.json (no payload.targets)")
    names: list[str] = []
    ids: list[str] = []
    types: list[str] = []
    for target in targets:
        target_type = target.get("type", {}) if isinstance(target, dict) else {}
        data = target_type.get("data", {}) if isinstance(target_type, dict) else {}
        descriptor = str(target_type.get("descriptor", "")).strip()
        names.append(_clean_text(data.get("name")))
        ids.append(_clean_text(data.get("id")))
        types.append(
            XENIUM_PANEL_DESCRIPTOR_TYPES.get(descriptor, f"descriptor {descriptor}")
        )
    return _raw_from_columns(
        names,
        symbols=None,
        native_ids=ids,
        feature_types=types,
        source=PanelSource(
            kind="xenium_gene_panel_json", path=str(path), sha256=digest
        ),
        native_id_column="id",
    )


def _h5ad_read_elem(node: Any) -> Any:
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def read_h5ad_var(path: Path | str) -> pd.DataFrame:
    """Read only the ``var`` table of an ``.h5ad`` file.

    Args:
        path: The file.

    Returns:
        ``var`` with feature names as index.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        var = _h5ad_read_elem(handle["var"])
    return pd.DataFrame(var)


CONTROL_FILTER_RECORD: Final = "uns/merxen_clustering_squidpy/control_feature_filter"


@dataclass(frozen=True)
class ControlFilterRecord:
    """What legacy clustering's control filter recorded in a clustered H5AD.

    ``remove_control_features`` stores the features it kept and removed
    before ``filter_genes(min_cells=...)`` drops rarely detected genes, so
    the record still lists the declared panel of a published table.

    Attributes:
        retained: Features kept by the control filter.
        removed: Control features it removed.
    """

    retained: tuple[str, ...]
    removed: tuple[str, ...]


def read_control_filter_record(path: Path | str) -> ControlFilterRecord | None:
    """Read the control-filter record of a clustered H5AD, if it has one.

    Args:
        path: The H5AD.

    Returns:
        The record, or ``None`` (prepared H5ADs have none).
    """
    import h5py

    with h5py.File(path, "r") as handle:
        node = handle.get(CONTROL_FILTER_RECORD)
        if node is None:
            return None
        record = _h5ad_read_elem(node)
    if not isinstance(record, Mapping) or "retained_features" not in record:
        return None
    return ControlFilterRecord(
        retained=_name_tuple(record.get("retained_features")),
        removed=_name_tuple(record.get("removed_control_features")),
    )


def _name_tuple(values: Any) -> tuple[str, ...]:
    """Feature names of an ``uns`` list (a numpy array once written)."""
    if values is None:
        return ()
    if isinstance(values, str | bytes):
        values = [values]
    return tuple(
        value.decode() if isinstance(value, bytes) else str(value)
        for value in np.asarray(values, dtype=object).ravel()
    )


def raw_panel_from_h5ad(
    path: Path | str,
    *,
    var: pd.DataFrame | None = None,
    kind: PanelSourceKind = "prepared_h5ad_var",
) -> RawPanel:
    """Return the declared features of an H5AD (plan §8.1, R39).

    A prepared H5AD declares its unfiltered ``var``. A published clustered
    H5AD declares what its control filter saw (the kept and the removed
    features of ``ControlFilterRecord``), so ``min_cells`` filtering never
    changes its panel hash; the features ``min_cells`` dropped carry no
    native ID and are resolved by symbol.

    Args:
        path: The H5AD.
        var: Its ``var`` if already read.
        kind: Source kind of a file without a control-filter record.

    Returns:
        The raw panel.
    """
    frame = var if var is not None else read_h5ad_var(path)
    record = read_control_filter_record(path)
    base = raw_panel_from_var(frame, source=PanelSource(kind=kind, path=str(path)))
    if record is None:
        return base
    declared = list(dict.fromkeys([*record.retained, *record.removed]))
    in_var = [str(name) for name in frame.index]
    if not set(in_var) <= set(declared):
        logger.warning(
            "%s: the control-filter record does not list every var feature; "
            "the var is the declared panel",
            path,
        )
        return base
    missing = [name for name in declared if name not in set(in_var)]
    extra = pd.DataFrame({column: [""] * len(missing) for column in base.features})
    extra["name"] = missing
    extra["symbol"] = missing
    if missing:
        logger.info(
            "%s: %d declared feature(s) absent from the min_cells-filtered var "
            "(e.g. %s) stay in the declared panel",
            path,
            len(missing),
            ", ".join(missing[:5]),
        )
    return RawPanel(
        features=pd.concat([base.features, extra], ignore_index=True),
        source=PanelSource(kind="clustered_h5ad_declared", path=str(path)),
        native_id_column=base.native_id_column,
    )


# --------------------------------------------------------------------------
# Declared panels


def pair_symbol_lookup(
    raw_panels: Sequence[RawPanel], species: Species
) -> dict[str, str]:
    """Return symbol -> native ID from the pair's own declared panels.

    Plan §8.4 step (2): a symbol-only platform (MERSCOPE) takes the ID the
    other platform (Xenium) declares for the same symbol. Symbols declared
    with two different IDs are left out.

    Args:
        raw_panels: The declared panels of the pair.
        species: Run species (IDs of another species are ignored).

    Returns:
        The lookup.
    """
    pattern = SPECIES_ID_PATTERNS[species]
    candidates: dict[str, set[str]] = {}
    for raw in raw_panels:
        for symbol, gene_id in zip(
            raw.features["symbol"], raw.features["native_id"], strict=True
        ):
            if gene_id and pattern.fullmatch(gene_id) and symbol:
                candidates.setdefault(str(symbol), set()).add(str(gene_id))
    return {
        symbol: next(iter(ids)) for symbol, ids in candidates.items() if len(ids) == 1
    }


def fallback_sources(
    species: Species,
    *,
    pair_lookup: Mapping[str, str] | None = None,
    fallback: GeneIdFallbackTable | None = None,
) -> GeneIdSources:
    """Return resolver sources from a pair lookup and an M0e fallback table.

    The packaged curated overrides are always included; the species test
    then has only the fallback table (the exact-case ratio).

    Args:
        species: Run species.
        pair_lookup: Symbol -> ID from the pair.
        fallback: The M0e fallback table (``load_gene_id_fallback_table``).

    Returns:
        The sources.
    """
    tables: dict[str, GeneTable] = {}
    if fallback is not None and fallback.n_species_rows:
        tables[species] = GeneTable.from_fallback_table(fallback)
    return GeneIdSources(
        species=species,
        gene_tables=tables,
        pair_lookup=dict(pair_lookup or {}),
        overrides=load_overrides(species),
    )


def declared_panel(
    raw: RawPanel,
    *,
    species: Species,
    platform: str | None,
    sample_id: str | None = None,
    registry: ControlRegistry | None = None,
    pair_lookup: Mapping[str, str] | None = None,
    fallback: GeneIdFallbackTable | None = None,
    sources: GeneIdSources | None = None,
    rules: ResolutionRules | None = None,
) -> DeclaredPanel:
    """Remove controls from a declared list and resolve its gene IDs.

    Controls follow the registry (feature type first; plan §8.4). The other
    features go through ``gene_ids.resolve_gene_ids`` (native ID, pair
    lookup, the run species' gene table, aliases, curated overrides; the
    exact-case species test and the refusal rules). Features that resolve
    to one ID are merged.

    Args:
        raw: The declared features.
        species: Run species.
        platform: The sample's platform (selects control name patterns).
        sample_id: The sample, for the record.
        registry: Control rules; the default keeps ``Gene Expression`` only.
        pair_lookup: Symbol -> ID from the pair (``pair_symbol_lookup``);
            ignored when ``sources`` is given (it carries its own).
        fallback: The M0e fallback table; ignored with ``sources``.
        sources: The resolver sources (``gene_ids.gene_id_sources``);
            default: ``fallback_sources(species, pair_lookup, fallback)``.
        rules: Refusal thresholds (default: the plan's).

    Returns:
        The declared panel; ``status`` is ``refused`` when the resolution
        is.
    """
    registry = registry or ControlRegistry()
    if sources is None:
        sources = fallback_sources(species, pair_lookup=pair_lookup, fallback=fallback)
    elif pair_lookup is not None:
        sources = sources.with_pair_lookup(pair_lookup)
    reference = sources.run_table
    controls: dict[str, list[str]] = {}
    kept_despite_token: list[str] = []
    inputs: list[FeatureInput] = []
    for row in raw.features.itertuples(index=False):
        name, symbol = str(row.name), str(row.symbol)
        native_id, feature_type = str(row.native_id), str(row.feature_type)
        is_reference_gene = reference is not None and reference.has_casefold(symbol)
        reason = registry.control_reason(
            name,
            platform=platform,
            feature_type=feature_type,
            has_native_id=bool(native_id),
            is_reference_gene=is_reference_gene,
        )
        if reason is not None:
            controls.setdefault(reason, []).append(name)
            continue
        if has_control_token(name):
            kept_despite_token.append(name)
        inputs.append(
            FeatureInput(
                name=name,
                symbol=symbol or name,
                native_value=str(getattr(row, "native_value", "") or native_id),
            )
        )
    resolution = resolve_gene_ids(
        inputs,
        species,
        sources,
        rules=rules,
        native_id_column=raw.native_id_column,
    )
    symbols_by_id: dict[str, list[str]] = {}
    sources_by_id: dict[str, ResolutionSource] = {}
    symbol_to_id: dict[str, str] = {}
    unresolved: dict[str, str] = {}
    other_species: list[str] = []
    for feature in resolution.features:
        if feature.gene_id is None:
            unresolved[feature.symbol] = feature.reason
            if feature.reason == "other_species_id":
                other_species.append(feature.native_value)
            continue
        symbol_to_id[feature.symbol] = feature.gene_id
        symbols_by_id.setdefault(feature.gene_id, []).append(feature.symbol)
        if feature.source is not None and feature.gene_id not in sources_by_id:
            sources_by_id[feature.gene_id] = feature.source
    ids = sorted(symbols_by_id)
    panel = DeclaredPanel(
        sample_id=sample_id,
        platform=platform,
        species=species,
        source=raw.source,
        panel_hash=compute_panel_hash(ids),
        ensembl_ids=ids,
        symbols=[symbols_by_id[gene_id][0] for gene_id in ids],
        id_sources={gene_id: sources_by_id[gene_id] for gene_id in ids},
        symbol_to_id=dict(sorted(symbol_to_id.items())),
        feature_ids=resolution.ids_by_name(),
        n_features_in=len(raw.features),
        controls_removed={reason: sorted(names) for reason, names in controls.items()},
        kept_despite_control_token=sorted(kept_despite_token),
        unresolved=dict(sorted(unresolved.items())),
        merged_duplicates={
            gene_id: symbols
            for gene_id, symbols in sorted(symbols_by_id.items())
            if len(symbols) > 1
        },
        other_species_ids=sorted(set(other_species)),
        resolution=resolution,
    )
    if panel.unresolved:
        logger.warning(
            "%s %s: %d of %d non-control features have no Ensembl ID: %s",
            sample_id or "panel",
            platform or "",
            len(panel.unresolved),
            panel.n_non_control,
            ", ".join(f"{s} ({r})" for s, r in list(panel.unresolved.items())[:20]),
        )
    if panel.status == "refused":
        logger.warning(
            "%s %s: declared panel refused: %s",
            sample_id or "panel",
            platform or "",
            panel.refusal_text(),
        )
    return panel


# --------------------------------------------------------------------------
# Panel families and modes


@dataclass(frozen=True)
class KnownPanelFamily:
    """A family record trust can be inherited from (``validated_panels.csv``).

    ``diagnostics.ValidatedPanelTable.known_families`` builds one per row.

    Attributes:
        family_id: Family id.
        species: Species.
        platforms: Platforms of the family's panel.
        panel_hash: Hash of the family's panel.
        ensembl_ids: The family panel's IDs.
        root_markers: Root marker IDs, all of which a member must contain.
    """

    family_id: str
    species: str
    platforms: frozenset[str]
    panel_hash: str
    ensembl_ids: frozenset[str]
    root_markers: frozenset[str] = frozenset()


def jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    """Return the Jaccard index of two gene sets (0 for two empty sets).

    Args:
        first: Gene IDs.
        second: Gene IDs.

    Returns:
        ``|A & B| / |A | B|``.
    """
    left, right = set(first), set(second)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def panel_family(
    ensembl_ids: Sequence[str],
    *,
    species: str,
    platforms: Sequence[str],
    known_families: Sequence[KnownPanelFamily] = (),
    min_jaccard: float = 0.95,
) -> PanelFamily:
    """Return the family a panel belongs to (plan §8.1, OD-E7).

    A family is a candidate when it has the panel's species and was
    validated on every platform the panel serves (the panel's platforms are
    a subset of the family's): a Xenium-only run of a panel validated on
    MERSCOPE and Xenium (``per_platform`` mode) keeps its family, while a
    Xenium panel never matches a MERSCOPE-only family. A panel whose hash
    is a candidate's row is ``listed`` in it. Otherwise it inherits a
    candidate when Jaccard >= ``min_jaccard`` with the family panel and it
    contains all its root markers; the most similar such family wins.
    Otherwise the panel is its own family,
    ``<species>_<platforms>_<hash prefix>``. A panel without platforms (a
    gene list) never matches a family.

    Args:
        ensembl_ids: The panel's IDs.
        species: Species.
        platforms: Platforms the panel serves.
        known_families: Families to inherit from
            (``diagnostics.load_validated_panels().known_families()``).
        min_jaccard: ``family_min_jaccard``.

    Returns:
        The family.
    """
    ids = set(ensembl_ids)
    own_hash = compute_panel_hash(sorted(ids))
    platform_set = frozenset(platform.upper() for platform in platforms)
    matched = sorted(platform_set)
    candidates = [
        family
        for family in known_families
        if family.species == species
        and platform_set
        and platform_set <= family.platforms
    ]
    listed = next((f for f in candidates if f.panel_hash == own_hash), None)
    if listed is not None:
        return PanelFamily(
            family_id=listed.family_id,
            basis="listed",
            reference_panel_hash=own_hash,
            jaccard=1.0,
            matched_platforms=matched,
        )
    best: tuple[float, KnownPanelFamily] | None = None
    for family in candidates:
        if not family.root_markers <= ids:
            continue
        score = jaccard(ids, family.ensembl_ids)
        if score >= min_jaccard and (best is None or score > best[0]):
            best = (score, family)
    if best is not None:
        return PanelFamily(
            family_id=best[1].family_id,
            basis="inherited",
            reference_panel_hash=best[1].panel_hash,
            jaccard=round(best[0], 6),
            matched_platforms=matched,
        )
    platform_token = "_".join(sorted(p.lower() for p in platform_set)) or "any"
    return PanelFamily(
        family_id=f"{species}_{platform_token}_{own_hash[:12]}",
        basis="own",
        reference_panel_hash=own_hash,
        jaccard=1.0,
    )


def resolve_panel_mode(
    panels: Sequence[DeclaredPanel],
    *,
    requested: Literal["auto", "intersection", "per_platform"] = "auto",
    min_jaccard: float = 0.9,
) -> tuple[PanelMode, float | None]:
    """Return the panel mode of a pair (plan §8.5).

    ``auto`` picks ``per_platform`` whenever exactly one of the two declared
    panels is refused, whatever their Jaccard: the intersection follows both
    declared panels and would be refused with the pair, while per platform
    the valid platform keeps its own panel and bundles (M3b review 2).

    Args:
        panels: The declared panels, one per platform.
        requested: ``annotation_panel_mode``.
        min_jaccard: ``auto`` picks ``intersection`` at or above this Jaccard.

    Returns:
        The mode (``single_sample`` for one panel) and the two panels'
        Jaccard (``None`` for one panel).

    Raises:
        ValueError: If there are no panels or more than two.
    """
    if not panels:
        raise ValueError("at least one declared panel is needed")
    if len(panels) == 1:
        return "single_sample", None
    if len(panels) > 2:
        raise ValueError(f"a pair has at most two platform panels, got {len(panels)}")
    score = jaccard(panels[0].ensembl_ids, panels[1].ensembl_ids)
    if requested == "intersection":
        return "intersection", score
    if requested == "per_platform":
        return "per_platform", score
    if sum(panel.status == "refused" for panel in panels) == 1:
        return "per_platform", score
    return ("intersection" if score >= min_jaccard else "per_platform"), score


def _platform_name(panel: DeclaredPanel) -> str:
    return (panel.platform or panel.sample_id or "sample").lower()


def declared_resolution(panel: DeclaredPanel) -> DeclaredResolution:
    """Return the resolution identity of a declared panel (provenance).

    Args:
        panel: The declared panel.

    Returns:
        Its resolution table sha256 and the gene tables consulted.
    """
    resolution = panel.resolution
    if resolution is None:
        return DeclaredResolution()
    return DeclaredResolution(
        resolution_table_sha256=resolution.table_sha256(),
        gene_tables=dict(resolution.species_check.tables),
    )


def platform_panel(
    panel: DeclaredPanel,
    *,
    kind: PanelKind,
    name: str,
    panel_mode: PanelMode,
    known_families: Sequence[KnownPanelFamily] = (),
    family_min_jaccard: float = 0.95,
) -> AnnotationPanel:
    """Return the annotation panel of one declared panel.

    Args:
        panel: The declared panel.
        kind: ``platform``, ``single_sample`` or ``gene_list``.
        name: Panel name.
        panel_mode: The pair's panel mode.
        known_families: Families to inherit from.
        family_min_jaccard: Family Jaccard threshold.

    Returns:
        The annotation panel.
    """
    platforms = [panel.platform] if panel.platform else []
    return AnnotationPanel(
        name=name,
        kind=kind,
        species=panel.species,
        platforms=platforms,
        sample_ids=[panel.sample_id] if panel.sample_id else [],
        panel_mode=panel_mode,
        panel_hash=panel.panel_hash,
        n_genes=panel.n_genes,
        ensembl_ids=list(panel.ensembl_ids),
        symbols=list(panel.symbols),
        symbols_by_platform=(
            {panel.platform: list(panel.symbols)} if panel.platform else {}
        ),
        declared_panel_hashes={_platform_name(panel): panel.panel_hash},
        declared_resolutions={_platform_name(panel): declared_resolution(panel)},
        panel_family=panel_family(
            panel.ensembl_ids,
            species=panel.species,
            platforms=platforms,
            known_families=known_families,
            min_jaccard=family_min_jaccard,
        ),
    )


def intersection_panel(
    panels: Sequence[DeclaredPanel],
    *,
    panel_mode: PanelMode,
    name: str = "intersection",
    known_families: Sequence[KnownPanelFamily] = (),
    family_min_jaccard: float = 0.95,
) -> AnnotationPanel:
    """Return the intersection of two platforms' declared panels.

    For same-panel human pairs this is E5 set a (296 genes in the evidence;
    297 once MERSCOPE ``H2AX`` resolves to Xenium's ``H2AFX`` ID, M0e).

    Args:
        panels: The two declared panels.
        panel_mode: ``intersection`` or ``per_platform``.
        name: Panel name.
        known_families: Families to inherit from.
        family_min_jaccard: Family Jaccard threshold.

    Returns:
        The intersection panel; ``symbols`` prefers the Xenium symbol.
    """
    shared = sorted(set.intersection(*(set(panel.ensembl_ids) for panel in panels)))
    symbol_maps = {
        _platform_name(panel).upper(): dict(
            zip(panel.ensembl_ids, panel.symbols, strict=True)
        )
        for panel in panels
    }
    order = sorted(symbol_maps, key=lambda platform: platform != "XENIUM")
    symbols = [
        next(
            (symbol_maps[p][gene_id] for p in order if symbol_maps[p].get(gene_id)),
            "",
        )
        for gene_id in shared
    ]
    platforms = sorted(panel.platform for panel in panels if panel.platform)
    species = panels[0].species
    return AnnotationPanel(
        name=name,
        kind="intersection",
        species=species,
        platforms=platforms,
        sample_ids=sorted(panel.sample_id for panel in panels if panel.sample_id),
        panel_mode=panel_mode,
        panel_hash=compute_panel_hash(shared),
        n_genes=len(shared),
        ensembl_ids=shared,
        symbols=symbols,
        symbols_by_platform={
            platform: [symbol_maps[platform].get(gene_id, "") for gene_id in shared]
            for platform in sorted(symbol_maps)
        },
        declared_panel_hashes={
            _platform_name(panel): panel.panel_hash for panel in panels
        },
        declared_resolutions={
            _platform_name(panel): declared_resolution(panel) for panel in panels
        },
        panel_family=panel_family(
            shared,
            species=species,
            platforms=platforms,
            known_families=known_families,
            min_jaccard=family_min_jaccard,
        ),
    )


# --------------------------------------------------------------------------
# Subset bundles (plan §3.3 step 1, §8.1, §8.7)

# MapMyCells lookup keys (``merxen.annotation.reference``): the root parent
# and the non-marker entries.
LOOKUP_ROOT_KEY: Final = "None"
LOOKUP_NON_PARENT_KEYS: Final[frozenset[str]] = frozenset({"metadata", "log"})
SUBSET_PANEL_SUFFIX: Final = "_subset"

SubsetAction = Literal["none", "subset", "own_family"]
SubsetReason = Literal["root_marker_missing", "weak_parent", "missing_frac"]


class SubsetBundleTrigger(_PanelModel):
    """Whether a dataset that lacks panel genes needs a subset bundle.

    MAP restricts a bundle's lookup to the genes a dataset has
    (``validate_lookup`` with auto-collapse). That is enough while few
    markers are lost; a **subset bundle** (markers found on the genes the
    dataset has) is needed when a missing gene is a root marker, a parent is
    left with fewer than ``weak_parent_markers`` markers, or more than
    ``subset_bundle_missing_frac`` of the panel is missing. Above
    ``own_family_missing_frac`` the subset is its own panel family with a
    full PREP (own resolvability and trust); otherwise it keeps the parent's
    family (plan §3.3, §8.1).

    Attributes:
        action: ``none``, ``subset`` or ``own_family``.
        reasons: Why a subset bundle is needed (empty for ``none``).
        n_panel_genes: Genes of the bundle's panel.
        n_missing: Panel genes the dataset lacks.
        missing_frac: ``n_missing / n_panel_genes``.
        missing_gene_ids: Their IDs (sorted).
        missing_root_markers: Root markers of the lookup among them.
        weak_parents: Parents that lost markers and keep fewer than
            ``weak_parent_markers`` (lookup key -> markers left).
        subset_bundle_missing_frac: The configured trigger share.
        own_family_missing_frac: The configured own-family share.
        weak_parent_markers: The configured weak-parent limit.
        subset_panel_hash: ``panel_hash`` of the genes the dataset has
            (``None`` for ``none``).
    """

    action: SubsetAction
    reasons: list[SubsetReason] = Field(default_factory=list)
    n_panel_genes: int
    n_missing: int
    missing_frac: float
    missing_gene_ids: list[str] = Field(default_factory=list)
    missing_root_markers: list[str] = Field(default_factory=list)
    weak_parents: dict[str, int] = Field(default_factory=dict)
    subset_bundle_missing_frac: float
    own_family_missing_frac: float
    weak_parent_markers: int
    subset_panel_hash: str | None = None

    @property
    def needs_subset(self) -> bool:
        """Return whether a subset bundle is needed."""
        return self.action != "none"


def subset_bundle_trigger(
    panel_ids: Iterable[str],
    present_ids: Iterable[str],
    lookup: Mapping[str, Any] | None = None,
    *,
    weak_parent_markers: int = 5,
    subset_bundle_missing_frac: float = 0.01,
    own_family_missing_frac: float = 0.05,
) -> SubsetBundleTrigger:
    """Decide whether a dataset's missing panel genes need a subset bundle.

    Args:
        panel_ids: The bundle panel's IDs.
        present_ids: IDs the dataset has (its resolved non-control features).
        lookup: The bundle's marker lookup (``query_markers.filtered.json``;
            parent key -> marker IDs). Without it only the missing share is
            tested.
        weak_parent_markers: A parent that loses markers and keeps fewer
            triggers a subset bundle.
        subset_bundle_missing_frac: A larger missing share triggers one.
        own_family_missing_frac: A larger missing share makes the subset its
            own family (full PREP).

    Returns:
        The decision.

    Raises:
        ValueError: If the panel is empty or the shares are not ordered.
    """
    panel = sorted(set(panel_ids))
    if not panel:
        raise ValueError("the panel has no genes")
    if own_family_missing_frac <= subset_bundle_missing_frac:
        raise ValueError(
            "own_family_missing_frac must exceed subset_bundle_missing_frac"
        )
    present = set(present_ids)
    missing = [gene_id for gene_id in panel if gene_id not in present]
    missing_set = set(missing)
    fraction = len(missing) / len(panel)
    root_lost: list[str] = []
    weak: dict[str, int] = {}
    if lookup is not None and missing:
        for key, markers in lookup.items():
            if key in LOOKUP_NON_PARENT_KEYS or not isinstance(markers, list):
                continue
            genes = [str(gene) for gene in markers]
            lost = [gene for gene in genes if gene in missing_set]
            if not lost:
                continue
            if key == LOOKUP_ROOT_KEY:
                root_lost = sorted(set(lost))
            left = len(genes) - len(lost)
            if left < weak_parent_markers:
                weak[key] = left
    reasons: list[SubsetReason] = []
    if root_lost:
        reasons.append("root_marker_missing")
    if weak:
        reasons.append("weak_parent")
    if fraction > subset_bundle_missing_frac:
        reasons.append("missing_frac")
    action: SubsetAction = "none"
    if fraction > own_family_missing_frac:
        action = "own_family"
    elif reasons:
        action = "subset"
    subset_hash = (
        compute_panel_hash([gene_id for gene_id in panel if gene_id in present])
        if action != "none" and len(missing) < len(panel)
        else None
    )
    return SubsetBundleTrigger(
        action=action,
        reasons=reasons,
        n_panel_genes=len(panel),
        n_missing=len(missing),
        missing_frac=round(fraction, 6),
        missing_gene_ids=missing,
        missing_root_markers=root_lost,
        weak_parents=dict(sorted(weak.items())),
        subset_bundle_missing_frac=subset_bundle_missing_frac,
        own_family_missing_frac=own_family_missing_frac,
        weak_parent_markers=weak_parent_markers,
        subset_panel_hash=subset_hash,
    )


def subset_trigger_for_config(
    panel_ids: Iterable[str],
    present_ids: Iterable[str],
    lookup: Mapping[str, Any] | None,
    config: AnnotationPanelConfig,
) -> SubsetBundleTrigger:
    """Return ``subset_bundle_trigger`` with the panel config's thresholds.

    Args:
        panel_ids: The bundle panel's IDs.
        present_ids: IDs the dataset has.
        lookup: The bundle's marker lookup.
        config: Panel settings (``weak_parent_markers``,
            ``subset_bundle_missing_frac``, ``own_family_missing_frac``).

    Returns:
        The decision.
    """
    return subset_bundle_trigger(
        panel_ids,
        present_ids,
        lookup,
        weak_parent_markers=config.weak_parent_markers,
        subset_bundle_missing_frac=config.subset_bundle_missing_frac,
        own_family_missing_frac=config.own_family_missing_frac,
    )


def subset_panel(
    panel: AnnotationPanel,
    present_ids: Iterable[str],
    trigger: SubsetBundleTrigger,
    *,
    name: str | None = None,
) -> AnnotationPanel:
    """Return the subset panel of a bundle panel on the genes a dataset has.

    The subset keeps its parent's family (``basis = "subset"``, trust and
    floors of the family; plan §8.1) unless the trigger says
    ``own_family``, which gives it a family of its own (full PREP with its
    own resolvability and trust). Its ``panel_hash`` keys the subset bundle
    (``merxen annotation-reference-prep --panel-genes`` builds it; MAP maps
    with it once the store has it).

    Args:
        panel: The bundle's annotation panel.
        present_ids: IDs the dataset has.
        trigger: ``subset_bundle_trigger`` for the panel and the dataset.
        name: Panel name (default ``<parent name>_subset``).

    Returns:
        The subset panel (``kind = "subset"``, ``parent_panel_hash`` the
        parent's hash, ``excluded_ids`` the missing genes).

    Raises:
        ValueError: If the trigger does not ask for a subset or no panel
            gene is present.
    """
    if not trigger.needs_subset:
        raise ValueError("the trigger does not ask for a subset bundle")
    present = set(present_ids)
    positions = [i for i, gene_id in enumerate(panel.ensembl_ids) if gene_id in present]
    if not positions:
        raise ValueError(f"no gene of panel {panel.name} is present")
    ids = [panel.ensembl_ids[i] for i in positions]
    excluded = [gene_id for gene_id in panel.ensembl_ids if gene_id not in present]
    parent_family = panel.panel_family
    if trigger.action == "subset" and parent_family is not None:
        family = PanelFamily(
            family_id=parent_family.family_id,
            basis="subset",
            reference_panel_hash=parent_family.reference_panel_hash,
            jaccard=round(len(ids) / panel.n_genes, 6),
        )
    else:
        family = panel_family(ids, species=panel.species, platforms=panel.platforms)
    return AnnotationPanel(
        name=name or f"{panel.name}{SUBSET_PANEL_SUFFIX}",
        kind="subset",
        species=panel.species,
        platforms=list(panel.platforms),
        sample_ids=list(panel.sample_ids),
        panel_mode=panel.panel_mode,
        panel_hash=compute_panel_hash(ids),
        n_genes=len(ids),
        ensembl_ids=ids,
        symbols=[panel.symbols[i] for i in positions],
        symbols_by_platform={
            platform: [symbols[i] for i in positions]
            for platform, symbols in panel.symbols_by_platform.items()
        },
        declared_panel_hashes=dict(panel.declared_panel_hashes),
        declared_resolutions=dict(panel.declared_resolutions),
        parent_panel_hash=panel.panel_hash,
        excluded_ids=excluded,
        panel_family=family,
    )


# --------------------------------------------------------------------------
# Set c: label-free platform deviation


@dataclass(frozen=True)
class SharedTissueMask:
    """The pair's shared tissue mask in the fixed platform's image frame.

    Attributes:
        mask_path: ``shared_tissue_mask.npy`` (non-zero = shared tissue).
        dataset_to_image: 3 x 3 affine from fixed-platform dataset
            coordinates (``obsm["spatial"]``) to mask ``(column, row)``.
        fixed_platform: Platform whose frame the mask uses (``XENIUM``).
        summary_path: ``registration_summary.json`` the affine came from.
    """

    mask_path: Path
    dataset_to_image: np.ndarray
    fixed_platform: str
    summary_path: Path | None = None

    def describe(self) -> dict[str, Any]:
        """Return the mask files and their sha256, for ``panel_report.json``."""
        return {
            "mask_path": str(self.mask_path),
            "mask_sha256": _file_sha256(self.mask_path),
            "summary_path": None
            if self.summary_path is None
            else str(self.summary_path),
            "summary_sha256": None
            if self.summary_path is None
            else _file_sha256(self.summary_path),
            "fixed_platform": self.fixed_platform,
        }

    def contains(self, xy: np.ndarray, *, block_rows: int = 2048) -> np.ndarray:
        """Return whether points fall inside the shared tissue.

        Args:
            xy: ``(n, 2)`` dataset coordinates in the fixed platform's frame.
            block_rows: Mask rows read per block (the mask is memory-mapped).

        Returns:
            Boolean array; points outside the image are ``False``.
        """
        mask = np.load(self.mask_path, mmap_mode="r")
        points = np.asarray(xy, dtype=np.float64)
        homogeneous = np.column_stack(
            [points[:, 0], points[:, 1], np.ones(len(points))]
        )
        image = homogeneous @ np.asarray(self.dataset_to_image, dtype=np.float64).T
        with np.errstate(invalid="ignore"):
            columns = np.floor(image[:, 0])
            rows = np.floor(image[:, 1])
        n_rows, n_columns = mask.shape[:2]
        valid = (
            np.isfinite(columns)
            & np.isfinite(rows)
            & (columns >= 0)
            & (columns < n_columns)
            & (rows >= 0)
            & (rows < n_rows)
        )
        inside = np.zeros(len(points), dtype=bool)
        if not valid.any():
            return inside
        valid_index = np.flatnonzero(valid)
        row_index = rows[valid_index].astype(np.int64)
        column_index = columns[valid_index].astype(np.int64)
        for start in range(int(row_index.min()), int(row_index.max()) + 1, block_rows):
            selected = (row_index >= start) & (row_index < start + block_rows)
            if not selected.any():
                continue
            block = np.asarray(mask[start : start + block_rows])
            inside[valid_index[selected]] = (
                block[row_index[selected] - start, column_index[selected]] > 0
            )
        return inside


def load_shared_tissue_mask(
    mask_path: Path | str,
    registration_summary_path: Path | str,
) -> SharedTissueMask:
    """Load the shared tissue mask written by the alignment stage.

    Args:
        mask_path: ``align_out/shared_tissue_mask.npy``.
        registration_summary_path: ``align_out/registration_summary.json``
            (supplies ``coordinate_frames.fixed_dataset_to_image_matrix`` and
            ``fixed_platform``).

    Returns:
        The mask.

    Raises:
        ValueError: If the mask is not a ``.npy`` file or the summary lacks
            the fixed-frame affine.
    """
    path = Path(mask_path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"the shared tissue mask must be a .npy file, got {path}")
    summary_path = Path(registration_summary_path)
    summary: object = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = summary.get("coordinate_frames") if isinstance(summary, dict) else None
    matrix = (
        frames.get("fixed_dataset_to_image_matrix")
        if isinstance(frames, dict)
        else None
    )
    if matrix is None:
        raise ValueError(
            f"{summary_path} has no coordinate_frames.fixed_dataset_to_image_matrix"
        )
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (3, 3):
        raise ValueError(
            f"fixed_dataset_to_image_matrix must be 3x3, got {affine.shape}"
        )
    fixed_platform = str((frames or {}).get("fixed_platform", "XENIUM")).upper()
    return SharedTissueMask(
        mask_path=path,
        dataset_to_image=affine,
        fixed_platform=fixed_platform,
        summary_path=summary_path,
    )


class PlatformPseudobulk(_PanelModel):
    """Label-free pseudobulk of one sample for the set-c rule.

    Attributes:
        sample_id: Sample.
        platform: Platform.
        n_objects: Objects in the count table.
        n_table_cells: Objects with ``total_counts >= min_counts`` (counts
            over non-control features).
        n_cells_used: Table cells inside the shared tissue mask (all table
            cells when no mask is applied).
        mask_applied: Whether the mask restricted the cells.
        mean_counts: Mean counts per used cell, per Ensembl ID.
    """

    sample_id: str
    platform: str
    n_objects: int
    n_table_cells: int
    n_cells_used: int
    mask_applied: bool
    mean_counts: dict[str, float]


class _H5adCounts:
    """Chunked read access to the counts of an ``.h5ad`` file (no full load)."""

    def __init__(self, path: Path) -> None:
        import h5py

        self.path = path
        self.handle = h5py.File(path, "r")
        node = (
            self.handle["layers/counts"]
            if "layers" in self.handle and "counts" in self.handle["layers"]
            else self.handle["X"]
        )
        self.node = node
        encoding = str(node.attrs.get("encoding-type", "array"))
        self.matrix: Any
        if encoding == "csr_matrix":
            try:
                from anndata.io import sparse_dataset
            except ImportError:  # anndata < 0.11
                from anndata.experimental import sparse_dataset
            self.matrix = sparse_dataset(node)
        elif encoding == "csc_matrix":
            # Row blocks of a backed CSC matrix would each read every column,
            # so a CSC matrix is loaded once and converted.
            self.matrix = _h5ad_read_elem(node).tocsr()
        else:
            self.matrix = node
        self.shape = tuple(int(value) for value in self.matrix.shape)

    def close(self) -> None:
        self.handle.close()

    def var(self) -> pd.DataFrame:
        return pd.DataFrame(_h5ad_read_elem(self.handle["var"]))

    def spatial(self) -> np.ndarray | None:
        if "obsm" in self.handle and "spatial" in self.handle["obsm"]:
            return np.asarray(_h5ad_read_elem(self.handle["obsm/spatial"]))[:, :2]
        return None

    def shape_key(self) -> str | None:
        key = "uns/merxen_clustering_squidpy/shape_key"
        if key in self.handle:
            value = _h5ad_read_elem(self.handle[key])
            return None if value is None else str(value)
        return None

    def row_blocks(self, block_rows: int) -> Iterator[tuple[int, Any]]:
        for start in range(0, self.shape[0], block_rows):
            yield start, self.matrix[start : min(start + block_rows, self.shape[0])]


def feature_columns(
    var: pd.DataFrame,
    *,
    declared: DeclaredPanel,
    platform: str | None,
    registry: ControlRegistry,
) -> tuple[np.ndarray, list[str]]:
    """Return a gene-column mask and the resolved ID of each ``var`` feature.

    A feature of the declared panel keeps its declared decision (control or
    gene, and its resolved ID), so MAP, set c and ``ANNOTATE_PANEL`` treat
    every feature alike. A feature the declared panel does not list (a
    vendor panel file that differs from the data) follows the registry and
    the declared symbol-to-ID table.

    Args:
        var: The data's ``var``.
        declared: The sample's declared panel.
        platform: The sample's platform.
        registry: Control rules.

    Returns:
        ``(is_gene, ids)``: per ``var`` feature whether it is a non-control
        feature, and its resolved ID (``""`` for controls and unresolved
        features).

    Raises:
        ValueError: If ``var`` holds a feature without a name.
    """
    raw = raw_panel_from_var(var, source=PanelSource(kind="h5ad_var"))
    if len(raw.features) != len(var):
        raise ValueError("var holds features without a name")
    control_by_name = declared.control_reason_by_name()
    declared_genes = set(declared.feature_ids) | {
        feature.name
        for feature in (declared.resolution.features if declared.resolution else [])
    }
    pattern = SPECIES_ID_PATTERNS[declared.species]
    is_gene = np.zeros(len(var), dtype=bool)
    ids: list[str] = []
    for position, row in enumerate(raw.features.itertuples(index=False)):
        name = str(row.name)
        if name in control_by_name:
            ids.append("")
            continue
        if name in declared_genes:
            is_gene[position] = True
            ids.append(declared.feature_ids.get(name, ""))
            continue
        reason = registry.control_reason(
            name,
            platform=platform,
            feature_type=str(row.feature_type),
            has_native_id=bool(row.native_id),
        )
        if reason is not None:
            ids.append("")
            continue
        is_gene[position] = True
        native = str(row.native_id)
        if native and pattern.fullmatch(native):
            ids.append(native)
        else:
            ids.append(declared.symbol_to_id.get(str(row.symbol), ""))
    return is_gene, ids


def platform_pseudobulk(
    h5ad_path: Path | str,
    *,
    declared: DeclaredPanel,
    min_counts: int,
    registry: ControlRegistry | None = None,
    mask: SharedTissueMask | None = None,
    block_rows: int = 100_000,
) -> PlatformPseudobulk:
    """Return the mean counts per table cell of each resolved gene.

    Table cells are objects whose counts over non-control features reach
    ``min_counts`` (``select_table_cells`` on the control-free table). The
    counts are read in row blocks, never loaded whole; ``layers["counts"]``
    is used when present, else ``X``.

    Args:
        h5ad_path: Prepared (or clustered) ``.h5ad``.
        declared: The sample's declared panel (symbol -> ID resolution).
        min_counts: Table-cell threshold (``ClusteringSquidpyConfig.min_counts``).
        registry: Control rules.
        mask: Shared tissue mask; ``None`` uses the whole section.
        block_rows: Rows per block.

    Returns:
        The pseudobulk.
    """
    registry = registry or ControlRegistry()
    reader = _H5adCounts(Path(h5ad_path))
    try:
        is_gene, column_ids = feature_columns(
            reader.var(),
            declared=declared,
            platform=declared.platform,
            registry=registry,
        )
        inside: np.ndarray | None = None
        if mask is not None:
            spatial = reader.spatial()
            if spatial is None:
                raise ValueError(f"{h5ad_path} has no obsm['spatial'] for the mask")
            inside = mask.contains(spatial)
        column_sums = np.zeros(reader.shape[1], dtype=np.float64)
        n_table_cells = 0
        n_used = 0
        gene_columns = np.flatnonzero(is_gene)
        for start, block in reader.row_blocks(block_rows):
            counts = block.tocsr() if hasattr(block, "tocsr") else np.asarray(block)
            totals = np.asarray(counts[:, gene_columns].sum(axis=1)).ravel()
            table = totals >= min_counts
            n_table_cells += int(table.sum())
            use = table
            if inside is not None:
                use = table & inside[start : start + len(totals)]
            n_used += int(use.sum())
            if use.any():
                column_sums += np.asarray(
                    counts[use].sum(axis=0, dtype=np.float64)
                ).ravel()
    finally:
        reader.close()
    sums_by_id: dict[str, float] = {}
    for position, gene_id in enumerate(column_ids):
        if gene_id:
            sums_by_id[gene_id] = sums_by_id.get(gene_id, 0.0) + float(
                column_sums[position]
            )
    denominator = max(n_used, 1)
    return PlatformPseudobulk(
        sample_id=declared.sample_id or "",
        platform=declared.platform or "",
        n_objects=int(reader.shape[0]),
        n_table_cells=n_table_cells,
        n_cells_used=n_used,
        mask_applied=inside is not None,
        mean_counts={
            gene_id: value / denominator
            for gene_id, value in sorted(sums_by_id.items())
        },
    )


@dataclass(frozen=True)
class CuratedSetcFamily:
    """A set-a family whose set c is a curated exclusion list (plan §3.2, D-A7).

    Attributes:
        family_id: Family id (``human_seta_e5``).
        species: Species.
        set_a_panel_hashes: Set-a panel hashes of the family (the 296-gene
            evidence panel and its post-M0e 297-gene form).
        excluded_ids: The curated platform-deviant gene IDs.
        excluded_symbols: Their symbols, by ID.
        rule: How the list was derived.
        source: Where it comes from.
        assets_sha256: sha256 of the family and exclusion asset files.
    """

    family_id: str
    species: str
    set_a_panel_hashes: frozenset[str]
    excluded_ids: tuple[str, ...]
    excluded_symbols: Mapping[str, str]
    rule: str
    source: str
    assets_sha256: Mapping[str, str]

    def describe(self) -> dict[str, Any]:
        """Return the family as it is recorded in ``panel_report.json``."""
        return {
            "family_id": self.family_id,
            "set_a_panel_hashes": sorted(self.set_a_panel_hashes),
            "n_listed": len(self.excluded_ids),
            "rule": self.rule,
            "source": self.source,
            "assets_sha256": dict(sorted(self.assets_sha256.items())),
        }


def load_curated_setc_families(species: str) -> list[CuratedSetcFamily]:
    """Load the packaged curated set-c families of a species.

    Args:
        species: ``"human"`` or ``"mouse"`` (mouse has none: set c is a
            human cross-platform sensitivity panel).

    Returns:
        The families.

    Raises:
        ValueError: If a family has no exclusions or an ID is not an
            Ensembl ID of the species.
    """
    if species not in SETC_FAMILY_FILES:
        return []
    from merxen.annotation.vocab import asset_path, load_asset_table

    family_file = SETC_FAMILY_FILES[species]
    exclusion_file = SETC_EXCLUSION_FILES[species]
    families = load_asset_table(family_file)
    exclusions = load_asset_table(exclusion_file)
    digests = {
        name: _file_sha256(asset_path(name)) for name in (family_file, exclusion_file)
    }
    pattern = SPECIES_ID_PATTERNS[species]
    result: list[CuratedSetcFamily] = []
    for family_id, rows in families.groupby("family_id", sort=True):
        listed = exclusions[exclusions["family_id"] == family_id]
        if listed.empty:
            raise ValueError(f"{exclusion_file}: family {family_id!r} lists no gene")
        bad = [gene for gene in listed["ensembl_id"] if not pattern.match(gene)]
        if bad:
            raise ValueError(f"{exclusion_file}: not {species} Ensembl IDs: {bad}")
        result.append(
            CuratedSetcFamily(
                family_id=str(family_id),
                species=species,
                set_a_panel_hashes=frozenset(rows["set_a_panel_hash"]),
                excluded_ids=tuple(sorted(listed["ensembl_id"])),
                excluded_symbols=dict(
                    zip(listed["ensembl_id"], listed["gene_symbol"], strict=True)
                ),
                rule=str(rows["rule"].iloc[0]),
                source=str(rows["source"].iloc[0]),
                assets_sha256=digests,
            )
        )
    return result


def curated_setc_family(
    set_a: AnnotationPanel, families: Sequence[CuratedSetcFamily]
) -> CuratedSetcFamily | None:
    """Return the curated set-c family of a set-a panel, if it has one.

    A set a belongs to a family when its panel hash, or the reference hash of
    the panel family it inherited (M3b), is one of the family's set-a hashes.

    Args:
        set_a: The set-a (intersection or gene-list) panel.
        families: Curated families.

    Returns:
        The family, or ``None`` (set c then follows the label-free rule).
    """
    hashes = {set_a.panel_hash}
    if set_a.panel_family is not None:
        hashes.add(set_a.panel_family.reference_panel_hash)
    for family in families:
        if family.species == set_a.species and hashes & family.set_a_panel_hashes:
            return family
    return None


class SetcReport(_PanelModel):
    """How set c was derived (``panel_report.json``; plan §3.2, D-A7).

    Attributes:
        basis: ``curated_family_list`` (the seeded set-a family: the E5
            exclusions) or ``label_free_rule`` (fallback for other families).
        rule: The rule applied.
        curated_family: The curated family (``basis`` curated), else ``None``.
        max_abs_log2_deviation: Threshold of the label-free rule.
        log2_pseudocount: Pseudocount added to both means.
        pair_median_log2_ratio: Median ``log2(mean X / mean M)`` over set a
            (``None`` without pseudobulk).
        mask_applied: Whether the shared tissue mask restricted the cells.
        mask_reason: Why the mask was not applied, when it was not.
        mask: The mask files and their sha256, when a mask was given.
        pseudobulk: Cell counts per platform (means omitted; empty without
            prepared data).
        n_set_a: Set-a genes.
        n_setc: Set-c genes.
        excluded: Centred log2 ratio of each excluded gene by ID (``None``
            without pseudobulk).
        excluded_symbols: Symbol of each excluded gene, by ID.
        centred_log2_ratio: Centred log2 ratio of every set-a gene, by ID.
        label_free_cross_check: For a curated set c with pseudobulk, what the
            label-free rule would drop and how it compares; else ``None``.
    """

    basis: Literal["curated_family_list", "label_free_rule"]
    rule: str
    curated_family: dict[str, Any] | None = None
    max_abs_log2_deviation: float
    log2_pseudocount: float
    pair_median_log2_ratio: float | None
    mask_applied: bool
    mask_reason: str | None = None
    mask: dict[str, Any] | None = None
    pseudobulk: dict[str, dict[str, int | str | bool]]
    n_set_a: int
    n_setc: int
    excluded: dict[str, float | None]
    excluded_symbols: dict[str, str]
    centred_log2_ratio: dict[str, float]
    label_free_cross_check: dict[str, Any] | None = None


SETC_RULE: Final = (
    "|log2((mean_X + c) / (mean_M + c)) - median over set a| > threshold, "
    "label-free pseudobulk over table cells (inside the shared tissue mask "
    "when available)"
)
SETC_CURATED_RULE: Final = "set a minus the curated family exclusion list"


def _centred_ratios(
    ids: Sequence[str],
    pseudobulk: Mapping[str, PlatformPseudobulk],
    log2_pseudocount: float,
) -> tuple[np.ndarray, float]:
    for platform in ("MERSCOPE", "XENIUM"):
        if platform not in pseudobulk:
            raise ValueError(f"set c needs a {platform} pseudobulk")
        if pseudobulk[platform].n_cells_used == 0:
            raise ValueError(f"set c: no {platform} table cells were used")
    xenium = pseudobulk["XENIUM"].mean_counts
    merscope = pseudobulk["MERSCOPE"].mean_counts
    ratios = np.array(
        [
            np.log2(
                (xenium.get(gene_id, 0.0) + log2_pseudocount)
                / (merscope.get(gene_id, 0.0) + log2_pseudocount)
            )
            for gene_id in ids
        ],
        dtype=np.float64,
    )
    median = float(np.median(ratios)) if len(ratios) else 0.0
    return ratios - median, median


def setc_panel(
    set_a: AnnotationPanel,
    pseudobulk: Mapping[str, PlatformPseudobulk] | None,
    *,
    curated: CuratedSetcFamily | None = None,
    max_abs_log2_deviation: float = 2.0,
    log2_pseudocount: float = 1e-3,
    mask_reason: str | None = None,
    mask: Mapping[str, Any] | None = None,
    known_families: Sequence[KnownPanelFamily] = (),
    family_min_jaccard: float = 0.95,
) -> tuple[AnnotationPanel, SetcReport]:
    """Return set c: set a minus the platform-deviant genes.

    * ``curated`` given (the seeded set-a family): set a minus the curated
      list, the E5 set c (32 genes excluded label-aware, in confident
      oligodendrocytes, in >= 3 of 4 pairs). The panel is the same for every
      pair of the family. With ``pseudobulk`` the label-free rule is computed
      too, as a cross-check only.
    * Otherwise the label-free rule (plan §3.2): ``r = log2((mean_X + c) /
      (mean_M + c))`` per set-a gene from pseudobulk over table cells; a gene
      is deviant when ``|r - median(r)| > max_abs_log2_deviation``. On the
      E5 pairs this rule drops 44-73 genes, including canonical markers, so
      it is a flagged fallback for families without a curated list.

    Args:
        set_a: The intersection (or gene-list) panel.
        pseudobulk: Pseudobulk per platform (``"MERSCOPE"``, ``"XENIUM"``);
            may be ``None`` only with ``curated``.
        curated: The curated family of ``set_a``.
        max_abs_log2_deviation: Label-free threshold (2).
        log2_pseudocount: ``c`` (1e-3, as the E5 ratios).
        mask_reason: Why no mask was applied, for the report.
        mask: The mask's files and sha256, for the report.
        known_families: Families to inherit from.
        family_min_jaccard: Family Jaccard threshold.

    Returns:
        The set-c panel and its report.

    Raises:
        ValueError: If the label-free rule lacks a platform's pseudobulk or
            a platform used no cells.
    """
    ids = list(set_a.ensembl_ids)
    centred: np.ndarray | None = None
    median: float | None = None
    cross_check_error: str | None = None
    if pseudobulk is not None:
        try:
            centred, median = _centred_ratios(ids, pseudobulk, log2_pseudocount)
        except ValueError as error:
            if curated is None:
                raise
            # The curated set c does not depend on the pseudobulk.
            cross_check_error = str(error)
            pseudobulk = None
    elif curated is None:
        raise ValueError("the label-free set-c rule needs a pseudobulk per platform")
    label_free = (
        None
        if centred is None
        else [
            gene_id
            for gene_id, value in zip(ids, centred, strict=True)
            if abs(value) > max_abs_log2_deviation
        ]
    )
    if curated is not None:
        listed = set(curated.excluded_ids)
        excluded = [gene_id for gene_id in ids if gene_id in listed]
    else:
        assert label_free is not None
        excluded = label_free
    excluded_set = set(excluded)
    kept = [gene_id for gene_id in ids if gene_id not in excluded_set]
    symbol_of = dict(zip(set_a.ensembl_ids, set_a.symbols, strict=True))
    position = {gene_id: index for index, gene_id in enumerate(set_a.ensembl_ids)}
    panel = AnnotationPanel(
        name="setc",
        kind="setc",
        species=set_a.species,
        platforms=list(set_a.platforms),
        sample_ids=list(set_a.sample_ids),
        panel_mode=set_a.panel_mode,
        panel_hash=compute_panel_hash(kept),
        n_genes=len(kept),
        ensembl_ids=kept,
        symbols=[symbol_of[gene_id] for gene_id in kept],
        symbols_by_platform={
            platform: [symbols[position[gene_id]] for gene_id in kept]
            for platform, symbols in set_a.symbols_by_platform.items()
        },
        declared_panel_hashes=dict(set_a.declared_panel_hashes),
        declared_resolutions=dict(set_a.declared_resolutions),
        parent_panel_hash=set_a.panel_hash,
        excluded_ids=excluded,
        panel_family=panel_family(
            kept,
            species=set_a.species,
            platforms=set_a.platforms,
            known_families=known_families,
            min_jaccard=family_min_jaccard,
        ),
    )
    ratio_of = (
        {}
        if centred is None
        else {
            gene_id: round(float(value), 6)
            for gene_id, value in zip(ids, centred, strict=True)
        }
    )
    cross_check: dict[str, Any] | None = (
        None if cross_check_error is None else {"error": cross_check_error}
    )
    if curated is not None and label_free is not None:
        label_free_set = set(label_free)
        cross_check = {
            "rule": SETC_RULE,
            "n_excluded": len(label_free),
            "n_in_curated_list": len(label_free_set & excluded_set),
            "only_label_free": {
                gene_id: symbol_of[gene_id]
                for gene_id in label_free
                if gene_id not in excluded_set
            },
            "only_curated": {
                gene_id: symbol_of[gene_id]
                for gene_id in excluded
                if gene_id not in label_free_set
            },
        }
    report = SetcReport(
        basis="curated_family_list" if curated is not None else "label_free_rule",
        rule=SETC_CURATED_RULE if curated is not None else SETC_RULE,
        curated_family=None
        if curated is None
        else {
            **curated.describe(),
            "n_listed_in_set_a": len(excluded),
            "listed_not_in_set_a": sorted(set(curated.excluded_ids) - set(ids)),
        },
        max_abs_log2_deviation=max_abs_log2_deviation,
        log2_pseudocount=log2_pseudocount,
        pair_median_log2_ratio=None if median is None else round(median, 6),
        mask_applied=pseudobulk is not None
        and all(item.mask_applied for item in pseudobulk.values()),
        mask_reason=mask_reason if pseudobulk is not None else "no prepared data",
        mask=None if mask is None else dict(mask),
        pseudobulk={}
        if pseudobulk is None
        else {
            platform: {
                "sample_id": item.sample_id,
                "n_objects": item.n_objects,
                "n_table_cells": item.n_table_cells,
                "n_cells_used": item.n_cells_used,
                "mask_applied": item.mask_applied,
            }
            for platform, item in sorted(pseudobulk.items())
        },
        n_set_a=len(ids),
        n_setc=len(kept),
        excluded={gene_id: ratio_of.get(gene_id) for gene_id in excluded},
        excluded_symbols={gene_id: symbol_of[gene_id] for gene_id in excluded},
        centred_log2_ratio=ratio_of,
        label_free_cross_check=cross_check,
    )
    return panel, report


# --------------------------------------------------------------------------
# Required bundles


@dataclass(frozen=True)
class PanelFile:
    """An annotation panel and the file it is written to.

    Attributes:
        panel: The panel.
        file_name: File name in the ANNOTATE_PANEL output directory.
    """

    panel: AnnotationPanel
    file_name: str


def required_bundles(
    panels: Mapping[str, PanelFile],
    *,
    references: Sequence[AnnotationReferenceSpec],
    species: Species,
    panel_mode: PanelMode,
    segmentation: str | None = None,
    xplat_sensitivity: str = "geneset_c",
    xplat_sensitivity_segmentations: Sequence[str] = ("proseg_hybrid",),
    refused_panels: Sequence[str] = (),
) -> list[RequiredBundle]:
    """Return the bundles one pair x segmentation needs (plan §3.1, §8.5).

    * Every reference with a panel role (primary, secondary, sensitivity,
      likelihood) needs a bundle on each annotation panel: the intersection
      panel (``intersection``), each platform panel (``per_platform``) or
      the sample's panel (``single_sample`` / gene list).
    * The primary reference also needs set c (same-panel pairs, when
      ``xplat_sensitivity`` is on for this segmentation) and, for
      ``per_platform`` pairs, the intersection panel (cross-platform runs).
      A ``per_platform`` human pair therefore needs 5 bundles.
    * Panel-independent references (region shares) need one bundle without
      a panel; resolvability references are built inside PREP.

    Args:
        panels: Annotation panels by name (``intersection``, ``setc``,
            platform names, ``sample``, ``gene_list``).
        references: The species' reference specs.
        species: Species.
        panel_mode: Resolved panel mode.
        segmentation: The segmentation (for the set-c segmentation filter).
        xplat_sensitivity: ``off`` disables the set-c bundle.
        xplat_sensitivity_segmentations: Segmentations with set-c runs.
        refused_panels: Panel names that get no bundle.

    Returns:
        Distinct bundles (one per key) in a stable order; a bundle serving
        several purposes lists each in ``uses``.
    """
    refused = set(refused_panels)
    if panel_mode == "intersection":
        annotation_names = ["intersection"]
    elif panel_mode == "per_platform":
        annotation_names = sorted(
            name for name, item in panels.items() if item.panel.kind == "platform"
        )
    else:
        annotation_names = [
            name
            for name, item in panels.items()
            if item.panel.kind in {"single_sample", "gene_list"}
        ]
    annotation_names = [name for name in annotation_names if name in panels]
    wants_setc = (
        xplat_sensitivity != "off"
        and "setc" in panels
        and (
            segmentation is None or segmentation in set(xplat_sensitivity_segmentations)
        )
    )
    bundles: list[RequiredBundle] = []
    by_key: dict[tuple[str, str, str | None], RequiredBundle] = {}

    def add(
        spec: AnnotationReferenceSpec, purpose: BundlePurpose, name: str | None
    ) -> None:
        panel_file = panels.get(name) if name is not None else None
        if name is not None and (panel_file is None or name in refused):
            return
        bundle = RequiredBundle(
            reference_id=spec.reference_id,
            role=spec.role,
            species=species,
            purpose=purpose,
            panel_name=name,
            panel_hash=None if panel_file is None else panel_file.panel.panel_hash,
            panel_file=None if panel_file is None else panel_file.file_name,
            n_panel_genes=None if panel_file is None else panel_file.panel.n_genes,
        )
        existing = by_key.get(bundle.key)
        if existing is None:
            by_key[bundle.key] = bundle
            bundles.append(bundle)
        elif bundle.uses[0] not in existing.uses:
            # One PREP bundle, several uses: keep every purpose for MAP.
            existing.uses.append(bundle.uses[0])

    for spec in references:
        if spec.species != species:
            continue
        if spec.role in PANEL_INDEPENDENT_ROLES:
            add(spec, "panel_independent", None)
            continue
        if spec.role not in PANEL_ROLES:
            continue
        for name in annotation_names:
            add(spec, "annotation", name)
        if spec.role == "primary":
            if wants_setc:
                add(spec, "setc_sensitivity", "setc")
            if panel_mode == "per_platform":
                add(spec, "intersection_xpanel", "intersection")
    return bundles


# --------------------------------------------------------------------------
# ANNOTATE_PANEL


@dataclass(frozen=True)
class PreparedSample:
    """One sample of the prepared directory.

    Attributes:
        sample_id: Sample id.
        platform: ``"MERSCOPE"`` or ``"XENIUM"``.
        h5ad_path: The prepared H5AD.
    """

    sample_id: str
    platform: str
    h5ad_path: Path


def prepared_samples(
    prepared_dir: Path | str,
    *,
    clustering_config: Mapping[str, Any] | None = None,
    platforms: Sequence[str] | None = None,
) -> list[PreparedSample]:
    """List the samples of a ``CLUSTERING_SQUIDPY_PREPARE`` output directory.

    Args:
        prepared_dir: Directory holding ``manifest.json`` and
            ``<platform>/<sid>_prepared.h5ad``.
        clustering_config: ``clustering_squidpy_config.json`` content; its
            ``samples[].platform`` wins over the directory name.
        platforms: Keep only these platforms.

    Returns:
        The samples, sorted by platform.

    Raises:
        FileNotFoundError: If the manifest or an H5AD is missing.
    """
    root = Path(prepared_dir)
    manifest: object = json.loads(
        (root / PREPARED_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    entries = manifest.get("samples") if isinstance(manifest, dict) else None
    if not isinstance(entries, dict):
        raise ValueError(f"{root / PREPARED_MANIFEST_FILE} has no samples mapping")
    configured: dict[str, str] = {}
    for sample in (clustering_config or {}).get("samples", []) or []:
        if isinstance(sample, dict) and sample.get("sample_id"):
            configured[str(sample["sample_id"])] = str(
                sample.get("platform", "")
            ).upper()
    wanted = {platform.upper() for platform in platforms} if platforms else None
    samples: list[PreparedSample] = []
    for sample_id, relative in entries.items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"prepared H5AD {path} is missing")
        platform = (
            configured.get(str(sample_id)) or Path(str(relative)).parent.name.upper()
        )
        if platform not in PLATFORMS:
            raise ValueError(f"cannot tell the platform of prepared sample {sample_id}")
        if wanted is None or platform in wanted:
            samples.append(
                PreparedSample(
                    sample_id=str(sample_id), platform=platform, h5ad_path=path
                )
            )
    samples.sort(key=lambda sample: (sample.platform, sample.sample_id))
    platforms_seen = [sample.platform for sample in samples]
    if len(platforms_seen) != len(set(platforms_seen)):
        raise ValueError(f"one sample per platform expected, got {platforms_seen}")
    return samples


def load_fallback_table(
    path: Path | str | None, species: Species
) -> GeneIdFallbackTable | None:
    """Load the configured M0e gene-ID fallback table (no network).

    Args:
        path: ``annotation_gene_id_fallback_csv`` (a local ``gene.csv`` or a
            reference ``.h5ad``), or ``None``.
        species: Run species.

    Returns:
        The table, or ``None`` without a path.
    """
    if path is None:
        return None
    from merxen.analysis.mapmycells import load_gene_id_fallback_table

    return load_gene_id_fallback_table(path, query_species=species)


@dataclass
class PanelComputation:
    """Everything ``compute_panel`` wrote.

    Attributes:
        report: ``panel_report.json`` content.
        panels: Annotation panels by name, with their file names.
        required: ``required_bundles.json`` content.
        paths: Written files by role (``report``, ``required``, panel names).
    """

    report: dict[str, Any]
    panels: dict[str, PanelFile]
    required: RequiredBundles
    paths: dict[str, Path] = field(default_factory=dict)


def _panel_file_name(name: str, panel_mode: PanelMode) -> str:
    if name == "setc":
        return PANEL_GENES_SETC_FILE
    if panel_mode == "per_platform":
        return f"panel_genes_{name}.json"
    return PANEL_GENES_FILE


def _cross_platform_id_matches(panels: Sequence[DeclaredPanel]) -> list[dict[str, str]]:
    """Shared IDs declared with different symbols (e.g. MERSCOPE H2AX, Xenium H2AFX)."""
    if len(panels) != 2:
        return []
    first, second = panels
    first_symbols = dict(zip(first.ensembl_ids, first.symbols, strict=True))
    second_symbols = dict(zip(second.ensembl_ids, second.symbols, strict=True))
    matches = []
    for gene_id in sorted(set(first_symbols) & set(second_symbols)):
        if first_symbols[gene_id] != second_symbols[gene_id]:
            matches.append(
                {
                    "ensembl_id": gene_id,
                    _platform_name(first): first_symbols[gene_id],
                    _platform_name(second): second_symbols[gene_id],
                    f"{_platform_name(first)}_source": first.id_sources[gene_id],
                    f"{_platform_name(second)}_source": second.id_sources[gene_id],
                }
            )
    return matches


def _declared_report(panel: DeclaredPanel) -> dict[str, Any]:
    resolution = panel.resolution
    return {
        "sample_id": panel.sample_id,
        "platform": panel.platform,
        "source": panel.source.model_dump(mode="json"),
        "status": panel.status,
        "refusal_reasons": panel.refusal_reasons,
        "refusal_details": (
            {} if resolution is None else dict(resolution.refusal_details)
        ),
        "panel_hash": panel.panel_hash,
        "n_features_in": panel.n_features_in,
        "n_controls_removed": sum(len(v) for v in panel.controls_removed.values()),
        "controls_removed": panel.controls_removed,
        "kept_despite_control_token": panel.kept_despite_control_token,
        "n_non_control": panel.n_non_control,
        "n_genes": panel.n_genes,
        "gene_id_resolution": panel.resolution_counts(),
        "gene_id_resolution_share": round(panel.resolution_share, 6),
        "resolved_by_pair_lookup": {
            symbol: gene_id
            for symbol, gene_id in panel.symbol_to_id.items()
            if panel.id_sources.get(gene_id) == "pair_lookup"
        },
        "resolved_by_fallback_table": _resolved_by(panel, "fallback_table"),
        "resolved_by_symbol_fallback": _resolved_by(panel, "symbol_fallback"),
        "resolved_by_alias": _resolved_by(panel, "alias"),
        "resolved_by_override": _resolved_by(panel, "override"),
        "unresolved": panel.unresolved,
        "merged_duplicates": panel.merged_duplicates,
        "other_species_ids": panel.other_species_ids,
        "species_check": (
            None
            if resolution is None
            else resolution.species_check.model_dump(mode="json")
        ),
        "native_id_column": None if resolution is None else resolution.native_id_column,
        "native_prefix_share": (
            None if resolution is None else resolution.native_prefix_share
        ),
        "other_species_share": (
            None if resolution is None else resolution.other_species_share
        ),
        "symbols_as_ids": [] if resolution is None else resolution.symbols_as_ids,
        "release_drift": {} if resolution is None else resolution.release_drift,
        "resolution_table_sha256": (
            None if resolution is None else resolution.table_sha256()
        ),
    }


def _resolved_by(panel: DeclaredPanel, source: str) -> dict[str, str]:
    """Features resolved by one source (symbol -> ID)."""
    if panel.resolution is None:
        return {
            symbol: gene_id
            for symbol, gene_id in panel.symbol_to_id.items()
            if panel.id_sources.get(gene_id) == source
        }
    return {
        feature.symbol: feature.gene_id
        for feature in panel.resolution.features
        if feature.gene_id is not None and feature.source == source
    }


def _panel_refusals(
    built: Mapping[str, AnnotationPanel],
    declared: Sequence[DeclaredPanel],
) -> dict[str, str]:
    """Annotation panels built from a declared panel the resolver refused.

    A platform panel follows its own declared panel; the intersection and
    set c follow every declared panel of the pair, since both platforms'
    data feed them.
    """
    refused_declared = {
        _platform_name(panel): panel for panel in declared if panel.status == "refused"
    }
    if not refused_declared:
        return {}
    text = "; ".join(
        f"declared {name} panel refused ({panel.refusal_text()})"
        for name, panel in sorted(refused_declared.items())
    )
    refusals: dict[str, str] = {}
    for name, panel in built.items():
        if panel.kind in {"platform", "single_sample", "gene_list"}:
            own = name if name in refused_declared else None
            if own is None and panel.kind != "platform" and refused_declared:
                own = next(iter(refused_declared))
            if own is not None:
                item = refused_declared[own]
                refusals[name] = f"declared {own} panel refused ({item.refusal_text()})"
        else:
            refusals[name] = text
    return refusals


def _mask_for_samples(
    mask: SharedTissueMask | None, samples: Sequence[PreparedSample]
) -> tuple[SharedTissueMask | None, str | None]:
    """Return the mask when every sample's coordinates share its frame."""
    if mask is None:
        return None, "no_shared_tissue_mask"
    for sample in samples:
        if sample.platform == mask.fixed_platform:
            continue
        reader = _H5adCounts(sample.h5ad_path)
        try:
            shape_key = reader.shape_key() or ""
        finally:
            reader.close()
        if "_aligned" not in shape_key:
            return None, (
                f"{sample.platform} coordinates are not in the {mask.fixed_platform} "
                f"frame (shape key {shape_key!r})"
            )
    return mask, None


def compute_panel(
    prepared_dir: Path | str,
    species: Species,
    platforms: Sequence[str] | None = None,
    *,
    output_dir: Path | str,
    config: AnnotationConfig | None = None,
    pair_id: str | None = None,
    segmentation: str | None = None,
    clustering_config: Mapping[str, Any] | None = None,
    panel_files: Mapping[str, Path] | None = None,
    shared_mask: SharedTissueMask | None = None,
    min_counts: int | None = None,
    known_families: Sequence[KnownPanelFamily] | None = None,
    setc_families: Sequence[CuratedSetcFamily] | None = None,
    require_shared_mask: bool = False,
    validated: ValidatedPanelTable | None = None,
) -> PanelComputation:
    """Run ``ANNOTATE_PANEL`` for one pair x segmentation (plan §3.2).

    Args:
        prepared_dir: ``CLUSTERING_SQUIDPY_PREPARE`` output directory.
        species: Run species.
        platforms: Platforms to use (default: all prepared samples).
        output_dir: Where the panel files go.
        config: Annotation config (references, panel settings); ``None``
            uses the species defaults.
        pair_id: Pair id (default: the clustering config's).
        segmentation: Segmentation (default: the clustering config's).
        clustering_config: ``clustering_squidpy_config.json`` content.
        panel_files: Declared-panel files keyed by sample id or platform
            (vendor panel file, codebook or source table); samples without
            one use the prepared H5AD's unfiltered ``var``.
        shared_mask: The pair's shared tissue mask, for set c.
        min_counts: Table-cell threshold (default: the clustering config's,
            then the annotation config's, then 10).
        known_families: Families to inherit from (default: the rows of the
            validated table).
        setc_families: Curated set-c families (default: the packaged ones).
        require_shared_mask: Refuse a label-free set c whose pseudobulk
            cannot use the shared tissue mask (pipeline runs of aligned
            pairs: a whole-section or stale mask must never define set c).
            A curated set c needs no mask.
        validated: The validated families (default:
            ``AnnotationPanelConfig.validated_panels_path``, else the packaged
            ``validated_panels.csv``).

    Returns:
        What was written.
    """
    config = config or AnnotationConfig(species=species)
    if config.species != species:
        raise ValueError(f"annotation config is for {config.species}, not {species}")
    if setc_families is None:
        setc_families = load_curated_setc_families(species)
    validated, known_families = _validated_families(
        config, species, validated, known_families
    )
    panel_config = config.panel
    registry = ControlRegistry.from_config(panel_config)
    clustering = dict(clustering_config or {})
    pair_id = pair_id or clustering.get("pair_id")
    samples = prepared_samples(
        prepared_dir, clustering_config=clustering, platforms=platforms
    )
    if not samples:
        raise ValueError(f"no prepared samples in {prepared_dir}")
    if segmentation is None:
        segmentations = {
            str(sample.get("segmentation"))
            for sample in clustering.get("samples", []) or []
            if isinstance(sample, dict) and sample.get("segmentation")
        }
        segmentation = segmentations.pop() if len(segmentations) == 1 else None
    min_counts_source = "argument"
    if min_counts is None and clustering.get("min_counts") is not None:
        min_counts, min_counts_source = (
            int(clustering["min_counts"]),
            "clustering_config",
        )
    if min_counts is None and config.min_counts is not None:
        min_counts, min_counts_source = config.min_counts, "annotation_config"
    if min_counts is None:
        min_counts, min_counts_source = DEFAULT_MIN_COUNTS, "default"
    files = dict(panel_files or {})
    raws: list[tuple[PreparedSample, RawPanel]] = []
    for sample in samples:
        panel_path = files.get(sample.sample_id) or files.get(sample.platform)
        if panel_path is not None:
            raw = read_panel_file(panel_path)
        else:
            raw = raw_panel_from_h5ad(sample.h5ad_path)
        raws.append((sample, raw))
    lookup = pair_symbol_lookup([raw for _, raw in raws], species)
    sources = gene_id_sources(panel_config, species, pair_lookup=lookup)
    rules = ResolutionRules.from_config(panel_config)
    declared = [
        declared_panel(
            raw,
            species=species,
            platform=sample.platform,
            sample_id=sample.sample_id,
            registry=registry,
            sources=sources,
            rules=rules,
        )
        for sample, raw in raws
    ]
    mode, pair_jaccard = resolve_panel_mode(
        declared,
        requested=panel_config.panel_mode,
        min_jaccard=panel_config.intersection_min_jaccard,
    )
    family_min_jaccard = panel_config.family_min_jaccard
    built: dict[str, AnnotationPanel] = {}
    if mode == "single_sample":
        built["sample"] = platform_panel(
            declared[0],
            kind="single_sample",
            name="sample",
            panel_mode=mode,
            known_families=known_families,
            family_min_jaccard=family_min_jaccard,
        )
    else:
        built["intersection"] = intersection_panel(
            declared,
            panel_mode=mode,
            known_families=known_families,
            family_min_jaccard=family_min_jaccard,
        )
        if mode == "per_platform":
            for panel in declared:
                name = _platform_name(panel)
                built[name] = platform_panel(
                    panel,
                    kind="platform",
                    name=name,
                    panel_mode=mode,
                    known_families=known_families,
                    family_min_jaccard=family_min_jaccard,
                )
    setc_report: SetcReport | None = None
    setc_skipped: str | None = None
    any_refused = any(panel.status == "refused" for panel in declared)
    if mode == "intersection" and species == "human" and any_refused:
        setc_skipped = "a declared panel of the pair is refused (gene-ID resolution)"
    elif mode == "intersection" and species == "human":
        curated = curated_setc_family(built["intersection"], setc_families)
        mask, mask_reason = _mask_for_samples(shared_mask, samples)
        if curated is None and mask is None and require_shared_mask:
            setc_skipped = (
                "label-free set c needs the shared tissue mask of this aligned "
                f"pair ({mask_reason}); set a has no curated set-c family"
            )
            logger.warning("Set c refused: %s", setc_skipped)
        else:
            if curated is None:
                logger.warning(
                    "Set a %s has no curated set-c family; set c follows the "
                    "label-free fallback rule (flagged label_free_rule)",
                    built["intersection"].panel_hash[:16],
                )
            pseudobulk = {
                sample.platform: platform_pseudobulk(
                    sample.h5ad_path,
                    declared=panel,
                    min_counts=min_counts,
                    registry=registry,
                    mask=mask,
                )
                for sample, panel in zip(samples, declared, strict=True)
            }
            built["setc"], setc_report = setc_panel(
                built["intersection"],
                pseudobulk,
                curated=curated,
                max_abs_log2_deviation=panel_config.setc_max_abs_log2_deviation,
                log2_pseudocount=panel_config.setc_log2_pseudocount,
                mask_reason=mask_reason,
                mask=None if shared_mask is None else shared_mask.describe(),
                known_families=known_families,
                family_min_jaccard=family_min_jaccard,
            )
    elif mode == "intersection":
        setc_skipped = "set c is a human cross-platform sensitivity panel"
    refused: dict[str, str] = _panel_refusals(built, declared)
    for name, annotation_panel in built.items():
        if name not in refused and (
            annotation_panel.n_genes < panel_config.min_mapped_genes
        ):
            refused[name] = (
                f"{annotation_panel.n_genes} resolved genes < min_mapped_genes "
                f"{panel_config.min_mapped_genes}"
            )
    panel_file_map = {
        name: PanelFile(panel=panel, file_name=_panel_file_name(name, mode))
        for name, panel in built.items()
    }
    bundles = required_bundles(
        panel_file_map,
        references=config.references,
        species=species,
        panel_mode=mode,
        segmentation=segmentation,
        xplat_sensitivity=config.xplat_sensitivity,
        xplat_sensitivity_segmentations=config.xplat_sensitivity_segmentations,
        refused_panels=list(refused),
    )
    annotation_names = [b.panel_name for b in bundles if b.purpose == "annotation"]
    status: Literal["ok", "refused"] = "ok" if annotation_names else "refused"
    reasons = [f"{name}: {reason}" for name, reason in sorted(refused.items())]
    if status == "refused":
        bundles = []
    required = RequiredBundles(
        pair_id=pair_id,
        segmentation=segmentation,
        species=species,
        panel_mode=mode,
        status=status,
        reasons=reasons,
        bundles=bundles,
        n_required=len(bundles),
    )
    report = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "pair_id": pair_id,
        "segmentation": segmentation,
        "species": species,
        "status": status,
        "reasons": reasons,
        "panel_mode_requested": panel_config.panel_mode,
        "panel_mode": mode,
        "refused_platforms": sorted(
            _platform_name(panel) for panel in declared if panel.status == "refused"
        ),
        "pair_jaccard": None if pair_jaccard is None else round(pair_jaccard, 6),
        "min_counts": min_counts,
        "min_counts_source": min_counts_source,
        "gene_id_fallback": _fallback_record(sources),
        "gene_id_sources": sources.describe(),
        "declared_panels": {
            panel.sample_id or _platform_name(panel): _declared_report(panel)
            for panel in declared
        },
        "cross_platform_id_matches": _cross_platform_id_matches(declared),
        "annotation_panels": {
            name: {
                "file": item.file_name,
                "kind": item.panel.kind,
                "panel_hash": item.panel.panel_hash,
                "n_genes": item.panel.n_genes,
                "platforms": item.panel.platforms,
                "panel_family": (
                    None
                    if item.panel.panel_family is None
                    else item.panel.panel_family.model_dump(mode="json")
                ),
                "refused": refused.get(name),
                "family_validation": _family_validation(item.panel, validated),
            }
            for name, item in panel_file_map.items()
        },
        "validated_panels": validated.describe(),
        "intersection_only_in": _only_in(declared) if len(declared) == 2 else {},
        "xplat_broad_only": (
            mode == "per_platform"
            and built["intersection"].n_genes
            < panel_config.xplat_min_intersection_genes
        ),
        "setc": None if setc_report is None else setc_report.model_dump(mode="json"),
        "setc_skipped": setc_skipped,
        "n_required_bundles": required.n_required,
    }
    return _write_panel_outputs(Path(output_dir), report, panel_file_map, required)


def _validated_families(
    config: AnnotationConfig,
    species: Species,
    validated: ValidatedPanelTable | None,
    known_families: Sequence[KnownPanelFamily] | None,
) -> tuple[ValidatedPanelTable, Sequence[KnownPanelFamily]]:
    """The validated table and the families a run's panels may inherit."""
    from merxen.annotation.diagnostics import load_validated_panels

    if validated is None:
        validated = load_validated_panels(config.panel.validated_panels_path)
    if known_families is None:
        known_families = validated.known_families(species)
    return validated, known_families


def _family_validation(
    panel: AnnotationPanel, validated: ValidatedPanelTable
) -> dict[str, Any]:
    """``family_validation_preview`` of one annotation panel (report item 1)."""
    from merxen.annotation.diagnostics import family_validation_preview

    return family_validation_preview(panel.panel_family, panel.species, validated)


def _fallback_record(sources: GeneIdSources) -> dict[str, Any] | None:
    """The run species' gene table, as M2's ``gene_id_fallback`` record."""
    table = sources.run_table
    if table is None:
        return None
    return {
        "path": table.path,
        "query_species": table.species,
        "n_symbols": table.n_symbols,
        "n_gene_ids": len(table.gene_ids),
    }


def _only_in(panels: Sequence[DeclaredPanel]) -> dict[str, list[str]]:
    first, second = panels
    only_first = sorted(set(first.ensembl_ids) - set(second.ensembl_ids))
    only_second = sorted(set(second.ensembl_ids) - set(first.ensembl_ids))
    return {
        _platform_name(first): [first.symbol_for(gene_id) for gene_id in only_first],
        _platform_name(second): [second.symbol_for(gene_id) for gene_id in only_second],
    }


def _write_panel_outputs(
    output_dir: Path,
    report: dict[str, Any],
    panel_file_map: Mapping[str, PanelFile],
    required: RequiredBundles,
) -> PanelComputation:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, item in panel_file_map.items():
        paths[name] = item.panel.write(output_dir / item.file_name)
    report_path = output_dir / PANEL_REPORT_FILE
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths["report"] = report_path
    required_path = output_dir / REQUIRED_BUNDLES_FILE
    required_path.write_text(
        json.dumps(required.model_dump(mode="json"), indent=2) + "\n"
    )
    paths["required"] = required_path
    logger.info(
        "Panel mode %s; %d annotation panel(s); %d required bundle(s) -> %s",
        report["panel_mode"],
        len(panel_file_map),
        required.n_required,
        output_dir,
    )
    return PanelComputation(
        report=report, panels=dict(panel_file_map), required=required, paths=paths
    )


def panel_from_gene_list(
    path: Path | str,
    species: Species,
    *,
    output_dir: Path | str,
    platform: str | None = None,
    config: AnnotationConfig | None = None,
    pair_id: str | None = None,
    segmentation: str | None = None,
    known_families: Sequence[KnownPanelFamily] | None = None,
    setc_families: Sequence[CuratedSetcFamily] | None = None,
    validated: ValidatedPanelTable | None = None,
) -> PanelComputation:
    """Build the panel files from a gene list (``annotation_panel_genes_path``).

    For prepare-only runs without prepared data: the list is the declared
    panel (any ``read_panel_file`` format), resolved with the configured
    fallback table.

    Args:
        path: The gene list.
        species: Run species.
        output_dir: Where the panel files go.
        platform: Platform the panel is for, if known.
        config: Annotation config; ``None`` uses the species defaults.
        pair_id: Pair id, for the record.
        segmentation: Segmentation, for the record.
        known_families: Families to inherit from (default: the rows of the
            validated table).
        setc_families: Curated set-c families (default: the packaged ones);
            a human gene list of such a family also gets its set c.
        validated: The validated families (default: the configured or
            packaged ``validated_panels.csv``).

    Returns:
        What was written.
    """
    config = config or AnnotationConfig(species=species)
    validated, known_families = _validated_families(
        config, species, validated, known_families
    )
    panel_config = config.panel
    registry = ControlRegistry.from_config(panel_config)
    if setc_families is None:
        setc_families = load_curated_setc_families(species)
    raw = read_panel_file(path)
    sources = gene_id_sources(panel_config, species)
    platform_value = platform.upper() if platform else None
    declared = declared_panel(
        raw,
        species=species,
        platform=platform_value,
        registry=registry,
        sources=sources,
        rules=ResolutionRules.from_config(panel_config),
    )
    panel = platform_panel(
        declared,
        kind="gene_list",
        name="gene_list",
        panel_mode="single_sample",
        known_families=known_families,
        family_min_jaccard=panel_config.family_min_jaccard,
    )
    refused: dict[str, str] = {}
    if declared.status == "refused":
        refused["gene_list"] = f"declared panel refused ({declared.refusal_text()})"
    elif panel.n_genes < panel_config.min_mapped_genes:
        refused["gene_list"] = f"{panel.n_genes} resolved genes < min_mapped_genes"
    panel_file_map = {"gene_list": PanelFile(panel=panel, file_name=PANEL_GENES_FILE)}
    # A gene list of the seeded set-a family also gets its curated set c, so a
    # prepare-only run builds every bundle a map_first pair of the family
    # needs (the label-free rule needs paired data and is not applied).
    setc_report: SetcReport | None = None
    curated = (
        curated_setc_family(panel, setc_families)
        if species == "human" and not refused
        else None
    )
    if curated is not None:
        setc, setc_report = setc_panel(
            panel,
            None,
            curated=curated,
            max_abs_log2_deviation=panel_config.setc_max_abs_log2_deviation,
            log2_pseudocount=panel_config.setc_log2_pseudocount,
            known_families=known_families,
            family_min_jaccard=panel_config.family_min_jaccard,
        )
        panel_file_map["setc"] = PanelFile(panel=setc, file_name=PANEL_GENES_SETC_FILE)
    bundles = required_bundles(
        panel_file_map,
        references=config.references,
        species=species,
        panel_mode="single_sample",
        segmentation=segmentation,
        xplat_sensitivity=config.xplat_sensitivity,
        xplat_sensitivity_segmentations=config.xplat_sensitivity_segmentations,
        refused_panels=list(refused),
    )
    status: Literal["ok", "refused"] = "refused" if refused else "ok"
    reasons = [f"{name}: {reason}" for name, reason in refused.items()]
    required = RequiredBundles(
        pair_id=pair_id,
        segmentation=segmentation,
        species=species,
        panel_mode="single_sample",
        status=status,
        reasons=reasons,
        bundles=[] if refused else bundles,
        n_required=0 if refused else len(bundles),
    )
    report = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "pair_id": pair_id,
        "segmentation": segmentation,
        "species": species,
        "status": status,
        "reasons": reasons,
        "panel_mode_requested": panel_config.panel_mode,
        "panel_mode": "single_sample",
        "pair_jaccard": None,
        "gene_id_fallback": _fallback_record(sources),
        "gene_id_sources": sources.describe(),
        "declared_panels": {"gene_list": _declared_report(declared)},
        "cross_platform_id_matches": [],
        "annotation_panels": {
            name: {
                "file": item.file_name,
                "kind": item.panel.kind,
                "panel_hash": item.panel.panel_hash,
                "n_genes": item.panel.n_genes,
                "platforms": item.panel.platforms,
                "panel_family": (
                    None
                    if item.panel.panel_family is None
                    else item.panel.panel_family.model_dump(mode="json")
                ),
                "refused": refused.get("gene_list"),
                "family_validation": _family_validation(item.panel, validated),
            }
            for name, item in panel_file_map.items()
        },
        "validated_panels": validated.describe(),
        "setc": None if setc_report is None else setc_report.model_dump(mode="json"),
        "setc_skipped": None
        if setc_report is not None
        else "no paired prepared data and no curated set-c family",
        "n_required_bundles": required.n_required,
    }
    return _write_panel_outputs(Path(output_dir), report, panel_file_map, required)
