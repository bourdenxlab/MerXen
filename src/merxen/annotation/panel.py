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
the shared registry ``merxen.control_features``; IDs come from the table's
own Ensembl column, then the pair's other platform (same symbol), then the
configured local fallback table (M0e, ``annotation_gene_id_fallback_csv``).
Aliases, overrides, the exact-case species test and trust states are M3b.

Panel modes (plan §3.2, §8.5): two platform panels with Jaccard >= 0.9 form an
``intersection`` panel (human: set a), and same-panel human pairs also get set
c = set a minus the genes whose label-free pseudobulk platform ratio deviates
by more than 2 log2 units from the pair median, over table cells inside the
shared tissue mask when one is available. Other pairs are ``per_platform``:
each platform's own panel plus their intersection for cross-platform
statistics. Unpaired samples use their own panel (``single_sample``).
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
from merxen.annotation.schema import PanelMode, ReferenceRole
from merxen.annotation.vocab import Species
from merxen.control_features import has_control_token, matches_control_name_pattern
from merxen.gene_ids import is_ensembl_gene_id

if TYPE_CHECKING:
    from merxen.analysis.mapmycells import GeneIdFallbackTable

logger = logging.getLogger(__name__)

PANEL_SCHEMA_VERSION: Final = 1
PANEL_GENES_FILE: Final = "panel_genes.json"
PANEL_GENES_SETC_FILE: Final = "panel_genes_setc.json"
PANEL_GENES_INTERSECTION_FILE: Final = "panel_genes_intersection.json"
PANEL_REPORT_FILE: Final = "panel_report.json"
REQUIRED_BUNDLES_FILE: Final = "required_bundles.json"
PREPARED_MANIFEST_FILE: Final = "manifest.json"

# ``var`` columns that may carry native Ensembl IDs, in the order of plan §8.4.
NATIVE_ID_COLUMNS: Final[tuple[str, ...]] = (
    "ensembl_id",
    "gene_ids",
    "gene_id",
    "feature_id",
)
SYMBOL_COLUMNS: Final[tuple[str, ...]] = (
    "gene",
    "gene_symbol",
    "gene_name",
    "feature_name",
    "symbol",
)
FEATURE_TYPE_COLUMNS: Final[tuple[str, ...]] = ("feature_types", "feature_type")
GENE_FEATURE_TYPE: Final = "Gene Expression"
# Xenium ``gene_panel.json`` target descriptors and the feature types they
# stand for (the cell-feature-matrix names).
XENIUM_PANEL_DESCRIPTOR_TYPES: Final[dict[str, str]] = {
    "gene": GENE_FEATURE_TYPE,
    "negative_control": "Negative Control Probe",
}
SPECIES_ID_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "human": re.compile(r"^ENSG\d+$"),
    "mouse": re.compile(r"^ENSMUSG\d+$"),
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

ResolutionSource = Literal["native", "pair_lookup", "fallback_table"]
PanelKind = Literal["intersection", "setc", "platform", "single_sample", "gene_list"]
PanelSourceKind = Literal[
    "xenium_gene_panel_json",
    "merscope_codebook",
    "gene_table",
    "h5ad_var",
    "prepared_h5ad_var",
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


def strip_version(gene_id: str) -> str:
    """Return an Ensembl ID without its ``.N`` version suffix.

    Args:
        gene_id: Candidate identifier.

    Returns:
        The stripped, whitespace-trimmed identifier.
    """
    return re.sub(r"\.\d+$", "", str(gene_id).strip())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


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
    """The family a panel belongs to (plan §8.1; trust inheritance is M3b).

    Attributes:
        family_id: Family id; the panel's own id unless it inherited one.
        basis: ``"own"`` or ``"inherited"`` (same species and platforms,
            Jaccard >= ``family_min_jaccard``, all root markers present).
        reference_panel_hash: Hash of the family's panel.
        jaccard: Jaccard of this panel with the family's panel.
    """

    family_id: str
    basis: Literal["own", "inherited"]
    reference_panel_hash: str
    jaccard: float


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
            ``fallback_table``).
        symbol_to_id: Resolved ID per non-control feature symbol.
        n_features_in: Features in the declared list.
        controls_removed: Removed control features per reason token.
        kept_despite_control_token: Features kept because they carry a native
            Ensembl ID although their name contains a control token.
        unresolved: Reason per non-control feature without an ID.
        merged_duplicates: Features merged into one ID (ID -> symbols).
        other_species_ids: Native IDs with another species' prefix.
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
    n_features_in: int
    controls_removed: dict[str, list[str]]
    kept_despite_control_token: list[str] = Field(default_factory=list)
    unresolved: dict[str, str]
    merged_duplicates: dict[str, list[str]] = Field(default_factory=dict)
    other_species_ids: list[str] = Field(default_factory=list)

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
            ``"sample"`` or ``"gene_list"``.
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
        parent_panel_hash: For set c, the hash of set a.
        excluded_ids: For set c, the set-a IDs it drops.
        panel_family: The panel's family.
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

        Part of ``build_hash`` (plan §8.4: the resolution table's sha256), so
        a bundle whose panel stub carries other symbols is a new build.

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


class RequiredBundle(_PanelModel):
    """One (reference, panel) bundle a pair x segmentation needs.

    Attributes:
        reference_id: Store id.
        role: Reference role.
        species: Species.
        purpose: Why the bundle is needed.
        panel_name: Annotation panel name (``None`` if panel-independent).
        panel_hash: Panel hash (``None`` if panel-independent).
        panel_file: ``panel_genes*.json`` file name in the ANNOTATE_PANEL
            output directory (``None`` if panel-independent).
    """

    reference_id: str
    role: ReferenceRole
    species: Species
    purpose: BundlePurpose
    panel_name: str | None = None
    panel_hash: str | None = None
    panel_file: str | None = None

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

    Wraps the shared registry ``merxen.control_features``. A feature type,
    when the source has one, decides alone (as for ProSeg transcripts);
    otherwise a configured extra pattern or the platform's anchored control
    name pattern marks a control; last, the ``CONTROL_TOKENS`` substring rule,
    which never removes a feature that carries a native Ensembl ID.

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
    ) -> str | None:
        """Return why a feature is a control, or ``None`` for a gene.

        Args:
            name: Feature name.
            platform: ``"MERSCOPE"`` or ``"XENIUM"`` (selects the name
                pattern); ``None`` checks both.
            feature_type: The source's feature type, ``""`` if none.
            has_native_id: Whether the feature carries a native Ensembl ID.

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
        if has_control_token(name) and not has_native_id:
            return "control_token"
        return None


def _token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


# --------------------------------------------------------------------------
# Reading declared panels


@dataclass(frozen=True)
class RawPanel:
    """A declared feature list before control removal and ID resolution.

    Attributes:
        features: One row per feature with columns ``name``, ``symbol``,
            ``native_id`` (unversioned, ``""`` if none) and ``feature_type``
            (``""`` if the source has none), in source order.
        source: Where it was read from.
    """

    features: pd.DataFrame
    source: PanelSource


def _raw_from_columns(
    names: Sequence[Any],
    *,
    symbols: Sequence[Any] | None,
    native_ids: Sequence[Any] | None,
    feature_types: Sequence[Any] | None,
    source: PanelSource,
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
    ids = (
        [strip_version(_clean_text(value)) for value in native_ids]
        if native_ids is not None
        else [""] * len(cleaned_names)
    )
    ids = [value if is_ensembl_gene_id(value) else "" for value in ids]
    types = (
        [_clean_text(value) for value in feature_types]
        if feature_types is not None
        else [""] * len(cleaned_names)
    )
    frame = pd.DataFrame(
        {
            "name": cleaned_names,
            "symbol": cleaned_symbols,
            "native_id": ids,
            "feature_type": types,
        }
    )
    frame = frame[frame["name"] != ""].reset_index(drop=True)
    return RawPanel(features=frame, source=source)


def raw_panel_from_var(
    var: pd.DataFrame,
    *,
    source: PanelSource,
) -> RawPanel:
    """Return the declared features of an AnnData ``var`` table.

    Args:
        var: ``var`` with feature names as index; optional symbol, ID and
            feature-type columns (``SYMBOL_COLUMNS``, ``NATIVE_ID_COLUMNS``,
            ``FEATURE_TYPE_COLUMNS``). An index of Ensembl IDs is used as the
            ID column when no ID column exists.

    Returns:
        The raw panel.
    """
    index = [str(name) for name in var.index]
    symbol_column = next((c for c in SYMBOL_COLUMNS if c in var.columns), None)
    id_column = _first_id_column(var)
    type_column = next((c for c in FEATURE_TYPE_COLUMNS if c in var.columns), None)
    symbols = var[symbol_column].tolist() if symbol_column is not None else None
    names: Sequence[Any] = index
    native_ids: Sequence[Any] | None = None
    if id_column is not None:
        native_ids = var[id_column].tolist()
    elif any(is_ensembl_gene_id(strip_version(name)) for name in index):
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
        feature_types=var[type_column].tolist() if type_column is not None else None,
        source=source,
    )


def _first_id_column(table: pd.DataFrame) -> str | None:
    """Return the first ID column that holds any Ensembl gene ID."""
    for column in NATIVE_ID_COLUMNS:
        if column in table.columns and any(
            is_ensembl_gene_id(strip_version(_clean_text(value)))
            for value in table[column]
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
        table = pd.read_csv(
            file_path,
            sep="\t" if is_tsv else ",",
            comment="#",
            dtype=str,
            keep_default_na=False,
        )
    else:
        raise ValueError(f"unsupported panel file type: {file_path}")
    columns = {str(column).strip(): column for column in table.columns}
    if "barcodeType" in columns and "name" in columns:
        return _raw_from_columns(
            table[columns["name"]].tolist(),
            symbols=None,
            native_ids=(table[columns["id"]].tolist() if "id" in columns else None),
            feature_types=None,
            source=PanelSource(
                kind="merscope_codebook", path=str(file_path), sha256=digest
            ),
        )
    symbol_column = next((c for c in (*SYMBOL_COLUMNS, "name") if c in columns), None)
    id_column = _first_id_column(table.rename(columns=lambda c: str(c).strip()))
    if symbol_column is None and id_column is None:
        raise ValueError(
            f"panel table {file_path} needs a symbol column ({SYMBOL_COLUMNS}) or "
            f"an Ensembl ID column ({NATIVE_ID_COLUMNS}); found {list(columns)}"
        )
    type_column = next((c for c in FEATURE_TYPE_COLUMNS if c in columns), None)
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
        feature_types=(
            table[columns[type_column]].tolist() if type_column is not None else None
        ),
        source=PanelSource(kind="gene_table", path=str(file_path), sha256=digest),
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


def declared_panel(
    raw: RawPanel,
    *,
    species: Species,
    platform: str | None,
    sample_id: str | None = None,
    registry: ControlRegistry | None = None,
    pair_lookup: Mapping[str, str] | None = None,
    fallback: GeneIdFallbackTable | None = None,
) -> DeclaredPanel:
    """Remove controls from a declared list and resolve its gene IDs.

    Resolution order (first hit wins; plan §8.4, M2 part): the feature's
    native Ensembl ID, the pair lookup (same symbol on the other platform),
    the configured local fallback table (a unique candidate). Features that
    resolve to one ID are merged.

    Args:
        raw: The declared features.
        species: Run species.
        platform: The sample's platform (selects control name patterns).
        sample_id: The sample, for the record.
        registry: Control rules; the default keeps ``Gene Expression`` only.
        pair_lookup: Symbol -> ID from the pair (``pair_symbol_lookup``).
        fallback: The M0e fallback table (``load_gene_id_fallback_table``).

    Returns:
        The declared panel.
    """
    registry = registry or ControlRegistry()
    pattern = SPECIES_ID_PATTERNS[species]
    lookup = dict(pair_lookup or {})
    controls: dict[str, list[str]] = {}
    kept_despite_token: list[str] = []
    unresolved: dict[str, str] = {}
    other_species: list[str] = []
    symbols_by_id: dict[str, list[str]] = {}
    sources_by_id: dict[str, ResolutionSource] = {}
    symbol_to_id: dict[str, str] = {}
    for row in raw.features.itertuples(index=False):
        name, symbol = str(row.name), str(row.symbol)
        native_id, feature_type = str(row.native_id), str(row.feature_type)
        reason = registry.control_reason(
            name,
            platform=platform,
            feature_type=feature_type,
            has_native_id=bool(native_id),
        )
        if reason is not None:
            controls.setdefault(reason, []).append(name)
            continue
        if native_id and has_control_token(name):
            kept_despite_token.append(name)
        gene_id, source, failure = _resolve_feature(
            symbol,
            native_id,
            pattern=pattern,
            pair_lookup=lookup,
            fallback=fallback,
        )
        if gene_id is None:
            unresolved[symbol] = failure
            if failure == "other_species_id":
                other_species.append(native_id)
            continue
        symbol_to_id[symbol] = gene_id
        symbols_by_id.setdefault(gene_id, []).append(symbol)
        if source is not None and gene_id not in sources_by_id:
            sources_by_id[gene_id] = source
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
    return panel


def _resolve_feature(
    symbol: str,
    native_id: str,
    *,
    pattern: re.Pattern[str],
    pair_lookup: Mapping[str, str],
    fallback: GeneIdFallbackTable | None,
) -> tuple[str | None, ResolutionSource | None, str]:
    if native_id:
        if pattern.fullmatch(native_id):
            return native_id, "native", ""
        return None, None, "other_species_id"
    if symbol in pair_lookup:
        return pair_lookup[symbol], "pair_lookup", ""
    if fallback is None:
        return None, None, "no_fallback_table"
    candidates = [
        candidate
        for candidate in fallback.candidate_ids(symbol)
        if pattern.fullmatch(candidate)
    ]
    if not candidates:
        return None, None, "not_in_fallback_table"
    if len(candidates) > 1:
        return None, None, "ambiguous_in_fallback_table"
    return candidates[0], "fallback_table", ""


# --------------------------------------------------------------------------
# Panel families and modes


@dataclass(frozen=True)
class KnownPanelFamily:
    """A family record trust can be inherited from (``validated_panels.csv``, M3b).

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

    A panel inherits a known family when it has the same species and
    platforms, Jaccard >= ``min_jaccard`` with the family panel and contains
    all its root markers; the most similar such family wins. Otherwise the
    panel is its own family, ``<species>_<platforms>_<hash prefix>``.

    Args:
        ensembl_ids: The panel's IDs.
        species: Species.
        platforms: Platforms the panel serves.
        known_families: Families to inherit from (none until M3b ships
            ``validated_panels.csv``).
        min_jaccard: ``family_min_jaccard``.

    Returns:
        The family.
    """
    ids = set(ensembl_ids)
    own_hash = compute_panel_hash(sorted(ids))
    platform_set = frozenset(platform.upper() for platform in platforms)
    best: tuple[float, KnownPanelFamily] | None = None
    for family in known_families:
        if family.species != species or family.platforms != platform_set:
            continue
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
    return ("intersection" if score >= min_jaccard else "per_platform"), score


def _platform_name(panel: DeclaredPanel) -> str:
    return (panel.platform or panel.sample_id or "sample").lower()


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
        panel_family=panel_family(
            shared,
            species=species,
            platforms=platforms,
            known_families=known_families,
            min_jaccard=family_min_jaccard,
        ),
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


def _column_ids_for_var(
    var: pd.DataFrame,
    *,
    declared: DeclaredPanel,
    platform: str | None,
    registry: ControlRegistry,
) -> tuple[np.ndarray, list[str]]:
    """Return a gene-column mask and the resolved ID of each ``var`` feature."""
    raw = raw_panel_from_var(var, source=PanelSource(kind="h5ad_var"))
    if len(raw.features) != len(var):
        raise ValueError("var holds features without a name")
    pattern = SPECIES_ID_PATTERNS[declared.species]
    is_gene = np.zeros(len(var), dtype=bool)
    ids: list[str] = []
    for position, row in enumerate(raw.features.itertuples(index=False)):
        reason = registry.control_reason(
            str(row.name),
            platform=platform,
            feature_type=str(row.feature_type),
            has_native_id=bool(row.native_id),
        )
        is_gene[position] = reason is None
        native = str(row.native_id)
        if reason is None and native and pattern.fullmatch(native):
            ids.append(native)
        elif reason is None:
            ids.append(declared.symbol_to_id.get(str(row.symbol), ""))
        else:
            ids.append("")
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
        is_gene, column_ids = _column_ids_for_var(
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


class SetcReport(_PanelModel):
    """How set c was derived (``panel_report.json``; plan §3.2, D-A7).

    Attributes:
        rule: The rule applied.
        max_abs_log2_deviation: Threshold on the centred log2 ratio.
        log2_pseudocount: Pseudocount added to both means.
        pair_median_log2_ratio: Median ``log2(mean X / mean M)`` over set a.
        mask_applied: Whether the shared tissue mask restricted the cells.
        mask_reason: Why the mask was not applied, when it was not.
        pseudobulk: Cell counts per platform (means omitted).
        n_set_a: Set-a genes.
        n_setc: Set-c genes.
        excluded: Centred log2 ratio of each excluded gene, by ID.
        excluded_symbols: Symbol of each excluded gene, by ID.
        centred_log2_ratio: Centred log2 ratio of every set-a gene, by ID.
    """

    rule: str
    max_abs_log2_deviation: float
    log2_pseudocount: float
    pair_median_log2_ratio: float
    mask_applied: bool
    mask_reason: str | None = None
    pseudobulk: dict[str, dict[str, int | str | bool]]
    n_set_a: int
    n_setc: int
    excluded: dict[str, float]
    excluded_symbols: dict[str, str]
    centred_log2_ratio: dict[str, float]


SETC_RULE: Final = (
    "|log2((mean_X + c) / (mean_M + c)) - median over set a| > threshold, "
    "label-free pseudobulk over table cells (inside the shared tissue mask "
    "when available)"
)


def setc_panel(
    set_a: AnnotationPanel,
    pseudobulk: Mapping[str, PlatformPseudobulk],
    *,
    max_abs_log2_deviation: float = 2.0,
    log2_pseudocount: float = 1e-3,
    mask_reason: str | None = None,
    known_families: Sequence[KnownPanelFamily] = (),
    family_min_jaccard: float = 0.95,
) -> tuple[AnnotationPanel, SetcReport]:
    """Return set c: set a minus the pair's platform-deviant genes.

    Rule (plan §3.2, D-A7): ``r = log2((mean_X + c) / (mean_M + c))`` per set-a
    gene from label-free pseudobulk over table cells; a gene is deviant when
    ``|r - median(r)| > max_abs_log2_deviation``. The E5 list of 32 genes
    (label-aware, in confident oligodendrocytes) is the expected result [M].

    Args:
        set_a: The intersection panel.
        pseudobulk: Pseudobulk per platform (``"MERSCOPE"``, ``"XENIUM"``).
        max_abs_log2_deviation: Threshold (2).
        log2_pseudocount: ``c`` (1e-3, as the E5 ratios).
        mask_reason: Why no mask was applied, for the report.
        known_families: Families to inherit from.
        family_min_jaccard: Family Jaccard threshold.

    Returns:
        The set-c panel and its report.

    Raises:
        ValueError: If a platform's pseudobulk is missing or empty.
    """
    for platform in ("MERSCOPE", "XENIUM"):
        if platform not in pseudobulk:
            raise ValueError(f"set c needs a {platform} pseudobulk")
        if pseudobulk[platform].n_cells_used == 0:
            raise ValueError(f"set c: no {platform} table cells were used")
    xenium = pseudobulk["XENIUM"].mean_counts
    merscope = pseudobulk["MERSCOPE"].mean_counts
    ids = list(set_a.ensembl_ids)
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
    centred = ratios - median
    deviant = np.abs(centred) > max_abs_log2_deviation
    kept = [gene_id for gene_id, drop in zip(ids, deviant, strict=True) if not drop]
    excluded = [gene_id for gene_id, drop in zip(ids, deviant, strict=True) if drop]
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
    report = SetcReport(
        rule=SETC_RULE,
        max_abs_log2_deviation=max_abs_log2_deviation,
        log2_pseudocount=log2_pseudocount,
        pair_median_log2_ratio=round(median, 6),
        mask_applied=all(item.mask_applied for item in pseudobulk.values()),
        mask_reason=mask_reason,
        pseudobulk={
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
        excluded={
            gene_id: round(float(value), 6)
            for gene_id, value, drop in zip(ids, centred, deviant, strict=True)
            if drop
        },
        excluded_symbols={gene_id: symbol_of[gene_id] for gene_id in excluded},
        centred_log2_ratio={
            gene_id: round(float(value), 6)
            for gene_id, value in zip(ids, centred, strict=True)
        },
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
        Distinct bundles in a stable order.
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
    seen: set[tuple[str, str, str | None]] = set()

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
        )
        if bundle.key not in seen:
            seen.add(bundle.key)
            bundles.append(bundle)

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
    return {
        "sample_id": panel.sample_id,
        "platform": panel.platform,
        "source": panel.source.model_dump(mode="json"),
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
        "resolved_by_fallback_table": {
            symbol: gene_id
            for symbol, gene_id in panel.symbol_to_id.items()
            if panel.id_sources.get(gene_id) == "fallback_table"
        },
        "unresolved": panel.unresolved,
        "merged_duplicates": panel.merged_duplicates,
        "other_species_ids": panel.other_species_ids,
    }


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
    known_families: Sequence[KnownPanelFamily] = (),
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
        known_families: Families to inherit from.

    Returns:
        What was written.
    """
    config = config or AnnotationConfig(species=species)
    if config.species != species:
        raise ValueError(f"annotation config is for {config.species}, not {species}")
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
            raw = raw_panel_from_var(
                read_h5ad_var(sample.h5ad_path),
                source=PanelSource(
                    kind="prepared_h5ad_var",
                    path=str(sample.h5ad_path),
                ),
            )
        raws.append((sample, raw))
    lookup = pair_symbol_lookup([raw for _, raw in raws], species)
    fallback = load_fallback_table(panel_config.gene_id_fallback_csv, species)
    declared = [
        declared_panel(
            raw,
            species=species,
            platform=sample.platform,
            sample_id=sample.sample_id,
            registry=registry,
            pair_lookup=lookup,
            fallback=fallback,
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
    if mode == "intersection" and species == "human":
        mask, mask_reason = _mask_for_samples(shared_mask, samples)
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
            max_abs_log2_deviation=panel_config.setc_max_abs_log2_deviation,
            log2_pseudocount=panel_config.setc_log2_pseudocount,
            mask_reason=mask_reason,
            known_families=known_families,
            family_min_jaccard=family_min_jaccard,
        )
    elif mode == "intersection":
        setc_skipped = "set c is a human cross-platform sensitivity panel"
    refused: dict[str, str] = {}
    for name, annotation_panel in built.items():
        if annotation_panel.n_genes < panel_config.min_mapped_genes:
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
        "pair_jaccard": None if pair_jaccard is None else round(pair_jaccard, 6),
        "min_counts": min_counts,
        "min_counts_source": min_counts_source,
        "gene_id_fallback": None if fallback is None else fallback.describe(),
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
            }
            for name, item in panel_file_map.items()
        },
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
    known_families: Sequence[KnownPanelFamily] = (),
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
        known_families: Families to inherit from.

    Returns:
        What was written.
    """
    config = config or AnnotationConfig(species=species)
    panel_config = config.panel
    registry = ControlRegistry.from_config(panel_config)
    raw = read_panel_file(path)
    fallback = load_fallback_table(panel_config.gene_id_fallback_csv, species)
    platform_value = platform.upper() if platform else None
    declared = declared_panel(
        raw,
        species=species,
        platform=platform_value,
        registry=registry,
        fallback=fallback,
    )
    panel = platform_panel(
        declared,
        kind="gene_list",
        name="gene_list",
        panel_mode="single_sample",
        known_families=known_families,
        family_min_jaccard=panel_config.family_min_jaccard,
    )
    refused = (
        {"gene_list": f"{panel.n_genes} resolved genes < min_mapped_genes"}
        if panel.n_genes < panel_config.min_mapped_genes
        else {}
    )
    panel_file_map = {"gene_list": PanelFile(panel=panel, file_name=PANEL_GENES_FILE)}
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
        "gene_id_fallback": None if fallback is None else fallback.describe(),
        "declared_panels": {"gene_list": _declared_report(declared)},
        "cross_platform_id_matches": [],
        "annotation_panels": {
            "gene_list": {
                "file": PANEL_GENES_FILE,
                "kind": "gene_list",
                "panel_hash": panel.panel_hash,
                "n_genes": panel.n_genes,
                "platforms": panel.platforms,
                "panel_family": (
                    None
                    if panel.panel_family is None
                    else panel.panel_family.model_dump(mode="json")
                ),
                "refused": refused.get("gene_list"),
            }
        },
        "setc": None,
        "setc_skipped": "no paired prepared data",
        "n_required_bundles": required.n_required,
    }
    return _write_panel_outputs(Path(output_dir), report, panel_file_map, required)
