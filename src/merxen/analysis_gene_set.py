"""Analysis gene sets: the genes an analysis uses on a panel (Atera plan AT-8 §1b).

An analysis gene set is a fixed list of Ensembl gene IDs derived from a
panel by a named recipe. It depends only on the panel and on pinned inputs,
never on one section's data, so every section of a panel gets the same set
and the same hash. Reference-based annotation keys its bundles on that hash
(AT-11), so a data-driven set (HVGs, markers) would give each section its
own reference preparation.

Recipes (``Recipe``):

* ``all_panel_genes``: every gene of the panel (the identity). This is the
  default for every platform; for ATERA it is the user's decision of
  2026-11-07 (RCA on Atera uses all 18,049 panel genes).
* ``intersect_public_panel:<key>``: the panel genes that are also in a
  pinned public panel list (``merxen.annotation.public_panels``).
* ``explicit:<path>``: the panel genes listed in a file of Ensembl gene IDs.

The panel is read with the resolver of ``ANNOTATE_PANEL``
(``merxen.annotation.panel.declared_panel``): controls are removed and every
feature gets an unversioned Ensembl gene ID. Recipes join on those IDs and
**never on gene symbols**: the Atera panel has 17 genes whose symbol is
itself an Ensembl ID, and a symbol can name different genes in two releases.
A requested gene the panel lacks is listed in ``missing_genes``; a panel gene
that only shares its symbol with it is reported in
``missing_symbol_matches`` and is not selected.

``gene_set_hash`` is ``annotation.panel.compute_panel_hash`` of the IDs (the
sha256 of the sorted, distinct IDs), so ``merxen annotation-panel
--panel-genes-path analysis_gene_set.csv`` gives a panel whose
``panel_hash`` equals it. The writer puts ``analysis_gene_set.json`` (this
record), ``analysis_gene_set.csv`` (``gene_symbol``, ``gene_id``: the layout
``--panel-genes-path`` reads) and ``analysis_gene_set_missing.csv`` in one
directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from merxen.annotation.gene_ids import (
    NATIVE_ID_COLUMNS,
    SPECIES_ID_PATTERNS,
    SYMBOL_COLUMNS,
    clean_native_value,
    clean_text,
)
from merxen.annotation.panel import compute_panel_hash
from merxen.annotation.vocab import Species
from merxen.gene_sets import WHOLE_TRANSCRIPTOME_MIN_GENES, is_whole_transcriptome

if TYPE_CHECKING:
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.panel import RawPanel

logger = logging.getLogger(__name__)

ANALYSIS_GENE_SET_SCHEMA_VERSION: Final = 1
ANALYSIS_GENE_SET_JSON: Final = "analysis_gene_set.json"
ANALYSIS_GENE_SET_CSV: Final = "analysis_gene_set.csv"
ANALYSIS_GENE_SET_MISSING_CSV: Final = "analysis_gene_set_missing.csv"
# The columns of a public panel gene list (``annotation-panel-fetch``), which
# ``annotation-panel --panel-genes-path`` reads as a gene table.
CSV_COLUMNS: Final[tuple[str, str]] = ("gene_symbol", "gene_id")

RECIPE_ALL_PANEL_GENES: Final = "all_panel_genes"
RECIPE_INTERSECT_PUBLIC_PANEL: Final = "intersect_public_panel"
RECIPE_EXPLICIT: Final = "explicit"
# Every platform defaults to the identity. For ATERA this is the user's
# decision of 2026-11-07: RCA uses all 18,049 panel genes for accuracy.
DEFAULT_RECIPE: Final = RECIPE_ALL_PANEL_GENES
# The default table of a SpatialData store (the vendor segmentation's).
DEFAULT_TABLE_KEY: Final = "table_original"
# The packaged pinned Prime 5K lists (``annotation.sim_inputs``).
PACKAGED_PUBLIC_PANEL_ASSET: Final = "panels__xenium_prime"

RecipeKind = Literal["all_panel_genes", "intersect_public_panel", "explicit"]
SourceKind = Literal["spatialdata_table", "panel_file"]
InputRole = Literal["source_panel", "public_panel", "explicit_list"]


class GeneSetError(ValueError):
    """An analysis gene set cannot be built from its inputs."""


def default_recipe(platform: str | None = None) -> str:
    """Return the default recipe of a platform.

    Args:
        platform: ``ATERA``, ``XENIUM``, ``MERSCOPE`` or ``None``.

    Returns:
        ``all_panel_genes`` for every platform (ATERA: user decision of
        2026-11-07; the 300-gene panels: the identity).
    """
    del platform
    return DEFAULT_RECIPE


# --------------------------------------------------------------------------
# Recipes


@dataclass(frozen=True)
class Recipe:
    """A parsed recipe.

    Attributes:
        kind: The recipe kind.
        argument: The public panel key (``intersect_public_panel``), the
            list path (``explicit``) or ``None`` (``all_panel_genes``).
    """

    kind: RecipeKind
    argument: str | None = None

    @classmethod
    def parse(cls, text: str) -> Recipe:
        """Parse ``all_panel_genes``, ``intersect_public_panel:<key>`` or
        ``explicit:<path>``.

        Raises:
            GeneSetError: If the text names no known recipe or lacks its
                argument.
        """
        value = text.strip()
        kind, separator, argument = value.partition(":")
        kind, argument = kind.strip(), argument.strip()
        if kind == RECIPE_ALL_PANEL_GENES:
            if separator:
                raise GeneSetError(f"{RECIPE_ALL_PANEL_GENES} takes no argument")
            return cls(RECIPE_ALL_PANEL_GENES)
        if kind in (RECIPE_INTERSECT_PUBLIC_PANEL, RECIPE_EXPLICIT):
            if not argument:
                raise GeneSetError(f"recipe {kind} needs an argument: {kind}:<value>")
            return cls(kind, argument)  # type: ignore[arg-type]
        raise GeneSetError(
            f"unknown recipe {text!r}; expected {RECIPE_ALL_PANEL_GENES}, "
            f"{RECIPE_INTERSECT_PUBLIC_PANEL}:<key> or {RECIPE_EXPLICIT}:<path>"
        )

    @property
    def label(self) -> str:
        """Return the recipe as recorded: an explicit list by file name only."""
        if self.kind == RECIPE_ALL_PANEL_GENES:
            return RECIPE_ALL_PANEL_GENES
        if self.kind == RECIPE_EXPLICIT:
            return f"{RECIPE_EXPLICIT}:{Path(str(self.argument)).name}"
        return f"{self.kind}:{self.argument}"

    @property
    def default_name(self) -> str:
        """Return the default gene-set name (a file-safe form of the label)."""
        if self.kind == RECIPE_ALL_PANEL_GENES:
            return RECIPE_ALL_PANEL_GENES
        if self.kind == RECIPE_EXPLICIT:
            return f"{RECIPE_EXPLICIT}__{Path(str(self.argument)).stem}"
        return f"{self.kind}__{self.argument}"


# --------------------------------------------------------------------------
# Models


class _GeneSetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeneSetRow(_GeneSetModel):
    """One gene: its unversioned Ensembl gene ID and a display symbol.

    Attributes:
        gene_id: Unversioned Ensembl gene ID (the join key).
        gene_symbol: Symbol, for display only (``""`` when unknown).
    """

    gene_id: str
    gene_symbol: str = ""


class GeneSetInput(_GeneSetModel):
    """One input file of a gene set.

    Attributes:
        role: What the input is for.
        kind: Its format (``spatialdata_table``, ``h5ad_var``,
            ``gene_table``, ``xenium_gene_panel_json``, ``merscope_codebook``,
            ``public_panel_gene_list``, ``packaged_asset``, ``gene_id_list``).
        path: Where it was read.
        sha256: sha256 of the file; for a SpatialData table, of its declared
            features (``declared_features_sha256``).
        n_rows: Rows read.
        key: The table key or the public panel key, where one applies.
    """

    role: InputRole
    kind: str
    path: str
    sha256: str
    n_rows: int
    key: str | None = None


class SourcePanelRecord(_GeneSetModel):
    """The panel a gene set is derived from (after ``declared_panel``).

    Attributes:
        kind: ``spatialdata_table`` or ``panel_file``.
        platform: The panel's platform, if known.
        n_features: Features read (controls included).
        n_genes: Resolved non-control genes.
        panel_hash: ``compute_panel_hash`` of those genes.
        declared_features_sha256: Order-independent sha256 of the declared
            features (name, symbol, native ID value, feature type).
        controls_removed: Removed controls per reason (counts).
        unresolved: Non-control features without an ID (name -> reason).
        id_sources: Genes per ID source (``native``, ``fallback_table``, ...).
        whole_transcriptome: Whether the panel has more than
            ``WHOLE_TRANSCRIPTOME_MIN_GENES`` genes.
    """

    kind: SourceKind
    platform: str | None
    n_features: int
    n_genes: int
    panel_hash: str
    declared_features_sha256: str
    controls_removed: dict[str, int] = Field(default_factory=dict)
    unresolved: dict[str, str] = Field(default_factory=dict)
    id_sources: dict[str, int] = Field(default_factory=dict)
    whole_transcriptome: bool


def gene_set_csv_text(genes: Sequence[GeneSetRow]) -> str:
    """Return the CSV of a gene set (``gene_symbol``, ``gene_id``).

    A missing or placeholder symbol (``""``, ``nan``, ``none``) is written as
    the gene ID, because ``annotation-panel --panel-genes-path`` drops a row
    whose name column is empty.

    Args:
        genes: The genes, in the order to write.

    Returns:
        The CSV text (``\\n`` line ends).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for gene in genes:
        writer.writerow([clean_text(gene.gene_symbol) or gene.gene_id, gene.gene_id])
    return buffer.getvalue()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AnalysisGeneSet(_GeneSetModel):
    """A fixed analysis gene set of one panel (frozen).

    Attributes:
        schema_version: Record layout version.
        name: Gene-set name (default: from the recipe).
        version: Version of this set's definition, set by its author.
        species: Run species.
        platform: Platform of the panel, if known.
        recipe: The recipe as recorded (``Recipe.label``).
        recipe_kind: Its kind.
        id_join: Always ``ensembl_gene_id``: recipes never join on symbols.
        inputs: The files read, with their sha256.
        source_panel: The panel the set is derived from.
        genes: The genes, sorted by ID.
        n_genes: ``len(genes)``.
        n_requested: Genes the recipe asked for (``None`` for
            ``all_panel_genes``).
        missing_genes: Requested genes the panel lacks.
        missing_symbol_matches: Missing ID -> the panel gene that has the same
            symbol (reported, never selected).
        n_excluded_panel_genes: Panel genes not in the set.
        whole_transcriptome: Whether the set has more than
            ``WHOLE_TRANSCRIPTOME_MIN_GENES`` genes.
        whole_transcriptome_min_genes: The threshold used.
        gene_set_hash: ``compute_panel_hash`` of the IDs (the
            ``panel_hash`` ``annotation-panel`` gives the CSV).
        csv_file: The CSV's file name.
        csv_sha256: sha256 of the CSV (a function of ``genes`` only).
    """

    schema_version: Literal[1] = ANALYSIS_GENE_SET_SCHEMA_VERSION
    name: str
    version: int = Field(default=1, ge=1)
    species: Species
    platform: str | None
    recipe: str
    recipe_kind: RecipeKind
    id_join: Literal["ensembl_gene_id"] = "ensembl_gene_id"
    inputs: list[GeneSetInput]
    source_panel: SourcePanelRecord
    genes: list[GeneSetRow]
    n_genes: int
    n_requested: int | None
    missing_genes: list[GeneSetRow] = Field(default_factory=list)
    missing_symbol_matches: dict[str, str] = Field(default_factory=dict)
    n_excluded_panel_genes: int
    whole_transcriptome: bool
    whole_transcriptome_min_genes: int = WHOLE_TRANSCRIPTOME_MIN_GENES
    gene_set_hash: str
    csv_file: str = ANALYSIS_GENE_SET_CSV
    csv_sha256: str

    @model_validator(mode="after")
    def _check(self) -> AnalysisGeneSet:
        ids = [gene.gene_id for gene in self.genes]
        if ids != sorted(set(ids)):
            raise ValueError("genes must be sorted by gene_id with no repeats")
        pattern = SPECIES_ID_PATTERNS[self.species]
        wrong = [gene_id for gene_id in ids if not pattern.fullmatch(gene_id)]
        if wrong:
            raise ValueError(
                f"{len(wrong)} gene ID(s) are not unversioned {self.species} "
                f"Ensembl gene IDs: {wrong[:5]}"
            )
        if self.n_genes != len(ids):
            raise ValueError(f"n_genes {self.n_genes} != {len(ids)} genes")
        if self.gene_set_hash != compute_panel_hash(ids):
            raise ValueError("gene_set_hash does not match the genes")
        if self.csv_sha256 != _sha256_text(gene_set_csv_text(self.genes)):
            raise ValueError("csv_sha256 does not match the genes")
        overlap = {gene.gene_id for gene in self.missing_genes} & set(ids)
        if overlap:
            raise ValueError(f"missing genes are in the set: {sorted(overlap)[:5]}")
        if self.whole_transcriptome != is_whole_transcriptome(len(ids)):
            raise ValueError("whole_transcriptome does not match n_genes")
        return self

    @property
    def gene_ids(self) -> list[str]:
        """Return the sorted gene IDs."""
        return [gene.gene_id for gene in self.genes]

    @property
    def gene_symbols(self) -> list[str]:
        """Return the symbols, in ``gene_ids`` order."""
        return [gene.gene_symbol for gene in self.genes]

    def csv_text(self) -> str:
        """Return the CSV (``gene_symbol``, ``gene_id``)."""
        return gene_set_csv_text(self.genes)

    @classmethod
    def load(cls, path: Path | str) -> AnalysisGeneSet:
        """Read ``analysis_gene_set.json``."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class GeneSetPaths:
    """The files ``write_analysis_gene_set`` wrote."""

    record: Path
    csv: Path
    missing_csv: Path


def write_analysis_gene_set(
    gene_set: AnalysisGeneSet, output_dir: Path | str
) -> GeneSetPaths:
    """Write the record, the gene CSV and the missing-gene CSV.

    Args:
        gene_set: The gene set.
        output_dir: Directory to write into (created).

    Returns:
        The written paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / gene_set.csv_file
    csv_path.write_text(gene_set.csv_text(), encoding="utf-8")
    missing_path = root / ANALYSIS_GENE_SET_MISSING_CSV
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*CSV_COLUMNS, "panel_gene_with_same_symbol"])
    for gene in gene_set.missing_genes:
        writer.writerow(
            [
                gene.gene_symbol,
                gene.gene_id,
                gene_set.missing_symbol_matches.get(gene.gene_id, ""),
            ]
        )
    missing_path.write_text(buffer.getvalue(), encoding="utf-8")
    record_path = root / ANALYSIS_GENE_SET_JSON
    record_path.write_text(
        json.dumps(gene_set.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return GeneSetPaths(record=record_path, csv=csv_path, missing_csv=missing_path)


# --------------------------------------------------------------------------
# The source panel


@dataclass(frozen=True)
class SourcePanel:
    """A resolved panel and the input it was read from.

    Attributes:
        record: What is recorded about it.
        genes: Its genes, sorted by ID.
        input: The input record.
    """

    record: SourcePanelRecord
    genes: tuple[GeneSetRow, ...]
    input: GeneSetInput


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_features_sha256(raw: RawPanel) -> str:
    """Order-independent sha256 of a raw panel's declared features."""
    columns = ["name", "symbol", "native_value", "feature_type"]
    rows = sorted(
        "\t".join(str(value) for value in row)
        for row in raw.features[columns].itertuples(index=False, name=None)
    )
    return _sha256_text("\n".join(rows))


def _read_elem(node: Any) -> Any:
    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem
    return read_elem(node)


def read_spatialdata_table_var(
    path: Path | str, table_key: str | None = None
) -> tuple[pd.DataFrame, Path, str]:
    """Read only the ``var`` of a SpatialData table (no counts are loaded).

    Args:
        path: A SpatialData store (with ``tables/``) or one table group.
        table_key: Table under ``tables/`` (default ``table_original``);
            ignored for a table group.

    Returns:
        ``(var, table_group_path, table_key)``.

    Raises:
        GeneSetError: If the path holds no such table.
    """
    import zarr

    root = Path(path)
    if (root / "tables").is_dir():
        key = table_key or DEFAULT_TABLE_KEY
        group_path = root / "tables" / key
    else:
        group_path, key = root, table_key or root.name
    if not (group_path / "var").exists():
        raise GeneSetError(f"{group_path} is not a SpatialData table (no var)")
    group = zarr.open_group(str(group_path), mode="r")
    var = pd.DataFrame(_read_elem(group["var"]))
    return var, group_path, key


def read_source_panel(
    path: Path | str,
    *,
    species: Species,
    platform: str | None = None,
    table_key: str | None = None,
    config: AnnotationConfig | None = None,
) -> SourcePanel:
    """Read a panel and resolve it as ``ANNOTATE_PANEL`` does.

    Args:
        path: A SpatialData store or table group (its table ``var``), or a
            panel file ``annotation-panel`` reads (``.h5ad`` var, gene table,
            Xenium ``gene_panel.json``, MERSCOPE codebook).
        species: Run species.
        platform: The panel's platform (selects control name patterns).
        table_key: Table of a SpatialData store (default
            ``table_original``).
        config: Annotation config whose panel settings give the resolver
            sources (gene tables, aliases, overrides) and control rules;
            default: the species defaults.

    Returns:
        The source panel.

    Raises:
        GeneSetError: If the panel's gene-ID resolution is refused.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.gene_ids import ResolutionRules, gene_id_sources
    from merxen.annotation.panel import (
        ControlRegistry,
        PanelSource,
        declared_panel,
        raw_panel_from_var,
        read_panel_file,
    )

    config = config or AnnotationConfig(species=species)
    source_path = Path(path)
    platform_value = platform.upper() if platform else None
    kind: SourceKind
    if source_path.is_dir():
        var, group_path, key = read_spatialdata_table_var(source_path, table_key)
        raw = raw_panel_from_var(
            var, source=PanelSource(kind="h5ad_var", path=str(group_path))
        )
        kind, input_kind, input_path, input_key = (
            "spatialdata_table",
            "spatialdata_table",
            str(group_path),
            key,
        )
        sha256 = _declared_features_sha256(raw)
    else:
        raw = read_panel_file(source_path)
        kind, input_kind, input_path, input_key = (
            "panel_file",
            raw.source.kind,
            str(source_path),
            None,
        )
        sha256 = raw.source.sha256 or _file_sha256(source_path)
    declared = declared_panel(
        raw,
        species=species,
        platform=platform_value,
        registry=ControlRegistry.from_config(config.panel),
        sources=gene_id_sources(config.panel, species),
        rules=ResolutionRules.from_config(config.panel),
    )
    if declared.status == "refused":
        raise GeneSetError(
            f"{source_path}: the panel's gene-ID resolution is refused "
            f"({declared.refusal_text()})"
        )
    genes = tuple(
        GeneSetRow(gene_id=gene_id, gene_symbol=symbol)
        for gene_id, symbol in zip(declared.ensembl_ids, declared.symbols, strict=True)
    )
    id_sources: dict[str, int] = {}
    for source in declared.id_sources.values():
        id_sources[source] = id_sources.get(source, 0) + 1
    record = SourcePanelRecord(
        kind=kind,
        platform=platform_value,
        n_features=len(raw.features),
        n_genes=len(genes),
        panel_hash=declared.panel_hash,
        declared_features_sha256=_declared_features_sha256(raw),
        controls_removed={
            reason: len(names)
            for reason, names in sorted(declared.controls_removed.items())
        },
        unresolved=dict(sorted(declared.unresolved.items())),
        id_sources=dict(sorted(id_sources.items())),
        whole_transcriptome=is_whole_transcriptome(len(genes)),
    )
    source_input = GeneSetInput(
        role="source_panel",
        kind=input_kind,
        path=input_path,
        sha256=sha256,
        n_rows=len(raw.features),
        key=input_key,
    )
    logger.info(
        "Source panel %s: %d features, %d genes, panel_hash %s",
        input_path,
        record.n_features,
        record.n_genes,
        record.panel_hash[:16],
    )
    return SourcePanel(record=record, genes=genes, input=source_input)


# --------------------------------------------------------------------------
# Requested gene lists


@dataclass(frozen=True)
class RequestedGenes:
    """The gene list a recipe asks for.

    Attributes:
        genes: Requested genes (IDs normalised, file order, no repeats).
        input: Where they came from.
    """

    genes: tuple[GeneSetRow, ...]
    input: GeneSetInput


def _normalised_ids(
    rows: Iterable[tuple[str, str]], *, species: Species, origin: str
) -> tuple[GeneSetRow, ...]:
    """Normalise ``(gene_id, symbol)`` rows and check the IDs' species.

    Raises:
        GeneSetError: If a value is not an Ensembl gene ID of ``species``
            (symbols are never accepted as IDs).
    """
    pattern = SPECIES_ID_PATTERNS[species]
    seen: dict[str, GeneSetRow] = {}
    wrong: list[str] = []
    for gene_id, symbol in rows:
        value = clean_native_value(gene_id)
        if not value:
            continue
        if not pattern.fullmatch(value):
            wrong.append(str(gene_id))
            continue
        seen.setdefault(
            value, GeneSetRow(gene_id=value, gene_symbol=clean_text(symbol))
        )
    if wrong:
        raise GeneSetError(
            f"{origin}: {len(wrong)} value(s) are not {species} Ensembl gene IDs "
            f"(gene sets join on Ensembl gene IDs, never on symbols): {wrong[:5]}"
        )
    if not seen:
        raise GeneSetError(f"{origin} lists no gene")
    return tuple(seen.values())


def public_panel_genes(
    key: str,
    *,
    species: Species,
    public_panel_dir: Path | str | None = None,
) -> RequestedGenes:
    """Return a pinned public panel's genes.

    With ``public_panel_dir``, ``<dir>/<key>.gene_list.csv`` from ``merxen
    annotation-panel-fetch`` (symbols and IDs; checked against its manifest
    and the pinned gene count). Without it, the packaged pinned list of the
    Prime 5K panels (IDs only).

    Args:
        key: ``PUBLIC_PANEL_LISTS`` key.
        species: Run species; must be the panel's.
        public_panel_dir: An ``annotation-panel-fetch`` output directory.

    Returns:
        The requested genes.

    Raises:
        GeneSetError: If the key is unknown, of another species, not
            available, or the list fails its pin.
    """
    from merxen.annotation.public_panels import (
        GENE_LIST_SUFFIX,
        MANIFEST_SUFFIX,
        PUBLIC_PANEL_LISTS,
    )

    item = PUBLIC_PANEL_LISTS.get(key)
    if item is None:
        raise GeneSetError(
            f"unknown public panel {key!r}; known: {sorted(PUBLIC_PANEL_LISTS)}"
        )
    if item.species != species:
        raise GeneSetError(
            f"public panel {key} is {item.species}, the run is {species}"
        )
    if public_panel_dir is not None:
        path = Path(public_panel_dir) / f"{key}{GENE_LIST_SUFFIX}"
        if not path.is_file():
            raise GeneSetError(
                f"{path} does not exist; write it with merxen annotation-panel-fetch "
                f"--panel {key} --out-dir {public_panel_dir}"
            )
        digest = _file_sha256(path)
        manifest_path = Path(public_panel_dir) / f"{key}{MANIFEST_SUFFIX}"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("gene_list_sha256")
            if expected and expected != digest:
                raise GeneSetError(f"{path} does not match its manifest's sha256")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(frame.columns[:2]) != list(CSV_COLUMNS):
            raise GeneSetError(f"{path} does not have the columns {CSV_COLUMNS}")
        genes = _normalised_ids(
            zip(frame["gene_id"], frame["gene_symbol"], strict=True),
            species=species,
            origin=str(path),
        )
        if len(genes) != item.n_genes:
            raise GeneSetError(f"{path} has {len(genes)} genes, pinned {item.n_genes}")
        return RequestedGenes(
            genes=genes,
            input=GeneSetInput(
                role="public_panel",
                kind="public_panel_gene_list",
                path=str(path),
                sha256=digest,
                n_rows=len(frame),
                key=key,
            ),
        )
    from merxen.annotation.sim_inputs import get_asset, read_asset_table

    asset = get_asset(PACKAGED_PUBLIC_PANEL_ASSET)
    table = read_asset_table(asset)
    rows = table[table["panel_key"] == key]
    if rows.empty:
        raise GeneSetError(
            f"public panel {key} is not packaged; fetch it with merxen "
            "annotation-panel-fetch and pass its directory (--public-panel-dir)"
        )
    asset_path = asset.path()
    genes = _normalised_ids(
        ((gene_id, "") for gene_id in rows["gene_id"]),
        species=species,
        origin=f"packaged asset {PACKAGED_PUBLIC_PANEL_ASSET} ({key})",
    )
    if len(genes) != item.n_genes:
        raise GeneSetError(
            f"packaged list {key} has {len(genes)} genes, pinned {item.n_genes}"
        )
    return RequestedGenes(
        genes=genes,
        input=GeneSetInput(
            role="public_panel",
            kind="packaged_asset",
            path=f"merxen/assets/annotation/sim_inputs/{asset_path.name}",
            sha256=_file_sha256(asset_path),
            n_rows=len(rows),
            key=key,
        ),
    )


def explicit_genes(path: Path | str, *, species: Species) -> RequestedGenes:
    """Read an explicit list of Ensembl gene IDs.

    Accepted: a CSV / TSV with an ID column (``ensembl_id``, ``gene_ids``,
    ``gene_id`` or ``feature_id``; an optional symbol column is kept for
    display), a JSON list, a ``panel_genes*.json`` (``ensembl_ids``), an
    ``analysis_gene_set.json`` or plain text with one ID per line. Every value
    must be an Ensembl gene ID of the run species: symbols are refused.

    Args:
        path: The list.
        species: Run species.

    Returns:
        The requested genes.

    Raises:
        GeneSetError: If the file holds no ID list or a value is not an ID.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise GeneSetError(f"explicit gene list {file_path} does not exist")
    suffix = file_path.suffix.lower()
    rows: list[tuple[str, str]]
    if suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(
            file_path,
            sep="\t" if suffix == ".tsv" else ",",
            dtype=str,
            keep_default_na=False,
            comment="#",
        )
        columns = {str(column).strip(): column for column in frame.columns}
        id_column = next((columns[c] for c in NATIVE_ID_COLUMNS if c in columns), None)
        if id_column is None:
            raise GeneSetError(
                f"{file_path} needs an Ensembl gene ID column ({NATIVE_ID_COLUMNS}); "
                f"found {list(columns)} (gene sets never join on symbols)"
            )
        symbol_column = next((columns[c] for c in SYMBOL_COLUMNS if c in columns), None)
        symbols = (
            frame[symbol_column].tolist()
            if symbol_column is not None
            else [""] * len(frame)
        )
        rows = list(zip(frame[id_column].tolist(), symbols, strict=True))
    elif suffix == ".json":
        payload: object = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [(str(value), "") for value in payload]
        elif isinstance(payload, dict) and "ensembl_ids" in payload:
            ids = [str(value) for value in payload["ensembl_ids"]]
            names = payload.get("symbols")
            symbols = (
                [str(value) for value in names]
                if isinstance(names, list) and len(names) == len(ids)
                else [""] * len(ids)
            )
            rows = list(zip(ids, symbols, strict=True))
        elif isinstance(payload, dict) and isinstance(payload.get("genes"), list):
            rows = [
                (str(gene.get("gene_id", "")), str(gene.get("gene_symbol", "")))
                for gene in payload["genes"]
                if isinstance(gene, dict)
            ]
        else:
            raise GeneSetError(f"{file_path} holds no gene ID list")
    else:
        rows = [
            (line.strip(), "")
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    genes = _normalised_ids(rows, species=species, origin=str(file_path))
    return RequestedGenes(
        genes=genes,
        input=GeneSetInput(
            role="explicit_list",
            kind="gene_id_list",
            path=str(file_path),
            sha256=_file_sha256(file_path),
            n_rows=len(rows),
        ),
    )


# --------------------------------------------------------------------------
# Building


@dataclass(frozen=True)
class GeneIdJoin:
    """The result of joining a requested list onto a panel by gene ID.

    Attributes:
        genes: Panel genes that were requested (sorted by ID; panel symbols).
        missing: Requested genes the panel lacks (request order).
        symbol_matches: Missing ID -> the panel gene that shares its symbol.
    """

    genes: tuple[GeneSetRow, ...]
    missing: tuple[GeneSetRow, ...]
    symbol_matches: Mapping[str, str]


def join_on_gene_ids(
    panel: Sequence[GeneSetRow], requested: Sequence[GeneSetRow]
) -> GeneIdJoin:
    """Select the panel genes a list requests, by Ensembl gene ID only.

    Symbols play no part in the selection: a panel gene whose symbol equals a
    requested ID (an ID-shaped symbol) or a requested symbol is not selected
    unless its own ID is requested. A missing gene whose symbol a panel gene
    carries under another ID is reported in ``symbol_matches``.

    Args:
        panel: The panel's genes.
        requested: The requested genes.

    Returns:
        The join.
    """
    by_id = {gene.gene_id: gene for gene in panel}
    wanted = {gene.gene_id for gene in requested}
    genes = tuple(
        sorted((by_id[i] for i in wanted if i in by_id), key=lambda g: g.gene_id)
    )
    missing = tuple(gene for gene in requested if gene.gene_id not in by_id)
    by_symbol: dict[str, list[str]] = {}
    for gene in panel:
        if gene.gene_symbol:
            by_symbol.setdefault(gene.gene_symbol, []).append(gene.gene_id)
    matches = {
        gene.gene_id: by_symbol[gene.gene_symbol][0]
        for gene in missing
        if gene.gene_symbol and len(by_symbol.get(gene.gene_symbol, [])) == 1
    }
    return GeneIdJoin(genes=genes, missing=missing, symbol_matches=matches)


def build_analysis_gene_set(
    source: SourcePanel,
    recipe: Recipe | str,
    *,
    species: Species,
    name: str | None = None,
    version: int = 1,
    public_panel_dir: Path | str | None = None,
) -> AnalysisGeneSet:
    """Apply a recipe to a source panel.

    Args:
        source: The resolved panel (``read_source_panel``).
        recipe: The recipe (or its text).
        species: Run species.
        name: Gene-set name (default: ``Recipe.default_name``).
        version: Version of the set's definition.
        public_panel_dir: ``annotation-panel-fetch`` output for
            ``intersect_public_panel`` (default: the packaged lists).

    Returns:
        The gene set.

    Raises:
        GeneSetError: If the recipe's inputs are invalid or no gene remains.
    """
    parsed = Recipe.parse(recipe) if isinstance(recipe, str) else recipe
    inputs = [source.input]
    if parsed.kind == RECIPE_ALL_PANEL_GENES:
        join = GeneIdJoin(genes=source.genes, missing=(), symbol_matches={})
        n_requested: int | None = None
    else:
        if parsed.kind == RECIPE_INTERSECT_PUBLIC_PANEL:
            requested = public_panel_genes(
                str(parsed.argument),
                species=species,
                public_panel_dir=public_panel_dir,
            )
        else:
            requested = explicit_genes(str(parsed.argument), species=species)
        inputs.append(requested.input)
        join = join_on_gene_ids(source.genes, requested.genes)
        n_requested = len(requested.genes)
    if not join.genes:
        raise GeneSetError(f"recipe {parsed.label} selects no gene of the panel")
    genes = list(join.genes)
    csv_text = gene_set_csv_text(genes)
    gene_set = AnalysisGeneSet(
        name=name or parsed.default_name,
        version=version,
        species=species,
        platform=source.record.platform,
        recipe=parsed.label,
        recipe_kind=parsed.kind,
        inputs=inputs,
        source_panel=source.record,
        genes=genes,
        n_genes=len(genes),
        n_requested=n_requested,
        missing_genes=list(join.missing),
        missing_symbol_matches=dict(join.symbol_matches),
        n_excluded_panel_genes=source.record.n_genes - len(genes),
        whole_transcriptome=is_whole_transcriptome(len(genes)),
        gene_set_hash=compute_panel_hash([gene.gene_id for gene in genes]),
        csv_sha256=_sha256_text(csv_text),
    )
    logger.info(
        "Analysis gene set %s (%s): %d of %d panel genes%s, gene_set_hash %s",
        gene_set.name,
        gene_set.recipe,
        gene_set.n_genes,
        source.record.n_genes,
        ""
        if n_requested is None
        else f"; {len(join.missing)} of {n_requested} requested genes missing",
        gene_set.gene_set_hash[:16],
    )
    return gene_set
