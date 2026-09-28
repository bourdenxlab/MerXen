"""Gene-ID resolution of declared panels and the species check (plan §8.4).

One resolver maps every non-control feature of a declared panel to an
unversioned Ensembl gene ID of the run species. The first source that
resolves a feature wins, and the source is recorded:

1. ``native``: the feature's own Ensembl gene ID (``var`` column
   ``ensembl_id``, ``gene_ids``, ``gene_id`` or ``feature_id``; version
   suffix stripped). Release drift: a native ID absent from the run
   species' local gene table whose symbol the table resolves to another ID
   takes the table's ID (``symbol_fallback``).
2. ``pair_lookup``: the ID the pair's other platform declares for the same
   symbol (Xenium IDs onto the symbol-only MERSCOPE side).
3. ``fallback_table``: the run species' local gene table (WHB / WMB
   ``gene.csv`` or a reference ``.h5ad`` ``var``), exact-case symbol match
   first, then case-insensitive; a unique candidate only. This generalises
   the M0e fallback (``annotation_gene_id_fallback_csv``).
4. ``alias``: an optional local alias table (HGNC / MGI previous and alias
   symbols); only aliases with a single target ID are used.
5. ``override``: curated rows of ``gene_id_overrides_<species>.csv`` (and
   an optional configured table), each with a reason.

Nothing is downloaded. Features several of which resolve to one ID are
merged (MAP sums their counts); unresolved features are listed with a
reason (reporter genes such as GFP, isoform probes, genes absent from the
tables).

**Refusals** (the panel is ``refused``, plan §8.2):

* ``species_mismatch``: the exact-case species test. HGNC symbols are upper
  case (``GFAP``) and MGI symbols are not (``Gfap``), so a mouse panel run
  under ``species = "human"`` would resolve through the case-insensitive
  fallback and pass every prefix check by construction. The test counts the
  panel symbols that match each species' gene table exactly and
  case-insensitively; it fails when the other species matches exactly more
  often, or when the run species' exact matches are fewer than
  ``species_exact_case_min_ratio`` of its case-insensitive matches.
* ``gene_id_resolution``: fewer than ``min_gene_id_resolution`` of the
  non-control features resolve.
* ``native_id_prefix``: fewer than ``min_native_prefix_share`` of the
  values in the native ID column are Ensembl gene IDs of the run species.
  Symbols stored in the ID column (the ag7 failure that compared symbols
  with Ensembl IDs and found 0 of 498 root markers) land here and are
  listed as ``symbols_as_ids``. Ensembl transcript or protein IDs (MERSCOPE
  codebooks, isoform probes) are not gene IDs and do not count.
* ``other_species_ids``: more than ``max_other_species_prefix_share`` of
  the native values are Ensembl gene IDs of another species.

The module imports numpy, pandas and pydantic only; the local gene tables
are read with the M0e loader of ``merxen.analysis.mapmycells``, imported
when a table is loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.vocab import SPECIES, Species, asset_path
from merxen.gene_ids import is_ensembl_gene_id

if TYPE_CHECKING:
    from merxen.analysis.mapmycells import GeneIdFallbackTable
    from merxen.annotation.config import AnnotationPanelConfig

logger = logging.getLogger(__name__)

SPECIES_ID_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "human": re.compile(r"^ENSG\d+$"),
    "mouse": re.compile(r"^ENSMUSG\d+$"),
}
# Ensembl transcript, protein and exon IDs (and isoform probe names built on
# them, e.g. Xenium's ``ENST00000262410.10_MAPT3R_exon9_exon11``): not gene
# IDs, so they neither resolve a feature nor count against the prefix rule.
ENSEMBL_NON_GENE_PATTERN: Final = re.compile(r"^ENS[A-Z]*[TPE]\d+")
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
OVERRIDE_FILES: Final[dict[str, str]] = {
    "human": "gene_id_overrides_human.csv",
    "mouse": "gene_id_overrides_mouse.csv",
}
OVERRIDE_COLUMNS: Final[tuple[str, ...]] = ("symbol", "ensembl_id", "reason")
# Alias-table layouts: HGNC ``hgnc_complete_set`` (symbol, alias_symbol,
# prev_symbol, ensembl_gene_id; lists separated by "|"), MGI marker lists
# ("Marker Symbol", "Marker Synonyms (pipe-separated)") and a plain table
# (alias, symbol and / or ensembl_id).
ALIAS_TARGET_SYMBOL_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "approved_symbol",
    "Marker Symbol",
    "gene_symbol",
)
ALIAS_TARGET_ID_COLUMNS: Final[tuple[str, ...]] = (
    "ensembl_id",
    "ensembl_gene_id",
    "gene_identifier",
)
ALIAS_COLUMNS: Final[tuple[str, ...]] = (
    "alias",
    "alias_symbol",
    "prev_symbol",
    "previous_symbol",
    "synonyms",
    "Marker Synonyms (pipe-separated)",
)
# Values of a native ID column that mean "no ID" (MERSCOPE codebooks write
# -1 for features without a transcript ID).
_MISSING_NATIVE_TOKENS: Final = frozenset({"", "nan", "none", "<na>", "-1", "na"})

GeneIdSource = Literal[
    "native",
    "pair_lookup",
    "fallback_table",
    "symbol_fallback",
    "alias",
    "override",
]
NativeKind = Literal["none", "run", "other_species", "non_gene", "non_id"]
RefusalReason = Literal[
    "no_features",
    "species_mismatch",
    "gene_id_resolution",
    "native_id_prefix",
    "other_species_ids",
]
SpeciesCheckStatus = Literal["pass", "species_mismatch", "not_evaluable"]


def strip_version(gene_id: str) -> str:
    """Return an Ensembl ID without its ``.N`` version suffix.

    Args:
        gene_id: Candidate identifier.

    Returns:
        The stripped, whitespace-trimmed identifier.
    """
    return re.sub(r"\.\d+$", "", str(gene_id).strip())


def clean_text(value: Any) -> str:
    """Return a table cell as stripped text, ``""`` for missing values.

    Args:
        value: A cell value.

    Returns:
        The text; ``None``, NaN and the tokens ``nan`` / ``none`` / ``<NA>``
        become ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def clean_native_value(value: Any) -> str:
    """Return a native ID cell unversioned, ``""`` when it means "no ID".

    Args:
        value: A cell of the native ID column.

    Returns:
        The unversioned value, or ``""`` for missing values and the
        placeholders ``-1`` / ``NA``. Ensembl gene IDs are upper-cased
        (upper case by definition; ``ensg00000141510`` is a human ID, not
        another species', M3b review 2).
    """
    text = strip_version(clean_text(value))
    if text.lower() in _MISSING_NATIVE_TOKENS:
        return ""
    return text.upper() if is_ensembl_gene_id(text) else text


def other_species(species: str) -> Species:
    """Return the other supported species.

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        ``"mouse"`` for human and ``"human"`` for mouse.

    Raises:
        ValueError: If the species is not supported.
    """
    if species not in SPECIES:
        raise ValueError(f"unsupported species {species!r}")
    return cast("Species", "mouse" if species == "human" else "human")


def native_kind(value: str, species: str) -> NativeKind:
    """Classify a value of a native ID column for a run species.

    Args:
        value: The unversioned value (``""`` when the feature has none).
        species: Run species.

    Returns:
        ``none`` (no value), ``run`` (an Ensembl gene ID of the run
        species), ``other_species`` (an Ensembl gene ID of another species),
        ``non_gene`` (an Ensembl transcript, protein or exon ID, or a probe
        named after one) or ``non_id`` (anything else, e.g. a symbol).
    """
    if not value:
        return "none"
    if is_ensembl_gene_id(value):
        # Ensembl gene IDs are upper case; a lower-case one is still an ID
        # of its prefix's species (``clean_native_value`` normalises too).
        return (
            "run"
            if SPECIES_ID_PATTERNS[species].fullmatch(value.strip().upper())
            else "other_species"
        )
    if ENSEMBL_NON_GENE_PATTERN.match(value):
        return "non_gene"
    return "non_id"


# --------------------------------------------------------------------------
# Local tables


@dataclass(frozen=True)
class GeneTable:
    """A species' local symbol-to-ID table (reference ``gene.csv`` or ``var``).

    Attributes:
        species: Species whose IDs the table holds.
        path: The file it was read from (``None`` for an in-memory table).
        ids_by_symbol: Distinct IDs per exact symbol.
        ids_by_casefold_symbol: Distinct IDs per case-folded symbol.
        gene_ids: Every ID of the table.
    """

    species: Species
    path: str | None
    ids_by_symbol: Mapping[str, tuple[str, ...]]
    ids_by_casefold_symbol: Mapping[str, tuple[str, ...]]
    gene_ids: frozenset[str]

    @classmethod
    def from_pairs(
        cls,
        species: Species,
        pairs: Iterable[tuple[str, str]],
        *,
        path: str | None = None,
    ) -> GeneTable:
        """Build a table from ``(symbol, gene_id)`` pairs.

        IDs of other species are skipped and version suffixes stripped, as
        the M0e loader does.

        Args:
            species: Species of the IDs to keep.
            pairs: ``(symbol, gene_id)`` pairs.
            path: Where the pairs came from, for the record.

        Returns:
            The table.
        """
        pattern = SPECIES_ID_PATTERNS[species]
        by_symbol: dict[str, set[str]] = {}
        for raw_symbol, raw_id in pairs:
            symbol, gene_id = clean_text(raw_symbol), strip_version(raw_id)
            if not symbol or not pattern.fullmatch(gene_id):
                continue
            by_symbol.setdefault(symbol, set()).add(gene_id)
        by_casefold: dict[str, set[str]] = {}
        for symbol, ids in by_symbol.items():
            by_casefold.setdefault(symbol.casefold(), set()).update(ids)
        return cls(
            species=species,
            path=path,
            ids_by_symbol={s: tuple(sorted(ids)) for s, ids in by_symbol.items()},
            ids_by_casefold_symbol={
                s: tuple(sorted(ids)) for s, ids in by_casefold.items()
            },
            gene_ids=frozenset(g for ids in by_symbol.values() for g in ids),
        )

    @classmethod
    def from_fallback_table(cls, table: GeneIdFallbackTable) -> GeneTable:
        """Wrap a table read by the M0e loader.

        Args:
            table: ``merxen.analysis.mapmycells.load_gene_id_fallback_table``
                output.

        Returns:
            The table.
        """
        species = cast("Species", table.query_species)
        return cls(
            species=species,
            path=str(table.path),
            ids_by_symbol=dict(table.ids_by_symbol),
            ids_by_casefold_symbol=dict(table.ids_by_casefold_symbol),
            gene_ids=frozenset(
                gene_id for ids in table.ids_by_symbol.values() for gene_id in ids
            ),
        )

    @property
    def n_symbols(self) -> int:
        """Return the number of distinct exact symbols."""
        return len(self.ids_by_symbol)

    def has_exact(self, symbol: str) -> bool:
        """Return whether the table holds the symbol in this exact case."""
        return symbol in self.ids_by_symbol

    def has_casefold(self, symbol: str) -> bool:
        """Return whether the table holds the symbol in any case."""
        return symbol.casefold() in self.ids_by_casefold_symbol

    def candidates(self, symbol: str) -> tuple[str, ...]:
        """Return the IDs of a symbol, preferring an exact-case match.

        Args:
            symbol: Gene symbol.

        Returns:
            Distinct candidate IDs (empty when absent).
        """
        return self.ids_by_symbol.get(symbol) or self.ids_by_casefold_symbol.get(
            symbol.casefold(), ()
        )

    def unique_id(self, symbol: str) -> tuple[str | None, str]:
        """Return the single ID of a symbol, or why there is none.

        Args:
            symbol: Gene symbol.

        Returns:
            ``(gene_id, "")``, or ``(None, "not_in_fallback_table")`` /
            ``(None, "ambiguous_in_fallback_table")``.
        """
        found = self.candidates(symbol)
        if not found:
            return None, "not_in_fallback_table"
        if len(found) > 1:
            return None, "ambiguous_in_fallback_table"
        return found[0], ""

    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable provenance."""
        return {
            "species": self.species,
            "path": self.path,
            "n_symbols": self.n_symbols,
            "n_gene_ids": len(self.gene_ids),
        }


def load_gene_table(path: Path | str, species: Species) -> GeneTable | None:
    """Read a local gene table for one species (no network).

    Uses the M0e loader: an Allen-style ``gene.csv`` (``gene_identifier``,
    ``gene_symbol``; ``ensembl_id`` / ``gene_id`` and ``symbol`` / ``gene``
    also work; ``.tsv`` tab-separated) or a reference ``.h5ad`` ``var``.

    Args:
        path: The table.
        species: Species whose IDs to keep.

    Returns:
        The table, or ``None`` when it holds no ID of that species (a WHB
        table read for a mouse run).
    """
    from merxen.analysis.mapmycells import load_gene_id_fallback_table

    table = load_gene_id_fallback_table(path, query_species=species)
    if table.n_species_rows == 0:
        return None
    return GeneTable.from_fallback_table(table)


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|;,]", value) if part.strip()]


@dataclass(frozen=True)
class AliasTable:
    """Previous and alias symbols of one species (HGNC / MGI; local only).

    Attributes:
        species: Species.
        path: The file read.
        symbols_by_alias: Approved symbols per case-folded alias.
        ids_by_alias: Ensembl IDs per case-folded alias, when the table has
            an ID column.
        ids_by_symbol: Ensembl IDs per case-folded approved symbol, when the
            table has an ID column.
        aliases_by_symbol: Previous and alias symbols per case-folded
            approved symbol (the reverse map: a current symbol whose gene
            table still lists a previous one, e.g. H2AX / H2AFX).
    """

    species: Species
    path: str | None
    symbols_by_alias: Mapping[str, frozenset[str]]
    ids_by_alias: Mapping[str, frozenset[str]] = field(default_factory=dict)
    ids_by_symbol: Mapping[str, frozenset[str]] = field(default_factory=dict)
    aliases_by_symbol: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def resolve(
        self, symbol: str, table: GeneTable | None
    ) -> tuple[str | None, str, str]:
        """Resolve a symbol through its aliases.

        Both directions count: the symbol as an alias or previous symbol
        (its approved symbols and their IDs), and the symbol as an approved
        symbol (the row's own ID, and the IDs the gene table gives its
        previous and alias symbols: the table may predate the rename). Only
        a single-target symbol resolves: everything it points to must give
        one Ensembl ID of the species.

        Args:
            symbol: The panel symbol.
            table: The run species' gene table, which turns approved,
                previous and alias symbols into IDs.

        Returns:
            ``(gene_id, detail, "")`` (``alias of <approved>``, or
            ``approved symbol of <previous>``), or ``(None, "", reason)``
            with reason ``not_an_alias`` or ``ambiguous_alias``.
        """
        key = symbol.casefold()
        approved = self.symbols_by_alias.get(key, frozenset())
        ids = set(self.ids_by_alias.get(key, frozenset()))
        ids.update(self.ids_by_symbol.get(key, frozenset()))
        previous = self.aliases_by_symbol.get(key, frozenset())
        if table is not None:
            for name in approved | previous:
                ids.update(table.candidates(name))
        pattern = SPECIES_ID_PATTERNS[self.species]
        ids = {gene_id for gene_id in ids if pattern.fullmatch(gene_id)}
        if not ids:
            return None, "", "not_an_alias"
        if len(ids) > 1:
            return None, "", "ambiguous_alias"
        if approved:
            detail = f"alias of {','.join(sorted(approved))}"
        elif previous:
            detail = f"approved symbol of {','.join(sorted(previous))}"
        else:
            detail = "approved symbol"
        return next(iter(ids)), detail, ""


def load_alias_table(path: Path | str, species: Species) -> AliasTable:
    """Read a local alias table (HGNC complete set, MGI list or plain CSV).

    Args:
        path: ``.csv``, ``.tsv`` or ``.txt`` (tab-separated) table with an
            alias column (``ALIAS_COLUMNS``; "|"-separated lists allowed)
            and an approved-symbol and / or Ensembl-ID column.
        species: Species of the table.

    Returns:
        The alias table.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it has no alias column or no target column.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"gene alias table {file_path} does not exist")
    is_tab = file_path.suffix.lower() in {".tsv", ".txt", ".rpt"}
    frame = pd.read_csv(
        file_path,
        sep="\t" if is_tab else ",",
        dtype=str,
        keep_default_na=False,
        comment=None,
    )
    alias_columns = [column for column in ALIAS_COLUMNS if column in frame.columns]
    symbol_column = next(
        (c for c in ALIAS_TARGET_SYMBOL_COLUMNS if c in frame.columns), None
    )
    id_column = next((c for c in ALIAS_TARGET_ID_COLUMNS if c in frame.columns), None)
    if not alias_columns or (symbol_column is None and id_column is None):
        raise ValueError(
            f"alias table {file_path} needs an alias column {ALIAS_COLUMNS} and "
            f"a target column {ALIAS_TARGET_SYMBOL_COLUMNS + ALIAS_TARGET_ID_COLUMNS}"
            f"; found {list(frame.columns)}"
        )
    symbols: dict[str, set[str]] = {}
    ids: dict[str, set[str]] = {}
    own_ids: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for record in frame.to_dict(orient="records"):
        target = clean_text(record.get(symbol_column, "")) if symbol_column else ""
        target_id = strip_version(record.get(id_column, "")) if id_column else ""
        if target and target_id:
            own_ids.setdefault(target.casefold(), set()).add(target_id)
        for column in alias_columns:
            for alias in _split_list(clean_text(record.get(column, ""))):
                key = alias.casefold()
                if target:
                    symbols.setdefault(key, set()).add(target)
                    reverse.setdefault(target.casefold(), set()).add(alias)
                if target_id:
                    ids.setdefault(key, set()).add(target_id)

    def frozen(mapping: dict[str, set[str]]) -> dict[str, frozenset[str]]:
        return {key: frozenset(value) for key, value in mapping.items()}

    return AliasTable(
        species=species,
        path=str(file_path),
        symbols_by_alias=frozen(symbols),
        ids_by_alias=frozen(ids),
        ids_by_symbol=frozen(own_ids),
        aliases_by_symbol=frozen(reverse),
    )


@dataclass(frozen=True)
class GeneIdOverride:
    """A curated symbol-to-ID row.

    Attributes:
        symbol: Panel symbol.
        ensembl_id: Its Ensembl gene ID.
        reason: Why the row exists.
        source: File it came from.
    """

    symbol: str
    ensembl_id: str
    reason: str
    source: str


def _read_overrides(path: Path, species: Species) -> dict[str, GeneIdOverride]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    missing = [column for column in OVERRIDE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"gene-ID overrides {path} lack columns {missing}")
    pattern = SPECIES_ID_PATTERNS[species]
    rows: dict[str, GeneIdOverride] = {}
    for record in frame.to_dict(orient="records"):
        if "species" in frame.columns and clean_text(record["species"]) not in {
            "",
            species,
        }:
            continue
        symbol = clean_text(record["symbol"])
        gene_id = strip_version(record["ensembl_id"])
        reason = clean_text(record["reason"])
        if not symbol:
            continue
        if not pattern.fullmatch(gene_id):
            raise ValueError(
                f"gene-ID overrides {path}: {symbol} -> {gene_id!r} is not a "
                f"{species} Ensembl gene ID"
            )
        if not reason:
            raise ValueError(f"gene-ID overrides {path}: {symbol} has no reason")
        rows[symbol] = GeneIdOverride(
            symbol=symbol, ensembl_id=gene_id, reason=reason, source=str(path)
        )
    return rows


def load_overrides(
    species: Species, path: Path | str | None = None
) -> dict[str, GeneIdOverride]:
    """Load the packaged curated overrides of a species, plus a configured table.

    Args:
        species: Run species.
        path: ``gene_id_overrides_csv`` (columns ``symbol``, ``ensembl_id``,
            ``reason``, optional ``species``); its rows replace packaged
            rows of the same symbol.

    Returns:
        Overrides by symbol.

    Raises:
        ValueError: If a row lacks a reason or has another species' ID.
    """
    rows = _read_overrides(asset_path(OVERRIDE_FILES[species]), species)
    if path is not None:
        rows.update(_read_overrides(Path(path), species))
    return rows


def _override_for(
    overrides: Mapping[str, GeneIdOverride], symbol: str
) -> GeneIdOverride | None:
    found = overrides.get(symbol)
    if found is not None:
        return found
    matches = [
        row for key, row in overrides.items() if key.casefold() == symbol.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class GeneIdSources:
    """Everything the resolver may consult for one run (no network).

    Attributes:
        species: Run species.
        gene_tables: Local gene tables by species; the run species' table
            is the symbol fallback, both are used by the species test.
        pair_lookup: Symbol -> ID declared by the pair's other platform.
        aliases: Optional alias table of the run species.
        overrides: Curated overrides by symbol.
    """

    species: Species
    gene_tables: Mapping[str, GeneTable] = field(default_factory=dict)
    pair_lookup: Mapping[str, str] = field(default_factory=dict)
    aliases: AliasTable | None = None
    overrides: Mapping[str, GeneIdOverride] = field(default_factory=dict)

    @property
    def run_table(self) -> GeneTable | None:
        """Return the run species' gene table, if any."""
        return self.gene_tables.get(self.species)

    def with_pair_lookup(self, pair_lookup: Mapping[str, str]) -> GeneIdSources:
        """Return a copy with another pair lookup."""
        return GeneIdSources(
            species=self.species,
            gene_tables=self.gene_tables,
            pair_lookup=dict(pair_lookup),
            aliases=self.aliases,
            overrides=self.overrides,
        )

    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable provenance of the sources."""
        return {
            "species": self.species,
            "gene_tables": {
                species: table.describe()
                for species, table in sorted(self.gene_tables.items())
            },
            "n_pair_lookup": len(self.pair_lookup),
            "alias_table": None if self.aliases is None else self.aliases.path,
            "n_overrides": len(self.overrides),
            "override_sources": sorted({row.source for row in self.overrides.values()}),
        }


def gene_id_sources(
    config: AnnotationPanelConfig,
    species: Species,
    *,
    pair_lookup: Mapping[str, str] | None = None,
) -> GeneIdSources:
    """Load the resolver's sources from the panel config.

    The run species' gene table is ``gene_tables[species]``, else the M0e
    ``gene_id_fallback_csv``; the other species' table is
    ``gene_tables[other]`` (the species test needs both, and without it only
    the exact-case ratio is tested).

    Args:
        config: Panel settings.
        species: Run species.
        pair_lookup: Symbol -> ID from the pair's other platform.

    Returns:
        The sources.
    """
    tables: dict[str, GeneTable] = {}
    for table_species in SPECIES:
        typed = cast("Species", table_species)
        path = config.gene_tables.get(typed)
        if path is None and typed == species:
            path = config.gene_id_fallback_csv
        if path is None:
            continue
        table = load_gene_table(path, typed)
        if table is None:
            logger.warning("Gene table %s has no %s Ensembl IDs", path, typed)
            continue
        tables[typed] = table
    aliases = (
        load_alias_table(config.gene_alias_table, species)
        if config.gene_alias_table is not None
        else None
    )
    return GeneIdSources(
        species=species,
        gene_tables=tables,
        pair_lookup=dict(pair_lookup or {}),
        aliases=aliases,
        overrides=load_overrides(species, config.gene_id_overrides_csv),
    )


# --------------------------------------------------------------------------
# Results


class _GeneIdModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolutionRules(_GeneIdModel):
    """The refusal thresholds of plan §8.2 / §8.4.

    Attributes:
        min_gene_id_resolution: Share of non-control features that must
            resolve.
        min_native_prefix_share: Share of native ID values that must be
            Ensembl gene IDs of the run species.
        max_other_species_prefix_share: Share of native ID values with
            another species' prefix above which the panel is refused.
        species_exact_case_min_ratio: Exact-case / case-insensitive match
            ratio the run species needs.
    """

    min_gene_id_resolution: float = 0.95
    min_native_prefix_share: float = 0.95
    max_other_species_prefix_share: float = 0.05
    species_exact_case_min_ratio: float = 0.5

    @classmethod
    def from_config(cls, config: AnnotationPanelConfig) -> ResolutionRules:
        """Return the rules of a panel config."""
        return cls(
            min_gene_id_resolution=config.min_gene_id_resolution,
            min_native_prefix_share=config.min_native_prefix_share,
            max_other_species_prefix_share=config.max_other_species_prefix_share,
            species_exact_case_min_ratio=config.species_exact_case_min_ratio,
        )


class SpeciesCheck(_GeneIdModel):
    """The exact-case species test (plan §8.4).

    Attributes:
        species: Run species.
        status: ``pass``, ``species_mismatch`` or ``not_evaluable`` (no run
            species table, or no symbol in any table).
        reasons: Why it failed or could not be evaluated.
        n_symbols: Distinct panel symbols tested (Ensembl IDs excluded).
        exact_matches: Symbols matching each species' table in exact case.
        casefold_matches: Symbols matching each species' table in any case.
        exact_case_ratio: Run species exact / case-insensitive matches.
        min_ratio: ``species_exact_case_min_ratio``.
        tables: The table path per species (``None`` if missing).
    """

    species: Species
    status: SpeciesCheckStatus
    reasons: list[str] = Field(default_factory=list)
    n_symbols: int
    exact_matches: dict[str, int]
    casefold_matches: dict[str, int]
    exact_case_ratio: float | None
    min_ratio: float
    tables: dict[str, str | None]


def species_check(
    symbols: Iterable[str],
    species: Species,
    gene_tables: Mapping[str, GeneTable],
    *,
    min_ratio: float = 0.5,
) -> SpeciesCheck:
    """Test whether panel symbols belong to the run species (exact case).

    Args:
        symbols: The panel's non-control feature symbols.
        species: Run species.
        gene_tables: Local gene tables by species (both, ideally).
        min_ratio: The run species' exact matches must be at least this
            share of its case-insensitive matches.

    Returns:
        The test result.
    """
    distinct = sorted(
        {
            text
            for text in (clean_text(symbol) for symbol in symbols)
            if text and not is_ensembl_gene_id(strip_version(text))
        }
    )
    exact: dict[str, int] = {}
    casefold: dict[str, int] = {}
    for table_species, table in sorted(gene_tables.items()):
        exact[table_species] = sum(1 for s in distinct if table.has_exact(s))
        casefold[table_species] = sum(1 for s in distinct if table.has_casefold(s))
    other = other_species(species)
    run_table = gene_tables.get(species)
    reasons: list[str] = []
    ratio: float | None = None
    status: SpeciesCheckStatus = "pass"
    if run_table is None:
        status = "not_evaluable"
        reasons.append(f"no {species} gene table")
    else:
        run_exact, run_casefold = exact[species], casefold[species]
        ratio = run_exact / run_casefold if run_casefold else None
        other_exact = exact.get(other)
        if other_exact is not None and other_exact > run_exact:
            status = "species_mismatch"
            reasons.append(
                f"{other_exact} symbols match the {other} gene table in exact case "
                f"vs {run_exact} for {species}"
            )
        if ratio is not None and ratio < min_ratio:
            status = "species_mismatch"
            reasons.append(
                f"only {run_exact} of {run_casefold} {species} symbol matches are "
                f"exact-case (ratio {ratio:.3f} < {min_ratio})"
            )
        if status == "pass" and run_casefold == 0 and not other_exact:
            status = "not_evaluable"
            reasons.append("no panel symbol is in the gene tables")
    if other not in gene_tables and status == "pass":
        reasons.append(f"no {other} gene table: only the exact-case ratio was tested")
    return SpeciesCheck(
        species=species,
        status=status,
        reasons=reasons,
        n_symbols=len(distinct),
        exact_matches=exact,
        casefold_matches=casefold,
        exact_case_ratio=None if ratio is None else round(ratio, 6),
        min_ratio=min_ratio,
        tables={
            table_species: (
                None
                if table_species not in gene_tables
                else gene_tables[table_species].path
            )
            for table_species in SPECIES
        },
    )


class FeatureResolution(_GeneIdModel):
    """How one non-control feature was resolved.

    Attributes:
        name: Feature name.
        symbol: Feature symbol.
        native_value: Unversioned value of the native ID column (``""``).
        native_kind: ``native_kind`` of that value.
        gene_id: Resolved ID (``None`` when unresolved).
        source: Resolution source.
        reason: Why the feature is unresolved (``""`` when resolved).
        detail: Extra detail (the alias target, the override reason, the
            replaced native ID of a release-drift fix).
    """

    name: str
    symbol: str
    native_value: str = ""
    native_kind: NativeKind = "none"
    gene_id: str | None = None
    source: GeneIdSource | None = None
    reason: str = ""
    detail: str = ""


class GeneIdResolution(_GeneIdModel):
    """``resolve_gene_ids`` output: per-feature IDs, checks and refusals.

    Attributes:
        species: Run species.
        status: ``ok`` or ``refused``.
        refusal_reasons: Why the panel is refused.
        refusal_details: A sentence per refusal reason.
        features: One row per non-control feature, in input order.
        species_check: The exact-case species test.
        native_id_column: The ``var`` column the native values came from.
        n_native_values: Native values that count for the prefix rules
            (Ensembl gene IDs of any species and non-ID values).
        native_prefix_share: Share of them with the run species' prefix.
        other_species_share: Share of them that are another species' IDs.
        symbols_as_ids: Native values that are the feature's own symbol.
        release_drift: Native (or pair-lookup) ID -> the gene table's ID.
        sources: The resolver sources' provenance.
    """

    species: Species
    status: Literal["ok", "refused"]
    refusal_reasons: list[RefusalReason] = Field(default_factory=list)
    refusal_details: dict[str, str] = Field(default_factory=dict)
    features: list[FeatureResolution]
    species_check: SpeciesCheck
    native_id_column: str | None = None
    n_native_values: int = 0
    native_prefix_share: float | None = None
    other_species_share: float | None = None
    symbols_as_ids: list[str] = Field(default_factory=list)
    release_drift: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Return the number of non-control features."""
        return len(self.features)

    @property
    def n_resolved(self) -> int:
        """Return the number of resolved features."""
        return sum(1 for feature in self.features if feature.gene_id)

    @property
    def resolution_share(self) -> float:
        """Return the resolved share of non-control features (0 if none)."""
        return self.n_resolved / self.n_features if self.features else 0.0

    @property
    def gene_ids(self) -> list[str]:
        """Return the sorted distinct resolved IDs."""
        return sorted({f.gene_id for f in self.features if f.gene_id})

    def ids_by_name(self) -> dict[str, str]:
        """Return the resolved ID per feature name (resolved features only)."""
        return {f.name: f.gene_id for f in self.features if f.gene_id}

    def source_counts(self) -> dict[str, int]:
        """Return the number of resolved features per source."""
        counts: dict[str, int] = {}
        for feature in self.features:
            if feature.source is not None:
                counts[feature.source] = counts.get(feature.source, 0) + 1
        return dict(sorted(counts.items()))

    def unmapped(self) -> dict[str, str]:
        """Return the reason per unresolved feature name."""
        return {f.name: f.reason for f in self.features if not f.gene_id}

    def merged_duplicates(self) -> dict[str, list[str]]:
        """Return the features merged into one ID (ID -> names, input order)."""
        names: dict[str, list[str]] = {}
        for feature in self.features:
            if feature.gene_id:
                names.setdefault(feature.gene_id, []).append(feature.name)
        return {
            gene_id: found for gene_id, found in sorted(names.items()) if len(found) > 1
        }

    def table_sha256(self) -> str:
        """Return the sha256 of the resolution table (name, symbol, ID, source).

        Returns:
            64 lower-case hex characters of the canonical JSON rows, sorted
            by feature name.
        """
        rows = sorted(
            [f.name, f.symbol, f.gene_id or "", f.source or "", f.reason]
            for f in self.features
        )
        text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        """Return the ``panel_report.json`` summary of the resolution."""
        return {
            "status": self.status,
            "refusal_reasons": list(self.refusal_reasons),
            "refusal_details": dict(self.refusal_details),
            "n_features": self.n_features,
            "n_resolved": self.n_resolved,
            "resolution_share": round(self.resolution_share, 6),
            "by_source": self.source_counts(),
            "unmapped": self.unmapped(),
            "merged_duplicates": self.merged_duplicates(),
            "native_id_column": self.native_id_column,
            "n_native_values": self.n_native_values,
            "native_prefix_share": self.native_prefix_share,
            "other_species_share": self.other_species_share,
            "symbols_as_ids": list(self.symbols_as_ids),
            "release_drift": dict(self.release_drift),
            "species_check": self.species_check.model_dump(mode="json"),
            "resolution_table_sha256": self.table_sha256(),
            "sources": self.sources,
        }


@dataclass(frozen=True)
class FeatureInput:
    """One non-control feature handed to the resolver.

    Attributes:
        name: Feature name (``var`` index).
        symbol: Gene symbol (the name when the source has no symbol).
        native_value: Raw value of the native ID column (``""`` if none).
    """

    name: str
    symbol: str
    native_value: str = ""


def _symbol_chain(
    symbol: str, sources: GeneIdSources
) -> tuple[str | None, GeneIdSource | None, str, str]:
    """Resolve a symbol through steps 2-5; return (id, source, reason, detail)."""
    pattern = SPECIES_ID_PATTERNS[sources.species]
    table = sources.run_table
    looked_up = sources.pair_lookup.get(symbol)
    if looked_up and pattern.fullmatch(looked_up):
        return looked_up, "pair_lookup", "", ""
    failure = "no_fallback_table"
    if table is not None:
        gene_id, failure = table.unique_id(symbol)
        if gene_id is not None:
            return gene_id, "fallback_table", "", ""
    if sources.aliases is not None:
        gene_id, detail, alias_failure = sources.aliases.resolve(symbol, table)
        if gene_id is not None:
            return gene_id, "alias", "", detail
        if alias_failure == "ambiguous_alias" and failure == "not_in_fallback_table":
            failure = alias_failure
    override = _override_for(sources.overrides, symbol)
    if override is not None:
        return override.ensembl_id, "override", "", override.reason
    return None, None, failure, ""


def _drift_fix(gene_id: str, symbol: str, table: GeneTable | None) -> tuple[str, bool]:
    """Apply the release-drift rule to a native or pair-lookup ID."""
    if table is None or gene_id in table.gene_ids:
        return gene_id, False
    replacement, _ = table.unique_id(symbol)
    if replacement is None or replacement == gene_id:
        return gene_id, False
    return replacement, True


def resolve_gene_ids(
    features: Sequence[FeatureInput] | pd.DataFrame,
    species: Species,
    sources: GeneIdSources,
    *,
    rules: ResolutionRules | None = None,
    native_id_column: str | None = None,
) -> GeneIdResolution:
    """Resolve the non-control features of a declared panel (plan §8.4).

    Args:
        features: Non-control features: ``FeatureInput`` rows, or a frame
            with columns ``name``, ``symbol`` and optional ``native_value``.
        species: Run species.
        sources: What the resolver may consult (``gene_id_sources``).
        rules: Refusal thresholds (default: the plan's).
        native_id_column: The native ID column, for the record.

    Returns:
        The resolution: per-feature IDs and sources, the species test and
        the refusal verdict.

    Raises:
        ValueError: If ``sources`` is for another species.
    """
    if sources.species != species:
        raise ValueError(f"resolver sources are for {sources.species}, not {species}")
    rules = rules or ResolutionRules()
    inputs = _feature_inputs(features)
    table = sources.run_table
    rows: list[FeatureResolution] = []
    drift: dict[str, str] = {}
    symbols_as_ids: list[str] = []
    counts = {"run": 0, "other_species": 0, "non_id": 0}
    for item in inputs:
        native = clean_native_value(item.native_value)
        kind = native_kind(native, species)
        if kind in counts:
            counts[kind] += 1
        if kind == "non_id" and native.casefold() == item.symbol.casefold():
            symbols_as_ids.append(item.name)
        if kind == "run":
            native_id, fixed = _drift_fix(native, item.symbol, table)
            if fixed:
                drift[native] = native_id
            rows.append(
                FeatureResolution(
                    name=item.name,
                    symbol=item.symbol,
                    native_value=native,
                    native_kind=kind,
                    gene_id=native_id,
                    source="symbol_fallback" if fixed else "native",
                    detail=f"native {native} absent from the gene table"
                    if fixed
                    else "",
                )
            )
            continue
        if kind == "other_species":
            rows.append(
                FeatureResolution(
                    name=item.name,
                    symbol=item.symbol,
                    native_value=native,
                    native_kind=kind,
                    reason="other_species_id",
                )
            )
            continue
        gene_id, source, reason, detail = _symbol_chain(item.symbol, sources)
        if source == "pair_lookup" and gene_id is not None:
            fixed_id, fixed = _drift_fix(gene_id, item.symbol, table)
            if fixed:
                drift[gene_id] = fixed_id
                gene_id, source = fixed_id, "symbol_fallback"
                detail = "pair-lookup ID absent from the gene table"
        rows.append(
            FeatureResolution(
                name=item.name,
                symbol=item.symbol,
                native_value=native,
                native_kind=kind,
                gene_id=gene_id,
                source=source,
                reason=reason,
                detail=detail,
            )
        )
    check = species_check(
        [item.symbol for item in inputs],
        species,
        sources.gene_tables,
        min_ratio=rules.species_exact_case_min_ratio,
    )
    n_native = sum(counts.values())
    prefix_share = counts["run"] / n_native if n_native else None
    other_share = counts["other_species"] / n_native if n_native else None
    result = GeneIdResolution(
        species=species,
        status="ok",
        features=rows,
        species_check=check,
        native_id_column=native_id_column,
        n_native_values=n_native,
        native_prefix_share=None if prefix_share is None else round(prefix_share, 6),
        other_species_share=None if other_share is None else round(other_share, 6),
        symbols_as_ids=symbols_as_ids,
        release_drift=dict(sorted(drift.items())),
        sources=sources.describe(),
    )
    reasons, details = _refusals(result, rules, prefix_share, other_share)
    if reasons:
        result = result.model_copy(
            update={
                "status": "refused",
                "refusal_reasons": reasons,
                "refusal_details": details,
            }
        )
        logger.warning(
            "Gene-ID resolution refused (%s): %s",
            species,
            "; ".join(details[reason] for reason in reasons),
        )
    return result


def _refusals(
    result: GeneIdResolution,
    rules: ResolutionRules,
    prefix_share: float | None,
    other_share: float | None,
) -> tuple[list[RefusalReason], dict[str, str]]:
    reasons: list[RefusalReason] = []
    details: dict[str, str] = {}
    if not result.features:
        reasons.append("no_features")
        details["no_features"] = "the declared panel has no non-control feature"
        return reasons, details
    if result.species_check.status == "species_mismatch":
        reasons.append("species_mismatch")
        details["species_mismatch"] = "; ".join(result.species_check.reasons)
    if prefix_share is not None and prefix_share < rules.min_native_prefix_share:
        reasons.append("native_id_prefix")
        text = (
            f"{prefix_share:.3f} of {result.n_native_values} native ID values are "
            f"{result.species} Ensembl gene IDs (< {rules.min_native_prefix_share})"
        )
        if result.symbols_as_ids:
            text += (
                f"; {len(result.symbols_as_ids)} hold the feature's symbol "
                f"(symbols as IDs, e.g. {', '.join(result.symbols_as_ids[:5])})"
            )
        details["native_id_prefix"] = text
    if other_share is not None and other_share > rules.max_other_species_prefix_share:
        reasons.append("other_species_ids")
        details["other_species_ids"] = (
            f"{other_share:.3f} of the native ID values carry another species' "
            f"prefix (> {rules.max_other_species_prefix_share})"
        )
    share = result.resolution_share
    if share < rules.min_gene_id_resolution:
        reasons.append("gene_id_resolution")
        details["gene_id_resolution"] = (
            f"{result.n_resolved} of {result.n_features} non-control features "
            f"resolve to {result.species} Ensembl IDs ({share:.3f} < "
            f"{rules.min_gene_id_resolution})"
        )
    return reasons, details


def _feature_inputs(
    features: Sequence[FeatureInput] | pd.DataFrame,
) -> list[FeatureInput]:
    if isinstance(features, pd.DataFrame):
        native = (
            features["native_value"].tolist()
            if "native_value" in features.columns
            else [""] * len(features)
        )
        return [
            FeatureInput(
                name=clean_text(name),
                symbol=clean_text(symbol) or clean_text(name),
                native_value=clean_text(value),
            )
            for name, symbol, value in zip(
                features["name"], features["symbol"], native, strict=True
            )
        ]
    return list(features)


def summing_matrix(
    feature_ids: Sequence[str], target_ids: Sequence[str], dtype: Any = np.float32
) -> Any:
    """Return the features x targets 0/1 matrix that sums duplicate features.

    ``counts @ summing_matrix(...)`` gives one column per target ID, the sum
    of every feature resolved to it (plan §8.4: duplicates are summed).

    Args:
        feature_ids: Resolved ID per feature (``""`` when unresolved).
        target_ids: The target IDs (e.g. a panel's genes present in data).
        dtype: Matrix dtype (the counts' dtype).

    Returns:
        A ``scipy.sparse.csr_matrix`` of shape
        ``(len(feature_ids), len(target_ids))``.
    """
    from scipy import sparse

    column = {gene_id: position for position, gene_id in enumerate(target_ids)}
    rows = [i for i, gene_id in enumerate(feature_ids) if gene_id in column]
    cols = [column[feature_ids[i]] for i in rows]
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=dtype), (rows, cols)),
        shape=(len(feature_ids), len(target_ids)),
    )
