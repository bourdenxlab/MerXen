"""Cell-type vocabularies for reference-based annotation.

The packaged tables under ``merxen/assets/annotation/`` map each reference node
to MerXen's species vocabularies:

- ``whb_supercluster_vocab.csv``: the 31 Allen WHB superclusters (human primary).
- ``seaad_mr_subclass_vocab.csv``: the 29 SEA-AD Multiregion subclasses (human
  second vote), plus supertype overrides that split "VLMC & Perivascular".
- ``wmb_class_vocab.csv``: the 34 Allen WMB classes (mouse primary).

``scripts/annotation/build_vocab_tables.py`` writes these tables from the Allen
taxonomy CSVs and ``overrides.yaml``; they are committed and reviewed, never
edited by hand.

Species vocabularies stay separate. Human uses seven broad classes plus
``Oligodendrocyte lineage`` as the lineage-level fallback; mouse uses the six
legacy WMB broad classes. Both use ``UNASSIGNED_LABEL``.

This module imports only the standard library and pandas (with its numpy),
because the GPU clustering environment imports it (it has neither
``cell_type_mapper`` nor SpatialData).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Literal, cast

import pandas as pd

logger = logging.getLogger(__name__)

ASSET_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "assets" / "annotation"

Species = Literal["human", "mouse"]
SPECIES: Final[tuple[str, ...]] = ("human", "mouse")

UNASSIGNED_LABEL: Final = "Mixed/Unknown"
UNRESOLVED_LABEL: Final = "unresolved"
NEURONS: Final = "Neurons"
OLIGODENDROCYTE_LINEAGE: Final = "Oligodendrocyte lineage"

HUMAN_BROAD_CLASSES: Final[tuple[str, ...]] = (
    NEURONS,
    "Astrocytes",
    "Oligodendrocytes",
    "Oligodendrocyte precursors",
    "Microglia",
    "Vascular cells",
    "Fibroblasts",
)
# The legacy WMB collapse (``collapse_atlas_label_to_broad_class``) and MECR's
# WMB ``target_broad_classes`` use these names, so map_first keeps them.
MOUSE_BROAD_CLASSES: Final[tuple[str, ...]] = (
    NEURONS,
    "Astrocytes/Ependymal",
    OLIGODENDROCYTE_LINEAGE,
    "OEC",
    "Vascular cells",
    "Microglia",
)
BROAD_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "human": HUMAN_BROAD_CLASSES,
    "mouse": MOUSE_BROAD_CLASSES,
}
HUMAN_LINEAGES: Final[tuple[str, ...]] = (
    NEURONS,
    OLIGODENDROCYTE_LINEAGE,
    "Astrocytes",
    "Microglia",
    "Vascular cells",
    "Fibroblasts",
)
HUMAN_LINEAGE_OF_BROAD_CLASS: Final[dict[str, str]] = {
    NEURONS: NEURONS,
    "Astrocytes": "Astrocytes",
    "Oligodendrocytes": OLIGODENDROCYTE_LINEAGE,
    "Oligodendrocyte precursors": OLIGODENDROCYTE_LINEAGE,
    "Microglia": "Microglia",
    "Vascular cells": "Vascular cells",
    "Fibroblasts": "Fibroblasts",
}
# Lineages that ``broad_class`` falls back to when a cell is confident only at
# lineage (§4.5); any other lineage gives ``UNASSIGNED_LABEL``.
MAP_FIRST_LINEAGE_FALLBACKS: Final[tuple[str, ...]] = (OLIGODENDROCYTE_LINEAGE, NEURONS)
# Labels map_first may write to the legacy ``broad_class`` column (§4.5): the
# species broad classes, the lineage-level fallbacks and ``UNASSIGNED_LABEL``.
MAP_FIRST_BROAD_LABELS: Final[dict[str, tuple[str, ...]]] = {
    "human": (*HUMAN_BROAD_CLASSES, OLIGODENDROCYTE_LINEAGE, UNASSIGNED_LABEL),
    "mouse": (*MOUSE_BROAD_CLASSES, UNASSIGNED_LABEL),
}

NT_EXCITATORY: Final = "Excitatory"
NT_INHIBITORY: Final = "Inhibitory"
NT_OTHER: Final = "Other"
NT_CLASSES: Final[tuple[str, ...]] = (NT_EXCITATORY, NT_INHIBITORY, NT_OTHER)

# Annotation levels per species, coarse to fine, as used by ``ct_final_level``
# (``none`` = no confident level). Report-only fine levels never enter it.
FINAL_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "human": ("none", "lineage", "broad", "nt", "supercluster"),
    "mouse": ("none", "broad", "class", "nt", "subclass"),
}

VocabTableId = Literal["whb_supercluster", "seaad_mr_subclass", "wmb_class"]
VOCAB_FILES: Final[dict[str, str]] = {
    "whb_supercluster": "whb_supercluster_vocab.csv",
    "seaad_mr_subclass": "seaad_mr_subclass_vocab.csv",
    "wmb_class": "wmb_class_vocab.csv",
}
VOCAB_SPECIES: Final[dict[str, str]] = {
    "whb_supercluster": "human",
    "seaad_mr_subclass": "human",
    "wmb_class": "mouse",
}
VOCAB_KEY_COLUMNS: Final[dict[str, str]] = {
    "whb_supercluster": "name",
    "seaad_mr_subclass": "subclass",
    "wmb_class": "class",
}
PRIMARY_VOCAB: Final[dict[str, VocabTableId]] = {
    "human": "whb_supercluster",
    "mouse": "wmb_class",
}
REGION_COLUMN_PREFIX: Final = "region_plausible_"
SUPERTYPE_PREFIX_COLUMN: Final = "supertype_prefix"
_BOOL_COLUMNS: Final[tuple[str, ...]] = ("sink", "never_drop")

# Floor classes (§5.4). Human floors follow the E2 eight-class scheme: neurons
# split by NT, COP separate at supercluster level only.
COP_SUPERCLUSTER: Final = "Committed oligodendrocyte precursor"
HUMAN_FLOOR_CLASSES: Final[tuple[str, ...]] = (
    "Exc",
    "Inh",
    "OtherNeuron",
    "Astro",
    "Oligo",
    "OPC",
    "COP",
    "Immune",
    "Vascular",
    "Fibroblast",
)
_HUMAN_NEURON_FLOOR_CLASS: Final[dict[str, str]] = {
    NT_EXCITATORY: "Exc",
    NT_INHIBITORY: "Inh",
    NT_OTHER: "OtherNeuron",
}
_HUMAN_GLIA_FLOOR_CLASS: Final[dict[str, str]] = {
    "Astrocytes": "Astro",
    "Oligodendrocytes": "Oligo",
    "Oligodendrocyte precursors": "OPC",
    "Microglia": "Immune",
    "Vascular cells": "Vascular",
    "Fibroblasts": "Fibroblast",
}
FLOOR_FILES: Final[dict[str, str]] = {
    "human": "floors_human.csv",
    "mouse": "floors_mouse.csv",
}
FLOOR_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "floor_class",
    "platform",
    "panel_family",
    "min_counts",
    "floor_source",
    "inherited_from",
    "note",
)
STATE_GENE_FILES: Final[dict[str, str]] = {
    "human": "state_genes_human.csv",
    "mouse": "state_genes_mouse.csv",
}
HELDOUT_MARKER_FILES: Final[dict[str, str]] = {
    "human": "heldout_markers_human.csv",
    "mouse": "heldout_markers_mouse.csv",
}


def _check_species(species: str) -> Species:
    if species not in SPECIES:
        raise ValueError(f"species must be one of {SPECIES}, got {species!r}")
    return cast(Species, species)


def _is_missing(value: object) -> bool:
    """Return whether a scalar is missing: None, any NaN or NA, or blank text.

    Covers ``pd.NA`` (nullable string and categorical columns), numpy float
    NaNs of any width and ``NaT`` as well as Python ``None`` / ``nan``.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Array-likes are not scalars and so not missing values.
        return False


def asset_path(filename: str) -> Path:
    """Return the path of a packaged annotation asset.

    Args:
        filename: File name under ``merxen/assets/annotation/``.

    Returns:
        The absolute path to the asset.

    Raises:
        FileNotFoundError: If the asset is not packaged.
    """
    path = ASSET_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Packaged annotation asset not found: {path}")
    return path


def load_asset_table(filename: str) -> pd.DataFrame:
    """Read a packaged annotation CSV with every column as a string.

    Empty fields stay empty strings; no value is turned into NaN, so labels
    such as ``"NA"`` survive.

    Args:
        filename: CSV file name under ``merxen/assets/annotation/``.

    Returns:
        A fresh dataframe (safe to modify).
    """
    return _read_asset_csv(filename).copy()


@cache
def _read_asset_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(asset_path(filename), dtype=str, keep_default_na=False)


def _parse_bool_column(frame: pd.DataFrame, column: str, filename: str) -> None:
    values = frame[column].str.strip().str.lower()
    invalid = sorted(set(values) - {"true", "false"})
    if invalid:
        raise ValueError(f"{filename}: column {column!r} has non-boolean {invalid}")
    frame[column] = values == "true"


def classify_neurotransmitter(term: str | None) -> str | None:
    """Map an Allen neurotransmitter annotation to MerXen's NT classes.

    Uses the rule of the legacy ``_neuron_split_for_neurotransmitter``: GABA or
    glycine without glutamate is inhibitory, glutamate without GABA or glycine
    is excitatory, and anything else (co-release, monoamines, choline) is
    ``Other``. It covers WHB (``"VGLUT1 VGLUT2"``, ``"GABA VGLUT3"``) and WMB
    (``"Glut"``, ``"GABA-Glyc"``, ``"Dopa"``) terms.

    Args:
        term: Neurotransmitter term, or ``None`` / NaN / blank when unannotated.

    Returns:
        ``"Excitatory"``, ``"Inhibitory"`` or ``"Other"``; ``None`` when the
        term is missing.
    """
    if _is_missing(term):
        return None
    label = str(term).upper()
    has_inhibitory = "GABA" in label or "GLY" in label
    has_excitatory = "GLUT" in label
    if has_inhibitory and not has_excitatory:
        return NT_INHIBITORY
    if has_excitatory and not has_inhibitory:
        return NT_EXCITATORY
    return NT_OTHER


@dataclass(frozen=True, eq=False)
class VocabTable:
    """One packaged vocabulary table, indexed by reference node name.

    Attributes:
        table_id: Registry id, e.g. ``"whb_supercluster"``.
        species: ``"human"`` or ``"mouse"``.
        key_column: Column holding the node name (``name``, ``subclass`` or
            ``class``).
        frame: One row per node, indexed by node name; boolean columns
            (``sink``, ``never_drop``, ``region_plausible_*``) are parsed.
        supertype_overrides: Rows that override a node's classes by supertype
            prefix (SEA-AD "VLMC & Perivascular"); empty for other tables.
    """

    table_id: str
    species: str
    key_column: str
    frame: pd.DataFrame
    supertype_overrides: pd.DataFrame

    @property
    def names(self) -> tuple[str, ...]:
        """Node names in table order."""
        return tuple(str(name) for name in self.frame.index)

    @property
    def regions(self) -> tuple[str, ...]:
        """Anatomical regions with a plausibility column in this table."""
        return tuple(
            column.removeprefix(REGION_COLUMN_PREFIX)
            for column in self.frame.columns
            if column.startswith(REGION_COLUMN_PREFIX)
        )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.frame.index

    def row(self, name: str) -> pd.Series:
        """Return the table row of one node.

        Args:
            name: Node name (the table's key column).

        Returns:
            The row as a series.

        Raises:
            KeyError: If the node is not in the table.
        """
        if name not in self.frame.index:
            raise KeyError(f"{name!r} is not a node of vocab table {self.table_id!r}")
        return self.frame.loc[name]

    def label_to_name(self) -> dict[str, str]:
        """Return the Allen term label (e.g. ``CS202210140_476``) to name map."""
        if "label" not in self.frame.columns:
            return {}
        return {
            str(label): str(name)
            for name, label in self.frame["label"].items()
            if str(label)
        }

    def _override_row(self, name: str, supertype: str | None) -> pd.Series | None:
        if _is_missing(supertype) or self.supertype_overrides.empty:
            return None
        candidates = self.supertype_overrides[
            self.supertype_overrides[self.key_column] == name
        ]
        for _, candidate in candidates.iterrows():
            if str(supertype).startswith(str(candidate[SUPERTYPE_PREFIX_COLUMN])):
                return candidate
        return None

    def _value(self, name: str, column: str, supertype: str | None) -> str | None:
        if column not in self.frame.columns:
            raise KeyError(f"vocab table {self.table_id!r} has no {column!r} column")
        override = self._override_row(name, supertype)
        value = override[column] if override is not None else self.row(name)[column]
        return None if _is_missing(value) else str(value)

    def broad_class(self, name: str, supertype: str | None = None) -> str:
        """Return the species broad class of a node.

        Args:
            name: Node name.
            supertype: Optional supertype name; applies the table's supertype
                overrides (SEA-AD "VLMC & Perivascular").

        Returns:
            A species broad class, ``Oligodendrocyte lineage`` (mouse) or
            ``UNASSIGNED_LABEL`` for sinks and nodes mapped to no broad class.

        Raises:
            KeyError: If the node is not in the table.
        """
        return self._value(name, "broad_class", supertype) or UNASSIGNED_LABEL

    def lineage(self, name: str, supertype: str | None = None) -> str:
        """Return the lineage of a node (human tables only).

        Args:
            name: Node name.
            supertype: Optional supertype name for the supertype overrides.

        Returns:
            A human lineage or ``UNASSIGNED_LABEL``.

        Raises:
            KeyError: If the table has no lineage column (mouse) or the node
                is unknown.
        """
        return self._value(name, "lineage", supertype) or UNASSIGNED_LABEL

    def nt(self, name: str, supertype: str | None = None) -> str | None:
        """Return the NT class of a node, or ``None`` for non-neurons.

        Args:
            name: Node name.
            supertype: Optional supertype name for the supertype overrides.

        Returns:
            ``"Excitatory"``, ``"Inhibitory"``, ``"Other"`` or ``None``.
        """
        return self._value(name, "nt", supertype)

    def is_sink(self, name: str) -> bool:
        """Return whether a node is a sink (a mixed-identity attractor).

        Args:
            name: Node name.

        Returns:
            ``True`` for sink nodes; ``False`` for tables without sinks.
        """
        if "sink" not in self.frame.columns:
            self.row(name)
            return False
        return bool(self.row(name)["sink"])

    def is_region_plausible(self, name: str, region: str) -> bool:
        """Return whether a node is plausible in an anatomical region.

        Args:
            name: Node name.
            region: Region token, e.g. ``"frontal_cortex"``.

        Returns:
            The table's ``region_plausible_<region>`` value.

        Raises:
            ValueError: If the table has no column for the region.
        """
        column = f"{REGION_COLUMN_PREFIX}{region}"
        if column not in self.frame.columns:
            raise ValueError(
                f"vocab table {self.table_id!r} has no plausibility column for "
                f"region {region!r} (available: {self.regions}); only frontal "
                "cortex is validated for human (OD-C7, plan §8.9)"
            )
        return bool(self.row(name)[column])

    def is_never_drop(self, name: str) -> bool:
        """Return whether mouse region pruning must keep a node.

        Args:
            name: Node name.

        Returns:
            The ``never_drop`` value; ``False`` for tables without it.
        """
        if "never_drop" not in self.frame.columns:
            self.row(name)
            return False
        return bool(self.row(name)["never_drop"])


def _build_vocab_table(table_id: str) -> VocabTable:
    filename = VOCAB_FILES[table_id]
    key_column = VOCAB_KEY_COLUMNS[table_id]
    frame = load_asset_table(filename)
    if key_column not in frame.columns:
        raise ValueError(f"{filename}: missing key column {key_column!r}")
    for column in frame.columns:
        if column in _BOOL_COLUMNS or column.startswith(REGION_COLUMN_PREFIX):
            _parse_bool_column(frame, column, filename)
    if SUPERTYPE_PREFIX_COLUMN in frame.columns:
        is_override = frame[SUPERTYPE_PREFIX_COLUMN].str.strip() != ""
        overrides = frame.loc[is_override].reset_index(drop=True)
        frame = frame.loc[~is_override]
    else:
        overrides = frame.iloc[0:0]
    duplicated = frame[key_column][frame[key_column].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{filename}: duplicated {key_column} values {duplicated}")
    unknown = sorted(set(overrides[key_column]) - set(frame[key_column]))
    if unknown:
        raise ValueError(f"{filename}: overrides for unknown nodes {unknown}")
    return VocabTable(
        table_id=table_id,
        species=VOCAB_SPECIES[table_id],
        key_column=key_column,
        frame=frame.set_index(key_column, drop=False).rename_axis(None),
        supertype_overrides=overrides,
    )


@cache
def load_vocab(table_id: VocabTableId) -> VocabTable:
    """Load one packaged vocabulary table (cached).

    Args:
        table_id: ``"whb_supercluster"``, ``"seaad_mr_subclass"`` or
            ``"wmb_class"``.

    Returns:
        The parsed table. Treat it as read-only; it is shared between calls.

    Raises:
        ValueError: If the id is unknown or the table is malformed.
    """
    if table_id not in VOCAB_FILES:
        raise ValueError(f"table_id must be one of {tuple(VOCAB_FILES)}")
    return _build_vocab_table(table_id)


def primary_vocab(species: Species) -> VocabTable:
    """Return the vocabulary of a species' primary reference.

    Args:
        species: ``"human"`` (WHB superclusters) or ``"mouse"`` (WMB classes).

    Returns:
        The primary vocabulary table.
    """
    return load_vocab(PRIMARY_VOCAB[_check_species(species)])


def lineage_for(name: str, species: Species = "human") -> str:
    """Return the lineage of a primary-reference node.

    Args:
        name: WHB supercluster name (human).
        species: Only ``"human"`` has a lineage level.

    Returns:
        A human lineage or ``UNASSIGNED_LABEL``.

    Raises:
        ValueError: For mouse, which has no lineage level.
    """
    if _check_species(species) == "mouse":
        raise ValueError("mouse annotation has no lineage level")
    return primary_vocab(species).lineage(name)


def nt_for(name: str, species: Species = "human") -> str | None:
    """Return the NT class of a primary-reference node.

    Args:
        name: WHB supercluster name (human) or WMB class name (mouse).
        species: ``"human"`` or ``"mouse"``.

    Returns:
        ``"Excitatory"``, ``"Inhibitory"``, ``"Other"``, or ``None`` for
        non-neuronal nodes.
    """
    return primary_vocab(species).nt(name)


def broad_class_for(name: str, species: Species = "human") -> str:
    """Return the vocabulary broad class of a primary-reference node.

    Args:
        name: WHB supercluster name (human) or WMB class name (mouse).
        species: ``"human"`` or ``"mouse"``.

    Returns:
        A species broad class or ``UNASSIGNED_LABEL``.
    """
    return primary_vocab(species).broad_class(name)


def is_sink(name: str, species: Species = "human") -> bool:
    """Return whether a primary-reference node is a sink.

    Args:
        name: WHB supercluster name (human) or WMB class name (mouse).
        species: ``"human"`` or ``"mouse"``.

    Returns:
        ``True`` for WHB Splatter and Miscellaneous; mouse has no sinks.
    """
    return primary_vocab(species).is_sink(name)


def is_region_plausible(name: str, region: str, species: Species = "human") -> bool:
    """Return whether a human primary-reference node is plausible in a region.

    Args:
        name: WHB supercluster name.
        region: Anatomical region token, e.g. ``"frontal_cortex"``.
        species: Only ``"human"`` has static region plausibility; mouse
            plausibility is decided per section by region pruning.

    Returns:
        ``True`` if the node is plausible in the region.

    Raises:
        ValueError: For mouse, or a region without a vocab column.
    """
    if _check_species(species) == "mouse":
        raise ValueError(
            "mouse plausibility is decided per section by region pruning, "
            "not by a vocab column"
        )
    return primary_vocab(species).is_region_plausible(name, region)


def seaad_broad_class(subclass: str, supertype: str | None = None) -> str:
    """Return the human broad class of a SEA-AD Multiregion call.

    Args:
        subclass: SEA-AD subclass name.
        supertype: SEA-AD supertype name; required to split
            "VLMC & Perivascular" into Fibroblasts (VLMC) and Vascular cells
            (Pericyte, SMC).

    Returns:
        A human broad class, or ``UNASSIGNED_LABEL`` (Ependymal, and
        "VLMC & Perivascular" without a supertype).
    """
    return load_vocab("seaad_mr_subclass").broad_class(subclass, supertype)


def _map_first_label(candidate: object, species: Species) -> str:
    if _is_missing(candidate):
        return UNASSIGNED_LABEL
    label = str(candidate)
    vocab = primary_vocab(species)
    # A node name (not a vocabulary label) goes through the vocab, so sinks
    # and nodes outside the vocabulary can never leak into ``broad_class``.
    if label in vocab and label not in MAP_FIRST_BROAD_LABELS[species]:
        if vocab.is_sink(label):
            return UNASSIGNED_LABEL
        label = vocab.broad_class(label)
    if label in MAP_FIRST_BROAD_LABELS[species]:
        return label
    logger.debug("%r is not a map_first broad label; using %s", label, UNASSIGNED_LABEL)
    return UNASSIGNED_LABEL


def broad_class_for_map_first(
    final_level: str,
    broad_name: str | None,
    lineage_name: str | None = None,
    *,
    species: Species,
) -> str:
    """Return the legacy ``broad_class`` value of one cell in map_first mode.

    Implements §4.5: the confident broad name when the final level is broad
    or deeper; when the final level is lineage, the lineage only if it is
    ``Oligodendrocyte lineage`` or ``Neurons`` (the two lineage fallbacks
    §4.5 allows; a single-class lineage whose broad level is not confident
    gives ``UNASSIGNED_LABEL``); ``UNASSIGNED_LABEL`` otherwise. The result
    is never a sink or any label outside ``MAP_FIRST_BROAD_LABELS[species]``.

    Args:
        final_level: ``ct_final_level`` of the cell (``FINAL_LEVELS``).
        broad_name: ``ct_broad_name`` of the cell.
        lineage_name: ``ct_lineage_name`` of the cell (human only).
        species: ``"human"`` or ``"mouse"``.

    Returns:
        A label from ``MAP_FIRST_BROAD_LABELS[species]``.

    Raises:
        ValueError: If ``final_level`` is not a level of the species.
    """
    levels = FINAL_LEVELS[_check_species(species)]
    if final_level not in levels:
        raise ValueError(
            f"final_level {final_level!r} is not a {species} level {levels}"
        )
    rank = levels.index(final_level)
    if rank >= levels.index("broad"):
        return _map_first_label(broad_name, species)
    if final_level == "lineage":
        lineage = _map_first_label(lineage_name, species)
        return lineage if lineage in MAP_FIRST_LINEAGE_FALLBACKS else UNASSIGNED_LABEL
    return UNASSIGNED_LABEL


def broad_classes_for_map_first(
    final_level: Iterable[str],
    broad_name: Iterable[str | None],
    lineage_name: Iterable[str | None] | None = None,
    *,
    species: Species,
) -> pd.Series:
    """Vectorised ``broad_class_for_map_first`` over a label table.

    Args:
        final_level: ``ct_final_level`` values.
        broad_name: ``ct_broad_name`` values, aligned with ``final_level``.
        lineage_name: ``ct_lineage_name`` values (human), or ``None``.
        species: ``"human"`` or ``"mouse"``.

    Returns:
        A categorical series with categories
        ``MAP_FIRST_BROAD_LABELS[species]``, in input order. When
        ``final_level`` is a series its index is kept.
    """
    index = final_level.index if isinstance(final_level, pd.Series) else None
    levels = [str(value) for value in final_level]
    broads = [None if _is_missing(value) else str(value) for value in broad_name]
    if lineage_name is None:
        lineages: list[str | None] = [None] * len(levels)
    else:
        lineages = [
            None if _is_missing(value) else str(value) for value in lineage_name
        ]
    if not len(levels) == len(broads) == len(lineages):
        raise ValueError("final_level, broad_name and lineage_name lengths differ")
    cache: dict[tuple[str, str | None, str | None], str] = {}
    labels = []
    for key in zip(levels, broads, lineages, strict=True):
        if key not in cache:
            cache[key] = broad_class_for_map_first(
                key[0], key[1], key[2], species=species
            )
        labels.append(cache[key])
    return pd.Series(
        pd.Categorical(labels, categories=list(MAP_FIRST_BROAD_LABELS[species])),
        index=index,
        name="broad_class",
    )


def floor_class_for(name: str, *, species: Species, level: str = "broad") -> str | None:
    """Return the floor class that keys a node's count floor (§5.4).

    Human floors use the E2 classes: neurons by NT (``Exc``, ``Inh``,
    ``OtherNeuron``), glia by broad class, and ``COP`` separately at
    supercluster level (at broad level COP is ``OPC``). Mouse floors are keyed
    by WMB class.

    Args:
        name: WHB supercluster name (human) or WMB class name (mouse).
        species: ``"human"`` or ``"mouse"``.
        level: Annotation level of the floor; only human ``"supercluster"``
            changes the result (COP).

    Returns:
        The floor class, or ``None`` for human nodes outside the seven broad
        classes (sinks, Ependymal, Choroid plexus), which are never confident.
    """
    vocab = primary_vocab(species)
    if species == "mouse":
        vocab.row(name)
        return name
    if level == "supercluster" and name == COP_SUPERCLUSTER:
        return "COP"
    broad = vocab.broad_class(name)
    if broad == NEURONS:
        return _HUMAN_NEURON_FLOOR_CLASS.get(vocab.nt(name) or NT_OTHER, "OtherNeuron")
    return _HUMAN_GLIA_FLOOR_CLASS.get(broad)


def load_floor_table(species: Species) -> pd.DataFrame:
    """Load the packaged v1 count floors of a species (§5.4).

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        One row per (level, floor_class, platform, panel_family) with an
        integer ``min_counts`` column.

    Raises:
        ValueError: If a column is missing or a key is duplicated.
    """
    filename = FLOOR_FILES[_check_species(species)]
    frame = load_asset_table(filename)
    missing = [column for column in FLOOR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")
    frame["min_counts"] = frame["min_counts"].astype(int)
    key = ["level", "floor_class", "platform", "panel_family"]
    duplicated = frame[frame.duplicated(key)]
    if not duplicated.empty:
        raise ValueError(
            f"{filename}: duplicated keys {duplicated[key].values.tolist()}"
        )
    return frame


def load_state_genes(species: Species) -> tuple[str, ...]:
    """Return the curated state genes excluded from negative-gene lists (§5.6).

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        Gene symbols in file order.
    """
    frame = load_asset_table(STATE_GENE_FILES[_check_species(species)])
    return tuple(frame["gene_symbol"].tolist())


def load_heldout_markers(species: Species) -> pd.DataFrame:
    """Load the independent held-out marker list of a species (§5.8).

    Args:
        species: ``"human"`` or ``"mouse"``.

    Returns:
        One row per (marker_class, gene_symbol) with the platforms on which
        the gene must not be used (``avoid_platforms``, ``;``-separated).
    """
    return load_asset_table(HELDOUT_MARKER_FILES[_check_species(species)])
