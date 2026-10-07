#!/usr/bin/env python
"""Build a new family's NP5 depth-profile asset from its map_first labels (M13 D8).

Gate P's NP5 judges each class's cells at the family's expected depth (plan
§14 NP5). M13 decision D8 (b) with CHECK K7 (pre-registration §23.9 item 6,
§23.10) fixes the input for a family with real sections: each class's
median from the family's first provisional ``map_first`` run's confident
broad calls, with the overall median for the classes without calls,
written as a ``sim_inputs`` asset (source sha256, deriving script,
``trust_effect: none``), frozen and recorded in the pre-registration
before gate P runs on the family. No other real-label input reaches gate P.

This script reads the family's RESOLVE label tables
(``<sid>_celltype_labels.parquet``, one per section, read only) and writes
the asset ``np5_depth__<family_id>.csv`` (role ``gate_p_profile``:
``class_code``, ``total_counts``, ``n_cells``) and its sidecar into
``src/merxen/assets/annotation/sim_inputs/``:

1. **Cells.** The table cells (``in_table``) of every section, pooled; each
   cell counts once. A confident broad call (``ct_broad_status`` is
   ``confident``) takes the class key gate P keys its decisions by, the E2
   floor class of its broad and NT names (``vocab.human_floor_class``:
   neurons by NT, glia by broad class), the key RESOLVE reads the
   resolvability decisions with. Every other table cell is coded
   ``__not_confident__``.
2. **Rows.** Per class code, one row per distinct total with its number of
   cells, so the asset is exact and stays small (``total_counts`` is an
   integer column of the label table).
3. **Readings** (pre-registration §23.20; adopted defaults, put to the user
   before any of the family's outputs are read):

   - (i) the scored segmentation (proseg_hybrid, D25) and the sections
     pooled (D8 (b)); each section's values are reported in the sidecar;
   - (ii) a class with fewer than ``PROFILE_MIN_CLASS_CELLS`` (100) pooled
     confident broad calls takes the overall median, as a class without
     calls (plan §8.3 v7.5's profile minimum; the asset keeps its calls and
     ``sim_inputs.FamilyDepthProfile.class_depths`` applies the minimum);
   - (iii) gate P keys every level by the same floor classes, so an NT or
     supercluster class takes the depths of the confident broad calls of
     its floor class; supercluster COP, which has no broad key, takes the
     overall median (A45 (a));
   - (iv) the overall median is that of the confident broad calls, every
     class pooled ("the profile's overall median", A45 (a) and §23.17 item
     7); the label-free median of every table cell is reported beside it
     and gives NP3's report-only depth histogram.

The sidecar records the asset as in-house data: its source file is the
source manifest this script writes under the evidence root
(``--manifest``), which names each label table by its path, size and
sha256; the sidecar's ``derived_from`` lists each table by its file name,
section, segmentation and sha256 only, so no local path outside the
evidence root reaches the repository. The script refuses a table that is
not a human ``map_first`` RESOLVE output of a provisional panel in frontal
cortex (D3 (a)), tables of another segmentation, panel, family or primary
bundle, a section given twice, a section ``--sections`` does not name or one
it names without a table (reading (i) pools every section of the family), a
section whose dataset gate failed or is not recorded, and a confident broad
call without a floor class. A failed section adds no confident call, but its
cells would still move the table median and NP3's report-only histogram, so
it goes back to the user rather than being pooled or dropped silently.

The asset's sidecar records the primary bundle's ``build_hash`` the tables
were resolved with (``run.primary_build_hash``); gate P refuses the asset
when its own PREP bundle has another (``gate_p_run.check_profile_bundle``),
so the profile and the decisions gate P scores come from one bundle.

Usage::

    python scripts/annotation/build_np5_depth_profile.py \\
        --labels <section 1 label table> ... --labels <section 4 label table> \\
        --sections P5822,P4815,P3518,P7417 \\
        --family-id human_merscope_aa25d5a241d0 --panel-hash aa25d5a241d0 \\
        --tissue brain_ff --date YYYY-MM-DD \\
        --evidence-root /srv/storage/MerXen/annotation_dev/evidence_20260926 \\
        --manifest <evidence root>/m13/npf_run/np5_profile/source_manifest.json

``--check`` writes nothing and exits 1 when the committed asset, its
sidecar or the manifest differs from what the tables give. After writing,
re-run ``scripts/annotation/build_sim_inputs.py`` so that the ``NOTICE``
lists the asset under "In-house assets".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation import sim_inputs as si
from merxen.annotation.diagnostics import own_family_id
from merxen.annotation.schema import Columns
from merxen.annotation.vocab import NEURONS, human_floor_class

logger = logging.getLogger("build_np5_depth_profile")

SCRIPT_PATH = "scripts/annotation/build_np5_depth_profile.py"
ASSET_PREFIX = "np5_depth__"
DEFAULT_SEGMENTATION = "proseg_hybrid"
MAP_FIRST = "map_first"
PROVISIONAL = "provisional"
CONFIDENT = "confident"
PRIMARY = "primary"
# D3 (a): the family's sections are frontal cortex, set explicitly.
FRONTAL_CORTEX = "frontal_cortex"
FAILED_GATE = "failed"
CHEMISTRY_OF_PLATFORM: dict[str, str] = {"MERSCOPE": "merscope"}
LABEL_COLUMNS: tuple[str, ...] = (
    Columns.SAMPLE_ID,
    Columns.PLATFORM,
    Columns.SEGMENTATION,
    Columns.SPECIES,
    Columns.PANEL_HASH,
    Columns.TOTAL_COUNTS,
    Columns.IN_TABLE,
    Columns.level("broad", "name"),
    Columns.level("broad", "status"),
    Columns.level("nt", "name"),
)
READINGS: dict[str, str] = {
    "segmentation_and_pooling": (
        "(i) the scored segmentation, the sections pooled (each table cell "
        "once; D8 (b)); each section's values reported"
    ),
    "small_classes": (
        "(ii) a class with fewer than min_class_cells pooled confident broad "
        "calls takes the overall median, as a class without calls (plan §8.3 "
        "v7.5's profile minimum)"
    ),
    "finer_levels": (
        "(iii) every gate-P level is keyed by the E2 floor classes, so an NT "
        "or supercluster class takes the depths of its floor class's "
        "confident broad calls; supercluster COP takes the overall median "
        "(A45 (a))"
    ),
    "overall_median": (
        "(iv) the median of the confident broad calls, every class pooled "
        "(A45 (a), pre-registration §23.17 item 7); the label-free median of "
        "every table cell is reported and gives NP3's report-only depth "
        "histogram"
    ),
}
_HEX = re.compile(r"[0-9a-f]{12,64}")


class ProfileBuildError(RuntimeError):
    """A label table is missing, inconsistent or of another run."""


@dataclass(frozen=True)
class SectionLabels:
    """One section's table cells, as the asset reads them.

    Attributes:
        path: The label table.
        sha256: Its sha256.
        size: Its bytes.
        sample_id: The section.
        pair_id: Its pair (``None`` for a table without the column).
        platform: Its platform.
        segmentation: Its segmentation.
        panel_hash: Its panel hash.
        totals: Total counts of every table cell.
        classes: The floor class of each table cell's confident broad call
            (``None``: not confident at broad).
        provenance: What the label table's provenance records of the run.
    """

    path: Path
    sha256: str
    size: int
    sample_id: str
    pair_id: str | None
    platform: str
    segmentation: str
    panel_hash: str
    totals: np.ndarray
    classes: np.ndarray
    provenance: dict[str, Any]

    @property
    def confident(self) -> np.ndarray:
        """Whether each table cell is a confident broad call."""
        return np.array([value is not None for value in self.classes], dtype=bool)

    def summary(self) -> dict[str, Any]:
        """Return the section's reported values (counts and medians)."""
        confident = self.confident
        per_class: dict[str, dict[str, Any]] = {}
        for name in sorted({str(v) for v in self.classes if v is not None}):
            values = self.totals[self.classes == name]
            per_class[name] = {
                "n_cells": int(values.size),
                "median_counts": float(np.median(values)),
            }
        return {
            "sample_id": self.sample_id,
            "n_table_cells": int(self.totals.size),
            "median_counts_table_cells": _median(self.totals),
            "n_confident_broad": int(confident.sum()),
            "median_counts_confident_broad": _median(self.totals[confident]),
            "classes": per_class,
            "gate_level": self.provenance.get("gate_level"),
        }


def _median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def sha256_file(path: Path) -> str:
    """Return a file's sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _single(frame: pd.DataFrame, column: str, path: Path) -> str:
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ProfileBuildError(
            f"{path.name}: {column} must hold one value, got {sorted(values)[:5]}"
        )
    return str(values[0])


def _run_provenance(provenance: Any, path: Path) -> dict[str, Any]:
    """The run facts a label table's provenance records (refusing others)."""
    if provenance is None:
        raise ProfileBuildError(
            f"{path.name} carries no annotation provenance: not a RESOLVE output"
        )
    if provenance.mode != MAP_FIRST:
        raise ProfileBuildError(
            f"{path.name} is a {provenance.mode} output, not a {MAP_FIRST} RESOLVE "
            "output (D8: the family's first provisional map_first run)"
        )
    if provenance.species != "human":
        raise ProfileBuildError(
            f"{path.name} is a {provenance.species} table; the floor classes of "
            "this asset are human (no mouse gate P before M6b)"
        )
    if provenance.anatomical_region != FRONTAL_CORTEX:
        raise ProfileBuildError(
            f"{path.name} was resolved for the region "
            f"{provenance.anatomical_region!r}, not {FRONTAL_CORTEX!r} (D3 (a): "
            "the family's sections are frontal cortex)"
        )
    trust = None if provenance.panel is None else provenance.panel.panel_trust
    if trust != PROVISIONAL:
        raise ProfileBuildError(
            f"{path.name}: the panel's trust is {trust!r}, not {PROVISIONAL!r} (D8: "
            "the family's first provisional map_first run)"
        )
    primary = [
        reference
        for reference in provenance.references.values()
        if reference.role == PRIMARY
    ]
    if len(primary) != 1 or not primary[0].build_hash:
        raise ProfileBuildError(
            f"{path.name}: the provenance names no single primary bundle build_hash"
        )
    gate = provenance.gate
    return {
        "mode": provenance.mode,
        "anatomical_region": provenance.anatomical_region,
        "merxen_version": provenance.merxen_version,
        "panel_trust": trust,
        "primary_reference": primary[0].reference_id,
        "primary_build_hash": primary[0].build_hash,
        "gate_level": None if gate is None else getattr(gate, "level", None),
    }


def read_section(path: Path) -> SectionLabels:
    """Read one section's label table (only the columns the asset needs).

    Args:
        path: ``<sid>_celltype_labels.parquet`` of a ``map_first`` RESOLVE run.

    Returns:
        The section.

    Raises:
        ProfileBuildError: If the table is not a human ``map_first`` RESOLVE
            output of a provisional panel, lacks a column, mixes sections,
            platforms, segmentations or panels, or holds a confident broad
            call without a floor class.
    """
    import pyarrow.parquet as pq

    from merxen.annotation.provenance import AnnotationProvenance
    from merxen.annotation.schema import LABELS_METADATA_KEY

    if not path.is_file():
        raise ProfileBuildError(f"label table {path} is missing")
    schema = pq.read_schema(path)
    missing = [column for column in LABEL_COLUMNS if column not in schema.names]
    if missing:
        raise ProfileBuildError(f"{path.name} lacks the columns {missing}")
    raw = (schema.metadata or {}).get(LABELS_METADATA_KEY)
    provenance = (
        None if raw is None else AnnotationProvenance.from_uns_json(raw.decode())
    )
    facts = _run_provenance(provenance, path)
    columns = list(LABEL_COLUMNS)
    if Columns.PAIR_ID in schema.names:
        columns.append(Columns.PAIR_ID)
    frame = pq.read_table(path, columns=columns).to_pandas()
    sample_id = _single(frame, Columns.SAMPLE_ID, path)
    pair_id = (
        _single(frame, Columns.PAIR_ID, path) if Columns.PAIR_ID in frame else None
    )
    if _single(frame, Columns.SPECIES, path) != "human":
        raise ProfileBuildError(f"{path.name}: the species column is not human")
    table = frame[frame[Columns.IN_TABLE].astype(bool).to_numpy()]
    totals = table[Columns.TOTAL_COUNTS].to_numpy(np.int64)
    confident = (
        table[Columns.level("broad", "status")].astype(str).to_numpy() == CONFIDENT
    )

    # The class key RESOLVE reads the decisions with (consensus: the E2
    # floor class of the assigned node's broad class and NT).
    def names(column: str) -> list[str | None]:
        values = table[Columns.level(column, "name")].astype(object).to_numpy()
        return [None if pd.isna(value) else str(value) for value in values[confident]]

    pairs = pd.DataFrame({"broad": names("broad"), "nt": names("nt")}, dtype=object)
    keys: dict[tuple[Any, Any], str | None] = {}
    for name, neuron_type in pairs.drop_duplicates().itertuples(index=False):
        keys[(name, neuron_type)] = human_floor_class(
            name, neuron_type if name == NEURONS else None
        )
    lacking = sorted({str(name) for (name, _), key in keys.items() if key is None})
    classes = np.full(len(table), None, dtype=object)
    classes[confident] = [
        keys[(name, neuron_type)] for name, neuron_type in pairs.itertuples(index=False)
    ]
    if lacking:
        raise ProfileBuildError(
            f"{path.name}: confident broad calls without a floor class: {lacking}"
        )
    return SectionLabels(
        path=path,
        sha256=sha256_file(path),
        size=path.stat().st_size,
        sample_id=sample_id,
        pair_id=pair_id,
        platform=_single(frame, Columns.PLATFORM, path).upper(),
        segmentation=_single(frame, Columns.SEGMENTATION, path),
        panel_hash=_single(frame, Columns.PANEL_HASH, path),
        totals=totals,
        classes=classes,
        provenance=facts,
    )


def _named_sections(
    sections: Sequence[SectionLabels], expected_sections: Sequence[str]
) -> None:
    """Refuse tables that are not exactly the family's named sections.

    An entry names a section by its sample id or its pair id. Reading (i)
    of pre-registration §23.20 pools every section of the family, so a
    named section without a table, a table of a section not named and an
    entry naming two sections are refused.
    """
    entries = [str(entry).strip() for entry in expected_sections if str(entry).strip()]
    if not entries:
        raise ProfileBuildError(
            "name the family's sections (--sections): reading (i) pools every one"
        )
    if len(set(entries)) != len(entries):
        raise ProfileBuildError(f"--sections names a section twice: {entries}")
    named: dict[str, list[str]] = {}
    for entry in entries:
        hits = [
            section.sample_id
            for section in sections
            if entry in (section.sample_id, section.pair_id)
        ]
        if len(hits) > 1:
            raise ProfileBuildError(
                f"--sections entry {entry!r} names more than one table ({hits}); "
                "give the sample ids"
            )
        named[entry] = hits
    missing = sorted(entry for entry, hits in named.items() if not hits)
    if missing:
        raise ProfileBuildError(
            f"the sections {missing} have no label table among those given: "
            "reading (i) pools every section of the family (pre-registration "
            "§23.20), so a missing one goes back to the user"
        )
    covered = {hit for hits in named.values() for hit in hits}
    extra = sorted(
        section.sample_id for section in sections if section.sample_id not in covered
    )
    if extra:
        raise ProfileBuildError(
            f"the label tables of {extra} are not among the sections --sections names"
        )


def check_sections(
    sections: Sequence[SectionLabels],
    *,
    family_id: str,
    segmentation: str,
    panel_hash: str | None,
    expected_sections: Sequence[str],
) -> None:
    """Refuse sections that are not one family's run on one segmentation.

    Raises:
        ProfileBuildError: For no section, a section given twice, tables that
            are not exactly ``expected_sections`` (each by its sample id or
            pair id), a section whose dataset gate failed or is not
            recorded, another segmentation, platforms, panels or primary
            bundles that differ, a panel hash other than ``panel_hash`` (a
            prefix), or a ``family_id`` other than the panel's own.
    """
    if not sections:
        raise ProfileBuildError("give at least one label table")
    ids = [section.sample_id for section in sections]
    if len(set(ids)) != len(ids):
        raise ProfileBuildError(f"a section is given twice: {sorted(ids)}")
    _named_sections(sections, expected_sections)
    for section in sections:
        level = section.provenance.get("gate_level")
        if level is None:
            raise ProfileBuildError(
                f"{section.path.name} records no dataset gate level: not a human "
                "RESOLVE output whose gate can be checked"
            )
        if level == FAILED_GATE:
            raise ProfileBuildError(
                f"the dataset gate of {section.sample_id} failed: the section has "
                "no confident call, but its cells would move the table median "
                "and NP3's report-only histogram; reading (i) pools the family's "
                "sections, so a failed one goes back to the user (pre-registration "
                "§23.20)"
            )
    for name, values in (
        ("segmentation", {section.segmentation for section in sections}),
        ("platform", {section.platform for section in sections}),
        ("panel_hash", {section.panel_hash for section in sections}),
        (
            "primary build_hash",
            {section.provenance["primary_build_hash"] for section in sections},
        ),
    ):
        if len(values) != 1:
            raise ProfileBuildError(
                f"the sections differ in their {name}: {sorted(values)} (D8: one "
                "map_first run of one family on one segmentation)"
            )
    first = sections[0]
    if first.segmentation != segmentation:
        raise ProfileBuildError(
            f"the tables are of {first.segmentation}, not the scored {segmentation} "
            "(reading (i), pre-registration §23.20)"
        )
    if panel_hash is not None and not first.panel_hash.startswith(panel_hash):
        raise ProfileBuildError(
            f"the tables' panel_hash {first.panel_hash[:16]} is not the frozen "
            f"{panel_hash} (D17)"
        )
    own = own_family_id("human", [first.platform], first.panel_hash)
    if own != family_id:
        raise ProfileBuildError(
            f"the tables are of the family {own}, not {family_id} (D16)"
        )


@dataclass(frozen=True)
class BuiltProfile:
    """The generated asset (CSV text and sidecar) and the source manifest."""

    asset: si.SimInputAsset
    csv_text: str
    manifest_text: str


def profile_rows(
    sections: Sequence[SectionLabels],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return the asset's rows and its class legend (steps 1 and 2).

    Returns:
        ``(rows, legend)``: rows ``class_code``, ``total_counts``,
        ``n_cells`` sorted by code and total; the legend one entry per code
        (``code``, ``class``, ``kind``, ``neuronal``, ``n_cells``,
        ``median_counts``), the confident classes sorted by name first and
        ``__not_confident__`` last.
    """
    totals = np.concatenate([section.totals for section in sections])
    classes = np.concatenate([section.classes for section in sections])
    confident = np.array([value is not None for value in classes], dtype=bool)
    names = sorted({str(value) for value in classes[confident]})
    labels = np.array(
        [
            str(value) if value is not None else si.PROFILE_NOT_CONFIDENT
            for value in classes
        ],
        dtype=object,
    )
    legend: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for code, name in enumerate([*names, si.PROFILE_NOT_CONFIDENT]):
        values = totals[labels == name]
        if name == si.PROFILE_NOT_CONFIDENT and values.size == 0:
            continue
        kind = (
            si.PROFILE_KIND_NOT_CONFIDENT
            if name == si.PROFILE_NOT_CONFIDENT
            else si.PROFILE_KIND_CONFIDENT
        )
        legend.append(
            {
                "code": code,
                "class": name,
                "kind": kind,
                "neuronal": None
                if kind == si.PROFILE_KIND_NOT_CONFIDENT
                else si.is_neuronal_class(name, "human"),
                "n_cells": int(values.size),
                "median_counts": float(np.median(values)),
            }
        )
        distinct, counts = np.unique(values, return_counts=True)
        frames.append(
            pd.DataFrame(
                {
                    "class_code": code,
                    "total_counts": distinct.astype(np.int64),
                    "n_cells": counts.astype(np.int64),
                }
            )
        )
    rows = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(si.GATE_P_PROFILE_COLUMNS))
    )
    return rows, legend


def _csv_text(rows: pd.DataFrame) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(si.GATE_P_PROFILE_COLUMNS))
    for code, total, count in rows[list(si.GATE_P_PROFILE_COLUMNS)].itertuples(
        index=False
    ):
        writer.writerow([int(code), int(total), int(count)])
    return buffer.getvalue()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ProfileBuildError(
            f"{path} is not under the evidence root {root}"
        ) from error


def build_profile(
    sections: Sequence[SectionLabels],
    *,
    family_id: str,
    tissue: str,
    date: str,
    evidence_root: Path,
    manifest_path: Path,
    repo_root: Path,
    chemistry: str | None = None,
) -> BuiltProfile:
    """Return the asset, its sidecar and the source manifest.

    Args:
        sections: The checked sections (``check_sections``).
        family_id: The family.
        tissue: Tissue and preservation (``brain_ff``).
        date: The derivation date (``YYYY-MM-DD``).
        evidence_root: The evidence root (``$A``).
        manifest_path: Where the source manifest goes (under the evidence
            root).
        repo_root: The repository root (for this script's sha256).
        chemistry: The chemistry (default: from the platform; MERSCOPE).

    Returns:
        The built asset.

    Raises:
        ProfileBuildError: For an unknown chemistry, a manifest outside the
            evidence root, or an asset above the repository limit.
    """
    first = sections[0]
    # Reading (ii)'s minimum, recorded; FamilyDepthProfile applies it.
    min_class_cells = si.PROFILE_MIN_CLASS_CELLS
    chemistry = chemistry or CHEMISTRY_OF_PLATFORM.get(first.platform)
    if chemistry not in si.CHEMISTRIES:
        raise ProfileBuildError(
            f"no chemistry for the platform {first.platform}; give --chemistry"
        )
    manifest_relative = _relative_to_root(manifest_path, evidence_root)
    rows, legend = profile_rows(sections)
    confident = [item for item in legend if item["kind"] == si.PROFILE_KIND_CONFIDENT]
    confident_totals = np.concatenate(
        [section.totals[section.confident] for section in sections]
    )
    table_totals = np.concatenate([section.totals for section in sections])
    pooled = {
        "n_sections": len(sections),
        "n_table_cells": int(table_totals.size),
        "n_confident_broad": int(confident_totals.size),
        "overall_median_confident_broad": _median(confident_totals),
        "median_counts_table_cells": _median(table_totals),
        "min_class_cells": int(min_class_cells),
        "classes_own_depths": sorted(
            item["class"] for item in confident if item["n_cells"] >= min_class_cells
        ),
        "classes_overall_median": sorted(
            item["class"] for item in confident if item["n_cells"] < min_class_cells
        ),
    }
    per_section = [section.summary() for section in sections]
    run = {
        key: first.provenance[key]
        for key in (
            "mode",
            "anatomical_region",
            "panel_trust",
            "primary_reference",
            "primary_build_hash",
        )
    }
    manifest = {
        "family_id": family_id,
        "panel_hash": first.panel_hash,
        "segmentation": first.segmentation,
        "platform": first.platform,
        "run": run,
        "label_tables": [
            {
                "path": str(section.path.resolve()),
                "file": section.path.name,
                "sample_id": section.sample_id,
                "size": section.size,
                "sha256": section.sha256,
                "provenance": section.provenance,
            }
            for section in sections
        ],
        "readings": READINGS,
        "pooled": pooled,
        "sections": per_section,
        "deriving_script": SCRIPT_PATH,
        "date": date,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_bytes = manifest_text.encode("utf-8")
    text = _csv_text(rows)
    payload = text.encode("utf-8")
    asset_id = f"{ASSET_PREFIX}{family_id}"
    sections_text = ", ".join(section.sample_id for section in sections)
    provenance: dict[str, Any] = {
        si.SOURCE_KIND_KEY: si.SOURCE_KIND_IN_HOUSE,
        "source_dataset": (
            f"the {family_id} family's map_first RESOLVE label tables, sections "
            f"{sections_text} (in-house; {first.segmentation})"
        ),
        "source_page": (
            "pre-registration §23.9 item 6, §23.10 (D8 (b) with CHECK K7) and "
            "§23.20 (docs/acceptance/annotation-v1-preregistration.md); "
            f"$A/{manifest_relative}"
        ),
        "source_files": [
            {
                "file": manifest_path.name,
                "path": f"$A/{manifest_relative}",
                "size": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
        ],
        "derived_from": [
            {
                "file": section.path.name,
                "sample_id": section.sample_id,
                "segmentation": section.segmentation,
                "size": section.size,
                "sha256": section.sha256,
            }
            for section in sections
        ],
        "licence": (
            "in-house data, no public licence; the asset holds derived per-class "
            "depth histograms only"
        ),
        "licence_note": "in-house data: no public licence text applies",
        "attribution": f"in-house sections of this project ({family_id})",
        "xoa_version": "not applicable: the sections were segmented by MerXen",
        "segmentation": f"MerXen {first.segmentation}",
        "deriving_script": SCRIPT_PATH,
        "deriving_script_sha256": si._sha256_file(repo_root / SCRIPT_PATH),
        "method": (
            "Every table cell of the sections, pooled, each cell once: a "
            "confident broad call coded by its E2 floor class "
            "(vocab.human_floor_class of ct_broad_name and ct_nt_name), every "
            "other table cell __not_confident__; one row per class and "
            "distinct total_counts with its cells. Gate P's NP5 gives a class "
            f"with >= {min_class_cells} confident broad calls its own depths and "
            "every other class the median of the confident broad calls "
            "(sim_inputs.FamilyDepthProfile; pre-registration §23.20)."
        ),
        "readings": READINGS,
        "run": run,
        "evidence": [
            f"$A/{manifest_relative}",
            "docs/acceptance/annotation-v1-preregistration.md §23.20",
        ],
        "evidence_root": f"$A = {evidence_root}",
        "date": date,
        "use": (
            "gate-P NP5 input only (plan §14 NP5; M13 decision D8 with CHECK "
            "K7); never trust evidence, never a PREP input"
        ),
        "summary": {"pooled": pooled, "sections": per_section},
    }
    asset = si.SimInputAsset(
        schema_version=si.SIM_INPUTS_SCHEMA_VERSION,
        asset_id=asset_id,
        asset_version=1,
        file=f"{asset_id}.csv",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        n_rows=text.count("\n") - 1,
        role="gate_p_profile",
        species="human",
        chemistry=chemistry,  # type: ignore[arg-type]
        tissue=tissue,
        reference=run["primary_reference"],
        label=(
            f"MerXen {first.segmentation} segmentation, map_first confident broad "
            f"calls of {family_id} ({len(sections)} sections)"
        ),
        description=(
            f"Per-class depth profile of {family_id} for gate P's NP5 (plan §14 "
            "NP5; M13 decision D8 with CHECK K7): its first provisional map_first "
            "run's confident broad calls per E2 floor class and its other table "
            "cells, as histograms of total counts; never a PREP input"
        ),
        columns={
            "class_code": "the class (legend in extra.class_legend)",
            "total_counts": "total counts of the cells",
            "n_cells": "table cells of the class with these total counts",
        },
        extra={
            "class_legend": legend,
            "family_id": family_id,
            "panel_hash": first.panel_hash,
            "min_class_cells": int(min_class_cells),
        },
        trust_effect="none",
        provenance=provenance,
    )
    if asset.size > si.MAX_ASSET_BYTES:
        raise ProfileBuildError(
            f"{asset.file}: {asset.size} bytes exceed the repository limit"
        )
    return BuiltProfile(asset=asset, csv_text=text, manifest_text=manifest_text)


def build_files(args: argparse.Namespace) -> tuple[BuiltProfile, dict[Path, str]]:
    """Return the built asset and every file it writes, by path."""
    sections = [read_section(Path(path)) for path in args.labels]
    check_sections(
        sections,
        family_id=args.family_id,
        segmentation=args.segmentation,
        panel_hash=args.panel_hash,
        expected_sections=args.sections,
    )
    built = build_profile(
        sections,
        family_id=args.family_id,
        tissue=args.tissue,
        date=args.date,
        evidence_root=args.evidence_root,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        chemistry=args.chemistry,
    )
    files = {
        args.manifest: built.manifest_text,
        args.output_dir / built.asset.file: built.csv_text,
        args.output_dir / f"{built.asset.asset_id}{si.SIDECAR_SUFFIX}": si.sidecar_json(
            built.asset
        ),
    }
    return built, files


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--labels",
        type=Path,
        action="append",
        required=True,
        help="A section's label table (repeat once per section).",
    )
    parser.add_argument(
        "--sections",
        action="append",
        required=True,
        help=(
            "The family's sections, each by its sample id or pair id "
            "(comma-separated or repeated); the label tables must be exactly "
            "these, each once."
        ),
    )
    parser.add_argument("--family-id", required=True)
    parser.add_argument(
        "--panel-hash",
        default=None,
        help="The frozen panel hash (or its first 12+ hex digits), checked.",
    )
    parser.add_argument("--segmentation", default=DEFAULT_SEGMENTATION)
    parser.add_argument("--tissue", required=True, help="e.g. brain_ff")
    parser.add_argument("--chemistry", default=None, choices=si.CHEMISTRIES)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=si.SIM_INPUTS_DIR)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing; exit 1 if a written file is stale.",
    )
    args = parser.parse_args(argv)
    args.sections = [
        entry.strip()
        for value in args.sections
        for entry in str(value).split(",")
        if entry.strip()
    ]
    if args.panel_hash is not None and not _HEX.fullmatch(args.panel_hash):
        parser.error("--panel-hash must be 12 to 64 lower-case hex digits")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must be YYYY-MM-DD")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the builder.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code (1: a stale file under ``--check``, or a
        refusal).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    try:
        built, files = build_files(args)
    except ProfileBuildError as error:
        logger.error("%s", error)
        return 1
    stale = []
    for path, text in files.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            logger.info("%s is up to date", path.name)
            continue
        stale.append(path.name)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            logger.info("Wrote %s (%d bytes)", path, len(text.encode("utf-8")))
    summary = built.asset.provenance["summary"]["pooled"]
    logger.info(
        "%s: %d table cells, %d confident broad calls; own depths: %s; overall "
        "median %s",
        built.asset.asset_id,
        summary["n_table_cells"],
        summary["n_confident_broad"],
        ", ".join(summary["classes_own_depths"]) or "none",
        summary["overall_median_confident_broad"],
    )
    if args.check and stale:
        logger.error("Stale generated files: %s", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
