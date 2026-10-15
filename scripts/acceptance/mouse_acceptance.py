#!/usr/bin/env python
"""M9 mouse acceptance: score gate M (plan §12 M9; pre-registration §9, §24, §25).

Two subcommands. Both read their inputs read-only and write only to ``--out``.

``score``
    Scores the mouse criteria of §9 (MO1-MO11) on MS1 and MS2 with the gate
    M scoring protocol of pre-registration §25, then writes ``summary.json``,
    ``summary.html`` and ``tables/`` to ``--out``.

``full-remap``
    The full pruned re-map of one MAP output, which MO2's full-re-map reading
    needs (§25.2). Every table cell is mapped again on the pruned tree, with
    the MAP run's bundle, its pruned lookup, ``--nodes_to_drop`` and its own
    engine parameters, process count included. It is a MapMyCells run (heavy
    compute; take the host's heavy-compute lock). ``score --mo2-full`` reads
    its output directory.

**Readings.** The defaults are the readings §25 (and §24's amendment of
2026-10-14) registered. ``summary.json`` records every reading with its
source. Another reading is accepted only with ``--exploratory``, which stamps
the summary "exploratory, not gate M"; otherwise the mode is "registered
readings". ``--label`` states the run's purpose ("gate M" for the gate M
score, "smoke test, not gate M" ...).

- **MO2** (``--mo2-remap``, registered ``both``, a tightening): the production
  (subset) re-map and the full re-map must both pass T2 after pruning <= 0.1%,
  class changes outside dropped nodes <= 0.25% and subclass changes <= 1.5%,
  on every segmentation ``SCORED_UNNAMED`` lists (one full re-map each:
  ``--mo2-full DATASET/SEGMENTATION=DIR``).
- **MO4** (``--mo4-window``, registered ``fixed``): MERFISH .31-.33. The
  AP-estimate window (the gate's) is reported beside it.
- **MO8 MAP memory** (``--mo8-map-memory``, registered ``trace_rss``): H14's
  reading, the largest Nextflow trace ``peak_rss`` of the section's MAP tasks
  over its four segmentations (GB = 1024^3 bytes). The memory probe's
  process-tree PSS (``--sampler``) is reported beside it.
- **MO8 MAP processes** (``--mo8-map-processes``, registered 4): a section
  whose MAP ran with another count (trace ``cpus``, the MAP manifest) has its
  MO8 rows ``UNSCORED (process count)``; they go back to the user.
- **MO8 PREP memory** (``--mo8-prep-memory``, registered ``tree_rss``): the
  larger of the largest ``peak_tree_rss_gb`` of the ctm steps in the bundle's
  ``bundle.json`` and the trace ``peak_rss`` of any PREP task of the bundle,
  for a bundle built with 8 processes (its resolvability ``mapping_runs``).
  The tree PSS and the largest single process are reported beside it.
- **MO10 T2** (``--mo10-t2``, registered ``allen_relative``): §24's amendment
  (U1; a LOOSENING, POST-HOC, approved in writing): the bin-mean T2 after
  pruning against the Allen labels' own T2 of the same cells and regions,
  read from the M6b-R ``mo10.json`` (``t2_bin_mean_ok_allen``). The
  registered 0.1% verdict is reported beside it ("both").

**Exceptions.** ``--exceptions`` names a JSON file of user-approved
exceptions (``{"exceptions": [...]}``), each with ``id``, ``criterion``,
``scope``, ``approved`` (ISO date), ``quote`` (the approval, verbatim) and
``recorded_in``. A scored ``FAIL`` row becomes ``FAIL (excepted)`` only when
the exception's criterion is the row's and every scope key matches the row:
``part``, ``dataset``, ``segmentation``, ``gene_set``, ``section``,
``region``, ``ap_bin`` and ``bundle`` (a string or a list of strings), and
``value_max`` / ``value_min`` (the row's value must lie within them). A
failure outside every scope stays ``FAIL``. Exceptions never touch a row that
is not measured (``NOT_AVAILABLE``) or not scored (``UNSCORED``). An MO8
exception names its parts and, for each, what it covers: the ``bundle``
(full build hashes) of a PREP part, the ``dataset`` of a section part
(§25.1.1: the written MO8 exception covers the PREP peak of the two gate M
panels and nothing else).

**Provenance.** Before anything is measured, every bundle path the inputs
name (the reports' provenance, the MAP manifests, the resolve summaries,
the PREP ``bundle_ref.json`` files, the negative control) must resolve to a
real path inside the reference store (``$MERXEN_REFERENCE_STORE``). A path
that is outside it as written is refused without being touched, so no SSD1
path is ever read. Any failure stops the scoring: every row is ``UNSCORED
(provenance)``, the summary is still written, and the exit status is 3.

**Sample names.** The datasets are keyed by their aliases (``ms1``, ``ms2``;
docs/local-configuration.md). Keys read from the command line or a recorded
file (``--dataset``, ``--pseudo-labels``, ``--sampler``, ``--mo2-full``, the
exceptions' scopes, ``mo10.json``'s gene sets) may name a section by its real
name or its alias; the outputs show the names through ``privacy.display``
(``sample_labels``), so on a host with the alias table they are as before.

**Verdicts** (``report_scoring``'s vocabulary): ``PASS``; ``FAIL``; ``FAIL
(excepted)``; ``NOT_AVAILABLE`` (a scored row that could not be measured);
``UNSCORED (...)``; and ``INFO pass`` / ``INFO fail`` / ``INFO`` / ``INFO not
measured`` for rows the flip rule does not score (other segmentations, MO9
[L], MO11, H18, readings reported beside). Every scored row other than
``PASS`` and ``FAIL (excepted)`` goes back to the user, and so does every
entry of ``problems``.

**Flip rule** (§9, §25.2): MO1-MO8 and MO10, with the user's written
exceptions. MO11 keeps its banner rule and MO9 [L] gates only the use of the
reseg and proseg outputs, so neither enters it. The flip verdict is ``PASS``
when every scored row of those criteria is ``PASS`` or ``FAIL (excepted)``
and no problem is open, ``FAIL`` when one is ``FAIL``, else ``INCOMPLETE``.

Scored segmentations: MO1, MO4 and MO7 name one per dataset
(``SCORED_SEGMENTATION``). MO2, MO3, MO5 and MO6 name none and are scored on
``SCORED_UNNAMED`` (§25.2.1, a tightening): §9's default, proseg_hybrid, and,
for the section whose named criteria use original_seg (MS2), original_seg
too; every scored segmentation must pass. MO5's enrichment is the second
section's only (§9). The other segmentations are reported. MO8 is per
4-segmentation section and per PREP bundle; MO9 compares the second
section's segmentations; MO10 is the MERFISH series (``mo10.json``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen import local_config, privacy
from merxen.annotation import report_scoring as sc
from merxen.local_config import LocalConfigError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mouse_corr_shadow as corr  # noqa: E402
from mouse_ap_axis import (  # noqa: E402
    CLASS_LEVEL,
    MANUAL_REGIONS,
    PRIMARY_RUN,
    SUBCLASS_LEVEL,
    candidate_by_label,
)
from run_acceptance import load_trace  # noqa: E402
from sample_labels import alias_key, display_tree  # noqa: E402

logger = logging.getLogger("mouse_acceptance")

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PREREGISTRATION: Final = REPO_ROOT / "docs/acceptance/annotation-v1-preregistration.md"
# §25.3: every bundle comes from this store ($MERXEN_REFERENCE_STORE; tests
# set this attribute); nothing from SSD1.
STORE_ROOT: Path | None = None
STORE_VARIABLE: Final = "MERXEN_REFERENCE_STORE"
EVIDENCE_VARIABLE: Final = "MERXEN_EVIDENCE_ROOT"
# MO1's marker pseudo-labels under the evidence root, whose mouse directories
# are named by the sections' display names (privacy.to_real(form="name")).
PSEUDO_LABELS: Final[dict[str, str]] = {
    "ms1": "data/mouse_ms1/out/tables/C_pseudo_labels_per_cell.csv",
    "ms2": "data/mouse_ms2/tables/C_pseudo_labels_per_cell_original_seg.csv.gz",
}

DATASETS: Final = ("ms1", "ms2")
SEGMENTATIONS: Final = ("original_seg", "proseg_hybrid", "reseg", "proseg_mask")
# §9: "on proseg_hybrid unless stated"; MS2's stated segmentation is
# original_seg (MO1, MO4, MO7); §24 constraint (i) names the same two.
SCORED_SEGMENTATION: Final[dict[str, str]] = {
    "ms1": "proseg_hybrid",
    "ms2": "original_seg",
}
# §9: "on proseg_hybrid unless stated". §25.2.1 (a tightening, recorded on
# 2026-10-14 before any gate M run): the criteria that name no segmentation
# (MO2 with both re-maps, MO3, MO5, MO6) are scored on that default and on
# the segmentation the dataset's named criteria use (MS2's original_seg,
# which M6's exit scored); every scored segmentation must pass.
DEFAULT_SEGMENTATION: Final = "proseg_hybrid"
SCORED_UNNAMED: Final[dict[str, tuple[str, ...]]] = {
    dataset: tuple(dict.fromkeys((segmentation, DEFAULT_SEGMENTATION)))
    for dataset, segmentation in SCORED_SEGMENTATION.items()
}
NEGATIVE_CONTROL: Final = "ms2_invalid"
FLIP_CRITERIA: Final = ("MO1", "MO2", "MO3", "MO4", "MO5", "MO6", "MO7", "MO8", "MO10")
REPORTED_CRITERIA: Final = ("MO9", "MO11", "H18")
TOLERANCE: Final = 1e-12

# Publish layout of a dataset's results root (``<outdir>/<pair>``).
REPORT_PATH: Final = (
    "{seg}/annotation_report/annotation_report_out/acceptance_metrics.json"
)
MAP_DIR: Final = "{seg}/annotation_map/annotation_map_out"
RESOLVE_DIR: Final = "{seg}/annotation_resolve/annotation_resolve_out"
MAP_MANIFEST: Final = "map_manifest.json"
FULL_REMAP_RECORD: Final = "full_remap.json"
FULL_REMAP_PARQUET: Final = "{sample_id}_mmc_wmb_panel_fullpruned.parquet"

# §9 thresholds (plan §14, verbatim there).
MO1_MIN: Final[dict[str, float]] = {"ms1": 0.89, "ms2": 0.84}
MO2_LIMITS: Final[dict[str, float]] = {
    "t2_after": 0.001,
    "class_changes_outside": 0.0025,
    "subclass_changes_outside": 0.015,
}
MO4_FIXED_WINDOW: Final = (
    "C57BL6J-638850.31",
    "C57BL6J-638850.32",
    "C57BL6J-638850.33",
)
ASTRO_EPEN: Final = "30 Astro-Epen"
IMMUNE: Final = "34 Immune"
# Points (percentage points of the allocated soft mass): MS2 original_seg
# Immune +-0.5 and Astro-Epen +-3; MS1 +-1 / +-5 [L; age caveat].
MO4_BANDS: Final[dict[str, dict[str, float]]] = {
    "ms2": {"astro_epen": 3.0, "immune": 0.5},
    "ms1": {"astro_epen": 5.0, "immune": 1.0},
}
MO4_CLASSES: Final[dict[str, str]] = {"astro_epen": ASTRO_EPEN, "immune": IMMUNE}
MO5_ENRICHMENT_MIN: Final = 10.0
MO5_FPR_MAX: Final = 0.005
MO6_RANGE: Final = (0.002, 0.010)
MO7_FAILED: Final = "failed"
MO8_SECTION_MAX_H: Final = 1.25
MO8_MAP_PEAK_MAX_GB: Final = 12.0
MO8_PREP_WALL_MAX_H: Final = 2.0
MO8_PREP_PEAK_MAX_GB: Final = 40.0
MO8_MAP_PROCESSES: Final = 4
MO8_PREP_PROCESSES: Final = 8
MO8_PANEL_REFERENCE: Final = "wmb_panel"
MO9_JSD_MAX: Final = 0.05
MO10_LIMITS: Final[dict[str, float]] = {
    "bin_mean_harm": 0.001,
    "section_max_harm": 0.003,
    "t2_bin_mean": 0.001,
    "missed_region": 0.001,
}
# §24: MO10 "passes when all three gene sets pass" (M6b-R's mo10.json keys).
MO10_GENE_SETS: Final = ("merfish500", "merfish_ms1", "merfish_ms2")
MO11_ANTERIOR_AP_MM: Final = 6.0

ANNOTATION_PROCESSES: Final = (
    "ANNOTATE_PANEL",
    "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
    "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE",
)
MAP_PROCESS: Final = "CLUSTERING_SQUIDPY_ANNOTATE_MAP"
PREP_PROCESS: Final = "ANNOTATE_REFERENCE_PREP"
KIB_PER_GIB: Final = 1024.0**2

# Verdicts (report_scoring; "FAIL (excepted)" for a user-approved exception).
PASS: Final = sc.PASS
FAIL: Final = sc.FAIL
EXCEPTED: Final = "FAIL (excepted)"
NOT_AVAILABLE: Final = sc.NOT_AVAILABLE
INFO: Final = "INFO"
INFO_PASS: Final = "INFO pass"
INFO_FAIL: Final = "INFO fail"
INFO_NOT_MEASURED: Final = "INFO not measured"
UNSCORED_PROCESSES: Final = "process count"
UNSCORED_PROVENANCE: Final = "provenance"
UNSCORED_REGION_RULE: Final = "region rule"
# A run states its purpose with --label (e.g. "gate M"); the mode only
# says whether the readings are the registered ones.
REGISTERED: Final = "registered readings"
EXPLORATORY: Final = "exploratory, not gate M"


# ---------------------------------------------------------------------------
# readings


@dataclass(frozen=True)
class Reading:
    """One reading of the scoring protocol.

    Attributes:
        name: Option name (``mo2_remap`` ...).
        registered: The registered value (§24, §25).
        choices: Accepted values (``()``: any positive integer).
        source: Where the registered value is recorded.
        beside: What is reported beside the scored value.
    """

    name: str
    registered: str
    choices: tuple[str, ...]
    source: str
    beside: str


READINGS: Final[dict[str, Reading]] = {
    reading.name: reading
    for reading in (
        Reading(
            "mo2_remap",
            "both",
            ("both", "subset", "full"),
            "pre-registration §25.2, MO2 (tightening): the production (subset) and "
            "the full re-map must both pass",
            "the re-map the reading does not score",
        ),
        Reading(
            "mo4_window",
            "fixed",
            ("fixed", "ap_estimate"),
            "pre-registration §25.2, MO4 (neutral): the registered fixed window, "
            "MERFISH .31-.33, with §9 MO4's bands",
            "the AP-estimate window (the gate's)",
        ),
        Reading(
            "mo8_map_memory",
            "trace_rss",
            ("trace_rss", "tree_pss"),
            "pre-registration §25.1 (neutral; H14's reading): the largest Nextflow "
            "trace peak_rss of the section's MAP tasks",
            "the memory probe's process-tree PSS (--sampler)",
        ),
        Reading(
            "mo8_map_processes",
            str(MO8_MAP_PROCESSES),
            (),
            "pre-registration §25.1: gate M's MAP runs with 4 MapMyCells processes; "
            "another count leaves the section's MO8 rows unscored",
            "the counts found",
        ),
        Reading(
            "mo8_prep_memory",
            "tree_rss",
            ("tree_rss", "tree_pss", "single_process"),
            "pre-registration §25.1: the summed process-tree RSS of the build "
            "(bundle.json ctm steps; the trace peak_rss of a PREP task), the "
            "strictest of three, at 8 processes",
            "the tree PSS and the largest single process",
        ),
        Reading(
            "mo10_t2",
            "allen_relative",
            ("allen_relative", "fixed"),
            "pre-registration §24, amendment of 2026-10-14 (U1; LOOSENING, "
            "POST-HOC, approved in writing): MO10's T2 clause against the Allen "
            'labels\' own T2 ("both")',
            "the registered 0.1% verdict",
        ),
    )
}


def resolve_readings(
    values: Mapping[str, str | None], *, exploratory: bool
) -> dict[str, dict[str, Any]]:
    """Return each reading's value and record; refuse an unregistered one.

    Args:
        values: Option name to the value given (``None``: the registered one).
        exploratory: Whether ``--exploratory`` was given.

    Returns:
        Reading name to ``value``, ``registered``, ``is_registered``,
        ``source`` and ``reported_beside``.

    Raises:
        ValueError: For a value outside the choices, or a non-registered
            value without ``exploratory``.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, reading in READINGS.items():
        value = values.get(name)
        value = reading.registered if value is None else str(value)
        if reading.choices and value not in reading.choices:
            raise ValueError(
                f"--{name.replace('_', '-')}: {value!r} not in {reading.choices}"
            )
        if not reading.choices and (not value.isdigit() or int(value) < 1):
            raise ValueError(f"--{name.replace('_', '-')}: {value!r} is not a count")
        registered = value == reading.registered
        if not registered and not exploratory:
            raise ValueError(
                f"--{name.replace('_', '-')} {value} is not the registered reading "
                f"({reading.registered}; {reading.source}); a non-registered "
                "reading needs --exploratory, which stamps the summary "
                f"{EXPLORATORY!r}"
            )
        out[name] = {
            "value": value,
            "registered": reading.registered,
            "is_registered": registered,
            "source": reading.source,
            "reported_beside": reading.beside,
        }
    return out


# ---------------------------------------------------------------------------
# rows and verdicts


@dataclass
class Row:
    """One scored (or reported) row of the gate M summary.

    Attributes:
        criterion: ``MO1`` ... ``MO11`` or ``H18``.
        part: The part of the criterion (``marker_consistency`` ...).
        dataset: ``ms1``, ``ms2``, ``ms2_invalid``, ``MERFISH`` or a bundle.
        segmentation: The segmentation (``all``: a 4-segmentation section).
        value: The measured value (``None``: not measured).
        comparator: ``<=``, ``>=``, ``==``, ``|x| <=``, ``in``, ``!=`` or
            ``reported``.
        threshold: The threshold.
        passes: The comparison (``None``: not measured or no threshold).
        scored: Whether the flip rule scores the row.
        source: Where the value comes from.
        note: What a reader needs.
        scope: Further scope keys an exception can match (``gene_set``,
            ``section``, ``region``, ``ap_bin``, ``bundle``).
        beside: Values reported beside the scored one (other readings).
        unscored: Why a scored row is not scored (``UNSCORED (<reason>)``).
        verdict: Set by ``finalize`` / ``apply_exceptions``.
        exception: The id of the exception that covers the row.
        exception_note: An exception that names the row but whose value
            bounds it falls outside.
    """

    criterion: str
    part: str
    dataset: str
    segmentation: str
    value: Any
    comparator: str
    threshold: Any
    passes: bool | None
    scored: bool
    source: str
    note: str = ""
    scope: dict[str, str] = field(default_factory=dict)
    beside: dict[str, Any] = field(default_factory=dict)
    unscored: str = ""
    verdict: str = ""
    exception: str = ""
    exception_note: str = ""

    def full_scope(self: Row) -> dict[str, str]:
        """Return every key an exception's scope can match."""
        return {
            "part": self.part,
            "dataset": self.dataset,
            "segmentation": self.segmentation,
            **{key: str(value) for key, value in self.scope.items()},
        }

    def back_to_user(self: Row) -> bool:
        """Whether the row goes back to the user (a scored row not passing)."""
        return self.scored and self.verdict not in (PASS, EXCEPTED)

    def to_json(self: Row) -> dict[str, Any]:
        """Return the JSON-safe record."""
        record = dataclasses.asdict(self)
        record["value"] = _safe(self.value)
        record["threshold"] = _safe(self.threshold)
        record["beside"] = _safe(self.beside)
        record["back_to_user"] = self.back_to_user()
        return record


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finalize(row: Row) -> Row:
    """Set a row's verdict from its comparison (exceptions come after)."""
    if row.unscored:
        row.verdict = f"UNSCORED ({row.unscored})"
    elif row.comparator == "reported":
        row.verdict = INFO if row.value is not None else INFO_NOT_MEASURED
    elif row.passes is None:
        row.verdict = NOT_AVAILABLE if row.scored else INFO_NOT_MEASURED
    elif not row.scored:
        row.verdict = INFO_PASS if row.passes else INFO_FAIL
    else:
        row.verdict = PASS if row.passes else FAIL
    return row


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def at_most(value: Any, limit: float) -> bool | None:
    """``value <= limit`` (absolute tolerance 1e-12), ``None`` when missing."""
    number = _float(value)
    return None if number is None else number <= limit + TOLERANCE


def at_least(value: Any, limit: float) -> bool | None:
    """``value >= limit`` (absolute tolerance 1e-12), ``None`` when missing."""
    number = _float(value)
    return None if number is None else number >= limit - TOLERANCE


# ---------------------------------------------------------------------------
# exceptions

EXCEPTION_KEYS: Final = frozenset(
    {
        "part",
        "dataset",
        "segmentation",
        "gene_set",
        "section",
        "region",
        "ap_bin",
        "bundle",
        "remap",
        "window",
        "level",
        "kind",
    }
)
EXCEPTION_BOUNDS: Final = frozenset({"value_max", "value_min"})
# MO8's parts and the scope key that names what an exception of each covers
# (one PREP bundle, one 4-segmentation section): an MO8 exception without it
# would cover every bundle or section (§25.1.1).
MO8_PART_UNITS: Final[dict[str, str]] = {
    "prep_peak": "bundle",
    "prep_wall": "bundle",
    "map_peak": "dataset",
    "section_time": "dataset",
}
BUILD_HASH_LENGTH: Final = 64
EXCEPTION_FIELDS: Final = frozenset(
    {"id", "criterion", "scope", "approved", "quote", "recorded_in"}
)
KNOWN_CRITERIA: Final = frozenset(FLIP_CRITERIA + REPORTED_CRITERIA)


@dataclass(frozen=True)
class ApprovedException:
    """A user-approved exception (written approval, quoted verbatim).

    Attributes:
        id: Its id (unique in the file).
        criterion: The criterion it covers (``MO10`` ...).
        scope: Scope keys (``EXCEPTION_KEYS``: a string or a list of
            strings) and value bounds (``value_max``, ``value_min``).
        approved: The approval date (ISO).
        quote: The user's approval, verbatim.
        recorded_in: Where the approval is recorded.
        match_scope: The scope rows are matched with: its strings through
            ``alias_key`` (an exception names a section by its real name or
            its alias; R2). ``scope`` is recorded as written.
    """

    id: str
    criterion: str
    scope: Mapping[str, Any]
    approved: str
    quote: str
    recorded_in: str = ""
    match_scope: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self: ApprovedException) -> dict[str, Any]:
        """Return the JSON record."""
        return {
            "id": self.id,
            "criterion": self.criterion,
            "scope": dict(self.scope),
            "approved": self.approved,
            "quote": self.quote,
            "recorded_in": self.recorded_in,
        }


def parse_exceptions(payload: Any) -> list[ApprovedException]:
    """Validate an exceptions file's content.

    Args:
        payload: ``{"exceptions": [...]}`` (or the list itself).

    Returns:
        The exceptions.

    Raises:
        ValueError: For a malformed entry: an unknown field (a bound beside
            ``scope`` instead of in it would widen the exception), no id, an
            unknown criterion, an empty or unknown scope key, a bound that is
            not a number, an approval date that is not an ISO date, or an
            empty quote.
    """
    entries = payload.get("exceptions") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ValueError('the exceptions file holds no "exceptions" list')
    out: list[ApprovedException] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"exception {index}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: not an object")
        extra = set(entry) - EXCEPTION_FIELDS
        if extra:
            raise ValueError(
                f"{where}: unknown keys {sorted(extra)} (bounds go in the scope)"
            )
        ident = str(entry.get("id") or "").strip()
        if not ident or ident in seen:
            raise ValueError(f"{where}: a unique id is required")
        seen.add(ident)
        criterion = str(entry.get("criterion") or "")
        if criterion not in KNOWN_CRITERIA:
            raise ValueError(f"{ident}: unknown criterion {criterion!r}")
        scope = entry.get("scope")
        if not isinstance(scope, Mapping) or not set(scope) & EXCEPTION_KEYS:
            raise ValueError(
                f"{ident}: the scope must name at least one of {sorted(EXCEPTION_KEYS)}"
            )
        unknown = set(scope) - EXCEPTION_KEYS - EXCEPTION_BOUNDS
        if unknown:
            raise ValueError(f"{ident}: unknown scope keys {sorted(unknown)}")
        for key, value in scope.items():
            if key in EXCEPTION_BOUNDS:
                if _float(value) is None:
                    raise ValueError(f"{ident}: {key} must be a number")
            elif not (
                isinstance(value, str)
                or (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, str) for item in value)
                )
            ):
                raise ValueError(f"{ident}: scope {key} must be a string or a list")
        approved = str(entry.get("approved") or "")
        try:
            date.fromisoformat(approved)
        except ValueError as error:
            raise ValueError(
                f"{ident}: approved {approved!r} is not an ISO date"
            ) from error
        quote = str(entry.get("quote") or "").strip()
        if not quote:
            raise ValueError(f"{ident}: the user's approval must be quoted")
        if criterion == "MO8":
            _check_mo8_scope(ident, scope)
        out.append(
            ApprovedException(
                id=ident,
                criterion=criterion,
                scope=dict(scope),
                approved=approved,
                quote=quote,
                recorded_in=str(entry.get("recorded_in") or ""),
                match_scope={
                    key: value if key in EXCEPTION_BOUNDS else _scope_keys(value)
                    for key, value in scope.items()
                },
            )
        )
    return out


def _scope_keys(value: str | list[str]) -> str | list[str]:
    """A scope value with section names as the script's keys (R2)."""
    keys = (*DATASETS, NEGATIVE_CONTROL, *MO10_GENE_SETS)
    if isinstance(value, str):
        return alias_key(value, keys)
    return [alias_key(item, keys) for item in value]


def _scope_values(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def _check_mo8_scope(ident: str, scope: Mapping[str, Any]) -> None:
    """Refuse an MO8 exception broader than named parts of named units.

    Raises:
        ValueError: When the scope names no part, an unknown part, or not
            the unit of each part (``MO8_PART_UNITS``), or names a bundle by
            anything but its full build hash.
    """
    parts = _scope_values(scope.get("part"))
    if not parts:
        raise ValueError(
            f"{ident}: an MO8 exception must name its part ({sorted(MO8_PART_UNITS)})"
        )
    unknown = sorted(set(parts) - set(MO8_PART_UNITS))
    if unknown:
        raise ValueError(f"{ident}: unknown MO8 part(s) {unknown}")
    for part in parts:
        unit = MO8_PART_UNITS[part]
        if unit not in scope:
            raise ValueError(
                f"{ident}: an MO8 exception for {part} names the {unit} it covers"
            )
    for bundle in _scope_values(scope.get("bundle")):
        if len(bundle) != BUILD_HASH_LENGTH or not all(
            char in "0123456789abcdef" for char in bundle
        ):
            raise ValueError(f"{ident}: MO8 bundle {bundle!r} is not a full build hash")


def exception_covers(exception: ApprovedException, row: Row) -> tuple[bool, bool]:
    """Return (names the row, covers it): scope keys, then value bounds."""
    if exception.criterion != row.criterion:
        return False, False
    keys = row.full_scope()
    for key, allowed in (exception.match_scope or exception.scope).items():
        if key in EXCEPTION_BOUNDS:
            continue
        options = [allowed] if isinstance(allowed, str) else list(allowed)
        if keys.get(key) not in options:
            return False, False
    value = _float(row.value)
    bounds_ok = True
    if "value_max" in exception.scope:
        bounds_ok = (
            bounds_ok
            and value is not None
            and (value <= float(exception.scope["value_max"]) + TOLERANCE)
        )
    if "value_min" in exception.scope:
        bounds_ok = (
            bounds_ok
            and value is not None
            and (value >= float(exception.scope["value_min"]) - TOLERANCE)
        )
    return True, bounds_ok


def apply_exceptions(
    rows: Sequence[Row], exceptions: Sequence[ApprovedException]
) -> list[dict[str, Any]]:
    """Turn scored ``FAIL`` rows inside an exception's scope into ``FAIL (excepted)``.

    Args:
        rows: Finalised rows (changed in place).
        exceptions: The approved exceptions.

    Returns:
        Per exception its record with the rows it covers (``applied_to``) and
        the failing rows it names whose value lies outside its bounds
        (``named_outside_bounds``; they stay ``FAIL``).
    """
    usage: dict[str, dict[str, list[str]]] = {
        exception.id: {"applied_to": [], "named_outside_bounds": []}
        for exception in exceptions
    }
    for row in rows:
        if row.verdict != FAIL or not row.scored:
            continue
        for exception in exceptions:
            named, covered = exception_covers(exception, row)
            if not named:
                continue
            label = _row_label(row)
            if covered:
                row.verdict = EXCEPTED
                row.exception = exception.id
                usage[exception.id]["applied_to"].append(label)
                break
            usage[exception.id]["named_outside_bounds"].append(label)
            row.exception_note = (
                f"{exception.id} names this row, but its value {row.value} lies "
                "outside the exception's bounds"
            )
    return [{**exception.to_json(), **usage[exception.id]} for exception in exceptions]


def _row_label(row: Row) -> str:
    extra = ",".join(f"{key}={value}" for key, value in sorted(row.scope.items()))
    return f"{row.criterion}/{row.part} {row.dataset} {row.segmentation}" + (
        f" [{extra}]" if extra else ""
    )


# ---------------------------------------------------------------------------
# provenance (§25.3)


@dataclass
class ProvenanceCheck:
    """Every bundle path the inputs name, and whether each lies in the store.

    Attributes:
        ok: Whether every path resolves inside the store.
        store_root: The store.
        bundles: One record per distinct path (``path``, ``sources``,
            ``real_path``, ``ok``, ``reason``).
        reasons: Why the check failed (empty when ``ok``).
    """

    ok: bool
    store_root: str
    bundles: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def bundle_paths(payload: Any, source: str) -> list[tuple[str, str]]:
    """Return every bundle path in a JSON payload, with where it was found.

    A bundle path is a ``bundle_path`` value, or the ``path`` of an object
    that also carries a ``build_hash`` (a bundle ref, a region-share record).

    Args:
        payload: Parsed JSON.
        source: A label for the payload.

    Returns:
        ``(path, "<source>:<json pointer>")`` pairs.
    """
    found: list[tuple[str, str]] = []

    def walk(node: Any, pointer: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                here = f"{pointer}/{key}"
                if isinstance(value, str) and (
                    key == "bundle_path" or (key == "path" and "build_hash" in node)
                ):
                    found.append((value, f"{source}:{here}"))
                else:
                    walk(value, here)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{pointer}/{index}")

    walk(payload, "")
    return found


def check_provenance(
    found: Sequence[tuple[str, str]], store_root: Path
) -> ProvenanceCheck:
    """Check that every bundle path resolves to a real path inside the store.

    A path outside the store as written (an SSD1 path, a relative path) is
    refused without touching the file system; a path inside it as written is
    resolved (symlinks followed) and must stay inside it and exist.

    Args:
        found: ``bundle_paths`` output over every input.
        store_root: The store (``STORE_ROOT``).

    Returns:
        The check (``ok`` is False when no bundle path was found).
    """
    root_text = os.path.normpath(str(store_root))
    real_root = Path(os.path.realpath(root_text))
    by_path: dict[str, list[str]] = {}
    for path, source in found:
        by_path.setdefault(path, []).append(source)
    check = ProvenanceCheck(ok=True, store_root=root_text)
    for path, sources in sorted(by_path.items()):
        text = os.path.normpath(path)
        record: dict[str, Any] = {
            "path": path,
            "sources": sources,
            "real_path": None,
            "ok": False,
            "reason": "",
        }
        if not os.path.isabs(text):
            record["reason"] = "not an absolute path"
        elif not text.startswith(root_text + os.sep):
            record["reason"] = f"outside {root_text} as written (not resolved)"
        else:
            real = Path(os.path.realpath(text))
            record["real_path"] = str(real)
            if real == real_root or not real.is_relative_to(real_root):
                record["reason"] = f"resolves outside the store: {real}"
            elif not real.is_dir():
                record["reason"] = "no such bundle directory"
            else:
                record["ok"] = True
        if not record["ok"]:
            check.reasons.append(f"{path}: {record['reason']} (from {sources[0]})")
        check.bundles.append(record)
    if not check.bundles:
        check.reasons.append("no bundle path found in the inputs")
    check.ok = not check.reasons
    return check


# ---------------------------------------------------------------------------
# inputs


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _one(paths: Iterable[Path]) -> Path | None:
    found = sorted(paths)
    return found[0] if len(found) == 1 else None


@dataclass
class SegOutputs:
    """One dataset x segmentation's published outputs (read-only).

    Attributes:
        dataset: ``ms1`` or ``ms2``.
        segmentation: The segmentation.
        root: The dataset's results root (``<outdir>/<pair>``).
        report: ``acceptance_metrics.json``, if present.
        manifest: ``map_manifest.json``, if present.
        resolve_summary: ``<pair>_resolve_summary.json``, if present.
        labels_path: The label table, if exactly one.
        files: Every file read (for the input digests).
    """

    dataset: str
    segmentation: str
    root: Path
    report: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    resolve_summary: dict[str, Any] | None = None
    labels_path: Path | None = None
    files: list[Path] = field(default_factory=list)

    @property
    def map_dir(self: SegOutputs) -> Path:
        """The published ``annotation_map_out``."""
        return self.root / MAP_DIR.format(seg=self.segmentation)

    @property
    def pair_id(self: SegOutputs) -> str:
        """The pair id the trace tags carry."""
        for source in (self.report, self.manifest, self.resolve_summary):
            if source and source.get("pair_id"):
                return str(source["pair_id"])
        return self.root.name

    @property
    def records(self: SegOutputs) -> list[dict[str, Any]]:
        """The report's metric records."""
        return list((self.report or {}).get("metrics") or [])

    def record(
        self: SegOutputs, criterion: str, name: str, **match: Any
    ) -> dict[str, Any] | None:
        """Return the first metric record of a criterion and name."""
        for record in self.records:
            if (
                record.get("criterion") == criterion
                and record.get("name") == name
                and all(record.get(key) == value for key, value in match.items())
            ):
                return record
        return None

    def value(self: SegOutputs, criterion: str, name: str, **match: Any) -> Any:
        """Return a measured record's value (``None`` otherwise)."""
        record = self.record(criterion, name, **match)
        if record is None or record.get("status") not in (None, "measured"):
            return None
        return record.get("value")

    @property
    def sample(self: SegOutputs) -> dict[str, Any]:
        """The resolve summary's (single) sample record."""
        samples = (self.resolve_summary or {}).get("samples") or {}
        return dict(next(iter(samples.values()))) if len(samples) == 1 else {}

    @property
    def map_sample(self: SegOutputs) -> dict[str, Any]:
        """The MAP manifest's (single) sample record."""
        samples = (self.manifest or {}).get("samples") or {}
        return dict(next(iter(samples.values()))) if len(samples) == 1 else {}

    @property
    def gate(self: SegOutputs) -> dict[str, Any]:
        """The mouse gate record."""
        return dict(self.sample.get("mouse_gate") or {})

    @property
    def region_record(self: SegOutputs) -> dict[str, Any]:
        """The MAP region step record (``mouse_regions``)."""
        return dict(self.map_sample.get("mouse_regions") or {})

    def map_processes(self: SegOutputs) -> list[int]:
        """Every MapMyCells process count the MAP manifest records."""
        counts: list[int] = []
        manifest = self.manifest or {}
        if manifest.get("n_processors") is not None:
            counts.append(int(manifest["n_processors"]))
        for run in (self.map_sample.get("runs") or {}).values():
            value = (run.get("engine_params") or {}).get("n_processors")
            if value is not None:
                counts.append(int(value))
        remap = self.region_record.get("remap") or {}
        value = (remap.get("engine_params") or {}).get("n_processors")
        if value is not None:
            counts.append(int(value))
        return counts

    def region_share_dir(self: SegOutputs) -> Path | None:
        """The region-share bundle the run used."""
        share = self.region_record.get("region_share") or {}
        if share.get("path"):
            return Path(str(share["path"]))
        refs = ((self.report or {}).get("provenance") or {}).get("references") or {}
        path = (refs.get("wmb_region_share") or {}).get("bundle_path")
        return Path(str(path)) if path else None


def load_outputs(dataset: str, root: Path) -> dict[str, SegOutputs]:
    """Read every segmentation's published outputs under a results root."""
    out: dict[str, SegOutputs] = {}
    for segmentation in SEGMENTATIONS:
        item = SegOutputs(dataset=dataset, segmentation=segmentation, root=root)
        report = root / REPORT_PATH.format(seg=segmentation)
        if report.is_file():
            item.report = _read_json(report)
            item.files.append(report)
        manifest = item.map_dir / MAP_MANIFEST
        if manifest.is_file():
            item.manifest = _read_json(manifest)
            item.files.append(manifest)
        resolve = root / RESOLVE_DIR.format(seg=segmentation)
        summary = _one(resolve.glob("*_resolve_summary.json"))
        if summary is not None:
            item.resolve_summary = _read_json(summary)
            item.files.append(summary)
        item.labels_path = _one(resolve.glob("*/*_celltype_labels.parquet"))
        if item.labels_path is not None:
            item.files.append(item.labels_path)
        out[segmentation] = item
    return out


def _missing(outputs: Mapping[str, Mapping[str, SegOutputs]], dataset: str) -> str:
    if dataset not in outputs:
        return f"no {dataset} results root given (--dataset {dataset}=...)"
    return ""


def _get(
    outputs: Mapping[str, Mapping[str, SegOutputs]], dataset: str, segmentation: str
) -> SegOutputs | None:
    return (outputs.get(dataset) or {}).get(segmentation)


def _scored_pairs() -> list[tuple[str, str]]:
    return [(dataset, SCORED_SEGMENTATION[dataset]) for dataset in DATASETS]


def _unnamed_pairs() -> list[tuple[str, str]]:
    return [
        (dataset, segmentation)
        for dataset in DATASETS
        for segmentation in SCORED_UNNAMED[dataset]
    ]


def _all_pairs() -> list[tuple[str, str]]:
    return [(dataset, seg) for dataset in DATASETS for seg in SEGMENTATIONS]


# ---------------------------------------------------------------------------
# MO1: class marker consistency (mouse_corr_shadow's pseudo-labels)


def mo1_measure(
    dataset: str, labels: pd.DataFrame, pseudo: pd.DataFrame
) -> dict[str, Any]:
    """Return MO1 over all marker-pseudo-confident table cells (M6's method).

    Args:
        dataset: ``ms1`` or ``ms2`` (the pseudo-label method).
        labels: The label table, indexed by cell id.
        pseudo: The dataset's marker pseudo-labels, indexed by cell id.

    Returns:
        ``mo1_pruned`` (the scored value: production labels), ``mo1_unpruned``,
        ``confident_cell_value`` (marker consistency of the confident class
        calls) and ``n_pseudo_confident``.
    """
    scorer = corr.score_ms1 if dataset == "ms1" else corr.score_ms2
    scored = scorer(labels, pseudo)
    frame, mo1 = corr.dataset_tables(labels, scored)
    row = frame[(frame["floor"] == "none") & (frame["depth_bin"] == "all")]
    confident = row["cons"].iloc[0] if len(row) else None
    return {
        **mo1,
        "confident_cell_value": None if confident is None else _float(confident),
    }


def mo1_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]],
    pseudo_paths: Mapping[str, Path],
    measured: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Row]:
    """MO1: MS1 proseg_hybrid >= 0.89, MS2 original_seg >= 0.84 (§9).

    Args:
        outputs: The datasets' outputs.
        pseudo_paths: Dataset to its pseudo-label CSV.
        measured: Precomputed ``mo1_measure`` results (tests).

    Returns:
        One scored row per dataset.
    """
    rows = []
    for dataset, segmentation in _scored_pairs():
        out = _get(outputs, dataset, segmentation)
        result = dict((measured or {}).get(dataset) or {})
        note = _missing(outputs, dataset)
        if not result and not note:
            pseudo = pseudo_paths.get(dataset)
            if out is None or out.labels_path is None:
                note = "no label table"
            elif pseudo is None or not Path(pseudo).is_file():
                note = f"no pseudo-labels at {pseudo}"
            else:
                labels = corr._read_labels(out.labels_path)
                frame = pd.read_csv(pseudo, index_col=0)
                frame.index = frame.index.astype(str)
                result = mo1_measure(dataset, labels, frame)
        value = result.get("mo1_pruned")
        if result and not result.get("n_pseudo_confident"):
            note = (
                "no marker-pseudo-confident table cell (pseudo-labels of other cells?)"
            )
            value = None
        g2 = out.value("MO1", "g2_marker_consistency") if out is not None else None
        rows.append(
            Row(
                "MO1",
                "marker_consistency",
                dataset,
                segmentation,
                value,
                ">=",
                MO1_MIN[dataset],
                at_least(value, MO1_MIN[dataset]),
                True,
                "label table + mouse_corr_shadow pseudo-labels",
                note=note
                or "pruned (production) class calls over all marker-pseudo-confident "
                "table cells (fixed denominator)",
                beside={
                    "mo1_unpruned": result.get("mo1_unpruned"),
                    "confident_cell_value": result.get("confident_cell_value"),
                    "n_pseudo_confident": result.get("n_pseudo_confident"),
                    "g2_marker_consistency": g2,
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# MO2: pruning mechanics (subset and full re-map)


def mo2_from_calls(
    *,
    unpruned_class: np.ndarray,
    unpruned_subclass: np.ndarray,
    pruned_class: np.ndarray,
    pruned_subclass: np.ndarray,
    pruned_subclass_name: np.ndarray,
    remapped: np.ndarray,
    t2: Iterable[str],
) -> dict[str, Any]:
    """Return MO2's three values for one re-map (M6 stage A's definitions).

    Args:
        unpruned_class: Unpruned class call per table cell.
        unpruned_subclass: Unpruned subclass call per table cell.
        pruned_class: The re-map's class call per table cell.
        pruned_subclass: The re-map's subclass call per table cell.
        pruned_subclass_name: The re-map's subclass name (T2 is by name).
        remapped: Cells whose unpruned class or subclass was dropped.
        t2: The section's T2 subclass names (``mouse_gate.t2_subclasses``).

    Returns:
        ``t2_after`` (share of table cells whose call is T2),
        ``class_changes_outside`` and ``subclass_changes_outside`` (share of
        the cells outside dropped nodes whose call changed), and the counts.
    """
    outside = ~np.asarray(remapped, dtype=bool)
    n_outside = int(outside.sum())
    n_cells = len(outside)
    t2_set = {str(name) for name in t2}
    in_t2 = np.array([str(name) in t2_set for name in pruned_subclass_name], dtype=bool)

    def changed(before: np.ndarray, after: np.ndarray) -> float | None:
        if not n_outside:
            return None
        differ = np.asarray(before, dtype=object) != np.asarray(after, dtype=object)
        return float(differ[outside].sum() / n_outside)

    return {
        "t2_after": float(in_t2.mean()) if n_cells else None,
        "class_changes_outside": changed(unpruned_class, pruned_class),
        "subclass_changes_outside": changed(unpruned_subclass, pruned_subclass),
        "n_cells": n_cells,
        "n_outside": n_outside,
        "n_remapped": n_cells - n_outside,
        "n_t2": int(in_t2.sum()),
    }


def mo2_measure(map_dir: Path, full_dir: Path | None = None) -> dict[str, Any]:
    """Return MO2 for the subset re-map and, if given, the full re-map.

    Reads the MAP output as RESOLVE does (``pipeline.load_resolve_runs``);
    T2 is G3's definition (``mouse_gate.t2_subclasses`` with the region
    step's present regions, the region-share bundle, the static never-drop
    classes and the cell minimum the step used).

    Args:
        map_dir: The published ``annotation_map_out``.
        full_dir: A ``full-remap`` output directory.

    Returns:
        ``subset`` and ``full`` (``mo2_from_calls`` results or ``None``) and
        the ``full_record`` (the full re-map's ``full_remap.json``), the MAP
        process count and the region step's status.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mapmycells_engine import level_frame, read_tidy_parquet
    from merxen.annotation.mouse_gate import t2_subclasses
    from merxen.annotation.mouse_resolve import subclass_classes
    from merxen.annotation.pipeline import load_map_manifest, load_resolve_runs

    manifest = load_map_manifest(map_dir / MAP_MANIFEST)
    (record,) = manifest.samples.values()
    run = load_resolve_runs(map_dir, record)[PRIMARY_RUN]
    regions = run.regions
    if regions is None or regions.region_share is None:
        raise ValueError(f"{map_dir}: no region step with a region-share bundle")
    ids = level_frame(regions.unpruned, CLASS_LEVEL).index.astype(str)
    remapped = np.array([value is not None for value in regions.dropped_level(ids)])
    config = AnnotationConfig.model_validate({"species": "mouse"})
    params = dict(regions.record.params or {})
    t2 = t2_subclasses(
        list(regions.record.present_regions),
        regions.region_share.shares,
        subclass_class=subclass_classes(run),
        never_drop_classes=list(
            params.get("never_drop_classes") or config.mouse_regions.never_drop_classes
        ),
        min_merfish_cells=int(
            params.get("min_merfish_cells_subclass")
            or config.mouse_regions.min_merfish_cells_subclass
        ),
        max_present_share=config.mouse_gate.g3_t2_max_present_share,
    )

    def calls(tidy: pd.DataFrame, level: str, column: str) -> np.ndarray:
        frame = level_frame(tidy, level)
        frame.index = frame.index.astype(str)
        return frame.reindex(ids)[column].astype(object).to_numpy()

    base = {
        "unpruned_class": calls(regions.unpruned, CLASS_LEVEL, "assignment"),
        "unpruned_subclass": calls(regions.unpruned, SUBCLASS_LEVEL, "assignment"),
    }

    def measure(tidy: pd.DataFrame) -> dict[str, Any]:
        return mo2_from_calls(
            **base,
            pruned_class=calls(tidy, CLASS_LEVEL, "assignment"),
            pruned_subclass=calls(tidy, SUBCLASS_LEVEL, "assignment"),
            pruned_subclass_name=calls(tidy, SUBCLASS_LEVEL, "name"),
            remapped=remapped,
            t2=t2,
        )

    result: dict[str, Any] = {
        "subset": measure(regions.pruned),
        "full": None,
        "full_record": None,
        "map_processes": int(manifest.n_processors),
        "region_step_status": str(regions.record.status),
        "n_t2_subclasses": len(t2),
    }
    if full_dir is not None:
        full_record = _read_json(full_dir / FULL_REMAP_RECORD)
        result["full_record"] = full_record
        nodes = sorted(tuple(node.as_pair()) for node in regions.record.nodes_to_drop)
        differ = [
            name
            for name, ok in (
                ("bundle", full_record.get("build_hash") == run.record.build_hash),
                (
                    "query",
                    full_record.get("query_fingerprint")
                    == run.record.query_fingerprint,
                ),
                (
                    "drop list",
                    sorted(
                        tuple(pair) for pair in full_record.get("nodes_to_drop") or []
                    )
                    == nodes,
                ),
            )
            if not ok
        ]
        if differ:
            result["full_problem"] = (
                f"the full re-map in {full_dir} was not made from this MAP output "
                f"({', '.join(differ)} differ)"
            )
        else:
            full, _ = read_tidy_parquet(full_dir / str(full_record["parquet"]))
            result["full"] = measure(full)
    return result


def mo2_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]],
    measured: Mapping[tuple[str, str], Mapping[str, Any]],
    reading: str,
    problems: Mapping[tuple[str, str], str] | None = None,
) -> list[Row]:
    """MO2 on every scored segmentation (``SCORED_UNNAMED``), per re-map and part.

    Args:
        outputs: The datasets' outputs.
        measured: (dataset, segmentation) to ``mo2_measure`` output.
        reading: ``both`` (registered), ``subset`` or ``full``.
        problems: (dataset, segmentation) to why it was not measured.

    Returns:
        Per dataset, re-map and part one row (the re-map the reading does not
        score is reported), plus the report's pruned-calls-in-dropped-nodes
        count (reported).
    """
    rows: list[Row] = []
    scored_remaps = {
        "both": ("subset", "full"),
        "subset": ("subset",),
        "full": ("full",),
    }[reading]
    for dataset, segmentation in _unnamed_pairs():
        result = measured.get((dataset, segmentation)) or {}
        why = (problems or {}).get((dataset, segmentation)) or _missing(
            outputs, dataset
        )
        for remap in ("subset", "full"):
            values = result.get(remap)
            note = why or (
                ""
                if values
                else (remap == "full" and result.get("full_problem"))
                or f"no {remap} re-map (score --mo2-full {dataset}/{segmentation}=DIR)"
            )
            unscored = ""
            if remap == "full" and values:
                full_record = result.get("full_record") or {}
                counts = {
                    "full re-map": full_record.get("n_processors"),
                    "MAP": result.get("map_processes"),
                }
                if counts["full re-map"] != counts["MAP"]:
                    unscored = UNSCORED_PROCESSES
                    note = (
                        f"the full re-map ran with {counts['full re-map']} processes, "
                        f"the MAP run with {counts['MAP']} (§25.2: the MAP run's own)"
                    )
            for part, limit in MO2_LIMITS.items():
                value = (values or {}).get(part)
                rows.append(
                    Row(
                        "MO2",
                        part,
                        dataset,
                        segmentation,
                        value,
                        "<=",
                        limit,
                        at_most(value, limit),
                        remap in scored_remaps,
                        f"MAP output ({remap} re-map)",
                        note=note,
                        scope={"remap": remap},
                        beside={
                            key: (values or {}).get(key)
                            for key in ("n_cells", "n_outside", "n_remapped", "n_t2")
                        },
                        unscored=unscored if remap in scored_remaps else "",
                    )
                )
        out = _get(outputs, dataset, segmentation)
        report_value = (
            out.value("MO2", "pruned_calls_in_dropped_nodes")
            if out is not None
            else None
        )
        rows.append(
            Row(
                "MO2",
                "pruned_calls_in_dropped_nodes",
                dataset,
                segmentation,
                report_value,
                "reported",
                None,
                None,
                False,
                "acceptance_metrics.json",
                beside={
                    "class_changes_outside_dropped (report, over table cells)": (
                        out.value("MO2", "class_changes_outside_dropped")
                        if out is not None
                        else None
                    )
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# MO3: inferred regions = the manual sets


def mo3_rows(outputs: Mapping[str, Mapping[str, SegOutputs]]) -> list[Row]:
    """MO3: the inferred regions equal the manual set, per segmentation.

    Scored on every segmentation of ``SCORED_UNNAMED`` (§25.2.1).
    """
    rows = []
    for dataset, segmentation in _all_pairs():
        out = _get(outputs, dataset, segmentation)
        inferred: list[str] | None = None
        if out is not None and out.region_record:
            inferred = [
                str(region)
                for region in out.region_record.get("inferred_regions") or []
            ]
        elif out is not None and out.value("MO3", "inferred_regions"):
            inferred = str(out.value("MO3", "inferred_regions")).split(";")
        manual = sorted(MANUAL_REGIONS[dataset])
        rows.append(
            Row(
                "MO3",
                "inferred_regions",
                dataset,
                segmentation,
                None if inferred is None else ";".join(sorted(inferred)),
                "==",
                ";".join(manual),
                None if inferred is None else sorted(inferred) == manual,
                segmentation in SCORED_UNNAMED[dataset],
                "map_manifest.json (mouse_regions)",
                note=_missing(outputs, dataset)
                or (
                    "" if out is None or out.region_record else "no region step record"
                ),
                beside={"source": (out.region_record.get("source") if out else None)},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# MO4: soft class composition vs the AP-matched MERFISH window


def window_shares(
    bundle: Path | None, sections: Sequence[str]
) -> dict[str, float] | None:
    """Return the pooled class shares of MERFISH sections (G4's expectation)."""
    from merxen.annotation.report_mouse import window_composition

    frame = window_composition(bundle, list(sections))
    if frame is None:
        return None
    return {str(row[0]): float(row[1]) for row in frame.itertuples(index=False)}


def gate_window(out: SegOutputs) -> list[str]:
    """The AP-estimate window the gate used (``mouse_gate.window``)."""
    gate = out.gate
    sections = (gate.get("window") or {}).get("sections") or (
        (gate.get("ap_estimate") or {}).get("window_sections") or []
    )
    return [str(section) for section in sections]


def mo4_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]],
    reading: str,
    windows: Mapping[tuple[str, tuple[str, ...]], dict[str, float] | None]
    | None = None,
) -> list[Row]:
    """MO4: soft Astro-Epen and Immune shares against the MERFISH window.

    Args:
        outputs: The datasets' outputs.
        reading: ``fixed`` (registered: .31-.33) or ``ap_estimate``.
        windows: Precomputed window shares by (dataset, sections) (tests).

    Returns:
        Per dataset and class one row in points (100 x (dataset - window)).
    """
    rows = []
    for dataset, segmentation in _scored_pairs():
        out = _get(outputs, dataset, segmentation)
        ap_window = gate_window(out) if out is not None else []
        bundle = out.region_share_dir() if out is not None else None

        def shares(
            sections: Sequence[str], d: str = dataset, b: Path | None = bundle
        ) -> Any:
            key = (d, tuple(sections))
            if windows is not None and key in windows:
                return windows[key]
            return window_shares(b, sections) if sections else None

        fixed = shares(MO4_FIXED_WINDOW)
        estimated = shares(ap_window)
        scored_window = fixed if reading == "fixed" else estimated
        for key, name in MO4_CLASSES.items():
            share = out.value("MO4", f"{key}_soft_share") if out is not None else None
            band = MO4_BANDS[dataset][key]

            def points(
                window: dict[str, float] | None, s: Any = share, n: str = name
            ) -> Any:
                if window is None or _float(s) is None or n not in window:
                    return None
                return 100.0 * (float(s) - window[n])

            value = points(scored_window)
            note = _missing(outputs, dataset) or (
                "" if scored_window is not None else f"no {reading} window composition"
            )
            rows.append(
                Row(
                    "MO4",
                    f"{key}_points",
                    dataset,
                    segmentation,
                    value,
                    "|x| <=",
                    band,
                    None if value is None else abs(value) <= band + TOLERANCE,
                    True,
                    "acceptance_metrics.json soft share; wmb_region_share "
                    "section_composition",
                    note=note,
                    scope={"window": reading},
                    beside={
                        "dataset_share": share,
                        "fixed_window": list(MO4_FIXED_WINDOW),
                        "fixed_window_share": (fixed or {}).get(name),
                        "fixed_points": points(fixed),
                        "ap_estimate_window": ap_window,
                        "ap_estimate_window_share": (estimated or {}).get(name),
                        "ap_estimate_points": points(estimated),
                        "gate_g4_points": (
                            out.value("MO4", f"g4_{key}_points")
                            if out is not None
                            else None
                        ),
                    },
                )
            )
    return rows


# ---------------------------------------------------------------------------
# MO5, MO6, MO7


def mo5_rows(outputs: Mapping[str, Mapping[str, SegOutputs]]) -> list[Row]:
    """MO5: MS2 held-out enrichment >= 10x; astrocyte FPR <= 0.5%.

    Both are scored on every segmentation of ``SCORED_UNNAMED`` (§25.2.1):
    the enrichment on the second section's (§9 names that section), the
    astrocyte FPR on both sections', as the M6 exit measured it. Other
    segmentations are reported.
    """
    rows = []
    for dataset, segmentation in _all_pairs():
        out = _get(outputs, dataset, segmentation)
        primary = segmentation in SCORED_UNNAMED[dataset]
        note = _missing(outputs, dataset)
        enrichment = out.value("MO5", "spillover_heldout_enrichment") if out else None
        rows.append(
            Row(
                "MO5",
                "heldout_enrichment",
                dataset,
                segmentation,
                enrichment,
                ">=",
                MO5_ENRICHMENT_MIN,
                at_least(enrichment, MO5_ENRICHMENT_MIN),
                primary and dataset == "ms2",
                "acceptance_metrics.json (resolve summary spillover_checks)",
                note=note,
            )
        )
        fpr = out.value("MO5", "spillover_astrocyte_fpr") if out else None
        rows.append(
            Row(
                "MO5",
                "astrocyte_fpr",
                dataset,
                segmentation,
                fpr,
                "<=",
                MO5_FPR_MAX,
                at_most(fpr, MO5_FPR_MAX),
                primary,
                "acceptance_metrics.json (resolve summary spillover_checks)",
                note=note,
            )
        )
    return rows


def mo6_rows(outputs: Mapping[str, Mapping[str, SegOutputs]]) -> list[Row]:
    """MO6: the F1 (region-incoherent) rate after pruning lies in 0.2-1.0%.

    Scored on every segmentation of ``SCORED_UNNAMED`` (§25.2.1).
    """
    rows = []
    for dataset, segmentation in _all_pairs():
        out = _get(outputs, dataset, segmentation)
        rate = out.value("MO6", "f1_region_incoherent_rate") if out else None
        low, high = MO6_RANGE
        number = _float(rate)
        rows.append(
            Row(
                "MO6",
                "f1_rate",
                dataset,
                segmentation,
                rate,
                "in",
                [low, high],
                None
                if number is None
                else low - TOLERANCE <= number <= high + TOLERANCE,
                segmentation in SCORED_UNNAMED[dataset],
                "acceptance_metrics.json (resolve summary region_coherence)",
                note=_missing(outputs, dataset),
            )
        )
    return rows


def gate_level(out: SegOutputs | None) -> str | None:
    """The mouse gate level (resolve summary, else the report)."""
    if out is None:
        return None
    level = out.gate.get("level") or out.value("MO7", "mouse_gate_level")
    return None if level is None else str(level)


def negative_control_files(path: Path) -> tuple[Path | None, Path | None]:
    """Return the negative control's resolve summary and MAP manifest.

    Args:
        path: The resolve summary itself, or a directory holding the
            standalone run's MAP and RESOLVE outputs (searched recursively;
            each file must be unique).

    Returns:
        ``(resolve summary, map_manifest.json)``, ``None`` where not found.
    """
    if path.is_file():
        return path, None
    if not path.is_dir():
        return None, None
    return (
        _one(path.rglob("*_resolve_summary.json")),
        _one(path.rglob(MAP_MANIFEST)),
    )


def mo7_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]],
    negative: Mapping[str, Any] | None,
    negative_path: Path | None,
    *,
    negative_bundles_checked: bool = True,
) -> list[Row]:
    """MO7: the invalid MS2 proseg_hybrid ``failed``; the scored sets not.

    Args:
        outputs: The datasets' outputs.
        negative: The negative control's resolve summary (§25.2: a standalone
            run on the legacy-snapshot H5AD with M6's G1 JSON).
        negative_path: Its path.
        negative_bundles_checked: False when its MAP manifest was not found,
            so its bundles were not checked (§25.3): the row is unscored.

    Returns:
        The negative-control row and one not-failed row per segmentation
        (scored on the scored segmentations).
    """
    rows = []
    samples = (negative or {}).get("samples") or {}
    sample = next(iter(samples.values())) if len(samples) == 1 else {}
    gate = dict(sample.get("mouse_gate") or {})
    level = gate.get("level")
    note, unscored = "", ""
    if not negative:
        note = "no negative-control run (score --mo7-negative)"
    elif not negative_bundles_checked:
        note = "its map_manifest.json was not found: its bundles are not checked"
        unscored = UNSCORED_PROVENANCE
    rows.append(
        Row(
            "MO7",
            "negative_control_failed",
            NEGATIVE_CONTROL,
            "proseg_hybrid",
            level,
            "==",
            MO7_FAILED,
            None if level is None else str(level) == MO7_FAILED,
            True,
            f"resolve summary {negative_path}" if negative_path else "resolve summary",
            note=note,
            unscored=unscored,
            beside={
                "level_reasons": gate.get("level_reasons"),
                "signal_status": gate.get("signal_status"),
                "warning_reasons": gate.get("warning_reasons"),
            },
        )
    )
    for dataset, segmentation in _all_pairs():
        out = _get(outputs, dataset, segmentation)
        level = gate_level(out)
        rows.append(
            Row(
                "MO7",
                "not_failed",
                dataset,
                segmentation,
                level,
                "!=",
                MO7_FAILED,
                None if level is None else level != MO7_FAILED,
                segmentation == SCORED_SEGMENTATION[dataset],
                "resolve summary mouse_gate",
                note=_missing(outputs, dataset),
                beside={
                    "warning": (out.gate.get("warning") if out else None),
                    "level_reasons": (out.gate.get("level_reasons") if out else None),
                    "warning_reasons": (
                        out.gate.get("warning_reasons") if out else None
                    ),
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# MO8: resources


def read_sampler(path: Path) -> dict[str, float]:
    """Return the peaks of a G0-style process-tree sampler CSV (GiB).

    Args:
        path: ``sampler.csv`` (``vmrss_kb``, ``vmhwm_kb``, ``pss_kb`` summed
            over the process tree per sample).

    Returns:
        ``tree_pss_gb``, ``summed_vmhwm_gb`` (the trace-style peak) and
        ``summed_vmrss_gb``.
    """
    frame = pd.read_csv(path)
    return {
        "tree_pss_gb": float(frame["pss_kb"].max()) / KIB_PER_GIB,
        "summed_vmhwm_gb": float(frame["vmhwm_kb"].max()) / KIB_PER_GIB,
        "summed_vmrss_gb": float(frame["vmrss_kb"].max()) / KIB_PER_GIB,
    }


def _tag_parts(tag: Any) -> list[str]:
    return [part for part in str(tag).split(":") if part]


def map_process_counts(trace: pd.DataFrame, out: SegOutputs) -> list[int]:
    """Every MapMyCells process count of a segmentation (trace and manifest)."""
    counts = list(out.map_processes())
    tasks = trace[
        (trace["process"] == MAP_PROCESS)
        & trace["tag"].map(
            lambda tag: _tag_parts(tag)[:2] == [out.pair_id, out.segmentation]
        )
    ]
    for value in tasks["cpus"]:
        if str(value).strip().isdigit():
            counts.append(int(str(value).strip()))
    return counts


def mo8_section_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]],
    trace: pd.DataFrame,
    samplers: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    memory_reading: str,
    processes: int,
) -> tuple[list[Row], list[str]]:
    """MO8 per 4-segmentation section: H14's time sum and MAP peak (§25.1).

    Args:
        outputs: The datasets' outputs.
        trace: ``run_acceptance.load_trace`` output.
        samplers: (dataset, segmentation) to ``read_sampler`` output.
        memory_reading: ``trace_rss`` (registered) or ``tree_pss``.
        processes: The registered MAP process count (4).

    Returns:
        The rows and the problems (a process count other than the reading's).
    """
    rows: list[Row] = []
    problems: list[str] = []
    done = trace["status"].isin(["COMPLETED", "CACHED"])
    for dataset in DATASETS:
        segs = outputs.get(dataset) or {}
        pair_ids = {out.pair_id for out in segs.values()}
        tasks = trace[
            trace["process"].isin(ANNOTATION_PROCESSES)
            & trace["tag"].map(
                lambda tag, p=pair_ids: (
                    bool(_tag_parts(tag)[:1]) and _tag_parts(tag)[0] in p
                )
            )
        ]
        covered = sorted(
            {_tag_parts(tag)[1] for tag in tasks["tag"] if len(_tag_parts(tag)) > 1}
        )
        missing = sorted(set(SEGMENTATIONS) - set(covered))
        incomplete = bool(missing) or tasks.empty or not bool(done[tasks.index].all())
        counts = sorted(
            {count for out in segs.values() for count in map_process_counts(trace, out)}
        )
        unscored = ""
        if counts != [processes] and dataset in outputs:
            unscored = UNSCORED_PROCESSES
            problems.append(
                f"MO8 {dataset}: MAP ran with {counts or 'unrecorded'} MapMyCells "
                f"processes, not {processes} (§25.1): the section's MO8 rows are "
                "not scored"
            )
        why = _missing(outputs, dataset)
        # A task without its realtime would drop out of the sum: not measured.
        no_time = bool(tasks["realtime_s"].isna().any())
        incomplete = incomplete or no_time
        wall_h = float(tasks["realtime_s"].sum()) / 3600.0 if not tasks.empty else None
        rows.append(
            Row(
                "MO8",
                "section_time",
                dataset,
                "all",
                None if incomplete else wall_h,
                "<=",
                MO8_SECTION_MAX_H,
                None if incomplete else at_most(wall_h, MO8_SECTION_MAX_H),
                True,
                "trace (PANEL + MAP + RESOLVE realtime)",
                note=why
                or (
                    f"{len(tasks)} tasks over {','.join(covered) or 'no segmentation'}"
                    + (f"; missing {','.join(missing)}" if missing else "")
                    + (
                        ""
                        if tasks.empty or done[tasks.index].all()
                        else "; a task did not complete"
                    )
                    + ("; a task has no realtime" if no_time else "")
                ),
                beside={"map_processes": counts},
                unscored=unscored,
            )
        )
        maps = tasks[tasks["process"] == MAP_PROCESS]
        # A MAP task without peak_rss: the others' maximum is not the peak.
        no_peak = bool(maps["peak_rss_gb"].isna().any())
        trace_peak = (
            float(maps["peak_rss_gb"].max()) if len(maps) and not no_peak else None
        )
        sampled = {seg: samplers.get((dataset, seg)) for seg in SEGMENTATIONS}
        pss_values = [values["tree_pss_gb"] for values in sampled.values() if values]
        pss_peak = max(pss_values) if len(pss_values) == len(SEGMENTATIONS) else None
        map_missing = sorted(
            set(SEGMENTATIONS)
            - {_tag_parts(tag)[1] for tag in maps["tag"] if len(_tag_parts(tag)) > 1}
        )
        value = (
            (trace_peak if not map_missing else None)
            if memory_reading == "trace_rss"
            else pss_peak
        )
        rows.append(
            Row(
                "MO8",
                "map_peak",
                dataset,
                "all",
                value,
                "<=",
                MO8_MAP_PEAK_MAX_GB,
                at_most(value, MO8_MAP_PEAK_MAX_GB),
                True,
                "trace peak_rss of the MAP tasks (GB = 1024^3 bytes)"
                if memory_reading == "trace_rss"
                else "sampler process-tree PSS (GiB)",
                note=why
                or (
                    f"MAP tasks missing for {','.join(map_missing)}"
                    if map_missing
                    else ""
                )
                or (
                    "a MAP task has no peak_rss"
                    if no_peak and memory_reading == "trace_rss"
                    else ""
                )
                or ("" if value is not None else "no sampler for every segmentation"),
                beside={
                    "trace_peak_rss_gb": trace_peak,
                    "sampler_tree_pss_gb": {
                        seg: (values or {}).get("tree_pss_gb")
                        for seg, values in sampled.items()
                    },
                    "sampler_summed_vmhwm_gb": {
                        seg: (values or {}).get("summed_vmhwm_gb")
                        for seg, values in sampled.items()
                    },
                    "map_processes": counts,
                },
                unscored=unscored,
            )
        )
    return rows, problems


def _bundle_record(path: Path) -> dict[str, Any]:
    # The store layout is <store>/<reference_id>/<build_hash>.
    record: dict[str, Any] = {
        "path": str(path),
        "reference_id": path.parent.name,
        "build_hash": path.name,
    }
    bundle_json = path / "bundle.json"
    if not bundle_json.is_file():
        record["error"] = "no bundle.json"
        return record
    bundle = _read_json(bundle_json)
    output = bundle.get("builder_output") or {}
    steps = output.get("ctm_steps") or {}
    tree_rss = [_float(step.get("peak_tree_rss_gb")) for step in steps.values()]
    tree_pss = [_float(step.get("peak_tree_pss_gb")) for step in steps.values()]
    single = [_float(step.get("peak_rss_gb")) for step in steps.values()]
    runs = list(output.get("mapping_runs") or [])
    summary_path = path / "resolvability_summary.json"
    would_raise = None
    if summary_path.is_file():
        summary = _read_json(summary_path)
        runs = runs or list(summary.get("mapping_runs") or [])
        would_raise = summary.get("validated_thresholds_would_raise")
    panel = bundle.get("panel") or {}
    record.update(
        {
            "reference_id": bundle.get("reference_id") or path.parent.name,
            "build_hash": bundle.get("build_hash") or path.name,
            "panel_hash": panel.get("panel_hash"),
            "wall_time_s": _float(bundle.get("wall_time_s")),
            "tree_rss_gb": max((v for v in tree_rss if v is not None), default=None),
            "tree_pss_gb": max((v for v in tree_pss if v is not None), default=None),
            "single_process_gb": max(
                (v for v in single if v is not None), default=None
            ),
            "ctm_steps": sorted(steps),
            "mapping_processes": sorted(
                {int(run["n_processors"]) for run in runs if run.get("n_processors")}
            ),
            "would_raise": would_raise,
        }
    )
    return record


def _bundle_label(record: Mapping[str, Any]) -> str:
    return f"{record.get('reference_id')} {str(record.get('build_hash') or '')[:8]}"


def mo8_prep_rows(
    bundles: Sequence[Path],
    trace: pd.DataFrame,
    *,
    memory_reading: str,
) -> list[Row]:
    """MO8 per PREP bundle: wall <= 2 h and peak <= 40 GB at 8 processes.

    The ``wmb_panel`` bundles are scored; every other bundle the run used
    (``wmb_region_share``, ``wmb_selfmap_testset``) is reported beside them.

    Args:
        bundles: The bundle directories (inside the store; provenance first).
        trace: ``load_trace`` output (PREP tasks: ``<reference>:<hash>`` tags).
        memory_reading: ``tree_rss`` (registered), ``tree_pss`` or
            ``single_process``.

    Returns:
        Two rows per bundle.
    """
    rows = []
    preps = trace[trace["process"] == PREP_PROCESS]
    for path in sorted(set(bundles)):
        record = _bundle_record(path)
        reference = str(record.get("reference_id") or "")
        scored = reference == MO8_PANEL_REFERENCE
        label = _bundle_label(record)
        prefix = str(record.get("panel_hash") or record.get("build_hash") or "")[:12]
        tasks = preps[
            preps["tag"].map(
                lambda tag, r=reference, p=prefix: (
                    _tag_parts(tag)[:1] == [r]
                    and len(_tag_parts(tag)) > 1
                    and bool(p)
                    and _tag_parts(tag)[1].startswith(p)
                )
            )
        ]
        trace_peak = float(tasks["peak_rss_gb"].max()) if len(tasks) else None
        wall = record.get("wall_time_s")
        rows.append(
            Row(
                "MO8",
                "prep_wall",
                label,
                "PREP",
                None if wall is None else wall / 3600.0,
                "<=",
                MO8_PREP_WALL_MAX_H,
                at_most(None if wall is None else wall / 3600.0, MO8_PREP_WALL_MAX_H),
                scored,
                "bundle.json wall_time_s (build incl. resolvability)",
                note=record.get("error", ""),
                scope={"bundle": str(record.get("build_hash") or path.name)},
            )
        )
        tree_rss = record.get("tree_rss_gb")
        strict = max((v for v in (tree_rss, trace_peak) if v is not None), default=None)
        value = {
            "tree_rss": strict,
            "tree_pss": record.get("tree_pss_gb"),
            "single_process": record.get("single_process_gb"),
        }[memory_reading]
        processes = record.get("mapping_processes") or []
        unscored = ""
        note = record.get("error", "")
        if scored and not record.get("error") and processes != [MO8_PREP_PROCESSES]:
            unscored = UNSCORED_PROCESSES
            note = (
                f"built with {processes or 'unrecorded'} processes, not "
                f"{MO8_PREP_PROCESSES} (resolvability mapping_runs)"
            )
        rows.append(
            Row(
                "MO8",
                "prep_peak",
                label,
                "PREP",
                value,
                "<=",
                MO8_PREP_PEAK_MAX_GB,
                at_most(value, MO8_PREP_PEAK_MAX_GB),
                scored,
                {
                    "tree_rss": "bundle.json ctm steps peak_tree_rss_gb and the "
                    "trace peak_rss",
                    "tree_pss": "bundle.json ctm steps peak_tree_pss_gb",
                    "single_process": "bundle.json ctm steps peak_rss_gb (GNU time)",
                }[memory_reading],
                note=note,
                scope={"bundle": str(record.get("build_hash") or path.name)},
                beside={
                    "tree_rss_gb": tree_rss,
                    "trace_peak_rss_gb": trace_peak,
                    "tree_pss_gb": record.get("tree_pss_gb"),
                    "single_process_gb": record.get("single_process_gb"),
                    "mapping_processes": processes,
                },
                unscored=unscored,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# MO9 [L]: MS2 registration and soft class JSD vs original_seg


def soft_class_vector(out: SegOutputs | None) -> dict[str, float] | None:
    """The report's soft class shares (allocated mass, the 34 WMB classes)."""
    if out is None:
        return None
    shares = {
        str(record.get("group")): float(record["value"])
        for record in out.records
        if record.get("criterion") == "MO4"
        and record.get("name") == "soft_share"
        and record.get("level") == "class"
        and record.get("value") is not None
    }
    return shares or None


def js_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Base-2 Jensen-Shannon distance of two renormalised compositions (M6)."""
    keys = sorted(set(p) | set(q))
    first = np.array([p.get(key, 0.0) for key in keys], dtype=np.float64)
    second = np.array([q.get(key, 0.0) for key in keys], dtype=np.float64)
    first, second = first / first.sum(), second / second.sum()
    middle = 0.5 * (first + second)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(first > 0, first * np.log2(first / middle), 0.0).sum()
        right = np.where(second > 0, second * np.log2(second / middle), 0.0).sum()
    return float(np.sqrt(max(0.5 * (left + right), 0.0)))


def mo9_rows(outputs: Mapping[str, Mapping[str, SegOutputs]]) -> list[Row]:
    """MO9 [L] (reported): MS2 registration passes; JSD vs original_seg <= 0.05.

    MO9 gates the use of the reseg and proseg outputs, not the flip (§9), so
    every row is reported. MS1's JSD against proseg_hybrid is information,
    as at the M6 exit.
    """
    rows = []
    for segmentation in SEGMENTATIONS:
        out = _get(outputs, "ms2", segmentation)
        status = (out.gate.get("signal_status") or {}).get("g1") if out else None
        rows.append(
            Row(
                "MO9",
                "registration_g1",
                "ms2",
                segmentation,
                status,
                "==",
                "pass",
                None if status is None else str(status) == "pass",
                False,
                "resolve summary mouse_gate signal_status",
                note=_missing(outputs, "ms2"),
                beside={
                    "g1_density_ratio": out.value("MO7", "g1_density_ratio")
                    if out
                    else None
                },
            )
        )
    for dataset, base in (("ms2", "original_seg"), ("ms1", "proseg_hybrid")):
        reference = soft_class_vector(_get(outputs, dataset, base))
        for segmentation in SEGMENTATIONS:
            if segmentation == base:
                continue
            other = soft_class_vector(_get(outputs, dataset, segmentation))
            value = js_distance(other, reference) if other and reference else None
            rows.append(
                Row(
                    "MO9",
                    f"soft_class_jsd_vs_{base}",
                    dataset,
                    segmentation,
                    value,
                    "<=",
                    MO9_JSD_MAX,
                    at_most(value, MO9_JSD_MAX),
                    False,
                    "acceptance_metrics.json MO4 soft_share (34 classes)",
                    note=_missing(outputs, dataset)
                    or (
                        "information (M6 exit practice)" if dataset == "ms1" else "[L]"
                    ),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# MO10: AP-axis harm on MERFISH-638850 (M6b-R's mo10.json)


def selected_candidates(text: str) -> list[str]:
    """Every ``Selected candidate (M6b): `<label>``` line, in document order."""
    marker = "Selected candidate (M6b): `"
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker) and stripped.endswith("`"):
            found.append(stripped[len(marker) : -1])
    return found


def mo10_rows(
    mo10: Mapping[str, Any] | None,
    *,
    reading: str,
    recorded: Sequence[str],
    rule_params: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Row], list[str]]:
    """MO10 per gene set and part, from M6b-R's ``mo10.json`` (§24).

    Args:
        mo10: The parsed ``mo10.json``.
        reading: ``allen_relative`` (registered) or ``fixed`` (0.1%).
        recorded: The pre-registration's selection lines, in order; the last
            one is the selected rule.
        rule_params: Dataset/segmentation to the region rule parameters its
            MAP region step recorded (``mouse_regions.params``).

    Returns:
        The rows and the problems (a candidate the pre-registration does not
        name last, a run whose region rule is not the candidate's: either
        leaves the scored rows unscored). A gene set of ``MO10_GENE_SETS``
        missing from ``mo10.json`` gives a not-evaluable row.
    """
    problems: list[str] = []
    rows: list[Row] = []
    if not mo10:
        rows.append(
            Row(
                "MO10",
                "mo10",
                "MERFISH",
                "MERFISH-638850",
                None,
                "==",
                True,
                None,
                True,
                "mo10.json",
                note="no mo10.json (score --mo10)",
            )
        )
        return rows, problems
    candidate = str(mo10.get("candidate") or "")
    selected = recorded[-1] if recorded else None
    unscored = ""
    if candidate != selected:
        unscored = "selection"
        problems.append(
            f"MO10: mo10.json scores {candidate!r}, but the pre-registration's last "
            f"selection line names {selected!r} (§24 order of record)"
        )
    mismatches = rule_mismatches(candidate, rule_params)
    if mismatches:
        # The runs gate M scores must use the rule MO10 scores.
        unscored = unscored or UNSCORED_REGION_RULE
        problems += mismatches
    mo10_reading = str(mo10.get("t2_reading") or "registered")
    for gene_set in sorted(set(MO10_GENE_SETS) - set(mo10.get("gene_sets") or {})):
        rows.append(
            Row(
                "MO10",
                "gene_set",
                "MERFISH",
                gene_set,
                None,
                "==",
                True,
                None,
                True,
                "mo10.json",
                note="this gene set is missing from mo10.json (§24: all three)",
                scope={"gene_set": gene_set},
                unscored=unscored,
            )
        )
    allen_reading = reading == "allen_relative"
    for gene_set, entry in sorted((mo10.get("gene_sets") or {}).items()):

        def add(
            part: str,
            value: Any,
            comparator: str,
            threshold: Any,
            passes: bool | None,
            scored: bool = True,
            *,
            note: str = "",
            scope: Mapping[str, str] | None = None,
            beside: Mapping[str, Any] | None = None,
            g: str = gene_set,
        ) -> None:
            rows.append(
                Row(
                    "MO10",
                    part,
                    "MERFISH",
                    g,
                    value,
                    comparator,
                    threshold,
                    passes,
                    scored,
                    "mo10.json",
                    note=note,
                    scope={"gene_set": g, **(scope or {})},
                    beside=dict(beside or {}),
                    unscored=unscored if scored else "",
                )
            )

        add(
            "bin_mean_harm",
            entry.get("max_bin_mean_harm"),
            "<=",
            MO10_LIMITS["bin_mean_harm"],
            _flag(entry.get("bin_mean_harm_ok")),
            note="the largest per-bin mean harm",
        )
        add(
            "section_max_harm",
            entry.get("max_section_harm"),
            "<=",
            MO10_LIMITS["section_max_harm"],
            _flag(entry.get("section_max_harm_ok")),
        )
        registered = _flag(
            entry.get(
                "t2_bin_mean_ok_registered",
                entry.get("t2_bin_mean_ok") if mo10_reading == "registered" else None,
            )
        )
        allen = _flag(entry.get("t2_bin_mean_ok_allen"))
        verdict_t2 = allen if allen_reading else registered
        bins = [
            item for item in entry.get("t2_bins") or [] if isinstance(item, Mapping)
        ]
        # No Allen-label T2 in any bin: mo10.json was made without the
        # reading (not evaluable). Otherwise a missing value fails (§24).
        no_allen = allen_reading and all(
            _float(item.get("t2_allen_mean")) is None for item in bins
        )
        bin_verdicts = []
        for item in bins:
            after = _float(item.get("t2_after_mean"))
            allen_mean = _float(item.get("t2_allen_mean"))
            fixed_ok = at_most(after, MO10_LIMITS["t2_bin_mean"])
            allen_ok = (
                None
                if after is None or allen_mean is None
                else after <= allen_mean + TOLERANCE
            )
            ok = allen_ok if allen_reading else fixed_ok
            # §24: "A missing value on either side fails."
            bin_verdicts.append(bool(ok))
            add(
                "t2_bin_mean",
                after,
                "<= T2_allen" if allen_reading else "<=",
                allen_mean if allen_reading else MO10_LIMITS["t2_bin_mean"],
                None if no_allen else bool(ok),
                note="mo10.json has no Allen-label T2" if no_allen else "",
                scope={"ap_bin": str(item.get("ap_bin"))},
                beside={
                    "t2_allen_mean": allen_mean,
                    "ok_allen_relative": allen_ok,
                    "ok_registered_0.1pct": fixed_ok,
                },
            )
        if bins and verdict_t2 is not None and all(bin_verdicts) != verdict_t2:
            problems.append(
                f"MO10 {gene_set}: the per-bin T2 verdicts ({all(bin_verdicts)}) "
                f"disagree with mo10.json's {reading} verdict ({verdict_t2})"
            )
        add(
            "t2_clause",
            entry.get("max_t2_bin_mean"),
            "<= T2_allen" if allen_reading else "<=",
            "the Allen labels' bin-mean T2"
            if allen_reading
            else MO10_LIMITS["t2_bin_mean"],
            verdict_t2,
            not bins,
            note=(
                "scored per AP bin (t2_bin_mean rows)"
                if bins
                else ""
                if verdict_t2 is not None
                else f"mo10.json carries no {reading} T2 verdict (its t2_reading: "
                f"{mo10_reading})"
            ),
            beside={
                "t2_reading_scored": reading,
                "ok_allen_relative": allen,
                "ok_registered_0.1pct": registered,
                "mo10_t2_reading": mo10_reading,
            },
        )
        missed = list(entry.get("missed_with_harm_above") or [])
        if not missed:
            add(
                "missed_region",
                None,
                "listed",
                MO10_LIMITS["missed_region"],
                _flag(entry.get("missed_only_where_harm_small")),
                note="no missed region at a section with harm above 0.1%",
            )
        for item in missed:
            regions = [
                region for region in str(item.get("missed") or "").split(";") if region
            ]
            for region in regions or [""]:
                add(
                    "missed_region",
                    _float(item.get("harm")),
                    "<=",
                    MO10_LIMITS["missed_region"],
                    at_most(item.get("harm"), MO10_LIMITS["missed_region"]),
                    note="a missed region where the section's harm is above 0.1%",
                    scope={"section": str(item.get("section")), "region": region},
                )
        add(
            "oracle_harm",
            entry.get("max_oracle_harm"),
            "reported",
            None,
            None,
            False,
            note="diagnostic",
        )
    for item in mo10.get("mo2") or []:
        rows.append(
            Row(
                "MO10",
                "m6b_exit_mo2",
                str(item.get("section")),
                "M6 MAP output",
                item.get("t2_after"),
                "reported",
                None,
                None,
                False,
                "mo10.json (MO2 re-checked with the selected rule, M6b exit)",
                beside={k: v for k, v in item.items() if k != "section"},
            )
        )
    for dataset, ok in sorted((mo10.get("mo3") or {}).items()):
        rows.append(
            Row(
                "MO10",
                "m6b_exit_mo3",
                str(dataset),
                "M6 MAP output",
                ok,
                "reported",
                None,
                None,
                False,
                "mo10.json (MO3 re-checked with the selected rule, M6b exit)",
            )
        )
    return rows, problems


def rule_mismatches(
    candidate: str, rule_params: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Return the runs whose region rule is not the selected candidate's.

    The candidate's ``MouseRegionConfig`` (``mouse_ap_axis.Candidate.config``)
    is compared key by key with the run's region record ``params`` (the
    production MAP records the whole config); a key the run does not record
    cannot be checked and counts as a difference. An (f)-only candidate
    keeps ``rule_variant`` v1 and differs only in its tile settings.

    Args:
        candidate: The selected candidate's label.
        rule_params: Run to its recorded ``mouse_regions.params``.

    Returns:
        One problem per run that differs (or for an unknown candidate).
    """
    try:
        expected = candidate_by_label(candidate).config().model_dump(mode="json")
    except (KeyError, ValueError) as error:
        return [f"MO10: the selected candidate {candidate!r}: {error}"]
    problems = []
    for where, params in sorted(rule_params.items()):
        differ = sorted(
            key if key in params else f"{key} (not recorded)"
            for key, value in expected.items()
            if key not in params or params[key] != value
        )
        if differ:
            problems.append(
                f"MO10: {where} ran another region rule than the selected "
                f"{candidate!r} (differs in {', '.join(differ)})"
            )
    return problems


def _flag(value: Any) -> bool | None:
    return bool(value) if isinstance(value, bool | np.bool_) else None


# ---------------------------------------------------------------------------
# MO11 and H18 (reported)


def ap_estimate(out: SegOutputs | None) -> float | None:
    """The section's AP estimate (gate record, else the report's MO11 field)."""
    if out is None:
        return None
    estimate = (out.gate.get("ap_estimate") or {}).get("ap_mm")
    if estimate is None:
        estimate = out.value("MO11", "ap_estimate_mm")
    return _float(estimate)


def mo11_rows(outputs: Mapping[str, Mapping[str, SegOutputs]]) -> list[Row]:
    """MO11 (reported): whether a scored section is anterior (AP < 6.0 mm).

    MO11 needs the first real anterior section's manual anatomy; it does not
    block the flip and only lifts the banner, so it is reported here.
    """
    rows = []
    for dataset, segmentation in _all_pairs():
        out = _get(outputs, dataset, segmentation)
        estimate = ap_estimate(out)
        anterior = None if estimate is None else estimate < MO11_ANTERIOR_AP_MM
        rows.append(
            Row(
                "MO11",
                "ap_estimate_mm",
                dataset,
                segmentation,
                estimate,
                "reported",
                MO11_ANTERIOR_AP_MM,
                None,
                False,
                "resolve summary mouse_gate.ap_estimate",
                note=(
                    _missing(outputs, dataset)
                    or (
                        "AP estimate pending"
                        if anterior is None
                        else "anterior (AP < 6.0 mm): MO11 applies; it needs this "
                        "section's manual anatomy (GM-REVIEW)"
                        if anterior
                        else "not anterior: MO11 is not tested here; the banner "
                        "rule stands"
                    )
                ),
                beside={"anterior": anterior},
            )
        )
    return rows


H18_NAMES: Final = (
    "prep_bins_emitted",
    "reweighted_bins_emitted",
    "thresholds_raised_by_local_rule",
)


def h18_rows(
    outputs: Mapping[str, Mapping[str, SegOutputs]], bundles: Sequence[Path]
) -> list[Row]:
    """H18 mouse rows (CHECK M2): the values, recorded with no exception."""
    rows = []
    for dataset, segmentation in _scored_pairs():
        out = _get(outputs, dataset, segmentation)
        for record in out.records if out else []:
            if (
                record.get("criterion") not in ("report", "H18")
                or record.get("name") not in H18_NAMES
            ):
                continue
            rows.append(
                Row(
                    "H18",
                    str(record["name"]),
                    dataset,
                    segmentation,
                    record.get("value"),
                    "reported",
                    None,
                    None,
                    False,
                    "acceptance_metrics.json (PREP and reweighted emission)",
                    scope={
                        "level": str(record.get("level")),
                        "kind": str(record.get("kind")),
                    },
                    beside={"n": record.get("n")},
                )
            )
    for path in sorted(set(bundles)):
        record = _bundle_record(path)
        if record.get("reference_id") != MO8_PANEL_REFERENCE:
            continue
        would_raise = record.get("would_raise")
        count = (
            len(would_raise) if isinstance(would_raise, list | dict) else would_raise
        )
        rows.append(
            Row(
                "H18",
                "validated_thresholds_would_raise",
                _bundle_label(record),
                "PREP",
                count,
                "reported",
                None,
                None,
                False,
                "resolvability_summary.json",
                scope={"bundle": str(record.get("build_hash") or path.name)},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# the flip rule and the summary


def flip_rule(
    rows: Sequence[Row], provenance_ok: bool, problems: Sequence[str] = ()
) -> dict[str, Any]:
    """The mouse flip rule: MO1-MO8 and MO10 with the approved exceptions.

    A criterion passes only when every scored row is ``PASS`` or ``FAIL
    (excepted)``. Every problem goes back to the user, so the flip is
    ``INCOMPLETE`` rather than ``PASS`` while one is open.
    """
    per_criterion: dict[str, str] = {}
    for criterion in FLIP_CRITERIA + REPORTED_CRITERIA:
        verdicts = [
            row.verdict for row in rows if row.criterion == criterion and row.scored
        ]
        if criterion in REPORTED_CRITERIA or not verdicts:
            per_criterion[criterion] = (
                "reported" if criterion in REPORTED_CRITERIA else NOT_AVAILABLE
            )
        elif FAIL in verdicts:
            per_criterion[criterion] = FAIL
        elif any(v not in (PASS, EXCEPTED) for v in verdicts):
            per_criterion[criterion] = "INCOMPLETE"
        elif EXCEPTED in verdicts:
            per_criterion[criterion] = EXCEPTED
        else:
            per_criterion[criterion] = PASS
    flip = [per_criterion[criterion] for criterion in FLIP_CRITERIA]
    if not provenance_ok:
        verdict = "NOT SCORED (provenance)"
    elif FAIL in flip:
        verdict = FAIL
    elif any(value in ("INCOMPLETE", NOT_AVAILABLE) for value in flip) or problems:
        verdict = "INCOMPLETE"
    else:
        verdict = PASS
    return {
        "rule": "MO1-MO8 and MO10, with the user's written exceptions (§9, §25.2); "
        "MO11 keeps its banner rule; MO9 [L] is reported",
        "verdict": verdict,
        "criteria": per_criterion,
        "problems": len(problems),
        "failing": [
            _row_label(row) for row in rows if row.scored and row.verdict == FAIL
        ],
        "excepted": [
            f"{_row_label(row)} ({row.exception})"
            for row in rows
            if row.verdict == EXCEPTED
        ],
        "not_scored": [
            f"{_row_label(row)}: {row.verdict}"
            for row in rows
            if row.scored and row.verdict not in (PASS, FAIL, EXCEPTED)
        ],
    }


def counts(rows: Sequence[Row]) -> dict[str, int]:
    """Rows per verdict."""
    out: dict[str, int] = {}
    for row in rows:
        out[row.verdict] = out.get(row.verdict, 0) + 1
    return dict(sorted(out.items()))


def file_sha256(path: Path) -> str:
    """A file's sha256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity() -> dict[str, Any]:
    """The code commit (``MERXEN_CODE_COMMIT``, git) and the scripts' sha256."""
    script = Path(__file__).resolve()
    commit: str | None = os.environ.get("MERXEN_CODE_COMMIT", "").strip() or None
    source = "env MERXEN_CODE_COMMIT" if commit else None
    dirty: bool | None = None
    if commit is None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(script.parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            source = "git"
            dirty = bool(
                subprocess.run(
                    ["git", "-C", str(script.parent), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            commit = None
    modules = {
        "mouse_acceptance.py": script,
        "mouse_corr_shadow.py": script.parent / "mouse_corr_shadow.py",
        "mouse_ap_axis.py": script.parent / "mouse_ap_axis.py",
        "run_acceptance.py": script.parent / "run_acceptance.py",
        "report_scoring.py": Path(sc.__file__).resolve(),
    }
    return {
        "git_commit": commit,
        "git_commit_source": source,
        "worktree_dirty": dirty,
        "sha256": {name: file_sha256(path) for name, path in modules.items()},
    }


_CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#57606a;--line:#d0d7de;--head:#f6f8fa;
--back:#fff4e5;--pass:#1a7f37;--fail:#cf222e;--stamp:#9a6700}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;
--line:#30363d;--head:#161b22;--back:#3d2a12;--pass:#3fb950;--fail:#f85149;--stamp:#d29922}}
body{font-family:system-ui,sans-serif;margin:24px;color:var(--fg);background:var(--bg);
max-width:1400px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{border:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}
th{background:var(--head)}.back{background:var(--back)}.small{color:var(--muted);font-size:12px}
.stamp{color:var(--stamp);font-weight:600}
"""


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.4g}"
    if isinstance(value, Mapping | list):
        return html.escape(json.dumps(_safe(value), default=str))
    return html.escape("" if value is None else str(value))


def _table(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for record in records:
        css = ' class="back"' if record.get("back_to_user") else ""
        cells = "".join(f"<td>{_cell(record.get(column))}</td>" for column in columns)
        body.append(f"<tr{css}>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


ROW_COLUMNS: Final = (
    "criterion",
    "part",
    "dataset",
    "segmentation",
    "scope",
    "value",
    "comparator",
    "threshold",
    "verdict",
    "exception",
    "source",
    "note",
    "beside",
)


def write_html(summary: Mapping[str, Any], path: Path) -> None:
    """Write the summary page (static, no external resources)."""
    rows = summary["rows"]
    back = [row for row in rows if row["back_to_user"]]
    stamp = summary["mode"] + (f"; {summary['label']}" if summary.get("label") else "")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Gate M Acceptance</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>M9 mouse acceptance (gate M)</h1>",
        f"<p class='stamp'>{html.escape(stamp)}</p>",
        f"<p class='small'>Date {html.escape(str(summary['date']))}; code "
        f"{html.escape(str(summary['code'].get('git_commit')))}. The reports record "
        "metrics; this page scores them (pre-registration §9, §24, §25). Rows marked "
        "in orange go back to the user.</p>",
        f"<h2>Flip rule: {html.escape(summary['flip_rule']['verdict'])}</h2>",
        _table(
            [
                {"criterion": k, "verdict": v}
                for k, v in summary["flip_rule"]["criteria"].items()
            ],
            ("criterion", "verdict"),
        ),
        "<h2>Readings</h2>",
        _table(
            [{"reading": k, **v} for k, v in summary["readings"].items()],
            (
                "reading",
                "value",
                "registered",
                "is_registered",
                "source",
                "reported_beside",
            ),
        ),
        "<h2>Provenance: "
        + ("passed" if summary["provenance"]["ok"] else "FAILED")
        + "</h2>",
        _table(summary["provenance"]["bundles"], ("path", "real_path", "ok", "reason")),
        "<h2>Exceptions</h2>",
        _table(
            summary["exceptions"],
            (
                "id",
                "criterion",
                "scope",
                "approved",
                "quote",
                "applied_to",
                "named_outside_bounds",
            ),
        ),
        f"<h2>Back to the user ({len(back)})</h2>",
        _table(back, ROW_COLUMNS),
        "<h2>Problems</h2><ul>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in summary["problems"])
        + "</ul>",
        "<h2>Every row</h2>",
        _table(rows, ROW_COLUMNS),
        "<h2>Inputs</h2>",
        _table(
            [{"input": k, "sha256": v} for k, v in summary["inputs"].items()],
            ("input", "sha256"),
        ),
        "</body></html>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# score


def _pairs(values: Sequence[str] | None, option: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        key, sep, rest = value.partition("=")
        if not sep or not key or not rest:
            raise ValueError(f"{option}: expected KEY=PATH, got {value!r}")
        out[key] = rest
    return out


def dataset_key(text: str) -> str:
    """Return the dataset key (``ms1``, ``ms2``) of a real name or alias (R2)."""
    return alias_key(text, (*DATASETS, NEGATIVE_CONTROL))


def _dataset_pairs(values: Sequence[str] | None, option: str) -> dict[str, str]:
    """``_pairs`` whose keys name a dataset, keyed by ``dataset_key``.

    A ``DATASET/SEGMENTATION`` key keeps its segmentation.
    """
    out: dict[str, str] = {}
    for key, value in _pairs(values, option).items():
        dataset, sep, segmentation = key.partition("/")
        out[dataset_key(dataset) + sep + segmentation] = value
    return out


def store_root() -> Path:
    """The gate M store (§25.3): ``STORE_ROOT``, else ``$MERXEN_REFERENCE_STORE``."""
    if STORE_ROOT is not None:
        return STORE_ROOT
    return local_config.require_path(STORE_VARIABLE, "the gate M store (§25.3)")


def pseudo_label_paths(
    evidence_root: Path | None, given: Mapping[str, str], datasets: Iterable[str]
) -> dict[str, Path]:
    """MO1's pseudo-label CSV per dataset: ``--pseudo-labels``, else the evidence's.

    The evidence archive names its mouse directories by the sections' display
    names (``privacy.to_real(form="name")``, R3).

    Args:
        evidence_root: ``--evidence-root`` (``None``: ``$MERXEN_EVIDENCE_ROOT``,
            read only when a scored dataset has no ``--pseudo-labels``).
        given: ``--pseudo-labels`` by dataset key.
        datasets: The datasets given (``--dataset``).

    Returns:
        Dataset to the CSV.
    """
    paths = {dataset: Path(value) for dataset, value in given.items()}
    missing = [d for d in PSEUDO_LABELS if d in set(datasets) and d not in paths]
    if missing:
        root = evidence_root or local_config.require_path(
            EVIDENCE_VARIABLE, "MO1's default pseudo-labels (or --pseudo-labels)"
        )
        for dataset in missing:
            paths[dataset] = root / privacy.to_real(PSEUDO_LABELS[dataset], form="name")
    return paths


def aliased_mo10(payload: Any) -> Any:
    """Return ``mo10.json`` with its section and gene-set keys aliased (R2).

    M6b-R wrote the sections' real names into its gene-set keys and its MO2
    and MO3 re-checks.
    """
    if not isinstance(payload, Mapping):
        return payload
    out = dict(payload)
    if isinstance(out.get("gene_sets"), Mapping):
        out["gene_sets"] = {
            alias_key(str(key), MO10_GENE_SETS): value
            for key, value in out["gene_sets"].items()
        }
    if isinstance(out.get("mo2"), list):
        out["mo2"] = [
            {**item, "section": dataset_key(str(item["section"]))}
            if isinstance(item, Mapping) and item.get("section") is not None
            else item
            for item in out["mo2"]
        ]
    if isinstance(out.get("mo3"), Mapping):
        out["mo3"] = {dataset_key(str(key)): ok for key, ok in out["mo3"].items()}
    return out


def full_remap_dirs(values: Sequence[str] | None) -> dict[tuple[str, str], Path]:
    """Parse ``--mo2-full``: one full re-map per scored (dataset, segmentation).

    Args:
        values: ``DATASET/SEGMENTATION=DIR`` items; a bare ``DATASET=DIR``
            stands for a dataset that ``SCORED_UNNAMED`` scores on one
            segmentation only.

    Returns:
        (dataset, segmentation) to the full-remap output directory.

    Raises:
        ValueError: For an unknown dataset, a segmentation MO2 does not
            score, an ambiguous bare dataset or a repeated key.
    """
    out: dict[tuple[str, str], Path] = {}
    for value in values or []:
        key, sep, rest = value.partition("=")
        if not sep or not key or not rest:
            raise ValueError(
                f"--mo2-full: expected DATASET/SEGMENTATION=DIR, got {value!r}"
            )
        dataset, _, segmentation = key.partition("/")
        dataset = dataset_key(dataset)
        if dataset not in SCORED_UNNAMED:
            raise ValueError(f"--mo2-full: unknown dataset {dataset!r}; use {DATASETS}")
        scored = SCORED_UNNAMED[dataset]
        if not segmentation:
            if len(scored) != 1:
                raise ValueError(
                    f"--mo2-full {key}: MO2 is scored on {', '.join(scored)} for "
                    f"{dataset} (§25.2.1); name the segmentation "
                    f"({dataset}/SEGMENTATION=DIR)"
                )
            segmentation = scored[0]
        if segmentation not in scored:
            raise ValueError(
                f"--mo2-full {key}: {segmentation} is not scored for MO2 on "
                f"{dataset} ({', '.join(scored)}; §25.2.1)"
            )
        if (dataset, segmentation) in out:
            raise ValueError(f"--mo2-full: {dataset}/{segmentation} given twice")
        out[(dataset, segmentation)] = Path(rest)
    return out


def _check_out(out: Path, protected: Sequence[Path]) -> None:
    target = Path(os.path.abspath(out))
    for root in protected:
        base = Path(os.path.abspath(root))
        if target == base or target.is_relative_to(base):
            raise ValueError(f"--out {out} lies inside an input tree ({root})")


def score(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Run ``score``: return the summary and the exit status."""
    readings = resolve_readings(
        {name: getattr(args, name) for name in READINGS}, exploratory=args.exploratory
    )
    mode = EXPLORATORY if args.exploratory else REGISTERED
    roots = {
        key: Path(value)
        for key, value in _dataset_pairs(args.dataset, "--dataset").items()
    }
    unknown = set(roots) - set(DATASETS)
    if unknown:
        raise ValueError(
            f"--dataset: unknown dataset(s) {sorted(unknown)}; use {DATASETS}"
        )
    store = store_root()
    _check_out(args.out, [*roots.values(), store])
    inputs: dict[str, str] = {}

    def track(path: Path) -> Path:
        if path.is_file():
            inputs[str(path)] = file_sha256(path)
        return path

    problems: list[str] = []
    outputs = {dataset: load_outputs(dataset, root) for dataset, root in roots.items()}
    for segs in outputs.values():
        for out in segs.values():
            for path in out.files:
                if path.suffix == ".json":
                    track(path)
            if out.report is None:
                problems.append(
                    f"{out.dataset} {out.segmentation}: no acceptance_metrics.json"
                )
    trace = load_trace([track(path) for path in args.trace or []])
    full_dirs = full_remap_dirs(args.mo2_full)
    samplers: dict[tuple[str, str], dict[str, float]] = {}
    for key, value in _dataset_pairs(args.sampler, "--sampler").items():
        dataset, _, segmentation = key.partition("/")
        samplers[(dataset, segmentation)] = read_sampler(track(Path(value)))
    negative_path: Path | None = None
    negative = None
    negative_manifest = None
    if args.mo7_negative is not None:
        negative_path, manifest_path = negative_control_files(Path(args.mo7_negative))
        if negative_path is not None:
            negative = _read_json(track(negative_path))
        else:
            problems.append(f"MO7: no single resolve summary under {args.mo7_negative}")
        if manifest_path is not None:
            negative_manifest = _read_json(track(manifest_path))
        else:
            problems.append(
                f"MO7: no single map_manifest.json under {args.mo7_negative}: the "
                "negative control's MAP bundles are not checked"
            )
    mo10 = None
    if args.mo10 is not None:
        mo10 = aliased_mo10(_read_json(track(Path(args.mo10))))
    exceptions: list[ApprovedException] = []
    if args.exceptions is not None:
        exceptions = parse_exceptions(_read_json(track(Path(args.exceptions))))
    prereg = Path(args.preregistration)
    recorded = (
        selected_candidates(prereg.read_text(encoding="utf-8"))
        if prereg.is_file()
        else []
    )
    if prereg.is_file():
        track(prereg)
    refs = sorted(
        path
        for root in args.prep_refs or []
        for path in Path(root).rglob("bundle_ref.json")
    )

    # Provenance first: nothing below reads a bundle before it passes.
    found: list[tuple[str, str]] = []
    for segs in outputs.values():
        for out in segs.values():
            for name, payload in (
                ("report", out.report),
                ("map_manifest", out.manifest),
                ("resolve_summary", out.resolve_summary),
            ):
                if payload is not None:
                    found += bundle_paths(
                        payload, f"{out.dataset}/{out.segmentation}/{name}"
                    )
    for ref in refs:
        found += bundle_paths(_read_json(track(ref)), str(ref))
    if negative is not None:
        found += bundle_paths(negative, f"{NEGATIVE_CONTROL}/resolve_summary")
    if negative_manifest is not None:
        found += bundle_paths(negative_manifest, f"{NEGATIVE_CONTROL}/map_manifest")
    # A full re-map that records another segmentation is not used (§25.2.1:
    # one full re-map per scored segmentation).
    full_mismatch: dict[tuple[str, str], str] = {}
    for (dataset, segmentation), full_dir in full_dirs.items():
        record_path = full_dir / FULL_REMAP_RECORD
        if record_path.is_file():
            full_record = _read_json(track(record_path))
            found += bundle_paths(full_record, f"{dataset}/{segmentation}/full_remap")
            recorded_seg = full_record.get("segmentation")
            if recorded_seg not in (None, segmentation):
                full_mismatch[(dataset, segmentation)] = (
                    f"the full re-map in {full_dir} records segmentation "
                    f"{recorded_seg}, not {segmentation}"
                )
                problems.append(
                    f"MO2 {dataset} {segmentation}: "
                    + full_mismatch[(dataset, segmentation)]
                )
    provenance = check_provenance(found, store)

    rows: list[Row] = []
    usage: list[dict[str, Any]] = []
    if provenance.ok:
        bundles = sorted(
            {
                Path(record["real_path"])
                for record in provenance.bundles
                if record["ok"] and record["real_path"]
            }
        )
        pseudo = pseudo_label_paths(
            args.evidence_root,
            _dataset_pairs(args.pseudo_labels, "--pseudo-labels"),
            roots,
        )
        rows += mo1_rows(outputs, pseudo)
        mo2: dict[tuple[str, str], Any] = {}
        mo2_problems: dict[tuple[str, str], str] = {}
        for pair in _unnamed_pairs():
            out = _get(outputs, *pair)
            if out is None or out.manifest is None:
                continue
            mismatch = full_mismatch.get(pair)
            try:
                mo2[pair] = mo2_measure(
                    out.map_dir, None if mismatch else full_dirs.get(pair)
                )
            except (OSError, ValueError, KeyError) as error:
                mo2_problems[pair] = f"MO2 not measured: {error}"
                problems.append(f"MO2 {pair[0]} {pair[1]}: {error}")
                continue
            if mismatch:
                mo2[pair] = {
                    **mo2[pair],
                    "full": None,
                    "full_record": None,
                    "full_problem": mismatch,
                }
        rows += mo2_rows(outputs, mo2, readings["mo2_remap"]["value"], mo2_problems)
        rows += mo3_rows(outputs)
        rows += mo4_rows(outputs, readings["mo4_window"]["value"])
        rows += mo5_rows(outputs)
        rows += mo6_rows(outputs)
        rows += mo7_rows(
            outputs,
            negative,
            negative_path,
            negative_bundles_checked=negative_manifest is not None,
        )
        section_rows, section_problems = mo8_section_rows(
            outputs,
            trace,
            samplers,
            memory_reading=readings["mo8_map_memory"]["value"],
            processes=int(readings["mo8_map_processes"]["value"]),
        )
        rows += section_rows
        problems += section_problems
        rows += mo8_prep_rows(
            bundles, trace, memory_reading=readings["mo8_prep_memory"]["value"]
        )
        rows += mo9_rows(outputs)
        params = {
            f"{out.dataset}/{out.segmentation}": dict(
                out.region_record.get("params") or {}
            )
            for segs in outputs.values()
            for out in segs.values()
            if out.region_record
        }
        mo10_list, mo10_problems = mo10_rows(
            mo10,
            reading=readings["mo10_t2"]["value"],
            recorded=recorded,
            rule_params=params,
        )
        rows += mo10_list
        problems += mo10_problems
        rows += mo11_rows(outputs)
        rows += h18_rows(outputs, bundles)
        rows = [finalize(row) for row in rows]
        usage = apply_exceptions(rows, exceptions)
    else:
        problems += [f"provenance: {reason}" for reason in provenance.reasons]
        rows = [
            finalize(
                Row(
                    criterion,
                    "all",
                    "all",
                    "all",
                    None,
                    "==",
                    None,
                    None,
                    True,
                    "provenance check (§25.3)",
                    note="a bundle path lies outside the store: nothing was measured",
                    unscored=UNSCORED_PROVENANCE,
                )
            )
            for criterion in FLIP_CRITERIA
        ]
        usage = [
            {**exception.to_json(), "applied_to": [], "named_outside_bounds": []}
            for exception in exceptions
        ]
    records = [row.to_json() for row in rows]
    summary: dict[str, Any] = {
        "schema": "merxen-m9-mouse-acceptance/1",
        "mode": mode,
        "label": args.label,
        "date": args.date,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_identity(),
        "protocol": {
            "preregistration": "docs/acceptance/annotation-v1-preregistration.md "
            "§9 (MO1-MO11), §24 (MO10, amendment U1), §25 (gate M)",
            "scored_segmentations": dict(SCORED_SEGMENTATION),
            "scored_segmentations_unnamed": {
                dataset: list(segmentations)
                for dataset, segmentations in SCORED_UNNAMED.items()
            },
            "selection_lines": recorded,
            "store_root": str(store),
        },
        "readings": readings,
        "datasets": {dataset: str(root) for dataset, root in roots.items()},
        "provenance": dataclasses.asdict(provenance),
        "flip_rule": flip_rule(rows, provenance.ok, problems),
        "exceptions": usage,
        "verdict_counts": counts(rows),
        "n_back_to_user": sum(1 for record in records if record["back_to_user"])
        + len(problems),
        "problems": problems,
        "rows": records,
        "inputs": inputs,
    }
    # Labels as privacy.display shows them (R5); paths verbatim. An approval
    # is recorded as its file holds it: its quote and record are the user's
    # and the record's own words, which may name a section both ways.
    summary = display_tree(_safe(summary))
    for record, exception in zip(summary["exceptions"], exceptions, strict=True):
        record.update(_safe(exception.to_json()))
    records = summary["rows"]
    out_dir = args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_dir / "tables" / "rows.csv", index=False)
    if len(trace):
        trace.to_csv(out_dir / "tables" / "trace_tasks.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(_safe(summary), indent=1, default=str) + "\n", encoding="utf-8"
    )
    write_html(summary, out_dir / "summary.html")
    return summary, 0 if provenance.ok else 3


# ---------------------------------------------------------------------------
# full-remap


# The refused disk (SSD1): nothing under this prefix is read.
REFUSED_PREFIX_VARIABLE: Final = "MERXEN_REFUSED_PATH_PREFIX"


def refused_prefix() -> str:
    """The refused path prefix (``$MERXEN_REFUSED_PATH_PREFIX``, the SSD1 disk)."""
    prefix = local_config.require_path(
        REFUSED_PREFIX_VARIABLE, "full-remap refuses MAP inputs on the refused disk"
    )
    return str(prefix).rstrip("/") or "/"


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _refused_strings(payload: Any, prefix: str) -> list[str]:
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and _under(node, prefix):
            found.append(node)

    walk(payload)
    return found


def full_remap(args: argparse.Namespace) -> dict[str, Any]:
    """Run ``full-remap``: the full pruned re-map of one MAP output (MO2).

    Every table cell is mapped again on the pruned tree: the MAP run's
    bundle, the pruned lookup (``mouse_regions.pruned_lookup``),
    ``--nodes_to_drop`` and the run's own engine parameters, process count
    included (§24 stage B, §25.2). The query is rebuilt from the H5AD the
    MAP run read, with the MAP run's own configuration (``--gene-table``
    replaces a gene table), and must reproduce the run's query fingerprint.

    Returns:
        The ``full_remap.json`` record.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mapmycells_engine import (
        MmcBundle,
        MmcEngineParams,
        run_mmc,
        write_query_h5ad,
    )
    from merxen.annotation.mouse_regions import pruned_lookup
    from merxen.annotation.panel import load_annotation_panel
    from merxen.annotation.pipeline import (
        MapSample,
        build_sample_query,
        load_map_manifest,
        load_samples,
    )
    from merxen.annotation.reference import lookup_sha256, read_lookup

    map_dir = Path(args.map_dir)
    out = Path(args.out)
    manifest = load_map_manifest(map_dir / MAP_MANIFEST)
    ((sample_id, record),) = manifest.samples.items()
    run = record.runs[PRIMARY_RUN]
    regions = record.mouse_regions
    if regions is None or not regions.nodes_to_drop:
        raise ValueError(
            f"{map_dir}: the region step dropped nothing; no re-map to compare"
        )
    check = check_provenance([(str(run.bundle_path), "map_manifest")], store_root())
    if not check.ok:
        raise ValueError(f"provenance: {'; '.join(check.reasons)}")
    payload = _read_json(map_dir / "annotation_config.json")
    tables = _pairs(args.gene_table, "--gene-table")
    panel_settings = payload.setdefault("panel", {})
    for species, path in tables.items():
        panel_settings.setdefault("gene_tables", {})[species] = path
        if species == "mouse":
            panel_settings["gene_id_fallback_csv"] = path
    for species in args.drop_gene_table or []:
        (panel_settings.get("gene_tables") or {}).pop(species, None)
    prefix = refused_prefix()
    refused = _refused_strings(payload, prefix)
    if refused or _under(str(record.h5ad_path), prefix):
        raise ValueError(
            "the MAP configuration reads from SSD1 (the refused disk, "
            f"${REFUSED_PREFIX_VARIABLE}): "
            + ", ".join(sorted(set(refused)))
            + " (replace each with --gene-table SPECIES=PATH or --drop-gene-table)"
        )
    config = AnnotationConfig.model_validate(payload).coupled_to_clustering(
        manifest.min_counts
    )
    panel_path = (
        Path(args.panel)
        if args.panel
        else (
            map_dir.parents[1]
            / "annotation_panel"
            / "annotation_panel_out"
            / "panel_genes.json"
        )
    )
    out.mkdir(parents=True, exist_ok=False)
    (loaded,) = load_samples(
        [MapSample(sample_id, record.platform, Path(record.h5ad_path), record.source)],
        config,
        min_counts=manifest.min_counts,
    )
    query = build_sample_query(loaded, load_annotation_panel(panel_path))
    if query.fingerprint != run.query_fingerprint:
        raise ValueError("the rebuilt query differs from the MAP run's (fingerprint)")
    bundle = MmcBundle.from_dir(run.bundle_path)
    nodes = regions.nodes_to_drop
    validation = pruned_lookup(
        read_lookup(bundle.lookup), bundle.tree(), nodes, query.gene_ids
    )
    scratch = Path(args.scratch or out / "scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    lookup_path = scratch / "lookup.pruned.json"
    lookup_path.write_text(json.dumps(validation.lookup) + "\n", encoding="utf-8")
    query_path = write_query_h5ad(
        query.counts, query.cell_ids, query.gene_ids, scratch / "query.h5ad"
    )
    engine = dict(run.engine_params or {})
    params = MmcEngineParams.model_validate(engine)
    parquet = out / FULL_REMAP_PARQUET.format(sample_id=sample_id)
    result = run_mmc(
        query_path,
        bundle,
        params,
        output_parquet=parquet,
        work_dir=scratch / "work",
        expected_ctm_version=config.ctm_version,
        lookup_path=lookup_path,
        nodes_to_drop=[node.as_pair() for node in nodes],
        log_dir=out / "logs",
        run_metadata={"purpose": "m9_mo2_full_pruned_remap", "sample_id": sample_id},
    )
    shutil.rmtree(scratch, ignore_errors=True)
    full_record = {
        "purpose": "MO2 full pruned re-map (pre-registration §25.2)",
        "map_dir": str(map_dir),
        "pair_id": manifest.pair_id,
        "segmentation": manifest.segmentation,
        "sample_id": sample_id,
        "parquet": parquet.name,
        "parquet_sha256": file_sha256(parquet),
        "build_hash": bundle.build_hash,
        "bundle_path": str(bundle.path),
        "n_processors": params.n_processors,
        "engine_params": params.model_dump(),
        "map_n_processors": manifest.n_processors,
        "n_cells": result.n_cells,
        "wall_s": result.wall_s,
        "peak_rss_gb": result.peak_rss_gb,
        "query_fingerprint": query.fingerprint,
        "lookup_sha256": lookup_sha256(validation.lookup),
        "nodes_to_drop": [list(node.as_pair()) for node in nodes],
        "gene_tables_replaced": tables,
        "gene_tables_dropped": list(args.drop_gene_table or []),
        "code": code_identity(),
    }
    (out / FULL_REMAP_RECORD).write_text(
        json.dumps(_safe(full_record), indent=1, default=str) + "\n", encoding="utf-8"
    )
    return full_record


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score", help="score gate M")
    scoring.add_argument(
        "--dataset",
        action="append",
        help="ms1=ROOT or ms2=ROOT (a section's alias or real name): the results "
        "root <outdir>/<pair> of a dataset",
    )
    scoring.add_argument(
        "--trace", type=Path, action="append", help="Nextflow trace.tsv"
    )
    scoring.add_argument(
        "--prep-refs",
        type=Path,
        action="append",
        help="PREP publish dir (bundle_ref.json)",
    )
    scoring.add_argument(
        "--sampler",
        action="append",
        help="DATASET/SEG=sampler.csv: a process-tree sampler of that MAP task",
    )
    scoring.add_argument(
        "--mo2-full",
        action="append",
        help="DATASET/SEGMENTATION=DIR: the full-remap output of that MAP output, "
        "one per segmentation MO2 scores (§25.2.1); DATASET=DIR for a dataset "
        "scored on one segmentation",
    )
    scoring.add_argument(
        "--mo7-negative", help="the negative control's resolve summary (file or dir)"
    )
    scoring.add_argument("--mo10", help="M6b-R's mo10.json")
    scoring.add_argument("--exceptions", help="JSON of user-approved exceptions")
    scoring.add_argument(
        "--pseudo-labels",
        action="append",
        help="DATASET=CSV (MO1; default: the evidence root's)",
    )
    scoring.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="the evidence archive: MO1's default pseudo-labels "
        f"(default: ${EVIDENCE_VARIABLE})",
    )
    scoring.add_argument("--preregistration", type=Path, default=PREREGISTRATION)
    for name, reading in READINGS.items():
        scoring.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            default=None,
            help=f"registered: {reading.registered} ({reading.source})",
        )
    scoring.add_argument(
        "--exploratory",
        action="store_true",
        help=f"allow a non-registered reading; stamps the summary {EXPLORATORY!r}",
    )
    scoring.add_argument(
        "--label", default="", help="free-text label stamped on the summary"
    )
    scoring.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    scoring.add_argument("--out", type=Path, required=True)
    remap = commands.add_parser(
        "full-remap", help="the full pruned re-map of a MAP output"
    )
    remap.add_argument("--map-dir", type=Path, required=True, help="annotation_map_out")
    remap.add_argument(
        "--panel", type=Path, help="panel_genes.json (default: the published one)"
    )
    remap.add_argument(
        "--gene-table",
        action="append",
        help="SPECIES=PATH replacing the MAP config's table",
    )
    remap.add_argument(
        "--drop-gene-table", action="append", help="SPECIES: leave that table out"
    )
    remap.add_argument(
        "--scratch", type=Path, help="scratch directory (removed afterwards)"
    )
    remap.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score gate M (``score``) or run MO2's full re-map (``full-remap``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "full-remap":
        try:
            record = full_remap(args)
        except (ValueError, LocalConfigError) as error:
            parser.error(str(error))
        print(
            json.dumps({k: record[k] for k in ("parquet", "n_cells", "n_processors")})
        )
        return 0
    try:
        summary, status = score(args)
    except (ValueError, LocalConfigError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "flip_rule": summary["flip_rule"]["verdict"],
                "verdict_counts": summary["verdict_counts"],
                "n_back_to_user": summary["n_back_to_user"],
            },
            indent=1,
        )
    )
    for problem in summary["problems"]:
        logger.warning("problem: %s", problem)
    return status


if __name__ == "__main__":
    sys.exit(main())
