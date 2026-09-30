"""Data model of the annotation report: items, metric records, the JSON file.

``acceptance_metrics.json`` (plan §9, §14) holds every metric the report
measures as a flat list of ``MetricRecord`` (criterion id, metric name,
scope, value, CI, n, definition, source), the coverage of each §14
criterion (which metrics the report provides and which the acceptance
scripts provide), a per-dataset digest and the provenance footer (item 12).
The report records metrics, never verdicts: pass / fail against the
pre-registered thresholds is the acceptance programme's job (M8, M9).

The file is deterministic for identical inputs (no timestamps or wall
times; floats rounded to 6 significant digits; sorted keys and records).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from merxen.annotation.report_figures import FigureRecord, write_csv
from merxen.annotation.report_panel import Banner

REPORT_SCHEMA_VERSION: Final = 1
METRICS_FILENAME: Final = "acceptance_metrics.json"
REPORT_FILENAME: Final = "report.html"
RUN_FILENAME: Final = "report_run.json"
FIGURES_DIR: Final = "figures"
TABLES_DIR: Final = "tables"

ItemStatus = Literal["ok", "partial", "not_available", "not_applicable", "failed"]
MetricStatus = Literal["measured", "not_available", "withheld"]

# §14 criteria: where each one's metric comes from. "report" = measured
# here (acceptance_metrics.json); "script:<name>" = the acceptance script
# the plan names; "report+script" = the report provides inputs or part.
HUMAN_CRITERIA_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "H1": (
        "report",
        "soft broad JSD, whole section and shared mask, 95% block-bootstrap CI",
    ),
    "H2": ("report", "flag_implausible share of table cells"),
    "H3": ("report", "WHB-SEA 7-class agreement, table cells >= 20 counts"),
    "H4": (
        "report+script",
        "held-out-gene enrichment: heldout_genes.py; item 8 when given",
    ),
    "H5": (
        "report",
        "confident COP supercluster, confident broad OPC, COP-derived OPC share",
    ),
    "H6": (
        "script:run_acceptance.py",
        "archived E2 HO simulation (not a pipeline output)",
    ),
    "H7": ("report", "confident broad coverage of table cells (and segmented objects)"),
    "H8": ("report", "gate level, warning flag, A, coverages"),
    "H9": (
        "script:marker_pseudo_labels.py",
        "marker pseudo-labels of the dataset reports",
    ),
    "H10": ("script:marker_referee.py", "needs the legacy broad_class"),
    "H12": ("report", "cortical-depth ordering, WM > GM, replication across platforms"),
    "H13": (
        "report+script",
        "table / label id sets, MENDER and depth outputs; zarr sha256 by the gate",
    ),
    "H14": ("report+script", "recorded wall times; peak RSS from the Nextflow trace"),
    "H15": ("script:run_acceptance.py", "re-run and seed-1 comparison"),
    "H16": (
        "report",
        "realised flag rates per flag x class x platform, uninformative marking",
    ),
    "H17": ("report", "the H1 / H2 metrics of the reseg report"),
    "H18": (
        "report+script",
        "PREP and reweighted emission tables (item 1); judged at M8",
    ),
}
MOUSE_CRITERIA_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "MO1": ("report+script", "G2 marker consistency here; MO1 by mouse_corr_shadow.py"),
    "MO2": (
        "report+script",
        "re-mapped cells and changes outside dropped nodes; T2 by script",
    ),
    "MO3": ("report", "inferred regions (compared with the manual sets at M9)"),
    "MO4": ("report", "soft class composition vs the AP-matched MERFISH window (G4)"),
    "MO5": ("report", "spill-over held-out enrichment and astrocyte FPR"),
    "MO6": ("report", "F1 (region-incoherent) rate after pruning"),
    "MO7": ("report", "mouse gate level"),
    "MO8": ("report+script", "recorded wall times; peak RSS from the trace"),
    "MO9": ("report+script", "per-segmentation reports; JSD vs original_seg by script"),
    "MO10": ("script:M6b", "AP-axis harm on MERFISH-638850 (M6b)"),
    "MO11": ("report+script", "first anterior section; AP estimate pending M6b"),
}


def _clean(value: Any) -> Any:
    """Return a JSON-safe value: finite floats rounded, NaN/inf -> None."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        return float(f"{number:.6g}")
    if isinstance(value, Mapping):
        return {
            str(key): _clean(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return [_clean(item) for item in value.tolist()]
    return str(value)


def clean_json(value: Any) -> Any:
    """Return ``value`` made JSON-safe and deterministic (see ``_clean``)."""
    return _clean(value)


class MetricRecord(BaseModel):
    """One measured metric (a row of ``acceptance_metrics.json``).

    Attributes:
        criterion: §14 id (``H1``, ``MO6``) or ``report`` for a report-only
            metric.
        name: Metric name (stable, snake case).
        scope: ``dataset`` (one sample) or ``pair``.
        sample_id: The sample (``None`` for pair metrics).
        platform: Its platform.
        region: ``whole_section`` / ``shared_mask`` where relevant.
        level: Annotation level where relevant.
        kind: Composition kind or variant where relevant.
        group: A class, supercluster or stratum where relevant.
        value: The value.
        ci_low: 95% CI lower bound, if any.
        ci_high: 95% CI upper bound, if any.
        n: The number of cells (or tiles, bins) behind it.
        definition: How it is computed.
        source: Where it comes from (``resolve_summary``, ``label_table``,
            ``report_metrics``, ``bundle``, ...).
        status: ``measured``, ``not_available`` or ``withheld`` (by the
            cross-platform scope).
        note: Anything a reader must know.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: str
    name: str
    scope: Literal["dataset", "pair"] = "dataset"
    sample_id: str | None = None
    platform: str | None = None
    region: str | None = None
    level: str | None = None
    kind: str | None = None
    group: str | None = None
    value: float | int | bool | str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    definition: str = ""
    source: str = ""
    status: MetricStatus = "measured"
    note: str = ""

    def sort_key(self: MetricRecord) -> tuple[str, ...]:
        """Return the deterministic ordering key."""
        return (
            _criterion_order(self.criterion),
            self.name,
            self.sample_id or "",
            self.region or "",
            self.level or "",
            self.kind or "",
            self.group or "",
        )


def _criterion_order(criterion: str) -> str:
    prefix = "".join(ch for ch in criterion if ch.isalpha())
    digits = "".join(ch for ch in criterion if ch.isdigit())
    return f"{prefix}{int(digits):04d}" if digits else f"zz_{criterion}"


def metric(
    criterion: str,
    name: str,
    value: Any,
    *,
    definition: str,
    source: str,
    **fields: Any,
) -> MetricRecord:
    """Build a ``MetricRecord`` with a JSON-safe value.

    Args:
        criterion: §14 id or ``report``.
        name: Metric name.
        value: The value (NaN becomes ``None`` with status ``not_available``
            unless a status is given).
        definition: How it is computed.
        source: Where it comes from.
        **fields: Other ``MetricRecord`` fields.

    Returns:
        The record.
    """
    cleaned = _clean(value)
    if isinstance(cleaned, list | dict):
        cleaned = json.dumps(cleaned, sort_keys=True)
    for key in ("ci_low", "ci_high"):
        if key in fields:
            fields[key] = _clean(fields[key])
    if "n" in fields and fields["n"] is not None:
        fields["n"] = int(fields["n"])
    status = fields.pop("status", None)
    if status is None:
        status = "measured" if cleaned is not None else "not_available"
    return MetricRecord(
        criterion=criterion,
        name=name,
        value=cleaned,
        definition=definition,
        source=source,
        status=status,
        **fields,
    )


class CriterionCoverage(BaseModel):
    """Where a §14 criterion's metric comes from.

    Attributes:
        criterion: The id.
        source: ``report``, ``report+script`` or ``script:<name>``.
        description: What is measured.
        n_records: Metric records the report wrote for it.
    """

    model_config = ConfigDict(extra="forbid")

    criterion: str
    source: str
    description: str
    n_records: int


class AcceptanceMetrics(BaseModel):
    """The ``acceptance_metrics.json`` document.

    Attributes:
        schema_version: Report schema version.
        kind: Always ``merxen_annotation_report_metrics``.
        species: ``human`` or ``mouse``.
        pair_id: Pair (or mouse section) id.
        segmentation: Segmentation.
        metrics: Every measured metric, sorted.
        criteria: Coverage of each §14 criterion.
        datasets: Per-sample digest (trust, gate, counts).
        items: Per report item: status and notes.
        provenance: The provenance footer (item 12).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = REPORT_SCHEMA_VERSION
    kind: str = "merxen_annotation_report_metrics"
    species: str
    pair_id: str
    segmentation: str
    metrics: list[MetricRecord] = Field(default_factory=list)
    criteria: list[CriterionCoverage] = Field(default_factory=list)
    datasets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    items: dict[str, dict[str, Any]] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    def write(self: AcceptanceMetrics, path: Path) -> Path:
        """Write the document (sorted keys, no NaN)."""
        payload = clean_json(self.model_dump(mode="python"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    def find(
        self: AcceptanceMetrics, criterion: str, name: str | None = None, **fields: Any
    ) -> list[MetricRecord]:
        """Return the records of a criterion (and name / field values)."""
        found = []
        for record in self.metrics:
            if record.criterion != criterion:
                continue
            if name is not None and record.name != name:
                continue
            if all(getattr(record, key) == value for key, value in fields.items()):
                found.append(record)
        return found


def criteria_coverage(
    species: str, metrics: Sequence[MetricRecord]
) -> list[CriterionCoverage]:
    """Return the coverage rows of a species' §14 criteria."""
    table = HUMAN_CRITERIA_SOURCES if species == "human" else MOUSE_CRITERIA_SOURCES
    counts: dict[str, int] = {}
    for record in metrics:
        counts[record.criterion] = counts.get(record.criterion, 0) + 1
    return [
        CriterionCoverage(
            criterion=criterion,
            source=source,
            description=description,
            n_records=counts.get(criterion, 0),
        )
        for criterion, (source, description) in table.items()
    ]


@dataclass(frozen=True)
class ReportOptions:
    """Settings of a report build.

    Attributes:
        n_bootstrap: Block-bootstrap replicates (200, §5.5).
        seed: Bootstrap seed.
        tile_um: Bootstrap tile edge (500 µm).
        density_bin_um: Aligned-bin edge of the density correlation (200 µm).
        max_dotplot_labels: Labels in the reference-expectation dotplot.
        genes_per_label: Specific genes per label in the dotplot.
        read_expression: Read the clustered counts (items 4, 7); off makes
            a report without them.
    """

    n_bootstrap: int = 200
    seed: int = 0
    tile_um: float = 500.0
    density_bin_um: float = 200.0
    max_dotplot_labels: int = 20
    genes_per_label: int = 2
    read_expression: bool = True


@dataclass
class ReportItem:
    """One report item (a §9 section).

    Attributes:
        number: §9 item number (1-12).
        slug: File-name slug, e.g. ``item01_annotatability``.
        title: Section title.
        catches: The failure the item makes visible (§9 "Catches").
        status: ``ok``, ``partial``, ``not_available`` or
            ``not_applicable``.
        figures: Written figures.
        tables: Extra written tables (name to path).
        summary: A small table shown in the HTML.
        notes: Notes shown in the HTML.
        metrics: Metric records.
        banners: Banners shown at the top of the section.
    """

    number: int
    slug: str
    title: str
    catches: str
    status: ItemStatus = "ok"
    figures: list[FigureRecord] = field(default_factory=list)
    tables: dict[str, Path] = field(default_factory=dict)
    summary: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    metrics: list[MetricRecord] = field(default_factory=list)
    banners: list[Banner] = field(default_factory=list)

    def digest(self: ReportItem) -> dict[str, Any]:
        """Return the item's status record for ``acceptance_metrics.json``."""
        return {
            "number": self.number,
            "title": self.title,
            "status": self.status,
            "notes": list(self.notes),
            "figures": [record.stem for record in self.figures],
            "tables": sorted(self.tables),
        }


@dataclass
class ItemWriter:
    """Writes an item's figures and tables into the report directory.

    Attributes:
        out_dir: The report directory.
        make_figures: Whether to draw figures (CSVs are always written).
    """

    out_dir: Path
    make_figures: bool = True

    @property
    def figures_dir(self: ItemWriter) -> Path:
        """Return the figures directory."""
        return self.out_dir / FIGURES_DIR

    @property
    def tables_dir(self: ItemWriter) -> Path:
        """Return the tables directory."""
        return self.out_dir / TABLES_DIR

    def figure(
        self: ItemWriter,
        item: ReportItem,
        stem: str,
        draw: Any,
        data: pd.DataFrame,
        caption: str,
    ) -> None:
        """Draw (``draw()`` returns a Figure) and save one figure with its CSV."""
        from merxen.annotation.report_figures import placeholder, save_report_figure

        full_stem = f"{item.slug}_{stem}"
        figure = draw() if self.make_figures else placeholder("figures disabled")
        item.figures.append(
            save_report_figure(figure, self.figures_dir, full_stem, data, caption)
        )

    def table(
        self: ItemWriter, item: ReportItem, name: str, frame: pd.DataFrame
    ) -> Path:
        """Write an extra table of an item."""
        path = self.tables_dir / f"{item.slug}__{name}.csv"
        write_csv(frame, path)
        item.tables[name] = path
        return path


__all__ = [
    "FIGURES_DIR",
    "HUMAN_CRITERIA_SOURCES",
    "METRICS_FILENAME",
    "MOUSE_CRITERIA_SOURCES",
    "REPORT_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "RUN_FILENAME",
    "TABLES_DIR",
    "AcceptanceMetrics",
    "CriterionCoverage",
    "ItemWriter",
    "MetricRecord",
    "ReportItem",
    "ReportOptions",
    "clean_json",
    "criteria_coverage",
    "metric",
]
