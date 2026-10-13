"""The annotation QC report (``ANNOTATION_REPORT``; plan §9, §3.6; M7).

``build_annotation_report`` reads the published outputs of one human pair ×
segmentation (or one mouse section × segmentation) through
``report_inputs`` and writes, into a fresh output directory:

- ``report.html``: a static page (inline CSS, no network assets) with the
  twelve §9 items, their banners, notes and figures;
- ``figures/<item>_<name>.{png,pdf,csv}``: every figure as PNG and PDF with
  the CSV of exactly what it draws;
- ``tables/<item>__<name>.csv``: the item tables (the panel card, gate,
  compositions, drop lists, ...), and ``<pair>_platform_gene_factors.csv``;
- ``acceptance_metrics.json``: every measured metric (``MetricRecord``) with
  the §14 criterion it serves, the coverage of each criterion and the
  provenance footer (item 12); deterministic for identical inputs;
- ``report_run.json``: wall time and versions of the build (not
  deterministic, so kept apart).

The report never writes into an input directory or a results tree unless
``allow_results_output`` is set, and never decides a gate: pass / fail
against the pre-registered thresholds is the acceptance programme's (M8,
M9). Items whose inputs are absent (cortical depth, the shared mask, the
held-out acceptance output, the clustered counts) are reported as
``not_available`` with the reason; refused and provisional panels get their
banners (plan §8.2).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from merxen.annotation.report_depth import item_cortical_depth
from merxen.annotation.report_expression import (
    item_cross_platform,
    item_reference_expectation,
)
from merxen.annotation.report_html import render_report
from merxen.annotation.report_inputs import (
    ReportInputs,
    ReportSources,
    load_report_inputs,
    recorded_bundle_dirs,
    report_results_root,
)
from merxen.annotation.report_items import (
    item_ad_ood,
    item_annotatability,
    item_composition,
    item_confidence_vs_counts,
    item_heldout,
    item_method_agreement,
    item_provenance,
    item_purity,
    new_item,
)
from merxen.annotation.report_model import (
    METRICS_FILENAME,
    REPORT_FILENAME,
    RUN_FILENAME,
    AcceptanceMetrics,
    ItemWriter,
    ReportItem,
    ReportOptions,
    clean_json,
    criteria_coverage,
)
from merxen.annotation.report_mouse import item_mouse_regions

logger = logging.getLogger(__name__)

ItemBuilder = Callable[[ReportInputs, ItemWriter, ReportOptions], ReportItem]
ITEM_BUILDERS: Final[tuple[tuple[int, str, str, ItemBuilder], ...]] = (
    (
        1,
        "annotatability",
        "Annotatability, resolvability and the panel",
        item_annotatability,
    ),
    (2, "composition", "Composition per level", item_composition),
    (3, "confidence_vs_counts", "Confidence vs counts", item_confidence_vs_counts),
    (
        4,
        "reference_expectation",
        "Reference-expectation dotplots",
        item_reference_expectation,
    ),
    (5, "negative_marker_purity", "Negative-marker purity", item_purity),
    (6, "method_agreement", "Method agreement", item_method_agreement),
    (7, "cross_platform", "Cross-platform concordance", item_cross_platform),
    (8, "heldout_genes", "Held-out-gene enrichment", item_heldout),
    (9, "cortical_depth", "Cortical depth QC", item_cortical_depth),
    (10, "mouse_regions", "Mouse region plausibility", item_mouse_regions),
    (11, "ad_ood", "AD and OOD panel", item_ad_ood),
)
# Failures of one item are recorded in the report (status "failed") so the
# other items are still written; these are the errors a malformed or
# unexpected input raises. ``strict`` re-raises them (tests, debugging).
ITEM_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    AttributeError,
    OSError,
    ZeroDivisionError,
)


class ReportOutputError(ValueError):
    """The report output directory is not allowed or not empty."""


@dataclass
class ReportResult:
    """What a report build wrote.

    Attributes:
        out_dir: The report directory.
        html: ``report.html``.
        metrics_path: ``acceptance_metrics.json``.
        metrics: The metrics document.
        items: The items, in §9 order.
        wall_time_s: Build wall time.
    """

    out_dir: Path
    html: Path
    metrics_path: Path
    metrics: AcceptanceMetrics
    items: list[ReportItem]
    wall_time_s: float


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def check_output_dir(
    out_dir: Path,
    sources: ReportSources,
    *,
    allow_results_output: bool = False,
    overwrite: bool = False,
) -> None:
    """Refuse an output directory inside the inputs, or a non-empty one.

    The report never writes into a results tree (rule R3, as MAP and
    RESOLVE): not inside an input directory or ``--results-root``, not at or
    below the results root of any input (``report_results_root``: the
    RESOLVE, MAP and PANEL outputs, the clustered H5ADs, the depth tables,
    the MENDER manifests, the alignment), not inside the reference store or
    a bundle RESOLVE's manifests record (stores only gain new build
    directories), and not beside the held-out-gene CSV.

    Args:
        out_dir: The requested report directory.
        sources: The report's inputs.
        allow_results_output: Allow writing inside the results tree / an
            input directory (an explicit publish into a results tree).
        overwrite: Allow a non-empty directory (files are replaced, none is
            deleted).

    Raises:
        ReportOutputError: If the directory is not allowed.
    """
    if not allow_results_output:
        for root in sources.input_roots():
            if _is_within(out_dir, root):
                raise ReportOutputError(
                    f"refusing to write the report into {out_dir}: it lies inside the "
                    f"input {root} (pass allow_results_output / --allow-results-output "
                    "to publish into a results tree)"
                )
        for directory, reason in guarded_directories(sources):
            if _is_within(out_dir, directory):
                raise ReportOutputError(
                    f"refusing to write the report into {out_dir}: it lies inside "
                    f"{directory}, {reason} (write the report elsewhere, or pass "
                    "allow_results_output / --allow-results-output)"
                )
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ReportOutputError(
            f"{out_dir} is not empty (pass overwrite / --overwrite)"
        )


def guarded_directories(sources: ReportSources) -> list[tuple[Path, str]]:
    """Return the directories a report output must stay out of, with the reason.

    Args:
        sources: The report's inputs.

    Returns:
        ``(directory, reason)`` pairs: every input's results root, the
        reference store, the bundles RESOLVE's manifests record and the
        held-out CSV's directory.
    """
    guarded: list[tuple[Path, str]] = []
    for path in sources.input_paths():
        root = report_results_root(path)
        if root is not None:
            guarded.append((root, f"the results tree of the input {path}"))
    if sources.store_root is not None:
        guarded.append((Path(sources.store_root), "the reference store"))
    for bundle in recorded_bundle_dirs(sources.resolve_dir, sources.pair_id):
        guarded.append((bundle, "a reference bundle RESOLVE's manifests record"))
    if sources.heldout_csv is not None:
        guarded.append(
            (Path(sources.heldout_csv).parent, "the held-out-gene CSV's directory")
        )
    return guarded


def _failed_item(
    number: int, slug: str, title: str, error: BaseException
) -> ReportItem:
    item = new_item(number, slug, title, "")
    item.status = "failed"
    item.notes.append(f"item failed: {type(error).__name__}: {error}")
    return item


def dataset_digest(inputs: ReportInputs) -> dict[str, dict[str, Any]]:
    """Return the per-sample digest (trust, gate, counts) of the report."""
    digest: dict[str, dict[str, Any]] = {}
    for sample in inputs.ordered_samples():
        resolution = sample.summary.get("resolution") or {}
        gate = resolution.get("gate") or {}
        trust = sample.summary.get("trust") or {}
        panel = sample.manifest.get("panel") or {}
        digest[sample.sample_id] = {
            "platform": sample.platform,
            "n_segmented": sample.summary.get("n_segmented")
            or sample.summary.get("n_objects"),
            "n_table": sample.summary.get("n_table"),
            "trust_state": trust.get("state") or panel.get("panel_trust"),
            "validation_basis": trust.get("validation_basis")
            or panel.get("validation_basis"),
            "panel_family": panel.get("panel_family") or trust.get("family_id"),
            "gate_level": gate.get("level"),
            "gate_warning": gate.get("warning"),
            "banner": sample.summary.get("banner"),
            "degraded_mode": (resolution.get("degraded_mode") or {}).get("name"),
            "coordinates": sample.coordinate_source,
            "cortical_depth": sample.depth_path is not None,
        }
    return digest


def build_annotation_report(
    sources: ReportSources,
    out_dir: Path | str,
    *,
    options: ReportOptions | None = None,
    allow_results_output: bool = False,
    overwrite: bool = False,
    make_figures: bool = True,
    strict: bool = False,
    items: Sequence[int] | None = None,
) -> ReportResult:
    """Build the annotation report of one pair (or mouse section) × segmentation.

    Args:
        sources: Where the inputs are (``report_inputs.discover_sources``).
        out_dir: A fresh output directory.
        options: Report settings.
        allow_results_output: Allow ``out_dir`` inside the results tree.
        overwrite: Allow a non-empty ``out_dir``.
        make_figures: Draw figures (``False``: placeholders; CSVs still
            written).
        strict: Re-raise an item's error instead of recording it.
        items: §9 item numbers to build (default: all; 12 is always built).

    Returns:
        What was written.

    Raises:
        ReportOutputError: If ``out_dir`` is not allowed.
        ReportInputError: If the RESOLVE outputs are missing.
    """
    started = time.perf_counter()
    options = options or ReportOptions()
    out = Path(out_dir)
    check_output_dir(
        out, sources, allow_results_output=allow_results_output, overwrite=overwrite
    )
    inputs = load_report_inputs(sources)
    out.mkdir(parents=True, exist_ok=True)
    writer = ItemWriter(out, make_figures=make_figures)
    built: list[ReportItem] = []
    selected = None if items is None else set(items)
    for number, slug, title, builder in ITEM_BUILDERS:
        if selected is not None and number not in selected:
            continue
        logger.info("report item %d: %s", number, title)
        try:
            built.append(builder(inputs, writer, options))
        except ITEM_ERRORS as error:
            if strict:
                raise
            logger.exception("report item %d failed", number)
            built.append(_failed_item(number, slug, title, error))
    provenance_item, footer = item_provenance(inputs, writer, options)
    built.append(provenance_item)
    metrics = [record for item in built for record in item.metrics]
    metrics.sort(key=lambda record: record.sort_key())
    document = AcceptanceMetrics(
        species=inputs.species,
        pair_id=sources.pair_id,
        segmentation=sources.segmentation,
        metrics=metrics,
        criteria=criteria_coverage(inputs.species, metrics),
        datasets=clean_json(dataset_digest(inputs)),
        items={item.slug: item.digest() for item in built},
        provenance=clean_json(footer),
    )
    metrics_path = document.write(out / METRICS_FILENAME)
    title = (
        f"Annotation report: {sources.pair_id} / {sources.segmentation} "
        f"({inputs.species})"
    )
    html_path = out / REPORT_FILENAME
    html_path.write_text(
        render_report(
            title=title,
            items=built,
            datasets=document.datasets,
            provenance=document.provenance,
            root=out,
        ),
        encoding="utf-8",
    )
    wall = time.perf_counter() - started
    from merxen import __version__

    (out / RUN_FILENAME).write_text(
        json.dumps(
            {
                "wall_time_s": round(wall, 3),
                "merxen_version": __version__,
                "options": options.__dict__,
                "items": {item.slug: item.status for item in built},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("annotation report written to %s in %.1f s", out, wall)
    return ReportResult(
        out_dir=out,
        html=html_path,
        metrics_path=metrics_path,
        metrics=document,
        items=built,
        wall_time_s=wall,
    )


__all__ = [
    "ITEM_BUILDERS",
    "ReportOutputError",
    "ReportResult",
    "build_annotation_report",
    "check_output_dir",
    "dataset_digest",
    "guarded_directories",
]
