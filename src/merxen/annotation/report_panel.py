"""The panel card and trust banners of the annotation report (plan §8.2, §9 item 1).

The panel card states what a dataset's gene panel supports, from the records
the annotation stages wrote (``panel_report.json``, ``bundle.json``,
``query_markers.filtered.json``, ``resolvability.parquet``,
``resolvability_summary.json``, the annotation manifest and RESOLVE's
summary); nothing here re-derives a decision:

- gene-ID resolution by source, unresolved features, controls removed by
  type (per declared panel);
- markers per parent (min / median / max and per parent), weak parents
  (< ``weak_parent_markers``, 5, panel markers) and collapsed parents, and
  the nodes MapMyCells patches with their ancestors' markers ("too few
  markers");
- resolvability precision-coverage curves per level and depth, D_max and the
  ``resolvability_extrapolated`` share per class, the unweighted (PREP) and
  the reweighted (RESOLVE) emission, the thresholds and floors with their
  source, missing declared-panel genes;
- the trust state, validation basis, validated share per level, real-data
  QC outcomes, and the banners for ``provisional``, ``broad_only`` and
  ``refused`` panels (and for gates that fail or are broad-only).

The glial-upper-bound note of the M3c version-7 families is rendered only
when a bundle's resolvability summary is version 7 or later (M3c is not
merged, so its wording is not fixed here; any note the bundle records is
shown verbatim).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

logger = logging.getLogger(__name__)

V7_RESOLVABILITY_VERSION: Final = 7
GENERIC_V7_NOTE: Final = (
    "Resolvability version {version} family: simulated glial coverage is an "
    "upper bound on real data, and precision is unmeasured on real data "
    "(M3c diagnostics)."
)
BANNER_TEXT: Final[dict[str, str]] = {
    "refused": (
        "REFUSED panel: MapMyCells was not run for the primary reference; every "
        "table cell is not_attempted_gate and the dataset gate is failed "
        "(panel_refused)."
    ),
    "broad_only": (
        "BROAD-ONLY panel: the leaf level is not resolvable on this panel; leaves "
        "are unresolved (subcluster_status not_resolvable_panel)."
    ),
    "provisional": (
        "PROVISIONAL panel: its family is not validated; emission uses the "
        "provisional margins, the gate warning flag is set and the dataset is "
        "listed separately in pooled statistics."
    ),
}
GATE_BANNER_TEXT: Final[dict[str, str]] = {
    "failed": "Dataset gate FAILED: every cell is exclude_hard ({reasons}).",
    "broad_only": (
        "Dataset gate BROAD-ONLY: supercluster and SEA-AD subclass are "
        "not_attempted_gate ({reasons})."
    ),
}


@dataclass(frozen=True)
class Banner:
    """A report banner.

    Attributes:
        severity: ``error`` (refused, failed), ``warning`` (provisional,
            broad-only, gate warnings) or ``info``.
        sample_id: The sample it concerns (``None``: the pair).
        code: A stable code (``panel_refused``, ``panel_provisional``, ...).
        text: The message.
    """

    severity: str
    sample_id: str | None
    code: str
    text: str


@dataclass
class PanelCard:
    """A dataset's panel card (report item 1).

    Attributes:
        summary: One row per sample: trust state, basis, family, hashes,
            counts of genes, controls, unresolved, markers.
        gene_ids: Rows ``sample_id``, ``source``, ``n_genes``.
        unresolved: Rows ``sample_id``, ``feature``, ``reason``.
        controls: Rows ``sample_id``, ``control_type``, ``n_features``.
        markers: Rows ``reference_id``, ``parent``, ``parent_name``,
            ``n_markers``, ``weak``, ``collapsed``.
        curves: Precision-coverage curves (``reference_id``, ``level``,
            ``class``, ``depth``, ``threshold``, ``precision``, ``coverage``).
        emission: Per (reference, level, class, depth): the unweighted PREP
            verdict (status, precision, Wilson bound, t*) and whether RESOLVE
            emitted it after reweighting, per sample.
        classes: Per (sample, class): D_max, extrapolated share.
        thresholds: Per sample: threshold and floor values with sources.
        floors: The packaged floors of the dataset's platform(s) and family.
        banners: Trust and gate banners.
        notes: Free-text notes (v7 notes, real-data QC, missing genes).
    """

    summary: pd.DataFrame
    gene_ids: pd.DataFrame
    unresolved: pd.DataFrame
    controls: pd.DataFrame
    markers: pd.DataFrame
    curves: pd.DataFrame
    emission: pd.DataFrame
    classes: pd.DataFrame
    thresholds: pd.DataFrame
    floors: pd.DataFrame
    banners: list[Banner] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def tables(self: PanelCard) -> dict[str, pd.DataFrame]:
        """Return the card's tables by name."""
        return {
            "panel_summary": self.summary,
            "panel_gene_ids": self.gene_ids,
            "panel_unresolved": self.unresolved,
            "panel_controls": self.controls,
            "panel_markers_per_parent": self.markers,
            "panel_resolvability_curves": self.curves,
            "panel_emission": self.emission,
            "panel_classes": self.classes,
            "panel_thresholds": self.thresholds,
            "panel_floors": self.floors,
        }


def _frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def trust_banners(
    sample_id: str,
    trust: Mapping[str, Any] | None,
    gate: Mapping[str, Any] | None,
    *,
    real_data_qc: Mapping[str, Any] | None = None,
) -> list[Banner]:
    """Return the banners of one sample (plan §8.2, §5.4, §8.8).

    Args:
        sample_id: Sample id.
        trust: The primary reference's trust record (RESOLVE summary
            ``trust``: ``state``, ``validation_basis``, ``effects``).
        gate: The dataset gate (``level``, ``warning``, reasons).
        real_data_qc: The real-data QC record, if any.

    Returns:
        Banners, most severe first.
    """
    banners: list[Banner] = []
    state = str((trust or {}).get("state") or "unknown")
    if state == "refused":
        banners.append(
            Banner("error", sample_id, "panel_refused", BANNER_TEXT["refused"])
        )
    elif state == "broad_only":
        banners.append(
            Banner("warning", sample_id, "panel_broad_only", BANNER_TEXT["broad_only"])
        )
    elif state == "provisional":
        banners.append(
            Banner(
                "warning", sample_id, "panel_provisional", BANNER_TEXT["provisional"]
            )
        )
    elif state == "unknown":
        banners.append(
            Banner(
                "warning",
                sample_id,
                "panel_trust_unknown",
                "Panel trust state not recorded.",
            )
        )
    level = str((gate or {}).get("level") or "")
    reasons = [
        str(item)
        for item in (
            list((gate or {}).get("level_reasons") or [])
            + list((gate or {}).get("reasons") or [])
        )
    ]
    if level in GATE_BANNER_TEXT:
        text = GATE_BANNER_TEXT[level].format(
            reasons="; ".join(reasons) or "no reason recorded"
        )
        banners.append(
            Banner(
                "error" if level == "failed" else "warning",
                sample_id,
                f"gate_{level}",
                text,
            )
        )
    if (gate or {}).get("warning"):
        warning_reasons = [
            str(item) for item in (gate or {}).get("warning_reasons") or []
        ]
        banners.append(
            Banner(
                "warning",
                sample_id,
                "gate_warning",
                "Dataset gate warning: " + ("; ".join(warning_reasons) or "set"),
            )
        )
    if real_data_qc:
        downgrades = real_data_qc.get("downgrades") or real_data_qc.get("effects") or []
        if downgrades:
            banners.append(
                Banner(
                    "warning",
                    sample_id,
                    "real_data_qc_downgrade",
                    "Real-data QC downgrade: " + json.dumps(downgrades, sort_keys=True),
                )
            )
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(banners, key=lambda banner: order.get(banner.severity, 3))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning("could not read %s: %s", path, error)
        return None
    return payload if isinstance(payload, dict) else None


def bundle_marker_rows(reference_id: str, bundle_dir: Path) -> list[dict[str, Any]]:
    """Return the markers per parent of a bundle's filtered lookup.

    Args:
        reference_id: Reference id.
        bundle_dir: The bundle directory.

    Returns:
        One row per parent (``None`` = the root), with the weak and
        collapsed marks from ``bundle.json``.
    """
    manifest = _read_json(bundle_dir / "bundle.json") or {}
    markers = (manifest.get("builder_output") or {}).get("markers") or {}
    lookup_name = str(markers.get("lookup_file") or "query_markers.filtered.json")
    lookup = _read_json(bundle_dir / lookup_name) or {}
    weak = {str(item) for item in markers.get("weak_parents") or []}
    collapsed = {
        str(item.get("parent") if isinstance(item, Mapping) else item)
        for item in markers.get("collapsed_parents") or []
    }
    names = _node_names(bundle_dir)
    rows = []
    for parent, genes in sorted(lookup.items()):
        if parent == "metadata" or not isinstance(genes, list):
            continue
        node = parent.split("/")[-1]
        rows.append(
            {
                "reference_id": reference_id,
                "parent": parent,
                "parent_name": "root" if parent == "None" else names.get(node, node),
                "n_markers": len(genes),
                "weak": parent in weak,
                "collapsed": parent in collapsed,
            }
        )
    return rows


def _node_names(bundle_dir: Path) -> dict[str, str]:
    path = bundle_dir / "profiles.parquet"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_parquet(path, columns=["node", "node_name"])
    except (ValueError, KeyError, OSError):
        return {}
    frame = frame.drop_duplicates("node")
    return dict(
        zip(frame["node"].astype(str), frame["node_name"].astype(str), strict=True)
    )


def resolvability_curves(reference_id: str, bundle_dir: Path) -> pd.DataFrame:
    """Return the PREP precision-coverage curves of a bundle.

    Args:
        reference_id: Reference id.
        bundle_dir: The bundle directory.

    Returns:
        Rows of the decision recipe's ``curve`` records (empty without a
        resolvability table).
    """
    columns = [
        "reference_id",
        "level",
        "class",
        "depth",
        "threshold",
        "precision",
        "coverage",
        "n_confident",
    ]
    path = bundle_dir / "resolvability.parquet"
    if not path.is_file():
        return _frame([], columns)
    frame = pd.read_parquet(path)
    if "kind" not in frame.columns:
        return _frame([], columns)
    curves = frame[frame["kind"] == "curve"].copy()
    recipe = _decision_recipe(bundle_dir)
    if recipe is not None and "recipe" in curves.columns:
        curves = curves[curves["recipe"] == recipe]
    curves.insert(0, "reference_id", reference_id)
    keep = [column for column in columns if column in curves.columns]
    return (
        curves[keep]
        .sort_values(["level", "class", "depth", "threshold"])
        .reset_index(drop=True)
    )


def _decision_recipe(bundle_dir: Path) -> str | None:
    summary = _read_json(bundle_dir / "resolvability_summary.json") or {}
    recipe = summary.get("decision_recipe")
    return None if recipe is None else str(recipe)


def resolvability_bins(reference_id: str, bundle_dir: Path) -> pd.DataFrame:
    """Return the PREP (unweighted) per-(level, class, depth) verdicts.

    Args:
        reference_id: Reference id.
        bundle_dir: The bundle directory.

    Returns:
        The ``decision`` rows of the decision recipe, one per (regime, level,
        class, depth) (``bin`` rows for tables without decisions; empty
        without a table).
    """
    columns = [
        "reference_id",
        "level",
        "class",
        "depth",
        "regime",
        "n_test",
        "n_confident",
        "precision",
        "coverage",
        "wilson_lb",
        "t_star",
        "target",
        "default_threshold",
        "status",
        "reason",
        "extrapolated",
        "pooled",
    ]
    path = bundle_dir / "resolvability.parquet"
    if not path.is_file():
        return _frame([], columns)
    frame = pd.read_parquet(path)
    if "kind" not in frame.columns:
        return _frame([], columns)
    kind = "decision" if (frame["kind"] == "decision").any() else "bin"
    bins = frame[frame["kind"] == kind].copy()
    recipe = _decision_recipe(bundle_dir)
    if recipe is not None and "recipe" in bins.columns:
        bins = bins[bins["recipe"] == recipe]
    bins.insert(0, "reference_id", reference_id)
    keep = [column for column in columns if column in bins.columns]
    order = [c for c in ("regime", "level", "class", "depth") if c in bins.columns]
    return bins[keep].sort_values(order).reset_index(drop=True)


def v7_notes(bundle_dir: Path) -> list[str]:
    """Return the version-7 (M3c) notes of a bundle's resolvability summary.

    Args:
        bundle_dir: The bundle directory.

    Returns:
        The notes the summary records, or the generic glial-upper-bound
        note; empty below version 7.
    """
    summary = _read_json(bundle_dir / "resolvability_summary.json") or {}
    try:
        version = int(summary.get("resolvability_version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version < V7_RESOLVABILITY_VERSION:
        return []
    recorded = [
        str(item)
        for key in ("panel_card_notes", "notes", "glial_upper_bound")
        for item in _as_list(summary.get(key))
        if item
    ]
    return recorded or [GENERIC_V7_NOTE.format(version=version)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _declared_rows(
    panel_report: Mapping[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    declared = (panel_report or {}).get("declared_panels") or {}
    return [(str(key), dict(value)) for key, value in sorted(declared.items())]


def _count(value: Any) -> int:
    if isinstance(value, Mapping | list | tuple | set):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_panel_card(
    samples: Sequence[Mapping[str, Any]],
    *,
    panel_report: Mapping[str, Any] | None,
    bundles: Mapping[str, Path],
    primary_reference: str | None,
    floors: pd.DataFrame | None = None,
) -> PanelCard:
    """Assemble the panel card of a report.

    Args:
        samples: Per sample ``{"sample_id", "platform", "summary",
            "manifest", "n_missing_panel_genes"}`` (the RESOLVE summary entry
            and the annotation manifest).
        panel_report: ``panel_report.json`` (``None`` when absent).
        bundles: Reference id to bundle directory.
        primary_reference: The primary reference id.
        floors: The packaged floors table (``vocab.load_floor_table``).

    Returns:
        The panel card.
    """
    summary_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    banners: list[Banner] = []
    notes: list[str] = []
    declared = dict(_declared_rows(panel_report))
    families: set[str] = set()
    platforms: set[str] = set()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        platform = str(sample.get("platform") or "")
        platforms.add(platform)
        entry = dict(sample.get("summary") or {})
        manifest = dict(sample.get("manifest") or {})
        panel = dict(manifest.get("panel") or {})
        trust = dict(entry.get("trust") or {})
        gate = dict((entry.get("resolution") or {}).get("gate") or {})
        family = panel.get("panel_family") or trust.get("family_id")
        if family:
            families.add(str(family))
        record = declared.get(sample_id, {})
        resolution = (
            record.get("gene_id_resolution") or panel.get("gene_id_resolution") or {}
        )
        for source, count in sorted(dict(resolution).items()):
            gene_rows.append(
                {"sample_id": sample_id, "source": source, "n_genes": _count(count)}
            )
        for feature, reason in sorted(dict(record.get("unresolved") or {}).items()):
            unresolved_rows.append(
                {"sample_id": sample_id, "feature": feature, "reason": reason}
            )
        controls = record.get("controls_removed") or panel.get("controls_removed") or {}
        for control_type, names in sorted(dict(controls).items()):
            control_rows.append(
                {
                    "sample_id": sample_id,
                    "control_type": control_type,
                    "n_features": _count(names),
                }
            )
        refs = manifest.get("references") or {}
        primary = refs.get(primary_reference or "", {}) if primary_reference else {}
        marker_record = dict(primary.get("markers") or {})
        summary_rows.append(
            {
                "sample_id": sample_id,
                "platform": platform,
                "trust_state": trust.get("state") or panel.get("panel_trust"),
                "validation_basis": trust.get("validation_basis")
                or panel.get("validation_basis"),
                "validated_max_level": trust.get("validated_max_level")
                or panel.get("validated_max_level"),
                "panel_family": family,
                "family_basis": trust.get("family_basis") or panel.get("family_basis"),
                "panel_hash": panel.get("panel_hash") or trust.get("panel_hash"),
                "declared_panel_hash": record.get("panel_hash"),
                "n_declared_genes": panel.get("n_declared_genes")
                or record.get("n_genes"),
                "n_features_in": record.get("n_features_in"),
                "n_controls_removed": record.get("n_controls_removed"),
                "gene_id_resolution_share": record.get("gene_id_resolution_share"),
                "n_unresolved": len(record.get("unresolved") or {}),
                "n_unmapped": panel.get("n_unmapped"),
                "n_missing_panel_genes": sample.get("n_missing_panel_genes"),
                "species_check": (record.get("species_check") or {}).get("status"),
                "n_query_genes_used": primary.get("n_query_genes_used"),
                "markers_per_parent_min": marker_record.get("markers_per_parent_min"),
                "markers_per_parent_median": marker_record.get(
                    "markers_per_parent_median"
                ),
                "trust_reasons": "; ".join(
                    str(item.get("code") if isinstance(item, Mapping) else item)
                    for item in (
                        trust.get("reasons") or panel.get("trust_reasons") or []
                    )
                ),
                "banner": bool(entry.get("banner") or panel.get("banner")),
                "validated_share": json.dumps(
                    panel.get("validated_share") or {}, sort_keys=True
                ),
            }
        )
        banners.extend(
            trust_banners(
                sample_id, trust, gate, real_data_qc=panel.get("real_data_qc")
            )
        )
        missing = sample.get("n_missing_panel_genes")
        if missing:
            notes.append(
                f"{sample_id}: {missing} declared panel genes absent from the data"
            )
        for reference_id, record_ in sorted(
            (manifest.get("resolvability") or {}).items()
        ):
            d_max = dict(record_.get("d_max") or {})
            extrapolated = dict(record_.get("extrapolated_share") or {})
            for cls in sorted(set(d_max) | set(extrapolated)):
                class_rows.append(
                    {
                        "sample_id": sample_id,
                        "reference_id": reference_id,
                        "class": cls,
                        "d_max": d_max.get(cls),
                        "extrapolated_share": extrapolated.get(cls),
                    }
                )
        thresholds = dict(manifest.get("thresholds") or {})
        for key, value in sorted(dict(thresholds.get("values") or {}).items()):
            threshold_rows.append(
                {
                    "sample_id": sample_id,
                    "name": key,
                    "value": value,
                    "threshold_source": thresholds.get("threshold_source"),
                    "floor_source": thresholds.get("floor_source"),
                    "floors_sha256": thresholds.get("floors_sha256"),
                    "calibration": thresholds.get("calibration"),
                }
            )
        for level, record_ in sorted(
            ((entry.get("resolution") or {}).get("levels") or {}).items()
        ):
            emission = dict(record_.get("emission") or {})
            threshold_rows.append(
                {
                    "sample_id": sample_id,
                    "name": f"emission:{level}",
                    "value": emission.get("regime"),
                    "threshold_source": emission.get("threshold_source"),
                    "floor_source": thresholds.get("floor_source"),
                    "floors_sha256": thresholds.get("floors_sha256"),
                    "calibration": thresholds.get("calibration"),
                }
            )
    marker_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    for reference_id, bundle_dir in sorted(bundles.items()):
        marker_rows.extend(bundle_marker_rows(reference_id, Path(bundle_dir)))
        curve_frames.append(resolvability_curves(reference_id, Path(bundle_dir)))
        bin_frames.append(resolvability_bins(reference_id, Path(bundle_dir)))
        notes.extend(f"{reference_id}: {note}" for note in v7_notes(Path(bundle_dir)))
        manifest = _read_json(Path(bundle_dir) / "bundle.json") or {}
        unsupported = (manifest.get("builder_output") or {}).get(
            "marker_unsupported_nodes"
        ) or {}
        n_unsupported = unsupported.get("n_nodes")
        if n_unsupported:
            notes.append(
                f"{reference_id}: {n_unsupported} nodes have too few markers of their "
                "own (MapMyCells patches them with their ancestors' markers)"
            )
    for sample in samples:
        calls = (sample.get("summary") or {}).get("marker_unsupported_calls")
        if calls and any(_count(value) for value in dict(calls).values()):
            notes.append(
                f"{sample['sample_id']}: calls on marker-unsupported nodes "
                + json.dumps(calls, sort_keys=True)
            )
    curves = (
        pd.concat([frame for frame in curve_frames if len(frame)], ignore_index=True)
        if any(len(frame) for frame in curve_frames)
        else resolvability_curves("", Path("/nonexistent"))
    )
    bins = (
        pd.concat([frame for frame in bin_frames if len(frame)], ignore_index=True)
        if any(len(frame) for frame in bin_frames)
        else resolvability_bins("", Path("/nonexistent"))
    )
    emission = _emission_table(bins, samples)
    floor_table = pd.DataFrame()
    if floors is not None and len(floors):
        floor_table = floors.copy()
        if "platform" in floor_table.columns and platforms:
            floor_table = floor_table[floor_table["platform"].isin(sorted(platforms))]
        if "panel_family" in floor_table.columns and families:
            matched = floor_table[floor_table["panel_family"].isin(sorted(families))]
            floor_table = matched if len(matched) else floor_table
        floor_table = floor_table.reset_index(drop=True)
    return PanelCard(
        summary=_frame(
            summary_rows, list(summary_rows[0]) if summary_rows else ["sample_id"]
        ),
        gene_ids=_frame(gene_rows, ["sample_id", "source", "n_genes"]),
        unresolved=_frame(unresolved_rows, ["sample_id", "feature", "reason"]),
        controls=_frame(control_rows, ["sample_id", "control_type", "n_features"]),
        markers=_frame(
            marker_rows,
            ["reference_id", "parent", "parent_name", "n_markers", "weak", "collapsed"],
        ),
        curves=curves,
        emission=emission,
        classes=_frame(
            class_rows,
            ["sample_id", "reference_id", "class", "d_max", "extrapolated_share"],
        ),
        thresholds=_frame(
            threshold_rows,
            [
                "sample_id",
                "name",
                "value",
                "threshold_source",
                "floor_source",
                "floors_sha256",
                "calibration",
            ],
        ),
        floors=floor_table,
        banners=banners,
        notes=notes,
    )


def _emission_table(
    bins: pd.DataFrame, samples: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Join PREP's unweighted verdicts with RESOLVE's reweighted emission."""
    frame = bins.copy()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        emitted: dict[tuple[str, str, str, int], bool] = {}
        manifest = dict(sample.get("manifest") or {})
        recorded = {str(key) for key in (manifest.get("resolvability") or {})}
        for reference_id, record in (manifest.get("resolvability") or {}).items():
            for level, classes in (record.get("emitted_depth_bins") or {}).items():
                for cls, depths in (classes or {}).items():
                    for depth in depths or []:
                        emitted[
                            (
                                str(reference_id),
                                str(level),
                                str(cls).lower(),
                                int(depth),
                            )
                        ] = True
        column = f"emitted_reweighted_{sample_id}"
        if len(frame) and emitted:
            keys = zip(
                frame["reference_id"].astype(str),
                frame["level"].astype(str),
                frame["class"].astype(str).str.lower(),
                frame["depth"].astype(int),
                strict=True,
            )
            frame[column] = [
                emitted.get(key, False) if key[0] in recorded else math.nan
                for key in keys
            ]
        else:
            frame[column] = pd.Series([math.nan] * len(frame), dtype=object)
    return frame


__all__ = [
    "BANNER_TEXT",
    "GENERIC_V7_NOTE",
    "Banner",
    "PanelCard",
    "build_panel_card",
    "bundle_marker_rows",
    "resolvability_bins",
    "resolvability_curves",
    "trust_banners",
    "v7_notes",
]
