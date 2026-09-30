"""Report item 10 (plan §9, §7.2-7.8): mouse region plausibility.

From the MAP record of the region step (``map_manifest.json`` →
``samples[*].mouse_regions``), the per-cell region table
(``<sid>_mouse_regions.parquet``), RESOLVE's summary (region step, F1
coherence, spill-over checks, the mouse gate G1-G5) and the label table:

- inferred vs explicit regions, tile counts, the tile map and the drop list;
- relabelled cells and their targets (the pruned re-map), with the
  mechanics MO2 needs (class changes outside dropped nodes, pruned calls
  still in a dropped node);
- subclass × region shares vs the MERFISH region shares (``wmb_region_share``
  bundle, when found);
- class composition vs the AP-matched MERFISH window (G4), Astro-Epen with
  and without low-count cells, F1 rate, spill-over prevalence and its map
  (never a count), the derived spill-over gene set, and the gate.

**M6b fields are optional and pending:** the AP estimate and bin (§7.8
step 4) and the M6b rule variant are read when a record carries them
(``ap_estimate_mm``, ``ap_bin``, ``m6b_rule_variant``); otherwise they are
reported as ``pending`` and nothing is inferred from them.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.report_figures import grouped_bars, tile_maps
from merxen.annotation.report_inputs import ReportInputs, SampleData
from merxen.annotation.report_items import (
    confident,
    names_array,
    new_item,
    soft_matrix,
    table_cells,
)
from merxen.annotation.report_metrics import tile_mean_map
from merxen.annotation.report_model import ItemWriter, ReportItem, ReportOptions, metric
from merxen.annotation.report_panel import Banner
from merxen.annotation.schema import Columns
from merxen.annotation.vocab import load_vocab

logger = logging.getLogger(__name__)

ASTRO_EPEN: Final = "30 Astro-Epen"
IMMUNE: Final = "34 Immune"
AP_SCOPE_MM: Final = (2.4, 10.4)
UNCOVERED_NOTE_AP_MM: Final = 7.5
# Plan §14 MO11: map_first results from anterior sections (AP < 6.0 mm) keep
# a banner until the first real anterior section passes MO11.
ANTERIOR_AP_MM: Final = 6.0
PENDING: Final = "pending_m6b"
UNCOVERED_SUBCLASSES: Final[tuple[str, ...]] = (
    "157 RN Spp1 Glut",
    "279 PSV Pax2 Gly-Gaba",
    "280 NLL-po Pax7 Gaba",
    "297 CU-ECU Pax2 Gly-Gaba",
)


def ap_fields(
    record: Mapping[str, Any] | None, gate: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return the M6b AP estimate, bin and rule variant, or ``pending``.

    Args:
        record: The MAP region record (``mouse_regions``).
        gate: The mouse gate record.

    Returns:
        ``ap_estimate_mm``, ``ap_bin``, ``ap_status``, ``m6b_rule_variant``,
        ``rule_variant`` (as recorded), ``rule_variant_status``,
        ``ap_in_scope`` (``None`` while pending) and ``warnings``.
    """
    sources = [dict(record or {}), dict(gate or {})]
    estimate = next(
        (
            s.get("ap_estimate_mm")
            for s in sources
            if s.get("ap_estimate_mm") is not None
        ),
        None,
    )
    ap_bin = next(
        (s.get("ap_bin") for s in sources if s.get("ap_bin") is not None), None
    )
    m6b_variant = next(
        (s.get("m6b_rule_variant") for s in sources if s.get("m6b_rule_variant")), None
    )
    variant = (record or {}).get("rule_variant") or (
        (record or {}).get("params") or {}
    ).get("rule_variant")
    warnings = []
    in_scope = None
    if estimate is not None:
        value = float(estimate)
        in_scope = AP_SCOPE_MM[0] <= value <= AP_SCOPE_MM[1]
        if not in_scope:
            warnings.append(
                f"AP estimate {value:.2f} mm is outside the validated coronal "
                f"scope {AP_SCOPE_MM[0]}-{AP_SCOPE_MM[1]} mm"
            )
    return {
        "ap_estimate_mm": estimate,
        "ap_bin": ap_bin,
        "ap_status": "measured" if estimate is not None else PENDING,
        "ap_in_scope": in_scope,
        "rule_variant": variant,
        "m6b_rule_variant": m6b_variant,
        "rule_variant_status": "selected" if m6b_variant else PENDING,
        "warnings": warnings,
    }


def _region_share_dir(
    record: Mapping[str, Any] | None, store: Path | None
) -> Path | None:
    share = (record or {}).get("region_share") or {}
    path = share.get("path")
    if path and Path(str(path)).is_dir():
        return Path(str(path))
    if store is not None and share.get("build_hash"):
        candidate = (
            store
            / str(share.get("reference_id") or "wmb_region_share")
            / str(share["build_hash"])
        )
        if candidate.is_dir():
            return candidate
    return None


def subclass_region_table(
    subclass: np.ndarray,
    region: np.ndarray,
    merfish: pd.DataFrame | None,
    present: list[str],
) -> pd.DataFrame:
    """Return each confident subclass's share per inferred region vs MERFISH.

    Args:
        subclass: Confident subclass per cell (``""`` otherwise).
        region: The cell's tile region (``""`` when unassigned).
        merfish: ``region_share.parquet`` rows (``level``, ``node_name``,
            ``region``, ``share``), if any.
        present: The section's present regions (MERFISH shares are
            renormalised over them).

    Returns:
        Rows ``subclass``, ``region``, ``n_cells``, ``share``,
        ``merfish_share`` (renormalised over present regions).
    """
    keep = (subclass != "") & (region != "")
    frame = pd.DataFrame({"subclass": subclass[keep], "region": region[keep]})
    if frame.empty:
        return pd.DataFrame(
            columns=["subclass", "region", "n_cells", "share", "merfish_share"]
        )
    counts = (
        frame.groupby(["subclass", "region"]).size().rename("n_cells").reset_index()
    )
    counts["share"] = counts["n_cells"] / counts.groupby("subclass")[
        "n_cells"
    ].transform("sum")
    counts["merfish_share"] = math.nan
    if merfish is not None and len(merfish):
        sub = merfish[
            (merfish["level"] == "subclass") & merfish["region"].isin(present)
        ]
        sub = sub.assign(
            renorm=sub["share"] / sub.groupby("node_name")["share"].transform("sum")
        )
        lookup = {
            (row.node_name, row.region): row.renorm
            for row in sub.itertuples(index=False)
        }
        counts["merfish_share"] = [
            lookup.get((a, b), 0.0 if a in set(sub["node_name"]) else math.nan)
            for a, b in zip(counts["subclass"], counts["region"], strict=True)
        ]
    return counts


def window_composition(bundle: Path | None, sections: list[str]) -> pd.DataFrame | None:
    """Return the class composition of the AP-matched MERFISH window (G4).

    Args:
        bundle: The ``wmb_region_share`` bundle.
        sections: The window's MERFISH sections (mouse gate record).

    Returns:
        Rows ``class``, ``merfish_share`` pooled over the sections, or ``None``.
    """
    if (
        bundle is None
        or not sections
        or not (bundle / "section_composition.parquet").is_file()
    ):
        return None
    frame = pd.read_parquet(bundle / "section_composition.parquet")
    frame = frame[(frame["level"] == "class") & frame["section"].isin(sections)]
    if frame.empty:
        return None
    pooled = frame.groupby("node_name")["n"].sum()
    return pd.DataFrame(
        {
            "class": pooled.index.astype(str),
            "merfish_share": (pooled / pooled.sum()).to_numpy(),
        }
    )


def allocated_class_shares(soft: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the class shares over the allocated soft mass and the unallocated share.

    MO4 and the mouse gate's G4 (``mouse_resolve.class_shares``) renormalise
    the residual away (plan §5.5), reporting the unallocated share beside.

    Args:
        soft: ``soft_matrix`` rows (classes, then the residual last).

    Returns:
        ``(shares, unallocated)``: per-class share of the allocated mass, and
        the residual's share of all soft mass.
    """
    allocated = np.asarray(soft, dtype=np.float64)[:, :-1]
    total = max(1e-12, float(allocated.sum()))
    residual = float(np.asarray(soft, dtype=np.float64)[:, -1].sum())
    return allocated.sum(axis=0) / total, residual / max(1e-12, total + residual)


def item_mouse_regions(
    inputs: ReportInputs, writer: ItemWriter, options: ReportOptions
) -> ReportItem:
    """Build item 10: mouse region plausibility (M6b fields optional)."""
    item = new_item(
        10,
        "mouse_regions",
        "Mouse region plausibility",
        "wrong region sets, harmful pruning, microglial spill-over read as microglia, "
        "low-count Astro-Epen sinks",
    )
    if inputs.species != "mouse":
        item.status = "not_applicable"
        item.notes.append("Mouse only.")
        return item
    region_rows = []
    drop_rows = []
    relabel_rows = []
    tile_frames = []
    spill_frames = []
    sub_region_frames = []
    comp_rows = []
    gate_rows = []
    gene_rows: list[dict[str, Any]] = []
    classes = list(load_vocab("wmb_class").names)
    for sample in inputs.ordered_samples():
        table = table_cells(sample)
        record = dict((sample.map_record or {}).get("mouse_regions") or {})
        step = dict(sample.summary.get("region_step") or {})
        gate = dict(
            sample.summary.get("mouse_gate")
            or (sample.summary.get("resolution") or {}).get("gate")
            or {}
        )
        ap = ap_fields(record, gate)
        for warning in ap["warnings"]:
            item.notes.append(f"{sample.sample_id}: {warning}")
        if ap["ap_status"] == PENDING:
            item.notes.append(
                f"{sample.sample_id}: AP estimate and bin pending (M6b); G4 uses "
                "the configured window"
            )
        estimate = ap["ap_estimate_mm"]
        if estimate is not None and float(estimate) < ANTERIOR_AP_MM:
            item.banners.append(
                Banner(
                    severity="warning",
                    sample_id=sample.sample_id,
                    code="anterior_section_mo11",
                    text=(
                        f"anterior coronal section (AP estimate {float(estimate):.2f} "
                        f"mm < {ANTERIOR_AP_MM} mm): MO11 not yet passed; map_first "
                        "results from anterior sections keep this banner until the "
                        "first real anterior section passes MO11 (plan §14)"
                    ),
                )
            )
        elif estimate is None:
            item.notes.append(
                f"{sample.sample_id}: AP estimate pending (M6b): whether the section "
                f"is anterior (AP < {ANTERIOR_AP_MM} mm, the MO11 banner) is not known"
            )
        if estimate is None or float(estimate) >= UNCOVERED_NOTE_AP_MM:
            item.notes.append(
                f"{sample.sample_id}: the WMB subclasses "
                f"{', '.join(UNCOVERED_SUBCLASSES)} have no 10Xv3 reference cells "
                "and cannot be assigned"
                + (
                    " (the AP estimate is pending; they can occur at AP >= 7.5 mm)"
                    if estimate is None
                    else " (AP >= 7.5 mm)"
                )
            )
        region_rows.append(
            {
                "sample_id": sample.sample_id,
                "platform": sample.platform,
                "status": record.get("status") or step.get("status"),
                "source": record.get("source") or step.get("source"),
                "requested": record.get("requested"),
                "inferred_regions": ";".join(record.get("inferred_regions") or []),
                "present_regions": ";".join(
                    record.get("present_regions") or step.get("present_regions") or []
                ),
                "explicit": (record.get("source") or step.get("source"))
                not in (None, "auto"),
                "n_assigned_tiles": record.get("n_assigned_tiles"),
                "n_confident_neurons": record.get("n_confident_neurons"),
                "n_nodes_dropped": step.get(
                    "n_nodes_dropped", len(record.get("nodes_to_drop") or [])
                ),
                "n_cells_region_dropped": step.get("n_cells_region_dropped"),
                "tile_counts": json.dumps(
                    record.get("tile_counts") or {}, sort_keys=True
                ),
                "largest_component": json.dumps(
                    record.get("largest_component") or {}, sort_keys=True
                ),
                "reasons": "; ".join(str(v) for v in record.get("reasons") or []),
                **{key: value for key, value in ap.items() if key != "warnings"},
            }
        )
        item.metrics.append(
            metric(
                "MO3",
                "inferred_regions",
                ";".join(
                    record.get("inferred_regions") or step.get("present_regions") or []
                )
                or None,
                definition=(
                    "regions inferred from confident neurons' MERFISH homes "
                    "(E7), or the explicit set"
                ),
                source="map_manifest",
                sample_id=sample.sample_id,
                platform=sample.platform,
                note=f"source {record.get('source') or step.get('source')}",
            )
        )
        for key in ("ap_estimate_mm", "m6b_rule_variant"):
            item.metrics.append(
                metric(
                    "MO11" if key == "ap_estimate_mm" else "MO10",
                    key,
                    ap[key],
                    definition="M6b field (optional until M6b is merged)",
                    source="map_manifest",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    status="measured" if ap[key] is not None else "not_available",
                    note=PENDING if ap[key] is None else "",
                )
            )
        dropped_names = set()
        for node in record.get("nodes_to_drop") or []:
            dropped_names.add(str(node.get("name")))
            drop_rows.append(
                {"sample_id": sample.sample_id, **{k: v for k, v in node.items()}}
            )
        # Relabelled cells and mechanics.
        if {"mmc_wmb_unpruned_class_name", "mmc_wmb_class_name"} <= set(table.columns):
            changed = (
                table["region_pruned_changed"].astype(bool).to_numpy()
                if "region_pruned_changed" in table.columns
                else np.zeros(len(table), bool)
            )
            unpruned_class = names_array(table, "mmc_wmb_unpruned_class_name")
            pruned_class = names_array(table, "mmc_wmb_class_name")
            unpruned_sub = names_array(table, "mmc_wmb_unpruned_subclass_name")
            pruned_sub = names_array(table, "mmc_wmb_subclass_name")
            outside_class = (~changed) & (unpruned_class != pruned_class)
            still_dropped = np.isin(pruned_sub, list(dropped_names)) | np.isin(
                pruned_class, list(dropped_names)
            )
            frame = pd.DataFrame(
                {
                    "unpruned_class": unpruned_class[changed],
                    "pruned_class": pruned_class[changed],
                    "unpruned_subclass": unpruned_sub[changed],
                    "pruned_subclass": pruned_sub[changed],
                }
            )
            if len(frame):
                moved = (
                    frame.groupby(
                        [
                            "unpruned_class",
                            "pruned_class",
                            "unpruned_subclass",
                            "pruned_subclass",
                        ]
                    )
                    .size()
                    .rename("n_cells")
                    .reset_index()
                )
                moved.insert(0, "sample_id", sample.sample_id)
                relabel_rows.append(moved)
            for name, value, definition in (
                (
                    "n_relabelled",
                    int(changed.sum()),
                    "cells whose unpruned subclass was dropped (re-mapped)",
                ),
                (
                    "class_changes_outside_dropped",
                    _share(outside_class, len(table)),
                    "share of table cells not re-mapped whose class differs from the "
                    "unpruned call",
                ),
                (
                    "pruned_calls_in_dropped_nodes",
                    int(still_dropped.sum()),
                    "pruned calls still naming a dropped node",
                ),
            ):
                item.metrics.append(
                    metric(
                        "MO2",
                        name,
                        value,
                        definition=definition,
                        source="label_table",
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        n=len(table),
                    )
                )
        # Tile map and spill-over map.
        regions = sample.mouse_regions
        tile_um = float((record.get("params") or {}).get("tile_um") or 150.0)
        if regions is not None and {"tile_i", "tile_j", "tile_region"} <= set(
            regions.columns
        ):
            tiles = (
                regions[["tile_i", "tile_j", "tile_region"]]
                .dropna()
                .drop_duplicates(["tile_i", "tile_j"])
            )
            tiles = tiles.assign(
                panel=sample.sample_id,
                x_um=(tiles["tile_i"].astype(float) + 0.5) * tile_um,
                y_um=(tiles["tile_j"].astype(float) + 0.5) * tile_um,
                value=tiles["tile_region"].astype(str),
            )
            tile_frames.append(tiles)
        spill_column = Columns.FLAG_MICROGLIAL_SPILLOVER
        if spill_column in table.columns:
            spill = (
                pd.to_numeric(table[spill_column].astype("object"), errors="coerce")
                .astype(float)
                .to_numpy()
            )
            xy = sample.table_xy()
            if (
                xy is None
                and regions is not None
                and {"tile_i", "tile_j"} <= set(regions.columns)
            ):
                joined = regions.set_index("cell_id").reindex(
                    table[Columns.CELL_ID].astype(str)
                )
                xy = np.column_stack(
                    [
                        (joined["tile_i"].to_numpy(float) + 0.5) * tile_um,
                        (joined["tile_j"].to_numpy(float) + 0.5) * tile_um,
                    ]
                )
            if xy is not None and np.isfinite(spill).any():
                spill_map = tile_mean_map(xy, spill, tile_um=max(tile_um, 300.0))
                spill_map.insert(0, "panel", sample.sample_id)
                spill_frames.append(spill_map)
            rate = float(np.nanmean(spill)) if np.isfinite(spill).any() else math.nan
            genes = spillover_genes(sample, inputs.gene_lookup)
            gene_rows.extend(
                {"sample_id": sample.sample_id, "set": name, "gene": gene}
                for name, members in (
                    ("microglial_spillover", genes),
                    ("astrocyte_fpr", _astro_genes(sample)),
                )
                for gene in members
            )
            item.metrics.append(
                metric(
                    "MO5",
                    "spillover_prevalence",
                    rate,
                    definition=(
                        "flag_microglial_spillover rate over table cells with "
                        "a non-null flag (prevalence, never a microglia count)"
                    ),
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    n=int(np.isfinite(spill).sum()),
                    note="derived gene set: " + ", ".join(genes) if genes else "",
                )
            )
        # Subclass x region vs MERFISH share.
        region_of_cell = np.full(len(table), "", dtype=object)
        if regions is not None and "tile_region" in regions.columns:
            joined = regions.set_index("cell_id").reindex(
                table[Columns.CELL_ID].astype(str)
            )
            region_of_cell = (
                joined["tile_region"].astype(object).fillna("").astype(str).to_numpy()
            )
        share_dir = _region_share_dir(record, inputs.sources.store_root)
        merfish = (
            pd.read_parquet(share_dir / "region_share.parquet")
            if share_dir is not None and (share_dir / "region_share.parquet").is_file()
            else None
        )
        subclass = np.where(
            confident(table, "subclass"),
            names_array(table, Columns.level("subclass", "name")),
            "",
        )
        present = list(
            record.get("present_regions") or step.get("present_regions") or []
        )
        sub_regions = subclass_region_table(subclass, region_of_cell, merfish, present)
        sub_regions.insert(0, "sample_id", sample.sample_id)
        sub_region_frames.append(sub_regions)
        if len(sub_regions) and sub_regions["merfish_share"].notna().any():
            weights = sub_regions["n_cells"].to_numpy(float)
            gaps = np.abs(
                sub_regions["share"].to_numpy(float)
                - sub_regions["merfish_share"].to_numpy(float)
            )
            ok = np.isfinite(gaps)
            item.metrics.append(
                metric(
                    "report",
                    "subclass_region_share_mean_abs_gap",
                    float(np.average(gaps[ok], weights=weights[ok]))
                    if ok.any()
                    else None,
                    definition=(
                        "cell-weighted mean |observed - MERFISH| share of "
                        "confident subclasses per present region"
                    ),
                    source="mouse_regions+wmb_region_share",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    n=int(weights[ok].sum()),
                )
            )
        # Composition vs window, Astro-Epen with and without low-count cells.
        soft = soft_matrix(table, "mouse")
        window = window_composition(
            share_dir, list((gate.get("window") or {}).get("sections") or [])
        )
        if soft is not None:
            # MO4 / G4 shares are over the allocated mass (the residual
            # renormalised away, as mouse_resolve.class_shares and §5.5 do);
            # the unallocated share is recorded beside them.
            allocated = soft[:, :-1]
            shares, unallocated = allocated_class_shares(soft)
            item.metrics.append(
                metric(
                    "MO4",
                    "unallocated_soft_share",
                    unallocated,
                    definition=(
                        "soft mass on no WMB class (1 - sum of soft_class_*) / all "
                        "soft mass, table cells"
                    ),
                    source="label_table",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    n=len(table),
                )
            )
            for index, name in enumerate(classes):
                comp_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "class": name,
                        "series": "soft (table cells)",
                        "share": float(shares[index]),
                    }
                )
            if window is not None:
                for row in window.itertuples(index=False):
                    comp_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class": row[0],
                            "series": "MERFISH window",
                            "share": float(row[1]),
                        }
                    )
            astro = classes.index(ASTRO_EPEN) if ASTRO_EPEN in classes else None
            if astro is not None:
                low = (
                    table[Columns.FLAG_ASTRO_LOWCOUNT].astype(bool).to_numpy()
                    if Columns.FLAG_ASTRO_LOWCOUNT in table.columns
                    else np.zeros(len(table), bool)
                )
                strict = allocated[~low]
                for name, value in (
                    ("astro_epen_soft_share", float(shares[astro])),
                    (
                        "astro_epen_soft_share_strict",
                        float(strict[:, astro].sum() / max(1e-12, strict.sum()))
                        if len(strict)
                        else math.nan,
                    ),
                ):
                    item.metrics.append(
                        metric(
                            "MO4" if name == "astro_epen_soft_share" else "report",
                            name,
                            value,
                            definition="soft Astro-Epen share of the allocated soft "
                            "mass of table cells"
                            + (
                                " without flag_astro_lowcount cells"
                                if name.endswith("strict")
                                else ""
                            ),
                            source="label_table",
                            sample_id=sample.sample_id,
                            platform=sample.platform,
                            n=len(table)
                            if not name.endswith("strict")
                            else int((~low).sum()),
                        )
                    )
            if IMMUNE in classes:
                immune = classes.index(IMMUNE)
                item.metrics.append(
                    metric(
                        "MO4",
                        "immune_soft_share",
                        float(shares[immune]),
                        definition=(
                            "soft Immune share of the allocated soft mass of table "
                            "cells"
                        ),
                        source="label_table",
                        sample_id=sample.sample_id,
                        platform=sample.platform,
                        n=len(table),
                    )
                )
        coherence = sample.summary.get("region_coherence") or {}
        item.metrics.append(
            metric(
                "MO6",
                "f1_region_incoherent_rate",
                coherence.get("rate"),
                definition="flag_region_incoherent rate after pruning (E7 F1)",
                source="resolve_summary",
                sample_id=sample.sample_id,
                platform=sample.platform,
                note=str(coherence.get("null_reason") or ""),
            )
        )
        signals = dict(gate.get("signals") or {})
        statuses = dict(gate.get("signal_status") or {})
        for key, value in sorted(signals.items()):
            gate_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "signal": key,
                    "value": value,
                    "status": statuses.get(key.split("_")[0]),
                }
            )
            item.metrics.append(
                metric(
                    "MO4"
                    if key.startswith("g4_")
                    else ("MO1" if key.startswith("g2_") else "MO7"),
                    key,
                    value,
                    definition=f"mouse gate signal {key.split('_')[0].upper()} (§7.6)",
                    source="resolve_summary",
                    sample_id=sample.sample_id,
                    platform=sample.platform,
                    note=f"status {statuses.get(key.split('_')[0])}",
                )
            )
        gate_rows.append(
            {
                "sample_id": sample.sample_id,
                "signal": "level",
                "value": gate.get("level"),
                "status": "warning" if gate.get("warning") else "ok",
            }
        )
        item.metrics.append(
            metric(
                "MO7",
                "mouse_gate_level",
                gate.get("level"),
                definition="mouse gate level (G1-G5)",
                source="resolve_summary",
                sample_id=sample.sample_id,
                platform=sample.platform,
                note="; ".join(
                    str(v)
                    for v in list(gate.get("level_reasons") or [])
                    + list(gate.get("warning_reasons") or [])
                ),
            )
        )
    regions_frame = pd.DataFrame(region_rows)
    writer.table(item, "regions", regions_frame)
    writer.table(item, "drop_list", pd.DataFrame(drop_rows))
    writer.table(
        item,
        "relabelled",
        pd.concat(relabel_rows, ignore_index=True)
        if relabel_rows
        else pd.DataFrame(
            columns=[
                "sample_id",
                "unpruned_class",
                "pruned_class",
                "unpruned_subclass",
                "pruned_subclass",
                "n_cells",
            ]
        ),
    )
    sub_regions_all = (
        pd.concat(sub_region_frames, ignore_index=True)
        if sub_region_frames
        else pd.DataFrame()
    )
    writer.table(item, "subclass_region_shares", sub_regions_all)
    tiles_all = (
        pd.concat(tile_frames, ignore_index=True) if tile_frames else pd.DataFrame()
    )
    writer.figure(
        item,
        "tile_map",
        lambda: tile_maps(
            tiles_all, panel="panel", categorical=True, title="Inferred region per tile"
        ),
        tiles_all,
        "Majority MERFISH home region of the confident neurons per tile (E7 region "
        "inference).",
    )
    spill_all = (
        pd.concat(spill_frames, ignore_index=True) if spill_frames else pd.DataFrame()
    )
    writer.figure(
        item,
        "spillover_map",
        lambda: tile_maps(
            spill_all,
            panel="panel",
            colour_label="spill-over rate",
            title="Microglial spill-over prevalence (rate per tile, not a count)",
        ),
        spill_all,
        "Rate of flag_microglial_spillover per tile: spill-over prevalence, never a "
        "microglia count.",
    )
    comp = pd.DataFrame(comp_rows)
    if len(comp):
        top = comp.groupby("class")["share"].max()
        comp = comp[comp["class"].isin(top[top >= 0.005].index)]
    writer.figure(
        item,
        "composition_vs_window",
        lambda: grouped_bars(
            comp,
            category="class",
            group="series",
            value="share",
            ylabel="share",
            title="Soft class composition vs the AP-matched MERFISH window",
        ),
        comp,
        "Soft WMB class composition of the table cells beside the pooled MERFISH "
        "window (G4).",
    )
    writer.table(item, "gate", pd.DataFrame(gate_rows))
    writer.table(
        item,
        "spillover_gene_sets",
        pd.DataFrame(gene_rows, columns=["sample_id", "set", "gene"]),
    )
    item.summary = regions_frame
    return item


def _symbols(sample: SampleData, lookup: Mapping[str, str]) -> dict[str, str]:
    """Return gene id -> symbol (clustered H5AD var, else the panel files)."""
    symbols: dict[str, str] = {}
    if sample.clustered_path is not None and Path(sample.clustered_path).is_file():
        from merxen.annotation.report_inputs import (
            read_clustered_table,
            resolved_gene_ids,
        )

        table = read_clustered_table(sample.clustered_path, with_counts=False)
        ids = resolved_gene_ids(table.gene_ids, lookup)
        symbols.update(zip(ids, table.gene_symbols, strict=True))
    return symbols


def spillover_genes(
    sample: SampleData, lookup: Mapping[str, str] | None = None
) -> list[str]:
    """Return the panel's derived microglial spill-over genes (§8.6), as symbols.

    Args:
        sample: The sample (its provenance records the gene set).
        lookup: Case-folded symbol -> Ensembl id of the panel files.

    Returns:
        Gene symbols (IDs where no symbol is known), in recorded order.
    """
    sets = (sample.manifest.get("flags") or {}).get("gene_sets") or {}
    ids = [str(gene) for gene in sets.get("microglial_spillover_genes") or []]
    symbols = _symbols(sample, lookup or {}) if ids else {}
    return [symbols.get(gene, gene) for gene in ids]


def _astro_genes(sample: SampleData) -> list[str]:
    checks = sample.summary.get("spillover_checks") or {}
    return [
        str(gene)
        for gene in (checks.get("astrocyte_fpr") or {}).get("astro_genes") or []
    ]


def _share(mask: np.ndarray, denominator: int) -> float:
    return float(np.count_nonzero(mask) / denominator) if denominator else math.nan


__all__ = [
    "AP_SCOPE_MM",
    "PENDING",
    "UNCOVERED_SUBCLASSES",
    "ap_fields",
    "item_mouse_regions",
    "subclass_region_table",
    "window_composition",
]
