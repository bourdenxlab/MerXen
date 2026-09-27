#!/usr/bin/env python
"""E8 segmentation comparison (plan §12 M3 item 2; feeds OD-B6 and OD-B7).

Compares the four published segmentations (original_seg, proseg_mask,
proseg_hybrid, reseg) of one human pair (both platforms) and one mouse
section (MERSCOPE), each mapped by ``merxen annotate``:

- **confident cells per mm²**: human confident lineage / broad /
  supercluster calls under the v1 shadow rules (SEA-AD second vote,
  floors, gate); mouse confident class (bp >= 0.9, >= 20 counts) and
  subclass (bp >= 0.8, >= 50 counts; no region pruning, M6). The area is the
  tissue covered by table cells (100 µm bins with >= 3 cells); densities are
  also given against the median area over the segmentations, so that their
  ratios are ratios of confident counts over the same tissue;
- **foreign-marker fraction** of confident cells: the share of a cell's
  counts on its broad class's negative genes (the bundle's
  ``negative_genes.parquet``; the §5.6 contamination score) and, for human,
  on the E1 canonical markers of the other broad classes;
- **platform soft JSD** (human): MERSCOPE vs Xenium soft / argmax /
  confident broad JSD with block-bootstrap CIs; mouse: WMB class
  composition JSD vs proseg_hybrid (no second platform);
- the gate (A, level), WHB-SEA agreement and the implausible share.

OD-B6 rule (plan): switch P1212's default segmentation if a segmentation
has >= 25% more confident broad cells per mm² on MERSCOPE than
proseg_hybrid. Writes ``e8_human.csv``, ``e8_human_jsd.csv`` and
``e8_mouse.csv`` to ``--out``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import MmcBundle
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import load_map_manifest
from merxen.annotation.shadow import (
    AGREEMENT_MIN_COUNTS,
    argmax_broad_names,
    block_bootstrap_jsd,
    foreign_marker_fraction,
    jensen_shannon_distance,
    label_agreement,
    mouse_confidence,
    negative_counts,
    negative_gene_mask,
    occupied_area_mm2,
    one_hot_broad_matrix,
    published_queries,
    soft_matrix_from_provisional,
    tile_codes,
    tile_sums,
)
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, MOUSE_BROAD_CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import PLATFORMS, _clustered_path, load_sample  # noqa: E402

logger = logging.getLogger("shadow_e8")

SEGMENTATIONS = ("original_seg", "proseg_mask", "proseg_hybrid", "reseg")
REFERENCE_SEG = "proseg_hybrid"
OD_B6_RATIO = 1.25
HUMAN_FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/"
    "WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
)
MOUSE_FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/20241115/"
    "gene.csv"
)


def _parse_runs(values: list[str]) -> dict[str, Path]:
    runs = {}
    for value in values:
        seg, _, path = value.partition("=")
        runs[seg] = Path(path)
    return runs


def _negative_score(
    counts: Any,
    gene_ids: list[str],
    labels: np.ndarray,
    totals: np.ndarray,
    negatives: pd.DataFrame,
    classes: tuple[str, ...],
) -> np.ndarray:
    mask = negative_gene_mask(negatives, gene_ids, classes)
    values = negative_counts(counts, labels, mask, classes)
    return values / np.maximum(totals, 1.0)


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def human_rows(
    args: argparse.Namespace,
    runs: dict[str, Path],
    n_segmented: dict[Any, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """E8 rows of the human pair."""
    pair = args.pair
    rows: list[dict[str, Any]] = []
    jsd_rows: list[dict[str, Any]] = []
    for seg, run_dir in runs.items():
        manifest = load_map_manifest(run_dir / "map_manifest.json")
        panel = AnnotationPanel.model_validate_json(
            (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
        )
        queries = published_queries(
            {p: _clustered_path(args.results_root, pair, seg, p) for p in PLATFORMS},
            panel,
            species="human",
            gene_id_fallback_csv=HUMAN_FALLBACK,
        )
        matrices: dict[str, dict[str, np.ndarray]] = {}
        codes: dict[str, np.ndarray] = {}
        for platform in PLATFORMS:
            key = (pair, seg, platform)
            sample = load_sample(
                run_dir, args.results_root, pair, seg, platform, n_segmented.get(key)
            )
            bundle = MmcBundle.from_dir(
                manifest.samples[sample.sample_id]
                .runs["whb_frontal_supc_clus"]
                .bundle_path
            )
            query = queries[platform]
            position = pd.Index(query.cell_ids).get_indexer(sample.labels.index)
            counts = query.counts[position]
            labels = sample.labels
            totals = labels["total_counts"].to_numpy(np.float64)
            v1 = sample.rules["seaad"]
            confident = np.where(v1.broad_confident, v1.broad_name, None)
            argmax = argmax_broad_names(labels)
            argmax_in = np.where(np.isin(argmax, HUMAN_BROAD_CLASSES), argmax, None)
            negatives = pd.read_parquet(bundle.path / "negative_genes.parquet")
            neg_conf = _negative_score(
                counts,
                list(query.gene_ids),
                confident,
                totals,
                negatives,
                HUMAN_BROAD_CLASSES,
            )
            neg_all = _negative_score(
                counts,
                list(query.gene_ids),
                argmax_in,
                totals,
                negatives,
                HUMAN_BROAD_CLASSES,
            )
            foreign_conf = foreign_marker_fraction(sample.scores, confident)
            foreign_all = foreign_marker_fraction(sample.scores, argmax_in)
            agreement, _ = label_agreement(
                argmax, sample.sea["broad"], totals >= AGREEMENT_MIN_COUNTS
            )
            area = occupied_area_mm2(sample.xy)
            rows.append(
                {
                    "pair": pair,
                    "segmentation": seg,
                    "platform": platform,
                    "n_segmented": n_segmented.get(key),
                    "n_table_cells": len(labels),
                    "median_counts": float(np.median(totals)),
                    "gate_A": v1.gate.frac_ge30,
                    "gate_level": v1.gate.level,
                    "gate_warning": v1.gate.warning,
                    "area_mm2": area,
                    "n_lineage_confident": int(v1.lineage_confident.sum()),
                    "n_broad_confident": int(v1.broad_confident.sum()),
                    "n_supercluster_confident": int(v1.supercluster_confident.sum()),
                    "broad_cov_table": float(v1.broad_confident.mean()),
                    "broad_cov_segmented": v1.gate.broad_coverage_segmented,
                    "table_cells_per_mm2": len(labels) / area if area else np.nan,
                    "broad_confident_per_mm2": v1.broad_confident.sum() / area
                    if area
                    else np.nan,
                    "supercluster_confident_per_mm2": v1.supercluster_confident.sum()
                    / area
                    if area
                    else np.nan,
                    "negative_fraction_confident_mean": _mean(neg_conf),
                    "negative_fraction_confident_median": _median(neg_conf),
                    "negative_fraction_all_mean": _mean(neg_all),
                    "foreign_marker_fraction_confident_mean": _mean(foreign_conf),
                    "foreign_marker_fraction_all_mean": _mean(foreign_all),
                    "whb_sea_agree_ge20": agreement,
                    "implausible_share": float(v1.implausible.mean()),
                    "n_missing_panel_genes": len(query.missing_gene_ids),
                }
            )
            matrices[platform] = {
                "soft": soft_matrix_from_provisional(labels),
                "argmax": one_hot_broad_matrix(argmax),
                "confident": one_hot_broad_matrix(
                    v1.broad_name, include=v1.broad_confident
                ),
            }
            codes[platform] = tile_codes(sample.xy)
        for kind in ("soft", "argmax", "confident"):
            result = block_bootstrap_jsd(
                tile_sums(matrices["MERSCOPE"][kind], codes["MERSCOPE"]),
                tile_sums(matrices["XENIUM"][kind], codes["XENIUM"]),
                n_reps=args.n_bootstrap,
            )
            jsd_rows.append(
                {
                    "pair": pair,
                    "segmentation": seg,
                    "kind": kind,
                    "jsd": result.jsd,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                }
            )
        logger.info("%s %s done", pair, seg)
    frame = pd.DataFrame(rows)
    for platform in PLATFORMS:
        subset = frame["platform"] == platform
        common = float(frame.loc[subset, "area_mm2"].median())
        frame.loc[subset, "common_area_mm2"] = common
        for level in ("broad", "supercluster"):
            frame.loc[subset, f"{level}_confident_per_common_mm2"] = (
                frame.loc[subset, f"n_{level}_confident"] / common
            )
        reference = frame.loc[
            subset & (frame["segmentation"] == REFERENCE_SEG), "n_broad_confident"
        ]
        if len(reference):
            frame.loc[subset, "broad_confident_ratio_vs_proseg_hybrid"] = frame.loc[
                subset, "n_broad_confident"
            ] / float(reference.iloc[0])
    frame["od_b6_switch_candidate"] = (
        frame["broad_confident_ratio_vs_proseg_hybrid"] >= OD_B6_RATIO
    )
    return frame.to_dict("records"), jsd_rows


def mouse_rows(args: argparse.Namespace, runs: dict[str, Path]) -> list[dict[str, Any]]:
    """E8 rows of the mouse section."""
    qc = pd.read_csv(args.evidence_root / "research/lowcount/qc_summary.csv")
    segmented = {
        str(r.seg): int(r.n_cells)
        for r in qc.itertuples(index=False)
        if str(r.sample) == args.mouse_sample
    }
    rows: list[dict[str, Any]] = []
    class_shares: dict[str, pd.Series] = {}
    for seg, run_dir in runs.items():
        manifest = load_map_manifest(run_dir / "map_manifest.json")
        sample_id = next(iter(manifest.samples))
        panel = AnnotationPanel.model_validate_json(
            (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
        )
        clustered = (
            args.mouse_results_root
            / seg
            / "clustering_squidpy"
            / "clustering_squidpy_out"
            / "merscope"
            / f"{sample_id}_clustered.h5ad"
        )
        query = published_queries(
            {"MERSCOPE": clustered},
            panel,
            species="mouse",
            gene_id_fallback_csv=MOUSE_FALLBACK,
        )["MERSCOPE"]
        labels = pd.read_parquet(
            run_dir / "merscope" / f"{sample_id}_ct_provisional.parquet"
        ).set_index("cell_id")
        labels = labels[labels["in_table"].to_numpy(bool)]
        position = pd.Index(query.cell_ids).get_indexer(labels.index)
        counts = query.counts[position]
        bundle = MmcBundle.from_dir(
            next(iter(manifest.samples[sample_id].runs.values())).bundle_path
        )
        with h5py.File(clustered, "r") as handle:
            xy = np.asarray(handle["obsm"]["spatial"][()], dtype=np.float64)[:, :2]
        area = occupied_area_mm2(xy)
        confidence = mouse_confidence(labels)
        totals = labels["total_counts"].to_numpy(np.float64)
        broad = labels["ct_broad_name"].astype(object).to_numpy()
        in_classes = np.isin(broad, MOUSE_BROAD_CLASSES)
        confident_broad = np.where(confidence.class_confident & in_classes, broad, None)
        negatives = pd.read_parquet(bundle.path / "negative_genes.parquet")
        neg_conf = _negative_score(
            counts,
            list(query.gene_ids),
            confident_broad,
            totals,
            negatives,
            MOUSE_BROAD_CLASSES,
        )
        classes = labels["mmc_wmb_class_name"].astype(str)
        class_shares[seg] = classes.value_counts(normalize=True)
        confident_classes = classes[confidence.class_confident]
        rows.append(
            {
                "sample": sample_id,
                "segmentation": seg,
                "n_segmented": segmented.get(seg),
                "n_table_cells": len(labels),
                "median_counts": float(np.median(totals)),
                "area_mm2": area,
                "n_class_confident": int(confidence.class_confident.sum()),
                "n_subclass_confident": int(confidence.subclass_confident.sum()),
                "class_cov_table": float(confidence.class_confident.mean()),
                "subclass_cov_table": float(confidence.subclass_confident.mean()),
                "class_confident_per_mm2": confidence.class_confident.sum() / area,
                "subclass_confident_per_mm2": confidence.subclass_confident.sum()
                / area,
                "negative_fraction_confident_mean": _mean(neg_conf),
                "negative_fraction_confident_median": _median(neg_conf),
                "immune_share_confident": float(
                    (confident_classes == "34 Immune").mean()
                ),
                "astro_epen_share_confident": float(
                    (confident_classes == "30 Astro-Epen").mean()
                ),
                "oligo_share_confident": float(
                    (confident_classes == "31 OPC-Oligo").mean()
                ),
                "n_microglia_confident": int((confident_classes == "34 Immune").sum()),
            }
        )
        logger.info("%s %s done", sample_id, seg)
    frame = pd.DataFrame(rows)
    common = float(frame["area_mm2"].median())
    frame["common_area_mm2"] = common
    for level in ("class", "subclass"):
        frame[f"{level}_confident_per_common_mm2"] = (
            frame[f"n_{level}_confident"] / common
        )
    reference = frame.loc[frame["segmentation"] == REFERENCE_SEG]
    if len(reference):
        for level in ("class", "subclass"):
            frame[f"{level}_confident_ratio_vs_proseg_hybrid"] = frame[
                f"n_{level}_confident"
            ] / float(reference[f"n_{level}_confident"].iloc[0])
    if REFERENCE_SEG in class_shares:
        union = sorted(set().union(*[set(s.index) for s in class_shares.values()]))
        base = class_shares[REFERENCE_SEG].reindex(union).fillna(0.0).to_numpy()
        frame["class_composition_jsd_vs_proseg_hybrid"] = [
            float(
                jensen_shannon_distance(
                    class_shares[seg].reindex(union).fillna(0.0).to_numpy(),
                    base,
                    columns=range(len(union)),
                )
            )
            for seg in frame["segmentation"]
        ]
    return frame.to_dict("records")


def main(argv: list[str] | None = None) -> int:
    """Compute the E8 comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pair", default="P1212")
    parser.add_argument("--human-run", action="append", default=[], metavar="SEG=DIR")
    parser.add_argument("--mouse-run", action="append", default=[], metavar="SEG=DIR")
    parser.add_argument("--mouse-sample", default="ag7")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--mouse-results-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    qc = pd.read_csv(args.evidence_root / "research/lowcount/qc_summary.csv")
    n_segmented = {
        (str(r.sample), str(r.seg), str(r.platform).upper()): int(r.n_cells)
        for r in qc.itertuples(index=False)
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.human_run:
        human, jsd = human_rows(args, _parse_runs(args.human_run), n_segmented)
        pd.DataFrame(human).to_csv(args.out / "e8_human.csv", index=False)
        pd.DataFrame(jsd).to_csv(args.out / "e8_human_jsd.csv", index=False)
    if args.mouse_run:
        mouse = mouse_rows(args, _parse_runs(args.mouse_run))
        pd.DataFrame(mouse).to_csv(args.out / "e8_mouse.csv", index=False)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
