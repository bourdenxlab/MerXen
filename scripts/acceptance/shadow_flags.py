#!/usr/bin/env python
"""Realised contamination and diffuse-profile flag rates (plan §5.6, H16; M3 item 5).

A prototype of the §5.6 flags on the MAP outputs, per class x platform
(production flags are M4):

- **Contamination:** ``contamination_score`` = counts on the assigned broad
  class's negative genes (``negative_genes.parquet`` of the primary bundle:
  human, detected in < 1% of the class in both WHB frontal and SEA-AD,
  minus state genes; mouse, the WMB panel precompute minus the mouse state
  list) / total counts. The null is a beta-binomial per (class, dataset)
  fitted on the class's confident cells in the top depth quartile; a cell is
  flagged at tail p < 0.01 with >= 3 negative counts. Strata with a realised
  rate > 15% are uninformative.
- **Diffuse profile:** ``n_genes`` (distinct query genes detected) above the
  q95 of distinct genes in 200 multinomial draws at the cell's query depth
  from the class profile (``profiles.parquet``), interpolated on a depth
  grid. The primary profile is the broad class's (cell-weighted over its
  nodes); the assigned node's profile (WHB supercluster, WMB class) is a
  sensitivity. Strata > 30% are uninformative (§4.3); H16's > 15% is
  reported too.

Cells are scored twice: confident broad calls (human: the v1 shadow rules;
mouse: class bp >= 0.9 and >= 20 counts) and every table cell's argmax
class. Writes ``flag_rates.csv``, ``flag_nulls.csv``, ``flag_scores.csv``
(per-class score quantiles) and ``flag_depth.csv`` (rate by depth bin) to
``--out``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import MmcBundle
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import load_map_manifest
from merxen.annotation.shadow import (
    CONTAMINATION_MAX_INFORMATIVE,
    DIFFUSE_MAX_INFORMATIVE,
    argmax_broad_names,
    class_profiles,
    contamination_flags,
    expected_genes_quantile,
    mouse_confidence,
    negative_counts,
    negative_gene_mask,
    published_queries,
    realised_rates,
)
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, MOUSE_BROAD_CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import (  # noqa: E402
    HELD_OUT_PAIRS,
    PAIRS,
    PLATFORMS,
    _clustered_path,
    load_sample,
)

logger = logging.getLogger("shadow_flags")

HUMAN_FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/"
    "WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
)
MOUSE_FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/20241115/"
    "gene.csv"
)
H16_MAX_INFORMATIVE = 0.15
DEPTH_BINS = (10, 30, 60, 120, 250, 500, 1_000_000)


@dataclass
class FlagInputs:
    """One sample's cells, labels and bundle files for the flags."""

    dataset: str
    species: str
    platform: str
    segmentation: str
    held_out: bool
    counts: Any
    gene_ids: list[str]
    total_counts: np.ndarray
    confident_class: np.ndarray
    argmax_class: np.ndarray
    node: np.ndarray
    negatives: pd.DataFrame
    profiles: pd.DataFrame
    vocab: pd.DataFrame
    node_level: str
    classes: tuple[str, ...]


def _vocab_map(vocab: pd.DataFrame, level: str, column: str) -> dict[str, str]:
    rows = vocab[vocab["level"].astype(str) == level]
    return dict(
        zip(rows["node_name"].astype(str), rows[column].astype(str), strict=True)
    )


def score(inputs: FlagInputs, *, seed: int) -> list[pd.DataFrame]:
    """Flag rates, nulls, score quantiles and depth profile of one sample."""
    mask = negative_gene_mask(inputs.negatives, inputs.gene_ids, inputs.classes)
    query_depth = np.asarray(inputs.counts.sum(axis=1)).ravel()
    n_genes = np.diff(inputs.counts.tocsr().indptr)
    broad_profiles = class_profiles(
        inputs.profiles,
        inputs.node_level,
        inputs.gene_ids,
        _vocab_map(inputs.vocab, inputs.node_level, "broad_class"),
        inputs.classes,
    )
    node_names = sorted({str(n) for n in inputs.node if n is not None})
    node_profiles = class_profiles(
        inputs.profiles,
        inputs.node_level,
        inputs.gene_ids,
        {name: name for name in node_names},
        node_names,
    )
    base = {
        "dataset": inputs.dataset,
        "species": inputs.species,
        "platform": inputs.platform,
        "segmentation": inputs.segmentation,
        "held_out": inputs.held_out,
    }
    rates, nulls, scores, depth_rows = [], [], [], []
    for class_set, labels in (
        ("confident", inputs.confident_class),
        ("argmax", inputs.argmax_class),
    ):
        negatives = negative_counts(inputs.counts, labels, mask, inputs.classes)
        flags = contamination_flags(
            negatives,
            inputs.total_counts,
            labels,
            inputs.confident_class == labels,
            inputs.classes,
        )
        q95_broad = expected_genes_quantile(
            query_depth, labels, broad_profiles, seed=seed
        )
        node_labels = np.where(pd.isna(labels), None, inputs.node)
        q95_node = expected_genes_quantile(
            query_depth, node_labels, node_profiles, seed=seed + 101
        )
        diffuse = {
            "diffuse_broad_profile": np.isfinite(q95_broad) & (n_genes > q95_broad),
            "diffuse_node_profile": np.isfinite(q95_node) & (n_genes > q95_node),
        }
        all_flags = {"contaminated": flags.flag, **diffuse}
        limits = {
            "contaminated": CONTAMINATION_MAX_INFORMATIVE,
            "diffuse_broad_profile": DIFFUSE_MAX_INFORMATIVE,
            "diffuse_node_profile": DIFFUSE_MAX_INFORMATIVE,
        }
        for name, values in all_flags.items():
            frame = realised_rates(
                values, labels, inputs.classes, max_informative=limits[name]
            )
            frame["informative_h16"] = frame["rate"] <= H16_MAX_INFORMATIVE
            if name == "contaminated":
                no_null = [cls for cls, fit in flags.fits.items() if fit is None]
                frame.loc[frame["class"].isin(no_null), "informative"] = False
            rates.append(frame.assign(**base, class_set=class_set, flag=name))
            names = np.asarray(labels, dtype=object)
            bins = pd.cut(query_depth, DEPTH_BINS, right=False)
            depth = (
                pd.DataFrame({"flag": values, "bin": bins, "labelled": pd.notna(names)})
                .query("labelled")
                .groupby("bin", observed=False)["flag"]
                .agg(["size", "mean"])
                .reset_index()
            )
            depth_rows.append(
                depth.assign(**base, class_set=class_set, flag_name=name).astype(
                    {"bin": str}
                )
            )
        for cls, fit in flags.fits.items():
            nulls.append(
                {
                    **base,
                    "class_set": class_set,
                    "class": cls,
                    "fitted": fit is not None,
                    "alpha": fit.alpha if fit else np.nan,
                    "beta": fit.beta if fit else np.nan,
                    "n_null_cells": fit.n_cells if fit else 0,
                    "null_mean_rate": fit.mean_rate if fit else np.nan,
                    "n_negative_genes": int(mask[inputs.classes.index(cls)].sum()),
                }
            )
        names = np.asarray(labels, dtype=object)
        for cls in inputs.classes:
            members = names == cls
            values = flags.score[members]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            q = np.quantile(values, [0.25, 0.5, 0.75, 0.95])
            scores.append(
                {
                    **base,
                    "class_set": class_set,
                    "class": cls,
                    "n_cells": len(values),
                    "score_mean": float(values.mean()),
                    "score_q25": q[0],
                    "score_median": q[1],
                    "score_q75": q[2],
                    "score_q95": q[3],
                    "share_score_zero": float((values == 0).mean()),
                }
            )
    return [
        pd.concat(rates, ignore_index=True),
        pd.DataFrame(nulls),
        pd.DataFrame(scores),
        pd.concat(depth_rows, ignore_index=True),
    ]


def human_inputs(
    args: argparse.Namespace, pair: str, seg: str, n_segmented: dict[Any, int]
) -> list[FlagInputs]:
    """Flag inputs of one human pair x segmentation."""
    run_dir = args.runs_root / pair / seg
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
    output = []
    for platform in PLATFORMS:
        sample = load_sample(
            run_dir,
            args.results_root,
            pair,
            seg,
            platform,
            n_segmented.get((pair, seg, platform)),
        )
        bundle = MmcBundle.from_dir(
            manifest.samples[sample.sample_id].runs["whb_frontal_supc_clus"].bundle_path
        )
        query = queries[platform]
        position = pd.Index(query.cell_ids).get_indexer(sample.labels.index)
        v1 = sample.rules["seaad"]
        output.append(
            FlagInputs(
                dataset=sample.sample_id,
                species="human",
                platform=platform,
                segmentation=seg,
                held_out=pair in HELD_OUT_PAIRS or seg != "proseg_hybrid",
                counts=query.counts[position],
                gene_ids=list(query.gene_ids),
                total_counts=sample.labels["total_counts"].to_numpy(np.float64),
                confident_class=np.where(v1.broad_confident, v1.broad_name, None),
                argmax_class=np.where(
                    np.isin(argmax_broad_names(sample.labels), HUMAN_BROAD_CLASSES),
                    argmax_broad_names(sample.labels),
                    None,
                ),
                node=sample.labels["mmc_whb_supercluster_name"].to_numpy(object),
                negatives=pd.read_parquet(bundle.path / "negative_genes.parquet"),
                profiles=pd.read_parquet(bundle.path / "profiles.parquet"),
                vocab=pd.read_csv(
                    bundle.path / "vocab_snapshot.csv", dtype=str, keep_default_na=False
                ),
                node_level="CCN202210140_SUPC",
                classes=HUMAN_BROAD_CLASSES,
            )
        )
    return output


def mouse_inputs(run_dir: Path, clustered: Path, seg: str) -> FlagInputs:
    """Flag inputs of one mouse section x segmentation."""
    manifest = load_map_manifest(run_dir / "map_manifest.json")
    panel = AnnotationPanel.model_validate_json(
        (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
    )
    query = published_queries(
        {"MERSCOPE": clustered},
        panel,
        species="mouse",
        gene_id_fallback_csv=MOUSE_FALLBACK,
    )["MERSCOPE"]
    sample_id = next(iter(manifest.samples))
    labels = pd.read_parquet(
        run_dir / "merscope" / f"{sample_id}_ct_provisional.parquet"
    ).set_index("cell_id")
    labels = labels[labels["in_table"].to_numpy(bool)]
    position = pd.Index(query.cell_ids).get_indexer(labels.index)
    bundle = MmcBundle.from_dir(
        next(iter(manifest.samples[sample_id].runs.values())).bundle_path
    )
    confidence = mouse_confidence(labels)
    broad = labels["ct_broad_name"].astype(object).to_numpy()
    in_classes = np.isin(broad, MOUSE_BROAD_CLASSES)
    return FlagInputs(
        dataset=sample_id,
        species="mouse",
        platform="MERSCOPE",
        segmentation=seg,
        held_out=False,
        counts=query.counts[position],
        gene_ids=list(query.gene_ids),
        total_counts=labels["total_counts"].to_numpy(np.float64),
        confident_class=np.where(confidence.class_confident & in_classes, broad, None),
        argmax_class=np.where(in_classes, broad, None),
        node=labels["mmc_wmb_class_name"].astype(object).to_numpy(),
        negatives=pd.read_parquet(bundle.path / "negative_genes.parquet"),
        profiles=pd.read_parquet(bundle.path / "profiles.parquet"),
        vocab=pd.read_csv(
            bundle.path / "vocab_snapshot.csv", dtype=str, keep_default_na=False
        ),
        node_level="CCN20230722_CLAS",
        classes=MOUSE_BROAD_CLASSES,
    )


def main(argv: list[str] | None = None) -> int:
    """Compute the realised flag rates."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default="proseg_hybrid,reseg")
    parser.add_argument(
        "--mouse",
        action="append",
        default=[],
        metavar="SEG=RUN_DIR=CLUSTERED_H5AD",
        help="a mouse MAP output and its published clustered H5AD",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    qc = pd.read_csv(args.evidence_root / "research/lowcount/qc_summary.csv")
    n_segmented = {
        (str(r.sample), str(r.seg), str(r.platform).upper()): int(r.n_cells)
        for r in qc.itertuples(index=False)
    }
    tables: list[list[pd.DataFrame]] = [[], [], [], []]
    work: list[FlagInputs] = []
    for seg in [s for s in args.segmentations.split(",") if s]:
        for pair in [p for p in args.pairs.split(",") if p]:
            if not (args.runs_root / pair / seg / "map_manifest.json").is_file():
                logger.warning("no MAP outputs for %s %s", pair, seg)
                continue
            for inputs in human_inputs(args, pair, seg, n_segmented):
                for table, frame in zip(
                    tables, score(inputs, seed=args.seed), strict=True
                ):
                    table.append(frame)
                logger.info("%s %s done", inputs.dataset, seg)
    for item in args.mouse:
        seg, run_dir, clustered = item.split("=", 2)
        work.append(mouse_inputs(Path(run_dir), Path(clustered), seg))
        for table, frame in zip(tables, score(work[-1], seed=args.seed), strict=True):
            table.append(frame)
        logger.info("%s %s done", work[-1].dataset, seg)
        work.clear()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frames in zip(
        ("flag_rates.csv", "flag_nulls.csv", "flag_scores.csv", "flag_depth.csv"),
        tables,
        strict=True,
    ):
        pd.concat(frames, ignore_index=True).to_csv(args.out / name, index=False)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
