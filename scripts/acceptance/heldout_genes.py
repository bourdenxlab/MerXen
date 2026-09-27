#!/usr/bin/env python
"""Held-out-gene enrichment (plan §5.8; H4; M3 items 3 and 4).

For each human pair (proseg_hybrid) and platform:

1. choose each broad class's held-out markers on the panel
   (``assets/annotation/heldout_markers_human.csv``: the plan's canonical
   markers, then canonical alternates for markers the panel lacks; SST never;
   GAD2 and P2RY12 not on MERSCOPE; 2-3 per class, a class with < 2 is
   skipped and reported);
2. remove them from the query **and** the lookup and re-map with WHB only
   (the production engine configuration);
3. score the held-out markers' counts in the assigned class vs the other
   classes: pooled fold enrichment and the AUROC of the per-cell held-out
   fraction, per class and platform. H4: fold >= 3 and AUROC >= 0.70 in >= 6
   of the 7 broad classes.

Variants (``--variants``): ``set_a`` (the §5.8 baseline, WHB set a bundle),
``set_c`` (the set-c panel and bundle) and ``x1`` (set a with the X1 platform
rescaling: per-gene reference-pseudobulk factors, capped at +/- 2 log2,
estimated from the held-out set-a re-map's cells with bp >= 0.8). Labels are
scored on all table cells of the held-out re-map, in the options the user
chooses H4's "assigned class" from (plan §5.8 does not define it; M3 PR):

- (a) ``heldout_argmax``: the argmax broad class (the M3 working metric);
- (b) ``heldout_argmax_cop_rule``: the argmax with the §5.2 COP rule applied
  (a COP call is broad OPC only with >= the COP supercluster floor and
  supercluster bp >= 0.69; otherwise it stays at lineage and is left out of
  the classes); ``heldout_argmax_cop_rule_seaad`` adds the rule's SEA-AD
  confident-OPC rescue from the production SEA-AD calls (which saw the
  held-out genes);
- (c) ``heldout_whb_confident``: the WHB-only confident broad calls (v1
  thresholds and floors, no second method: SEA-AD saw the held-out genes);

and ``production_argmax`` (circular, for comparison).

Writes ``heldout_markers.csv``, ``heldout_enrichment.csv``, ``heldout_h4.csv``
and ``runs/<pair>/<sid>_<variant>.parquet`` to ``--out``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.config import AnnotationThresholds
from merxen.annotation.mapmycells_engine import (
    MmcBundle,
    level_frame,
    read_tidy_parquet,
)
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import SampleQuery, load_map_manifest
from merxen.annotation.schema import meets_threshold
from merxen.annotation.shadow import (
    OPC,
    X1_LABEL_MIN_BP,
    argmax_broad_names,
    cop_rule_broad_names,
    evaluate_human_rules,
    heldout_enrichment,
    map_query_variant,
    profile_matrix,
    published_queries,
    reference_pseudobulk_log2_factors,
    rule_inputs_from_provisional,
    seaad_broad_calls,
    select_heldout_markers,
    whb_labels_from_tidy,
)
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES, load_heldout_markers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import (  # noqa: E402
    HELD_OUT_PAIRS,
    PAIRS,
    PLATFORMS,
    _clustered_path,
    add_fallback_arguments,
)

logger = logging.getLogger("heldout_genes")

VARIANTS = ("set_a", "set_c", "x1")
RUN_IDS = {
    "set_a": "whb_frontal_supc_clus",
    "set_c": "whb_frontal_supc_clus_setc",
    "x1": "whb_frontal_supc_clus",
}
PANEL_FILES = {
    "set_a": "panel_genes.json",
    "set_c": "panel_genes_setc.json",
    "x1": "panel_genes.json",
}
PREFIXES = {"set_a": "mmc_whb", "set_c": "mmc_whb_setc", "x1": "mmc_whb"}
WHB_SUPC = "CCN202210140_SUPC"
H4_PAIRS = ("P7513", "P1212", "P7113")
H4_MIN_CLASSES = 6


def marker_ids(
    panel: AnnotationPanel, platform: str, symbols: tuple[str, ...]
) -> dict[str, str]:
    """Return symbol -> Ensembl ID of held-out markers on a panel/platform."""
    by_platform = panel.symbols_by_platform.get(platform) or panel.symbols
    lookup = {
        str(symbol).upper(): gene_id
        for symbol, gene_id in zip(by_platform, panel.ensembl_ids, strict=True)
        if symbol
    }
    return {symbol: lookup[symbol.upper()] for symbol in symbols}


def marker_count_frame(
    query: SampleQuery, ids: dict[str, str], cell_ids: pd.Index
) -> pd.DataFrame:
    """Return the held-out markers' counts (symbols as columns) by cell."""
    column = {gene: i for i, gene in enumerate(query.gene_ids)}
    position = pd.Index(query.cell_ids).get_indexer(cell_ids)
    symbols = list(ids)
    block = query.counts[position][:, [column[ids[symbol]] for symbol in symbols]]
    return pd.DataFrame(np.asarray(block.toarray()), index=cell_ids, columns=symbols)


def enrichment_rows(
    results: list[Any], base: dict[str, Any], label_set: str
) -> list[dict[str, Any]]:
    """Flatten ``heldout_enrichment`` results into CSV rows."""
    return [
        {
            **base,
            "label_set": label_set,
            "broad_class": item.broad_class,
            "markers": ";".join(item.markers),
            "n_assigned": item.n_assigned,
            "n_other": item.n_other,
            "rate_assigned_per_1k": item.rate_assigned * 1000,
            "rate_other_per_1k": item.rate_other * 1000,
            "fold": item.fold,
            "detection_assigned": item.detection_assigned,
            "detection_other": item.detection_other,
            "auroc": item.auroc,
            "passes": item.passes,
        }
        for item in results
    ]


def run_pair(
    pair: str,
    args: argparse.Namespace,
    marker_table: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Held-out re-maps and enrichment of one pair."""
    run_dir = args.runs_root / pair / "proseg_hybrid"
    manifest = load_map_manifest(run_dir / "map_manifest.json")
    clustered = {
        platform: _clustered_path(args.results_root, pair, "proseg_hybrid", platform)
        for platform in PLATFORMS
    }
    out_dir = args.out / "runs" / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    heldout_labels: dict[str, pd.DataFrame] = {}
    for variant in [v for v in args.variants.split(",") if v]:
        panel = AnnotationPanel.model_validate_json(
            (run_dir / "panel" / PANEL_FILES[variant]).read_text(encoding="utf-8")
        )
        queries = published_queries(
            clustered,
            panel,
            species="human",
            gene_id_fallback_csv=args.gene_id_fallback_csv,
        )
        for platform in PLATFORMS:
            sample_id = f"{pair}_{platform}"
            record = manifest.samples[sample_id].runs[RUN_IDS[variant]]
            bundle = MmcBundle.from_dir(record.bundle_path)
            vocab = pd.read_csv(
                bundle.path / "vocab_snapshot.csv", dtype=str, keep_default_na=False
            )
            symbols = panel.symbols_by_platform.get(platform) or panel.symbols
            selection = select_heldout_markers(marker_table, symbols, platform)
            ids = marker_ids(panel, platform, selection.all_markers)
            query = queries[platform]
            for cls in HUMAN_BROAD_CLASSES:
                selection_rows.append(
                    {
                        "pair": pair,
                        "platform": platform,
                        "variant": variant,
                        "panel_hash": panel.panel_hash[:16],
                        "broad_class": cls,
                        "markers": ";".join(selection.markers.get(cls, ())),
                        "skipped": selection.skipped.get(cls, ""),
                        "not_on_panel": ";".join(selection.not_on_panel.get(cls, ())),
                        "avoided_on_platform": ";".join(selection.avoided.get(cls, ())),
                    }
                )
            factors = None
            if variant == "x1":
                labels_a = heldout_labels[f"set_a_{platform}"]
                kept = [g for g in query.gene_ids if g not in set(ids.values())]
                columns = [query.gene_ids.index(g) for g in kept]
                profiles = profile_matrix(
                    pd.read_parquet(bundle.path / "profiles.parquet"), WHB_SUPC, kept
                )
                aligned = labels_a.reindex(pd.Index(query.cell_ids))
                capped, _ = reference_pseudobulk_log2_factors(
                    query.counts[:, columns],
                    aligned["mmc_whb_supercluster_name"].to_numpy(object),
                    profiles,
                    include=aligned["mmc_whb_supercluster_bp"].to_numpy(float)
                    >= X1_LABEL_MIN_BP,
                )
                factors = dict(zip(kept, capped.tolist(), strict=True))
            parquet = map_query_variant(
                query,
                bundle,
                out_dir / f"{sample_id}_{variant}_heldout.parquet",
                work_dir=args.work_dir / f"{pair}_{platform}_{variant}",
                drop_gene_ids=list(ids.values()),
                log2_factors=factors,
                n_processors=args.n_processors,
                run_metadata={"heldout_markers": sorted(ids)},
            )
            tidy, _ = read_tidy_parquet(parquet)
            total = pd.Series(query.total_counts, index=pd.Index(query.cell_ids))
            labels = whb_labels_from_tidy(tidy, vocab, total)
            heldout_labels[f"{variant}_{platform}"] = labels
            cells = labels.index
            counts = marker_count_frame(query, ids, cells)
            argmax = argmax_broad_names(labels)
            sea_tidy, _ = read_tidy_parquet(
                run_dir / platform.lower() / f"{sample_id}_mmc_seaad_mr_panel.parquet"
            )
            sea = seaad_broad_calls(
                level_frame(sea_tidy, "subclass"),
                level_frame(sea_tidy, "supertype"),
                class_level=level_frame(sea_tidy, "class"),
            ).reindex(cells)
            sea_confident_opc = (sea["broad"].to_numpy(object) == OPC) & (
                meets_threshold(sea["broad_raw"], AnnotationThresholds().seaad_broad)
            )
            cop_rule = {
                "heldout_argmax_cop_rule": None,
                "heldout_argmax_cop_rule_seaad": sea_confident_opc,
            }
            whb_only = evaluate_human_rules(
                rule_inputs_from_provisional(labels),
                platform=platform,
                second_vote="none",
            )
            production = pd.read_parquet(
                run_dir / platform.lower() / f"{sample_id}_ct_provisional.parquet"
            ).set_index("cell_id")
            production = production.reindex(cells)
            production_argmax = argmax_broad_names(production, prefix=PREFIXES[variant])
            base = {
                "pair": pair,
                "platform": platform,
                "variant": variant,
                "held_out_pair": pair in HELD_OUT_PAIRS,
                "n_cells": len(cells),
                "n_query_genes": len(query.gene_ids) - len(ids),
                "n_heldout_genes": len(ids),
            }
            total_counts = labels["total_counts"].to_numpy(np.float64)
            markers = selection.markers
            primary = heldout_enrichment(counts, total_counts, argmax, markers)
            rows += enrichment_rows(primary, base, "heldout_argmax")
            for label_set, rescue in cop_rule.items():
                cop_labels = cop_rule_broad_names(
                    argmax,
                    labels["mmc_whb_supercluster_name"].to_numpy(object),
                    labels["mmc_whb_supercluster_bp"].to_numpy(np.float64),
                    total_counts,
                    platform=platform,
                    sea_confident_opc=rescue,
                )
                rows += enrichment_rows(
                    heldout_enrichment(counts, total_counts, cop_labels, markers),
                    base,
                    label_set,
                )
            rows += enrichment_rows(
                heldout_enrichment(
                    counts,
                    total_counts,
                    argmax,
                    markers,
                    include=whb_only.broad_confident,
                ),
                base,
                "heldout_whb_confident",
            )
            rows += enrichment_rows(
                heldout_enrichment(counts, total_counts, production_argmax, markers),
                base,
                "production_argmax",
            )
            logger.info(
                "%s %s %s: %d held-out genes, H4 passes %d of %d scored classes",
                pair,
                platform,
                variant,
                len(ids),
                sum(item.passes for item in primary),
                len(primary),
            )
    return selection_rows, rows


def h4_rows(enrichment: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarise H4 (classes passing of 7) per pair, platform and variant."""
    rows = []
    for (pair, platform, variant, label_set), frame in enrichment.groupby(
        ["pair", "platform", "variant", "label_set"]
    ):
        n_pass = int(frame["passes"].sum())
        rows.append(
            {
                "pair": pair,
                "platform": platform,
                "variant": variant,
                "label_set": label_set,
                "n_scored": len(frame),
                "n_pass": n_pass,
                "failing": ";".join(
                    frame.loc[~frame["passes"].astype(bool), "broad_class"]
                ),
                "h4_scored_pair": pair in H4_PAIRS,
                "h4_pass": n_pass >= H4_MIN_CLASSES,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the held-out-gene re-maps and score the enrichment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--n-processors", type=int, default=6)
    add_fallback_arguments(parser)
    parser.add_argument("--tag", default="", help="suffix of the output CSVs")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if "x1" in args.variants.split(",") and "set_a" not in args.variants.split(","):
        parser.error("the x1 variant needs set_a (its factors use set a's labels)")
    marker_table = load_heldout_markers("human")
    selection_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pair in [p for p in args.pairs.split(",") if p]:
        selection, enrichment = run_pair(pair, args, marker_table)
        selection_rows += selection
        rows += enrichment
    args.out.mkdir(parents=True, exist_ok=True)
    enrichment_frame = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    pd.DataFrame(selection_rows).to_csv(
        args.out / f"heldout_markers{suffix}.csv", index=False
    )
    enrichment_frame.to_csv(args.out / f"heldout_enrichment{suffix}.csv", index=False)
    pd.DataFrame(h4_rows(enrichment_frame)).to_csv(
        args.out / f"heldout_h4{suffix}.csv", index=False
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
