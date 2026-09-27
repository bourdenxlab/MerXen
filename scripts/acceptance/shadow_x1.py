#!/usr/bin/env python
"""X1 platform rescaling vs set a and set c (plan §12 M3 item 3, D-A7, OD-B9).

X1 divides each query gene by a per-platform factor before mapping: the
log2 ratio of the dataset's observed gene totals to the reference
pseudobulk expected from its cells' production WHB supercluster calls
(bp >= 0.8; ``profiles.parquet`` expected fractions), centred on the median
and capped at +/- 2 log2 (``reference_pseudobulk_log2_factors``). The
rescaled query is re-mapped with the production WHB set-a configuration.

For each human pair (proseg_hybrid) this compares set a (production),
set c (production sensitivity run) and X1 on:

- MERSCOPE-vs-Xenium broad JSD (soft, argmax, confident under the v1
  shadow rules with the production SEA-AD calls), with 200-replicate block
  bootstrap CIs and paired bootstrap CIs of the differences;
- the E1 marker referee on X1-vs-set-a, X1-vs-set-c and set-c-vs-set-a broad
  disputes;
- confident broad coverage and the broad compositions;
- the factors themselves (spread, share capped, agreement across pairs).

The held-out-gene comparison (H4 with X1) is ``heldout_genes.py --variants
set_a,set_c,x1``. **X1 passes** (becomes the ``geneset_c+rescale``
sensitivity, OD-B9) only if, on both development pairs (P7513, P1212), its
soft JSD is below set a's with the paired 95% CI of the difference below 0,
the referee sides with X1 in more X1-vs-set-a disputes than with set a, and
it passes H4 in no fewer classes than set a on any development sample.

Writes ``x1_factors.csv``, ``x1_jsd.csv``, ``x1_jsd_differences.csv``,
``x1_referee.csv``, ``x1_samples.csv``, ``x1_compositions.csv`` and
``runs/<pair>/<sid>_x1.parquet`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import MmcBundle, read_tidy_parquet
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.pipeline import load_map_manifest
from merxen.annotation.shadow import (
    X1_LABEL_MIN_BP,
    argmax_broad_names,
    block_bootstrap_jsd,
    composition_shares,
    evaluate_human_rules,
    map_query_variant,
    marker_referee,
    one_hot_broad_matrix,
    paired_block_bootstrap_jsd_difference,
    profile_matrix,
    published_queries,
    reference_pseudobulk_log2_factors,
    rule_inputs_from_provisional,
    soft_matrix_from_provisional,
    tile_codes,
    tile_sums,
    whb_labels_from_tidy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import (  # noqa: E402
    HELD_OUT_PAIRS,
    PAIRS,
    PLATFORMS,
    Sample,
    _clustered_path,
    load_sample,
)

logger = logging.getLogger("shadow_x1")

WHB_SUPC = "CCN202210140_SUPC"
SET_A_RUN = "whb_frontal_supc_clus"
SET_C_RUN = "whb_frontal_supc_clus_setc"
LABELLINGS = ("set_a", "set_c", "x1")
KINDS = ("soft", "argmax", "confident")
FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/"
    "WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
)


def labelling_matrices(
    labels: pd.DataFrame, sample: Sample, n_segmented: int | None
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Soft, argmax and confident broad matrices of one WHB labelling."""
    inputs = rule_inputs_from_provisional(labels, sample.sea)
    rules = evaluate_human_rules(
        inputs, platform=sample.platform, second_vote="seaad", n_segmented=n_segmented
    )
    argmax = argmax_broad_names(labels)
    matrices = {
        "soft": soft_matrix_from_provisional(labels),
        "argmax": one_hot_broad_matrix(argmax),
        "confident": one_hot_broad_matrix(
            rules.broad_name, include=rules.broad_confident
        ),
    }
    return matrices, argmax, float(rules.broad_confident.mean())


def main(argv: list[str] | None = None) -> int:
    """Run X1 and compare it with set a and set c."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--n-processors", type=int, default=6)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--gene-id-fallback-csv", default=FALLBACK)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    qc = pd.read_csv(args.evidence_root / "research/lowcount/qc_summary.csv")
    n_segmented = {
        (str(r.sample), str(r.seg), str(r.platform).upper()): int(r.n_cells)
        for r in qc.itertuples(index=False)
    }
    factor_rows: list[dict[str, Any]] = []
    jsd_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    referee_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    seg = "proseg_hybrid"
    for pair in [p for p in args.pairs.split(",") if p]:
        run_dir = args.runs_root / pair / seg
        manifest = load_map_manifest(run_dir / "map_manifest.json")
        panel = AnnotationPanel.model_validate_json(
            (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
        )
        queries = published_queries(
            {p: _clustered_path(args.results_root, pair, seg, p) for p in PLATFORMS},
            panel,
            species="human",
            gene_id_fallback_csv=args.gene_id_fallback_csv,
        )
        out_dir = args.out / "runs" / pair
        out_dir.mkdir(parents=True, exist_ok=True)
        per_platform: dict[str, dict[str, Any]] = {}
        for platform in PLATFORMS:
            sample_id = f"{pair}_{platform}"
            key = (pair, seg, platform)
            sample = load_sample(
                run_dir, args.results_root, pair, seg, platform, n_segmented.get(key)
            )
            record = manifest.samples[sample_id].runs[SET_A_RUN]
            bundle = MmcBundle.from_dir(record.bundle_path)
            vocab = pd.read_csv(
                bundle.path / "vocab_snapshot.csv", dtype=str, keep_default_na=False
            )
            query = queries[platform]
            genes = list(query.gene_ids)
            profiles = profile_matrix(
                pd.read_parquet(bundle.path / "profiles.parquet"), WHB_SUPC, genes
            )
            production = sample.labels.reindex(pd.Index(query.cell_ids))
            capped, uncapped = reference_pseudobulk_log2_factors(
                query.counts,
                production["mmc_whb_supercluster_name"].to_numpy(object),
                profiles,
                include=production["mmc_whb_supercluster_bp"].to_numpy(float)
                >= X1_LABEL_MIN_BP,
            )
            symbols = dict(zip(panel.ensembl_ids, panel.symbols, strict=True))
            factor_rows.extend(
                {
                    "pair": pair,
                    "platform": platform,
                    "gene_id": gene,
                    "symbol": symbols.get(gene, ""),
                    "log2_factor": float(c),
                    "log2_factor_uncapped": float(u),
                    "capped": bool(abs(u) > 2.0),
                }
                for gene, c, u in zip(genes, capped, uncapped, strict=True)
            )
            parquet = map_query_variant(
                query,
                bundle,
                out_dir / f"{sample_id}_x1.parquet",
                work_dir=args.work_dir / f"{pair}_{platform}_x1",
                log2_factors=dict(zip(genes, capped.tolist(), strict=True)),
                n_processors=args.n_processors,
                run_metadata={"x1_cap_log2": 2.0, "x1_label_min_bp": X1_LABEL_MIN_BP},
            )
            total = pd.Series(query.total_counts, index=pd.Index(query.cell_ids))
            x1_labels = whb_labels_from_tidy(
                read_tidy_parquet(parquet)[0], vocab, total
            )
            x1_labels = x1_labels.reindex(sample.labels.index)
            setc_tidy, _ = read_tidy_parquet(
                run_dir / platform.lower() / f"{sample_id}_mmc_{SET_C_RUN}.parquet"
            )
            setc_bundle = MmcBundle.from_dir(
                manifest.samples[sample_id].runs[SET_C_RUN].bundle_path
            )
            setc_vocab = pd.read_csv(
                setc_bundle.path / "vocab_snapshot.csv",
                dtype=str,
                keep_default_na=False,
            )
            setc_labels = whb_labels_from_tidy(setc_tidy, setc_vocab, total).reindex(
                sample.labels.index
            )
            labellings = {
                "set_a": sample.labels,
                "set_c": setc_labels,
                "x1": x1_labels,
            }
            matrices, argmaxes, coverage = {}, {}, {}
            for name, labels in labellings.items():
                matrices[name], argmaxes[name], coverage[name] = labelling_matrices(
                    labels, sample, n_segmented.get(key)
                )
            per_platform[platform] = {
                "sample": sample,
                "matrices": matrices,
                "codes": tile_codes(sample.xy),
            }
            row: dict[str, Any] = {
                "pair": pair,
                "sample_id": sample_id,
                "platform": platform,
                "held_out": pair in HELD_OUT_PAIRS,
                "n_cells": len(sample.labels),
                "share_factors_capped": float((np.abs(uncapped) > 2).mean()),
                "factor_sd_log2": float(np.std(uncapped)),
                "n_query_genes": len(genes),
            }
            for name in LABELLINGS:
                row[f"broad_cov_{name}"] = coverage[name]
                shares = composition_shares(matrices[name]["argmax"])
                for cls, value in shares.items():
                    comp_rows.append(
                        {
                            "pair": pair,
                            "platform": platform,
                            "labelling": name,
                            "kind": "argmax",
                            "class": cls,
                            "share": value,
                        }
                    )
            row["x1_vs_set_a_label_agreement"] = float(
                (argmaxes["x1"] == argmaxes["set_a"]).mean()
            )
            row["set_c_vs_set_a_label_agreement"] = float(
                (argmaxes["set_c"] == argmaxes["set_a"]).mean()
            )
            sample_rows.append(row)
            for first, second in (("x1", "set_a"), ("x1", "set_c"), ("set_c", "set_a")):
                result = marker_referee(
                    sample.scores, argmaxes[first], argmaxes[second]
                )
                referee_rows.append(
                    {
                        "pair": pair,
                        "sample_id": sample_id,
                        "first": first,
                        "second": second,
                        "n_disputes": result.n_disputes,
                        "dispute_share": result.n_disputes / max(result.n_cells, 1),
                        "first_wins": result.first_wins,
                        "second_wins": result.second_wins,
                        "undecided": result.undecided,
                        "top_disputes": json.dumps(result.top_disputes),
                    }
                )
        merscope, xenium = per_platform["MERSCOPE"], per_platform["XENIUM"]
        tiles = {
            (name, kind): (
                tile_sums(merscope["matrices"][name][kind], merscope["codes"]),
                tile_sums(xenium["matrices"][name][kind], xenium["codes"]),
            )
            for name in LABELLINGS
            for kind in KINDS
        }
        for (name, kind), (tiles_m, tiles_x) in tiles.items():
            result = block_bootstrap_jsd(tiles_m, tiles_x, n_reps=args.n_bootstrap)
            jsd_rows.append(
                {
                    "pair": pair,
                    "labelling": name,
                    "kind": kind,
                    "held_out": pair in HELD_OUT_PAIRS,
                    "jsd": result.jsd,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                }
            )
        for kind in KINDS:
            for first, second in (("x1", "set_a"), ("x1", "set_c"), ("set_c", "set_a")):
                a_m, a_x = tiles[(first, kind)]
                b_m, b_x = tiles[(second, kind)]
                diff = paired_block_bootstrap_jsd_difference(
                    a_m, a_x, b_m, b_x, n_reps=args.n_bootstrap
                )
                diff_rows.append(
                    {
                        "pair": pair,
                        "kind": kind,
                        "first": first,
                        "second": second,
                        "held_out": pair in HELD_OUT_PAIRS,
                        "first_jsd": diff.first_jsd,
                        "second_jsd": diff.second_jsd,
                        "difference": diff.difference,
                        "ci_low": diff.ci_low,
                        "ci_high": diff.ci_high,
                        "share_positive": diff.share_positive,
                    }
                )
        logger.info("%s done", pair)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "x1_factors.csv": factor_rows,
        "x1_jsd.csv": jsd_rows,
        "x1_jsd_differences.csv": diff_rows,
        "x1_referee.csv": referee_rows,
        "x1_samples.csv": sample_rows,
        "x1_compositions.csv": comp_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
