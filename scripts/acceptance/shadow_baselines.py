#!/usr/bin/env python
"""M3 shadow baselines on the human datasets (plan §12 M3 item 1, §14).

Reads the standalone ``merxen annotate`` outputs (``<runs-root>/<pair>/<seg>/``:
``map_manifest.json``, ``<plat>/<sid>_ct_provisional.parquet``,
``<plat>/<sid>_mmc_seaad_mr_panel.parquet``) and the published inputs they
were mapped from (``<results-root>/<pair>/<seg>/clustering_squidpy/
clustering_squidpy_out/<plat>/<sid>_clustered.h5ad``: coordinates, counts,
legacy ``broad_class``; ``<results-root>/<pair>/alignment/align_out/`` for the
shared tissue mask), all read-only, and writes to ``--out``:

- ``sample_metrics.csv``: per sample x segmentation, the dataset gate (A,
  level, warning), confident lineage / broad / supercluster coverage of table
  cells (and broad of segmented objects) under the shadow v1 rules
  (``merxen.annotation.shadow.evaluate_human_rules``) for the second-vote
  variants ``none`` / ``seaad_from60`` / ``seaad`` (v1), the implausible-call
  share, COP shares, WHB-SEA 7-class agreement (>= 20 counts);
- ``jsd.csv`` / ``compositions.csv``: soft, argmax, confident and >= 30-count
  soft broad compositions per platform and their MERSCOPE-vs-Xenium JSD with
  95% spatial block-bootstrap CIs, whole section and inside the shared mask
  (plus the set-c soft / argmax JSD where a set-c run exists);
- ``referee.csv``: the E1 canonical-marker referee on WHB-vs-SEA and
  new-vs-legacy broad disputes;
- ``second_vote_cost.csv``: on the E2 30k native cells (``--e2-native``), the
  confident broad coverage under no vote, the SEA-AD rules and E2's LL rule
  with the same production WHB calls, and the OD-B13 trigger;
- ``exit_checks.csv``: production vs the pilot configuration (E1 (iii) pilot
  for P7513 / P1212, E2's runs of it for P7113 / P5011) on shared cells;
- ``runtime.csv``: MAP wall time, processor-seconds per 1k cells, peak RSS.

Usage::

    python scripts/acceptance/shadow_baselines.py \\
        --runs-root <shadow>/baselines/runs --results-root <results> \\
        --evidence-root <evidence> --out <shadow>/baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.mapmycells_engine import (
    level_frame,
    parse_extended_json_tidy,
    read_tidy_parquet,
)
from merxen.annotation.panel import load_shared_tissue_mask
from merxen.annotation.pipeline import load_map_manifest, read_h5ad_counts
from merxen.annotation.shadow import (
    AGREEMENT_MIN_COUNTS,
    COMPOSITION_COLUMNS,
    N_BOOTSTRAP,
    OPC,
    TILE_UM,
    HumanRuleResult,
    argmax_broad_names,
    block_bootstrap_jsd,
    composition_shares,
    evaluate_human_rules,
    label_agreement,
    marker_class_scores,
    marker_referee,
    one_hot_broad_matrix,
    rule_inputs_from_provisional,
    seaad_broad_calls,
    soft_matrix_from_provisional,
    tile_codes,
    tile_sums,
)
from merxen.annotation.vocab import COP_SUPERCLUSTER, HUMAN_BROAD_CLASSES

logger = logging.getLogger("shadow_baselines")

PLATFORMS = ("MERSCOPE", "XENIUM")
PAIRS = ("P7513", "P1212", "P7113", "P5011")
SEGMENTATIONS = ("proseg_hybrid", "reseg")
HELD_OUT_PAIRS = ("P7113", "P5011")
VOTES = ("none", "seaad_from60", "seaad")
WHB_LEVEL = "CCN202210140_SUPC"
SEA_LEVEL = "CCN20260630_LEVEL_1"
OD_B13_TRIGGER = 0.08
PILOT_BP = 0.8


@dataclass
class Sample:
    """One sample's table cells, labels and published context."""

    sample_id: str
    platform: str
    labels: pd.DataFrame
    sea: pd.DataFrame
    xy: np.ndarray
    legacy_broad: np.ndarray
    scores: np.ndarray
    aligned_frame: bool
    rules: dict[str, HumanRuleResult]


def _clustered_path(results: Path, pair: str, seg: str, platform: str) -> Path:
    return (
        results
        / pair
        / seg
        / "clustering_squidpy"
        / "clustering_squidpy_out"
        / platform.lower()
        / f"{pair}_{platform}_clustered.h5ad"
    )


def _read_obs_context(
    path: Path, obs_names: pd.Index
) -> tuple[np.ndarray, np.ndarray, str]:
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as handle:
        xy = np.asarray(handle["obsm"]["spatial"][()], dtype=np.float64)[:, :2]
        legacy = np.asarray(read_elem(handle["obs"]["broad_class"]), dtype=object)
        key = "uns/merxen_clustering_squidpy/shape_key"
        shape_key = str(read_elem(handle[key])) if key in handle else ""
    if len(xy) != len(obs_names):
        raise ValueError(f"{path}: obsm['spatial'] does not fit obs")
    return xy, legacy, shape_key


def load_sample(
    run_dir: Path,
    results: Path,
    pair: str,
    seg: str,
    platform: str,
    n_segmented: int | None,
) -> Sample:
    """Load one sample's outputs and published context, then apply the rules."""
    sample_id = f"{pair}_{platform}"
    folder = run_dir / platform.lower()
    labels = pd.read_parquet(folder / f"{sample_id}_ct_provisional.parquet")
    labels = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
    tidy, _ = read_tidy_parquet(folder / f"{sample_id}_mmc_seaad_mr_panel.parquet")
    sea = seaad_broad_calls(
        level_frame(tidy, "subclass"), level_frame(tidy, "supertype")
    )
    path = _clustered_path(results, pair, seg, platform)
    obs_names, var, counts, _ = read_h5ad_counts(path, "clustered")
    xy, legacy, shape_key = _read_obs_context(path, obs_names)
    position = pd.Index(obs_names).get_indexer(labels.index)
    if (position < 0).any():
        raise ValueError(f"{sample_id}: labelled cells missing from {path}")
    symbols = [str(value) for value in var.index]
    scores, _ = marker_class_scores(counts[position], symbols)
    inputs = rule_inputs_from_provisional(labels, sea)
    rules = {
        vote: evaluate_human_rules(
            inputs,
            platform=platform,
            second_vote=vote,
            n_segmented=n_segmented,  # type: ignore[arg-type]
        )
        for vote in VOTES
    }
    return Sample(
        sample_id=sample_id,
        platform=platform,
        labels=labels,
        sea=sea.reindex(labels.index),
        xy=xy[position],
        legacy_broad=legacy[position],
        scores=scores,
        aligned_frame=platform == "XENIUM" or "_aligned" in shape_key,
        rules=rules,
    )


def sample_metrics(sample: Sample, pair: str, seg: str) -> dict[str, Any]:
    """Return the per-sample baseline row."""
    labels = sample.labels
    counts = labels["total_counts"].to_numpy(np.float64)
    v1 = sample.rules["seaad"]
    supercluster = labels["mmc_whb_supercluster_name"].astype(object).to_numpy()
    is_cop = supercluster == COP_SUPERCLUSTER
    opc = v1.broad_confident & (v1.broad_name == OPC)
    whb_broad = argmax_broad_names(labels)
    ge20 = counts >= AGREEMENT_MIN_COUNTS
    agree, n_ge20 = label_agreement(whb_broad, sample.sea["broad"], ge20)
    agree_strict, _ = label_agreement(whb_broad, sample.sea["broad"], ge20, strict=True)
    agree_all, _ = label_agreement(whb_broad, sample.sea["broad"])
    row: dict[str, Any] = {
        "pair": pair,
        "segmentation": seg,
        "sample_id": sample.sample_id,
        "platform": sample.platform,
        "held_out": pair in HELD_OUT_PAIRS or seg != "proseg_hybrid",
        "n_table_cells": len(labels),
        "median_counts": float(np.median(counts)),
        "gate_A_frac_ge30": v1.gate.frac_ge30,
        "gate_level": v1.gate.level,
        "gate_warning": v1.gate.warning,
        "gate_reasons": "; ".join(v1.gate.reasons),
        "implausible_share": float(v1.implausible.mean()),
        "implausible_share_lineage_raw_ge_threshold": float(
            (v1.implausible & (labels["ct_lineage_raw"].to_numpy() >= 0.73)).mean()
        ),
        "implausible_top_nodes": json.dumps(
            pd.Series(supercluster[v1.implausible], dtype=object)
            .value_counts()
            .head(4)
            .to_dict()
        ),
        "sink_share": float(
            np.isin(supercluster, ["Splatter", "Miscellaneous"]).mean()
        ),
        "whb_sea_agree_ge20": agree,
        "whb_sea_agree_ge20_strict": agree_strict,
        "whb_sea_agree_all": agree_all,
        "n_ge20": n_ge20,
        "cop_argmax_share": float(is_cop.mean()),
        "cop_confident_supercluster_share": float(
            (v1.supercluster_confident & is_cop).mean()
        ),
        "opc_confident_broad_share": float(opc.mean()),
        "cop_derived_share_of_confident_opc": (
            float((opc & is_cop).sum() / opc.sum()) if opc.any() else math.nan
        ),
        "cop_suppressed_share": float(v1.cop_suppressed.mean()),
    }
    for vote, result in sample.rules.items():
        row[f"lineage_cov_{vote}"] = float(result.lineage_confident.mean())
        row[f"broad_cov_{vote}"] = float(result.broad_confident.mean())
        row[f"supercluster_cov_{vote}"] = float(result.supercluster_confident.mean())
        row[f"broad_cov_segmented_{vote}"] = result.gate.broad_coverage_segmented
    row["below60_sea_rule_cost"] = (
        row["broad_cov_seaad_from60"] - row["broad_cov_seaad"]
    )
    return row


def _compositions(sample: Sample, *, include_setc: bool) -> dict[str, np.ndarray]:
    labels = sample.labels
    counts = labels["total_counts"].to_numpy(np.float64)
    v1 = sample.rules["seaad"]
    soft = soft_matrix_from_provisional(labels)
    kinds = {
        "soft": soft,
        "argmax": one_hot_broad_matrix(argmax_broad_names(labels)),
        "confident": one_hot_broad_matrix(v1.broad_name, include=v1.broad_confident),
        "soft_ge30": np.where((counts >= 30)[:, None], soft, 0.0),
    }
    if include_setc and "mmc_whb_setc_supercluster_name" in labels.columns:
        kinds["setc_soft"] = soft_matrix_from_provisional(labels, prefix="mmc_whb_setc")
        kinds["setc_argmax"] = one_hot_broad_matrix(
            argmax_broad_names(labels, prefix="mmc_whb_setc")
        )
    return kinds


def pair_jsd(
    samples: dict[str, Sample],
    pair: str,
    seg: str,
    results: Path,
    *,
    n_reps: int,
    tile_um: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return JSD rows and composition rows of one pair x segmentation."""
    merscope, xenium = samples["MERSCOPE"], samples["XENIUM"]
    regions: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "whole_section": (
            np.ones(len(merscope.labels), bool),
            np.ones(len(xenium.labels), bool),
        )
    }
    align = results / pair / "alignment" / "align_out"
    mask_path, summary_path = (
        align / "shared_tissue_mask.npy",
        align / "registration_summary.json",
    )
    mask_note = "no shared_tissue_mask.npy"
    if mask_path.is_file() and summary_path.is_file():
        mask = load_shared_tissue_mask(mask_path, summary_path)
        if merscope.aligned_frame and mask.fixed_platform == "XENIUM":
            regions["shared_mask"] = (
                mask.contains(merscope.xy),
                mask.contains(xenium.xy),
            )
            mask_note = "applied"
        else:
            mask_note = "MERSCOPE coordinates not in the fixed frame"
    kinds_m = _compositions(merscope, include_setc=True)
    kinds_x = _compositions(xenium, include_setc=True)
    jsd_rows, comp_rows = [], []
    for region, selection in regions.items():
        assert selection is not None
        keep_m, keep_x = selection
        codes_m = tile_codes(merscope.xy[keep_m], tile_um)
        codes_x = tile_codes(xenium.xy[keep_x], tile_um)
        for kind in kinds_m:
            if kind not in kinds_x:
                continue
            matrix_m, matrix_x = kinds_m[kind][keep_m], kinds_x[kind][keep_x]
            result = block_bootstrap_jsd(
                tile_sums(matrix_m, codes_m),
                tile_sums(matrix_x, codes_x),
                n_reps=n_reps,
                seed=seed,
            )
            shares_m, shares_x = (
                composition_shares(matrix_m),
                composition_shares(matrix_x),
            )
            jsd_rows.append(
                {
                    "pair": pair,
                    "segmentation": seg,
                    "kind": kind,
                    "region": region,
                    "held_out": pair in HELD_OUT_PAIRS or seg != "proseg_hybrid",
                    "jsd": result.jsd,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_reps": result.n_reps,
                    "tile_um": tile_um,
                    "n_tiles_merscope": result.n_tiles_a,
                    "n_tiles_xenium": result.n_tiles_b,
                    "n_cells_merscope": int(keep_m.sum()),
                    "n_cells_xenium": int(keep_x.sum()),
                    "mass_merscope": float(matrix_m.sum()),
                    "mass_xenium": float(matrix_x.sum()),
                    "unallocated_merscope": shares_m["unallocated"],
                    "unallocated_xenium": shares_x["unallocated"],
                    "mask_note": mask_note if region == "shared_mask" else "",
                }
            )
            for platform, shares, matrix in (
                ("MERSCOPE", shares_m, matrix_m),
                ("XENIUM", shares_x, matrix_x),
            ):
                renormalised = matrix[:, : len(HUMAN_BROAD_CLASSES)].sum(axis=0)
                total = renormalised.sum()
                comp_rows.append(
                    {
                        "pair": pair,
                        "segmentation": seg,
                        "platform": platform,
                        "kind": kind,
                        "region": region,
                        **{
                            f"share_{column}": shares[column]
                            for column in COMPOSITION_COLUMNS
                        },
                        **{
                            f"share7_{cls}": (
                                float(value / total) if total else math.nan
                            )
                            for cls, value in zip(
                                HUMAN_BROAD_CLASSES, renormalised, strict=True
                            )
                        },
                    }
                )
    if "shared_mask" not in regions:
        logger.warning("%s %s: shared mask not used (%s)", pair, seg, mask_note)
    return jsd_rows, comp_rows


def referee_rows(sample: Sample, pair: str, seg: str) -> list[dict[str, Any]]:
    """Return the E1 marker-referee rows of one sample."""
    labels = sample.labels
    v1 = sample.rules["seaad"]
    whb = argmax_broad_names(labels)
    new = np.where(v1.broad_confident, v1.broad_name, None)
    comparisons = {
        ("WHB argmax", "SEA-AD argmax"): (whb, sample.sea["broad"].to_numpy(object)),
        ("new confident (v1 shadow)", "legacy broad_class"): (new, sample.legacy_broad),
        ("WHB argmax", "legacy broad_class"): (whb, sample.legacy_broad),
        ("new confident (v1 shadow)", "SEA-AD argmax"): (
            new,
            sample.sea["broad"].to_numpy(object),
        ),
    }
    rows = []
    for (first, second), (a, b) in comparisons.items():
        result = marker_referee(sample.scores, a, b)
        rows.append(
            {
                "pair": pair,
                "segmentation": seg,
                "sample_id": sample.sample_id,
                "first": first,
                "second": second,
                "n_cells": result.n_cells,
                "n_disputes": result.n_disputes,
                "dispute_share": result.n_disputes / max(result.n_cells, 1),
                "first_wins": result.first_wins,
                "second_wins": result.second_wins,
                "undecided": result.undecided,
                "consensus_top_marker_ok": result.consensus_top_marker_ok,
                "top_disputes": json.dumps(result.top_disputes),
            }
        )
    return rows


def second_vote_rows(
    sample: Sample, pair: str, native: pd.DataFrame, e2_coverage: pd.DataFrame
) -> dict[str, Any]:
    """Coverage under no vote / SEA-AD / E2's LL rule on the E2 native cells."""
    cells = native[native["ds"] == sample.sample_id]
    ll = cells["ll_broad"].astype(object)
    ll.index = ll.index.astype(str)
    shared = sample.labels.index.intersection(ll.index)
    labels = sample.labels.loc[shared]
    inputs = rule_inputs_from_provisional(labels, sample.sea, ll)
    row: dict[str, Any] = {
        "pair": pair,
        "sample_id": sample.sample_id,
        "held_out": pair in HELD_OUT_PAIRS,
        "n_e2_native": len(cells),
        "n_shared": len(shared),
        "share_below60_shared": float((labels["total_counts"] < 60).mean()),
    }
    for vote in ("none", "seaad_from60", "seaad", "seaad_ll_below60", "ll_below60"):
        result = evaluate_human_rules(
            inputs,
            platform=sample.platform,
            second_vote=vote,  # type: ignore[arg-type]
        )
        row[f"broad_cov_{vote}"] = float(result.broad_confident.mean())
    base = row["broad_cov_seaad_from60"]
    row["cost_sea_below60"] = base - row["broad_cov_seaad"]
    row["cost_ll_below60"] = base - row["broad_cov_seaad_ll_below60"]
    # OD-B13: coverage the below-60 SEA-AD rule removes vs the LL rule.
    row["ll_minus_sea"] = row["broad_cov_seaad_ll_below60"] - row["broad_cov_seaad"]
    row["e2_ll_rule_minus_sea"] = row["broad_cov_ll_below60"] - row["broad_cov_seaad"]
    row["od_b13_trigger"] = bool(
        max(row["ll_minus_sea"], row["e2_ll_rule_minus_sea"]) > OD_B13_TRIGGER
    )
    whole = float(sample.rules["seaad"].broad_confident.mean())
    row["prod_broad_cov_seaad_whole"] = whole
    if sample.sample_id in e2_coverage.index:
        e2 = e2_coverage.loc[sample.sample_id]
        row["e2_broad_cov_table"] = float(e2["broad_cov_table"])
        row["e2_broad_cov_table_ll_rule"] = float(e2["broad_cov_table_withLLagree"])
        row["e2_published_ll_minus_prod_sea_whole"] = (
            row["e2_broad_cov_table_ll_rule"] - whole
        )
    return row


def _pilot_paths(evidence: Path, sample_id: str) -> dict[str, Path]:
    pair = sample_id.split("_")[0]
    if pair in ("P7513", "P1212"):
        return {
            "whb": evidence / "research/pilot/whb_region" / f"{sample_id}.panel.json",
            "seaad": evidence / "research/pilot/seaad_mr" / f"{sample_id}.json",
        }
    return {
        "whb": evidence / "exp/E2/mmc" / f"WHBF__{sample_id}_sub30k.json",
        "seaad": evidence / "exp/E2/mmc" / f"SEAAD__{sample_id}_sub30k.json",
    }


def exit_check_rows(run_dir: Path, evidence: Path, pair: str) -> list[dict[str, Any]]:
    """Production vs pilot-configuration calls on shared cells (M3 exit)."""
    rows = []
    for platform in PLATFORMS:
        sample_id = f"{pair}_{platform}"
        for key, path in _pilot_paths(evidence, sample_id).items():
            if not path.is_file():
                logger.warning("no pilot output %s", path)
                continue
            level = WHB_LEVEL if key == "whb" else SEA_LEVEL
            run_id = "whb_frontal_supc_clus" if key == "whb" else "seaad_mr_panel"
            pilot = level_frame(parse_extended_json_tidy(path), level)
            tidy, meta = read_tidy_parquet(
                run_dir / platform.lower() / f"{sample_id}_mmc_{run_id}.parquet"
            )
            production = level_frame(tidy, level)
            shared = production.index.intersection(pilot.index)
            same = (
                production.loc[shared, "assignment"].astype(str).to_numpy()
                == pilot.loc[shared, "assignment"].astype(str).to_numpy()
            )
            bp = production.loc[shared, "bp"].to_numpy(np.float64)
            for subset, mask in (
                ("all", np.ones(len(shared), bool)),
                (f"prod_bp>={PILOT_BP}", bp >= PILOT_BP),
            ):
                rows.append(
                    {
                        "pair": pair,
                        "sample_id": sample_id,
                        "reference": key,
                        "level": level,
                        "pilot": str(path.relative_to(evidence)),
                        "subset": subset,
                        "n_shared": len(shared),
                        "n": int(mask.sum()),
                        "agreement": float(same[mask].mean())
                        if mask.any()
                        else math.nan,
                        "prod_n_query_genes": meta.get("n_query_genes"),
                    }
                )
    return rows


def runtime_rows(run_dir: Path, pair: str, seg: str) -> list[dict[str, Any]]:
    """MAP wall time and memory of one pair x segmentation."""
    manifest = load_map_manifest(run_dir / "map_manifest.json")
    rows = [
        {
            "pair": pair,
            "segmentation": seg,
            "sample_id": "ALL",
            "run_id": "map_total",
            "wall_s": manifest.wall_time_s,
        }
    ]
    for sample_id, record in manifest.samples.items():
        for run_id, run in record.runs.items():
            rows.append(
                {
                    "pair": pair,
                    "segmentation": seg,
                    "sample_id": sample_id,
                    "run_id": run_id,
                    "build_hash": run.build_hash[:16],
                    "panel_hash": run.panel_hash[:16],
                    "n_cells": run.n_cells,
                    "n_query_genes": run.n_query_genes,
                    "n_missing_panel_genes": run.n_missing_panel_genes,
                    "lookup_restricted": run.lookup_restricted,
                    "wall_s": run.wall_s,
                    "processor_s_per_1k": run.wall_s
                    * manifest.n_processors
                    / run.n_cells
                    * 1000,
                    "peak_rss_gb": run.peak_rss_gb,
                    "reused": run.reused,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Compute the baselines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default=",".join(SEGMENTATIONS))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--tile-um", type=float, default=TILE_UM)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    evidence = args.evidence_root
    qc = pd.read_csv(evidence / "research/lowcount/qc_summary.csv")
    n_segmented = {
        (str(row.sample), str(row.seg), str(row.platform).upper()): int(row.n_cells)
        for row in qc.itertuples(index=False)
    }
    native = pd.read_csv(evidence / "exp/E2/out/cells_native.csv.gz", index_col=0)
    e2_coverage = pd.read_csv(
        evidence / "exp/E2/out/final_coverage_and_gate.csv", index_col=0
    )
    metric_rows, jsd_rows, comp_rows, ref_rows = [], [], [], []
    vote_rows, exit_rows, time_rows = [], [], []
    for seg in [item for item in args.segmentations.split(",") if item]:
        for pair in [item for item in args.pairs.split(",") if item]:
            run_dir = args.runs_root / pair / seg
            if not (run_dir / "map_manifest.json").is_file():
                logger.warning("no MAP outputs for %s %s", pair, seg)
                continue
            logger.info("%s %s", pair, seg)
            samples = {
                platform: load_sample(
                    run_dir,
                    args.results_root,
                    pair,
                    seg,
                    platform,
                    n_segmented.get((pair, seg, platform)),
                )
                for platform in PLATFORMS
            }
            for sample in samples.values():
                metric_rows.append(sample_metrics(sample, pair, seg))
                ref_rows.extend(referee_rows(sample, pair, seg))
                if seg == "proseg_hybrid":
                    vote_rows.append(
                        second_vote_rows(sample, pair, native, e2_coverage)
                    )
            jsd, comp = pair_jsd(
                samples,
                pair,
                seg,
                args.results_root,
                n_reps=args.n_bootstrap,
                tile_um=args.tile_um,
                seed=args.seed,
            )
            jsd_rows.extend(jsd)
            comp_rows.extend(comp)
            time_rows.extend(runtime_rows(run_dir, pair, seg))
            if seg == "proseg_hybrid":
                exit_rows.extend(exit_check_rows(run_dir, evidence, pair))
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sample_metrics.csv": metric_rows,
        "jsd.csv": jsd_rows,
        "compositions.csv": comp_rows,
        "referee.csv": ref_rows,
        "second_vote_cost.csv": vote_rows,
        "exit_checks.csv": exit_rows,
        "runtime.csv": time_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.out / name, index=False)
        logger.info("wrote %s (%d rows)", args.out / name, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
