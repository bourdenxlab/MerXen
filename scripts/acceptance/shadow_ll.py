#!/usr/bin/env python
"""M3 LL value test and OD-B13 trigger (plan §12 M3 item 6, §5.2, §5.3).

Types every table cell of the human MAP outputs (``<runs-root>/<pair>/<seg>/``)
with the likelihood typer LL (vii) (``merxen.annotation.likelihood``: the
spill-over mixture model with per-gene platform factors, on the WHB frontal
bundle's cluster profiles), then scores:

- **OD-B8 (does LL join v1.1?)**: confident broad coverage under the v1.1
  rule (below 60 counts SEA-AD **or** LL agrees) minus v1 (SEA-AD must
  agree), and the marker-referee outcomes of the confident labels with and
  without LL: new-vs-legacy disputes (H10) and the canonical-marker
  plausibility of confident labels. LL joins v1.1 if, on a development
  dataset, coverage rises by > 0.05 or one of these referee outcomes
  improves by > 0.02. In v1.1 LL only adds confident cells below 60 counts
  (§5.3) and never renames one, so these are the outcomes it can move.
  For information: WHB-vs-LL disputes (E1's referee) and WHB-vs-SEA-AD
  disputes resolved with LL as a tie-breaker (not a v1.1 behaviour; LL and
  the referee read the same marker counts, so it favours LL by
  construction).
- **OD-B13 (promote LL before M8?)**: coverage the below-60 SEA-AD rule
  removes vs the LL rules (E2's rule with LL (vii) alone, and v1 with LL as
  the below-60 vote); trigger > 0.08 on any dataset.

Both factor variants are run: capped at +/- 2 log2 (the M10 specification)
and uncapped (E1's (vii)). Writes ``ll_sample_metrics.csv``,
``ll_referee.csv``, ``ll_factors.csv``, ``ll_decision.csv`` and per-sample
calls ``calls/<pair>_<seg>_<sid>_<variant>.parquet`` to ``--out``.

Usage::

    python scripts/acceptance/shadow_ll.py --runs-root <shadow>/baselines/runs \\
        --results-root <results> --evidence-root <evidence> \\
        --bundle <store>/whb_frontal_supc_clus/<hash> --out <shadow>/ll
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation.likelihood import (
    LikelihoodReference,
    reference_from_profiles,
    run_ll_vii,
)
from merxen.annotation.panel import AnnotationPanel
from merxen.annotation.reference import TaxonomyTreeView
from merxen.annotation.shadow import (
    E1_REFEREE_MARKERS,
    argmax_broad_names,
    evaluate_human_rules,
    marker_referee,
    published_queries,
    rule_inputs_from_provisional,
)
from merxen.annotation.vocab import HUMAN_BROAD_CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_baselines import (  # noqa: E402
    HELD_OUT_PAIRS,
    PAIRS,
    PLATFORMS,
    Sample,
    _clustered_path,
    load_sample,
)

logger = logging.getLogger("shadow_ll")

WHB_LEAF = "CCN202210140_CLUS"
WHB_PARENT = "CCN202210140_SUPC"
VARIANTS = {"capped": 2.0, "uncapped": None}
RULES = (
    "none",
    "seaad_from60",
    "seaad",
    "seaad_or_ll",
    "seaad_ll_below60",
    "ll_below60",
)
DEVELOPMENT = ("P7513", "P1212")
OD_B8_COVERAGE = 0.05
OD_B8_REFEREE = 0.02
OD_B13_TRIGGER = 0.08
FALLBACK = (
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/"
    "WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
)


class ReferenceCache:
    """LL references of one bundle, built once per query gene set."""

    def __init__(self, bundle: Path) -> None:
        """Read the bundle's profiles, mapping tree and vocab snapshot."""
        self.profiles = pd.read_parquet(bundle / "profiles.parquet")
        self.tree = TaxonomyTreeView.from_tree_dict(
            json.loads((bundle / "mapping_tree.json").read_text(encoding="utf-8"))
        )
        vocab = pd.read_csv(
            bundle / "vocab_snapshot.csv", dtype=str, keep_default_na=False
        )
        parents = vocab[vocab["level"] == WHB_PARENT]
        self.broad_of = dict(zip(parents["node"], parents["broad_class"], strict=True))
        self.profile_genes = set(self.profiles["gene_id"].astype(str))
        self._cache: dict[tuple[str, ...], LikelihoodReference] = {}

    def get(self, genes: list[str]) -> tuple[LikelihoodReference, list[int]]:
        """Return the reference on the profiled query genes and their columns."""
        columns = [i for i, gene in enumerate(genes) if gene in self.profile_genes]
        key = tuple(genes[i] for i in columns)
        if key not in self._cache:
            self._cache[key] = reference_from_profiles(
                self.profiles,
                self.tree,
                leaf_level=WHB_LEAF,
                parent_level=WHB_PARENT,
                broad_of_parent=self.broad_of,
                gene_ids=list(key),
            )
        return self._cache[key], columns


def _plausibility(sample: Sample, labels: np.ndarray, confident: np.ndarray) -> float:
    """Share of confident cells whose top canonical-marker class is the label."""
    scores = sample.scores
    has_marker = scores.sum(axis=1) > 0
    keep = confident & has_marker
    if not keep.any():
        return math.nan
    classes = list(E1_REFEREE_MARKERS)
    top = np.asarray(classes, dtype=object)[scores.argmax(axis=1)]
    return float((top[keep] == labels[keep]).mean())


def _resolved_referee(
    sample: Sample, whb: np.ndarray, sea: np.ndarray, ll: np.ndarray
) -> dict[str, Any]:
    """Referee WHB-vs-SEA disputes with and without LL as the tie-breaker."""
    in_classes = np.isin(whb, HUMAN_BROAD_CLASSES) & np.isin(sea, HUMAN_BROAD_CLASSES)
    disputed = in_classes & (whb != sea)
    classes = list(E1_REFEREE_MARKERS)
    column = {cls: i for i, cls in enumerate(classes)}
    rows = np.flatnonzero(disputed)
    score_whb = np.array([sample.scores[r, column[str(whb[r])]] for r in rows])
    score_sea = np.array([sample.scores[r, column[str(sea[r])]] for r in rows])
    ll_sides_sea = ll[rows] == sea[rows]
    ll_sides_whb = ll[rows] == whb[rows]
    chosen = np.where(ll_sides_sea, score_sea, score_whb)
    other = np.where(ll_sides_sea, score_whb, score_sea)
    n = len(rows)
    return {
        "n_whb_sea_disputes": n,
        "ll_sides_whb": float(ll_sides_whb.mean()) if n else math.nan,
        "ll_sides_sea": float(ll_sides_sea.mean()) if n else math.nan,
        "ll_sides_neither": float((~ll_sides_whb & ~ll_sides_sea).mean())
        if n
        else math.nan,
        "whb_policy_wins": float((score_whb > score_sea).mean()) if n else math.nan,
        "ll_tiebreak_wins": float((chosen > other).mean()) if n else math.nan,
        "ll_sides_sea_markers_side_sea": float(
            (score_sea[ll_sides_sea] > score_whb[ll_sides_sea]).mean()
        )
        if ll_sides_sea.any()
        else math.nan,
    }


def score_sample(
    sample: Sample,
    calls: pd.DataFrame,
    *,
    pair: str,
    seg: str,
    variant: str,
    n_segmented: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Coverage under the rule variants and the referee outcomes of one sample."""
    labels = sample.labels
    ll = calls["ll_broad_name"].reindex(labels.index)
    inputs = rule_inputs_from_provisional(labels, sample.sea, ll, ll_scheme="broad7")
    results = {
        rule: evaluate_human_rules(
            inputs,
            platform=sample.platform,
            second_vote=rule,  # type: ignore[arg-type]
            n_segmented=n_segmented,
        )
        for rule in RULES
    }
    counts = labels["total_counts"].to_numpy(np.float64)
    below = counts < 60
    whb = argmax_broad_names(labels)
    sea = sample.sea["broad"].to_numpy(object)
    ll_values = ll.to_numpy(object)
    row: dict[str, Any] = {
        "pair": pair,
        "segmentation": seg,
        "sample_id": sample.sample_id,
        "platform": sample.platform,
        "variant": variant,
        "held_out": pair in HELD_OUT_PAIRS or seg != "proseg_hybrid",
        "n_table_cells": len(labels),
        "share_below60": float(below.mean()),
        "ll_whb_agree_all": float((ll_values == whb).mean()),
        "ll_whb_agree_below60": float((ll_values[below] == whb[below]).mean())
        if below.any()
        else math.nan,
        "ll_sea_agree_all": float((ll_values == sea).mean()),
        "ll_mixed_unknown_share": float((ll_values == "Mixed/Unknown").mean()),
        "ll_mix_fraction_mean": float(calls["ll_mix_fraction"].mean()),
    }
    for rule, result in results.items():
        row[f"broad_cov_{rule}"] = float(result.broad_confident.mean())
        row[f"supercluster_cov_{rule}"] = float(result.supercluster_confident.mean())
    row["od_b8_coverage_gain"] = row["broad_cov_seaad_or_ll"] - row["broad_cov_seaad"]
    row["cost_sea_below60"] = row["broad_cov_seaad_from60"] - row["broad_cov_seaad"]
    row["ll_vote_minus_sea"] = (
        row["broad_cov_seaad_ll_below60"] - row["broad_cov_seaad"]
    )
    row["e2_ll_rule_minus_sea"] = row["broad_cov_ll_below60"] - row["broad_cov_seaad"]
    row["od_b13_trigger"] = bool(
        max(row["ll_vote_minus_sea"], row["e2_ll_rule_minus_sea"]) > OD_B13_TRIGGER
    )
    v1, v11 = results["seaad"], results["seaad_or_ll"]
    referee_rows = []
    for rule, result in (("v1", v1), ("v1.1", v11)):
        new = np.where(result.broad_confident, result.broad_name, None)
        legacy = marker_referee(sample.scores, new, sample.legacy_broad)
        row[f"h10_new_wins_{rule}"] = legacy.first_wins
        row[f"h10_disputes_{rule}"] = legacy.n_disputes
        row[f"plausibility_{rule}"] = _plausibility(
            sample, result.broad_name, result.broad_confident
        )
    gained = v11.broad_confident & ~v1.broad_confident
    row["n_gained_v11"] = int(gained.sum())
    row["plausibility_gained_v11"] = _plausibility(sample, v11.broad_name, gained)
    row["h10_delta"] = row["h10_new_wins_v1.1"] - row["h10_new_wins_v1"]
    row["plausibility_delta"] = row["plausibility_v1.1"] - row["plausibility_v1"]
    resolved = _resolved_referee(sample, whb, sea, ll_values)
    row.update(resolved)
    row["tiebreak_delta"] = resolved["ll_tiebreak_wins"] - resolved["whb_policy_wins"]
    whb_vs_ll = marker_referee(sample.scores, whb, ll_values)
    row["whb_vs_ll_disputes"] = whb_vs_ll.n_disputes
    row["whb_vs_ll_whb_wins"] = whb_vs_ll.first_wins
    row["whb_vs_ll_ll_wins"] = whb_vs_ll.second_wins
    referee_rows.append(
        {
            "pair": pair,
            "segmentation": seg,
            "sample_id": sample.sample_id,
            "variant": variant,
            "comparison": "WHB argmax vs LL (vii)",
            "n_disputes": whb_vs_ll.n_disputes,
            "first_wins": whb_vs_ll.first_wins,
            "second_wins": whb_vs_ll.second_wins,
            "undecided": whb_vs_ll.undecided,
            "top_disputes": json.dumps(whb_vs_ll.top_disputes),
        }
    )
    return row, referee_rows


def decision_rows(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """The OD-B8 and OD-B13 verdicts from the per-sample metrics."""
    rows = []
    base = metrics[metrics["segmentation"] == "proseg_hybrid"]
    for variant, frame in base.groupby("variant"):
        dev = frame[frame["pair"].isin(DEVELOPMENT)]
        referee_gain = dev[["h10_delta", "plausibility_delta"]].max(axis=None)
        coverage_gain = dev["od_b8_coverage_gain"].max()
        rows.append(
            {
                "variant": variant,
                "decision": "OD-B8",
                "max_coverage_gain_dev": float(coverage_gain),
                "max_referee_gain_dev": float(referee_gain),
                "max_coverage_gain_all": float(frame["od_b8_coverage_gain"].max()),
                "max_referee_gain_all": float(
                    frame[["h10_delta", "plausibility_delta"]].max(axis=None)
                ),
                "max_tiebreak_delta_info": float(frame["tiebreak_delta"].max()),
                "passes": bool(
                    coverage_gain > OD_B8_COVERAGE or referee_gain > OD_B8_REFEREE
                ),
            }
        )
        rows.append(
            {
                "variant": variant,
                "decision": "OD-B13",
                "max_cost_sea_below60": float(frame["cost_sea_below60"].max()),
                "max_ll_vote_minus_sea": float(frame["ll_vote_minus_sea"].max()),
                "max_e2_ll_rule_minus_sea": float(frame["e2_ll_rule_minus_sea"].max()),
                "passes": bool(frame["od_b13_trigger"].any()),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run LL (vii) on the MAP outputs and score OD-B8 / OD-B13."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--segmentations", default="proseg_hybrid,reseg")
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
    cache = ReferenceCache(args.bundle)
    (args.out / "calls").mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    referee_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    for seg in [s for s in args.segmentations.split(",") if s]:
        for pair in [p for p in args.pairs.split(",") if p]:
            run_dir = args.runs_root / pair / seg
            if not (run_dir / "map_manifest.json").is_file():
                logger.warning("no MAP outputs for %s %s", pair, seg)
                continue
            panel = AnnotationPanel.model_validate_json(
                (run_dir / "panel" / "panel_genes.json").read_text(encoding="utf-8")
            )
            queries = published_queries(
                {
                    p: _clustered_path(args.results_root, pair, seg, p)
                    for p in PLATFORMS
                },
                panel,
                species="human",
                gene_id_fallback_csv=args.gene_id_fallback_csv,
            )
            for platform in PLATFORMS:
                sample = load_sample(
                    run_dir,
                    args.results_root,
                    pair,
                    seg,
                    platform,
                    n_segmented.get((pair, seg, platform)),
                )
                query = queries[platform]
                reference, columns = cache.get(list(query.gene_ids))
                counts = query.counts[:, columns]
                for variant, cap in VARIANTS.items():
                    result = run_ll_vii(
                        counts, reference, cell_ids=query.cell_ids, cap_log2=cap
                    )
                    calls = result.calls
                    calls.to_parquet(
                        args.out
                        / "calls"
                        / f"{pair}_{seg}_{sample.sample_id}_{variant}.parquet"
                    )
                    row, referee = score_sample(
                        sample,
                        calls,
                        pair=pair,
                        seg=seg,
                        variant=variant,
                        n_segmented=n_segmented.get((pair, seg, platform)),
                    )
                    row["n_factors_beyond_2log2"] = int(
                        (np.abs(result.uncapped_log2_factors) > 2).sum()
                    )
                    row["n_genes"] = len(reference.gene_ids)
                    row["n_leaves"] = reference.n_leaves
                    metric_rows.append(row)
                    referee_rows.extend(referee)
                    if variant == "capped":
                        factor_rows.extend(
                            {
                                "pair": pair,
                                "segmentation": seg,
                                "sample_id": sample.sample_id,
                                "gene_id": gene,
                                "log2_factor_uncapped": float(value),
                            }
                            for gene, value in zip(
                                reference.gene_ids,
                                result.uncapped_log2_factors,
                                strict=True,
                            )
                        )
                    logger.info(
                        "%s %s %s: v1 %.3f v1.1 %.3f E2-LL %.3f",
                        pair,
                        seg,
                        variant,
                        row["broad_cov_seaad"],
                        row["broad_cov_seaad_or_ll"],
                        row["broad_cov_ll_below60"],
                    )
    metrics = pd.DataFrame(metric_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "ll_sample_metrics.csv", index=False)
    pd.DataFrame(referee_rows).to_csv(args.out / "ll_referee.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(args.out / "ll_factors.csv", index=False)
    pd.DataFrame(decision_rows(metrics)).to_csv(
        args.out / "ll_decision.csv", index=False
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
