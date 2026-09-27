#!/usr/bin/env python
"""M3 LL value test and OD-B13 trigger (plan §12 M3 item 6, §5.2, §5.3).

Types every table cell of the human MAP outputs (``<runs-root>/<pair>/<seg>/``)
with LL (vii)'s recipe (``merxen.annotation.likelihood``: E1
``14_ll_contam.py``'s spill-over mixture model with per-gene platform
factors) on the **WHB frontal bundle's cluster profiles** (122 clusters under
17 superclusters, sinks included), not on E1's 204-cluster whole-WHB cortex
reference (plan §3.2 ``ll_whb_ctx_profiles``). Its fidelity to E1 (vii) is
measured on E1's 30k pilot subsets (``ll_vs_e1_vii.csv``).

- **OD-B8 (does LL join v1.1?)**. The plan asks whether LL moves referee
  outcomes by > 2 points or coverage by > 5 points; it does not say how LL
  would be used. Two readings are scored, neither pre-registered (the first
  was chosen in this script at M3 C2):

  1. *v1.1 below-60 vote* (§5.3: below 60 counts SEA-AD **or** LL agrees):
     coverage v1.1 - v1, and H10 / canonical-marker plausibility of the
     confident labels. v1.1's confident cells are a subset of "v1 without
     the below-60 rule" (the same >= 60 veto and rescues), so its coverage
     gain is bounded by the below-60 SEA-AD cost (``od_b8_coverage_bound``;
     0.2-1.0 points at M3), far below 5, and H10 can move by at most
     ``h10_ceiling_gain`` (every added cell a won dispute). This reading
     cannot pass by construction and says little about LL itself.
  2. *LL as tie-breaker* of WHB-vs-SEA-AD disputes: markers side with the
     label LL picks vs WHB alone (``tiebreak_delta``). Circular: LL and the
     referee read the same marker counts.

- **OD-B13 (promote LL before M8?)**: coverage the below-60 SEA-AD rule
  removes vs the LL rules (E2's rule with LL (vii) alone, and v1 with LL as
  the below-60 vote); trigger > 0.08 on any dataset.

Both factor variants are run: capped at +/- 2 log2 (the M10 specification)
and uncapped (E1's (vii)). Writes ``ll_sample_metrics.csv``,
``ll_referee.csv``, ``ll_factors.csv``, ``ll_decision.csv``,
``ll_vs_e1_vii.csv`` and per-sample calls
``calls/<pair>_<seg>_<sid>_<variant>.parquet`` to ``--out``.

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
    add_fallback_arguments,
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
    # v1.1 confident cells are a subset of v1 without the below-60 rule.
    row["od_b8_coverage_bound"] = row["cost_sea_below60"]
    row["v11_within_bound"] = bool(
        (
            results["seaad_or_ll"].broad_confident
            & ~results["seaad_from60"].broad_confident
        ).sum()
        == 0
    )
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
    # H10 ceiling: every added cell a dispute the new label wins.
    disputes = row["h10_disputes_v1"]
    wins = row["h10_new_wins_v1"] * disputes
    row["h10_ceiling_gain"] = (
        (wins + row["n_gained_v11"]) / (disputes + row["n_gained_v11"])
        - row["h10_new_wins_v1"]
        if disputes
        else math.nan
    )
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
    """The OD-B8 readings and the OD-B13 verdict from the per-sample metrics.

    OD-B8 is reported under both readings (module docstring); neither was
    pre-registered, so the rows are inputs for the user's decision.
    """
    rows = []
    base = metrics[metrics["segmentation"] == "proseg_hybrid"]
    for variant, frame in base.groupby("variant"):
        dev = frame[frame["pair"].isin(DEVELOPMENT)]
        referee_gain = dev[["h10_delta", "plausibility_delta"]].max(axis=None)
        coverage_gain = dev["od_b8_coverage_gain"].max()
        rows.append(
            {
                "variant": variant,
                "decision": "OD-B8 (v1.1 below-60 vote)",
                "max_coverage_gain_dev": float(coverage_gain),
                "max_referee_gain_dev": float(referee_gain),
                "max_coverage_gain_all": float(frame["od_b8_coverage_gain"].max()),
                "max_referee_gain_all": float(
                    frame[["h10_delta", "plausibility_delta"]].max(axis=None)
                ),
                "max_coverage_bound_all": float(frame["od_b8_coverage_bound"].max()),
                "max_h10_ceiling_gain_all": float(frame["h10_ceiling_gain"].max()),
                "v11_within_bound_all": bool(frame["v11_within_bound"].all()),
                "passes": bool(
                    coverage_gain > OD_B8_COVERAGE or referee_gain > OD_B8_REFEREE
                ),
                "note": "cannot pass by construction: coverage gain <= below-60 "
                "SEA-AD cost",
            }
        )
        rows.append(
            {
                "variant": variant,
                "decision": "OD-B8 (LL as tie-breaker)",
                "max_referee_gain_dev": float(dev["tiebreak_delta"].max()),
                "min_referee_gain_dev": float(dev["tiebreak_delta"].min()),
                "max_referee_gain_all": float(frame["tiebreak_delta"].max()),
                "min_referee_gain_all": float(frame["tiebreak_delta"].min()),
                "passes": bool(dev["tiebreak_delta"].max() > OD_B8_REFEREE),
                "note": "circular: LL and the referee read the same marker counts",
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


def e1_fidelity_rows(
    calls_dir: Path, preds_dir: Path, *, seg: str = "proseg_hybrid"
) -> list[dict[str, Any]]:
    """Agreement of the port with E1 (vii) on E1's 30k pilot subsets.

    E1's ``LLmix_wbctx+platform`` predictions (uncapped factors, the
    204-cluster whole-WHB cortex reference) against the port's calls on the
    same cells, at broad class and supercluster.
    """
    rows = []
    for path in sorted(preds_dir.glob("LLmix_wbctx+platform__*.pred.csv")):
        sample_id = path.name.split("__", 1)[1].removesuffix(".pred.csv")
        pair = sample_id.split("_")[0]
        e1 = pd.read_csv(path, index_col=0)
        e1.index = e1.index.astype(str)
        for variant in VARIANTS:
            calls_path = calls_dir / f"{pair}_{seg}_{sample_id}_{variant}.parquet"
            if not calls_path.is_file():
                continue
            calls = pd.read_parquet(calls_path)
            calls.index = calls.index.astype(str)
            shared = e1.index.intersection(calls.index)
            port, ref = calls.loc[shared], e1.loc[shared]
            broad_port = port["ll_broad_name"].astype(str).to_numpy()
            broad_e1 = ref["pred_broad"].astype(str).to_numpy()
            super_port = port["ll_supercluster_name"].astype(str).to_numpy()
            super_e1 = ref["pred_supercluster"].astype(str).to_numpy()
            rows.append(
                {
                    "sample_id": sample_id,
                    "variant": variant,
                    "e1_variant": "LLmix_wbctx+platform (uncapped)",
                    "n_e1": len(e1),
                    "n_shared": len(shared),
                    "broad_agreement": float((broad_port == broad_e1).mean()),
                    "supercluster_agreement": float((super_port == super_e1).mean()),
                    "e1_superclusters": int(pd.Series(super_e1).nunique()),
                    "port_superclusters": int(pd.Series(super_port).nunique()),
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
    add_fallback_arguments(parser)
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
    preds_dir = args.evidence_root / "exp" / "E1" / "preds" / "real"
    pd.DataFrame(e1_fidelity_rows(args.out / "calls", preds_dir)).to_csv(
        args.out / "ll_vs_e1_vii.csv", index=False
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
