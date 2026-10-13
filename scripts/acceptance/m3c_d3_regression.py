#!/usr/bin/env python
"""M3c pre-registration §21 (ii-b): replay phase 1's D3 on the public 5K section.

Profile mode (plan §8.3 v7.5) with D3's settings — recipe name
``R3_measured_realdepth``, version 1, seed 0, table rule ``all_measured``,
uniform 25% spill, the public 5K depth-profile asset, ``M_c = min(max(2 n_c,
2000), 8 n_c)`` — on the phase-1 test set (``wmb_selfmap_testset``
``6c94b240``, 13,059 cells, not topped up), mapped with the engine of bundle
``c30f7eab`` and the production mapping configuration on 6 processes, and
tabulated as phase 1's ``analyse.py`` (class rule bp >= 0.9 and >= 20 counts;
subclass rule bp >= 0.8, >= 50 counts and the class rule; simulated cells
weighted to the real called composition, class weight x within-class subclass
weight trimmed at 10x the class median). Pass: for each of the 14 major
classes and ALL, at class and subclass level, ``|M3c - D3| <= 0.01``. Reported
beside it (no pass / fail): the same predictions with the production table
rule ``restricted``.

Read-only on the bundles, the test set and the evidence; writes only into
``--out-dir``.

Usage::

    python scripts/acceptance/m3c_d3_regression.py \\
        --out-dir $A/m3c/d3_regression \\
        --scratch-dir $A/m3c/d3_regression_scratch --n-processors 6

where ``$A`` is ``/srv/storage/MerXen/annotation_dev/evidence_20260926``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation import reference as ref
from merxen.annotation import resolvability as res
from merxen.annotation import sim_inputs as si
from merxen.annotation import simulate
from merxen.annotation.config import (
    KNOWN_REFERENCES,
    AnnotationConfig,
    AnnotationReferenceSpec,
)
from merxen.annotation.mapmycells_engine import MmcBundle
from merxen.annotation.memory import ProcessTreeSampler

logger = logging.getLogger("m3c_d3_regression")

EVIDENCE = Path("/srv/storage/MerXen/annotation_dev/evidence_20260926")
BUNDLE = Path(
    "/srv/storage/MerXen/annotation_references_large/wmb_panel/"
    "c30f7eab413cba9e5e142c337d01140750f28967adff2100903f04893eced4f8"
)
TEST_SET = Path(
    "/srv/storage/MerXen/annotation_references_large/wmb_selfmap_testset/"
    "6c94b24030d2a0e4376ce4a7171b02d900cd87ad63a9322af4815ce54da3b21f"
)
CONFIG = EVIDENCE / "m3b/simulate/configs/mouse.annotation_config.json"
DATA = Path(__file__).resolve().parents[2] / "tests/test_annotation/data"
KEY_NAME = "R3_measured_realdepth"
TOLERANCE = 0.01
METRICS = ("cov_class_rule73", "cov_subclass_rule73")


def replay(
    rule: str,
    *,
    test: res.HeldOutCells,
    engine: MmcBundle,
    config: AnnotationConfig,
    spec: AnnotationReferenceSpec,
    summary: dict[str, Any],
    composition: pd.DataFrame,
    out_dir: Path,
    scratch_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate, map and tabulate one table rule; return predictions and record."""
    started = time.monotonic()
    vocab = engine.vocab()
    specs = ref.level_specs_for("wmb_panel", engine, config)
    truth = simulate.profile_truth_classes(test, specs, "mouse", vocab)
    table = si.efficiency_table(si.get_asset(si.EFFICIENCY_MOUSE_PRIME))
    efficiency = si.r3_efficiency(test.genes, table, rule=rule, seed=0)
    profile = si.profile_from_asset(si.get_asset(si.PROFILE_MOUSE_PRIME_FF))
    recipe = res.SimulationRecipe(
        KEY_NAME,
        1,
        None,
        simulate.PROFILE_SPILL_FRACTION,
        0,
        efficiency_source="measured",
        efficiency_table=si.EFFICIENCY_MOUSE_PRIME,
        table_rule=rule,
        residual_sd_log2=si.R3_RESIDUAL_SD_LOG2,
    )
    grid = [int(value) for value in summary["depth_grid"]]
    query = simulate.simulate_on_profile(
        test, profile, truth, recipe, grid, efficiency=efficiency.efficiency
    )
    simulated_s = time.monotonic() - started
    runs: list[dict[str, Any]] = []
    map_fn = ref.mmc_map_function(
        engine,
        spec=spec,
        config=config,
        scratch_dir=scratch_dir / rule,
        log_dir=out_dir / "logs" / rule,
        runs=runs,
    )
    with ProcessTreeSampler() as sampler:
        tidy = map_fn(query, f"d3_replay_{rule}", 0)
    cells = res.level_cells(tidy, query, test, specs, seed=0)
    for rule_fn in ref.cells_rules_for("wmb_panel", config):
        cells = rule_fn(cells)
    cells = res.as_stored(cells)
    cells.to_parquet(out_dir / f"cells_{rule}.parquet", index=False)
    query.obs.reset_index().to_parquet(out_dir / f"sims_{rule}.parquet", index=False)
    names = simulate._node_names(vocab)
    class_of = simulate._subclass_class_map(vocab)
    frame = simulate.profile_cell_table(
        cells, species="mouse", summary=summary, names=names
    )
    counts = composition.groupby("subclass")["n_cells"].sum()
    share = counts / counts.sum()
    weights = simulate.composition_weights_to_real(frame["truth_leaf"], share, class_of)
    predictions = simulate.per_class_predictions(
        frame, weights, simulate.profile_metrics(frame)
    )
    predictions.to_csv(out_dir / f"predictions_{rule}.csv", index=False)
    record = {
        "rule": rule,
        "efficiency": efficiency.summary(),
        "n_simulated": int(len(query.obs)),
        "truncated_share": float(query.obs["truncated"].mean()),
        "profile_sources": dict(profile.used),
        "simulate_s": round(simulated_s, 1),
        "mapping_runs": runs,
        "mapping_process_tree_peak": sampler.peak().to_json(),
        "wall_s": round(time.monotonic() - started, 1),
    }
    return predictions, record


def compare(predictions: pd.DataFrame, fixture: pd.DataFrame) -> pd.DataFrame:
    """Return M3c - D3 per major class and ALL at class and subclass level."""
    ours = predictions.set_index("class")
    theirs = fixture.set_index("class")
    rows = []
    for cls in theirs.index:
        for metric in METRICS:
            d3 = float(theirs.loc[cls, metric])
            m3c = float(ours.loc[cls, metric]) if cls in ours.index else float("nan")
            rows.append(
                {
                    "class": cls,
                    "metric": metric,
                    "d3": d3,
                    "m3c": m3c,
                    "delta": m3c - d3,
                    "pass": bool(np.isfinite(m3c) and abs(m3c - d3) <= TOLERANCE),
                    "m3c_kish_n": float(ours.loc[cls, "kish_n"])
                    if cls in ours.index
                    else None,
                    "d3_kish_n": float(theirs.loc[cls, "kish_n"]),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay; exit 0 when (ii-b) passes, 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--n-processors", type=int, default=6)
    parser.add_argument(
        "--rules", default="all_measured,restricted", help="Comma-separated rules."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    ref.set_prep_resources(n_processors=args.n_processors)
    started = time.monotonic()
    config = AnnotationConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    spec = AnnotationReferenceSpec(
        reference_id="wmb_panel", **KNOWN_REFERENCES["wmb_panel"]
    )
    summary = json.loads((BUNDLE / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    engine = MmcBundle.from_dir(BUNDLE)
    test = res.load_test_cells(TEST_SET)
    composition = pd.read_csv(DATA / "m3c_real_composition_mouse5k.csv")
    fixture = pd.read_csv(DATA / "m3c_d3_predictions_mouse5k.csv")
    report: dict[str, Any] = {
        "test": "M3c pre-registration §21 (ii-b)",
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "bundle": str(BUNDLE),
        "test_set": str(TEST_SET),
        "n_test_cells": int(len(test.obs)),
        "n_processors": args.n_processors,
        "tolerance": TOLERANCE,
        "runs": {},
    }
    lines = [
        "M3c pre-registration §21 (ii-b): replay of phase 1's D3 on the public "
        "5K section",
        f"bundle {BUNDLE.name[:16]}, test set {TEST_SET.name[:16]} "
        f"({len(test.obs)} cells), {args.n_processors} processes",
        "",
    ]
    exit_code = 0
    for rule in [item.strip() for item in args.rules.split(",") if item.strip()]:
        predictions, record = replay(
            rule,
            test=test,
            engine=engine,
            config=config,
            spec=spec,
            summary=summary,
            composition=composition,
            out_dir=args.out_dir,
            scratch_dir=args.scratch_dir,
        )
        table = compare(predictions, fixture)
        table.to_csv(args.out_dir / f"comparison_{rule}.csv", index=False)
        passes = bool(table["pass"].all())
        record["passes"] = passes if rule == "all_measured" else None
        record["max_abs_delta"] = float(table["delta"].abs().max())
        report["runs"][rule] = record
        verdict = (
            ("PASS" if passes else "FAIL")
            if rule == "all_measured"
            else "reported (no pass / fail)"
        )
        if rule == "all_measured" and not passes:
            exit_code = 1
        lines.append(
            f"== table rule {rule}: {verdict}; max |M3c - D3| "
            f"{record['max_abs_delta']:.4f} (tolerance {TOLERANCE}); "
            f"{record['n_simulated']} simulated cells, truncated "
            f"{record['truncated_share']:.4f}; wall {record['wall_s']} s"
        )
        lines.append(f"   efficiency: {json.dumps(record['efficiency'])}")
        lines.append(
            f"   {'class':22s} {'metric':20s} {'D3':>7s} {'M3c':>7s} {'delta':>8s}"
        )
        for row in table.to_dict("records"):
            lines.append(
                f"   {str(row['class'])[:22]:22s} {row['metric']:20s} "
                f"{row['d3']:7.4f} {row['m3c']:7.4f} {row['delta']:+8.4f}"
                + ("" if row["pass"] else "  <-- outside .01")
            )
        lines.append("")
    report["wall_s"] = round(time.monotonic() - started, 1)
    (args.out_dir / "D3_REGRESSION.json").write_text(
        json.dumps(simulate._json_native(report), indent=2) + "\n"
    )
    (args.out_dir / "D3_REGRESSION.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
