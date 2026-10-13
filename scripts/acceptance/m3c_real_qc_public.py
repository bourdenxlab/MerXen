#!/usr/bin/env python
"""M3c stage D: the real-data QC of ``real_qc`` on the public 5K mouse section.

An in-sample demonstration (reported, never pass / fail, never trust evidence):
the section is the one the stored factor table and depth profile were measured
on, so the factor re-measure should reproduce the stored table and the
coverage warning repeats the calibration of pre-registration §21 (iv). With
the M3c version-7 5K mouse bundle and the phase-1 calls:

1. ``factor_remeasure``: per-gene factors of the section's confident class
   calls (class bp >= 0.9, >= 20 counts) against the bundle's class-level
   reference profiles, compared with the stored ``member`` table on the genes
   informative in both (warning when r < 0.9);
2. ``nonneuronal_high_depth_flags`` and ``nonneuronal_depth_trend`` on the
   version-7 provisional coverage of the real cells;
3. ``coverage_vs_simulation`` (as in (iv)): the class-depth predictor
   decides each per-class warning; with ``--profile-predictions`` (an
   ``annotation-panel-simulate`` ``profile_predictions.csv``) the profile-mode
   member mean is reported beside it (D4 of 2026-09-29);
4. ``gene_complexity_check``: native genes per cell against cells simulated
   with ``R1_contam_HO@0`` (version-7 conventions) from the bundle's test set
   at the grid values 250-2000.

Read-only on the inputs; writes ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

A = Path("/srv/storage/MerXen/annotation_dev/evidence_20260926")
CALLS = A / "5k_real/mouse5k/qc/percell_calls_mouse5k.parquet"
MATRIX = Path(
    "/srv/storage/MerXen/public_validation/xenium_prime_5k_mouse_brain_ff/"
    "Xenium_Prime_Mouse_Brain_Coronal_FF_cell_feature_matrix.h5"
)
SIM_DEPTHS = (250, 350, 500, 700, 1000, 1400, 2000)


def read_counts(path: Path, cell_ids: Sequence[str]) -> tuple[Any, list[str]]:
    """Return the section's cells x Ensembl-gene counts (gene-expression features)."""
    import h5py
    from scipy import sparse

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        barcodes = [value.decode() for value in group["barcodes"][:]]
        features = group["features"]
        ids = [value.decode() for value in features["id"][:]]
        types = [value.decode() for value in features["feature_type"][:]]
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=tuple(group["shape"][:]),
        )
    keep = [index for index, kind in enumerate(types) if kind == "Gene Expression"]
    genes = [ids[index] for index in keep]
    cells = matrix[keep, :].T.tocsr()
    position = {barcode: index for index, barcode in enumerate(barcodes)}
    rows = [position[str(cell)] for cell in cell_ids]
    return cells[rows], genes


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    from merxen.annotation import real_qc, simulate
    from merxen.annotation import resolvability as res
    from merxen.annotation import sim_inputs as si
    from merxen.annotation.config import AnnotationResolvabilityConfig
    from merxen.annotation.shadow import profile_matrix

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--calls", type=Path, default=CALLS)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--profile-predictions",
        type=Path,
        default=None,
        help="profile_predictions.csv of profile mode (reported beside)",
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calls = pd.read_parquet(args.calls)
    summary = json.loads((args.bundle / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    class_depth = pd.read_parquet(args.bundle / res.CLASS_DEPTH_FILE)
    counts, genes = read_counts(args.matrix, calls["cell_id"].astype(str).tolist())
    totals = np.asarray(counts.sum(axis=1)).ravel()
    record: dict[str, Any] = {
        "n_cells": int(counts.shape[0]),
        "n_genes": len(genes),
        "totals_match_calls": bool(
            np.allclose(totals, calls["total_counts"].to_numpy(np.float64))
        ),
    }
    lines = [
        "M3c stage D: real_qc on the public 5K mouse section (in-sample demonstration;",
        "reported only, never trust evidence)",
        f"bundle {args.bundle}; {counts.shape[0]} cells x {len(genes)} genes; totals "
        f"equal the phase-1 calls: {record['totals_match_calls']}",
        "",
    ]
    # 1. Factor re-measure.
    profiles = pd.read_parquet(args.bundle / "profiles.parquet")
    matrix = profile_matrix(profiles, "CCN20230722_CLAS", genes, key="node_name")
    member = si.member_table_for("mouse", "xenium_prime", "wmb_10xv3")
    stored = si.efficiency_table(member) if member is not None else None
    counts_arr = calls["total_counts"].to_numpy(np.float64)
    confident = (calls["class_bp"].to_numpy(np.float64) >= 0.9) & (counts_arr >= 20)
    remeasure = real_qc.factor_remeasure(
        counts,
        genes,
        calls["class"].astype(str).to_numpy(),
        matrix,
        stored,
        include=confident,
        asset_id=None if member is None else member.asset_id,
    )
    record["factor_remeasure"] = remeasure.summary()
    if remeasure.factors is not None:
        remeasure.factors.to_csv(args.out_dir / "factor_remeasure.csv", index=False)
    lines += [
        "1. factor re-measure (confident class calls; genes informative in both):",
        f"   r {remeasure.pearson_r:.4f} (Spearman {remeasure.spearman_r:.4f}) on "
        f"{remeasure.n_informative} genes; warning "
        f"{bool(remeasure.outcome and remeasure.outcome.fired)}",
        "",
    ]
    # 2. Non-neuronal high depth and the glial trend on the v7 provisional coverage.
    cls = calls["class"].astype(str).to_numpy()
    grid = [int(value) for value in summary["depth_grid"]]
    bins = res.depth_bin(counts_arr, grid)
    emitted = simulate.emission_frame(summary, "provisional")
    class_ok = simulate.provisional_ok(
        "class", cls, bins, calls["class_bp"].to_numpy(np.float64), counts_arr, emitted
    )
    subclass_ok = (
        simulate.provisional_ok(
            "subclass",
            cls,
            bins,
            calls["subclass_bp"].to_numpy(np.float64),
            counts_arr,
            emitted,
        )
        & (counts_arr >= 60)
        & class_ok
    )
    nonneuronal = sorted(
        {value for value in set(cls) if si.is_neuronal_class(value, "mouse") is False}
    )
    flags = real_qc.nonneuronal_high_depth_flags(counts_arr, cls, class_depth)
    trend, outcomes = real_qc.nonneuronal_depth_trend(
        counts_arr, cls, {"class": class_ok, "subclass": subclass_ok}, nonneuronal
    )
    trend.to_csv(args.out_dir / "nonneuronal_depth_trend.csv", index=False)
    in_nn = np.isin(cls, nonneuronal)
    record["nonneuronal_high_depth"] = {
        "share_of_nonneuronal_flagged": float(flags[in_nn].mean())
        if in_nn.any()
        else None,
        "share_of_nonneuronal_at_1000": float((counts_arr[in_nn] >= 1000).mean())
        if in_nn.any()
        else None,
        "trend_outcomes": [item.to_json() for item in outcomes],
    }
    high = record["nonneuronal_high_depth"]
    lines += [
        "2. non-neuronal high depth (version-7 provisional decisions):",
        "   non-neuronal cells flagged flag_nonneuronal_high_depth: "
        f"{high['share_of_nonneuronal_flagged']:.3f} (at >= 1,000 counts: "
        f"{high['share_of_nonneuronal_at_1000']:.3f})",
        "   coverage by depth band (500-999 / 1,000-1,999 / >= 2,000):",
    ]
    for (level, name), group in trend.groupby(["level", "class"], sort=True):
        values = group.set_index("band")
        lines.append(
            f"   {level:8s} {name:18s} "
            + " / ".join(
                f"{values.loc[band, 'coverage']:.3f} "
                f"({int(values.loc[band, 'n_cells'])})"
                for band in ("500-999", "1,000-1,999", ">= 2,000")
                if band in values.index
            )
        )
    lines.append(f"   falling above 1,000 counts (> 2 SE): {len(outcomes)}")
    lines += [f"     {item.level} / {item.cls}" for item in outcomes]
    lines.append("")
    # 3. Coverage vs simulation.
    shares = real_qc.class_bin_shares(
        counts_arr, cls, sorted(set(class_depth["depth"]))
    )
    predicted = real_qc.predicted_class_coverage(
        class_depth, shares, levels=("class", "subclass")
    )
    real = real_qc.real_class_coverage(
        cls, {"class": class_ok, "subclass": subclass_ok}
    )
    profile_predicted = (
        real_qc.profile_coverage_table(
            pd.read_csv(args.profile_predictions),
            {"class": "cov_class_prov", "subclass": "cov_subclass_prov"},
        )
        if args.profile_predictions is not None
        else None
    )
    comparison = real_qc.coverage_vs_simulation(
        real, predicted, profile_predicted=profile_predicted
    )
    comparison.table.to_csv(args.out_dir / "coverage_vs_simulation.csv", index=False)
    record["coverage_vs_simulation"] = comparison.summary()
    lines += [
        "3. coverage vs simulation (real prov < class-depth prediction - 0.10; "
        "the profile-mode prediction, when given, reported beside it):",
        f"   {len(comparison.flagged)} flagged: "
        + ", ".join(f"{level}/{name}" for level, name in comparison.flagged),
        *[
            f"   {row['level']:8s} {row['class']:22s} real {row['real_coverage']:.3f}"
            f" class-depth {row['predicted_coverage']:.3f}"
            f" profile {row['profile_coverage']:.3f}"
            for row in comparison.table[comparison.table["warn"]].to_dict("records")
        ],
        "",
    ]
    # 4. Gene complexity against simulated cells.
    test = res.load_test_cells(args.test_set)
    recipe = res.member_recipe("R1_contam_HO", 0, AnnotationResolvabilityConfig())
    query = res.thin_and_contaminate_v7(test, list(SIM_DEPTHS), recipe)
    simulated_genes = np.diff(query.counts.tocsr().indptr)
    simulated_depth = query.obs["total_counts"].to_numpy(np.float64)
    native_genes = np.diff(counts.tocsr().indptr)
    # The M3c check bins realised totals at the lower edge, as it was run
    # (M13's interpolated matching, ruling C3 (b), needs per-cell pairs).
    table, outcome = real_qc.gene_complexity_check(
        native_genes,
        counts_arr,
        simulated_genes,
        simulated_depth,
        grid,
        matching=real_qc.GENE_COMPLEXITY_LOWER_EDGE,
    )
    table.to_csv(args.out_dir / "gene_complexity.csv", index=False)
    record["gene_complexity"] = outcome.to_json()
    lines += [
        "4. gene complexity (native vs R1_contam_HO@0 simulated cells, median genes):",
        *[
            f"   bin {int(row['depth']):5d}: native {row['native_median_genes']:.0f} "
            f"({int(row['n_native'])}) vs simulated "
            f"{row['simulated_median_genes']:.0f} "
            f"({int(row['n_simulated'])}); gap {row['gap']:+.3f}"
            for row in table[table["judged"]].to_dict("records")
        ],
        f"   warning (> 45%): {outcome.fired}",
    ]
    (args.out_dir / "REAL_QC_PUBLIC.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n"
    )
    (args.out_dir / "REAL_QC_PUBLIC.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
