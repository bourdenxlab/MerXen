#!/usr/bin/env python
"""M3c pre-registration §21 (iv): per-class calibration on the public 5K section.

Reported, in-sample, never pass / fail. For the M3c version-7 5K mouse bundle:

* **Real**: the public section's 63,147 vendor cells with the phase-1 calls
  (``$A/5k_real/mouse5k/qc/percell_calls_mouse5k.parquet``) when the M3c
  bundle's query markers equal those of ``c30f7eab`` (sha256; otherwise the
  script stops: the cells must be re-mapped with the M3c bundle first). Per
  (level, called class): the raw §7.3 rule (class bp >= 0.9 and >= 20 counts;
  subclass bp >= 0.8, >= 50 counts and the class rule) and the coverage under
  the version-7 ensemble's provisional decisions applied to the real cells
  (bin of the total counts, applied threshold, floors, mouse subclass floor
  60 and the class).
* **Simulated**: profile mode on the public profile for each emission member
  and the member mean, weighted to the real called composition
  (``annotation-panel-simulate``'s ``<reference>/profile_predictions.csv``),
  and the class-depth predictor ``sum_d s_c(d) cov(L, c, d)`` at the
  section's own per-class bin shares (``real_qc.predicted_class_coverage``,
  the predictor RESOLVE will use).
* Per major class (the 14 of (ii)) and level (class, subclass): prediction -
  observed for the raw rule and for the provisional coverage; which (level,
  class) the per-class warning flags (``real_qc.coverage_vs_simulation``:
  real < class-depth prediction - 0.10, >= 200 cells); simulated precision
  (member mean; no real counterpart). The ALL row is shown beside the classes.
* With ``--churn-dir`` (the §21 (iii) diagnostic output of the bundle): the
  real provisional coverage (class, subclass) under ensemble A (the bundle)
  and under the fresh ensemble B.

Read-only on the inputs; writes ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

A = Path("/srv/storage/MerXen/annotation_dev/evidence_20260926")
CALLS = A / "5k_real/mouse5k/qc/percell_calls_mouse5k.parquet"
PHASE1_BUNDLE = Path(
    "/srv/storage/MerXen/annotation_references_large/wmb_panel/"
    "c30f7eab413cba9e5e142c337d01140750f28967adff2100903f04893eced4f8"
)
MAJOR = (
    "01 IT-ET Glut",
    "02 NP-CT-L6b Glut",
    "04 DG-IMN Glut",
    "06 CTX-CGE GABA",
    "07 CTX-MGE GABA",
    "09 CNU-LGE GABA",
    "11 CNU-HYa GABA",
    "12 HY GABA",
    "13 CNU-HYa Glut",
    "18 TH Glut",
    "30 Astro-Epen",
    "31 OPC-Oligo",
    "33 Vascular",
    "34 Immune",
)
GLIA = ("30 Astro-Epen", "31 OPC-Oligo", "33 Vascular", "34 Immune")
LEVELS = ("class", "subclass")
SUBCLASS_FLOOR = 60.0


def _file_sha(bundle: Path, prefix: str) -> dict[str, str]:
    manifest = json.loads((bundle / "bundle.json").read_text())
    return {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files") or []
        if str(item["path"]).startswith(prefix)
    }


def marker_content_sha256(bundle: Path) -> dict[str, str]:
    """Return the sha256 of each query-marker file's marker lists alone.

    ``query_markers*.json`` embed a ``log`` and a ``metadata`` record (input
    paths, timestamps) besides the per-parent marker lists MapMyCells reads;
    the canonical JSON of the lists (keys sorted) identifies the lookup.
    """
    import hashlib

    result = {}
    for path in sorted(bundle.glob("query_markers*.json")):
        payload = json.loads(path.read_text())
        lists = {
            key: value
            for key, value in payload.items()
            if key not in ("log", "metadata")
        }
        text = json.dumps(lists, sort_keys=True, separators=(",", ":"))
        result[path.name] = hashlib.sha256(text.encode()).hexdigest()
    return result


def query_markers_equal(bundle: Path) -> tuple[bool, dict[str, Any]]:
    """Whether the bundle's query markers equal ``c30f7eab``'s (sha256).

    Equal when the files are byte-identical, or when their marker lists are
    (the files then differ only in the embedded log and metadata records);
    the mapping precompute must be byte-identical too.
    """
    files = {
        "m3c": _file_sha(bundle, "query_markers"),
        "c30f7eab": _file_sha(PHASE1_BUNDLE, "query_markers"),
    }
    content = {
        "m3c": marker_content_sha256(bundle),
        "c30f7eab": marker_content_sha256(PHASE1_BUNDLE),
    }
    precompute = {
        "m3c": _file_sha(bundle, "mapping_precompute"),
        "c30f7eab": _file_sha(PHASE1_BUNDLE, "mapping_precompute"),
    }
    equal_files = bool(files["m3c"]) and files["m3c"] == files["c30f7eab"]
    equal_content = bool(content["m3c"]) and content["m3c"] == content["c30f7eab"]
    equal_precompute = (
        bool(precompute["m3c"]) and precompute["m3c"] == precompute["c30f7eab"]
    )
    record = {
        "file_sha256": files,
        "marker_lists_sha256": content,
        "mapping_precompute_sha256": precompute,
        "files_identical": equal_files,
        "marker_lists_identical": equal_content,
        "mapping_precompute_identical": equal_precompute,
    }
    return (equal_files or equal_content) and equal_precompute, record


def real_coverage(
    calls: pd.DataFrame, summary: Mapping[str, Any], regime: str = "provisional"
) -> pd.DataFrame:
    """Per real cell: the raw rule and the version-7 provisional coverage."""
    from merxen.annotation import resolvability as res
    from merxen.annotation import simulate

    counts = calls["total_counts"].to_numpy(np.float64)
    grid = [int(value) for value in summary["depth_grid"]]
    bins = res.depth_bin(counts, grid)
    emitted = simulate.emission_frame(summary, regime)
    cls = calls["class"].astype(str).to_numpy()
    class_bp = calls["class_bp"].to_numpy(np.float64)
    subclass_bp = calls["subclass_bp"].to_numpy(np.float64)
    table = pd.DataFrame({"class": cls, "total_counts": counts})
    table["cov_class_rule73"] = (class_bp >= 0.9) & (counts >= 20)
    table["cov_subclass_rule73"] = (
        table["cov_class_rule73"] & (subclass_bp >= 0.8) & (counts >= 50)
    )
    class_ok = simulate.provisional_ok("class", cls, bins, class_bp, counts, emitted)
    subclass_ok = (
        simulate.provisional_ok("subclass", cls, bins, subclass_bp, counts, emitted)
        & (counts >= SUBCLASS_FLOOR)
        & class_ok
    )
    table["cov_class_prov"] = class_ok
    table["cov_subclass_prov"] = subclass_ok
    return table


def per_class_real(table: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Return the real coverage per called class and ALL."""
    rows = []
    for cls, frame in [*table.groupby("class", sort=True), ("ALL", table)]:
        rows.append(
            {"class": str(cls), "n_real": len(frame)}
            | {metric: float(frame[metric].mean()) for metric in metrics}
        )
    return pd.DataFrame(rows)


def class_depth_prediction(bundle: Path, calls: pd.DataFrame) -> pd.DataFrame:
    """The class-depth predictor at the section's own per-class bin shares."""
    from merxen.annotation import real_qc

    class_depth = pd.read_parquet(bundle / "resolvability_class_depth.parquet")
    grid = sorted({int(value) for value in class_depth["depth"]})
    shares = real_qc.class_bin_shares(
        calls["total_counts"].to_numpy(np.float64),
        calls["class"].astype(str).to_numpy(),
        grid,
    )
    return real_qc.predicted_class_coverage(class_depth, shares, levels=LEVELS)


def coverage_warnings(
    real: pd.DataFrame,
    predicted: pd.DataFrame,
    profile: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """``real_qc.coverage_vs_simulation`` on the provisional coverage.

    The class-depth predictor decides the warnings; profile mode's member
    mean, when given, is reported beside it (D4 of 2026-09-29).
    """
    from merxen.annotation import real_qc

    long = []
    for level in LEVELS:
        for row in real[real["class"] != "ALL"].to_dict("records"):
            long.append(
                {
                    "level": level,
                    "class": row["class"],
                    "n_cells": int(row["n_real"]),
                    "real_coverage": float(row[f"cov_{level}_prov"]),
                }
            )
    profile_predicted = (
        None
        if profile is None
        else real_qc.profile_coverage_table(
            profile, {level: f"cov_{level}_prov" for level in LEVELS}
        )
    )
    comparison = real_qc.coverage_vs_simulation(
        pd.DataFrame(long), predicted, profile_predicted=profile_predicted
    )
    return comparison.table, [outcome.message for outcome in comparison.outcomes]


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "   -  "
    return (
        f"{value:+.3f}" if isinstance(value, float) and abs(value) < 1 else f"{value}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="simulate output")
    parser.add_argument("--calls", type=Path, default=CALLS)
    parser.add_argument("--churn-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    equal, markers = query_markers_equal(args.bundle)
    if not equal:
        (args.out_dir / "CALIBRATION.txt").write_text(
            "query markers differ from c30f7eab: re-map the public cells with the "
            f"M3c bundle first\n{json.dumps(markers, indent=2)}\n"
        )
        return 2
    calls = pd.read_parquet(args.calls)
    summary = json.loads((args.bundle / "resolvability_summary.json").read_text())
    metrics = [
        "cov_class_rule73",
        "cov_subclass_rule73",
        "cov_class_prov",
        "cov_subclass_prov",
    ]
    real = per_class_real(real_coverage(calls, summary), metrics)
    profile = pd.read_csv(args.run_dir / "wmb_panel" / "profile_predictions.csv")
    profile["class"] = profile["class"].astype(str)
    predicted = class_depth_prediction(args.bundle, calls)
    warn_table, warnings = coverage_warnings(real, predicted, profile)
    rows = []
    members = sorted(profile["member"].astype(str).unique())
    for cls in [*MAJOR, "ALL"]:
        real_row = real[real["class"] == cls]
        if real_row.empty:
            continue
        for level in LEVELS:
            record: dict[str, Any] = {
                "class": cls,
                "level": level,
                "n_real": int(real_row["n_real"].iloc[0]),
                "real_raw": float(real_row[f"cov_{level}_rule73"].iloc[0]),
                "real_prov": float(real_row[f"cov_{level}_prov"].iloc[0]),
            }
            for member in members:
                sim = profile[(profile["member"] == member) & (profile["class"] == cls)]
                if sim.empty:
                    continue
                tag = "mean" if member == "member_mean" else member
                record[f"sim_raw[{tag}]"] = float(sim[f"cov_{level}_rule73"].iloc[0])
                record[f"sim_prov[{tag}]"] = float(sim[f"cov_{level}_prov"].iloc[0])
                precision = f"prec_{level}_prov"
                if precision in sim.columns:
                    record[f"sim_prec_prov[{tag}]"] = float(sim[precision].iloc[0])
            if cls != "ALL":
                match = predicted[
                    (predicted["level"] == level) & (predicted["class"] == cls)
                ]
                record["class_depth_pred"] = (
                    float(match["predicted_coverage"].iloc[0]) if len(match) else None
                )
                flag = warn_table[
                    (warn_table["level"] == level) & (warn_table["class"] == cls)
                ]
                record["warn"] = bool(flag["warn"].iloc[0]) if len(flag) else False
            record["delta_raw_mean"] = (
                record.get("sim_raw[mean]", np.nan) - record["real_raw"]
            )
            record["delta_prov_mean"] = (
                record.get("sim_prov[mean]", np.nan) - record["real_prov"]
            )
            if record.get("class_depth_pred") is not None:
                record["delta_class_depth"] = (
                    record["class_depth_pred"] - record["real_prov"]
                )
            rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "calibration_per_class.csv", index=False)
    warn_table.to_csv(args.out_dir / "coverage_warning_table.csv", index=False)
    lines = [
        "M3c pre-registration §21 (iv): per-class calibration on the public 5K section",
        "(reported; in-sample for depth, composition and R3 factors: a best case)",
        f"date {datetime.now(UTC).isoformat(timespec='seconds')}; bundle {args.bundle}",
        f"real calls: phase-1 calls (query markers equal to c30f7eab: {equal}); "
        f"{len(calls)} vendor cells",
        f"simulated: {args.run_dir / 'wmb_panel' / 'profile_predictions.csv'} "
        f"(members {', '.join(members)})",
        "",
        "delta = prediction - observed. raw = the §7.3 rule; prov = coverage under the",
        "version-7 ensemble's provisional decisions; cd = class-depth predictor at the",
        "section's own bin shares; warn = real prov < cd - 0.10 (>= 200 cells).",
        "",
        f"{'class':22s} {'level':8s} {'n':>6s} {'real_raw':>8s} {'d_raw':>7s} "
        f"{'real_prov':>9s} {'d_prov':>7s} {'cd':>6s} {'d_cd':>7s} {'prec':>6s} warn",
    ]
    for record in rows:
        cd = record.get("class_depth_pred")
        lines.append(
            f"{record['class']:22s} {record['level']:8s} {record['n_real']:6d} "
            f"{record['real_raw']:8.3f} {record['delta_raw_mean']:+7.3f} "
            f"{record['real_prov']:9.3f} {record['delta_prov_mean']:+7.3f} "
            + (
                f"{cd:6.3f} {record['delta_class_depth']:+7.3f} "
                if cd is not None
                else f"{'-':>6s} {'-':>7s} "
            )
            + f"{record.get('sim_prec_prov[mean]', float('nan')):6.3f} "
            + ("WARN" if record.get("warn") else "")
        )
    glia = table[table["class"].isin(GLIA)]
    neurons = table[table["class"].isin([c for c in MAJOR if c not in GLIA])]

    def span(frame: pd.DataFrame, column: str) -> str:
        return f"{frame[column].min():+.3f} to {frame[column].max():+.3f}"

    lines += [
        "",
        "ranges of prediction - observed (member mean):",
        f"  major neurons: raw {span(neurons, 'delta_raw_mean')}; "
        f"prov {span(neurons, 'delta_prov_mean')}",
        f"  glia:          raw {span(glia, 'delta_raw_mean')}; "
        f"prov {span(glia, 'delta_prov_mean')}",
        "  phase-1 reference (D3): major neurons -.03 to +.09, glia +.06 to +.18",
        "",
        f"per-class coverage warnings (all classes with >= 200 cells): {len(warnings)}",
        *[f"  {message.split('. Simulated')[0]}" for message in warnings],
    ]
    per_member = []
    for member in members:
        if member == "member_mean":
            continue
        subset = profile[(profile["member"] == member) & (profile["class"] == "ALL")]
        if len(subset):
            per_member.append(
                f"  {member}: ALL raw class {subset['cov_class_rule73'].iloc[0]:.3f} "
                f"subclass {subset['cov_subclass_rule73'].iloc[0]:.3f}; prov class "
                f"{subset['cov_class_prov'].iloc[0]:.3f} subclass "
                f"{subset['cov_subclass_prov'].iloc[0]:.3f}"
            )
    real_all = real[real["class"] == "ALL"].iloc[0]
    lines += [
        "",
        "ALL rows per member (simulated, weighted to the real composition):",
        *per_member,
        f"  real: raw class {real_all['cov_class_rule73']:.3f} subclass "
        f"{real_all['cov_subclass_rule73']:.3f}; prov class "
        f"{real_all['cov_class_prov']:.3f} subclass "
        f"{real_all['cov_subclass_prov']:.3f}",
    ]
    churn_record: dict[str, Any] = {}
    if args.churn_dir is not None:
        b_summary_path = next(
            args.churn_dir.rglob("ensemble_B/resolvability_summary.json")
        )
        b_summary = json.loads(b_summary_path.read_text())
        real_b = per_class_real(real_coverage(calls, b_summary), metrics)
        lines += [
            "",
            "real 5K provisional coverage under ensemble A (bundle) and B (fresh):",
        ]
        for cls in ["ALL", *MAJOR]:
            a_row = real[real["class"] == cls]
            b_row = real_b[real_b["class"] == cls]
            if a_row.empty or b_row.empty:
                continue
            a_class = float(a_row["cov_class_prov"].iloc[0])
            b_class = float(b_row["cov_class_prov"].iloc[0])
            a_sub = float(a_row["cov_subclass_prov"].iloc[0])
            b_sub = float(b_row["cov_subclass_prov"].iloc[0])
            churn_record[cls] = {
                "class_A": a_class,
                "class_B": b_class,
                "subclass_A": a_sub,
                "subclass_B": b_sub,
            }
            lines.append(
                f"  {cls:22s} class A {a_class:.3f} B {b_class:.3f} "
                f"({b_class - a_class:+.3f}); "
                f"subclass A {a_sub:.3f} B {b_sub:.3f} ({b_sub - a_sub:+.3f})"
            )
    (args.out_dir / "CALIBRATION.txt").write_text("\n".join(lines) + "\n")
    (args.out_dir / "CALIBRATION.json").write_text(
        json.dumps(
            {
                "query_markers": markers,
                "rows": json.loads(table.to_json(orient="records")),
                "warnings": warnings,
                "real_coverage_A_B": churn_record,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
