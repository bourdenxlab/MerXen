#!/usr/bin/env python
"""M8 human acceptance: score every §14 row with the §18 protocol (plan §12 M8).

Two subcommands, both read-only on their inputs; they write only to ``--out``.

``score``
    Collects the stage B outputs and scores every human criterion row
    against the pre-registered thresholds (plan §14 = pre-registration §9)
    with the M8 scoring protocol (pre-registration §18), then writes
    ``summary.json``, ``summary.html`` and ``tables/`` to ``--out``.

``h15``
    Compares a re-run of MAP + RESOLVE with the pipeline's outputs (H15:
    "identical re-run -> identical DataFrame content"; "seed 0 vs 1 changes
    <= 1% of confident broad labels") and writes one JSON that ``score``
    reads.

Row sources of ``score`` (``--runs-root`` is the acceptance run's publish
root, ``<runs>/<pair>/<seg>/``):

- ``resolve_criteria.py`` (``--criteria``): H1, H2, H3, H4 (the WHB-only
  headline set, D6; the ``H4[<set>]`` rows are reported), H5, H7, H8 and
  its warning row, H9, H10, H16, H17. Each value is cross-checked with the
  pair x segmentation's ``ANNOTATION_REPORT`` ``acceptance_metrics.json``
  (H1 soft whole-section JSD, H2, H3, H5, H7, the gate level and warning,
  the warning's coverage value, H16's marking): a difference beyond the
  report's 6-significant-digit rounding is ``CROSSCHECK-MISMATCH`` and goes
  back to the user.
- ``marker_referee.py`` (``--referee``): H10 must equal ``resolve_criteria``'s.
- The report: H12 (``report_scoring.score_h12``: the square 500 µm tile CI,
  AND over both platforms, WM > GM on the tile CI; the tangential blocks
  beside it, not scored, M7 D23) and H13's MENDER, id-set and depth parts.
- ``compare_legacy.py untouched`` (``--legacy``): H13 "legacy tables
  untouched before the flip (sha256)".
- ``draw_spread.py`` (``--draw-spread``): H18 on the scored draw ``c0e0``
  (D4's scope with ``--d4-reference``); ``draw_spread.csv`` is copied as the
  acceptance output that never changes a verdict (§18 item 2).
- The Nextflow trace(s) (``--trace``): H14 (annotation wall = PANEL + MAP +
  RESOLVE realtime summed over the pair's segmentations <= 2.5 h; MAP peak
  RSS <= 16 GB; PREP <= 2.5 h per validated panel from each bundle's
  recorded build time; no GPU process and CUDA hidden in every task).
- ``h15`` JSONs (``--h15``): H15 on P7513 and P1212 proseg_hybrid.
- The archived E2 HO contaminated simulation (``--e2-cells``): H6, first
  measured here (definition below).

Verdicts (``report_scoring``): ``PASS``; ``INFO pass`` / ``INFO fail`` (not
scored by the flip rule: held-out or reported rows, rule 3);
``EXCEPTION (Dn)`` inside an approved exception's stated scope (D3, D4, D5,
D7 only, §18 item 6); ``EXCEPTION-RECHECK (Dn)`` outside it; ``FAIL-OUTSIDE``
for a failure no decision names, which includes every failure of H6 and
H12-H15 (first measurements, no pre-approved exception, item 4);
``NOT_AVAILABLE`` for a scored row that could not be measured. Everything
but ``PASS``, ``INFO *`` and ``EXCEPTION (Dn)`` goes back to the user, and so
does any failure of H6 or H12-H15 in a row the flip rule does not score
(``INFO fail`` on a held-out pair or another segmentation; item 4 names every
failure of a first-measured criterion).

Before scoring, the P5 check (protocol item 2): every PREP bundle the run
used names one of the three D1 bundles and was reused (its reuse logged by
PREP, or its ``bundle.json`` created before ``--run-start``), or its
self-map records 8 MapMyCells workers; every MAP run mapped with one of
those bundles. Otherwise nothing is scored: every row is ``UNSCORED (P5)``
and goes back to the user.

H6 as implemented: the E2 in-silico cells of the ``contam`` scenario (the
held-out-donor WHB engine, ``ho_*`` columns) at each depth; a call is
confident at the v1 raw thresholds (broad bp >= 0.73; supercluster bp >=
0.69 with a confident broad, rule 4), allowing 1e-6 below them as
``schema.meets_threshold``; each cell weighted by (the dataset's share of
its truth class) / (the simulation's share at that depth), E2's
composition reweighting, where the dataset's composition is the §5.5 soft
mass of its proseg_hybrid table cells on the WHB superclusters (assigned
node and runner-ups, bootstrap probabilities; the mass the production
reweighting uses), restricted to the simulated truth superclusters and
aggregated to E2's broad classes through the simulation's own truth
mapping. Per level, depth (broad D >= 15, supercluster D >= 30) and class
with >= 50 simulated test cells of that truth class: the reweighted
precision of the confident calls of that class must reach 0.90 (broad) /
0.85 (supercluster); a class without a confident call is not evaluable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen.annotation import report_scoring as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_criteria import (  # noqa: E402
    DEV_PAIRS,
    H4_HEADLINE_SET,
)

logger = logging.getLogger("run_acceptance")

PAIRS: Final = ("P7513", "P1212", "P7113", "P5011")
PLATFORMS: Final = ("MERSCOPE", "XENIUM")
SEGMENTATIONS: Final = ("proseg_hybrid", "reseg", "proseg_mask", "original_seg")
SCORED_SEGMENTATION: Final = "proseg_hybrid"
# The D1 bundles (pre-registration §18; stage A2) and the self-map's workers.
D1_BUNDLES: Final[dict[str, str]] = {
    "ffc6dd2d96e2a6cce6f37fe95754ba6e057aa412da7df2e0688ad921395739c6": "set a WHB",
    "dc3c500f2ed8e5f43ec2900a43ef5214957eea095694b2d9e6d8886868f5c240": "set a SEA-AD",
    "2ed1c2f6dbc7108e409ba157e3cc350eecb2211e2bb250de190324e2dc6f39b4": "set c WHB",
}
SET_A_WHB: Final = "ffc6dd2d96e2a6cce6f37fe95754ba6e057aa412da7df2e0688ad921395739c6"
SELF_MAP_WORKERS: Final = 8
SCORED_DRAW: Final = "c0e0"

VERDICT_NOT_AVAILABLE: Final = "NOT_AVAILABLE"
VERDICT_UNSCORED: Final = "UNSCORED (P5)"
CROSSCHECK_MISMATCH: Final = "CROSSCHECK-MISMATCH"
FIRST_MEASUREMENTS: Final = frozenset({"H6", "H12", "H13", "H14", "H15"})
# acceptance_metrics.json stores float(f"{x:.6g}") (report_model._clean), up to
# 5e-6 relative from the full-precision value the criteria table keeps; the
# cross-check allows twice that, so only a difference in the first 5
# significant digits is a mismatch.
REPORT_REL_TOL: Final = 1e-5
REFEREE_NEW: Final = "new confident (RESOLVE)"  # marker_referee.NEW
REFEREE_LEGACY: Final = "legacy broad_class"  # marker_referee.LEGACY
H6_POOLED: Final = "pooled"

# H6 (plan §14): v1 raw thresholds, targets, depths and the class-size floor.
H6_THRESHOLDS: Final[dict[str, float]] = {"broad": 0.73, "supercluster": 0.69}
H6_TARGETS: Final[dict[str, float]] = {"broad": 0.90, "supercluster": 0.85}
H6_DEPTHS: Final[dict[str, tuple[int, ...]]] = {
    "broad": (15, 30, 60, 120),
    "supercluster": (30, 60, 120),
}
H6_MIN_TEST_CELLS: Final = 50
H6_SCENARIO: Final = "contam"
H6_COLUMNS: Final[dict[str, tuple[str, str, str]]] = {
    # level: (truth, prediction, probability)
    "broad": ("true_broad", "ho_broad", "ho_broad_p"),
    "supercluster": ("true_super", "ho_super", "ho_super_p"),
}
THRESHOLD_SLACK: Final = 1e-6  # schema.meets_threshold (float32 probabilities)
N_RUNNERS_UP: Final = 5

# H14 (plan §14).
ANNOTATION_PROCESSES: Final = (
    "ANNOTATE_PANEL",
    "CLUSTERING_SQUIDPY_ANNOTATE_MAP",
    "CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE",
)
MAP_PROCESS: Final = "CLUSTERING_SQUIDPY_ANNOTATE_MAP"
PREP_PROCESS: Final = "ANNOTATE_REFERENCE_PREP"
GPU_PROCESSES: Final = frozenset(
    {
        "CLUSTERING_SQUIDPY_COMPUTE",
        "CELLPOSE_SEGMENT",
        "CELLPOSE_NUCLEI_SEGMENT",
        "ALIGN",
    }
)
H14_ANNOTATION_WALL_MAX_H: Final = 2.5
H14_MAP_PEAK_RSS_MAX_GB: Final = 16.0
H14_PREP_WALL_MAX_H: Final = 2.5
# How a task script exports an empty CUDA_VISIBLE_DEVICES (Nextflow writes the
# config env block with single quotes; a script's own export uses double).
CUDA_HIDDEN: Final = ("CUDA_VISIBLE_DEVICES=''", 'CUDA_VISIBLE_DEVICES=""')

# H15 (plan §14).
H15_PAIRS: Final = ("P7513", "P1212")
H15_SEED_CHANGE_MAX: Final = 0.01

# Layout of the acceptance run's publish root (STAGE_B_PLAN option (c)).
REPORT_PATH: Final = (
    "{pair}/{seg}/annotation_report/annotation_report_out/acceptance_metrics.json"
)
STATE_REPORT_PATH: Final = (
    "{pair}/proseg_hybrid/mender_state_policy/annotation_report/"
    "annotation_report_out/acceptance_metrics.json"
)
DEPTH_DIR: Final = (
    "{pair}/{seg}/compute_cortical_depth_m8/{platform}/compute_cortical_depth_out/{seg}"
)
MAP_MANIFEST_PATH: Final = (
    "{pair}/{seg}/annotation_map/annotation_map_out/map_manifest.json"
)
# The second MENDER policy's run (launch Bstate) maps proseg_hybrid again.
STATE_LABEL: Final = "mender_state_policy"
STATE_MAP_MANIFEST_PATH: Final = (
    "{pair}/proseg_hybrid/" + STATE_LABEL + "/annotation_map/annotation_map_out/"
    "map_manifest.json"
)
MENDER_POLICIES: Final[dict[str, str]] = {
    "main": "exclude_from_features",
    "state": "state",
}


# ---------------------------------------------------------------------------
# rows and verdicts


@dataclass
class Row:
    """One scored (or reported) row of the acceptance summary.

    Attributes:
        criterion: Criterion id (``H7``, ``H17/H2``, ``H13/mender`` ...).
        pair: Pair id, or ``all`` / ``PREP``.
        segmentation: Segmentation.
        dataset: Sample id, pair id or label.
        value: The measured value (``None``: not measured).
        comparator: ``<=``, ``>=``, ``==`` or ``listed``.
        threshold: The pre-registered threshold.
        passes: The comparison (``None``: not measured).
        scored: Whether the flip rule scores the row.
        source: Where the value comes from.
        note: What a reader needs.
        exception: The decision whose scope was checked (``D3`` ...).
        scope_check: What the scope check found.
        crosscheck: ``ok``, ``CROSSCHECK-MISMATCH: ...`` or empty.
        verdict: Set by ``finalize``.
    """

    criterion: str
    pair: str
    segmentation: str
    dataset: str
    value: Any
    comparator: str
    threshold: Any
    passes: bool | None
    scored: bool
    source: str
    note: str = ""
    exception: str = ""
    scope_check: str = ""
    crosscheck: str = ""
    verdict: str = ""
    exception_verdict: str = ""

    @property
    def base(self: Row) -> str:
        """The criterion id without its part (``H13/mender`` -> ``H13``)."""
        return self.criterion.split("/")[0].split("[")[0]

    def to_json(self: Row) -> dict[str, Any]:
        """Return the JSON-safe record."""
        record = dataclasses.asdict(record_safe(self))
        record["first_measurement"] = self.base in FIRST_MEASUREMENTS
        record["back_to_user"] = back_to_user(self)
        record["back_to_user_reason"] = back_to_user_reason(self)
        return record


def record_safe(row: Row) -> Row:
    """Return a copy with NaN values replaced by ``None``."""
    value = row.value
    if isinstance(value, float | np.floating) and not math.isfinite(float(value)):
        value = None
    elif isinstance(value, np.generic):
        value = value.item()
    threshold = row.threshold
    if isinstance(threshold, float | np.floating) and not math.isfinite(
        float(threshold)
    ):
        threshold = None
    return dataclasses.replace(row, value=value, threshold=threshold)


def finalize(row: Row, exception: sc.ExceptionVerdict | None = None) -> Row:
    """Set a row's verdict with the §18 rules (``report_scoring.verdict``).

    A scored row without a measurement is ``NOT_AVAILABLE``; a cross-check
    mismatch keeps the verdict and is sent back to the user
    (``back_to_user``).
    """
    if exception is not None:
        row.exception = exception.decision
        row.scope_check = "; ".join(exception.reasons)
        row.exception_verdict = exception.verdict
    if row.passes is None:
        row.verdict = VERDICT_NOT_AVAILABLE if row.scored else "INFO not measured"
        return row
    row.verdict = sc.verdict(bool(row.passes), row.scored, exception)
    return row


def back_to_user_reason(row: Row) -> str:
    """Why a row goes back to the user (§18 items 4 and 6; empty: it does not).

    A cross-check mismatch, a verdict outside ``PASS`` / ``INFO *`` /
    ``EXCEPTION (Dn)``, and any failure of a first-measured criterion (H6,
    H12-H15), also in a row the flip rule does not score (item 4: "any
    failure comes back to the user").
    """
    if row.crosscheck.startswith(CROSSCHECK_MISMATCH):
        return "cross-check mismatch"
    if row.verdict.startswith(
        ("EXCEPTION-RECHECK", sc.FAIL_OUTSIDE, VERDICT_NOT_AVAILABLE, "UNSCORED")
    ):
        return f"verdict {row.verdict}"
    if row.base in FIRST_MEASUREMENTS and row.passes is False:
        return "first-measured criterion failed (not scored here; item 4)"
    return ""


def back_to_user(row: Row) -> bool:
    """Whether a row goes back to the user (``back_to_user_reason``)."""
    return bool(back_to_user_reason(row))


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def _missing_set(value: Any) -> set[int]:
    if _is_missing(value):
        return set()
    return {int(float(item)) for item in str(value).split(",") if item.strip()}


# ---------------------------------------------------------------------------
# P5: the PREP bundles were the D1 bundles, reused or mapped with 8 workers


@dataclass
class PrepCheck:
    """The P5 check of the run's PREP bundles (protocol item 2).

    Attributes:
        ok: Whether scoring may proceed.
        bundles: One record per bundle ref.
        map_runs: One record per MAP run (pair, seg, sample, run, build hash).
        reasons: Why it failed (empty when ``ok``).
    """

    ok: bool
    bundles: list[dict[str, Any]] = field(default_factory=list)
    map_runs: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_prep_bundles(
    bundle_refs: Sequence[Path],
    run_start: datetime,
    *,
    reuse_logged: Mapping[str, bool] | None = None,
    map_manifests: Mapping[tuple[str, str], Path] | None = None,
) -> PrepCheck:
    """Check that the run used the D1 bundles as the protocol requires (P5).

    Args:
        bundle_refs: The run's PREP ``bundle_ref.json`` files.
        run_start: When the run started (a bundle created later was built).
        reuse_logged: Build hash -> whether PREP logged ``reused`` for it.
        map_manifests: (pair, seg) -> ``map_manifest.json``: every MAP run
            must name one of the checked bundles.

    Returns:
        The check.
    """
    reuse_logged = reuse_logged or {}
    check = PrepCheck(ok=True)
    seen: dict[str, dict[str, Any]] = {}
    for path in bundle_refs:
        ref = _json(path)
        build_hash = str(ref.get("build_hash"))
        record: dict[str, Any] = {
            "bundle_ref": str(path),
            "reference_id": ref.get("reference_id"),
            "build_hash": build_hash,
            "d1_bundle": D1_BUNDLES.get(build_hash),
        }
        bundle_dir = Path(str(ref.get("path")))
        created: datetime | None = None
        workers: list[int] = []
        try:
            created = _parse_time(_json(bundle_dir / "bundle.json")["created_at"])
        except (OSError, KeyError, ValueError) as error:
            record["bundle_json_error"] = str(error)
        try:
            summary = _json(bundle_dir / "resolvability_summary.json")
            workers = sorted(
                {int(run["n_processors"]) for run in summary.get("mapping_runs", [])}
            )
        except (OSError, KeyError, ValueError, TypeError) as error:
            record["resolvability_summary_error"] = str(error)
        record["created_at"] = None if created is None else created.isoformat()
        record["created_before_run"] = created is not None and created < run_start
        record["reuse_logged"] = bool(reuse_logged.get(build_hash, False))
        record["self_map_workers"] = workers
        reused = record["created_before_run"] or record["reuse_logged"]
        eight = workers == [SELF_MAP_WORKERS]
        record["ok"] = record["d1_bundle"] is not None and (reused or eight)
        if record["d1_bundle"] is None:
            check.reasons.append(
                f"{path}: build hash {build_hash[:16]} is not a D1 bundle"
            )
        elif not (reused or eight):
            check.reasons.append(
                f"{path}: bundle {build_hash[:16]} was built by the run and its "
                "self-map "
                f"records {workers} workers, not {SELF_MAP_WORKERS}"
            )
        check.bundles.append(record)
        seen[build_hash] = record
    if not check.bundles:
        check.reasons.append("no PREP bundle_ref.json found")
    for (pair, segmentation), manifest_path in sorted((map_manifests or {}).items()):
        try:
            manifest = _json(manifest_path)
        except (OSError, ValueError) as error:
            check.reasons.append(f"{manifest_path}: {error}")
            continue
        for sample_id, sample in sorted(manifest.get("samples", {}).items()):
            for run_id, run in sorted(sample.get("runs", {}).items()):
                build_hash = str(run.get("build_hash"))
                ok = build_hash in seen and seen[build_hash]["ok"]
                check.map_runs.append(
                    {
                        "pair": pair,
                        "segmentation": segmentation,
                        "sample_id": sample_id,
                        "run_id": run_id,
                        "build_hash": build_hash,
                        "ok": ok,
                    }
                )
                if not ok:
                    check.reasons.append(
                        f"MAP {pair} {segmentation} {sample_id} {run_id} mapped with "
                        f"{build_hash[:16]}, not a checked D1 bundle"
                    )
    check.ok = not check.reasons
    return check


REUSED_LINE = re.compile(
    r"annotation-reference-prep: (reused|built) (\S+) ([0-9a-f]{16})"
)


def prep_reuse_from_logs(texts: Iterable[str]) -> dict[str, bool]:
    """Return 16-hex build-hash prefix -> reused, from PREP task logs."""
    reuse: dict[str, bool] = {}
    for text in texts:
        for kind, _reference, prefix in REUSED_LINE.findall(text):
            reuse[prefix] = reuse.get(prefix, True) and kind == "reused"
    return reuse


# ---------------------------------------------------------------------------
# Nextflow trace (H14) and task evidence

_DURATION = re.compile(r"([0-9]*\.?[0-9]+)\s*(ms|s|m|h|d)")
_DURATION_UNITS: Final[dict[str, float]] = {
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}
_MEMORY = re.compile(r"([0-9]*\.?[0-9]+)\s*(B|KB|MB|GB|TB|PB)?", re.IGNORECASE)
_MEMORY_UNITS: Final[dict[str, float]] = {
    "B": 1024.0**-3,
    "KB": 1024.0**-2,
    "MB": 1024.0**-1,
    "GB": 1.0,
    "TB": 1024.0,
    "PB": 1024.0**2,
}


def parse_duration(text: Any) -> float:
    """Return seconds of a Nextflow duration (``1h 2m 3s``, ``57.7s``, ``300ms``)."""
    raw = "" if _is_missing(text) else str(text).strip()
    matches = _DURATION.findall(raw)
    if not matches or _DURATION.sub("", raw).strip():
        return math.nan
    return float(sum(float(number) * _DURATION_UNITS[unit] for number, unit in matches))


def parse_memory_gb(text: Any) -> float:
    """Return GB (1024^3 bytes, as Nextflow prints) of a trace memory value."""
    raw = "" if _is_missing(text) else str(text).strip()
    match = _MEMORY.fullmatch(raw)
    if not match:
        return math.nan
    unit = (match.group(2) or "B").upper()
    return float(match.group(1)) * _MEMORY_UNITS[unit]


def short_process(process: str) -> str:
    """Return the last component of a fully qualified process name."""
    return str(process).rsplit(":", 1)[-1]


def load_trace(paths: Sequence[Path]) -> pd.DataFrame:
    """Return one row per task name over one or more traces.

    As the M5 e2e ``trace_summary.py``: the last record of a task name wins,
    except that a ``CACHED`` record keeps an earlier computed record.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for row in frame.to_dict("records"):
            name = row["name"]
            if row.get("status") == "CACHED" and name in records:
                continue
            records[name] = {**row, "trace": Path(path).name}
    rows = []
    for row in records.values():
        rows.append(
            {
                "name": row["name"],
                "process": short_process(row.get("process", row["name"])),
                "tag": row.get("tag", ""),
                "status": row.get("status", ""),
                "exit": row.get("exit", ""),
                "cpus": row.get("cpus", ""),
                "realtime_s": parse_duration(row.get("realtime")),
                "peak_rss_gb": parse_memory_gb(row.get("peak_rss")),
                "workdir": row.get("workdir", ""),
                "trace": row["trace"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "process",
            "tag",
            "status",
            "exit",
            "cpus",
            "realtime_s",
            "peak_rss_gb",
            "workdir",
            "trace",
        ],
    )


def _tag_parts(tag: str) -> list[str]:
    return [part for part in str(tag).split(":") if part]


def h14_rows(
    trace: pd.DataFrame,
    pairs: Sequence[str],
    segmentations: Sequence[str],
    prep_bundles: Sequence[Mapping[str, Any]],
) -> list[Row]:
    """Return the H14 rows (resources; first measured at M8).

    Args:
        trace: ``load_trace`` output.
        pairs: The pairs.
        segmentations: The segmentations each pair must cover (all four:
            "annotation wall per pair (all segmentations)").
        prep_bundles: ``PrepCheck.bundles`` (each bundle's build time).

    Returns:
        Per pair the annotation wall and MAP peak RSS rows (scored on the
        development pairs), one PREP row per bundle and the no-GPU row.
    """
    rows: list[Row] = []
    done = trace["status"].isin(["COMPLETED", "CACHED"])
    for pair in pairs:
        tasks = trace[
            trace["process"].isin(ANNOTATION_PROCESSES)
            & trace["tag"].map(lambda tag, p=pair: _tag_parts(tag)[:1] == [p])
        ]
        covered = sorted(
            {_tag_parts(tag)[1] for tag in tasks["tag"] if len(_tag_parts(tag)) > 1}
        )
        missing = sorted(set(segmentations) - set(covered))
        incomplete = bool(missing) or not bool(done[tasks.index].all()) or tasks.empty
        wall_h = float(tasks["realtime_s"].sum()) / 3600.0
        scored = pair in DEV_PAIRS
        rows.append(
            Row(
                criterion="H14/annotation_wall",
                pair=pair,
                segmentation="all",
                dataset=pair,
                value=None if incomplete else wall_h,
                comparator="<=",
                threshold=H14_ANNOTATION_WALL_MAX_H,
                passes=None
                if incomplete
                else wall_h <= H14_ANNOTATION_WALL_MAX_H + 1e-12,
                scored=scored,
                source="trace",
                note=(
                    f"PANEL + MAP + RESOLVE realtime over {len(tasks)} tasks, "
                    f"segmentations {','.join(covered) or 'none'}"
                    + (f"; missing {','.join(missing)}" if missing else "")
                    + (
                        "; a task did not complete"
                        if not done[tasks.index].all()
                        else ""
                    )
                ),
            )
        )
        maps = tasks[tasks["process"] == MAP_PROCESS]
        peak = float(maps["peak_rss_gb"].max()) if len(maps) else math.nan
        rows.append(
            Row(
                criterion="H14/map_peak_rss",
                pair=pair,
                segmentation="all",
                dataset=pair,
                value=peak if math.isfinite(peak) else None,
                comparator="<=",
                threshold=H14_MAP_PEAK_RSS_MAX_GB,
                passes=(peak <= H14_MAP_PEAK_RSS_MAX_GB)
                if math.isfinite(peak)
                else None,
                scored=scored,
                source="trace",
                note=f"{len(maps)} MAP tasks (peak_rss, GB = 1024^3 bytes)",
            )
        )
    preps = trace[trace["process"] == PREP_PROCESS]
    prep_realtime = (
        f"; the run's {len(preps)} PREP tasks took "
        f"{float(preps['realtime_s'].max()):.0f} s at most"
        if len(preps)
        else ""
    )
    for bundle in prep_bundles:
        wall = _float(bundle.get("build_wall_time_s"))
        rows.append(
            Row(
                criterion="H14/prep_wall",
                pair="PREP",
                segmentation="all",
                dataset=(
                    f"{bundle.get('reference_id')} {str(bundle.get('build_hash'))[:16]}"
                ),
                value=wall / 3600.0 if math.isfinite(wall) else None,
                comparator="<=",
                threshold=H14_PREP_WALL_MAX_H,
                passes=(wall / 3600.0 <= H14_PREP_WALL_MAX_H)
                if math.isfinite(wall)
                else None,
                scored=True,
                source="bundle.json wall_time_s",
                note="the bundle's recorded build time incl. resolvability"
                + prep_realtime,
            )
        )
    gpu = sorted(set(trace["process"]) & GPU_PROCESSES)
    hidden = cuda_hidden_counts(trace)
    no_gpu = not gpu and hidden["without"] == 0
    rows.append(
        Row(
            criterion="H14/no_gpu",
            pair="all",
            segmentation="all",
            dataset="run",
            value=float(len(gpu) + hidden["without"]),
            comparator="==",
            threshold=0.0,
            passes=None if hidden["checked"] == 0 else no_gpu,
            scored=True,
            source="trace + task .command.run",
            note=(
                f"GPU processes run: {gpu or 'none'}; CUDA hidden in {hidden['with']} "
                f"of {hidden['checked']} task scripts checked ({hidden['absent']} "
                "work directories absent)"
            ),
        )
    )
    return [finalize(row) for row in rows]


def cuda_hidden_counts(trace: pd.DataFrame) -> dict[str, int]:
    """Count the tasks whose ``.command.run`` hides the GPUs (``CUDA_HIDDEN``)."""
    counts = {"checked": 0, "with": 0, "without": 0, "absent": 0}
    for workdir in trace["workdir"]:
        path = Path(str(workdir)) / ".command.run"
        if not workdir or not path.is_file():
            counts["absent"] += 1
            continue
        counts["checked"] += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        hidden = any(f"export {form}" in text for form in CUDA_HIDDEN)
        counts["with" if hidden else "without"] += 1
    return counts


# ---------------------------------------------------------------------------
# the report (acceptance_metrics.json)


def load_reports(
    runs_root: Path, pairs: Sequence[str], segmentations: Sequence[str]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return (pair, seg) -> metric records of each report found."""
    reports: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        for segmentation in segmentations:
            path = runs_root / REPORT_PATH.format(pair=pair, seg=segmentation)
            if path.is_file():
                reports[(pair, segmentation)] = list(_json(path)["metrics"])
    return reports


def _records(
    records: Iterable[Mapping[str, Any]], criterion: str, name: str, **where: Any
) -> list[Mapping[str, Any]]:
    out = []
    for record in records:
        if record.get("criterion") != criterion or record.get("name") != name:
            continue
        if all(record.get(key) == value for key, value in where.items()):
            out.append(record)
    return out


def _close(a: Any, b: Any) -> bool:
    x, y = _float(a), _float(b)
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isclose(x, y, rel_tol=REPORT_REL_TOL, abs_tol=1e-9)


# (criterion, note prefix) -> report metric name, for the dataset-level rows.
REPORT_METRICS: Final[dict[str, tuple[str, str]]] = {
    "H2": ("H2", "flag_implausible_share"),
    "H17/H2": ("H2", "flag_implausible_share"),
    "H3": ("H3", "whb_sea_agreement_ge20"),
    "H7": ("H7", "confident_coverage_table"),
    "H8/warning": ("H8", "gate_broad_coverage_segmented"),
}


def report_value(
    row: Row, reports: Mapping[tuple[str, str], list[dict[str, Any]]]
) -> tuple[bool, Any, str]:
    """Return (has a counterpart, the report's value, what was compared)."""
    records = reports.get((row.pair, row.segmentation))
    if records is None:
        return False, None, "no report"
    criterion = row.criterion
    if criterion in ("H1", "H17/H1"):
        found = [
            record
            for record in _records(records, "H1", "broad_jsd", region="whole_section")
            if str(record.get("kind") or "").endswith(":soft")
            and str(record.get("kind")).startswith("set_a")
        ]
        return (
            bool(found),
            found[0]["value"] if found else None,
            "H1 broad_jsd whole_section set_a:soft",
        )
    if criterion == "H5":
        name = (
            "confident_cop_supercluster_share"
            if row.note.startswith("confident COP")
            else "confident_broad_opc_share"
        )
        found = _records(records, "H5", name, sample_id=row.dataset)
        return bool(found), found[0]["value"] if found else None, f"H5 {name}"
    if criterion == "H16":
        found = _records(records, "H16", "uninformative_marking_consistent")
        return bool(found), found[0]["value"] if found else None, "H16 marking"
    if criterion in REPORT_METRICS:
        report_criterion, name = REPORT_METRICS[criterion]
        where: dict[str, Any] = {"sample_id": row.dataset}
        if report_criterion == "H7":
            where["level"] = "broad"
        found = _records(records, report_criterion, name, **where)
        return (
            bool(found),
            found[0]["value"] if found else None,
            f"{report_criterion} {name}",
        )
    return False, None, ""


def crosscheck(
    row: Row,
    reports: Mapping[tuple[str, str], list[dict[str, Any]]],
    samples: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    """Compare a criteria-table row with the report's value (empty: no counterpart)."""
    if row.criterion == "H8":
        records = reports.get((row.pair, row.segmentation))
        entry = samples.get((row.segmentation, row.dataset))
        if records is None or entry is None:
            return ""
        level = _records(records, "H8", "gate_level", sample_id=row.dataset)
        warning = _records(records, "H8", "gate_warning", sample_id=row.dataset)
        if not level or not warning:
            return f"{CROSSCHECK_MISMATCH}: no H8 gate records in the report"
        same = str(level[0]["value"]) == str(entry["gate_level"]) and _bool(
            warning[0]["value"]
        ) == _bool(entry["gate_warning"])
        return (
            "ok"
            if same
            else f"{CROSSCHECK_MISMATCH}: report gate {level[0]['value']} / "
            f"{warning[0]['value']} vs {entry['gate_level']} / {entry['gate_warning']}"
        )
    has, value, what = report_value(row, reports)
    if not has:
        if (row.pair, row.segmentation) in reports and what:
            return f"{CROSSCHECK_MISMATCH}: the report has no {what}"
        return ""
    if row.criterion == "H16":
        same = _bool(value) == bool(row.passes)
        return "ok" if same else f"{CROSSCHECK_MISMATCH}: report {what} {value}"
    return (
        "ok"
        if _close(value, row.value)
        else f"{CROSSCHECK_MISMATCH}: report {what} {value} vs {row.value}"
    )


# ---------------------------------------------------------------------------
# resolve_criteria.py rows (H1-H5, H7-H10, H16, H17) with D3 / D5 / D7


def criteria_rows(
    table: pd.DataFrame,
    samples: pd.DataFrame,
    h2_breakdown: pd.DataFrame,
    enrichment: pd.DataFrame | None,
    reports: Mapping[tuple[str, str], list[dict[str, Any]]],
    referee: pd.DataFrame | None,
) -> list[Row]:
    """Score ``resolve_criteria.py``'s criteria table with the §18 scopes.

    As the stage A3 scorer (``report_scoring`` checks): D3 on the H8
    warning row, D5 on H2 / H17-H2 (with ``h2_breakdown``'s confident broad
    and supercluster counts of the flagged cells), D7 on H4 (the headline
    set's classes of ``heldout_enrichment_m4.csv``). ``H4[<set>]`` rows are
    reported, never scored (D6). H10 is also compared with
    ``marker_referee.py``'s rows.
    """
    level = {
        (str(row["segmentation"]), str(row["sample_id"])): row
        for row in samples.to_dict("records")
    }
    h2_all = {
        (str(row["segmentation"]), str(row["sample_id"])): row
        for row in h2_breakdown[h2_breakdown["node"] == "all"].to_dict("records")
    }
    if referee is not None and {"first", "second"} <= set(referee.columns):
        # A referee.csv of resolve_criteria.py also holds its other two comparisons.
        referee = referee[
            (referee["first"] == REFEREE_NEW) & (referee["second"] == REFEREE_LEGACY)
        ]
    referee_index = (
        {}
        if referee is None
        else {
            (str(row["segmentation"]), str(row["sample_id"])): row
            for row in referee.to_dict("records")
        }
    )
    rows: list[Row] = []
    for record in table.to_dict("records"):
        criterion = str(record["criterion"])
        passes = _bool(record["passes"])
        row = Row(
            criterion=criterion,
            pair=str(record["pair"]),
            segmentation=str(record["segmentation"]),
            dataset=str(record["dataset"]),
            value=None if _is_missing(record["value"]) else record["value"],
            comparator=str(record["comparator"]),
            threshold=None if _is_missing(record["threshold"]) else record["threshold"],
            passes=passes,
            scored=bool(_bool(record["scored"])),
            source="resolve_criteria.py",
            note="" if _is_missing(record.get("note")) else str(record["note"])[:240],
        )
        if criterion.startswith("H4["):
            row.scored = False
            row.note = f"reported label set (D6); {row.note}"[:240]
            rows.append(finalize(row))
            continue
        row.crosscheck = crosscheck(row, reports, level)
        if criterion == "H10" and referee is not None:
            theirs = referee_index.get((row.segmentation, row.dataset))
            if theirs is None:
                row.crosscheck = f"{CROSSCHECK_MISMATCH}: not in marker_referee.py"
            elif not _close(theirs["first_wins"], row.value):
                row.crosscheck = (
                    f"{CROSSCHECK_MISMATCH}: marker_referee.py {theirs['first_wins']}"
                )
            else:
                row.crosscheck = "ok (marker_referee.py)"
        exception = None
        if passes is False and row.scored:
            exception = _criteria_exception(row, record, level, h2_all, enrichment)
        rows.append(finalize(row, exception))
    return rows


def _criteria_exception(
    row: Row,
    record: Mapping[str, Any],
    level: Mapping[tuple[str, str], Mapping[str, Any]],
    h2_all: Mapping[tuple[str, str], Mapping[str, Any]],
    enrichment: pd.DataFrame | None,
) -> sc.ExceptionVerdict | None:
    key = (row.segmentation, row.dataset)
    if row.criterion == "H8/warning":
        entry = level.get(key) or {}
        return sc.check_d3(
            segmentation=row.segmentation,
            dataset=row.dataset,
            value=_float(record["value"]),
            level=entry.get("gate_level"),
        )
    if row.criterion in ("H2", "H17/H2"):
        breakdown = h2_all.get(key) or {}
        broad = breakdown.get("n_broad_confident")
        supercluster = breakdown.get("n_supercluster_confident")
        return sc.check_d5(
            criterion=row.criterion,
            segmentation=row.segmentation,
            dataset=row.dataset,
            value=_float(record["value"]),
            n_confident_broad=None if _is_missing(broad) else int(broad),
            n_confident_supercluster=None
            if _is_missing(supercluster)
            else int(supercluster),
        )
    if row.criterion == "H4" and enrichment is not None:
        pair, platform = row.dataset.split("_", 1)
        classes = enrichment[
            (enrichment["label_set"] == H4_HEADLINE_SET)
            & (enrichment["pair"] == pair)
            & (enrichment["platform"] == platform)
        ]
        return sc.check_d7(
            segmentation=row.segmentation, dataset=row.dataset, classes=classes
        )
    return None


# ---------------------------------------------------------------------------
# H12 and H13 from the reports


def h12_rows(
    reports: Mapping[tuple[str, str], list[dict[str, Any]]], pairs: Sequence[str]
) -> tuple[list[Row], list[dict[str, Any]]]:
    """Score H12 per pair on the proseg_hybrid report (the square tiles).

    Other segmentations' reports are scored the same way and reported.
    """
    rows: list[Row] = []
    scores: list[dict[str, Any]] = []
    for pair in pairs:
        for segmentation in SEGMENTATIONS:
            records = reports.get((pair, segmentation))
            scored = segmentation == SCORED_SEGMENTATION
            if records is None:
                if scored:
                    rows.append(
                        finalize(
                            Row(
                                "H12",
                                pair,
                                segmentation,
                                pair,
                                None,
                                "==",
                                None,
                                None,
                                True,
                                "report",
                                note="no annotation report",
                            )
                        )
                    )
                continue
            try:
                score = sc.score_h12(records, pair_id=pair)
            except ValueError as error:
                rows.append(
                    finalize(
                        Row(
                            "H12",
                            pair,
                            segmentation,
                            pair,
                            None,
                            "==",
                            None,
                            None,
                            scored,
                            "report",
                            note=f"inconsistent report: {error}",
                        )
                    )
                )
                continue
            record = score.to_json() | {"segmentation": segmentation}
            scores.append(record)
            passes = (
                None if score.verdict == sc.NOT_AVAILABLE else score.verdict == sc.PASS
            )
            beside = record["reported_beside"]["ordering"]
            note = (
                f"ci_scored {score.ci_scored}; ordering {score.ordering} -> "
                f"{score.ordering_verdict} (required {score.ordering_required}); WM>GM "
                + "; ".join(
                    f"{platform} {value.get('value')} [{value.get('ci_low')}, "
                    f"{value.get('ci_high')}] {value['verdict']}"
                    for platform, value in sorted(score.wm_gm.items())
                )
                + f"; beside (tangential blocks, not scored): {beside}"
                + ("; " + "; ".join(score.notes) if score.notes else "")
            )
            rows.append(
                finalize(
                    Row(
                        "H12",
                        pair,
                        segmentation,
                        pair,
                        score.verdict,
                        "==",
                        "PASS",
                        passes,
                        scored,
                        "report_scoring.score_h12",
                        note=note[:400],
                    )
                )
            )
    return rows, scores


def _provenance_mender(path: Path) -> dict[str, dict[str, Any]]:
    try:
        samples = _json(path)["provenance"]["samples"]
    except (OSError, KeyError, ValueError):
        return {}
    return {sid: dict(value.get("mender") or {}) for sid, value in samples.items()}


def h13_rows(
    runs_root: Path,
    reports: Mapping[tuple[str, str], list[dict[str, Any]]],
    legacy_summary: Mapping[str, Any] | None,
    pairs: Sequence[str],
    *,
    state_report_path: str = STATE_REPORT_PATH,
) -> list[Row]:
    """Return H13's parts (contracts; first measured at M8).

    - ``H13/mender``: MENDER completes on the pair under both unassigned
      policies (the main run's proseg_hybrid report and the second policy's
      report: every platform's MENDER status ``complete`` with the expected
      policy). Scored on all 4 pairs (the criterion names them).
    - ``H13/depth_violins``: the published cortical-depth copy of every
      segmentation and platform holds violin plots.
    - ``H13/id_sets``: the zarr table's ids equal the clustered H5AD's
      (``table_ids_equal_clustered``) on every segmentation and platform.
    - ``H13/legacy_untouched``: ``compare_legacy.py untouched``.
    - ``H13/viewer``: the viewer on reseg is a manual check (not measured).

    The last four are scored on P7513 and P1212 (the flip rule).
    """
    rows: list[Row] = []
    for pair in pairs:
        dev = pair in DEV_PAIRS
        main_path = runs_root / REPORT_PATH.format(pair=pair, seg=SCORED_SEGMENTATION)
        state_path = runs_root / state_report_path.format(pair=pair)
        found = []
        for label, path in (("main", main_path), ("state", state_path)):
            mender = _provenance_mender(path) if path.is_file() else {}
            for platform in PLATFORMS:
                entry = mender.get(f"{pair}_{platform}", {})
                found.append(
                    (
                        label,
                        platform,
                        entry.get("status"),
                        entry.get("unassigned_state_policy"),
                    )
                )
        missing = [item for item in found if item[2] is None]
        ok = all(
            status == "complete" and policy == MENDER_POLICIES[label]
            for label, _platform, status, policy in found
        )
        rows.append(
            finalize(
                Row(
                    "H13/mender",
                    pair,
                    SCORED_SEGMENTATION,
                    pair,
                    float(sum(1 for item in found if item[2] == "complete")),
                    "==",
                    float(len(found)),
                    None if missing else ok,
                    True,
                    "report provenance",
                    note="; ".join(f"{a} {b} {c} {d}" for a, b, c, d in found),
                )
            )
        )
        violins: list[str] = []
        absent: list[str] = []
        for segmentation in SEGMENTATIONS:
            for platform in PLATFORMS:
                directory = runs_root / DEPTH_DIR.format(
                    pair=pair, seg=segmentation, platform=platform.lower()
                )
                (violins if any(directory.glob("*violin*")) else absent).append(
                    f"{segmentation}/{platform.lower()}"
                )
        rows.append(
            finalize(
                Row(
                    "H13/depth_violins",
                    pair,
                    "all",
                    pair,
                    float(len(violins)),
                    "==",
                    float(len(violins) + len(absent)),
                    not absent,
                    dev,
                    "published depth",
                    note=f"missing: {absent}"
                    if absent
                    else "every segmentation and platform",
                )
            )
        )
        states = []
        for segmentation in SEGMENTATIONS:
            records = reports.get((pair, segmentation), [])
            for platform in PLATFORMS:
                found_ids = _records(
                    records,
                    "H13",
                    "table_ids_equal_clustered",
                    sample_id=f"{pair}_{platform}",
                )
                states.append(_bool(found_ids[0]["value"]) if found_ids else None)
        rows.append(
            finalize(
                Row(
                    "H13/id_sets",
                    pair,
                    "all",
                    pair,
                    float(sum(1 for state in states if state)),
                    "==",
                    float(len(states)),
                    None if None in states else all(states),
                    dev,
                    "report",
                    note=(
                        f"table_ids_equal_clustered over {len(states)} datasets: "
                        f"{states}"
                    ),
                )
            )
        )
    passes = None if legacy_summary is None else bool(legacy_summary.get("passes"))
    rows.append(
        finalize(
            Row(
                "H13/legacy_untouched",
                "all",
                "all",
                "stores",
                None
                if legacy_summary is None
                else float(len(legacy_summary.get("failing_groups", []))),
                "==",
                0.0,
                passes,
                True,
                "compare_legacy.py untouched",
                note=(
                    "no compare_legacy output"
                    if legacy_summary is None
                    else f"{legacy_summary.get('n_legacy_tables')} legacy tables, "
                    f"{legacy_summary.get('n_hashed_files')} hashed files; failing "
                    f"{legacy_summary.get('failing_groups')}"
                ),
            )
        )
    )
    rows.append(
        finalize(
            Row(
                "H13/viewer",
                "all",
                "reseg",
                "viewer",
                None,
                "==",
                None,
                None,
                True,
                "manual",
                note="viewer on reseg: a manual check by the user",
            )
        )
    )
    return rows


# ---------------------------------------------------------------------------
# H6: the archived E2 HO contaminated simulation, reweighted per dataset


def soft_supercluster_mass(labels: pd.DataFrame) -> pd.Series:
    """Return the §5.5 soft mass of table cells per WHB supercluster name.

    Each table cell adds the bootstrap probability of its assigned WHB
    supercluster and of each runner-up (``mmc_whb_supercluster_*``).
    """
    table = (
        labels[labels["in_table"].to_numpy(bool)] if "in_table" in labels else labels
    )
    prefix = "mmc_whb_supercluster"
    columns = [(f"{prefix}_name", f"{prefix}_bp")] + [
        (f"{prefix}_runner_up_{index}_name", f"{prefix}_runner_up_{index}_bp")
        for index in range(1, N_RUNNERS_UP + 1)
    ]
    parts = []
    for name, bp in columns:
        if name not in table or bp not in table:
            continue
        part = pd.DataFrame(
            {
                "name": table[name].astype(object).to_numpy(),
                "mass": np.nan_to_num(table[bp].to_numpy(np.float64), nan=0.0),
            }
        )
        parts.append(part[part["name"].notna()])
    if not parts:
        return pd.Series(dtype=float)
    mass = pd.concat(parts).groupby("name")["mass"].sum()
    return mass[mass > 0]


def truth_map(cells: pd.DataFrame) -> dict[str, str]:
    """Return the simulation's own supercluster -> broad truth mapping."""
    pairs = cells[["true_super", "true_broad"]].dropna().drop_duplicates()
    duplicated = pairs["true_super"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "truth superclusters with two broad classes: "
            f"{sorted(pairs[duplicated]['true_super'])}"
        )
    return dict(zip(pairs["true_super"], pairs["true_broad"], strict=True))


def dataset_composition(
    mass: pd.Series, supercluster_to_broad: Mapping[str, str]
) -> dict[str, pd.Series]:
    """Return a dataset's shares over the simulation's truth classes per level."""
    kept = mass[mass.index.isin(list(supercluster_to_broad))]
    total = float(kept.sum())
    if total <= 0:
        return {"supercluster": pd.Series(dtype=float), "broad": pd.Series(dtype=float)}
    supercluster = kept / total
    broad = supercluster.groupby(supercluster.index.map(supercluster_to_broad)).sum()
    return {"supercluster": supercluster, "broad": broad}


def _confident(cells: pd.DataFrame, level: str) -> np.ndarray:
    broad = cells["ho_broad_p"].to_numpy(np.float64) >= (
        H6_THRESHOLDS["broad"] - THRESHOLD_SLACK
    )
    if level == "broad":
        return broad
    return broad & (
        cells["ho_super_p"].to_numpy(np.float64)
        >= H6_THRESHOLDS["supercluster"] - THRESHOLD_SLACK
    )


def h6_precision(
    cells: pd.DataFrame, composition: Mapping[str, pd.Series] | None
) -> list[dict[str, Any]]:
    """Return the H6 precision rows of one composition (``None``: unweighted).

    Args:
        cells: The E2 in-silico cells of the scored scenario.
        composition: ``dataset_composition`` output, or ``None``.

    Returns:
        One row per level, depth and class with >= 50 test cells, plus per
        level and depth a ``pooled`` row (class ``pooled``): the reweighted
        precision of every confident call of those classes together, the
        other reading of "classes with n >= 50", for information only.
    """
    rows = []
    for level, (truth_column, call_column, _) in H6_COLUMNS.items():
        for depth in H6_DEPTHS[level]:
            at = cells[cells["D"].astype(int) == depth]
            truth = at[truth_column].astype(object).to_numpy()
            call = at[call_column].astype(object).to_numpy()
            confident = _confident(at, level)
            simulated = pd.Series(truth).value_counts(normalize=True)
            if composition is None:
                weights = np.ones(len(at))
            else:
                shares = composition[level]
                weights = np.array(
                    [
                        float(shares.get(value, 0.0)) / float(simulated[value])
                        for value in truth
                    ]
                )
            counts = pd.Series(truth).value_counts()
            evaluated = [
                cls for cls in sorted(counts.index) if counts[cls] >= H6_MIN_TEST_CELLS
            ]
            pooled_calls = confident & np.isin(
                call, np.asarray(evaluated, dtype=object)
            )
            pooled_mass = float(weights[pooled_calls].sum())
            pooled = (
                float(weights[pooled_calls & (call == truth)].sum()) / pooled_mass
                if pooled_mass > 0
                else math.nan
            )
            rows.append(
                {
                    "level": level,
                    "depth": depth,
                    "class": H6_POOLED,
                    "n_test_cells": int(sum(counts[cls] for cls in evaluated)),
                    "n_confident_calls": int(pooled_calls.sum()),
                    "precision": pooled,
                    "target": H6_TARGETS[level],
                    "evaluable": math.isfinite(pooled),
                    "passes": (pooled >= H6_TARGETS[level] - 1e-12)
                    if math.isfinite(pooled)
                    else None,
                }
            )
            for cls in evaluated:
                calls = confident & (call == cls)
                mass = float(weights[calls].sum())
                correct = float(weights[calls & (truth == cls)].sum())
                precision = correct / mass if mass > 0 else math.nan
                rows.append(
                    {
                        "level": level,
                        "depth": depth,
                        "class": cls,
                        "n_test_cells": int(counts[cls]),
                        "n_confident_calls": int(calls.sum()),
                        "precision": precision,
                        "target": H6_TARGETS[level],
                        "evaluable": math.isfinite(precision),
                        "passes": (precision >= H6_TARGETS[level] - 1e-12)
                        if math.isfinite(precision)
                        else None,
                    }
                )
    return rows


def h6_rows(
    cells: pd.DataFrame, masses: Mapping[str, pd.Series]
) -> tuple[list[Row], pd.DataFrame]:
    """Return H6's rows (one per proseg_hybrid dataset) and its detail table."""
    cells = cells[cells["scenario"] == H6_SCENARIO]
    mapping = truth_map(cells)
    detail = []
    for row in h6_precision(cells, None):
        detail.append({"dataset": "unweighted", **row})
    rows: list[Row] = []
    for dataset, mass in sorted(masses.items()):
        composition = dataset_composition(mass, mapping)
        everything = h6_precision(cells, composition)
        detail += [{"dataset": dataset, **row} for row in everything]
        measured = [row for row in everything if row["class"] != H6_POOLED]
        pooled = [row for row in everything if row["class"] == H6_POOLED]
        evaluable = [row for row in measured if row["evaluable"]]
        failing = [
            f"{row['level']} {row['class']} D{row['depth']} {row['precision']:.3f}"
            for row in evaluable
            if not row["passes"]
        ]
        pair = dataset.split("_")[0]
        rows.append(
            finalize(
                Row(
                    "H6",
                    pair,
                    SCORED_SEGMENTATION,
                    dataset,
                    float(len(failing)),
                    "==",
                    0.0,
                    None if not evaluable else not failing,
                    True,
                    "archived E2 HO contam simulation",
                    note=(
                        f"{len(evaluable)} class x depth rows evaluable of "
                        f"{len(measured)}; "
                        + (
                            f"failing: {'; '.join(failing[:8])}"
                            if failing
                            else "none failing"
                        )
                    )[:400],
                )
            )
        )
        pooled_failing = [
            f"{row['level']} D{row['depth']} {row['precision']:.3f}"
            for row in pooled
            if row["evaluable"] and not row["passes"]
        ]
        rows.append(
            finalize(
                Row(
                    "H6/pooled",
                    pair,
                    SCORED_SEGMENTATION,
                    dataset,
                    float(len(pooled_failing)),
                    "==",
                    0.0,
                    None if not pooled else not pooled_failing,
                    False,
                    "archived E2 HO contam simulation",
                    note=(
                        "information: every class with >= 50 test cells pooled per "
                        "level and depth (the other reading of H6); "
                        + (
                            f"failing: {'; '.join(pooled_failing)}"
                            if pooled_failing
                            else "none failing"
                        )
                    )[:400],
                )
            )
        )
    return rows, pd.DataFrame(detail)


# ---------------------------------------------------------------------------
# H15


def frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str]:
    """Return whether two DataFrames have identical content, and the difference."""
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as error:
        return False, " ".join(str(error).split())[:300]
    return True, ""


def compare_parquets(first: Path, second: Path) -> list[dict[str, Any]]:
    """Compare every parquet under two output directories (same relative paths)."""
    rows = []
    names = sorted(
        {str(path.relative_to(first)) for path in first.rglob("*.parquet")}
        | {str(path.relative_to(second)) for path in second.rglob("*.parquet")}
    )
    for name in names:
        a, b = first / name, second / name
        if not (a.is_file() and b.is_file()):
            rows.append(
                {"file": name, "equal": False, "difference": "missing on one side"}
            )
            continue
        equal, difference = frames_equal(pd.read_parquet(a), pd.read_parquet(b))
        rows.append({"file": name, "equal": equal, "difference": difference})
    return rows


def confident_broad(labels: pd.DataFrame) -> pd.Series:
    """Return each table cell's confident broad name (``None`` if not confident)."""
    table = labels[labels["in_table"].to_numpy(bool)].set_index("cell_id")
    confident = table["ct_broad_status"].astype(str).to_numpy() == "confident"
    return pd.Series(
        np.where(confident, table["ct_broad_name"].astype(object).to_numpy(), None),
        index=table.index.astype(str),
        dtype=object,
    )


def seed_change(seed0: pd.DataFrame, seed1: pd.DataFrame) -> dict[str, Any]:
    """Return the share of seed-0 confident broad labels that seed 1 changes.

    A label changes when a cell is confident under one seed only, or under
    both with different names; the share is over the seed-0 confident cells.
    """
    a = confident_broad(seed0)
    b = confident_broad(seed1).reindex(a.index)
    conf_a = a.notna().to_numpy()
    conf_b = b.notna().to_numpy()
    same = conf_a & conf_b & (a.to_numpy() == b.to_numpy())
    changed = (conf_a | conf_b) & ~same
    n_a = int(conf_a.sum())
    return {
        "n_cells": len(a),
        "n_confident_seed0": n_a,
        "n_confident_seed1": int(conf_b.sum()),
        "n_changed": int(changed.sum()),
        "share_changed": float(changed.sum() / n_a) if n_a else math.nan,
        "cells_missing_in_seed1": int(
            confident_broad(seed1).index.symmetric_difference(a.index).size
        ),
    }


def output_dir_problem(path: Path | None) -> str:
    """Why a task output directory cannot be compared (empty: it can)."""
    if path is None:
        return "not given"
    if not path.is_dir():
        return f"{path} does not exist"
    if not any(path.rglob("*.parquet")):
        return f"{path} holds no parquet"
    return ""


def h15_compare(
    pair: str,
    pipeline_map: Path,
    pipeline_resolve: Path,
    rerun_map: Path | None,
    rerun_resolve: Path | None,
    seed1_resolve: Path | None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the H15 record of one pair (``h15`` subcommand).

    A re-run output that is absent or empty (its task failed or never ran)
    leaves the comparison unmeasured (``identical`` / ``seed1`` ``None``,
    with the reason), never a difference: H15 measures the re-run's content,
    not whether the re-run could be staged.
    """
    record: dict[str, Any] = {
        "pair": pair,
        "segmentation": SCORED_SEGMENTATION,
        "notes": list(notes),
    }
    if rerun_map is not None or rerun_resolve is not None:
        problems = [
            f"{what}: {problem}"
            for what, path in (
                ("pipeline MAP", pipeline_map),
                ("pipeline RESOLVE", pipeline_resolve),
                ("re-run MAP", rerun_map),
                ("re-run RESOLVE", rerun_resolve),
            )
            if (problem := output_dir_problem(path))
        ]
        if problems or rerun_map is None or rerun_resolve is None:
            record["identical"] = None
            record["rerun_problem"] = "; ".join(problems)
        else:
            files = compare_parquets(pipeline_map, rerun_map) + compare_parquets(
                pipeline_resolve, rerun_resolve
            )
            record["identical_files"] = files
            record["identical"] = bool(files) and all(item["equal"] for item in files)
    if seed1_resolve is not None:
        names = {
            f"{pair}_{platform}": f"{platform.lower()}/{pair}_{platform}"
            "_celltype_labels.parquet"
            for platform in PLATFORMS
        }
        absent = [
            str(root / name)
            for name in names.values()
            for root in (pipeline_resolve, seed1_resolve)
            if not (root / name).is_file()
        ]
        if absent:
            record["seed1"] = None
            record["seed1_problem"] = f"labels missing: {absent}"
            return record
        per_sample = {}
        for sample, name in names.items():
            per_sample[sample] = seed_change(
                pd.read_parquet(pipeline_resolve / name),
                pd.read_parquet(seed1_resolve / name),
            )
        n_changed = sum(item["n_changed"] for item in per_sample.values())
        n_confident = sum(item["n_confident_seed0"] for item in per_sample.values())
        record["seed1"] = {
            "per_sample": per_sample,
            "n_changed": n_changed,
            "n_confident_seed0": n_confident,
            "share_changed": n_changed / n_confident if n_confident else math.nan,
        }
    return record


def h15_rows(records: Mapping[str, Mapping[str, Any]]) -> list[Row]:
    """Return H15's rows on P7513 and P1212 (first measured at M8)."""
    rows: list[Row] = []
    for pair in H15_PAIRS:
        record = records.get(pair, {})
        identical = record.get("identical")
        files = record.get("identical_files", [])
        rows.append(
            finalize(
                Row(
                    "H15/identical_rerun",
                    pair,
                    SCORED_SEGMENTATION,
                    pair,
                    None
                    if identical is None
                    else float(sum(1 for f in files if not f["equal"])),
                    "==",
                    0.0,
                    None if identical is None else bool(identical),
                    True,
                    "h15",
                    note=(
                        "not measured: "
                        + str(record.get("rerun_problem") or "no re-run record")
                        if identical is None
                        else f"{len(files)} tables compared; differing: "
                        + str([f["file"] for f in files if not f["equal"]][:6])
                    ),
                )
            )
        )
        seed1 = record.get("seed1")
        share = None if seed1 is None else _float(seed1.get("share_changed"))
        rows.append(
            finalize(
                Row(
                    "H15/seed1",
                    pair,
                    SCORED_SEGMENTATION,
                    pair,
                    share,
                    "<=",
                    H15_SEED_CHANGE_MAX,
                    None
                    if share is None or not math.isfinite(share)
                    else share <= H15_SEED_CHANGE_MAX + 1e-12,
                    True,
                    "h15",
                    note="not measured: "
                    + str(record.get("seed1_problem") or "no seed-1 record")
                    if seed1 is None
                    else f"{seed1['n_changed']} of {seed1['n_confident_seed0']} "
                    "seed-0 confident broad labels",
                )
            )
        )
    return rows


# ---------------------------------------------------------------------------
# H18 from the draw spread (the scored draw), D4


def h18_rows(
    draw_dir: Path, reference: pd.DataFrame | None
) -> tuple[list[Row], list[str]]:
    """Score H18 on the scored draw (D4's scope), as the stage A3 scorer.

    Returns:
        The rows and the draw-spread run problems (back to the user).
    """
    problems: list[str] = []
    run = _json(draw_dir / "draw_spread_run.json")
    if run.get("bundle_build_hash") != SET_A_WHB:
        problems.append(
            f"draw spread bundle {run.get('bundle_build_hash')} is not set a WHB D1"
        )
    if SCORED_DRAW not in run.get("draws", []):
        problems.append(f"the scored draw {SCORED_DRAW} is not in the draw spread")
    check = run.get("scored_draw_check") or {}
    if not check:
        problems.append(
            "the draw spread records no scored_draw_check: whether the scored draw "
            "reproduces the stored rows is unknown"
        )
    elif check.get("skipped"):
        problems.append(
            "the draw spread skipped the scored-draw check (--no-check-scored): "
            "whether the scored draw reproduces the stored rows is unknown"
        )
    elif check.get("reproduces") is not True:
        problems.append(f"the scored draw does not reproduce the stored rows: {check}")
    if (
        run.get("n_processors") != SELF_MAP_WORKERS
        or run.get("self_map_n_processors") != SELF_MAP_WORKERS
    ):
        problems.append(
            f"draw spread workers {run.get('n_processors')} / self-map "
            f"{run.get('self_map_n_processors')} (protocol item 2: {SELF_MAP_WORKERS})"
        )
    rows: list[Row] = []
    h18 = pd.read_csv(draw_dir / "draw_spread_h18.csv")
    h18 = h18[(h18["draw"] == SCORED_DRAW) & h18["required"].astype(bool)]
    for record in h18.to_dict("records"):
        passes = bool(_bool(record["passes"]))
        missing = _missing_set(record["missing"])
        exception = (
            None
            if passes
            else sc.check_d4_h18(
                dataset=str(record["dataset"]),
                level=str(record["level"]),
                cls=str(record["class"]),
                missing=missing,
            )
        )
        rows.append(
            finalize(
                Row(
                    f"H18/{record['level']}",
                    str(record["dataset"]).split("_")[0],
                    SCORED_SEGMENTATION,
                    f"{record['dataset']} {record['class']}",
                    float(len(missing)),
                    "==",
                    0.0,
                    passes,
                    True,
                    "draw_spread.py c0e0",
                    note=f"expected {record['expected']}; emitted {record['emitted']}; "
                    f"missing {sorted(missing)}; floor {record['floor']} "
                    f"({record.get('floor_family')})",
                ),
                exception,
            )
        )
    raised = pd.read_csv(draw_dir / "draw_spread_would_raise.csv")
    raised = raised[raised["draw"] == SCORED_DRAW]
    reference_of: dict[str, list[tuple[str, str, int]]] = {}
    if reference is not None:
        reference_of = {
            str(dataset): [
                (str(a), str(b), int(c))
                for a, b, c in group[["level", "class", "depth"]].itertuples(
                    index=False
                )
            ]
            for dataset, group in reference.groupby("dataset")
        }
    datasets = sorted(
        set(h18["dataset"].astype(str)) | set(raised["dataset"].astype(str))
    )
    for dataset in datasets:
        bins = [
            (str(a), str(b), int(c))
            for a, b, c in raised[raised["dataset"] == dataset][
                ["level", "class", "depth"]
            ].itertuples(index=False)
        ]
        exception = sc.check_d4_would_raise(
            dataset=dataset,
            bins=bins,
            reference=None if reference is None else reference_of.get(dataset, []),
        )
        rows.append(
            finalize(
                Row(
                    "H18/would_raise",
                    dataset.split("_")[0],
                    SCORED_SEGMENTATION,
                    dataset,
                    float(len(bins)),
                    "listed",
                    None,
                    exception is None,
                    True,
                    "draw_spread.py c0e0",
                    note="; ".join(f"{a} {b} {c}" for a, b, c in bins)[:300],
                ),
                exception,
            )
        )
    rows.append(
        finalize(
            Row(
                "H18/draw_check",
                "all",
                SCORED_SEGMENTATION,
                "draw_spread_run.json",
                float(len(problems)),
                "==",
                0.0,
                not problems,
                True,
                "draw_spread.py",
                note="; ".join(problems)
                or "set a WHB D1, scored draw c0e0 stored and reproduced, 8 workers",
            )
        )
    )
    return rows, problems


# ---------------------------------------------------------------------------
# provenance, summary and HTML


def file_sha256(path: Path) -> str:
    """Return a file's sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity() -> dict[str, Any]:
    """Return the code commit (``MERXEN_CODE_COMMIT``, ``COMMIT``, git) and hashes."""
    script = Path(__file__).resolve()
    root = script.parents[2]
    commit: str | None = None
    source = None
    if os.environ.get("MERXEN_CODE_COMMIT", "").strip():
        commit, source = (
            os.environ["MERXEN_CODE_COMMIT"].strip(),
            "env MERXEN_CODE_COMMIT",
        )
    elif (root / "COMMIT").is_file():
        commit, source = (
            (root / "COMMIT").read_text(encoding="utf-8").strip(),
            str(root / "COMMIT"),
        )
    else:
        try:
            top = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            if top and Path(top[0]).resolve() == root:
                commit, source = top[1], "git"
        except (OSError, subprocess.CalledProcessError):
            pass
    modules = {
        "run_acceptance.py": script,
        "resolve_criteria.py": script.parent / "resolve_criteria.py",
        "report_scoring.py": Path(sc.__file__).resolve(),
    }
    return {
        "git_commit": commit,
        "git_commit_source": source,
        "sha256": {name: file_sha256(path) for name, path in modules.items()},
    }


def counts(rows: Sequence[Row]) -> dict[str, int]:
    """Return the number of rows per verdict."""
    out: dict[str, int] = {}
    for row in rows:
        out[row.verdict] = out.get(row.verdict, 0) + 1
    return dict(sorted(out.items()))


_CSS = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1f2328;background:#fff;max-width:1400px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;font-size:12px;width:100%}
th,td{border:1px solid #d0d7de;padding:3px 6px;text-align:left;vertical-align:top}
th{background:#f6f8fa}.back{background:#fff4e5}.pass{color:#1a7f37}.fail{color:#cf222e}
.small{color:#57606a;font-size:12px}
@media (prefers-color-scheme: dark){body{background:#0d1117;color:#e6edf3}
th{background:#161b22}th,td{border-color:#30363d}.back{background:#3d2a12}.small{color:#9198a1}}
"""


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.4g}"
    return html.escape("" if value is None else str(value))


def _table(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for record in records:
        css = ' class="back"' if record.get("back_to_user") else ""
        cells = "".join(f"<td>{_cell(record.get(column))}</td>" for column in columns)
        body.append(f"<tr{css}>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


ROW_COLUMNS: Final = (
    "criterion",
    "pair",
    "segmentation",
    "dataset",
    "value",
    "comparator",
    "threshold",
    "verdict",
    "back_to_user_reason",
    "exception",
    "scope_check",
    "crosscheck",
    "source",
    "note",
)


def write_html(summary: Mapping[str, Any], path: Path) -> None:
    """Write the summary page (static; no external resources)."""
    rows = summary["rows"]
    back = [row for row in rows if row["back_to_user"]]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>M8 Human Acceptance</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>M8 human acceptance (gate H): stage B scoring</h1>",
        f"<p class='small'>Date {html.escape(str(summary['date']))}; code "
        f"{html.escape(str(summary['code'].get('git_commit')))}; scored CI "
        f"{html.escape(summary['protocol']['h12_ci_scored'])}; scored draw "
        f"{html.escape(summary['protocol']['scored_draw'])}; H4 set "
        f"{html.escape(summary['protocol']['h4_scored_set'])}. The report records "
        "metrics; this page scores them (pre-registration §18). Rows marked in "
        "orange go back to the user.</p>",
        "<h2>P5 check (the D1 bundles): "
        + ("passed" if summary["p5_check"]["ok"] else "FAILED: nothing scored")
        + "</h2>",
        _table(
            summary["p5_check"]["bundles"],
            (
                "reference_id",
                "build_hash",
                "d1_bundle",
                "created_before_run",
                "reuse_logged",
                "self_map_workers",
                "ok",
            ),
        ),
        "<h2>Verdicts</h2>",
        _table(
            [{"verdict": k, "rows": v} for k, v in summary["verdict_counts"].items()],
            ("verdict", "rows"),
        ),
        f"<h2>Back to the user ({len(back)})</h2>",
        _table(back, ROW_COLUMNS),
        "<h2>Problems</h2><ul>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in summary["problems"])
        + "</ul>",
        "<h2>Every row</h2>",
        _table(rows, ROW_COLUMNS),
        "<h2>H12 (square tiles scored; tangential blocks beside, not scored)</h2>",
        _table(
            [
                {
                    **item,
                    "ordering": json.dumps(item["ordering"]),
                    "wm_gm": json.dumps(item["wm_gm"]),
                    "reported_beside": json.dumps(item["reported_beside"]),
                }
                for item in summary["h12"]
            ],
            (
                "pair_id",
                "segmentation",
                "verdict",
                "ci_scored",
                "ordering",
                "ordering_verdict",
                "wm_gm",
                "wm_gm_verdict",
                "reported_beside",
            ),
        ),
        "<h2>Inputs</h2>",
        _table(
            [{"input": k, "sha256": v} for k, v in summary["inputs"].items()],
            ("input", "sha256"),
        ),
        "</body></html>",
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _unscored(rows: Sequence[Row], reasons: Sequence[str]) -> list[Row]:
    note = "; ".join(reasons)[:200]
    for row in rows:
        row.verdict = VERDICT_UNSCORED
        row.note = f"{row.note}; P5: {note}" if row.note else f"P5: {note}"
    return list(rows)


# ---------------------------------------------------------------------------
# CLI


def _bundle_refs(prep_refs: Path) -> list[Path]:
    return sorted(prep_refs.rglob("bundle_ref.json")) if prep_refs.is_dir() else []


def _prep_logs(trace: pd.DataFrame) -> list[str]:
    texts = []
    for workdir in trace.loc[trace["process"] == PREP_PROCESS, "workdir"]:
        for name in (".command.out", ".command.log", ".command.err"):
            path = Path(str(workdir)) / name
            if workdir and path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return texts


def map_manifests(
    runs_root: Path, pairs: Sequence[str], segmentations: Sequence[str]
) -> dict[tuple[str, str], Path]:
    """Return (pair, run) -> ``map_manifest.json`` of every MAP run published.

    The main layout's runs are keyed by segmentation; the second MENDER
    policy's proseg_hybrid MAP by ``proseg_hybrid/mender_state_policy``, so
    the P5 check covers every MAP the acceptance run made.
    """
    found: dict[tuple[str, str], Path] = {}
    for pair in pairs:
        for segmentation in segmentations:
            path = runs_root / MAP_MANIFEST_PATH.format(pair=pair, seg=segmentation)
            if path.is_file():
                found[(pair, segmentation)] = path
        state = runs_root / STATE_MAP_MANIFEST_PATH.format(pair=pair)
        if state.is_file():
            found[(pair, f"{SCORED_SEGMENTATION}/{STATE_LABEL}")] = state
    return found


def _read_optional(path: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path is not None and path.is_file() else None


def score(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ``score`` subcommand; return the summary."""
    pairs = [item for item in args.datasets.split(",") if item]
    segmentations = (
        list(SEGMENTATIONS)
        if args.segmentations == "all"
        else [item for item in args.segmentations.split(",") if item]
    )
    inputs: dict[str, str] = {}

    def track(path: Path) -> Path:
        if path.is_file():
            inputs[str(path)] = file_sha256(path)
        return path

    problems: list[str] = []
    trace = load_trace([track(path) for path in args.trace])
    reuse = prep_reuse_from_logs(_prep_logs(trace))
    refs = [track(path) for path in _bundle_refs(args.prep_refs)]
    manifests = map_manifests(args.runs_root, pairs, segmentations)
    for path in manifests.values():
        track(path)
    prep = check_prep_bundles(
        refs,
        _parse_time(args.run_start),
        reuse_logged={
            ref_hash: reuse[ref_hash[:16]]
            for ref_hash in D1_BUNDLES
            if ref_hash[:16] in reuse
        },
        map_manifests=manifests,
    )
    for bundle in prep.bundles:
        path = Path(_json(Path(bundle["bundle_ref"]))["path"]) / "bundle.json"
        bundle["build_wall_time_s"] = (
            _json(path).get("wall_time_s") if path.is_file() else None
        )
    reports = load_reports(args.runs_root, pairs, segmentations)
    for pair, segmentation in reports:
        track(args.runs_root / REPORT_PATH.format(pair=pair, seg=segmentation))
    criteria = args.criteria
    table = pd.read_csv(track(criteria / "criteria_table.csv"))
    samples = pd.read_csv(track(criteria / "criteria_samples.csv"))
    h2 = pd.read_csv(track(criteria / "h2_breakdown.csv"))
    enrichment = _read_optional(track(criteria / "heldout_enrichment_m4.csv"))
    if enrichment is None:
        problems.append("no heldout_enrichment_m4.csv: D7's scope cannot be checked")
    referee = (
        _read_optional(track(args.referee / "referee.csv")) if args.referee else None
    )
    rows = criteria_rows(table, samples, h2, enrichment, reports, referee)
    h12, h12_scores = h12_rows(reports, pairs)
    rows += h12
    legacy = None
    if args.legacy is not None and (args.legacy / "legacy_untouched.json").is_file():
        legacy = _json(track(args.legacy / "legacy_untouched.json"))
    else:
        problems.append("no compare_legacy.py untouched output (H13 legacy tables)")
    rows += h13_rows(args.runs_root, reports, legacy, pairs)
    rows += h14_rows(trace, pairs, segmentations, prep.bundles)
    h15_records = {}
    if args.h15 is not None:
        for path in sorted(args.h15.glob("h15_*.json")):
            record = _json(track(path))
            h15_records[str(record["pair"])] = record
    rows += h15_rows(h15_records)
    masses = {}
    for pair in pairs:
        summary_path = (
            args.runs_root
            / pair
            / SCORED_SEGMENTATION
            / "annotation_resolve"
            / "annotation_resolve_out"
        )
        for platform in PLATFORMS:
            sample = f"{pair}_{platform}"
            labels_path = (
                summary_path / platform.lower() / f"{sample}_celltype_labels.parquet"
            )
            if labels_path.is_file():
                masses[sample] = soft_supercluster_mass(
                    pd.read_parquet(track(labels_path))
                )
    h6_detail = pd.DataFrame()
    if args.e2_cells is not None and args.e2_cells.is_file():
        cells = pd.read_csv(track(args.e2_cells), dtype={"D": str})
        h6, h6_detail = h6_rows(cells, masses)
        rows += h6
    else:
        problems.append("no E2 in-silico cells: H6 not measured")
        rows.append(
            finalize(
                Row(
                    "H6",
                    "all",
                    SCORED_SEGMENTATION,
                    "all",
                    None,
                    "==",
                    0.0,
                    None,
                    True,
                    "E2",
                    note="not measured",
                )
            )
        )
    reference = _read_optional(track(args.d4_reference)) if args.d4_reference else None
    if (
        args.draw_spread is not None
        and (args.draw_spread / "draw_spread_run.json").is_file()
    ):
        for name in (
            "draw_spread_run.json",
            "draw_spread_h18.csv",
            "draw_spread_would_raise.csv",
            "draw_spread.csv",
        ):
            track(args.draw_spread / name)
        h18, draw_problems = h18_rows(args.draw_spread, reference)
        rows += h18
        problems += draw_problems
    else:
        problems.append("no draw spread: H18 not measured")
        rows.append(
            finalize(
                Row(
                    "H18",
                    "all",
                    SCORED_SEGMENTATION,
                    "all",
                    None,
                    "==",
                    0.0,
                    None,
                    True,
                    "draw_spread.py",
                    note="not measured",
                )
            )
        )
    if not prep.ok:
        rows = _unscored(rows, prep.reasons)
        problems += [f"P5: {reason}" for reason in prep.reasons]
    records = [row.to_json() for row in rows]
    summary = {
        "schema": "merxen-m8-acceptance/1",
        "date": args.date,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_identity(),
        "protocol": {
            "preregistration": "docs/acceptance/annotation-v1-preregistration.md §18",
            "scored_draw": SCORED_DRAW,
            "scored_draw_workers": SELF_MAP_WORKERS,
            "h4_scored_set": H4_HEADLINE_SET,
            "h12_ci_scored": sc.SQUARE_TILE_CI,
            "h12_reported_beside": "tangential_block_500um (M7 D23, not scored)",
            "exception_tolerances": sc.TOLERANCES,
            "first_measurements": sorted(FIRST_MEASUREMENTS),
        },
        "datasets": pairs,
        "segmentations": segmentations,
        "p5_check": dataclasses.asdict(prep),
        "verdict_counts": counts(rows),
        "n_back_to_user": sum(1 for record in records if record["back_to_user"]),
        "all_scored_rows_pass_or_inside_an_exception": not any(
            record["back_to_user"] and record["scored"] for record in records
        ),
        "nothing_back_to_user": not problems
        and not any(record["back_to_user"] for record in records),
        "problems": problems,
        "h12": h12_scores,
        "rows": records,
        "inputs": inputs,
    }
    out = args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "tables" / "criteria_scored.csv", index=False)
    if len(h6_detail):
        h6_detail.to_csv(out / "tables" / "h6_precision.csv", index=False)
    trace.to_csv(out / "tables" / "trace_tasks.csv", index=False)
    pd.DataFrame(prep.bundles).to_csv(out / "tables" / "p5_bundles.csv", index=False)
    pd.DataFrame(prep.map_runs).to_csv(out / "tables" / "p5_map_runs.csv", index=False)
    if (
        args.draw_spread is not None
        and (args.draw_spread / "draw_spread.csv").is_file()
    ):
        shutil.copy2(
            args.draw_spread / "draw_spread.csv", out / "tables" / "draw_spread.csv"
        )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8"
    )
    write_html(summary, out / "summary.html")
    return summary


def h15(args: argparse.Namespace) -> dict[str, Any]:
    """Run the ``h15`` subcommand; return the record."""
    record = h15_compare(
        args.pair,
        args.pipeline_map,
        args.pipeline_resolve,
        args.rerun_map,
        args.rerun_resolve,
        args.seed1_resolve,
        notes=args.note,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    """Score the M8 acceptance (``score``) or compare an H15 re-run (``h15``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scoring = commands.add_parser("score", help="score the stage B outputs")
    scoring.add_argument("--runs-root", type=Path, required=True)
    scoring.add_argument(
        "--criteria", type=Path, required=True, help="resolve_criteria.py --out"
    )
    scoring.add_argument("--referee", type=Path, help="marker_referee.py --out")
    scoring.add_argument(
        "--legacy", type=Path, help="compare_legacy.py untouched --out"
    )
    scoring.add_argument("--draw-spread", type=Path, help="draw_spread.py --out")
    scoring.add_argument(
        "--d4-reference", type=Path, help="stage A3 d4_would_raise_reference.csv"
    )
    scoring.add_argument("--trace", type=Path, action="append", required=True)
    scoring.add_argument(
        "--prep-refs", type=Path, required=True, help="PREP publish dir"
    )
    scoring.add_argument("--run-start", required=True, help="ISO time the run started")
    scoring.add_argument("--e2-cells", type=Path, help="E2 out/cells_insil.csv.gz")
    scoring.add_argument("--h15", type=Path, help="directory of h15_<pair>.json")
    scoring.add_argument("--datasets", default=",".join(PAIRS))
    scoring.add_argument("--segmentations", default="all")
    scoring.add_argument("--date", default=datetime.now(UTC).date().isoformat())
    scoring.add_argument("--out", type=Path, required=True)
    rerun = commands.add_parser("h15", help="compare an H15 re-run")
    rerun.add_argument("--pair", required=True)
    rerun.add_argument("--pipeline-map", type=Path, required=True)
    rerun.add_argument("--pipeline-resolve", type=Path, required=True)
    rerun.add_argument("--rerun-map", type=Path)
    rerun.add_argument("--rerun-resolve", type=Path)
    rerun.add_argument("--seed1-resolve", type=Path)
    rerun.add_argument(
        "--note",
        action="append",
        default=[],
        help="recorded in the JSON (e.g. a re-run task that failed)",
    )
    rerun.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "h15":
        record = h15(args)
        print(
            json.dumps(
                {k: record.get(k) for k in ("pair", "identical")}
                | {"seed1": (record.get("seed1") or {}).get("share_changed")}
            )
        )
        return 0
    summary = score(args)
    print(
        json.dumps(
            {k: summary[k] for k in ("verdict_counts", "n_back_to_user")}, indent=1
        )
    )
    for problem in summary["problems"]:
        logger.warning("problem: %s", problem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
