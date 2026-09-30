#!/usr/bin/env python
"""The M8 draw-spread table: 9 simulation draws of a human self-map (D2).

User decision 2026-09-30 (M8 D2, pre-registration §18): the gate scores the
pre-registered seed-0 draw of the resolvability self-map, and the spread over
the 3 x 3 grid of per-cell seeds x efficiency seeds (0-2 each) is reported
beside it. It never changes a verdict. For one self-map bundle (the rebuilt
set a WHB bundle, M8 D1) this script:

1. simulates and maps every draw of the decision recipe on the bundle's own
   test cells (``reference.self_map_test_cells``) with the production
   simulation, mapper (onto the held-out bundle, bootstrap settings and
   seed 0 as recorded) and cells rules (``draw_spread.simulate_draw``); the
   scored draw is the bundle's stored rows, and its re-simulation is checked
   against them (``--no-check-scored`` skips that control);
2. decides each draw alone with the unchanged rules: PREP unweighted (H18
   with each platform's floors) and, with ``--resolve-root`` and
   ``--runs-root``, reweighted to every dataset's soft composition as RESOLVE
   builds it (``draw_spread.dataset_composition``: the seed-0 RESOLVE label
   tables and the MAP WHB calls);
3. with ``--resolve`` (and ``--store``, ``--qc-summary``), runs RESOLVE per
   draw and pair on the MAP outputs with the draw's tables in place of the
   bundle's, for H7, the H8 warning value and the gate level.

Writes to ``--out``: ``draw_spread_rows.csv`` (long: draw x dataset x
metric), ``draw_spread.csv`` (per dataset and metric: the scored value, the
min-max over the draws and the draws failing), ``draw_spread_h18.csv``,
``draw_spread_bins.csv``, ``draw_spread_bin_spread.csv``,
``draws/<tag>.cells.parquet`` (cache) and ``draw_spread_run.json`` (inputs,
code commit, script sha256, the scored-draw check). Inputs are read-only;
RESOLVE outputs go under ``--out/resolve/<tag>/<pair>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from merxen.annotation import draw_spread as ds
from merxen.annotation import reference as ref
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.mapmycells_engine import (
    MmcBundle,
    level_frame,
    read_tidy_parquet,
)
from merxen.annotation.pipeline import (
    annotate_resolve,
    current_store_bundles,
    read_label_table,
)
from merxen.annotation.store import ReferenceStore
from merxen.annotation.vocab import load_floor_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_criteria import (  # noqa: E402
    H7_THRESHOLDS,
    H8_BROAD_ONLY,
    H8_WARNING,
    n_segmented_table,
)
from shadow_baselines import PAIRS, PLATFORMS, add_fallback_arguments  # noqa: E402

logger = logging.getLogger("draw_spread")

WHB_RUN = "whb_frontal_supc_clus"
SEGMENTATION = "proseg_hybrid"
DECISION_KEYS = ["level", "sim_id"]


def human_config(fallback_csv: Path | None) -> AnnotationConfig:
    """The human annotation config RESOLVE ran with (as ``resolve_criteria``)."""
    config = AnnotationConfig(species="human")
    return config.model_copy(
        update={
            "panel": config.panel.model_copy(
                update={"gene_id_fallback_csv": fallback_csv}
            )
        }
    ).coupled_to_clustering(10)


def compare_cells(stored: pd.DataFrame, redrawn: pd.DataFrame) -> dict[str, Any]:
    """Compare a re-simulated draw with the bundle's stored rows (the control).

    Args:
        stored: The bundle's decision-recipe rows.
        redrawn: ``simulate_draw`` rows of the same draw.

    Returns:
        Row counts and the shares of calls, bp values and correctness that
        differ (all 0 when the draw path reproduces the bundle).
    """
    left = stored.sort_values(DECISION_KEYS).reset_index(drop=True)
    right = redrawn.sort_values(DECISION_KEYS).reset_index(drop=True)
    same_rows = len(left) == len(right) and left["sim_id"].astype(str).equals(
        right["sim_id"].astype(str)
    )
    record: dict[str, Any] = {
        "n_stored": len(left),
        "n_redrawn": len(right),
        "same_rows": bool(same_rows),
    }
    if not same_rows:
        return {**record, "reproduces": False}
    call_a = left["call"].astype(object).fillna("<none>").astype(str).to_numpy()
    call_b = right["call"].astype(object).fillna("<none>").astype(str).to_numpy()
    bp_a = np.nan_to_num(left["bp"].to_numpy(np.float64), nan=-1.0)
    bp_b = np.nan_to_num(right["bp"].to_numpy(np.float64), nan=-1.0)
    correct = left["correct"].to_numpy(bool) != right["correct"].to_numpy(bool)
    record.update(
        {
            "calls_changed": float(np.mean(call_a != call_b)),
            "bp_changed": float(np.mean(np.abs(bp_a - bp_b) > 1e-6)),
            "correct_changed": float(np.mean(correct)),
        }
    )
    record["reproduces"] = (
        record["calls_changed"] == 0.0
        and record["bp_changed"] == 0.0
        and record["correct_changed"] == 0.0
    )
    return record


@contextmanager
def draw_tables_installed(
    bundle_dir: Path, tables: res.ResolvabilityTables
) -> Iterator[None]:
    """Serve ``tables`` wherever RESOLVE loads ``bundle_dir``'s resolvability.

    RESOLVE reads a bundle's tables with ``resolvability.load_resolvability``
    (imported when it runs), so replacing the module function for the one
    bundle directory gives RESOLVE the draw's tables and every other input
    unchanged, as the M4 and M8-prep counterfactuals did.
    """
    original = res.load_resolvability
    target = bundle_dir.resolve()

    def load(directory: Path | str) -> res.ResolvabilityTables | None:
        if Path(directory).resolve() == target:
            return tables
        return original(directory)

    res.load_resolvability = load  # type: ignore[assignment]
    try:
        yield
    finally:
        res.load_resolvability = original  # type: ignore[assignment]


def resolve_rows(
    draw: ds.Draw, result: Any, pair: str, config: AnnotationConfig
) -> list[dict[str, Any]]:
    """Return one draw's RESOLVE rows of a pair: H7, the H8 warning, the level.

    Args:
        draw: The draw.
        result: ``annotate_resolve`` output.
        pair: The pair id.
        config: The config (the gate's warning threshold).

    Returns:
        ``draw_spread.metric_row`` rows.
    """
    warn_below = float(config.gate.warn_segmented_broad_coverage)
    rows: list[dict[str, Any]] = []
    for sample_id, resolution in sorted(result.samples.items()):
        platform = str(resolution.platform).upper()
        labels = resolution.labels
        table = labels[labels["in_table"].to_numpy(bool)]
        confident = table["ct_broad_status"].astype(str).to_numpy() == "confident"
        gate = (resolution.summary.get("resolution") or {}).get("gate") or {}
        expected_level = "broad_only" if sample_id in H8_BROAD_ONLY else "full"
        warning_expected = sample_id in H8_WARNING[SEGMENTATION]
        rows += [
            ds.metric_row(
                draw,
                sample_id,
                "H7",
                float(confident.mean()) if len(table) else None,
                threshold=H7_THRESHOLDS.get((pair, platform)),
                comparator=">=",
            ),
            ds.metric_row(
                draw,
                sample_id,
                "H8/warning_value",
                gate.get("broad_coverage_segmented"),
                threshold=warn_below,
                comparator=">=",
                passes=bool(gate.get("warning")) == warning_expected,
                note=(
                    f"warning {bool(gate.get('warning'))} (expected {warning_expected})"
                ),
            ),
            ds.metric_row(
                draw,
                sample_id,
                "H8/level",
                None,
                passes=gate.get("level") == expected_level,
                note=f"level {gate.get('level')} (expected {expected_level})",
            ),
        ]
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_draw_cells(
    draw: ds.Draw,
    cache: Path,
    *,
    stored: pd.DataFrame,
    simulate: Any,
) -> pd.DataFrame:
    if draw.is_scored:
        return stored
    path = cache / f"{draw.tag}.cells.parquet"
    if path.is_file():
        return res.restore_labels(pd.read_parquet(path))
    cells = simulate(draw)
    cells.to_parquet(path, index=False)
    return res.restore_labels(pd.read_parquet(path))


def dataset_compositions(
    args: argparse.Namespace,
    depths: Sequence[int],
    min_bin_mass: float,
) -> dict[str, tuple[str, res.DatasetComposition]]:
    """Return sample id -> (platform, composition) of the seed-0 RESOLVE runs."""
    out: dict[str, tuple[str, res.DatasetComposition]] = {}
    if args.resolve_root is None or args.runs_root is None:
        return out
    for pair in args.pairs.split(","):
        for platform in PLATFORMS:
            sample_id = f"{pair}_{platform}"
            labels_path = (
                args.resolve_root
                / pair
                / SEGMENTATION
                / platform.lower()
                / f"{sample_id}_celltype_labels.parquet"
            )
            tidy_path = (
                args.runs_root
                / pair
                / SEGMENTATION
                / platform.lower()
                / f"{sample_id}_mmc_{WHB_RUN}.parquet"
            )
            if not (labels_path.is_file() and tidy_path.is_file()):
                logger.warning("%s: no RESOLVE labels or MAP calls; skipped", sample_id)
                continue
            labels, _ = read_label_table(labels_path)
            tidy, _ = read_tidy_parquet(tidy_path)
            leaf = level_frame(tidy, ref.WHB_SUPC)
            out[sample_id] = (
                platform,
                ds.dataset_composition(leaf, labels, depths, min_bin_mass=min_bin_mass),
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """Build the draw-spread tables of one human self-map bundle."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, ds.GRID_SEEDS)))
    parser.add_argument("--resolve-root", type=Path, help="seed-0 RESOLVE outputs")
    parser.add_argument("--runs-root", type=Path, help="MAP outputs")
    parser.add_argument("--pairs", default=",".join(PAIRS))
    parser.add_argument("--resolve", action="store_true", help="RESOLVE per draw")
    parser.add_argument("--store", type=Path, help="reference store (RESOLVE)")
    parser.add_argument("--qc-summary", type=Path, help="segmented objects (RESOLVE)")
    parser.add_argument("--n-processors", type=int, default=8)
    parser.add_argument("--max-gb", type=float, default=60.0)
    parser.add_argument("--no-check-scored", action="store_true")
    parser.add_argument(
        "--allow-pre-d1",
        action="store_true",
        help="accept a bundle without the M8 D1 test-set revision (diagnostic)",
    )
    add_fallback_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.resolve and (
        args.store is None or args.qc_summary is None or args.runs_root is None
    ):
        parser.error("--resolve needs --store, --qc-summary and --runs-root")
    draws = ds.draw_grid([int(seed) for seed in args.seeds.split(",") if seed])
    bundle_dir = args.bundle
    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    tables = res.load_resolvability(bundle_dir)
    if tables is None:
        parser.error(f"{bundle_dir} has no resolvability tables")
    summary = tables.summary
    exclusion = summary.get("test_set_exclusion") or {}
    if (
        exclusion.get("revision") != ref.HO_SELF_MAP_TEST_SET_REVISION
        and not args.allow_pre_d1
    ):
        parser.error(
            f"{bundle_dir}: self-map test-set revision {exclusion.get('revision')} "
            f"is not {ref.HO_SELF_MAP_TEST_SET_REVISION} (M8 D1); rebuild it or "
            "pass --allow-pre-d1"
        )
    test_ref = summary["test_set_bundle"]
    ho_dir = Path(test_ref["path"])
    engine_record = summary.get("engine") or {}
    engine = MmcBundle.from_dir(bundle_dir if engine_record.get("self") else ho_dir)
    # The bundle's own test cells: the M8 D1 exclusion when the bundle
    # recorded it (a pre-D1 bundle, --allow-pre-d1, simulated all of them).
    test = res.load_test_cells(ho_dir)
    recorded: dict[str, Any] | None = None
    if exclusion:
        test, recorded = ref.self_map_test_cells(test, str(test_ref["reference_id"]))
        if recorded is None or recorded["n_excluded"] != exclusion.get("n_excluded"):
            parser.error(
                f"{bundle_dir}: the test-set exclusion does not reproduce the "
                f"bundle's ({exclusion.get('n_excluded')} cells left out)"
            )
    config = human_config(args.gene_id_fallback_csv)
    reference_id = str(manifest.get("reference_id") or WHB_RUN)
    mapping = (manifest.get("recorded_settings") or {}).get("mapping")
    if not mapping:
        parser.error(f"{bundle_dir}: bundle.json records no mapping settings")
    spec = AnnotationReferenceSpec(
        reference_id=reference_id, species="human", role="primary", **mapping
    )
    recipe = res.simulation_recipes(config.resolvability, seed=ref.TEST_SET_SEED)[0]
    if recipe.to_json() != summary["recipes"][0]:
        parser.error("the decision recipe differs from the bundle's recorded one")
    ref.set_prep_resources(n_processors=args.n_processors, max_gb=args.max_gb)
    specs = ref.level_specs_for(reference_id, engine, config)
    rules = ref.cells_rules_for(reference_id, config)
    depths = tables.depth_grid
    decision = str(summary["decision_recipe"])
    stored = tables.cells[
        (tables.cells["recipe"].astype(str) == decision)
        & (tables.cells["seed"].to_numpy() == ds.MAPPING_SEED)
    ].reset_index(drop=True)
    cache = args.out / "draws"
    cache.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    map_fn = ref.mmc_map_function(
        engine,
        spec=spec,
        config=config,
        scratch_dir=args.out / "scratch",
        log_dir=args.out / "logs",
        runs=runs,
    )

    def simulate(draw: ds.Draw) -> pd.DataFrame:
        logger.info("simulating and mapping draw %s", draw.tag)
        return ds.simulate_draw(
            test, depths, recipe, draw, specs=specs, map_fn=map_fn, cells_rules=rules
        )

    check: dict[str, Any] = {"skipped": True}
    if not args.no_check_scored:
        check = compare_cells(stored, simulate(draws[0]))
        logger.info("scored-draw control: %s", check)
    floors = load_floor_table("human")
    compositions = dataset_compositions(
        args, depths, float(config.resolvability.composition_min_bin_cells)
    )
    segmented = n_segmented_table(args.qc_summary) if args.qc_summary else {}
    rows: list[dict[str, Any]] = []
    h18_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    for draw in draws:
        cells = _load_draw_cells(draw, cache, stored=stored, simulate=simulate)
        drawn = ds.tables_with_draw(tables, cells)
        prep = drawn.decisions()
        for platform in PLATFORMS:
            metric_rows, h18, bins = ds.decision_metric_rows(
                prep,
                draw=draw,
                dataset=f"PREP[{platform} floors]",
                platform=platform,
                depths=depths,
                floors=floors,
                settings=drawn.settings,
            )
            rows += metric_rows
            h18_frames.append(h18)
            bin_frames.append(bins)
        for sample_id, (platform, composition) in sorted(compositions.items()):
            weighted = drawn.decisions(composition=composition)
            metric_rows, h18, bins = ds.decision_metric_rows(
                weighted,
                draw=draw,
                dataset=sample_id,
                platform=platform,
                depths=depths,
                floors=floors,
                settings=drawn.settings,
            )
            rows += metric_rows
            h18_frames.append(h18)
            bin_frames.append(bins)
        if args.resolve:
            with draw_tables_installed(bundle_dir, drawn):
                for pair in args.pairs.split(","):
                    n_segmented = {
                        f"{pair}_{platform}": segmented[(pair, SEGMENTATION, platform)]
                        for platform in PLATFORMS
                        if (pair, SEGMENTATION, platform) in segmented
                    }
                    result = annotate_resolve(
                        args.runs_root / pair / SEGMENTATION,
                        config,
                        output_dir=args.out / "resolve" / draw.tag / pair,
                        bundle_finder=current_store_bundles(ReferenceStore(args.store)),
                        n_segmented=n_segmented,
                        n_bootstrap=1,
                    )
                    rows += resolve_rows(draw, result, pair, config)
        logger.info("draw %s decided", draw.tag)
    out = args.out
    frame = pd.DataFrame(rows, columns=list(ds.ROW_COLUMNS))
    frame.to_csv(out / "draw_spread_rows.csv", index=False)
    ds.spread_table(frame).to_csv(out / "draw_spread.csv", index=False)
    pd.concat(h18_frames, ignore_index=True).to_csv(
        out / "draw_spread_h18.csv", index=False
    )
    bins_frame = pd.concat(bin_frames, ignore_index=True)
    bins_frame.to_csv(out / "draw_spread_bins.csv", index=False)
    ds.bin_spread(bins_frame).to_csv(out / "draw_spread_bin_spread.csv", index=False)
    record: Mapping[str, Any] = {
        "bundle": str(bundle_dir),
        "bundle_build_hash": manifest.get("build_hash"),
        "test_set_bundle": test_ref,
        "test_set_exclusion": recorded,
        "resolvability_version": summary.get("resolvability_version"),
        "draws": [draw.tag for draw in draws],
        "scored_draw": draws[0].tag,
        "scored_draw_check": check,
        "mapping_runs": runs,
        "datasets_reweighted": sorted(compositions),
        "resolve": bool(args.resolve),
        "git_commit": _git_commit(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "draw_spread_run.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0 if check.get("skipped") or check.get("reproduces") else 1


if __name__ == "__main__":
    sys.exit(main())
