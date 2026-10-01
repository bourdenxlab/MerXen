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
``draw_spread_would_raise.csv`` (per draw and dataset, every validated
threshold the local rule would raise at H18's depths: D4's list),
``draws/<build hash[:16]>_np<workers>/<tag>.cells.parquet`` (the draw cache,
keyed by the bundle and the worker count, with ``cache.json``; a cache of
other inputs is refused) and ``draw_spread_run.json`` (inputs, code commit,
script and module sha256, the scored-draw check, which draws were computed
or read from the cache). Inputs are read-only; RESOLVE outputs go under
``--out/resolve/<tag>/<pair>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
from merxen.annotation.shadow import SEED_PANEL_FAMILY
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
DEFAULT_WORKERS = 8
COMMIT_ENV = "MERXEN_CODE_COMMIT"
COMMIT_FILE = "COMMIT"
CACHE_RECORD = "cache.json"


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


class ServedTables:
    """What ``draw_tables_installed`` served: the patched and the other loads."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.patched = 0
        self.other: list[str] = []

    def check_served(self, since: int, what: str) -> None:
        """Raise unless the draw's tables were served since ``since`` loads.

        Args:
            since: ``patched`` before the RESOLVE call.
            what: The call, for the message.

        Raises:
            RuntimeError: When RESOLVE never loaded the bundle whose tables
                were replaced (it resolved with another bundle, so its rows
                would silently be the stored seed-0 tables).
        """
        if self.patched > since:
            return
        others = sorted(set(self.other)) or ["no resolvability tables"]
        raise RuntimeError(
            f"{what}: RESOLVE never loaded {self.bundle_dir} (it loaded "
            f"{', '.join(others)}): the draw's tables were not used; pass the "
            "bundle the store lookup picks, or resolve with --bundle"
        )


@contextmanager
def draw_tables_installed(
    bundle_dir: Path, tables: res.ResolvabilityTables
) -> Iterator[ServedTables]:
    """Serve ``tables`` wherever RESOLVE loads ``bundle_dir``'s resolvability.

    RESOLVE reads a bundle's tables with ``resolvability.load_resolvability``
    (imported when it runs), so replacing the module function for the one
    bundle directory gives RESOLVE the draw's tables and every other input
    unchanged, as the M4 and M8-prep counterfactuals did. The yielded
    ``ServedTables`` counts the loads, so a caller can check that RESOLVE
    used this bundle at all (``check_served``).
    """
    original = res.load_resolvability
    target = bundle_dir.resolve()
    served = ServedTables(bundle_dir)

    def load(directory: Path | str) -> res.ResolvabilityTables | None:
        if Path(directory).resolve() == target:
            served.patched += 1
            return tables
        served.other.append(str(directory))
        return original(directory)

    res.load_resolvability = load  # type: ignore[assignment]
    try:
        yield served
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


def self_map_workers(summary: Mapping[str, Any]) -> int | None:
    """The MapMyCells worker count the bundle's self-map mapped with.

    ctm splits the query into chunks of ``min(chunk_size, ceil(n / n_processors))``
    rows and seeds each chunk from the master generator, so the bootstrap
    draws, and hence the calls, depend on the worker count: the draws of the
    grid are the bundle's realisation only when they are mapped with the
    same count (M8 stage A2: 6 workers changed 3.3% of the scored draw's
    calls; 8, the count PREP used, reproduced it exactly).

    Args:
        summary: The bundle's ``resolvability_summary.json``.

    Returns:
        The one ``n_processors`` its mapping runs record, or ``None`` when
        they record none or disagree.
    """
    counts = {
        int(run["n_processors"])
        for run in summary.get("mapping_runs") or []
        if isinstance(run, Mapping) and run.get("n_processors") is not None
    }
    return counts.pop() if len(counts) == 1 else None


def choose_workers(
    requested: int | None, summary: Mapping[str, Any]
) -> tuple[int, int | None]:
    """Return the worker count to map the draws with, and the recorded one.

    The self-map's recorded count (``self_map_workers``) unless one is
    requested (a warning follows in ``main`` when they differ), else 8.

    Args:
        requested: ``--n-processors``.
        summary: The bundle's ``resolvability_summary.json``.

    Returns:
        ``(workers, recorded)``.
    """
    recorded = self_map_workers(summary)
    return int(requested or recorded or DEFAULT_WORKERS), recorded


def require_test_set_revision(
    summary: Mapping[str, Any], *, allow_pre_d1: bool
) -> dict[str, Any]:
    """Return the bundle's test-set exclusion record, or refuse a pre-D1 bundle.

    Args:
        summary: The bundle's ``resolvability_summary.json``.
        allow_pre_d1: Accept a bundle without the M8 D1 revision
            (``--allow-pre-d1``, a diagnostic).

    Returns:
        ``test_set_exclusion`` (empty for an accepted pre-D1 bundle).

    Raises:
        ValueError: If the revision is not ``HO_SELF_MAP_TEST_SET_REVISION``
            and ``allow_pre_d1`` is false.
    """
    exclusion = dict(summary.get("test_set_exclusion") or {})
    if exclusion.get("revision") != ref.HO_SELF_MAP_TEST_SET_REVISION and not (
        allow_pre_d1
    ):
        raise ValueError(
            f"self-map test-set revision {exclusion.get('revision')} is not "
            f"{ref.HO_SELF_MAP_TEST_SET_REVISION} (M8 D1); rebuild it or pass "
            "--allow-pre-d1"
        )
    return exclusion


def bundle_test_cells(
    test: res.HeldOutCells,
    summary: Mapping[str, Any],
    exclusion: Mapping[str, Any],
) -> tuple[res.HeldOutCells, dict[str, Any] | None]:
    """Return the test cells the bundle's self-map simulated.

    With the M8 D1 exclusion recorded, the rule is re-applied
    (``reference.self_map_test_cells``) and must leave out as many cells as
    the bundle records; a pre-D1 bundle (``--allow-pre-d1``) simulated all.

    Args:
        test: ``resolvability.load_test_cells`` of the test-set bundle.
        summary: The bundle's ``resolvability_summary.json``.
        exclusion: ``require_test_set_revision`` output.

    Returns:
        ``(cells, record)``; ``record`` is ``None`` for a pre-D1 bundle.

    Raises:
        ValueError: If the exclusion does not reproduce the bundle's.
    """
    if not exclusion:
        return test, None
    reference_id = str((summary.get("test_set_bundle") or {})["reference_id"])
    kept, recorded = ref.self_map_test_cells(test, reference_id)
    if recorded is None or recorded["n_excluded"] != exclusion.get("n_excluded"):
        raise ValueError(
            "the test-set exclusion does not reproduce the bundle's "
            f"({exclusion.get('n_excluded')} cells left out; got "
            f"{None if recorded is None else recorded['n_excluded']})"
        )
    return kept, recorded


def scored_rows(tables: res.ResolvabilityTables) -> pd.DataFrame:
    """Return the bundle's stored rows of the scored draw.

    The decision recipe's rows at mapping seed 0 (a bundle may store other
    mapping seeds of it for the fine-level seed check).

    Args:
        tables: ``resolvability.load_resolvability`` output.

    Returns:
        The rows, index reset.
    """
    decision = str(tables.summary["decision_recipe"])
    cells = tables.cells
    return cells[
        (cells["recipe"].astype(str) == decision)
        & (cells["seed"].to_numpy() == ds.MAPPING_SEED)
    ].reset_index(drop=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity(script: Path | None = None) -> dict[str, Any]:
    """Return the code identity of this run.

    In order: the ``MERXEN_CODE_COMMIT`` environment variable, the ``COMMIT``
    file at the tree's root (next to ``src/``; an exported tree, as the
    stage A2 runs used), ``git rev-parse HEAD`` of the checkout the script
    sits in (only when the git top level is the script's own tree, never an
    enclosing repository). The sha256 of the script and of the imported
    ``draw_spread`` and ``resolvability`` modules tie the run to its code
    whichever way it ran.

    Args:
        script: This script (default: ``__file__``).

    Returns:
        ``git_commit``, ``git_commit_source``, ``script_sha256`` and
        ``module_sha256`` (module name to ``{path, sha256}``).
    """
    path = Path(script or __file__).resolve()
    root = path.parents[2] if len(path.parents) > 2 else path.parent
    commit: str | None = None
    source: str | None = None
    if os.environ.get(COMMIT_ENV, "").strip():
        commit, source = os.environ[COMMIT_ENV].strip(), f"env {COMMIT_ENV}"
    elif (root / COMMIT_FILE).is_file():
        commit = (root / COMMIT_FILE).read_text(encoding="utf-8").strip() or None
        source = str(root / COMMIT_FILE)
    else:
        try:
            top, head = subprocess.run(
                ["git", "rev-parse", "--show-toplevel", "HEAD"],
                cwd=path.parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, ValueError, subprocess.CalledProcessError):
            top, head = "", ""
        if top and Path(top).resolve() == root:
            commit, source = head, "git"
    modules = {
        module.__name__: {
            "path": str(Path(str(module.__file__)).resolve()),
            "sha256": _sha256(Path(str(module.__file__))),
        }
        for module in (ds, res)
    }
    return {
        "git_commit": commit,
        "git_commit_source": source if commit else None,
        "script_sha256": _sha256(path),
        "module_sha256": modules,
    }


def draw_cache(
    out: Path,
    *,
    build_hash: str,
    workers: int,
    recipe: Mapping[str, Any],
    n_test_cells: int,
    code: Mapping[str, Any],
) -> Path:
    """Return (and create) the draw cache of these inputs.

    ``<out>/draws/<build hash[:16]>_np<workers>/`` with ``cache.json``
    (bundle, worker count, decision recipe, test cells and the module
    sha256): a re-run into the same ``--out`` reuses only draws of the same
    bundle, count and code, and refuses a cache written with other inputs.

    Args:
        out: ``--out``.
        build_hash: The bundle's build hash.
        workers: The MapMyCells worker count.
        recipe: The decision recipe (``to_json``).
        n_test_cells: The test cells simulated.
        code: ``code_identity`` output.

    Returns:
        The cache directory.

    Raises:
        ValueError: If the directory holds a cache of other inputs.
    """
    directory = out / "draws" / f"{str(build_hash)[:16]}_np{int(workers)}"
    record = {
        "bundle_build_hash": str(build_hash),
        "n_processors": int(workers),
        "recipe": dict(recipe),
        "n_test_cells": int(n_test_cells),
        "module_sha256": {
            name: value["sha256"]
            for name, value in (code.get("module_sha256") or {}).items()
        },
    }
    path = directory / CACHE_RECORD
    expected = json.loads(json.dumps(record))
    if path.is_file():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != expected:
            differs = sorted(key for key in expected if found.get(key) != expected[key])
            raise ValueError(
                f"{directory} caches draws of other inputs ({', '.join(differs)} "
                "differ); use a new --out"
            )
    else:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return directory


def load_draw_cells(
    draw: ds.Draw,
    cache: Path,
    *,
    stored: pd.DataFrame,
    simulate: Any,
) -> tuple[pd.DataFrame, str]:
    """Return one draw's cells and where they came from.

    Args:
        draw: The draw.
        cache: ``draw_cache`` output.
        stored: The bundle's scored rows (``scored_rows``).
        simulate: ``draw -> simulate_draw`` rows.

    Returns:
        ``(cells, source)``: ``stored`` for the scored draw, ``cached`` or
        ``computed``.
    """
    if draw.is_scored:
        return stored, "stored"
    path = cache / f"{draw.tag}.cells.parquet"
    if path.is_file():
        return res.restore_labels(pd.read_parquet(path)), "cached"
    cells = simulate(draw)
    cells.to_parquet(path, index=False)
    return res.restore_labels(pd.read_parquet(path)), "computed"


def panel_family_of(summary_path: Path, sample_id: str) -> str | None:
    """The panel family a dataset's RESOLVE run trusted (``trust.family_id``)."""
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trust = ((summary.get("samples") or {}).get(sample_id) or {}).get("trust") or {}
    family = trust.get("family_id")
    return None if family is None else str(family)


def prep_panel_family(
    requested: str | None, families: Mapping[str, str | None]
) -> tuple[str | None, str]:
    """Return the panel family of PREP's H18 rows and how it was chosen.

    Args:
        requested: ``--panel-family``.
        families: Sample id to its RESOLVE trust family.

    Returns:
        ``(family, basis)``.

    Raises:
        ValueError: If the datasets trust more than one family.
    """
    if requested:
        return requested, "--panel-family"
    found = {family for family in families.values() if family is not None}
    if len(found) > 1:
        raise ValueError(
            f"the datasets trust several panel families {sorted(found)}: pass "
            "--panel-family"
        )
    if found:
        return found.pop(), "the datasets' RESOLVE trust family"
    return SEED_PANEL_FAMILY, "seed family (no dataset trust record)"


def dataset_compositions(
    args: argparse.Namespace,
    depths: Sequence[int],
    min_bin_mass: float,
) -> dict[str, tuple[str, res.DatasetComposition, str | None]]:
    """Return sample id -> (platform, composition, panel family) of seed 0."""
    out: dict[str, tuple[str, res.DatasetComposition, str | None]] = {}
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
            family = panel_family_of(
                args.resolve_root
                / pair
                / SEGMENTATION
                / f"{pair}_resolve_summary.json",
                sample_id,
            )
            out[sample_id] = (
                platform,
                ds.dataset_composition(leaf, labels, depths, min_bin_mass=min_bin_mass),
                family,
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
    parser.add_argument(
        "--n-processors",
        type=int,
        default=None,
        help="MapMyCells workers (default: the count the self-map recorded, else 8)",
    )
    parser.add_argument("--max-gb", type=float, default=60.0)
    parser.add_argument("--no-check-scored", action="store_true")
    parser.add_argument(
        "--panel-family",
        help=(
            "floor-table family of PREP's H18 rows (default: the datasets' "
            "RESOLVE trust family, else human_set_a)"
        ),
    )
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
    test_ref = summary["test_set_bundle"]
    ho_dir = Path(test_ref["path"])
    engine_record = summary.get("engine") or {}
    engine = MmcBundle.from_dir(bundle_dir if engine_record.get("self") else ho_dir)
    try:
        exclusion = require_test_set_revision(summary, allow_pre_d1=args.allow_pre_d1)
        # The bundle's own test cells: the M8 D1 exclusion when the bundle
        # recorded it (a pre-D1 bundle, --allow-pre-d1, simulated all of them).
        test, recorded = bundle_test_cells(
            res.load_test_cells(ho_dir), summary, exclusion
        )
    except ValueError as error:
        parser.error(f"{bundle_dir}: {error}")
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
    workers, recorded_workers = choose_workers(args.n_processors, summary)
    if recorded_workers is not None and workers != recorded_workers:
        logger.warning(
            "mapping the draws with %d workers, but the self-map mapped with %d: "
            "ctm's chunk seeding makes these draws another realisation, and the "
            "scored-draw control will not reproduce the stored rows",
            workers,
            recorded_workers,
        )
    ref.set_prep_resources(n_processors=workers, max_gb=args.max_gb)
    specs = ref.level_specs_for(reference_id, engine, config)
    rules = ref.cells_rules_for(reference_id, config)
    depths = tables.depth_grid
    stored = scored_rows(tables)
    code = code_identity()
    try:
        cache = draw_cache(
            args.out,
            build_hash=str(manifest.get("build_hash")),
            workers=workers,
            recipe=recipe.to_json(),
            n_test_cells=len(test.obs),
            code=code,
        )
    except ValueError as error:
        parser.error(str(error))
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
    try:
        prep_family, family_basis = prep_panel_family(
            args.panel_family,
            {sample: family for sample, (_, _, family) in compositions.items()},
        )
    except ValueError as error:
        parser.error(str(error))
    segmented = n_segmented_table(args.qc_summary) if args.qc_summary else {}
    rows: list[dict[str, Any]] = []
    h18_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    raise_frames: list[pd.DataFrame] = []
    sources: dict[str, str] = {}
    for draw in draws:
        cells, sources[draw.tag] = load_draw_cells(
            draw, cache, stored=stored, simulate=simulate
        )
        drawn = ds.tables_with_draw(tables, cells)
        decision_sets = [
            (f"PREP[{platform} floors]", platform, prep_family, drawn.decisions())
            for platform in PLATFORMS
        ] + [
            (sample_id, platform, family, drawn.decisions(composition=composition))
            for sample_id, (platform, composition, family) in sorted(
                compositions.items()
            )
        ]
        for dataset, platform, family, decisions in decision_sets:
            metric_rows, h18, bins = ds.decision_metric_rows(
                decisions,
                draw=draw,
                dataset=dataset,
                platform=platform,
                depths=depths,
                floors=floors,
                settings=drawn.settings,
                panel_family=family,
            )
            rows += metric_rows
            h18_frames.append(h18)
            bin_frames.append(bins)
            raise_frames.append(
                ds.would_raise_h18_bins(
                    decisions, drawn.settings, draw=draw, dataset=dataset
                )
            )
        if args.resolve:
            with draw_tables_installed(bundle_dir, drawn) as served:
                for pair in args.pairs.split(","):
                    n_segmented = {
                        f"{pair}_{platform}": segmented[(pair, SEGMENTATION, platform)]
                        for platform in PLATFORMS
                        if (pair, SEGMENTATION, platform) in segmented
                    }
                    before = served.patched
                    result = annotate_resolve(
                        args.runs_root / pair / SEGMENTATION,
                        config,
                        output_dir=args.out / "resolve" / draw.tag / pair,
                        bundle_finder=current_store_bundles(ReferenceStore(args.store)),
                        n_segmented=n_segmented,
                        n_bootstrap=1,
                    )
                    served.check_served(before, f"draw {draw.tag}, pair {pair}")
                    rows += resolve_rows(draw, result, pair, config)
        logger.info("draw %s decided (%s)", draw.tag, sources[draw.tag])
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
    pd.concat(raise_frames, ignore_index=True).to_csv(
        out / "draw_spread_would_raise.csv", index=False
    )
    record: Mapping[str, Any] = {
        "bundle": str(bundle_dir),
        "bundle_build_hash": manifest.get("build_hash"),
        "test_set_bundle": test_ref,
        "test_set_exclusion": recorded,
        "resolvability_version": summary.get("resolvability_version"),
        "draws": [draw.tag for draw in draws],
        "draw_sources": sources,
        "draw_cache": str(cache),
        "scored_draw": draws[0].tag,
        "scored_draw_check": check,
        "mapping_runs": runs,
        "n_processors": workers,
        "self_map_n_processors": recorded_workers,
        "datasets_reweighted": sorted(compositions),
        "panel_family": {
            "PREP": prep_family,
            "PREP_basis": family_basis,
            **{sample: family for sample, (_, _, family) in compositions.items()},
        },
        "resolve": bool(args.resolve),
        **code,
    }
    (out / "draw_spread_run.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0 if check.get("skipped") or check.get("reproduces") else 1


if __name__ == "__main__":
    sys.exit(main())
