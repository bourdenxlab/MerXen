#!/usr/bin/env python
"""M13 C17: the real-data QC regression on the M8 human pairs (set a).

Plan §12 M13 exit ("real-data QC is run as a regression on P7513, P1212,
P7113, P5011"; the ag7 and VZG2 rows wait for M6b, chunk C22). The M8 RESOLVE
outputs (``$A/m8/resolve/<pair>/{proseg_hybrid,reseg}``, read only) name the
MAP run, the bundles (their ``resolved_build_hash``), the gate denominators
and the pair's alignment; this script re-runs human RESOLVE on that MAP
output with the same inputs and the real-data QC as wired in M13 (C12, C13,
C15), and writes the outcome table under ``$A/m13/regression/``:

1. ``registration``: the M0a registration check (human G1, §7.6 / §8.8,
   NR9) of every section x segmentation. The M8 pairs' QC stage outputs
   predate M0a, so the check is computed here as the QC stage computes it
   (``merxen.qc.registration.compute_segmentation_registration_qc`` on the
   published SpatialData store, read only: the segmentation's shapes, the
   platform's own cells as the offset reference, the store's primary
   points), into ``registration/<seg>/<sample>_registration_qc.json``.
2. ``resolve``: ``merxen annotate-resolve`` per pair x segmentation and
   variant, into ``resolve/<variant>/<pair>/<seg>/``:

   * ``qc``: the registered configuration (species defaults: the QC on,
     the ``node`` referee comparator, G1 warn-only by D23 (b), the seeded
     families warn-only by D20 (b));
   * ``qc_free``: ``real_qc.enabled`` false, the QC-free re-run NR1
     compares with;
   * ``class_comparator``: the referee's ``class`` comparator, reported
     beside the registered statistic (D18; never decides).

3. ``table``: the outcome table and the readings the M13 decisions ask of
   C17, with a report:

   * ``outcomes.csv``: every outcome of the ``qc`` variant, with the effect
     each would have once the human gate merges (D20 (b): the seeded set a
     family only warns until then);
   * ``marker_consistency.csv`` (D18): the derived marker statistic
     re-measured on set a; any value below the warning threshold (0.75)
     sends the thresholds back to the user before any number of the
     new-panel human MERSCOPE family;
   * ``paired_concordance.csv`` (D21): the soft broad JSD on the shared
     mask (scored, point estimate) beside the whole section (reported);
     P5011 proseg_hybrid is expected to warn (.235 / .227), the other pairs
     to pass (<= .154);
   * ``flag_rates.csv`` (D22): the literal reading of §8.8 (CHECK K6, H16's
     15% marking on class x platform strata) beside option (a) (each flag's
     own null switch over flag x class x platform), which needs separate
     written approval;
   * ``registration_g1.csv`` (D23): whether G1's fail rule fires on any set
     a section (a false failure); without one, the move to (a) (``failed``
     + ``exclude_hard``, a tightening) can be decided and recorded before
     any QC output of the new family is read;
   * ``nr1.csv``: NR1's downgrade-only comparison of ``qc`` with
     ``qc_free``, and the identity of both with the M8 label tables (the
     version-7 column ``flag_nonneuronal_high_depth`` that M8 predates must
     be null on these version-6 bundles);
   * ``regression_summary.json`` and ``REPORT.txt``.

Reads (never writes) the M8 RESOLVE tree, the MAP runs, the published
results and the reference store; ``--out`` must lie outside all of them.
CPU only (RESOLVE runs one at a time with single-threaded BLAS, as M8 did).
No MapMyCells, no PREP and no Nextflow.

Usage::

    PYTHONPATH=src python scripts/acceptance/m13_real_qc_regression.py all
    PYTHONPATH=src python scripts/acceptance/m13_real_qc_regression.py table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("m13_real_qc_regression")

A = Path("/srv/storage/MerXen/annotation_dev/evidence_20260926")
M8_RESOLVE = A / "m8" / "resolve"
OUT = A / "m13" / "regression"
RESULTS = Path("/srv/storage/MerXen/results")
STORE = Path("/media/mathieubo/SSD1/MerXen/annotation_references")
FALLBACK = Path(
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/"
    "WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
)
REPO = Path(__file__).resolve().parents[2]

PAIRS: tuple[str, ...] = ("P7513", "P1212", "P7113", "P5011")
SEGMENTATIONS: tuple[str, ...] = ("proseg_hybrid", "reseg")
# The scored segmentation of gate H (pre-registration §9); reseg is reported.
SCORED_SEGMENTATION = "proseg_hybrid"
VARIANT_QC = "qc"
VARIANT_QC_FREE = "qc_free"
VARIANT_CLASS = "class_comparator"
VARIANTS: tuple[str, ...] = (VARIANT_QC, VARIANT_QC_FREE, VARIANT_CLASS)

# The layer keys of ``workflows/main.nf`` ``analysisLayerKeys`` and the
# registration reference of ``workflows/modules/qc.nf`` (the platform's own
# cells).
SHAPE_KEYS: dict[str, str] = {
    "proseg_hybrid": "MOSAIK_proseg_hybrid",
    "reseg": "MOSAIK_proseg",
    "proseg": "MOSAIK_proseg",
    "proseg_mask": "MOSAIK_cellpose",
    "cellpose": "MOSAIK_cellpose",
}
REFERENCE_SHAPE_KEYS: dict[str, str] = {
    "MERSCOPE": "merscope_cell_boundaries",
    "XENIUM": "xenium_cell_boundaries",
}

# D21 (plan §4): on the scored segmentation P5011 warns (shared mask .235,
# whole section .227) and the other pairs pass (<= .154).
PAIRED_EXPECTED_WARN: dict[str, tuple[float, float]] = {"P5011": (0.235, 0.227)}
PAIRED_EXPECTED_PASS_MAX = 0.154
# The new column of M13 C14 that M8's label tables predate: null on the
# version-6 bundles of set a.
NEW_NULL_COLUMNS: frozenset[str] = frozenset({"flag_nonneuronal_high_depth"})

# Run the merxen CLI from the interpreter (and PYTHONPATH) running this script.
CLI_SNIPPET = "import sys; from merxen.cli import main; sys.exit(main())"
SINGLE_THREAD_ENV: dict[str, str] = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Outcome detail that is the check's headline number.
VALUE_KEYS: dict[str, str] = {
    "marker_consistency": "consistency",
    "registration_g1": "density_ratio",
    "paired_concordance": "jsd",
}


class RegressionError(RuntimeError):
    """An input of the regression is missing or inconsistent."""


# --------------------------------------------------------------------------
# Inputs


@dataclass(frozen=True)
class M8Run:
    """What one M8 RESOLVE run (pair x segmentation) resolved with.

    Attributes:
        pair: The pair id.
        segmentation: The segmentation.
        resolve_dir: The M8 RESOLVE output directory.
        map_dir: The MAP output it resolved.
        samples: Sample id to platform (``MERSCOPE`` / ``XENIUM``).
        n_segmented: Sample id to the gate warning's denominator.
        bundles: MAP run id to ``(reference_id, resolved_build_hash)``.
        alignment_dir: The pair's ``align_out`` (shared tissue mask).
        summary_sha256: The M8 summary's sha256 (recorded by its run record).
    """

    pair: str
    segmentation: str
    resolve_dir: Path
    map_dir: Path
    samples: dict[str, str]
    n_segmented: dict[str, int]
    bundles: dict[str, tuple[str, str]]
    alignment_dir: Path | None = None
    summary_sha256: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object file's content.

    Args:
        path: The file.

    Returns:
        The object.

    Raises:
        RegressionError: If the file is missing or holds no object.
    """
    if not path.is_file():
        raise RegressionError(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegressionError(f"{path} does not hold a JSON object")
    return payload


def read_m8_run(root: Path, pair: str, segmentation: str) -> M8Run:
    """Read what an M8 RESOLVE run resolved with (its run record and summary).

    Args:
        root: The M8 RESOLVE root (``<root>/<pair>/<seg>/``).
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        The run.

    Raises:
        RegressionError: If a file is missing, the summary is of another
            pair, segmentation or species, or its samples resolved a run
            with different bundles.
    """
    directory = root / pair / segmentation
    record = read_json(directory / f"{pair}_resolve_run.json")
    summary = read_json(directory / f"{pair}_resolve_summary.json")
    if summary.get("pair_id") != pair or summary.get("segmentation") != segmentation:
        raise RegressionError(
            f"{directory}: the summary is pair {summary.get('pair_id')} / "
            f"{summary.get('segmentation')}, not {pair} / {segmentation}"
        )
    if summary.get("species") != "human":
        raise RegressionError(f"{directory}: not a human RESOLVE run")
    samples: dict[str, str] = {}
    n_segmented: dict[str, int] = {}
    bundles: dict[str, tuple[str, str]] = {}
    for sample_id, item in sorted((summary.get("samples") or {}).items()):
        samples[sample_id] = str(item["platform"]).upper()
        if item.get("n_segmented") is not None:
            n_segmented[sample_id] = int(item["n_segmented"])
        for run_id, bundle in sorted((item.get("bundles") or {}).items()):
            value = (str(bundle["reference_id"]), str(bundle["resolved_build_hash"]))
            if bundles.setdefault(run_id, value) != value:
                raise RegressionError(
                    f"{directory}: run {run_id} resolved with {bundles[run_id]} "
                    f"and {value} in different samples"
                )
    if not samples:
        raise RegressionError(f"{directory}: the summary records no sample")
    alignment = (summary.get("pair") or {}).get("alignment_dir")
    return M8Run(
        pair=pair,
        segmentation=segmentation,
        resolve_dir=directory,
        map_dir=Path(str(record["map_manifest"])).parent,
        samples=samples,
        n_segmented=n_segmented,
        bundles=bundles,
        alignment_dir=None if alignment is None else Path(str(alignment)),
        summary_sha256=record.get("summary_sha256"),
    )


def check_out_dir(out_dir: Path, protected: Iterable[Path]) -> None:
    """Refuse an output directory inside (or around) a read-only input tree.

    Args:
        out_dir: The output directory.
        protected: Trees the regression reads and never writes.

    Raises:
        RegressionError: If ``out_dir`` lies inside a protected tree or holds
            one.
    """
    target = out_dir.resolve()
    for root in protected:
        tree = Path(root).resolve()
        if target == tree or tree in target.parents or target in tree.parents:
            raise RegressionError(
                f"--out {out_dir} overlaps the read-only input tree {root}"
            )


def layer_keys(platform: str, segmentation: str) -> tuple[str, str]:
    """Return a layer's shape key and its registration reference shape key.

    Args:
        platform: ``MERSCOPE`` or ``XENIUM``.
        segmentation: The segmentation (``main.nf`` ``analysisLayerKeys``).

    Returns:
        ``(shape_key, reference_shape_key)``.

    Raises:
        RegressionError: For an unknown platform or segmentation.
    """
    key = platform.upper()
    if key not in REFERENCE_SHAPE_KEYS:
        raise RegressionError(f"unknown platform {platform!r}")
    if segmentation not in SHAPE_KEYS:
        raise RegressionError(f"no shape key for segmentation {segmentation!r}")
    return SHAPE_KEYS[segmentation], REFERENCE_SHAPE_KEYS[key]


def latest_zarr(results: Path, pair: str, platform: str) -> Path:
    """Return a section's published SpatialData store (the QC stage's input)."""
    return results / pair / platform.lower() / "latest" / "latest_spatialdata.zarr"


def registration_path(out_dir: Path, segmentation: str, sample_id: str) -> Path:
    """Return where a section's registration check goes.

    The name is the QC stage's (``<sample id lower-case>_registration_qc.json``)
    so ``mouse_gate.find_registration_check`` finds it too.
    """
    return (
        out_dir
        / "registration"
        / segmentation
        / f"{sample_id.lower()}_registration_qc.json"
    )


def variant_dir(out_dir: Path, variant: str, pair: str, segmentation: str) -> Path:
    """Return a RESOLVE re-run's output directory."""
    return out_dir / "resolve" / variant / pair / segmentation


def variant_config(variant: str) -> dict[str, Any] | None:
    """Return the annotation config of a variant (``None``: species defaults).

    Args:
        variant: ``qc``, ``qc_free`` or ``class_comparator``.

    Returns:
        The config record (``AnnotationConfig`` JSON), or ``None`` for the
        registered configuration, which M8 also ran with (no config file).

    Raises:
        RegressionError: For an unknown variant.
    """
    if variant == VARIANT_QC:
        return None
    if variant == VARIANT_QC_FREE:
        return {"species": "human", "real_qc": {"enabled": False}}
    if variant == VARIANT_CLASS:
        return {"species": "human", "real_qc": {"marker_referee_comparator": "class"}}
    raise RegressionError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def resolve_command(
    run: M8Run,
    *,
    out_dir: Path,
    store: Path,
    registration: Mapping[str, Path],
    config_path: Path | None,
    protected_roots: Sequence[Path],
    fallback: Path | None = FALLBACK,
    python: str = sys.executable,
) -> list[str]:
    """Return the ``merxen annotate-resolve`` command of one re-run.

    The M8 invocation (``$A/m8/stageA2/scripts/run_resolve.sh``) with M8's
    bundles pinned by MAP run id (its ``resolved_build_hash`` in the store,
    not the store's current bundle), its gate denominators and the pair's
    alignment, plus the sections' registration checks.

    Args:
        run: The M8 run.
        out_dir: The re-run's output directory.
        store: The reference store holding the bundles.
        registration: Sample id to its registration check JSON.
        config_path: The variant's annotation config (``None``: defaults).
        protected_roots: Trees ``--out`` must stay out of.
        fallback: The gene-ID fallback table MAP used.
        python: The interpreter.

    Returns:
        The command.
    """
    command = [
        python,
        "-c",
        CLI_SNIPPET,
        "annotate-resolve",
        "--map-dir",
        str(run.map_dir),
        "--out",
        str(out_dir),
    ]
    for run_id, (reference_id, build_hash) in sorted(run.bundles.items()):
        command += ["--bundle", f"{run_id}={store / reference_id / build_hash}"]
    if fallback is not None:
        command += ["--gene-id-fallback-csv", str(fallback)]
    for sample_id, number in sorted(run.n_segmented.items()):
        command += ["--n-segmented", f"{sample_id}={number}"]
    if run.alignment_dir is not None:
        command += ["--alignment-dir", str(run.alignment_dir)]
    for sample_id, path in sorted(registration.items()):
        command += ["--registration-qc", f"{sample_id}={path}"]
    if config_path is not None:
        command += ["--annotation-config", str(config_path)]
    for root in protected_roots:
        command += ["--results-root", str(root)]
    return command


# --------------------------------------------------------------------------
# Registration (M0a, read only)


@dataclass
class _LayerStore:
    """The SpatialData elements the registration check reads (``sdata``-like)."""

    attrs: dict[str, Any]
    shapes: dict[str, Any] = field(default_factory=dict)
    points: dict[str, Any] = field(default_factory=dict)


def registration_record(
    zarr: Path, *, shape_key: str, reference_shape_key: str
) -> dict[str, Any]:
    """Run the QC stage's registration check on a published store (read only).

    ``compute_segmentation_registration_qc`` on the segmentation's shapes,
    the platform's own cells (when the store holds them) and the store's
    primary points (``choose_primary_points_key`` on the store's
    attributes), each read alone instead of the whole store.

    Args:
        zarr: The section's ``latest_spatialdata.zarr``.
        shape_key: The segmentation's shapes.
        reference_shape_key: The platform's own cells.

    Returns:
        ``RegistrationCheckResult.to_dict`` with its ``inputs``.

    Raises:
        RegressionError: If the store lacks the shapes or any points.
    """
    import dask.dataframe as dd
    import geopandas as gpd

    from merxen.io.spatialdata_schema import choose_primary_points_key
    from merxen.qc.registration import compute_segmentation_registration_qc

    root = read_json(zarr / "zarr.json")
    store = _LayerStore(attrs=dict(root.get("attributes") or {}))
    shape_file = zarr / "shapes" / shape_key / "shapes.parquet"
    if not shape_file.exists():
        raise RegressionError(f"{zarr} has no shapes {shape_key}")
    store.shapes[shape_key] = gpd.read_parquet(shape_file)
    reference_file = zarr / "shapes" / reference_shape_key / "shapes.parquet"
    if reference_file.exists():
        store.shapes[reference_shape_key] = gpd.read_parquet(reference_file)
    points_dir = zarr / "points"
    if points_dir.is_dir():
        for child in sorted(points_dir.iterdir()):
            if (child / "points.parquet").exists():
                store.points[child.name] = dd.read_parquet(child / "points.parquet")
    points_key = choose_primary_points_key(store)
    if points_key is None:
        raise RegressionError(f"{zarr} has no points")
    result = compute_segmentation_registration_qc(
        store,
        shape_key=shape_key,
        points_key=points_key,
        reference_shape_key=reference_shape_key,
    )
    record = result.to_dict()
    record["inputs"] = {
        "zarr": str(zarr),
        "shape_key": shape_key,
        "reference_shape_key": reference_shape_key
        if reference_shape_key in store.shapes
        else None,
        "points_key": points_key,
        "computed_by": "scripts/acceptance/m13_real_qc_regression.py",
    }
    return record


def run_registration(
    out_dir: Path,
    *,
    pairs: Sequence[str],
    segmentations: Sequence[str],
    m8_root: Path,
    results: Path,
    force: bool = False,
    threads: int = 4,
) -> list[Path]:
    """Compute the registration check of every section x segmentation.

    Args:
        out_dir: The regression's output directory.
        pairs: The pairs.
        segmentations: The segmentations.
        m8_root: The M8 RESOLVE root (the sections of each pair).
        results: The published results root.
        force: Recompute existing checks.
        threads: Dask workers reading the points (the QC stage's
            ``DASK_NUM_WORKERS``).

    Returns:
        The check files.
    """
    import dask

    dask.config.set(num_workers=threads)
    written: list[Path] = []
    for segmentation in segmentations:
        for pair in pairs:
            run = read_m8_run(m8_root, pair, segmentation)
            for sample_id, platform in run.samples.items():
                path = registration_path(out_dir, segmentation, sample_id)
                written.append(path)
                if path.exists() and not force:
                    logger.info("registration %s %s: kept", sample_id, segmentation)
                    continue
                shape_key, reference = layer_keys(platform, segmentation)
                started = time.time()
                record = registration_record(
                    latest_zarr(results, pair, platform),
                    shape_key=shape_key,
                    reference_shape_key=reference,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(record, indent=1, default=float) + "\n",
                    encoding="utf-8",
                )
                logger.info(
                    "registration %s %s: %s ratio %.3f shift %s (%.0f s)",
                    sample_id,
                    segmentation,
                    record.get("status"),
                    float(record.get("density_ratio") or float("nan")),
                    record.get("shift_um"),
                    time.time() - started,
                )
    return written


# --------------------------------------------------------------------------
# RESOLVE re-runs


def run_resolve(
    out_dir: Path,
    *,
    pairs: Sequence[str],
    segmentations: Sequence[str],
    variants: Sequence[str],
    m8_root: Path,
    store: Path,
    protected_roots: Sequence[Path],
    fallback: Path | None = FALLBACK,
    force: bool = False,
    dry_run: bool = False,
) -> list[list[str]]:
    """Re-run RESOLVE per pair x segmentation x variant (one at a time).

    Args:
        out_dir: The regression's output directory.
        pairs: The pairs.
        segmentations: The segmentations.
        variants: The variants.
        m8_root: The M8 RESOLVE root.
        store: The reference store.
        protected_roots: Trees the outputs must stay out of.
        fallback: The gene-ID fallback table.
        force: Re-run when a summary exists.
        dry_run: Only return the commands.

    Returns:
        The commands (run, or to run).

    Raises:
        RegressionError: If a registration check is missing or a re-run
            fails.
    """
    configs = out_dir / "configs"
    commands: list[list[str]] = []
    for variant in variants:
        config = variant_config(variant)
        config_path = None
        if config is not None:
            config_path = configs / f"{variant}.json"
            if not dry_run:
                configs.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(config, indent=1) + "\n")
        for segmentation in segmentations:
            for pair in pairs:
                run = read_m8_run(m8_root, pair, segmentation)
                registration = {
                    sample_id: registration_path(out_dir, segmentation, sample_id)
                    for sample_id in run.samples
                }
                missing = [str(p) for p in registration.values() if not p.exists()]
                if missing and not dry_run:
                    raise RegressionError(
                        f"registration checks missing (run the registration step "
                        f"first): {missing}"
                    )
                target = variant_dir(out_dir, variant, pair, segmentation)
                command = resolve_command(
                    run,
                    out_dir=target,
                    store=store,
                    registration=registration,
                    config_path=config_path,
                    protected_roots=protected_roots,
                    fallback=fallback,
                )
                commands.append(command)
                summary = target / f"{pair}_resolve_summary.json"
                if dry_run or (summary.exists() and not force):
                    continue
                target.mkdir(parents=True, exist_ok=True)
                started = time.time()
                with (target / "resolve.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, **SINGLE_THREAD_ENV},
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RegressionError(
                        f"RESOLVE {variant} {pair} {segmentation} failed "
                        f"(exit {completed.returncode}; {target / 'resolve.log'})"
                    )
                logger.info(
                    "resolve %s %s %s: %.0f s",
                    variant,
                    pair,
                    segmentation,
                    time.time() - started,
                )
    return commands


# --------------------------------------------------------------------------
# Tables


def _effect_after_gate(outcome: Mapping[str, Any]) -> str:
    """Return the effect an outcome has once the species gate merges (D20).

    ``none`` for an outcome that did not fire; for a fired one, the lowering
    effect the seeded families' warn-only demotion withheld, else its own.
    """
    if not outcome.get("fired"):
        return "none"
    details = outcome.get("details") or {}
    return str(details.get("warn_only_effect") or outcome.get("effect"))


def _value(outcome: Mapping[str, Any]) -> float | None:
    details = outcome.get("details") or {}
    key = VALUE_KEYS.get(str(outcome.get("check")))
    if key is not None:
        value = details.get(key)
        return None if value is None else float(value)
    if outcome.get("check") == "flag_rates":
        n, k = details.get("n_strata"), details.get("n_uninformative")
        return float(k) / float(n) if n else None
    return None


def outcome_rows(
    summary: Mapping[str, Any], *, pair: str, segmentation: str
) -> list[dict[str, Any]]:
    """Return one row per real-data QC outcome of a resolve summary.

    Args:
        summary: A ``<pair>_resolve_summary.json`` of the ``qc`` variant.
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        Rows with the outcome as applied and the effect it would have once
        the human gate merges (``effect_after_gate``, D20 (b)).

    Raises:
        RegressionError: If a sample records no enabled real-data QC.
    """
    rows: list[dict[str, Any]] = []
    for sample_id, item in sorted((summary.get("samples") or {}).items()):
        block = item.get("real_qc") or {}
        if not block.get("enabled"):
            raise RegressionError(f"{pair} {segmentation} {sample_id}: QC not enabled")
        for outcome in block.get("outcomes") or []:
            details = outcome.get("details") or {}
            rows.append(
                {
                    "pair": pair,
                    "segmentation": segmentation,
                    "sample_id": sample_id,
                    "platform": str(item.get("platform")).upper(),
                    "check": outcome.get("check"),
                    "level": outcome.get("level"),
                    "class": outcome.get("class"),
                    "state": outcome.get("state"),
                    "outcome": outcome.get("outcome"),
                    "fired": bool(outcome.get("fired")),
                    "effect": outcome.get("effect"),
                    "effect_after_gate": _effect_after_gate(outcome),
                    "gate_cap_after_gate": details.get("warn_only_gate_cap")
                    or outcome.get("gate_cap"),
                    "warn_only": bool(block.get("warn_only")),
                    "value": _value(outcome),
                    "reason": outcome.get("reason"),
                    "message": outcome.get("message"),
                }
            )
    return rows


def _gate(item: Mapping[str, Any]) -> dict[str, Any]:
    return dict((item.get("resolution") or {}).get("gate") or {})


def sample_rows(
    qc_summary: Mapping[str, Any],
    free_summary: Mapping[str, Any] | None,
    *,
    pair: str,
    segmentation: str,
) -> list[dict[str, Any]]:
    """Return one row per sample: gate before / after QC and the check tokens.

    Args:
        qc_summary: The ``qc`` variant's summary.
        free_summary: The ``qc_free`` variant's summary (``None``: not run).
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        The rows.
    """
    rows: list[dict[str, Any]] = []
    free_samples = (free_summary or {}).get("samples") or {}
    for sample_id, item in sorted((qc_summary.get("samples") or {}).items()):
        block = item.get("real_qc") or {}
        gate = _gate(item)
        free_gate = _gate(free_samples.get(sample_id) or {})
        row: dict[str, Any] = {
            "pair": pair,
            "segmentation": segmentation,
            "sample_id": sample_id,
            "platform": str(item.get("platform")).upper(),
            "trust_state": (item.get("trust") or {}).get("state"),
            "seeded": block.get("seeded"),
            "warn_only": block.get("warn_only"),
            "gate_level_qc_free": free_gate.get("level"),
            "gate_level_qc": gate.get("level"),
            "gate_warning_qc": gate.get("warning"),
            "qc_warning_reasons": "; ".join(
                str(reason)
                for reason in (block.get("effects") or {}).get("warning_reasons", [])
            ),
            "downgrades": "; ".join(str(d) for d in block.get("downgrades") or []),
        }
        for check, token in sorted((block.get("per_check") or {}).items()):
            row[f"check:{check}"] = token
        rows.append(row)
    return rows


def _outcome(item: Mapping[str, Any], check: str) -> dict[str, Any] | None:
    for outcome in (item.get("real_qc") or {}).get("outcomes") or []:
        if outcome.get("check") == check:
            return dict(outcome)
    return None


def _referee_fields(outcome: Mapping[str, Any] | None, prefix: str) -> dict[str, Any]:
    details = dict((outcome or {}).get("details") or {})
    sets = details.get("marker_sets")
    sets = sets if isinstance(sets, Mapping) else {}
    groups = sets.get("markers")
    groups = groups if isinstance(groups, Mapping) else {}
    return {
        f"{prefix}_outcome": None if outcome is None else outcome.get("outcome"),
        f"{prefix}_consistency": details.get("consistency"),
        f"{prefix}_n_marker_groups": details.get("n_marker_groups"),
        f"{prefix}_n_scored": details.get("n_pseudo_labelled", details.get("n_scored")),
        f"{prefix}_groups": ";".join(
            f"{group}:{len(markers)}" for group, markers in groups.items()
        )
        or None,
        f"{prefix}_markers": "; ".join(
            f"{group}: {','.join(str(m) for m in markers)}"
            for group, markers in groups.items()
        )
        or None,
        f"{prefix}_left_out": ";".join(str(c) for c in sets.get("left_out") or [])
        or None,
        f"{prefix}_fingerprint": details.get("fingerprint"),
        f"{prefix}_reason": None if outcome is None else outcome.get("reason"),
    }


def marker_rows(
    qc_summary: Mapping[str, Any],
    class_summary: Mapping[str, Any] | None,
    *,
    pair: str,
    segmentation: str,
) -> list[dict[str, Any]]:
    """Return the derived marker statistic per sample (D18 re-measure).

    Args:
        qc_summary: The ``qc`` variant (the registered ``node`` comparator).
        class_summary: The ``class_comparator`` variant (reported only).
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        One row per sample.
    """
    class_samples = (class_summary or {}).get("samples") or {}
    rows: list[dict[str, Any]] = []
    for sample_id, item in sorted((qc_summary.get("samples") or {}).items()):
        node = _outcome(item, "marker_consistency")
        alternative = _outcome(class_samples.get(sample_id) or {}, "marker_consistency")
        details = dict((node or {}).get("details") or {})
        rows.append(
            {
                "pair": pair,
                "segmentation": segmentation,
                "sample_id": sample_id,
                "platform": str(item.get("platform")).upper(),
                "warn": details.get("warn"),
                "broad_only": details.get("broad_only"),
                **_referee_fields(node, "node"),
                **_referee_fields(alternative, "class"),
            }
        )
    return rows


def d18_verdict(rows: Sequence[Mapping[str, Any]], *, warn: float) -> dict[str, Any]:
    """Return D18's reading of the derived statistic on set a.

    Only the registered statistic (the ``node`` comparator) decides; the
    ``class`` comparator is reported beside it.

    Args:
        rows: ``marker_rows`` of the set a sections.
        warn: The warning threshold (0.75).

    Returns:
        ``verdict``: ``below_warn`` (any value below ``warn``: the thresholds
        go back to the user before any number of the new family),
        ``not_evaluable`` (no section evaluable: not a pass, the user
        decides), ``partly_not_evaluable`` or ``at_or_above_warn``; with the
        counts and the ``class`` comparator's reading.
    """

    def reading(prefix: str) -> dict[str, Any]:
        values = [row.get(f"{prefix}_consistency") for row in rows]
        evaluated = [float(value) for value in values if value is not None]
        below = [value for value in evaluated if value < warn]
        if below:
            verdict = "below_warn"
        elif not evaluated:
            verdict = "not_evaluable"
        elif len(evaluated) < len(values):
            verdict = "partly_not_evaluable"
        else:
            verdict = "at_or_above_warn"
        return {
            "verdict": verdict,
            "n_sections": len(values),
            "n_evaluated": len(evaluated),
            "n_below_warn": len(below),
            "min": min(evaluated) if evaluated else None,
            "max": max(evaluated) if evaluated else None,
        }

    node = reading("node")
    return {
        **node,
        "warn": warn,
        "decides": "node",
        "thresholds_back_to_user": node["verdict"] == "below_warn",
        "user_decision_needed": node["verdict"] != "at_or_above_warn",
        "class_comparator_reported": reading("class"),
    }


def _jsd_row(rows: Sequence[Mapping[str, Any]], region: str) -> dict[str, Any]:
    for row in rows:
        if row.get("kind") == "soft" and row.get("region") == region:
            return dict(row)
    return {}


def paired_rows(
    summary: Mapping[str, Any], *, pair: str, segmentation: str
) -> list[dict[str, Any]]:
    """Return the pair's paired-concordance reading (D21) with its expectation.

    Args:
        summary: The ``qc`` variant's summary.
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        One row: the shared-mask soft broad JSD (scored point estimate, with
        its CI), the whole section (reported), the outcome both sections
        record, and on the scored segmentation whether D21's expectation
        holds (P5011 warns at .235 / .227, the other pairs pass at <= .154).
    """
    jsd = list((summary.get("pair") or {}).get("jsd") or [])
    shared, whole = _jsd_row(jsd, "shared_mask"), _jsd_row(jsd, "whole_section")
    outcomes = [
        outcome
        for item in (summary.get("samples") or {}).values()
        if (outcome := _outcome(item, "paired_concordance")) is not None
    ]
    tokens = sorted({str(outcome.get("outcome")) for outcome in outcomes})
    after = sorted({_effect_after_gate(outcome) for outcome in outcomes})
    token = tokens[0] if len(tokens) == 1 else "/".join(tokens)
    row: dict[str, Any] = {
        "pair": pair,
        "segmentation": segmentation,
        "shared_mask_jsd": shared.get("jsd"),
        "shared_mask_ci_low": shared.get("ci_low"),
        "shared_mask_ci_high": shared.get("ci_high"),
        "whole_section_jsd": whole.get("jsd"),
        "outcome": token,
        "effect_after_gate": "/".join(after),
        "cross_platform_statistics_level": (
            (summary.get("pair") or {}).get("cross_platform") or {}
        ).get("statistics_level"),
        "expected": None,
        "expectation_met": None,
    }
    if segmentation == SCORED_SEGMENTATION:
        value = shared.get("jsd")
        if pair in PAIRED_EXPECTED_WARN:
            shared_value, whole_value = PAIRED_EXPECTED_WARN[pair]
            row["expected"] = f"warn ({shared_value:.3f} / {whole_value:.3f})"
            row["expectation_met"] = (
                token == "warn"
                and value is not None
                and round(float(value), 3) == shared_value
                and whole.get("jsd") is not None
                and round(float(whole["jsd"]), 3) == whole_value
            )
        else:
            row["expected"] = f"pass (<= {PAIRED_EXPECTED_PASS_MAX:.3f})"
            row["expectation_met"] = (
                token == "pass"
                and value is not None
                and float(value) <= PAIRED_EXPECTED_PASS_MAX
            )
    return [row]


def flag_rate_rows(
    summary: Mapping[str, Any], *, pair: str, segmentation: str
) -> list[dict[str, Any]]:
    """Return both D22 readings of the flag rates per sample.

    Args:
        summary: The ``qc`` variant's summary.
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        One row per sample: the literal reading (scored; class x platform
        strata, H16's 15% marking) and option (a) (each flag's own null
        switch over flag x class x platform; reported).
    """
    rows: list[dict[str, Any]] = []
    for sample_id, item in sorted((summary.get("samples") or {}).items()):
        record = (item.get("real_qc") or {}).get("flag_rates") or {}
        reported = record.get("reported") or {}
        option_a = reported.get("own_switch_all_flags") or {}
        rows.append(
            {
                "pair": pair,
                "segmentation": segmentation,
                "sample_id": sample_id,
                "platform": str(item.get("platform")).upper(),
                "literal_n_strata": record.get("n_strata"),
                "literal_n_uninformative": record.get("n_uninformative"),
                "literal_share": record.get("share_uninformative"),
                "literal_warn": record.get("warn"),
                "literal_uninformative_strata": ";".join(
                    f"{stratum.get('class')}/{stratum.get('platform')}"
                    for stratum in record.get("uninformative_strata") or []
                ),
                "option_a_flags": ";".join(option_a.get("flags") or []),
                "option_a_n_strata": option_a.get("n_strata"),
                "option_a_n_uninformative": option_a.get("n_uninformative"),
                "option_a_share": option_a.get("share"),
                "option_a_would_warn": option_a.get("would_warn"),
                "warn_frac": record.get("warn_frac"),
            }
        )
    return rows


def d22_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return how often each D22 reading warns on the given samples."""
    return {
        "n_samples": len(rows),
        "literal_warnings": sum(bool(row.get("literal_warn")) for row in rows),
        "option_a_warnings": sum(bool(row.get("option_a_would_warn")) for row in rows),
        "literal_max_share": max(
            (float(row["literal_share"]) for row in rows if row.get("literal_share")),
            default=0.0,
        ),
        "option_a_max_share": max(
            (float(row["option_a_share"]) for row in rows if row.get("option_a_share")),
            default=0.0,
        ),
        "scored": "literal (CHECK K6); option (a) needs separate written approval",
    }


def g1_rows(
    summary: Mapping[str, Any], *, pair: str, segmentation: str
) -> list[dict[str, Any]]:
    """Return registration G1 per sample (D23).

    Args:
        summary: The ``qc`` variant's summary.
        pair: The pair id.
        segmentation: The segmentation.

    Returns:
        One row per sample: the check's density ratio and shift, which of
        §7.6's rules fires (``fail``, ``warn`` or none) and the outcome.
    """
    rows: list[dict[str, Any]] = []
    for sample_id, item in sorted((summary.get("samples") or {}).items()):
        outcome = _outcome(item, "registration_g1") or {}
        details = outcome.get("details") or {}
        rows.append(
            {
                "pair": pair,
                "segmentation": segmentation,
                "sample_id": sample_id,
                "platform": str(item.get("platform")).upper(),
                "state": outcome.get("state"),
                "outcome": outcome.get("outcome"),
                "rule": details.get("rule"),
                "density_ratio": details.get("density_ratio"),
                "shift_um": details.get("shift_um"),
                "density_ratio_fail": details.get("density_ratio_fail"),
                "density_ratio_warn": details.get("density_ratio_warn"),
                "shift_fail_um": details.get("shift_fail_um"),
                "effect_setting": details.get("effect_setting"),
                "source": details.get("source"),
                "reason": outcome.get("reason"),
            }
        )
    return rows


def d23_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return D23's reading of G1 on set a (a false failure or none).

    Every set a section is accepted data (gate H), so G1's fail rule firing
    on one is a false failure.

    Args:
        rows: ``g1_rows`` of the set a sections.

    Returns:
        ``verdict``: ``false_failure`` (G1 stays warn-only, D23 (b)),
        ``incomplete`` (a section not evaluated: nothing can be decided) or
        ``no_false_failure`` (the move to (a), a tightening, can be decided
        and recorded before any QC output of the new family is read); with
        the sections behind it.
    """
    failed = [row for row in rows if row.get("rule") == "fail"]
    warned = [row for row in rows if row.get("rule") == "warn"]
    unevaluated = [row for row in rows if row.get("state") != "evaluated"]
    if failed:
        verdict = "false_failure"
    elif unevaluated or not rows:
        verdict = "incomplete"
    else:
        verdict = "no_false_failure"

    def names(items: Sequence[Mapping[str, Any]]) -> list[str]:
        return [f"{row['sample_id']}/{row['segmentation']}" for row in items]

    return {
        "verdict": verdict,
        "n_sections": len(rows),
        "fail_rule": names(failed),
        "warn_rule": names(warned),
        "not_evaluated": names(unevaluated),
        "decision": "the move to D23 (a) is the user's (recorded in the "
        "pre-registration before any QC output of the new family is read)",
    }


def label_identity(new: pd.DataFrame, old: pd.DataFrame) -> list[str]:
    """List how a re-run's label table differs from the M8 table.

    Columns M8 predates are allowed only if listed in ``NEW_NULL_COLUMNS``
    and null throughout.

    Args:
        new: The re-run's label table.
        old: The M8 label table.

    Returns:
        One message per difference (empty: identical).
    """
    problems: list[str] = []
    if len(new) != len(old):
        return [f"{len(new)} rows, M8 {len(old)}"]
    missing = sorted(set(old.columns) - set(new.columns))
    if missing:
        problems.append(f"columns missing: {missing}")
    for column in sorted(set(new.columns) - set(old.columns)):
        if column not in NEW_NULL_COLUMNS:
            problems.append(f"new column {column}")
        elif not new[column].isna().all():
            problems.append(f"new column {column} is not null throughout")
    for column in old.columns:
        if column in new.columns and not new[column].equals(old[column]):
            before = old[column].astype(object).reset_index(drop=True)
            after = new[column].astype(object).reset_index(drop=True)
            same = (before == after) | (before.isna() & after.isna())
            problems.append(f"column {column} differs ({int((~same).sum())} rows)")
    return problems


def _sample_files(directory: Path, item: Mapping[str, Any]) -> tuple[Path, Path]:
    return directory / str(item["labels"]), directory / str(item["annotation_manifest"])


def nr1_rows(
    *,
    pair: str,
    segmentation: str,
    qc_dir: Path,
    qc_summary: Mapping[str, Any],
    free_dir: Path | None,
    free_summary: Mapping[str, Any] | None,
    m8_dir: Path,
    m8_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return NR1 (``qc`` vs ``qc_free``) and the identity with M8 per sample.

    Args:
        pair: The pair id.
        segmentation: The segmentation.
        qc_dir: The ``qc`` re-run's directory.
        qc_summary: Its summary.
        free_dir: The ``qc_free`` re-run's directory (``None``: not run).
        free_summary: Its summary.
        m8_dir: The M8 RESOLVE directory.
        m8_summary: The M8 summary.

    Returns:
        One row per sample: NR1's violations and both identities with M8.
    """
    from merxen.annotation.real_qc import downgrade_only_violations

    rows: list[dict[str, Any]] = []
    m8_samples = m8_summary.get("samples") or {}
    free_samples = (free_summary or {}).get("samples") or {}
    for sample_id, item in sorted((qc_summary.get("samples") or {}).items()):
        labels_path, manifest_path = _sample_files(qc_dir, item)
        applied = pd.read_parquet(labels_path)
        old_item = m8_samples.get(sample_id)
        m8_labels = (
            None
            if old_item is None
            else pd.read_parquet(_sample_files(m8_dir, old_item)[0])
        )
        row: dict[str, Any] = {
            "pair": pair,
            "segmentation": segmentation,
            "sample_id": sample_id,
            "platform": str(item.get("platform")).upper(),
            "n_cells": len(applied),
        }
        row["qc_vs_m8"] = (
            "no M8 table"
            if m8_labels is None
            else "; ".join(label_identity(applied, m8_labels)) or "identical"
        )
        free_item = free_samples.get(sample_id)
        if free_dir is None or free_item is None:
            row["nr1_violations"] = "qc_free not run"
            row["qc_free_vs_m8"] = "qc_free not run"
        else:
            free_labels_path, free_manifest_path = _sample_files(free_dir, free_item)
            free = pd.read_parquet(free_labels_path)
            problems = downgrade_only_violations(
                free,
                applied,
                species="human",
                free_provenance=read_json(free_manifest_path),
                applied_provenance=read_json(manifest_path),
                free_gate=_gate(free_item),
                applied_gate=_gate(item),
                qc=(item.get("real_qc") or {}).get("effects"),
            )
            row["nr1_violations"] = "; ".join(problems) or "none"
            row["qc_free_vs_m8"] = (
                "no M8 table"
                if m8_labels is None
                else "; ".join(label_identity(free, m8_labels)) or "identical"
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Assembly and report


def _code_record() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import hashlib

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def build_tables(
    out_dir: Path,
    *,
    pairs: Sequence[str],
    segmentations: Sequence[str],
    m8_root: Path,
) -> dict[str, Any]:
    """Write the outcome table, the decision readings and the report.

    Args:
        out_dir: The regression's output directory (with ``resolve/``).
        pairs: The pairs.
        segmentations: The segmentations.
        m8_root: The M8 RESOLVE root.

    Returns:
        The regression summary (also written as JSON).

    Raises:
        RegressionError: If the ``qc`` re-run of a pair x segmentation is
            missing.
    """
    from merxen.annotation.config import AnnotationConfig

    config = AnnotationConfig(species="human")
    warn = float(config.real_qc.marker_consistency_warn or 0.75)
    tables: dict[str, list[dict[str, Any]]] = {
        "outcomes": [],
        "samples": [],
        "marker_consistency": [],
        "paired_concordance": [],
        "flag_rates": [],
        "registration_g1": [],
        "nr1": [],
    }
    inputs: list[dict[str, Any]] = []
    for segmentation in segmentations:
        for pair in pairs:
            run = read_m8_run(m8_root, pair, segmentation)
            m8_summary = read_json(run.resolve_dir / f"{pair}_resolve_summary.json")
            directories = {
                variant: variant_dir(out_dir, variant, pair, segmentation)
                for variant in VARIANTS
            }
            summaries: dict[str, dict[str, Any] | None] = {}
            for variant, directory in directories.items():
                path = directory / f"{pair}_resolve_summary.json"
                summaries[variant] = read_json(path) if path.exists() else None
            qc_summary = summaries[VARIANT_QC]
            if qc_summary is None:
                raise RegressionError(
                    f"no qc re-run of {pair} {segmentation} under {out_dir}: run "
                    "the resolve step first"
                )
            inputs.append(
                {
                    "pair": pair,
                    "segmentation": segmentation,
                    "m8_resolve_dir": str(run.resolve_dir),
                    "m8_summary_sha256": run.summary_sha256,
                    "map_dir": str(run.map_dir),
                    "bundles": {
                        key: list(value) for key, value in sorted(run.bundles.items())
                    },
                    "variants_run": sorted(
                        variant for variant, value in summaries.items() if value
                    ),
                }
            )
            keys = {"pair": pair, "segmentation": segmentation}
            tables["outcomes"] += outcome_rows(qc_summary, **keys)
            tables["samples"] += sample_rows(
                qc_summary, summaries[VARIANT_QC_FREE], **keys
            )
            tables["marker_consistency"] += marker_rows(
                qc_summary, summaries[VARIANT_CLASS], **keys
            )
            tables["paired_concordance"] += paired_rows(qc_summary, **keys)
            tables["flag_rates"] += flag_rate_rows(qc_summary, **keys)
            tables["registration_g1"] += g1_rows(qc_summary, **keys)
            free_summary = summaries[VARIANT_QC_FREE]
            tables["nr1"] += nr1_rows(
                **keys,
                qc_dir=directories[VARIANT_QC],
                qc_summary=qc_summary,
                free_dir=None if free_summary is None else directories[VARIANT_QC_FREE],
                free_summary=free_summary,
                m8_dir=run.resolve_dir,
                m8_summary=m8_summary,
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        _frame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    scored = [
        row
        for row in tables["marker_consistency"]
        if row["segmentation"] == SCORED_SEGMENTATION
    ]
    paired = tables["paired_concordance"]
    nr1 = tables["nr1"]
    result: dict[str, Any] = {
        "code": _code_record(),
        "inputs": inputs,
        "config": {
            "real_qc": config.real_qc.model_dump(mode="json"),
            "variants": {variant: variant_config(variant) for variant in VARIANTS},
        },
        "d18_marker_consistency": {
            "scored_segmentation": d18_verdict(scored, warn=warn),
            "all_segmentations": d18_verdict(tables["marker_consistency"], warn=warn),
        },
        "d20_warn_only": {
            "warn_only_samples": sum(bool(r["warn_only"]) for r in tables["samples"]),
            "n_samples": len(tables["samples"]),
            "withheld_effects": sorted(
                {
                    f"{row['sample_id']}/{row['segmentation']}:{row['check']}:"
                    f"{row['effect_after_gate']}"
                    for row in tables["outcomes"]
                    if row["fired"] and row["effect_after_gate"] != row["effect"]
                }
            ),
        },
        "d21_paired_concordance": {
            "expectation_met": all(
                bool(row["expectation_met"])
                for row in paired
                if row["expectation_met"] is not None
            ),
            "rows": paired,
        },
        "d22_flag_rates": d22_summary(tables["flag_rates"]),
        "d23_registration_g1": d23_verdict(tables["registration_g1"]),
        "nr1": {
            "violations": [
                f"{row['sample_id']}/{row['segmentation']}: {row['nr1_violations']}"
                for row in nr1
                if row["nr1_violations"] not in ("none",)
            ],
            "qc_identical_to_m8": all(row["qc_vs_m8"] == "identical" for row in nr1),
            "qc_free_identical_to_m8": all(
                row["qc_free_vs_m8"] == "identical" for row in nr1
            ),
        },
        "not_covered": [
            "ag7 and VZG2 (mouse) rows: chunk C22, after M6b",
            "proseg_mask and original_seg: no M8 RESOLVE outputs",
        ],
    }
    (out_dir / "regression_summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    report = render_report(result, tables)
    (out_dir / "REPORT.txt").write_text(report, encoding="utf-8")
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(
    result: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]
) -> str:
    """Return the regression report text.

    Args:
        result: The regression summary.
        tables: The tables' rows.

    Returns:
        The report.
    """
    code = result["code"]
    lines = [
        "M13 C17: real-data QC regression on the M8 human pairs (set a)",
        "=" * 66,
        f"code {code.get('commit')} ({code.get('branch')}; dirty "
        f"{code.get('dirty')}); M8 RESOLVE outputs read only; RESOLVE re-run "
        "with M8's MAP output, bundles, gate denominators and alignment, plus "
        "the registration checks computed here.",
        "",
        "1. Outcomes (registered configuration; the seeded set a family is "
        "warn-only until gate H merges into main, D20 (b)). dataset_gate is "
        "the gate's own verdict (broad_only or failed: fail), recorded, never "
        "re-applied by the QC.",
    ]
    checks = sorted(
        {key for row in tables["samples"] for key in row if key.startswith("check:")}
    )
    lines.append(
        "   "
        + f"{'sample':16s} {'seg':13s} "
        + " ".join(name.removeprefix("check:")[:14].ljust(14) for name in checks)
    )
    for row in tables["samples"]:
        lines.append(
            "   "
            + f"{row['sample_id']:16s} {row['segmentation']:13s} "
            + " ".join(str(row.get(name) or "-")[:14].ljust(14) for name in checks)
        )
    d20 = result["d20_warn_only"]
    lines += [
        f"   warn-only samples {d20['warn_only_samples']} of {d20['n_samples']}; "
        "effects withheld until the gate merges: "
        + (", ".join(d20["withheld_effects"]) or "none"),
        "",
        "2. D18: the derived marker statistic re-measured on set a "
        "(node comparator decides; class comparator reported)",
    ]
    for row in tables["marker_consistency"]:
        lines.append(
            f"   {row['sample_id']:16s} {row['segmentation']:13s} node "
            f"{_fmt(row['node_consistency'])} ({row['node_n_marker_groups']} groups"
            f" {row['node_groups'] or '-'}; {row['node_reason'] or 'evaluated'}) | "
            f"class {_fmt(row['class_consistency'])} "
            f"({row['class_n_marker_groups']} groups {row['class_groups'] or '-'}; "
            f"{row['class_reason'] or 'evaluated'})"
        )
    d18 = result["d18_marker_consistency"]["scored_segmentation"]
    lines += [
        f"   scored segmentation: {d18['verdict']} ({d18['n_evaluated']} of "
        f"{d18['n_sections']} evaluated, {d18['n_below_warn']} below "
        f"{d18['warn']}); class comparator (reported): "
        f"{d18['class_comparator_reported']['verdict']}",
        "   D18: below 0.75 on set a sends the thresholds back to the user; "
        "a not_evaluable statistic is not a pass and also needs the user's "
        "decision before the new family is scored.",
        "",
        "3. D21: paired concordance (soft broad JSD; shared mask scored, whole "
        "section reported)",
    ]
    for row in tables["paired_concordance"]:
        lines.append(
            f"   {row['pair']:6s} {row['segmentation']:13s} shared "
            f"{_fmt(row['shared_mask_jsd'])} [{_fmt(row['shared_mask_ci_low'])}, "
            f"{_fmt(row['shared_mask_ci_high'])}] whole "
            f"{_fmt(row['whole_section_jsd'])}: {row['outcome']} (after the gate: "
            f"{row['effect_after_gate']}); expected {row['expected'] or 'reported'}"
            f"; met {_fmt(row['expectation_met'])}"
        )
    d21 = result["d21_paired_concordance"]
    d22 = result["d22_flag_rates"]
    lines += [
        f"   expectation met: {d21['expectation_met']}",
        "",
        "4. D22: flag rates, literal reading (scored) vs option (a) (reported)",
    ]
    for row in tables["flag_rates"]:
        lines.append(
            f"   {row['sample_id']:16s} {row['segmentation']:13s} literal "
            f"{row['literal_n_uninformative']}/{row['literal_n_strata']} "
            f"warn {row['literal_warn']} | option (a) "
            f"{row['option_a_n_uninformative']}/{row['option_a_n_strata']} "
            f"warn {row['option_a_would_warn']}"
        )
    lines += [
        f"   warnings: literal {d22['literal_warnings']}, option (a) "
        f"{d22['option_a_warnings']} of {d22['n_samples']} samples",
        "",
        "5. D23: registration G1 (§7.6 rules on the M0a check; warn-only in M13)",
    ]
    for row in tables["registration_g1"]:
        lines.append(
            f"   {row['sample_id']:16s} {row['segmentation']:13s} ratio "
            f"{_fmt(row['density_ratio'])} shift {_fmt(row['shift_um'], 1)} um: "
            f"rule {row['rule'] or 'none'}, outcome {row['outcome']}"
        )
    d23 = result["d23_registration_g1"]
    lines += [
        f"   verdict: {d23['verdict']} (fail rule: "
        f"{', '.join(d23['fail_rule']) or 'none'}; warn rule: "
        f"{', '.join(d23['warn_rule']) or 'none'}; not evaluated: "
        f"{', '.join(d23['not_evaluated']) or 'none'})",
        f"   {d23['decision']}",
        "",
        "6. NR1 (qc vs qc_free) and identity with the M8 label tables",
    ]
    for row in tables["nr1"]:
        lines.append(
            f"   {row['sample_id']:16s} {row['segmentation']:13s} NR1 "
            f"{row['nr1_violations']}; qc vs M8 {row['qc_vs_m8']}; qc_free vs M8 "
            f"{row['qc_free_vs_m8']}"
        )
    nr1 = result["nr1"]
    lines += [
        f"   NR1 violations: {len(nr1['violations'])}; qc identical to M8: "
        f"{nr1['qc_identical_to_m8']}; qc_free identical to M8: "
        f"{nr1['qc_free_identical_to_m8']}",
        "",
        "Not covered: " + "; ".join(result["not_covered"]) + ".",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "step", choices=("registration", "resolve", "table", "all"), help="step"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--m8-resolve", type=Path, default=M8_RESOLVE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--gene-id-fallback", type=Path, default=FALLBACK)
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS))
    parser.add_argument("--segmentations", nargs="+", default=list(SEGMENTATIONS))
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--force", action="store_true", help="recompute outputs")
    parser.add_argument(
        "--threads", type=int, default=4, help="registration: dask workers"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve: print the commands only"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    protected = [args.m8_resolve, args.results, args.store]
    check_out_dir(args.out, protected)
    if args.step in ("registration", "all"):
        run_registration(
            args.out,
            pairs=args.pairs,
            segmentations=args.segmentations,
            m8_root=args.m8_resolve,
            results=args.results,
            force=args.force,
            threads=args.threads,
        )
    if args.step in ("resolve", "all"):
        commands = run_resolve(
            args.out,
            pairs=args.pairs,
            segmentations=args.segmentations,
            variants=args.variants,
            m8_root=args.m8_resolve,
            store=args.store,
            protected_roots=[args.m8_resolve, args.results],
            fallback=args.gene_id_fallback,
            force=args.force,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            for command in commands:
                print(" ".join(command))
            return 0
    if args.step in ("table", "all"):
        build_tables(
            args.out,
            pairs=args.pairs,
            segmentations=args.segmentations,
            m8_root=args.m8_resolve,
        )
        print((args.out / "REPORT.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
