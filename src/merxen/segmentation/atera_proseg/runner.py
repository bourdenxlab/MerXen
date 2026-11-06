"""Command line, version check, run and log parsing of one ATERA ProSeg tile.

The command is the AT-5(b) benchmark command in Xenium parquet layout
(:func:`build_command`, golden-tested). The prior weight is explicit: ProSeg
multiplies ``--prior-weight`` by ``max(mean transcripts per cell / 100, 1)``
(``main.rs``), so the multiplier of each tile is chosen to make the effective
weight ProSeg prints equal the requested one. ProSeg takes the mean after its
own filters (the cell-free filter drops rows the multiplier counted), so a tile
whose printed weight is off is restarted once with the multiplier rescaled by
requested / printed (:func:`merxen.segmentation.atera_proseg.tile.run_tile`);
it fails if the weight is still off.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from merxen.config import AteraProsegModelParams
from merxen.memory import RssWatchdog
from merxen.segmentation.atera_proseg import (
    REQUIRED_PROSEG_VERSION,
    AteraProsegError,
    PriorWeightMismatchError,
    ProsegRunError,
)

logger = logging.getLogger(__name__)

OUTPUT_SPATIALDATA = "proseg.zarr"
OUTPUT_CELL_METADATA = "cell-metadata.parquet"
OUTPUT_TRANSCRIPT_METADATA = "transcript-metadata.parquet"
OUTPUT_CELL_POLYGONS = "cell-polygons.geojson.gz"

_PRIOR_WEIGHT_LINE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+prior weight\s*$")
_TIMESTAMP = re.compile(r"^\[(\S+Z) ")
_SAMPLING = re.compile(
    r"sampling times: morphology: ([\d.]+)(ns|µs|us|ms|s), "
    r"repo: ([\d.]+)(ns|µs|us|ms|s), params: ([\d.]+)(ns|µs|us|ms|s)"
)
_PHASE = re.compile(
    r"\] (loaded transcripts|initialized voxels|write_transcript_metadata"
    r"|write consensus polygons|write SpatialData): ([\d.]+)(ns|µs|us|ms|s)"
)
_UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def prior_weight_multiplier(
    effective_weight: float, n_rows: int, n_labels: int
) -> float:
    """Return the ``--prior-weight`` that gives ``effective_weight``.

    Args:
        effective_weight: Weight ProSeg should report.
        n_rows: Transcripts given to ProSeg.
        n_labels: Prior cells in the window.

    Returns:
        ``W / max(n_rows / n_labels / 100, 1)``.
    """
    mean = float(n_rows) / max(int(n_labels), 1)
    return float(effective_weight) / max(mean / 100.0, 1.0)


def _number(value: float) -> str:
    """Format a float for the command line without losing precision."""
    return repr(float(value))


def build_command(
    binary: str | Path,
    *,
    parquet_path: str | Path,
    prior_path: str | Path,
    output_dir: str | Path,
    model: AteraProsegModelParams,
    prior_weight: float,
    pixel_size_um: float,
    col0: int,
    row0: int,
    num_threads: int,
) -> list[str]:
    """Return the ProSeg argument list of one tile.

    Args:
        binary: The ProSeg executable.
        parquet_path: The Xenium-layout input.
        prior_path: The uint32 vendor mask of the window.
        output_dir: ProSeg ``--output-path``.
        model: Model and schedule.
        prior_weight: ``--prior-weight`` (the multiplier).
        pixel_size_um: Microns per mask pixel.
        col0: First column of the window.
        row0: First row of the window.
        num_threads: ``--nthreads``.

    Returns:
        The argument list (``binary`` first).
    """
    ps = float(pixel_size_um)
    command = [
        str(binary),
        str(parquet_path),
        "xenium",
        "-x",
        "x_location",
        "-y",
        "y_location",
        "-z",
        "z_location",
        "--gene-column",
        "feature_name",
        "--cell-id-column",
        "cell_id",
        "--cell-id-unassigned",
        "0",
        "--transcript-id-column",
        "transcript_id",
        "--qv-column",
        "qv",
        "--min-qv",
        _number(model.min_qv),
        "--compartment-column",
        "overlaps_nucleus",
        "--compartment-nuclear",
        "1",
        "--fov-column",
        "fov_name",
        "--voxel-size",
        _number(model.voxel_size),
        "--burnin-voxel-size",
        _number(model.burnin_voxel_size),
        "--voxel-layers",
        str(int(model.voxel_layers)),
        "--nuclear-reassignment-prob",
        _number(model.nuclear_reassignment_prob),
        "--diffusion-probability",
        _number(model.diffusion_probability),
        "--cell-compactness",
        _number(model.cell_compactness),
        "--burnin-samples",
        str(int(model.burnin_samples)),
        "--samples",
        str(int(model.samples)),
        "--hillclimb",
        str(int(model.hillclimb)),
        "--recorded-samples",
        str(int(model.recorded_samples)),
        "--prior-weight",
        _number(prior_weight),
    ]
    if model.prior_seg_reassignment_prob is not None:
        command += [
            "--prior-seg-reassignment-prob",
            _number(model.prior_seg_reassignment_prob),
        ]
    if model.use_cell_initialization:
        command.append("--use-cell-initialization")
    command += [
        "--cellpose-masks",
        str(prior_path),
        "--cellpose-x-transform",
        _number(ps),
        "0",
        _number(int(col0) * ps),
        "--cellpose-y-transform",
        "0",
        _number(ps),
        _number(int(row0) * ps),
        "--nthreads",
        str(int(num_threads)),
        "--output-path",
        str(output_dir),
        "--output-spatialdata",
        OUTPUT_SPATIALDATA,
        "--exclude-spatialdata-transcripts",
        "--output-cell-metadata",
        OUTPUT_CELL_METADATA,
        "--output-transcript-metadata",
        OUTPUT_TRANSCRIPT_METADATA,
        "--output-cell-polygons",
        OUTPUT_CELL_POLYGONS,
        "--overwrite",
    ]
    return command


def proseg_version(binary: str | Path) -> str:
    """Return the version ProSeg reports (``proseg 3.2.0`` -> ``3.2.0``)."""
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise AteraProsegError(f"ProSeg --version failed: {result.stderr.strip()}")
    return result.stdout.strip().split()[-1]


def check_version(binary: str | Path, *, allow_untested: bool = False) -> str:
    """Return the ProSeg version, refusing any other than the verified one.

    Args:
        binary: The ProSeg executable.
        allow_untested: Run another version anyway (with a warning), as
            ``atera_proseg_allow_untested_version`` lets the preflight do.

    Raises:
        AteraProsegError: For another version unless ``allow_untested``.
    """
    version = proseg_version(binary)
    if version != REQUIRED_PROSEG_VERSION and allow_untested:
        logger.warning(
            "Running the ATERA ProSeg route with ProSeg %s; it is verified with "
            "%s only (atera_proseg_allow_untested_version)",
            version,
            REQUIRED_PROSEG_VERSION,
        )
    elif version != REQUIRED_PROSEG_VERSION:
        raise AteraProsegError(
            f"The ATERA ProSeg route needs ProSeg {REQUIRED_PROSEG_VERSION} "
            f"(Xenium-layout parquet reader and flags); found {version}"
        )
    return version


def parse_prior_weight(stdout: str) -> float | None:
    """Return the effective prior weight ProSeg printed, or ``None``."""
    for line in stdout.splitlines():
        match = _PRIOR_WEIGHT_LINE.match(line)
        if match:
            return float(match.group(1))
    return None


def _timestamp(line: str) -> float | None:
    match = _TIMESTAMP.match(line)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_timings(stderr: str, burnin_samples: int) -> dict[str, Any]:
    """Summarise ProSeg's ``RUST_LOG=info`` lines.

    Args:
        stderr: ProSeg's log.
        burnin_samples: Burn-in iterations (to split the iteration times).

    Returns:
        Phase durations in seconds (``loaded transcripts`` and so on), the
        number of logged iterations and the mean burn-in and sampling seconds
        per iteration (from the iteration timestamps).
    """
    phases: dict[str, float] = {}
    stamps: list[float] = []
    for line in stderr.splitlines():
        phase = _PHASE.search(line)
        if phase:
            phases[phase.group(1)] = round(
                float(phase.group(2)) * _UNITS[phase.group(3)], 3
            )
        if _SAMPLING.search(line):
            stamp = _timestamp(line)
            if stamp is not None:
                stamps.append(stamp)
    result: dict[str, Any] = {"phases_s": phases, "iterations_logged": len(stamps)}
    burnin = min(int(burnin_samples), len(stamps))
    if burnin > 1:
        result["burnin_s_per_iter"] = round(
            (stamps[burnin - 1] - stamps[0]) / (burnin - 1), 3
        )
    if len(stamps) > burnin + 1:
        result["sampling_s_per_iter"] = round(
            (stamps[-1] - stamps[burnin]) / (len(stamps) - 1 - burnin), 3
        )
    return result


@dataclass
class ProsegRun:
    """Outcome of one ProSeg run.

    Attributes:
        command: The argument list.
        version: ProSeg version.
        returncode: Exit status.
        wall_s: Wall time in seconds.
        peak_anonymous_rss_gb: Peak anonymous RSS of the task's process tree.
        reported_prior_weight: Effective weight ProSeg printed.
        timings: Parsed log timings.
        stdout_path: Saved stdout.
        stderr_path: Saved stderr.
    """

    command: list[str]
    version: str
    returncode: int = 0
    wall_s: float = 0.0
    peak_anonymous_rss_gb: float = 0.0
    reported_prior_weight: float | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    stdout_path: Path | None = None
    stderr_path: Path | None = None


def _early_prior_weight_problem(
    stdout_path: Path, expected: float | None, tolerance: float
) -> PriorWeightMismatchError | None:
    """Return why the printed prior weight is wrong, once ProSeg printed it.

    ProSeg prints its dataset summary (transcripts, cells, genes, fovs, then
    the prior weight when it exceeds 1) right after voxel initialisation, a
    few minutes into a run of tens of minutes, so a wrong weight stops the run
    early instead of after the sampling.
    """
    if expected is None:
        return None
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if " fovs" not in text:
        return None
    reported = parse_prior_weight(text)
    try:
        check_prior_weight(reported, expected, tolerance)
    except PriorWeightMismatchError as error:
        return error
    return None


def run_proseg(
    command: Sequence[str],
    *,
    log_dir: Path,
    burnin_samples: int,
    max_rss_gb: float | None,
    cleanup: Sequence[Path] = (),
    version: str = REQUIRED_PROSEG_VERSION,
    expected_prior_weight: float | None = None,
    prior_weight_tolerance: float = 0.02,
    poll_s: float = 5.0,
    cwd: Path | None = None,
) -> ProsegRun:
    """Run ProSeg under the process-tree memory watchdog.

    Args:
        command: The argument list.
        log_dir: Where ``proseg.stdout.log`` and ``proseg.stderr.log`` go.
        burnin_samples: For the timing summary.
        max_rss_gb: Watchdog cap; above it the task exits 99.
        cleanup: Paths the watchdog removes before exiting.
        version: The checked ProSeg version (recorded).
        expected_prior_weight: Effective weight ProSeg must report; the run is
            stopped as soon as it reports another (``None``: not checked).
        prior_weight_tolerance: Relative tolerance of that check.
        poll_s: Seconds between two looks at ProSeg's output.
        cwd: Working directory of ProSeg, which writes debug files there
            (``membership_debug.txt``); ``None`` uses ``log_dir``. Paths in
            ``command`` must not be relative to the caller's directory.

    Returns:
        The run record.

    Raises:
        ProsegRunError: If ProSeg exits non-zero.
        PriorWeightMismatchError: If ProSeg reports another prior weight.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "proseg.stdout.log"
    stderr_path = log_dir / "proseg.stderr.log"
    env = {**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "info")}
    start = time.monotonic()
    watchdog = RssWatchdog(max_rss_gb, interval_s=5.0, name="ATERA_PROSEG_TILE")
    watchdog.cleanup.extend(cleanup)
    weight_problem: PriorWeightMismatchError | None = None
    weight_checked = expected_prior_weight is None
    with (
        stdout_path.open("w", encoding="utf-8") as out,
        stderr_path.open("w", encoding="utf-8") as err,
        watchdog,
    ):
        process = subprocess.Popen(
            list(command),
            stdout=out,
            stderr=err,
            env=env,
            cwd=str(cwd if cwd is not None else log_dir),
        )
        while True:
            try:
                process.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            if not weight_checked:
                out.flush()
                text = stdout_path.read_text(encoding="utf-8", errors="replace")
                if " fovs" in text:
                    weight_checked = True
                    weight_problem = _early_prior_weight_problem(
                        stdout_path, expected_prior_weight, prior_weight_tolerance
                    )
                    if weight_problem is not None:
                        process.terminate()
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        break
    wall = time.monotonic() - start
    if weight_problem is not None:
        raise PriorWeightMismatchError(
            f"{weight_problem}; ProSeg was stopped",
            reported=weight_problem.reported,
            requested=weight_problem.requested,
        )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    run = ProsegRun(
        command=list(command),
        version=version,
        returncode=int(process.returncode),
        wall_s=round(wall, 1),
        peak_anonymous_rss_gb=round(watchdog.peak_gb, 3),
        reported_prior_weight=parse_prior_weight(stdout),
        timings=parse_timings(stderr, burnin_samples),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if process.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-20:])
        raise ProsegRunError(
            f"ProSeg exited with status {process.returncode}; last log lines:\n{tail}"
        )
    return run


def check_prior_weight(
    reported: float | None, requested: float, tolerance: float
) -> None:
    """Fail when ProSeg's effective prior weight is not the requested one.

    Raises:
        PriorWeightMismatchError: When the weight is missing or off by more
            than ``tolerance`` (relative).
    """
    if reported is None:
        raise PriorWeightMismatchError(
            "ProSeg printed no prior weight; the effective weight could not be "
            f"checked against {requested}",
            reported=None,
            requested=requested,
        )
    if abs(reported - requested) > tolerance * requested:
        raise PriorWeightMismatchError(
            f"ProSeg's effective prior weight {reported} differs from the requested "
            f"{requested} by more than {tolerance:.0%}",
            reported=reported,
            requested=requested,
        )
