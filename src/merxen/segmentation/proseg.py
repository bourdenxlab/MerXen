"""ProSeg subprocess orchestration vendored and trimmed from MOSAIK."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

logger = logging.getLogger(__name__)


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    """Parse a semantic version string into integer components."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _check_proseg_available(proseg_binary: str | Path) -> str:
    """Verify ProSeg is executable and return its reported version."""
    try:
        result = subprocess.run(  # noqa: S603
            [str(proseg_binary), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        msg = (
            f"Proseg binary '{proseg_binary}' not found or not executable. "
            "Nextflow runs ENSURE_PROSEG automatically; for direct CLI use, "
            "install it with `cargo install proseg`."
        )
        raise RuntimeError(msg) from exc

    stdout = (result.stdout or "").strip()
    semver = _parse_semver(stdout)
    if semver is not None:
        return ".".join(str(x) for x in semver)
    parts = stdout.split()
    return parts[1] if len(parts) > 1 else "unknown"


def _check_proseg_version_gte(version: str, target: str) -> bool:
    """Return True when ``version`` is >= ``target``.

    If parsing fails, this function defaults to True to preserve compatibility
    with mixed binary outputs across ProSeg versions.
    """
    parsed_version = _parse_semver(version)
    parsed_target = _parse_semver(target)
    if parsed_version is None or parsed_target is None:
        logger.warning(
            "Could not parse ProSeg version '%s'; assuming it supports modern flags.",
            version,
        )
        return True
    return parsed_version >= parsed_target


def _validate_columns_from_csv(csv_path: Path, required_columns: list[str]) -> None:
    """Validate that required columns exist in an input CSV header."""
    header_columns = list(pd.read_csv(csv_path, nrows=0).columns)
    missing_columns = [col for col in required_columns if col not in header_columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in transcript CSV {csv_path}: {missing_columns}"
        )


def _to_optional_path_array(
    value: np.ndarray | str | Path | None,
    tmp_path: Path,
    filename: str,
    dtype: DTypeLike,
) -> Path | None:
    """Materialize array-like input to disk and return a path."""
    if value is None:
        return None
    if isinstance(value, str | Path):
        return Path(value)

    out_path = tmp_path / filename
    np.save(out_path, np.asarray(value, dtype=dtype))
    return out_path


_PROSEG_ITER_RE = re.compile(r"[Ii]ter(?:ation)?\s*[:\s]\s*(\d+)(?:\s*/\s*(\d+))?")


def run_proseg_refinement(
    transcripts_df: pd.DataFrame | str | Path,
    output_path: str | Path,
    proseg_binary: str | Path,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    gene_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    transcript_id_col: str | None = None,
    qv_col: str | None = None,
    samples: int = 1000,
    burnin_voxel_size: float | None = None,
    voxel_size: float = 0.5,
    voxel_layers: int = 1,
    nuclear_reassignment_prob: float = 0.2,
    diffusion_probability: float = 0.2,
    cell_compactness: float | None = None,
    expand_initialized_cells: int | None = None,
    use_cell_initialization: bool = False,
    prior_seg_reassignment_prob: float | None = None,
    morphology_steps_per_iter: int | None = None,
    max_transcript_nucleus_distance: float | None = None,
    cellpose_masks: np.ndarray | str | Path | None = None,
    cellpose_cellprobs: np.ndarray | str | Path | None = None,
    cellpose_scale: float | None = None,
    cellpose_x_transform: tuple[float, float, float] | None = None,
    cellpose_y_transform: tuple[float, float, float] | None = None,
    diffusion_sigma_far: float | None = None,
    num_threads: int = 12,
    overwrite: bool = True,
    logger: logging.Logger | None = None,
    progress_callback: Any = None,
    proseg_samples: int | None = None,
) -> Path:
    """Run ProSeg refinement on transcript data.

    Args:
        transcripts_df: Input transcript table or CSV path.
        output_path: Output SpatialData zarr path.
        proseg_binary: Path to the ProSeg executable.
        x_col: X coordinate column name.
        y_col: Y coordinate column name.
        z_col: Z coordinate column name.
        gene_col: Gene/feature column name.
        cell_id_col: Seed assignment column name.
        transcript_id_col: Optional stable numeric transcript identifier column.
        qv_col: Optional transcript quality-value column.
        samples: Number of MCMC samples.
        burnin_voxel_size: Optional burn-in voxel size (must be >= voxel_size).
        voxel_size: Voxel size for inference.
        voxel_layers: Number of voxel layers in z.
        nuclear_reassignment_prob: ProSeg nuclear reassignment probability.
        diffusion_probability: ProSeg diffusion probability.
        cell_compactness: Optional cell compactness prior.
        expand_initialized_cells: Optional cell expansion in voxels.
        use_cell_initialization: Whether to initialize from seeded assignments.
        prior_seg_reassignment_prob: Optional prior reassignment probability.
        morphology_steps_per_iter: Optional morphology sub-steps per iteration.
        max_transcript_nucleus_distance: Optional transcript distance cutoff.
        cellpose_masks: Optional Cellpose mask array or path.
        cellpose_cellprobs: Optional Cellpose probability array or path.
        cellpose_scale: Optional isotropic mask scale in microns per pixel.
        cellpose_x_transform: Optional affine x tuple (a, b, c).
        cellpose_y_transform: Optional affine y tuple (a, b, c).
        diffusion_sigma_far: Optional far-field diffusion sigma.
        num_threads: Number of CPU threads for ProSeg.
        overwrite: Whether to pass `--overwrite`.
        logger: Optional logger instance.

    Returns:
        Path to the produced output zarr.
    """
    log = logger or logging.getLogger(__name__)
    output_path = Path(output_path)
    proseg_binary = Path(proseg_binary)

    if int(samples) < 1:
        raise ValueError(f"samples must be >= 1 (got {samples})")
    if int(voxel_layers) < 1:
        raise ValueError(f"voxel_layers must be >= 1 (got {voxel_layers})")
    if int(voxel_layers) > 256:
        raise ValueError(f"voxel_layers exceeds ProSeg maximum of 256 ({voxel_layers})")
    if burnin_voxel_size is not None and burnin_voxel_size < voxel_size:
        raise ValueError(
            "burnin_voxel_size must be >= voxel_size "
            f"(got burnin_voxel_size={burnin_voxel_size}, voxel_size={voxel_size})"
        )

    required_columns = [x_col, y_col, z_col, gene_col, cell_id_col]
    if transcript_id_col is not None:
        required_columns.append(transcript_id_col)
    if qv_col is not None:
        required_columns.append(qv_col)
    transcript_csv_path: Path | None = None

    if isinstance(transcripts_df, str | Path):
        transcript_csv_path = Path(transcripts_df)
        if not transcript_csv_path.exists():
            raise FileNotFoundError(f"Transcript CSV not found: {transcript_csv_path}")
        _validate_columns_from_csv(transcript_csv_path, required_columns)
    else:
        missing_columns = [
            col for col in required_columns if col not in transcripts_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in transcripts_df: {missing_columns}"
            )

    proseg_version = _check_proseg_available(proseg_binary)
    use_zarr_output = _check_proseg_version_gte(proseg_version, "3.0.0")
    log.info("Using ProSeg version: %s", proseg_version)

    os.environ["RAYON_NUM_THREADS"] = str(int(num_threads))
    log.info("Using %s CPU threads for ProSeg", num_threads)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        if transcript_csv_path is None:
            if not isinstance(transcripts_df, pd.DataFrame):
                raise TypeError(
                    "transcripts_df must be a DataFrame when not using CSV input."
                )
            transcript_csv_path = tmp_path / "transcripts.csv"
            transcript_view = transcripts_df.loc[:, required_columns]
            transcript_view.to_csv(transcript_csv_path, index=False)
            log.info(
                "Saved %s transcripts to %s", len(transcript_view), transcript_csv_path
            )
        else:
            log.info("Using transcript CSV: %s", transcript_csv_path)

        cellpose_mask_path = _to_optional_path_array(
            value=cellpose_masks,
            tmp_path=tmp_path,
            filename="cellpose_masks.npy",
            dtype=np.uint32,
        )
        cellpose_cellprobs_path = _to_optional_path_array(
            value=cellpose_cellprobs,
            tmp_path=tmp_path,
            filename="cellpose_cellprobs.npy",
            dtype=np.float32,
        )

        if cellpose_mask_path is not None:
            has_scale = cellpose_scale is not None
            has_affine = (
                cellpose_x_transform is not None and cellpose_y_transform is not None
            )
            if not (has_scale or has_affine):
                raise ValueError(
                    "Cellpose masks were supplied but no transform was provided. "
                    "Provide cellpose_scale or both cellpose_x_transform/"
                    "cellpose_y_transform."
                )

        cmd = [
            str(proseg_binary),
            str(transcript_csv_path),
            "-x",
            x_col,
            "-y",
            y_col,
            "-z",
            z_col,
            "--gene-column",
            gene_col,
            "--cell-id-column",
            cell_id_col,
            "--cell-id-unassigned",
            "0",
            "--samples",
            str(int(samples)),
            "--voxel-size",
            str(float(voxel_size)),
            "--voxel-layers",
            str(int(voxel_layers)),
            "--nuclear-reassignment-prob",
            str(float(nuclear_reassignment_prob)),
            "--diffusion-probability",
            str(float(diffusion_probability)),
        ]

        if int(samples) < 100:
            # ProSeg's default recorded-samples=100 can fail for small sample counts.
            cmd.extend(["--recorded-samples", str(int(samples))])
        if transcript_id_col is not None:
            cmd.extend(["--transcript-id-column", transcript_id_col])
        if qv_col is not None:
            cmd.extend(["--qv-column", qv_col])
        if cell_compactness is not None:
            cmd.extend(["--cell-compactness", str(float(cell_compactness))])
        if expand_initialized_cells is not None:
            cmd.extend(
                ["--expand-initialized-cells", str(int(expand_initialized_cells))]
            )
        if use_cell_initialization:
            cmd.append("--use-cell-initialization")
        if prior_seg_reassignment_prob is not None:
            cmd.extend(
                [
                    "--prior-seg-reassignment-prob",
                    str(float(prior_seg_reassignment_prob)),
                ]
            )
        if morphology_steps_per_iter is not None:
            cmd.extend(
                ["--morphology-steps-per-iter", str(int(morphology_steps_per_iter))]
            )
        if max_transcript_nucleus_distance is not None:
            cmd.extend(
                [
                    "--max-transcript-nucleus-distance",
                    str(float(max_transcript_nucleus_distance)),
                ]
            )
        if burnin_voxel_size is not None:
            cmd.extend(["--burnin-voxel-size", str(float(burnin_voxel_size))])
        if diffusion_sigma_far is not None:
            cmd.extend(["--diffusion-sigma-far", str(float(diffusion_sigma_far))])
        if cellpose_mask_path is not None:
            cmd.extend(["--cellpose-masks", str(cellpose_mask_path)])
        if cellpose_cellprobs_path is not None:
            cmd.extend(["--cellpose-cellprobs", str(cellpose_cellprobs_path)])

        if cellpose_x_transform is not None and cellpose_y_transform is not None:
            x_transform = cellpose_x_transform
            y_transform = cellpose_y_transform
            if len(x_transform) != 3 or len(y_transform) != 3:
                raise ValueError(
                    "cellpose_x_transform and cellpose_y_transform must be 3-tuples"
                )
            cmd.extend(["--cellpose-x-transform"] + [str(x) for x in x_transform])
            cmd.extend(["--cellpose-y-transform"] + [str(y) for y in y_transform])
        elif cellpose_scale is not None:
            cmd.extend(["--cellpose-scale", str(float(cellpose_scale))])

        if use_zarr_output:
            cmd.extend(["--output-spatialdata", str(output_path)])
        if overwrite:
            cmd.append("--overwrite")

        log.info("Running ProSeg command: %s", " ".join(cmd))
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        streamed_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            streamed_lines.append(line)
            log.info("[proseg] %s", line)
            if progress_callback is not None:
                m = _PROSEG_ITER_RE.search(line)
                if m:
                    done = int(m.group(1))
                    total = int(m.group(2)) if m.group(2) else proseg_samples
                    pct = round(100 * done / total, 1) if total else None
                    progress_callback(
                        "proseg_sampling",
                        samples_done=done,
                        samples_total=total,
                        pct=pct,
                    )

        return_code = process.wait()
        if return_code != 0:
            tail = "\n".join(streamed_lines[-100:])
            if tail:
                log.error("Last ProSeg output lines:\n%s", tail)
            raise RuntimeError(
                f"ProSeg execution failed with return code {return_code}"
            )

    if not output_path.exists():
        raise RuntimeError(f"ProSeg output not found at {output_path}")

    log.info("ProSeg completed successfully: %s", output_path)
    return output_path
