# Running the pipeline

Once you have a conda environment, a samplesheet, and, for full runs, a ProSeg
binary (see [Getting started](getting-started.md) and
[Samplesheet format](samplesheet.md)), the pipeline is driven by a single
`nextflow run` invocation.

## Basic invocation

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --proseg_binary /path/to/proseg
```

Required parameter:

| Flag | Description |
|------|-------------|
| `--samplesheet` | Path to your CSV. |

Common optional parameters:

| Flag | Description |
|------|-------------|
| `--outdir` | Where all outputs are published. Defaults to `./results`. |
| `--proseg_binary` | Absolute path to the ProSeg binary. Required when `SEGMENT` runs. |
| `--analysis_mode` | `paired` (default), `merscope`, or `xenium`. Controls which platform columns are required and which stages are active. |
| `--force_spatialdata_build` | Force rebuilding the SpatialData zarr even when a cached one exists. Defaults to `false`. |
| `--enable_alignment` | Run optional Spateo alignment and alignment QC before comparison. Paired mode only. Defaults to `false`. |
| `--start_stage` / `--stop_stage` | Run a contiguous stage range. Defaults to the full pipeline. |
| `--only_stage` | Convenience alias for setting `start_stage` and `stop_stage` to the same stage. |

Every other parameter has a default in
[workflows/nextflow.config](../workflows/nextflow.config). See
[Configuration](configuration.md) for the full list.

To use alignment, pass `--enable_alignment true`. Nextflow runs `ALIGN` in
`environment.alignment.yml`, bootstraps Spateo/Dynamo from pinned Git refs if
the shimmed import check fails, then restores modern AnnData for SpatialData
compatibility. Other stages continue to use the regular `environment.yml`.
On single-GPU systems, `ALIGN` and RAPIDS-backed `CLUSTERING_SQUIDPY` are
serialized by default (`alignment_max_forks = 1` and
`clustering_squidpy_max_forks = 1`) so multiple jobs do not exhaust VRAM.

## Analysis mode

`--analysis_mode paired` is the default and expects both MERSCOPE and Xenium
inputs for every samplesheet row. It enables the paired-only stages:
`align`, `align_qc`, and `compare`.

Use single-platform mode when a row has only one dataset:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/xenium_only.csv \
    --analysis_mode xenium \
    --outdir ./results \
    --proseg_binary /path/to/proseg
```

In `--analysis_mode merscope` or `--analysis_mode xenium`, the workflow runs
`build_spatialdata → segment → enrich → qc → visualize → clustering_squidpy`
for the selected platform. Visualization writes single-dataset alternatives
for paired plots, including gene-abundance, one-platform transcript overview,
and one-platform sanity crop outputs. `mapmycells` remains available after
clustering. Alignment and comparison are rejected in single-platform mode
because they require both datasets.

## Resuming a run

Nextflow caches every successful process by hash. To pick up after a failure
or mid-pipeline interruption, add `-resume`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --proseg_binary /path/to/proseg \
    --enable_alignment true \
    -resume
```

Completed stages are skipped; only the failed stage and its downstreams re-run.

## Forcing a full rebuild

- `--force_spatialdata_build true` — ignore cached SpatialData zarrs and
  rebuild from raw exports. Requires raw-dir columns for the active
  platform(s) to be set in the samplesheet.
- `nextflow run ...` without `-resume` — blow away Nextflow's cache and
  re-run every process. The `work/` directory will grow; clean it with
  `rm -rf work/` when you're done.

## Overriding parameters on the command line

Any `nextflow.config` parameter can be overridden with `--<name> <value>`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --proseg_binary /path/to/proseg \
    --cellpose_gpu false \
    --cellpose_diameter 30.0 \
    --max_ram_gb 256
```

See [Configuration → Nextflow parameters](configuration.md#nextflow-parameters)
for the complete list.

## Monitoring a run

Three reports are written to `${outdir}/nextflow/` automatically
(see [workflows/nextflow.config](../workflows/nextflow.config)):

| File | What it shows |
|------|---------------|
| `report.html` | HTML summary of each process's status, CPU, memory, and duration. |
| `timeline.html` | Gantt chart of process execution. |
| `trace.tsv` | Tab-separated per-process metrics incl. peak RSS, realtime, workdir. |

Nextflow also writes live logs to `.nextflow.log` and a rolling history in
`.nextflow.log.1`, `.nextflow.log.2`, ...; tail the most recent to debug
failures.

## Running a subset of stages

Use `--start_stage`, `--stop_stage`, or `--only_stage` when you want to run a
piece of the pipeline without relying on Nextflow's `-resume` cache lineage.
Skipped upstream stages are not invoked. Instead, the workflow checks for their
published outputs under `--outdir` and fails early if an expected file is
missing.

Accepted stages are:

`build_spatialdata`, `segment`, `enrich`, `qc`, `align`, `align_qc`, `compare`,
`visualize`, `clustering_squidpy`, and `mapmycells`. Alignment stages are only
active when `--enable_alignment true` is set, and `align`, `align_qc`, and
`compare` are active only in `--analysis_mode paired`.

Run one stage:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --enable_alignment false \
    --only_stage visualize
```

Run from a stage through the end:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --enable_alignment false \
    --start_stage qc
```

Run a bounded range:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --enable_alignment false \
    --start_stage qc \
    --stop_stage compare
```

Starting at `segment` still needs `--proseg_binary`; starting later does not.
Starting at `compare`, `visualize`, or `clustering_squidpy` with
`--enable_alignment false` reads
`${outdir}/${pair_id}/{merscope,xenium}/latest/latest_spatialdata.zarr`.
In single-platform mode, starting at `visualize` or `clustering_squidpy` reads
only `${outdir}/${pair_id}/<selected-platform>/latest/latest_spatialdata.zarr`.
With `--enable_alignment true`, `ALIGN` updates
`${outdir}/${pair_id}/merscope/latest/latest_spatialdata.zarr` in place with
alignment metadata and `*_aligned_nonrigid` vector elements. Later stages read
that updated MERSCOPE zarr and keep using
`${outdir}/${pair_id}/xenium/latest/latest_spatialdata.zarr` as the fixed
reference.

`mapmycells` is downstream of `clustering_squidpy`. By default it runs both the
cached whole-brain reference and a configurable Allen WHB region reference
(`mapmycells_reference_mode=both`). Run through it with `--stop_stage mapmycells`,
or rerun only that stage with `--only_stage mapmycells` after clustering outputs
already exist. Whole-brain-only runs require `--mapmycells_marker_lookup_path`
and `--mapmycells_precomputed_stats_path`; region runs require
`--mapmycells_region_labels`.

If the mapper outputs already exist and only the final annotated H5AD/plots need
to be regenerated, add `--mapmycells_plots_only true` to `--only_stage
mapmycells`. The process copies the previously published
`${outdir}/<pair_id>/mapmycells/mapmycells_out/` directory into the work
directory, skips MapMyCells execution, and rewrites the plots from the existing
CSV/extended JSON outputs.

## Running on a cluster

The default executor is local (`executor = 'local'` in
[nextflow.config:36-40](../workflows/nextflow.config#L36-L40)) with a hard
ceiling of 75 CPUs and 600 GB memory. To target an HPC scheduler, add a
profile or edit the `executor` block — see the
[Nextflow executor docs](https://www.nextflow.io/docs/latest/executor.html).
Per-process CPU and memory requests are already declared in the `process {}`
block and will carry over to most schedulers unchanged.
