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
| `--force_spatialdata_build` | Force rebuilding the SpatialData zarr even when a cached one exists. Defaults to `false`. |
| `--enable_alignment` | Run optional Spateo alignment and alignment QC before comparison. Defaults to `false`. |
| `--start_stage` / `--stop_stage` | Run a contiguous stage range. Defaults to the full pipeline. |
| `--only_stage` | Convenience alias for setting `start_stage` and `stop_stage` to the same stage. |

Every other parameter has a default in
[workflows/nextflow.config](../workflows/nextflow.config). See
[Configuration](configuration.md) for the full list.

To use alignment, install Spateo and then restore modern AnnData for
SpatialData compatibility:

```bash
pip install spateo-release==1.1.1
pip install "anndata>=0.12.10"
```

## Resuming a run

Nextflow caches every successful process by hash. To pick up after a failure
or mid-pipeline interruption, add `-resume`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --proseg_binary /path/to/proseg \
    -resume
```

Completed stages are skipped; only the failed stage and its downstreams re-run.

## Forcing a full rebuild

- `--force_spatialdata_build true` — ignore cached SpatialData zarrs and
  rebuild from raw exports. Requires `merscope_dir` / `xenium_dir` to be set
  in the samplesheet.
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
and `visualize`. Alignment stages are only active when `--enable_alignment true`
is set.

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
Starting at `compare` or `visualize` with `--enable_alignment false` reads
`${outdir}/${pair_id}/{merscope,xenium}/latest/latest_spatialdata.zarr`.
With `--enable_alignment true`, `ALIGN` updates
`${outdir}/${pair_id}/merscope/latest/latest_spatialdata.zarr` in place with
alignment metadata and `*_aligned_nonrigid` vector elements. Later stages read
that updated MERSCOPE zarr and keep using
`${outdir}/${pair_id}/xenium/latest/latest_spatialdata.zarr` as the fixed
reference.

## Running on a cluster

The default executor is local (`executor = 'local'` in
[nextflow.config:36-40](../workflows/nextflow.config#L36-L40)) with a hard
ceiling of 75 CPUs and 600 GB memory. To target an HPC scheduler, add a
profile or edit the `executor` block — see the
[Nextflow executor docs](https://www.nextflow.io/docs/latest/executor.html).
Per-process CPU and memory requests are already declared in the `process {}`
block and will carry over to most schedulers unchanged.
