# Running the pipeline

Once you have a conda environment, a ProSeg binary, and a samplesheet (see
[Getting started](getting-started.md) and [Samplesheet format](samplesheet.md)),
the pipeline is driven by a single `nextflow run` invocation.

## Basic invocation

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --proseg_binary /path/to/proseg
```

Required parameters:

| Flag | Description |
|------|-------------|
| `--samplesheet` | Path to your CSV. |
| `--proseg_binary` | Absolute path to the ProSeg binary. |

Optional parameter everyone hits sooner or later:

| Flag | Description |
|------|-------------|
| `--outdir` | Where all outputs are published. Defaults to `./results`. |
| `--force_spatialdata_build` | Force rebuilding the SpatialData zarr even when a cached one exists. Defaults to `false`. |
| `--enable_alignment` | Run optional Spateo alignment and alignment QC before comparison. Defaults to `false`. |

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
(see [workflows/nextflow.config:75-96](../workflows/nextflow.config#L75-L96)):

| File | What it shows |
|------|---------------|
| `report.html` | HTML summary of each process's status, CPU, memory, and duration. |
| `timeline.html` | Gantt chart of process execution. |
| `trace.tsv` | Tab-separated per-process metrics incl. peak RSS, realtime, workdir. |

Nextflow also writes live logs to `.nextflow.log` and a rolling history in
`.nextflow.log.1`, `.nextflow.log.2`, ...; tail the most recent to debug
failures.

## Running a subset of stages

The workflow in [workflows/main.nf](../workflows/main.nf) is a single DSL2
`workflow {}` block, so there is no built-in `--stages` flag. Two options:

1. **Resume.** If the pipeline has already produced the stage outputs for some
   pairs, `-resume` will short-circuit them.
2. **Invoke the CLI directly.** Every stage is also a plain Python entry
   point — write your own JSON config and run `merxen <subcommand> --config ...`.
   See [CLI reference](cli.md) for argument shapes.

## Running on a cluster

The default executor is local (`executor = 'local'` in
[nextflow.config:36-40](../workflows/nextflow.config#L36-L40)) with a hard
ceiling of 75 CPUs and 600 GB memory. To target an HPC scheduler, add a
profile or edit the `executor` block — see the
[Nextflow executor docs](https://www.nextflow.io/docs/latest/executor.html).
Per-process CPU and memory requests are already declared in the `process {}`
block and will carry over to most schedulers unchanged.
