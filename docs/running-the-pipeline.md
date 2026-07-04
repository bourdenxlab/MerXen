# Running the pipeline

Once you have a conda environment and a samplesheet (see
[Getting started](getting-started.md) and [Samplesheet format](samplesheet.md)),
the pipeline is driven by a single `nextflow run` invocation. ProSeg is resolved
automatically from configured search paths and installed with Cargo if needed.

## Basic invocation

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results
```

Required parameter:

| Flag | Description |
|------|-------------|
| `--samplesheet` | Path to your CSV. |

Common optional parameters:

| Flag | Description |
|------|-------------|
| `--outdir` | Where all outputs are published. Defaults to `./results`. |
| `--analysis_mode` | `paired` (default), `merscope`, or `xenium`. Controls which platform columns are required and which stages are active. |
| `--analysis_segmentation` | `both` (default), `reseg`, or `original_seg`. Controls whether downstream analysis runs on resegmented data, original instrument segmentation, or both. |
| `--force_spatialdata_build` | Force rebuilding the SpatialData zarr even when a cached one exists. Defaults to `false`. |
| `--enable_alignment` | Run optional Spateo alignment and alignment QC before comparison. Paired mode only. Defaults to `false`. |
| `--cortical_depth_enabled` | Run cortical-depth coordinate computation before QC. Requires boundary GeoJSON annotations. Defaults to `false`. |
| `--start_stage` / `--stop_stage` | Run a contiguous stage range. Defaults to the full pipeline. |
| `--only_stage` | Convenience alias for setting `start_stage` and `stop_stage` to the same stage. |

The samplesheet may also include `analysis_mode`, `enable_alignment`,
`analysis_segmentation`, `cortical_depth_enabled`, `start_stage`, `stop_stage`,
and `only_stage` columns.
Non-empty row values override these command-line settings for that row only;
blank cells inherit the command-line/config value. Every other parameter has a
default in
[workflows/nextflow.config](../workflows/nextflow.config). See
[Configuration](configuration.md) for the full list.

To use alignment for all paired rows by default, pass `--enable_alignment true`.
To choose per row, add an `enable_alignment` column to the samplesheet and set
paired rows to `true` or `false`; blank cells inherit `--enable_alignment`.
Nextflow runs `ALIGN` in `environment.alignment.yml`, bootstraps Spateo/Dynamo
from pinned Git refs if the shimmed import check fails, then restores modern
AnnData for SpatialData compatibility. Other stages continue to use the regular
`environment.yml`. GPU-heavy `SEGMENT` and `ALIGN` default to one task at a
time. RAPIDS-backed `CLUSTERING_SQUIDPY` allows up to four queued local tasks,
but GPU execution is serialized by the shared workstation GPU lock when it is
enabled.

Before any task inputs are emitted, the workflow runs stage-aware preflight
checks for reference files required by the selected stage range. For example,
`clustering_squidpy` with hierarchical mode checks the broad marker lookup and
Allen taxonomy paths, while `mapmycells` checks whole-brain marker/stat files
only when that module is selected and the requested reference mode needs them.
When `compute_cortical_depth` is selected, preflight checks pial/tissue-edge
annotation GeoJSONs or a combined role-labelled annotation GeoJSON for every
active platform. Gray/white boundaries are optional for pial-only mask/QC
pieces.
Missing references stop the run immediately with the selected stages and paths
that need attention.

## Analysis mode

`--analysis_mode paired` is the default fallback and expects both MERSCOPE and
Xenium inputs for rows that do not override `analysis_mode`. It enables the
paired-only stages for those paired rows:
`align`, `align_qc`, and `compare`.

`align` and `align_qc` are active only for paired rows whose effective
`enable_alignment` value is `true`.

Use single-platform mode when a row has only one dataset:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/xenium_only.csv \
    --analysis_mode xenium \
    --outdir ./results
```

In `analysis_mode=merscope` or `analysis_mode=xenium`, either from the command
line or a row-level samplesheet value, the workflow runs
`build_spatialdata → segment → enrich → mask_image_quantification → qc →
visualize → spatial_gene_analysis → clustering_squidpy` for the selected
platform. If
`cortical_depth_enabled=true`, `compute_cortical_depth` runs after
`clustering_squidpy` so the per-cell cluster annotations are available for its
depth violin plots (depth columns are not consumed by other stages).
Visualization writes single-dataset alternatives for
paired plots, including gene-abundance, one-platform transcript overview, and
one-platform sanity crop outputs. `mapmycells` remains available after
clustering. Alignment and comparison are rejected for single-platform rows
because they require both datasets.

## Analysis segmentation

By default, stages from QC onward run twice per active sample: once on
`reseg` layers (`table_MOSAIK_proseg` / `MOSAIK_proseg`) and once on
`original_seg` layers (`table_original` plus the platform's original cell
boundaries). Restrict the branch set when you only need one result family:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --analysis_segmentation original_seg \
    --outdir ./results
```

The upstream build, segmentation, and enrichment stages remain shared for each
row. The enriched SpatialData zarr contains both segmentation families;
downstream processes receive explicit table and shape keys for the selected
branch. A row-level `analysis_segmentation` value can restrict branches for one
sample while other rows continue to use the global default.

## Resuming a run

Nextflow caches every successful process by hash. To pick up after a failure
or mid-pipeline interruption, add `-resume`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --enable_alignment true \
    -resume
```

Completed stages are skipped; only the failed stage and its downstreams re-run.

## Failure behavior

Task failures use Nextflow's `ignore` error strategy with
`workflow.failOnIgnore = true`. This means a failed task does not terminate the
whole run immediately. Instead, that task emits no outputs, so only channel
branches that depend on those missing outputs stop:

- A failed per-platform stage prevents later stages for that same
  `pair_id`/platform branch.
- A failed paired dependency prevents paired comparison and later paired
  downstream stages for that `pair_id`/segmentation branch.
- Other rows, platforms, or segmentation branches that do not depend on the
  failed task continue running.

The final Nextflow exit status is still non-zero if any task failure was
ignored, so batch schedulers and shell scripts can detect that the run was not
fully clean.

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
    --cellpose_gpu false \
    --cellpose_diameter 30.0 \
    --max_ram_gb 256
```

See [Configuration → Nextflow parameters](configuration.md#nextflow-parameters)
for the complete list.

By default, local GPU-heavy stages share a workstation GPU lock. This prevents
Cellpose segmentation, GPU alignment, and GPU Squidpy clustering from starting
at the same time on GPU 0 and failing with CUDA out-of-memory. Disable it only
when a scheduler or multi-GPU setup is already isolating GPU work.

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

The same fields can be set per row in the samplesheet. A row-level `only_stage`
overrides that row's start/stop values. If a row sets either `start_stage` or
`stop_stage`, the global `--only_stage` fallback is ignored for that row.
For an already-aligned paired row, keep `enable_alignment=true` and start at
`compare`; the workflow will read the published alignment outputs instead of
rerunning `ALIGN` or `ALIGN_QC`.

Accepted stages are:

`build_spatialdata`, `segment`, `enrich`, `mask_image_quantification`,
`qc`, `align`, `align_qc`, `compare`, `visualize`,
`spatial_gene_analysis`, `clustering_squidpy`, `compute_cortical_depth`, and
`mapmycells`.
`compute_cortical_depth` is only active
for rows whose effective `cortical_depth_enabled` value is `true`. Alignment
stages are only active for rows whose effective `enable_alignment` value is
`true`, and `align`, `align_qc`, and `compare` are active only in paired rows.

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

Starting at `segment` runs the ProSeg resolver first; starting later does not
need ProSeg because it reads published segmentation outputs.
Starting at `compare`, `visualize`, `spatial_gene_analysis`, or
`clustering_squidpy` with effective `enable_alignment=false` reads
`${outdir}/${pair_id}/{merscope,xenium}/latest/latest_spatialdata.zarr`.
In single-platform mode, starting at `visualize`, `spatial_gene_analysis`, or
`clustering_squidpy` reads only
`${outdir}/${pair_id}/<selected-platform>/latest/latest_spatialdata.zarr`.
With effective `enable_alignment=true`, `ALIGN` updates
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
`${outdir}/<pair_id>/<analysis_segmentation>/mapmycells/mapmycells_out/` directory into the work
directory, skips MapMyCells execution, and rewrites the plots from the existing
CSV/extended JSON outputs.

## Running on a cluster

The default executor is local (`executor = 'local'` in
[nextflow.config:36-40](../workflows/nextflow.config#L36-L40)) with a hard
ceiling of 72 CPUs and 640 GB memory. To target an HPC scheduler, add a
profile or edit the `executor` block — see the
[Nextflow executor docs](https://www.nextflow.io/docs/latest/executor.html).
Per-process CPU and memory requests are already declared in the `process {}`
block and will carry over to most schedulers unchanged.
