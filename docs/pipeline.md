# Pipeline architecture

MerXen is a Nextflow DSL2 pipeline that drives Python stages through a single
CLI. Each Nextflow process calls one `merxen <subcommand>` with a JSON config
emitted by the workflow, and stages hand zarrs and masks to each other through
channels.

## Stage graph

```
┌───────────────────────┐
│ samplesheet.csv (row) │
└──────────┬────────────┘
           │ flatMap → (MERSCOPE | XENIUM) inputs
           ▼
  ┌─────────────────┐
  │ BUILD_SPATIAL-  │   raw MERSCOPE / Xenium export
  │ DATA            │ → source_spatialdata.zarr
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ SEGMENT         │   Cellpose-SAM tiled segmentation,
  │                 │   then ProSeg transcript-based
  │                 │   refinement with Cellpose as prior
  │                 │ → proseg_base_latest.zarr,
  │                 │   cellpose_masks_tiled.npy
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ ENRICH          │   per-shape gene assignment tables
  │                 │ → latest_input.zarr (with tables)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ QC              │   cell-level + transcript-assignment
  │                 │   metrics, histograms, violins
  └────────┬────────┘
           ▼
      (paired on pair_id)
           │
           ▼
  ┌─────────────────┐
  │ ALIGN           │   optional Spateo MERSCOPE→Xenium
  │ ALIGN_QC        │   registration and QC
  └────────┬────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌───────────┐
│COMPARE │   │ VISUALIZE │  gene scatter, density,
│        │   │           │  sanity overlays
└────────┘   └───────────┘
```

Both platforms traverse `BUILD_SPATIALDATA → SEGMENT → ENRICH → QC`
independently. They are rejoined after QC. If `--enable_alignment true` is set,
`ALIGN` and `ALIGN_QC` run before `COMPARE` / `VISUALIZE`; otherwise the paired
stages consume the enriched zarrs directly.

## Channel keys and joins

Per-platform stages key on `"${pair_id}|${platform}"` (e.g. `P0001|MERSCOPE`).
The paired stages key on `pair_id` alone. The workflow
[workflows/main.nf:280-292](../workflows/main.nf#L280-L292) filters the QC
channel into MERSCOPE and XENIUM sub-channels and joins them by `pair_id`.

## Data flow for one row

For a samplesheet row with `pair_id=EXAMPLE01`:

| Step | Nextflow process | CLI | Input | Output |
|------|------------------|-----|-------|--------|
| 1 | `BUILD_SPATIALDATA` × 2 | `merxen build-spatialdata` | raw export folders (or cached zarr) | `source_spatialdata.zarr` per platform |
| 2 | `SEGMENT` × 2 | `merxen segment` | `source_spatialdata.zarr` | durable `latest/latest_spatialdata.zarr`, `cellpose_masks_tiled.npy`, `transcripts_for_proseg.csv` |
| 3 | `ENRICH` × 2 | `merxen enrich` | latest zarr + Cellpose mask | same durable `latest/latest_spatialdata.zarr`, now enriched with per-shape counts tables |
| 4 | `QC` × 2 | `merxen qc` | enriched zarr | `qc_out/` (metrics CSV, plots) |
| 5 | `ALIGN` × 1 | `merxen align` | both platforms' enriched zarrs | aligned zarrs + transform metadata, when enabled |
| 6 | `ALIGN_QC` × 1 | `merxen alignment-qc` | aligned zarrs | `alignment_qc_out/`, when enabled |
| 7 | `COMPARE` × 1 | `merxen compare` | aligned zarrs if enabled; otherwise enriched zarrs | `compare_out/` (gene comparison CSVs + metrics JSON) |
| 8 | `VISUALIZE` × 1 | `merxen visualize` | aligned zarrs if enabled; otherwise enriched zarrs | `visualize_out/` (PNG plots) |

All published artifacts land under
`${params.outdir}/${pair_id}/<stage>/...`. See [Outputs](outputs.md).

## Caching and reuse

Two independent caching layers:

- **SpatialData reuse.** A samplesheet can point `merscope_spatialdata_path` /
  `xenium_spatialdata_path` at an existing built zarr. `build_spatialdata`
  short-circuits to that artifact unless `--force_spatialdata_build true` is
  passed. Implemented in [src/merxen/io/builders/pipeline.py:14](../src/merxen/io/builders/pipeline.py#L14).
- **Nextflow work-dir caching.** Resume a run with `nextflow run ... -resume`
  and completed processes will be skipped. `publishDir` modes are set so that
  SpatialData-heavy stages are symlinked rather than copied.

## Why Cellpose *and* ProSeg?

Cellpose-SAM runs on DAPI/PolyT/18S images and is authoritative for nuclear
boundaries but ignores transcript density. ProSeg refines those masks using
transcript coordinates, so final cell boundaries reflect where mRNA actually
sits. Cellpose acts as a prior for ProSeg's MCMC sampler. Details in
[Segmentation](stages/segmentation.md).

## How a Nextflow run talks to Python

1. The workflow builds a nested Groovy map for each stage — cellpose params,
   proseg params, memory limits, dataset metadata — then serializes it with
   `JsonOutput.prettyPrint(JsonOutput.toJson(...))`.
2. The process writes the JSON into a heredoc inside its work directory.
3. The process runs `merxen <subcommand> --config <file>.json`.
4. The CLI loads the JSON, validates it against the matching Pydantic model
   from [src/merxen/config.py](../src/merxen/config.py), and calls the
   stage function.

This means **the Pydantic models in `config.py` are the authoritative contract**
between the workflow and the Python code. Changing a config field requires
updating both the model and the Groovy emitter in
[workflows/main.nf](../workflows/main.nf).
