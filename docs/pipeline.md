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
                  │
                  ▼
           ┌───────────────────┐
           │ CLUSTERING_       │  Scanpy/Squidpy QC,
           │ SQUIDPY           │  UMAP, Leiden, spatial scatter
           └───────────────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ MAPMYCELLS        │  local reference-based
           │                   │  cell type assignment
           └───────────────────┘
```

Both platforms traverse `BUILD_SPATIALDATA → SEGMENT → ENRICH → QC`
independently. They are rejoined after QC. If `--enable_alignment true` is set,
`ALIGN` and `ALIGN_QC` run before `COMPARE` / `VISUALIZE` /
`CLUSTERING_SQUIDPY`; otherwise the paired stages consume the enriched zarrs
directly. `MAPMYCELLS` consumes the AnnData files written by
`CLUSTERING_SQUIDPY` and is opt-in because it requires local reference files.

## Channel keys and joins

Per-platform stages key on `"${pair_id}|${platform}"` (e.g. `P0001|MERSCOPE`).
The paired stages key on `pair_id` alone. The workflow filters the QC channel
into MERSCOPE and XENIUM sub-channels and joins them by `pair_id`.

## Data flow for one row

For a samplesheet row with `pair_id=EXAMPLE01`:

| Step | Nextflow process | CLI | Input | Output |
|------|------------------|-----|-------|--------|
| 1 | `BUILD_SPATIALDATA` × 2 | `merxen build-spatialdata` | raw export folders (or cached zarr) | `source_spatialdata.zarr` per platform |
| 2 | `SEGMENT` × 2 | `merxen segment` | `source_spatialdata.zarr` | durable `latest/latest_spatialdata.zarr`, `cellpose_masks_tiled.npy`, `transcripts_for_proseg.csv` |
| 3 | `ENRICH` × 2 | `merxen enrich` | latest zarr + Cellpose mask | same durable `latest/latest_spatialdata.zarr`, now enriched with per-shape counts tables |
| 4 | `QC` × 2 | `merxen qc` | enriched zarr | `qc_out/` (metrics CSV, plots) |
| 5 | `ALIGN` × 1 | `merxen align` | both platforms' enriched zarrs | in-place MERSCOPE aligned elements + transform metadata, when enabled |
| 6 | `ALIGN_QC` × 1 | `merxen alignment-qc` | updated MERSCOPE zarr + original Xenium zarr | `alignment_qc_out/`, when enabled |
| 7 | `COMPARE` × 1 | `merxen compare` | updated MERSCOPE zarr if enabled; otherwise enriched zarrs | `compare_out/` (gene comparison CSVs + metrics JSON) |
| 8 | `VISUALIZE` × 1 | `merxen visualize` | updated MERSCOPE zarr if enabled; otherwise enriched zarrs | `visualize_out/` (PNG plots) |
| 9 | `CLUSTERING_SQUIDPY` × 1 | `merxen clustering-squidpy` | same paired zarrs, after visualization in full runs | `clustering_squidpy_out/` (QC plots, UMAP/spatial plots, `.h5ad`) |
| 10 | `MAPMYCELLS` × 1 | `merxen mapmycells` | clustered `.h5ad` files from `clustering_squidpy_out/` | `mapmycells_out/` (query `.h5ad`, CSV/JSON assignments, annotated `.h5ad`) |

All published artifacts land under
`${params.outdir}/${pair_id}/<stage>/...`. See [Outputs](outputs.md).

## Caching and reuse

Two independent reuse paths and one Nextflow cache:

- **SpatialData reuse.** A samplesheet can point `merscope_spatialdata_path` /
  `xenium_spatialdata_path` at an existing built zarr. `build_spatialdata`
  short-circuits to that artifact unless `--force_spatialdata_build true` is
  passed. Implemented in [src/merxen/io/builders/pipeline.py:14](../src/merxen/io/builders/pipeline.py#L14).
- **Published-output stage starts.** `--start_stage`, `--stop_stage`, and
  `--only_stage` select a contiguous process range without invoking earlier
  stages. When an upstream stage is skipped, the workflow reads the expected
  artifact from `${outdir}` and errors if it is missing. This is useful when
  `-resume` would pick the wrong Nextflow run lineage.
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
