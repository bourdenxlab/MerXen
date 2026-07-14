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
           │ flatMap → row-level active platform inputs
           │ (analysis_mode paired|merscope|xenium)
           ▼
  ┌─────────────────┐
  │ BUILD_SPATIAL-  │   raw MERSCOPE / Xenium export
  │ DATA            │ → source_spatialdata.zarr
  └────────┬────────┘
           ▼
  ┌──────────────────────┐
  │ SEGMENT subworkflow  │
  │ CELLPOSE_SEGMENT GPU │ → mask, transcripts, affine
  │ PROSEG_SEGMENT CPU   │ → proseg_base_latest.zarr
  └──────────┬───────────┘
           ▼
  ┌─────────────────┐
  │ ENRICH          │   per-shape gene assignment tables
  │                 │ → latest_input.zarr (with tables)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ MASK_IMAGE_     │   Cellpose-mask image-channel
  │ QUANTIFICATION  │   min/median/mean/max/IQR table
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │ QC              │   cell-level + transcript-assignment
  │                 │   metrics, histograms, violins
  └────────┬────────┘
           ▼
      (paired on pair_id when both platforms are active)
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
│COMPARE │   │ VISUALIZE │  paired or single-platform
│paired  │   │           │  plots
└────────┘   └───────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ SPATIAL_GENE_     │  per-gene Moran's I,
           │ ANALYSIS          │  Geary's C, spatial gene plots
           └───────────────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ CLUSTERING_       │  Scanpy/Squidpy QC,
           │ SQUIDPY           │  UMAP, Leiden, spatial scatter
           └───────────────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ COMPUTE_          │  optional tissue annotation +
           │ CORTICAL_DEPTH    │  Laplace/equal-area depth
           └───────────────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ DISTANCE_FROM_    │  optional registered object-edge
           │ OBJECT            │  annotation + paired pseudobulk DE
           └───────────────────┘
                  │
                  ▼
           ┌───────────────────┐
           │ MAPMYCELLS        │  local reference-based
           │                   │  cell type assignment
           └───────────────────┘
```

Rows inherit `analysis_mode`, `enable_alignment`, `analysis_segmentation`,
object-distance settings, `start_stage`, `stop_stage`, and `only_stage` from
Nextflow params unless those
columns are set in the samplesheet. For rows with `analysis_mode=paired`, both
platforms traverse `BUILD_SPATIALDATA → SEGMENT → ENRICH →
MASK_IMAGE_QUANTIFICATION → QC` independently and
are rejoined after QC. When the row's effective `cortical_depth_enabled` value is
`true`, `COMPUTE_CORTICAL_DEPTH` runs as a terminal stage after
`CLUSTERING_SQUIDPY` (consuming the clustering-updated zarr) so per-cell cluster
annotations are available for its depth violin plots; its depth columns are not
consumed by the main comparison branch. When the row's effective
`distance_from_object_enabled` value is `true`, `DISTANCE_FROM_OBJECT_ANNOTATE`
runs after cortical depth (when selected) or after clustering and consumes
`cortical_depth_annotation`. Pair-level pseudobulks are then grouped by
platform for `DISTANCE_FROM_OBJECT_COHORT`. If
mask image quantification is disabled or skipped by a stage range, downstream
stages consume the enriched zarr directly. If the row's effective
`enable_alignment` value is `true`, `ALIGN` and `ALIGN_QC` run before
`COMPARE` / `VISUALIZE` / `SPATIAL_GENE_ANALYSIS` /
`CLUSTERING_SQUIDPY`; otherwise the paired stages
consume the quantified/enriched zarrs directly. In `analysis_mode=merscope` or
`analysis_mode=xenium`, only the selected platform traverses those stages, and
paired-only `ALIGN`, `ALIGN_QC`, and `COMPARE` are inactive for that row.
`MAPMYCELLS` consumes the AnnData files written by
`CLUSTERING_SQUIDPY` and is opt-in because it requires local reference files.

## Channel keys and joins

Per-platform stages key on `"${pair_id}|${platform}"` (e.g. `P0001|MERSCOPE`).
Segmentation-specific analysis branches add the segmentation key. Paired-only
stages join MERSCOPE and XENIUM branches by `pair_id` and segmentation, so a
missing output from one platform prunes only the paired downstream branch that
depends on it. For visualization, spatial gene analysis, clustering, and
MapMyCells, the workflow passes a JSON `samples` list so those stages can
handle either one or two platforms.

## Data flow for one row

For a samplesheet row with `pair_id=EXAMPLE01`:

| Step | Nextflow process | CLI | Input | Output |
|------|------------------|-----|-------|--------|
| 1 | `BUILD_SPATIALDATA` × 2 | `merxen build-spatialdata` | raw export folders (or cached zarr) | `source_spatialdata.zarr` per platform |
| 2a | `CELLPOSE_SEGMENT` × 2 | `merxen cellpose-segment` | `source_spatialdata.zarr` | `cellpose_masks_tiled.npy`, stitching stats, seeded transcript CSV, affine JSON |
| 2b | `PROSEG_SEGMENT` × 2 | `merxen proseg-segment` | Cellpose artifacts | durable `latest/latest_spatialdata.zarr` |
| 3 | `ENRICH` × 2 | `merxen enrich` | latest zarr + Cellpose mask | same durable `latest/latest_spatialdata.zarr`, now enriched with per-shape counts tables |
| 4 | `MASK_IMAGE_QUANTIFICATION` × 2 | `merxen mask-image-quantification` | enriched zarr + Cellpose mask | same durable zarr, now with `table_MOSAIK_cellpose_image_quantification` plus sidecars |
| 5 | `QC` × 2 | `merxen qc` | quantified/enriched zarr | `qc_out/` (metrics CSV, plots) |
| 6 | `ALIGN` × 1 | `merxen align` | both platforms' latest analysis-ready zarrs | in-place MERSCOPE aligned elements + transform metadata, when enabled |
| 7 | `ALIGN_QC` × 1 | `merxen alignment-qc` | updated MERSCOPE zarr + original Xenium zarr | `alignment_qc_out/`, when enabled |
| 8 | `COMPARE` × 1 | `merxen compare` | updated MERSCOPE zarr if enabled; otherwise analysis-ready zarrs | `compare_out/` (gene comparison CSVs + metrics JSON) |
| 9 | `VISUALIZE` × 1 | `merxen visualize` | updated MERSCOPE zarr if enabled; otherwise analysis-ready zarrs | `visualize_out/` (PNG plots) |
| 10 | `SPATIAL_GENE_ANALYSIS` × 1 | `merxen spatial-gene-analysis` | same paired zarrs, after visualization in full runs | `spatial_gene_analysis_out/` (Moran's I, Geary's C, rankings, spatial gene plots) |
| 11 | `CLUSTERING_SQUIDPY` × 1 | `merxen clustering-squidpy` | same paired zarrs, after spatial gene analysis in full runs | `clustering_squidpy_out/` plus derived clustered tables in each durable zarr |
| 12 | `COMPUTE_CORTICAL_DEPTH` × platform | `merxen compute-cortical-depth` | durable zarr + boundary GeoJSON annotations | tissue/depth columns and sidecars, when enabled |
| 13a | `DISTANCE_FROM_OBJECT_ANNOTATE` × platform | `merxen distance-from-object` | durable zarr + registered object GeoJSON | table metadata, per-cell sidecars, and pair-level near/far pseudobulks, when enabled |
| 13b | `DISTANCE_FROM_OBJECT_COHORT` × platform | `merxen distance-from-object-cohort` | all pair-level platform pseudobulks | paired PyDESeq2 results for each segmentation branch |
| 14 | `MAPMYCELLS` × 1 | `merxen mapmycells` | clustered `.h5ad` files from `clustering_squidpy_out/` | `mapmycells_out/` (query `.h5ad`, CSV/JSON assignments, annotated `.h5ad`) |

In single-platform mode, platform-local steps run once per row, paired-only
alignment and comparison are skipped, and visualization/spatial-gene/clustering
stages consume a one-sample config for the selected platform.

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
  `-resume` would pick the wrong Nextflow run lineage. The same fields can be
  set per row in the samplesheet.
- **Nextflow work-dir caching.** Resume a run with `nextflow run ... -resume`
  and completed processes will be skipped. `publishDir` modes are set so that
  SpatialData-heavy stages are symlinked rather than copied.
- **Branch-local failures.** Process failures use Nextflow's `ignore` error
  strategy. A failed task emits no outputs, which prevents only dependent
  downstream joins from firing. `workflow.failOnIgnore = true` keeps the final
  run status non-zero when any task was ignored.

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
2. Each process writes the JSON into a heredoc inside its own work directory.
3. The segmentation subworkflow runs separate Cellpose and ProSeg subcommands;
   Nextflow stages their mask, transcript, and affine handoff files.
4. The CLI loads the JSON, validates it against the matching Pydantic model
   from [src/merxen/config.py](../src/merxen/config.py), and calls the
   stage function.

This means **the Pydantic models in `config.py` are the authoritative contract**
between the workflow and the Python code. Changing a config field requires
updating both the model and the Groovy emitter in
[workflows/main.nf](../workflows/main.nf).
