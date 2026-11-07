# Spatial Gene Analysis

Runs two complementary analyses after visualization and before Squidpy
clustering:

1. cell-centroid Moran's I and Geary's C; and
2. tissue-level marked point-pattern analysis on the native location of every
   transcript, independent of vendor or ProSeg transcript assignments.

## What it does

### Cell-level autocorrelation

1. Open the latest SpatialData zarr and copy the selected AnnData table.
2. Populate `.obsm["spatial"]` from the matching shape centroids, using aligned
   MERSCOPE coordinates when alignment is enabled.
3. Optionally remove blank/negative/control-like genes.
4. Filter low-support genes, normalize cell totals, and log-transform
   expression.
5. Build a Squidpy spatial nearest-neighbor graph from generic xy coordinates.
6. Compute Moran's I and Geary's C for every retained gene.
7. Write a full metrics table, top/bottom ranking table, a distribution plot,
   and individual spatial expression plots for ranked genes.

Moran's I and Geary's C therefore use one expression value and one centroid
per selected cell; they do not operate directly on transcript coordinates.

### Transcript-level tissue patterns

1. Stream native transcript `x`, `y`, and gene labels in 500,000-row chunks.
   Existing transcript-to-cell assignments are never read.
2. Restrict the point process to the tissue polygon bounded by the pial and
   tissue-edge annotations, subtracting any exclusion polygons.
3. Classify every retained transcript by geometry. A point inside a
   `cellpose_nuclei` polygon is **nuclear**; otherwise a point inside the active
   cell segmentation is **cytoplasmic**; otherwise it is **extracellular**.
   Nuclear membership takes precedence when masks overlap.
4. Measure distance to the nearest cell and nuclear boundary. Distances are
   positive inside the relevant mask and negative outside it. For each signed
   distance bin, compare the gene fraction with the all-transcript background
   using log2 odds, hypergeometric tail probabilities, and Benjamini-Hochberg
   FDR.
5. Count same-gene transcript pairs in 0–2, 2–5, 5–20, 20–50, and 50–200 µm
   annuli. The reported pair-correlation enrichment is observed pair count
   divided by the random-label mean on the observed transcript support.
6. Evaluate two nested random-label nulls with 100 routine draws:
   - `global`: sample the same number of coordinates without replacement from
     all tissue transcripts;
   - `compartment_stratified`: preserve that gene's nuclear, cytoplasmic, and
     extracellular counts while sampling coordinates within each compartment.

The second null asks whether clustering remains after gross compartment
preference is controlled. It helps distinguish a gene that merely occurs in
cells from one that has residual tissue-scale clustering within the same
compartment. This is a marked random-label pair enrichment rather than a
classical area-normalized unmarked `g(r)` estimate.

Genes with fewer than 100 transcripts skip pair correlation. More abundant
genes are deterministically and uniformly thinned to at most 5,000 points per
gene for pair calculations; full points are still used for compartments and
signed distances. Up to 12 pair-correlation worker threads are used per
sample.

### Engine selection

The transcript-level analysis has two engines. Python picks one per sample by
capability, before the cell-level work starts:

| The layer… | Engine |
|------------|--------|
| has cell and nucleus label elements (ATERA: `cell_labels` or, for `reseg`, `proseg_cell_labels`, with `nucleus_labels`) | label-raster engine (below) |
| has none, and its table has more than 5,000 non-control genes (`merxen.gene_sets.WHOLE_TRANSCRIPTOME_MIN_GENES`, e.g. a Xenium Prime 5K row) | refused with a message: the polygon engine keeps every transcript in memory. Set `spatial_gene_analysis_transcript_analysis_enabled=false` for the row. |
| has none (the 300-gene XENIUM and MERSCOPE layers) | the polygon engine described above, unchanged |

Both engines reduce their transcripts to the same per-gene counts
(`GenePatternCounts`) and share the statistics that turn them into the
summary and signed-distance tables, so a XENIUM or MERSCOPE table is
byte-identical to the one before the label-raster engine existed (a test
compares them with a frozen copy of the former code).

### Label-raster engine (ATERA)

Whole-transcriptome sections hold more than a billion transcripts, so this
engine streams every transcript instead of keeping points:

1. **Tissue support.** When the row names tissue annotations (the GeoJSON
   columns above), they win: the polygon is rasterised on a 10 µm grid and
   transcripts in grid bins its boundary crosses get an exact
   point-in-polygon test. Otherwise the tissue comes from the density of
   every transcript (controls included) on a 10 µm grid: a threshold (Otsu
   on log10 of the positive bins, which separates the off-tissue background
   of a few transcripts per bin from the tissue; capped at 1% of their 99th
   percentile so a window inside the tissue is not split into dense and
   sparse tissue) applied to every bin, a closing with a 50 µm disk, hole
   filling, and
   removal of components smaller than 0.05 mm². The manifest records the
   method, the threshold and the areas; the mask itself is written as
   `<sample_id>_transcript_tissue_mask.npz` and drawn over the density in
   `plots/transcript_patterns/<sample_id>_tissue_support.png`.
2. **Compartments.** Each transcript's pixel in the layer's cell labels and
   the nucleus labels, with the platform's pixel rule (the vendor's float32
   corner rule for ATERA, `PlatformSpec.mask_pixel_convention`): nuclear on a
   nucleus label, else cytoplasmic on a cell label, else extracellular. Label
   rasters cannot overlap, so the overlap counters are 0 by construction.
3. **Signed distances.** For each label raster, a tiled exact Euclidean
   distance transform (`scipy.ndimage.distance_transform_edt`, 4,096-pixel
   tiles) of the pixels that touch a label change (8-connected), with a halo
   just above the largest distance a statistic needs (200 µm by default, 944
   pixels on ATERA). A pixel's distance is the distance from its centre to
   the nearest such pixel centre plus half a pixel, so it is within one pixel
   of the point-to-boundary distance of the polygon engine (a test checks
   this on pixel-aligned shapes). Distances are stored as `uint16`
   (0.01 µm steps) in memory-mapped files in task scratch (about 5 GB per
   raster on a full ATERA section) and removed at the end; a distance beyond
   the halo is reported as beyond the outermost bin edge.
4. **One streaming pass** adds every finite, in-tissue, non-control
   transcript (controls by `is_gene`, else by the control-name rules of
   `merxen.control_features`) to per-gene compartment counts, signed-distance
   bin counts, 0.05 µm distance histograms for the medians, and the
   pericellular and membrane-proximal counters. Nothing is thinned:
   `n_transcripts_used` in the manifest is every such transcript.
5. **Pair correlation is not computed** on this engine (AT-8 decision; an
   all-transcript method is a follow-up). The summary's `paircorr_status`
   column says `not_computed_label_raster_engine`, the pair-correlation table
   is empty, and the manifest records it under `transcript_engine.paircorr`.
6. **Diagnostics** are rasterised over every transcript of the selected
   genes, never scattered or subsampled: the gene's in-tissue density on the
   10 µm grid with the tissue outline; its densest 250 µm window (chosen from
   that raster) at 1 µm as a three-colour nuclear / cytoplasmic /
   extracellular composite over the cell and nucleus label edges; the
   signed-distance profile; and the compartment fractions against all
   transcripts.

The manifest's `transcript_counts` adds, on this engine, the vendor-layer
consistency counters: `n_gene_transcripts_on_cell_label` (gene transcripts
on any cell label, in or out of the tissue: on the vendor layer this equals
the gene total of `table_original`) and its in-tissue part, plus the
transcripts outside the label grid. `transcript_engine` records the pixel
rule, the distance encoding, tile counts, timings and the scratch size.

## Nextflow process

[`SPATIAL_GENE_ANALYSIS`](../../workflows/modules/spatial_gene_analysis.nf) —
one instance per `pair_id` and analysis segmentation branch.

- **Input:** `tuple(pair_id, segmentation, samples_json)`, where
  `samples_json` has one or two `{sample_id, platform, zarr_path, table_key,
  shape_key}` records plus the nuclei key and tissue annotation paths.
- **CLI:** `merxen spatial-gene-analysis --config spatial_gene_analysis_config.json`.
- **Output:** `tuple(pair_id, segmentation, spatial_gene_analysis_out/)`.
- **publishDir:** `${outdir}/${pair_id}/${segmentation}/spatial_gene_analysis/`
  (copy mode).

When `VISUALIZE` is active, this stage waits for visualization completion. In a
default run through `clustering_squidpy`, clustering waits for
`SPATIAL_GENE_ANALYSIS`.

Set `spatial_gene_analysis_enabled=false` on an individual samplesheet row to
skip this entire stage while allowing that row to continue through clustering
and MapMyCells. Set
`spatial_gene_analysis_transcript_analysis_enabled=false` instead to retain the
cell-level autocorrelation outputs but skip the annotation-dependent
transcript-level analysis.

ATERA rows run the cell-level autocorrelation on every retained gene of the
layer's table and plot only the top-N ranked genes. Their transcript-level
analysis runs on the [label-raster engine](#label-raster-engine-atera), on
every layer the row analyses (`original_seg`, and `reseg` for
`native_proseg` rows), without tissue annotations.

The process requests 12 CPUs and 120 GB. Under the implicit `standard`
(Dwight) profile, `spatial_gene_analysis_max_forks = 4`, so four samples may
run at once, using at most 480/640 GB (75%) and 48/72 configured CPUs. Those
capacity and concurrency values are Dwight host settings; another execution
profile must supply limits appropriate to its own host.

## Tissue annotation requirement

On XENIUM and MERSCOPE rows, transcript analysis requires a bounded
native-coordinate tissue area for each active platform (ATERA rows do not:
they fall back to the density-derived tissue, and preflight only checks the
annotation files a row names). Supply either a combined role-labelled GeoJSON containing
`pia` and `tissue_edge` features, or separate pial and tissue-edge GeoJSONs.
The same platform-specific and shared samplesheet column aliases used by
cortical depth are accepted. White-matter lines are not used as the outer
tissue boundary; optional exclusion polygons are removed from the support.
Preflight checks validate these paths before expensive analysis begins.
They are not required when the row's effective
`spatial_gene_analysis_transcript_analysis_enabled` value is `false`.

## Python entry points

| Function | File |
|----------|------|
| CLI `spatial_gene_analysis_command` | [cli/run_spatial_gene_analysis.py](../../src/merxen/cli/run_spatial_gene_analysis.py) |
| `run_spatial_gene_analysis` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `compute_spatial_autocorrelation` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `ranked_spatial_autocorr_genes` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `run_transcript_pattern_analysis` | [analysis/transcript_spatial_patterns.py](../../src/merxen/analysis/transcript_spatial_patterns.py) |
| `compute_signed_distance_enrichment` | [analysis/transcript_spatial_patterns.py](../../src/merxen/analysis/transcript_spatial_patterns.py) |
| `compute_multiscale_pair_correlation` | [analysis/transcript_spatial_patterns.py](../../src/merxen/analysis/transcript_spatial_patterns.py) |
| `select_transcript_engine` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `run_raster_transcript_pattern_analysis` | [analysis/transcript_patterns_raster.py](../../src/merxen/analysis/transcript_patterns_raster.py) |
| `density_tissue_mask` | [analysis/transcript_patterns_raster.py](../../src/merxen/analysis/transcript_patterns_raster.py) |

## Config schema

`SpatialGeneAnalysisConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `spatial_gene_analysis_out/` is populated. |
| `samples` | Sample configs including table/cell shape keys, `cellpose_nuclei`, and pial/tissue-edge annotation paths. |
| `drop_control_features` | Remove blank/negative/control-like genes before analysis. |
| `min_counts` / `min_cells` | Optional cell and gene filters before normalization. |
| `normalize_target_sum` | Optional `scanpy.pp.normalize_total` target sum. `null` uses Scanpy's default. |
| `normalize_exclude_highly_expressed` / `normalize_max_fraction` | Optional Scanpy size-factor controls for very highly expressed genes. |
| `n_neighbors` | Number of spatial neighbors used by Squidpy. |
| `top_n` | Number of top and bottom genes retained per metric. |
| `spatial_point_size` | Point size for ranked gene spatial expression plots. |
| `figure_dpi` | PNG output DPI. |
| `transcript_analysis_enabled` | Enable assignment-independent transcript point analysis. Default `true`. |
| `transcript_min_count` / `paircorr_min_count` | Minimum support for distance/ranking and pair analyses. Defaults 50/100. |
| `signed_distance_edges_um` | Signed cell/nuclear boundary distance bin edges. |
| `paircorr_distance_edges_um` | Pair-distance annulus edges. Default `[0, 2, 5, 20, 50, 200]`. |
| `paircorr_max_transcripts_per_gene` | Deterministic per-gene cap. Default 5,000. |
| `paircorr_permutations` / `paircorr_seed` | Random-label draws and reproducibility seed. Defaults 100/0. |
| `paircorr_n_jobs` | Worker threads; Nextflow supplies `task.cpus` (12). |
| `pericellular_distance_um` / `membrane_distance_um` | Convenience proximity summaries. Defaults 5/2 µm. |
| `transcript_diagnostic_*` | Representative gene plot count, cap, local window size, and point cap. |

The label-raster engine's own parameters (tissue grid 10 µm, smoothing sigma
0 (off), closing radius 50 µm, smallest kept component 50,000 µm², threshold cap
0.01 of the 99th percentile, distance tile 4,096 px, median bin 0.05 µm, fine
diagnostic bin 1 µm, scratch directory, worker threads from
`DASK_NUM_WORKERS`) are `RasterEngineSettings` in
[analysis/transcript_patterns_raster.py](../../src/merxen/analysis/transcript_patterns_raster.py),
not config fields: the Nextflow JSON never sets them, and `merxen/config.py`
is part of the map_first hierarchy fingerprint, so a field there would re-run
every row's annotation `COMPUTE_CPU` task.

## Outputs

Written under `spatial_gene_analysis_out/<platform>/`:

Each listed `.png` plot is also written as a same-stem `.pdf`.

| Kind | File | Contents |
|------|------|----------|
| Metrics table | `<sample_id>_spatial_gene_autocorrelation.csv` | One row per retained gene with Moran's I, Geary's C, and available p-values. |
| Ranking table | `<sample_id>_spatial_gene_autocorrelation_rankings.csv` | Top and bottom `top_n` genes for both Moran's I and Geary's C. |
| Distribution plot | `plots/distributions/<sample_id>_spatial_autocorrelation_distribution.png` | Side-by-side histograms for Moran's I and Geary's C. |
| Spatial gene plots | `plots/spatial_genes/<metric>/<top_or_bottom>/*.png` | One plot per ranked gene, colored by log-normalized expression. |
| Transcript summary | `<sample_id>_transcript_spatial_patterns.csv` | Per-gene compartment counts/enrichment, signed-distance summaries, pair-band summaries, overlap QC, and conservative pattern label. |
| Signed-distance table | `<sample_id>_transcript_signed_distance.parquet` | Long table for every gene × boundary × signed-distance bin, including fractions, log2 odds, p-values, and BH FDR. |
| Pair-correlation table | `<sample_id>_transcript_pair_correlation.parquet` | Long table for every eligible gene × null model × distance band, including thinning, null envelope, enrichment, empirical p-values, and BH FDR. Empty (same columns) on the label-raster engine. |
| Transcript rankings | `<sample_id>_transcript_spatial_pattern_rankings.csv` | Separate rankings for compartments, proximity, signed-distance bands, and both pair nulls. |
| Transcript diagnostics | `plots/transcript_patterns/*.png` | Polygon engine: tissue map, local cell/nuclear outline view, signed-distance profile, and both pair-null profiles for representative genes. Label-raster engine: density rasters over every transcript (see above). |
| Tissue mask (label-raster engine) | `<sample_id>_transcript_tissue_mask.npz` | `inside` (bool grid, row = y), `origin_um`, `bin_um`. |
| Tissue support plot (label-raster engine) | `plots/transcript_patterns/<sample_id>_tissue_support.png` | The tissue outline over the all-transcript density. |
| Manifest | `<sample_id>_spatial_gene_analysis_manifest.json` | Parameters, retained counts, output paths and, on the label-raster engine, `transcript_engine` (method, tissue support, pair-correlation status, timings). |
