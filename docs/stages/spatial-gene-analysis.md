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

The default process request is 12 CPUs and 120 GB. Four samples may run at
once, using at most 480/640 GB (75%) and 48/72 configured CPUs.

## Tissue annotation requirement

Transcript analysis requires a bounded native-coordinate tissue area for each
active platform. Supply either a combined role-labelled GeoJSON containing
`pia` and `tissue_edge` features, or separate pial and tissue-edge GeoJSONs.
The same platform-specific and shared samplesheet column aliases used by
cortical depth are accepted. White-matter lines are not used as the outer
tissue boundary; optional exclusion polygons are removed from the support.
Preflight checks validate these paths before expensive analysis begins.

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
| Pair-correlation table | `<sample_id>_transcript_pair_correlation.parquet` | Long table for every eligible gene × null model × distance band, including thinning, null envelope, enrichment, empirical p-values, and BH FDR. |
| Transcript rankings | `<sample_id>_transcript_spatial_pattern_rankings.csv` | Separate rankings for compartments, proximity, signed-distance bands, and both pair nulls. |
| Transcript diagnostics | `plots/transcript_patterns/*.png` | Tissue map, local cell/nuclear outline view, signed-distance profile, and both pair-null profiles for representative genes. |
| Manifest | `<sample_id>_spatial_gene_analysis_manifest.json` | Parameters, retained counts, and output paths. |
