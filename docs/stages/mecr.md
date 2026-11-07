# Mutually exclusive co-expression rate (MECR)

MECR measures unexpected co-detection of genes that are mutually exclusive
between broad cell classes in a single-cell RNA-seq reference. It follows the
metric introduced by Hartman and Satija in
[Comparative analysis of multiplexed in situ gene expression profiling technologies](https://doi.org/10.7554/eLife.96949.1).
A lower value indicates greater molecular-assignment specificity; an elevated
value can reflect off-target signal or overly permissive cell segmentation. The
stage is enabled by default for human runs and runs after QC, independently of
alignment. Mouse runs may opt in; they default off because the complete WMB
reference download is about 151 GB.

## Method

The stage uses the species-matched complete Allen reference: Whole Human Brain
10x v3 (WHB) for human or every raw Whole Mouse Brain 10Xv2, 10Xv3, and 10X
Multiome shard (WMB) for mouse. It joins raw H5AD cell labels to atlas taxonomy
metadata and the same broad-class collapse used by Squidpy clustering. The WHB
default classes are:

- Neurons
- Oligodendrocytes
- Oligodendrocyte precursors
- Astrocytes
- Microglia
- Fibroblasts
- Vascular cells

The WMB defaults are neurons, Astrocytes/Ependymal, Oligodendrocyte lineage,
OEC, Vascular cells, and Immune cells collapsed to the existing Microglia
output class.

Reference preparation is restricted to genes present in the spatial panel,
but each reference cell is normalized using its full-library count before the
panel is selected. Expression is normalized to 10,000 counts per cell and
log-transformed. Python/Scanpy Wilcoxon tests compare each broad class with the
rest. Following the paper, a gene is retained for a class only when it is
detected in strictly more than 25% of that class and strictly less than 1% of
the other retained cells. A gene that qualifies for more than one class is
removed.

The raw WMB matrices contain a small number of cells absent from the published
taxonomy metadata. They cannot be assigned to a broad class, so mouse MECR
skips them and records the count in `mecr_reference_manifest.json`; all
taxonomy-annotated cells are retained.

Every unordered pair of retained genes from different broad classes is then
scored in each spatial cell-count table:

```text
MECR(gene 1, gene 2) = cells detecting both genes / cells detecting either gene
```

The sample-level MECR is the unweighted arithmetic mean of all finite pair
rates. Pairs for which neither gene is detected have an undefined (NaN) rate;
they remain in the audit table and are excluded from the aggregate mean.

## Plots

Reference preparation writes a histogram of MECR across every eligible atlas
marker pair, with mean and median lines for description only. Unlike the
exploratory notebook, no reference-MECR cutoff is used to select pairs.

Each spatial branch writes:

- the complete pair-rate distribution by platform;
- a MERSCOPE-versus-Xenium scatter with an identity line, restricted to the
  exact eligible pairs shared by both platforms;
- one broad-class-pair median-MECR heatmap per platform; and
- barnyard count plots for up to `mecr_barnyard_top_n_pairs` pairs.

Barnyard pairs are selected deterministically from eligible production pairs:
canonical pairs are prioritized when available, followed by pairs with the
highest mean spatial MECR and those detected in the most cells. The selection
CSV records every reason. Display coordinates use natural raw cell counts by
default. XENIUM and MERSCOPE scatterplots may be downsampled to
`mecr_barnyard_max_points` randomly chosen cells; ATERA plots draw every cell
as a 2D histogram of the two counts (one bin per count, at most 120 bins per
axis, logarithmic colour scale of cells per bin), with no subsampling. The MECR
shown in the title is always calculated from every cell. Set
`mecr_barnyard_log1p=true` to opt into the earlier `log1p` display.

## Workflow behaviour

`MECR_REFERENCE` runs once per workflow invocation over the merged panels of
every XENIUM and MERSCOPE row and streams every configured raw matrix in
bounded row chunks. Its output is shared by every selected XENIUM/MERSCOPE
sample and segmentation branch. ATERA rows get their own reference task,
`MECR_REFERENCE_ATERA` (below), so an ATERA row in the samplesheet leaves the
`MECR_REFERENCE` config, and its cached result, unchanged. `MECR` then scores
each branch separately for MERSCOPE, Xenium or Atera with the reference of its
row. Missing WMB inputs are downloaded into the durable
reference cache with four resumable concurrent transfers. Because reference
preparation is the expensive step, keep the Nextflow work directory and use
`-resume` when rerunning the same panel and reference settings.

Enable WMB MECR explicitly with `--species mouse --mecr_enabled true`. The
Dwight profile stores the reference under
`$MERXEN_LEGACY_ABC_ATLAS_DIR` (set in its site config); other profiles should set
`--mecr_reference_cache_dir` to a location with at least 160 GB free.

MECR can be disabled globally with:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --mecr_enabled false
```

It can also be selected alone with `--only_stage mecr`, provided the published
`latest_spatialdata.zarr` inputs already exist. A samplesheet `mecr_enabled`
column overrides the global switch per row.

## Whole-transcriptome panels and ATERA rows

The in-memory reference keeps the panel-restricted matrix of every retained
reference cell (about 3 million WHB cells) and runs the Wilcoxon test on every
panel gene. For an 18,000-gene Atera panel that matrix holds most of WHB's
16 billion non-zeros, so the reference task chooses a mode (`reference_mode`
in its config JSON; Atera plan AT-8):

- `in_memory`: as above.
- `streaming`: two passes over the raw matrices, neither of which keeps the
  panel matrix. Pass 1 counts, for every panel gene, the cells of each broad
  class in which it is detected (a classes x genes table). The detection
  fractions, and so the 25% / 1% marker rule, follow from these counts alone.
  Pass 2 reads the reference again and keeps only the detection candidates
  (the genes that pass both thresholds in at least one class; a few hundred),
  normalised exactly as the in-memory path does. The Wilcoxon test and the
  reference marker-pair MECR use this candidate matrix.
- `auto` (the default): `streaming` when the panel has more than 5,000
  non-control genes (`merxen.gene_sets.WHOLE_TRANSCRIPTOME_MIN_GENES`, the
  shared whole-transcriptome rule, whatever the platform; Xenium Prime 5K
  panels count), otherwise `in_memory`. The 300-gene XENIUM and MERSCOPE
  panels stay in memory.

Marker eligibility uses detection only, so both modes give the same markers,
detection fractions and MECR values (spatial and reference). The Wilcoxon
columns of `mecr_reference_gene_statistics.csv` (and of the marker table)
differ by design in streaming mode:

- `wilcoxon_score`, `logfoldchange`, `p_value` and `p_value_adjusted` are NaN
  for genes that are not detection candidates;
- `p_value_adjusted` is Benjamini-Hochberg over the tested candidates of each
  class instead of over every panel gene. The scores and raw p-values of the
  tested genes are unchanged.

The pass-2 matrix is held column-wise (CSC) so the per-gene Wilcoxon ranking
reads each gene once; `logfoldchange` can differ from the in-memory value in
the last float32 digit. `mecr_reference_manifest.json` records the requested
and resolved mode and why, the number of tested genes and the BH scope.

Reference row chunks can be decoded by several processes
(`reference_workers`, default 1; same results in the same order).
`MECR_REFERENCE` keeps the default and writes no mode, so its JSON is the one
of earlier runs. `MECR_REFERENCE_ATERA` (8 CPUs, 64 GB, one fork) writes
`"reference_mode": "streaming"` and `"reference_workers": <task.cpus>`, and
publishes to `<outdir>/mecr_reference_atera/`. Its panel is the union of the
ATERA rows' tables; every other setting is the one of `MECR_REFERENCE`.

The spatial tables are read straight from their zarr group (only the selected
table), not by opening the whole SpatialData store, whose labels, shapes and
other tables take minutes to parse on a whole-transcriptome store.

## Main parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mecr_enabled` | human: `true`; mouse: `false` | Enable the species-matched stage for a row. Mouse is opt-in due to reference size. |
| `mecr_reference_h5ad_paths` | `[]` | Explicit complete raw reference shard list; auto-download fills this for WMB. |
| `mecr_neurons_h5ad_path` | WHB-10Xv3 neuron raw H5AD | Legacy WHB neuronal reference matrix. |
| `mecr_nonneurons_h5ad_path` | WHB-10Xv3 non-neuron raw H5AD | Legacy WHB non-neuronal reference matrix. |
| `mecr_cell_metadata_path` | atlas-dependent | Maps reference cell labels to cluster aliases. |
| `mecr_taxonomy_metadata_path` | atlas-dependent | Resolves taxonomy labels. |
| `mecr_cluster_membership_path` | atlas-dependent | Maps cluster aliases to the selected taxonomy level. |
| `mecr_reference_cache_dir` | `<outdir>/mapmycells_cache` | Durable cache for automatic WMB downloads. |
| `mecr_auto_download_reference` | `true` | Download or reuse every complete WMB reference input for mouse. |
| `mecr_marker_min_target_fraction` | `0.25` | Strict lower detection threshold in the target class. |
| `mecr_marker_max_other_fraction` | `0.01` | Strict upper detection threshold outside the target class. |
| `mecr_reference_chunk_rows` | `5000` | Number of reference cells read per chunk. |
| `mecr_figure_dpi` | `180` | Distribution plot resolution. |
| `mecr_barnyard_top_n_pairs` | `6` | Maximum number of barnyard gene-pair plots. |
| `mecr_barnyard_max_points` | `50000` | Maximum displayed cells per platform and barnyard plot. |
| `mecr_barnyard_log1p` | `false` | Opt into `log1p` rather than natural count axes. |

See [Configuration](../configuration.md#mecr) for the full parameter list and
[Outputs](../outputs.md#mecr) for generated artifacts.
