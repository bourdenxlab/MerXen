# Mutually exclusive co-expression rate (MECR)

MECR measures unexpected co-detection of genes that are mutually exclusive
between broad cell classes in a single-cell RNA-seq reference. It follows the
metric introduced by Hartman and Satija in
[Comparative analysis of multiplexed in situ gene expression profiling technologies](https://doi.org/10.7554/eLife.96949.1).
A lower value indicates greater molecular-assignment specificity; an elevated
value can reflect off-target signal or overly permissive cell segmentation. The
stage is enabled by default and runs after QC, independently of alignment.

## Method

The stage uses the complete Allen Whole Human Brain 10x v3 reference. It joins
the neuron and non-neuron raw H5AD cell labels to the same WHB taxonomy metadata
and broad-class collapse used by the Squidpy clustering stage. The default
classes are:

- Neurons
- Oligodendrocytes
- Oligodendrocyte precursors
- Astrocytes
- Microglia
- Fibroblasts
- Vascular cells

Reference preparation is restricted to genes present in the spatial panel,
but each reference cell is normalized using its full-library count before the
panel is selected. Expression is normalized to 10,000 counts per cell and
log-transformed. Python/Scanpy Wilcoxon tests compare each broad class with the
rest. Following the paper, a gene is retained for a class only when it is
detected in strictly more than 25% of that class and strictly less than 1% of
the other retained cells. A gene that qualifies for more than one class is
removed.

Every unordered pair of retained genes from different broad classes is then
scored in each spatial cell-count table:

```text
MECR(gene 1, gene 2) = cells detecting both genes / cells detecting either gene
```

The sample-level MECR is the unweighted arithmetic mean of all finite pair
rates. Pairs for which neither gene is detected have an undefined (NaN) rate;
they remain in the audit table and are excluded from the aggregate mean.

## Plots

Reference preparation writes a histogram of MECR across every eligible WHB
marker pair, with mean and median lines for description only. Unlike the
exploratory notebook, no reference-MECR cutoff is used to select pairs.

Each spatial branch writes:

- the complete pair-rate distribution by platform;
- a MERSCOPE-versus-Xenium scatter with an identity line, restricted to the
  exact eligible pairs shared by both platforms;
- one broad-class-pair median-MECR heatmap per platform; and
- barnyard count scatterplots for up to `mecr_barnyard_top_n_pairs` pairs.

Barnyard pairs are selected deterministically from eligible production pairs:
canonical pairs are prioritized when available, followed by pairs with the
highest mean spatial MECR and those detected in the most cells. The selection
CSV records every reason. Display coordinates use natural raw cell counts by
default and may be downsampled to `mecr_barnyard_max_points`, while the MECR
shown in the title is always calculated from every cell. Set
`mecr_barnyard_log1p=true` to opt into the earlier `log1p` display.

## Workflow behaviour

`MECR_REFERENCE` runs once per workflow invocation and streams both complete
WHB raw matrices in bounded row chunks. Its output is shared by every selected
sample and segmentation branch. `MECR` then scores each branch separately for
MERSCOPE and/or Xenium. Because the reference preparation is the expensive
step, keep the Nextflow work directory and use `-resume` when rerunning the same
panel and reference settings.

MECR can be disabled globally with:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --mecr_enabled false
```

It can also be selected alone with `--only_stage mecr`, provided the published
`latest_spatialdata.zarr` inputs already exist. A samplesheet `mecr_enabled`
column overrides the global switch per row.

## Main parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `mecr_enabled` | `true` | Enable the stage for a row. |
| `mecr_neurons_h5ad_path` | WHB-10Xv3 neuron raw H5AD | Complete neuronal reference matrix. |
| `mecr_nonneurons_h5ad_path` | WHB-10Xv3 non-neuron raw H5AD | Complete non-neuronal reference matrix. |
| `mecr_cell_metadata_path` | WHB cell metadata CSV | Maps reference cell labels to cluster aliases. |
| `mecr_taxonomy_metadata_path` | WHB taxonomy CSV | Resolves taxonomy labels. |
| `mecr_cluster_membership_path` | WHB membership CSV | Maps cluster aliases to the selected taxonomy level. |
| `mecr_marker_min_target_fraction` | `0.25` | Strict lower detection threshold in the target class. |
| `mecr_marker_max_other_fraction` | `0.01` | Strict upper detection threshold outside the target class. |
| `mecr_reference_chunk_rows` | `5000` | Number of reference cells read per chunk. |
| `mecr_figure_dpi` | `180` | Distribution plot resolution. |
| `mecr_barnyard_top_n_pairs` | `6` | Maximum number of barnyard gene-pair plots. |
| `mecr_barnyard_max_points` | `50000` | Maximum displayed cells per platform and barnyard plot. |
| `mecr_barnyard_log1p` | `false` | Opt into `log1p` rather than natural count axes. |

See [Configuration](../configuration.md#mecr) for the full parameter list and
[Outputs](../outputs.md#mecr) for generated artifacts.
