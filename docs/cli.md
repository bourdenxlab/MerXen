# CLI reference

The `merxen` CLI is a Click command group registered by `pyproject.toml`
([line 44](../pyproject.toml#L44)) and defined at
[src/merxen/cli/__init__.py](../src/merxen/cli/__init__.py). Each subcommand
corresponds to exactly one Nextflow process, takes a single `--config
<path>.json` argument, and validates it against the matching Pydantic model
in [src/merxen/config.py](../src/merxen/config.py).

```
$ merxen --help
Usage: merxen [OPTIONS] COMMAND [ARGS]...

  MerXen spatial transcriptomics pipeline CLI.

Commands:
  build-spatialdata  Build a platform-specific SpatialData zarr from raw input
  segment            Run Cellpose + ProSeg segmentation for one dataset
  cellpose-nuclei-segment
                      Run reusable DAPI-only Cellpose nuclei segmentation
  enrich             Enrich a segmented zarr with per-shape tables
  mask-image-quantification
                      Quantify image channels over final Cellpose masks
  compute-cortical-depth
                      Compute Laplace/equal-area cortical-depth coordinates
  distance-from-object
                      Assign registered polygon-edge distances and pseudobulks
  distance-from-object-cohort
                      Run paired near-vs-far PyDESeq2 by platform
  qc                 Compute geometry and assignment QC metrics
  align              Align MERSCOPE into paired Xenium coordinates
  materialize-alignment
                      Reconcile aligned vectors, rasters, and image outputs
  alignment-qc       Compute post-alignment QC metrics
  compare            Run cross-platform gene-level comparison
  visualize          Generate visualization artifacts for a pair
  spatial-gene-analysis
                      Run cell and transcript-coordinate spatial gene analysis
  mecr-reference     Discover MECR markers in a complete whole-brain reference
  mecr               Score mutually exclusive co-expression rates
  clustering-squidpy Run Scanpy/Squidpy clustering analysis
  mapmycells         Run local MapMyCells cell type assignment
  annotation-panel   Resolve the declared panels of a pair and...
  annotation-reference-prep
                      Get or build one reference bundle and...
  annotation-store   Inspect the annotation reference store...
```

The reference-based annotation commands (`annotation-*`, plan
`docs/plans/robust-celltype-annotation-plan.md` §3.2) take explicit
options instead of a single `--config`.

Logging is configured in the root `main()` group and streams to stderr at
`INFO` level.

---

## `merxen build-spatialdata`

Build or reuse a platform-specific SpatialData zarr.

```bash
merxen build-spatialdata --config build_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON file validated against [`SpatialDataBuildConfig`](../src/merxen/config.py#L112). |
| `--force-rerun` | Rebuild even if a cached zarr is available. |

Details: [Stage 1 — SpatialData build](stages/spatialdata-build.md).

---

## `merxen segment`

Run Cellpose tiled segmentation, export transcripts, and run ProSeg.

```bash
merxen segment --config segment_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`SegmentationConfig`](../src/merxen/config.py#L146). |
| `--force-rerun` | Ignore cached `proseg_base_latest.zarr` / `proseg_base_raw.zarr` in the output dir. |

Details: [Stage 2 — Segmentation](stages/segmentation.md).

---

## `merxen cellpose-nuclei-segment`

Run only the reusable DAPI-only Cellpose nuclei process. It uses the `nuclei`
model, cell Cellpose inference/tiling settings, and its own 5–400 µm² mask
filter.

```bash
merxen cellpose-nuclei-segment --config segment_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `SegmentationConfig`. |
| `--force-rerun` | Recompute the nuclei mask even if the durable output exists. |

Details: [Stage 2 — Segmentation](stages/segmentation.md).

---

## `merxen enrich`

Enrich a segmented zarr with explicit shape layers and per-shape gene
tables.

```bash
merxen enrich --config enrich_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`EnrichmentConfig`](../src/merxen/config.py#L157). |
| `--force-rerun` | Overwrite existing shape layers and tables. |

Details: [Stage 3 — Enrichment](stages/enrichment.md).

---

## `merxen mask-image-quantification`

Quantify all SpatialData image channels over final Cellpose label-mask pixels.

```bash
merxen mask-image-quantification --config mask_image_quantification_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `MaskImageQuantificationConfig`. |
| `--force-rerun` | Recompute the SpatialData table and sidecar exports even when present. |

Details: [Stage 4 — Mask image quantification](stages/mask-image-quantification.md).

---

## `merxen qc`

Compute per-cell geometry and transcript-assignment metrics.

```bash
merxen qc --config qc_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`QCConfig`](../src/merxen/config.py#L169). |

Details: [Stage 4 — QC](stages/qc.md).

---

## `merxen compute-cortical-depth`

Compute cortical-depth coordinates and update selected AnnData tables.

```bash
merxen compute-cortical-depth --config cortical_depth_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `CorticalDepthConfig`. |

Details: [Cortical depth](stages/cortical-depth.md).

---

## `merxen distance-from-object`

Annotate one platform zarr and create tissue-block near/far pseudobulks.

```bash
merxen distance-from-object --config distance_from_object_config.json
```

## `merxen distance-from-object-cohort`

Combine pair-level pseudobulks for one platform and run paired PyDESeq2.

```bash
merxen distance-from-object-cohort --config distance_from_object_cohort_config.json
```

The commands validate `DistanceFromObjectConfig` and
`DistanceFromObjectCohortConfig`, respectively. See [Distance from
object](stages/distance-from-object.md).

---

## `merxen align`

Align a MERSCOPE section into paired Xenium coordinates.

```bash
merxen align --config align_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `AlignmentConfig`. |

The default JSON backend is `valis`; set `backend` to `legacy_spateo` to run
the former implementation. The recommended reproducible setup is
`envs/environment.alignment.yml`, which installs `requirements/requirements.alignment.lock` and
the exact VALIS package. Validate either backend with
`merxen check-alignment-deps --backend valis|legacy_spateo`.

Details: [Section alignment](stages/alignment.md).

---

## `merxen materialize-alignment`

Reconcile all enabled aligned artifacts from a saved or embedded transform.

```bash
merxen materialize-alignment --config align_config.json \
  --summary materialization_summary.json
```

The command validates `AlignmentConfig.materialization`, writes JSON status to
stdout and optionally to the machine-readable `--summary` path, and refuses
stale or missing output when `reconcile=false` unless `force=true`.

---

## `merxen alignment-qc`

Collate the selected backend's post-alignment QC. VALIS runs report DAPI
morphology metrics and an image overlay; only the legacy backend uses the
former expression-grid and centroid QC.

```bash
merxen alignment-qc --config alignment_qc_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `AlignmentQCConfig`. |

Details: [Section alignment](stages/alignment.md).

---

## `merxen compare`

Run cross-platform gene-level comparison (one pair at a time).

```bash
merxen compare --config compare_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`ComparisonConfig`](../src/merxen/config.py#L177). |

Details: [Stage 5 — Comparison](stages/comparison.md).

---

## `merxen visualize`

Generate visualization artifacts for a paired or single-platform dataset.

```bash
merxen visualize --config visualize_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`VisualizationConfig`](../src/merxen/config.py#L186). |

Details: [Stage 6 — Visualization](stages/visualization.md).

---

## `merxen spatial-gene-analysis`

Run cell-level Moran's I/Geary's C and assignment-independent transcript
signed-distance and nested pair-correlation analyses.

```bash
merxen spatial-gene-analysis --config spatial_gene_analysis_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `SpatialGeneAnalysisConfig`. |

Details: [Spatial gene analysis](stages/spatial-gene-analysis.md).

---

## `merxen mecr-reference`

Discover paper-standard mutually exclusive broad-class markers in the complete
species-matched WHB or WMB reference for the selected spatial panel.

```bash
merxen mecr-reference --config mecr_reference_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `MecrReferenceConfig`. |

The workflow normally calls this command once and shares its marker table with
all MECR branch tasks.

## `merxen mecr`

Calculate pair-level and aggregate MECR values for one pair and segmentation.

```bash
merxen mecr --config mecr_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `MecrConfig`. |

Details: [Mutually exclusive co-expression rate](stages/mecr.md).

---

## `merxen clustering-squidpy`

Run per-platform Scanpy/Squidpy QC, clustering, UMAP, and spatial plots for
one pair.

```bash
merxen clustering-squidpy --config clustering_squidpy_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ClusteringSquidpyConfig`. |

Details: [Squidpy clustering](stages/clustering-squidpy.md).

---

## `merxen mapmycells`

Run local Allen Institute MapMyCells cell type assignment on clustered AnnData
outputs.

```bash
merxen mapmycells --config mapmycells_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `MapMyCellsConfig`. |

The active Python environment must include Allen's `cell_type_mapper` package.

Details: [MapMyCells](stages/mapmycells.md).

---

## `merxen annotation-panel`

`ANNOTATE_PANEL` for one pair x segmentation: reads each platform's
**declared** panel, removes control features, resolves Ensembl IDs, picks
the panel mode and writes the panel files and the bundles the pair needs.

```bash
merxen annotation-panel --prepared-dir clustering_prepare_out --species human \
  --clustering-config clustering_squidpy_config.json \
  --annotation-config annotation_config.json \
  --shared-tissue-mask align_out/shared_tissue_mask.npy \
  --registration-summary align_out/registration_summary.json \
  --output-dir annotation_panel
```

| Option | Description |
|--------|-------------|
| `--prepared-dir PATH` | `CLUSTERING_SQUIDPY_PREPARE` output (`manifest.json` + `<platform>/<sid>_prepared.h5ad`). |
| `--panel-genes-path PATH` | Instead of `--prepared-dir`: a gene list for prepare-only runs (`annotation_panel_genes_path`). |
| `--species human\|mouse` | Run species. |
| `--platforms LIST` | Comma-separated platforms to use (default: every prepared sample). |
| `--output-dir PATH` | Where the panel files go. |
| `--annotation-config PATH` | `AnnotationConfig` JSON (references, panel settings); default: species defaults. |
| `--clustering-config PATH` | `clustering_squidpy_config.json`: pair id, sample platforms, segmentation, `min_counts`. |
| `--pair-id`, `--segmentation` | Override the clustering config's values. |
| `--panel-file KEY=PATH` | Declared-panel file per sample id or platform: Xenium `gene_panel.json`, MERSCOPE codebook or a gene table. Without one, the unfiltered `var` of the prepared H5AD is the declared panel. |
| `--shared-tissue-mask PATH`, `--registration-summary PATH` | The pair's `shared_tissue_mask.npy` and the `registration_summary.json` giving its frame; a label-free set c then uses table cells inside the mask (their sha256 go into `panel_report.json`). |
| `--require-shared-tissue-mask` | Refuse a label-free set c without a usable mask (pipeline runs of aligned pairs). The seeded set-a family's curated set c needs no mask. |
| `--min-counts N` | Table-cell threshold (default: the clustering config's). |
| `--gene-id-fallback-csv PATH` | Local gene table for the symbol -> Ensembl fallback (M0e); overrides the config. |
| `--panel-mode auto\|intersection\|per_platform` | Override `annotation_panel_mode`. |

Outputs: `panel_report.json` (resolution by source, unresolved features,
controls removed and why, merged duplicates, set c with its basis: the
curated family list or the label-free fallback rule), one `panel_genes*.json`
per annotation panel (`panel_genes.json` = set a or the sample's panel,
`panel_genes_setc.json`, or `panel_genes_<platform>.json` +
`panel_genes_intersection.json` for `per_platform` pairs) and
`required_bundles.json` (each bundle's reference, role, purpose, panel file,
`panel_hash` and `n_panel_genes`, `uses` (every purpose and panel file one
bundle serves, e.g. set c equal to set a) and `n_required`). Set c of the
seeded set-a family is set a minus the curated E5 list
(`setc_exclusions_human.csv`), also for a gene list of that family; other
families use the label-free rule
([Set c](stages/annotation.md#set-c)). `panel_hash` is the
sha256 of the sorted resolved IDs of the declared panel, so cells, zero-count
probes and `var` order never change it. `--annotation-config` may name
references by id only (`"references": ["whb_frontal_supc_clus", ...]`, or
`{"reference_id": "wmb_panel", "max_cells_per_cluster": 50}`); their known
taxonomy settings fill the rest.

## `merxen annotation-reference-prep`

`ANNOTATE_REFERENCE_PREP` for one (reference, panel): returns the bundle
whose `build_hash` matches, building it once if needed, and writes
`bundle_ref.json` (the bundle's identity only: a re-run writes the same
bytes; whether the bundle was built or reused is printed). Store, builder
and input errors end the command with a one-line message and exit code 1. A
build waiting for another build of the same reference logs what it waits
for.

```bash
merxen annotation-reference-prep --reference-id whb_frontal_supc_clus \
  --species human --panel-genes annotation_panel/panel_genes.json \
  --store /media/mathieubo/SSD1/MerXen/annotation_references \
  --store-large /srv/storage/MerXen/annotation_references \
  --source region_precompute=/path/to/region_frontal_a44_a45_a46_a32_acc \
  --auto-download --n-processors 8 --max-gb 40 --output bundle_ref.json
```

| Option | Description |
|--------|-------------|
| `--reference-id ID` | Reference from the annotation config or the known references (plan §3.2). |
| `--species human\|mouse` | Run species. |
| `--panel-genes PATH` | `panel_genes*.json` (omit for panel-independent references such as `wmb_region_share`). |
| `--store PATH`, `--store-large PATH` | Store roots; panels above 1,000 genes go to the large store. |
| `--annotation-config PATH` | `AnnotationConfig` JSON. |
| `--source NAME=PATH` | Reference source files, added to the spec's sources. |
| `--scratch-dir PATH` | Parent of the build scratch directory (never inside a store). |
| `--output PATH` | `bundle_ref.json` to write. |
| `--download-dir PATH` | Cache of pinned reference downloads (default `<store>/.downloads`). |
| `--auto-download / --no-auto-download` | Download missing pinned files (SEA-AD Multiregion, MERFISH-C57BL6J-638850-CCF cell metadata; default off). |
| `--download-seed KEY=PATH` | A local copy of a pinned file (e.g. `precomputed_stats=...`), copied into the cache only when its sha256 matches. |
| `--n-processors N`, `--max-gb N` | `cell_type_mapper` processes (default 8) and reference-marker memory bound (default 40 GB); not part of `build_hash`. |

Sources per reference (`merxen.annotation.reference`; directories are
expanded to the files a builder reads before `build_hash` is computed):

| Reference | Sources |
|---|---|
| `whb_frontal_supc_clus` (also set c) | `region_precompute` (the region precompute, or its reference directory with `region_reference_manifest.json`), or for a rebuild `whb_metadata_dir` + `whb_h5ad_dir`; `seaad_precomputed_stats` (negative genes; from the download cache when absent) |
| `seaad_mr_panel` | `seaad_precomputed_stats`, optional `seaad_metadata_dir` (taxonomy tables); pinned files come from the download cache when absent |
| `wmb_panel` | `wmb_h5ad_dir` (WMB-10Xv3 `*-raw.h5ad`; only those files are hashed), `wmb_metadata_dir` (`cell_metadata.csv`, taxonomy tables), `wmb_mapping_stats` (Allen `precomputed_stats_ABC_revision_230821.h5`), `wmb_selfmap_test_cells` (required while resolvability is enabled; a copy matching the pinned sha256 is seeded into `<download-dir>/local/wmb_selfmap/` and used from there), optional `wmb_marker_gene_universe` (default: the validated ag7 ∪ VZG2 union for panels inside it) |
| `wmb_region_share` | `merfish_ccf_metadata` (`cell_metadata_with_parcellation_annotation.csv`; from the download cache when absent, key `merfish_ccf_metadata`) |
| `whb_whole_ctx_panel` (optional) | `whb_whole_precompute` |

Each `cell_type_mapper` step runs in its own interpreter with one BLAS
thread and no GPU; its input JSON, logs and peak RSS are kept under the
bundle's `logs/` and in `bundle.json`.

Store layout: `<store>/<reference_id>/<build_hash>/` (complete bundles,
read-only: files 0444, directories 0555), `<store>/.tmp-<uuid>/` (a build in
progress or killed), `<store>/.failed-<uuid>/` (a build that raised, with
`build_error.txt`), `<store>/.downloads/` (verified pinned inputs) and
`<store>/.source_digests/` (full source sha256 by file identity, so large
sources are hashed once). Sources are copied with a checksum, never
symlinked, and nothing is ever deleted automatically. Reuse checks every
bundle file's size and the sha256 of files up to 64 MB.

## `merxen annotation-store`

```bash
merxen annotation-store list --store DIR [--store-large DIR] [--json]
merxen annotation-store prune --store DIR --unreferenced-by RESULTS_ROOT --dry-run [--json]
```

`list` shows bundles, temporary and failed builds with their size; both
commands refuse a store path that does not exist. `prune`
requires `--dry-run` and only lists the bundles no `bundle_ref.json`,
`map_manifest.json`, `required_bundles.json` or `*_annotation_manifest.json`
under the results root references, plus failed and dead temporary builds.
Delete by hand after review (OD-D4).

---

## Writing a standalone config

The Nextflow workflow writes these JSON configs for you, but you can also
hand-roll them to drive a single stage outside of Nextflow. Example for
`merxen qc`:

```json
{
  "dataset_name": "EXAMPLE01_MERSCOPE",
  "latest_zarr_path": "/path/to/proseg_base_latest.zarr",
  "output_dir": "./qc_out"
}
```

Save as `qc_config.json` and run:

```bash
merxen qc --config qc_config.json
```

The Pydantic layer will complain loudly about anything missing or
mis-typed.
