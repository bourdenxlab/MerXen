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
  annotate           Map published or prepared samples with...
  annotate-resolve   Resolve MAP outputs into label tables...
  annotation-panel-fetch
                      Fetch pinned public panel gene lists (URL, size and...
  annotation-panel-simulate
                      Simulate a candidate panel: predicted levels, trust,...
```

The reference-based annotation commands (`annotation-*`, `annotate` and
`annotate-resolve`, plan `docs/plans/robust-celltype-annotation-plan.md`
§3.2–§3.4) take explicit options instead of a single `--config`.

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
  --store-large /srv/storage/MerXen/annotation_references_large \
  --source region_precompute=/path/to/region_frontal_a44_a45_a46_a32_acc \
  --auto-download --n-processors 8 --max-gb 40 --output bundle_ref.json
```

| Option | Description |
|--------|-------------|
| `--reference-id ID` | Reference from the annotation config or the known references (plan §3.2). |
| `--species human\|mouse` | Run species. |
| `--panel-genes PATH` | `panel_genes*.json` (omit for panel-independent references such as `wmb_region_share`). |
| `--store PATH`, `--store-large PATH` | Store roots; bundles of panels above 1,000 genes go to the large store (dwight: `/srv/storage/MerXen/annotation_references_large`) with their reference markers kept (plan §8.7); the per-parent marker prefilter is opt-in (`large_panel_marker_prefilter`). A builder refuses what it cannot build safely before any source is downloaded or hashed (`LargePanelRefusedError`): `whb_whole_ctx_panel` above 1,000 genes, and a large `wmb_panel` without the prefilter whose predicted query-marker peak (the measured 37.7 GB up to 5,006 genes, scaled beyond) exceeds the PREP reserve (`--max-gb` / 0.625; the prefilter is mandatory above it, OD-E8). |
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
| `whb_frontal_supc_clus` (also set c) | `region_precompute` (the region precompute, or its reference directory with `region_reference_manifest.json` and `region_cell_metadata.csv`), or for a rebuild `whb_metadata_dir` + `whb_h5ad_dir`; `seaad_precomputed_stats` (negative genes; from the download cache when absent). While resolvability is enabled also `whb_h5ad_dir` and `whb_metadata_dir` (the held-out test set: raw WHB h5ads, taxonomy tables, the WHB cell metadata for the other-region non-neuronal test cells, and the region cell metadata from the region directory) |
| `seaad_mr_panel` | `seaad_precomputed_stats`, optional `seaad_metadata_dir` (taxonomy tables); pinned files come from the download cache when absent. While resolvability is enabled also `whb_region_dir` (the WHB region reference directory), `whb_h5ad_dir` and `whb_metadata_dir` (the shared held-out test set) |
| `whb_frontal_supc_clus_ho` (resolvability) | `whb_region_dir` (or `whb_region_cell_metadata`, or `whb_metadata_dir` with the WHB cell metadata and ROI map to rebuild it), `whb_h5ad_dir`, `whb_metadata_dir` (always with the WHB cell metadata: the other-region non-neuronal test cells); built by the WHB and SEA-AD builders through the store, or directly |
| `wmb_selfmap_testset` (resolvability) | `wmb_selfmap_test_cells`, `wmb_h5ad_dir`, `wmb_metadata_dir`; built by the `wmb_panel` builder through the store, or directly |
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

## `merxen annotation-panel-fetch`

```bash
merxen annotation-panel-fetch --out-dir DIR [--panel KEY ...]
```

Downloads the public vendor panel gene lists the panel simulation starts from
(plan §8.8; `merxen.annotation.public_panels`), pinned by URL, size, sha256
and gene count: `xenium_prime_5k_human` (5,001 genes), `xenium_prime_5k_mouse`
(5,006), `xenium_human_brain_v1` (266) and `xenium_mouse_brain_v1` (248), all
from `https://cdn.10xgenomics.com/raw/upload/software-support/Xenium-panels/`.
Without `--panel`, every pinned list. A verified copy under `DIR/raw/` is
reused; a download whose sha256 differs from the pin is refused (a changed
vendor file is a new panel family). Writes `DIR/<key>.gene_list.csv`
(`gene_symbol`, `gene_id`) and `DIR/<key>.manifest.json` (URL, size,
sha256, gene count, time). Keep `DIR` outside the repository. These are
panel definitions, not datasets: no public validation data are downloaded
(OD-E1 / OD-E9).

## `merxen annotation-panel-simulate`

The in-silico design aid for a panel before its data exist (plan §8.8,
`merxen.annotation.simulate`).

```bash
merxen annotation-panel-simulate --public-panel xenium_prime_5k_mouse \
  --panel-dir /srv/storage/MerXen/annotation_dev/evidence_20260926/m3b/panels \
  --store /media/mathieubo/SSD1/MerXen/annotation_references \
  --store-large /srv/storage/MerXen/annotation_references_large \
  --annotation-config mouse.annotation_config.json \
  --param annotation_wmb_h5ad_dir=/path/WMB-10Xv3/20230630 \
  --param annotation_wmb_metadata_dir=/path/abc_atlas/metadata \
  --param annotation_wmb_mapping_stats_path=/path/precomputed_stats_ABC_revision_230821.h5 \
  --param annotation_wmb_selfmap_test_cells_path=/path/truth.csv \
  --scratch-dir /srv/storage/.../scratch --out-dir /srv/storage/.../simulate/5k_mouse \
  --n-processors 8 --max-gb 40 --expected-depth 250
```

1. Resolves the gene list as a declared panel (`annotation-panel
   --panel-genes-path`: gene IDs, controls, species test, family).
2. Builds the species' self-map references through the store with the
   production configuration (human `whb_frontal_supc_clus` and
   `seaad_mr_panel`, mouse `wmb_panel`; `--references` to choose): the same
   bundles a pipeline PREP would build, reused when they exist. Panels above
   1,000 genes go to the large store with their reference markers kept.
3. Reports the predicted emission per (level, class, depth bin) from the
   bundle's resolvability decisions (resolvability version 3: split halves,
   local isotonic thresholds, Wilson bound and point precision, pooled deep
   bins) in the panel's regime (`validated` only for a real-data-validated
   family, else `provisional` with the provisional margins), the trust state,
   weak and collapsed parents; with `--expected-depth` the classes emitted at
   that median depth, with `--depth-profile` (CSV of per-cell panel counts)
   each class's share of cells in emitted bins.
4. For prefiltered panels (the prefilter is opt-in; `--prefilter-compare
   auto`; `always` / `off`),
   finds the unfiltered lookup on the self-map engine's marker precompute,
   maps the same simulated cells (decision recipe, seed 0) with it and
   compares per (level, class): agreement >= 0.95 at bp >= 0.8 for every
   class with >= 50 unfiltered confident calls of an emitted level, and no
   parent of the prefiltered lookup below 5 markers (NP9, the pre-registered
   rule; the relaxed reading, no parent made weak by the prefilter, is
   reported as `no_parent_made_weak`). This is also the unfiltered marker
   memory measurement (OD-E8).
5. Records wall time per step, peak RSS (largest process, as `/usr/bin/time`
   reports it, and the summed PSS of each step's process tree) and disk
   (bundle, kept reference markers, test set).

Outputs in `--out-dir`: `simulate_report.json`, `SIMULATE_REPORT.txt`,
`predicted_levels.csv` (+ `<reference>/predicted_levels_unfiltered.csv`),
`prefilter_comparison.csv`, `resources.csv`, `panel/` (the panel files) and
`<reference>/logs/`.

| Option | Description |
|--------|-------------|
| `--gene-list PATH` / `--public-panel KEY --panel-dir DIR` | The panel: any `read_panel_file` format, or a pinned public list (fetched or verified into `DIR`). |
| `--species`, `--platform`, `--name` | Run species (default: the public list's), panel platform, report name. |
| `--references LIST` | References to build (default: the species' self-map references). |
| `--store`, `--store-large`, `--download-dir`, `--auto-download` | As `annotation-reference-prep`. |
| `--param NAME=PATH` | Reference sources by their Nextflow param name (`annotation_whb_h5ad_dir=...`); mapped to each reference's sources as `AnnotationReferences.SOURCE_PARAMS` does. |
| `--annotation-config PATH` | `AnnotationConfig` JSON (production settings). |
| `--scratch-dir PATH` | Build scratch and the unfiltered markers (large for 5K panels: use `/srv/storage`). |
| `--n-processors N`, `--max-gb N` | ctm processes and reference-marker memory bound. |
| `--prefilter-compare auto\|always\|off` | See step 4. |
| `--expected-depth N`, `--depth-profile PATH` | See step 3. |
| `--gate-p` | The gate-P programme (NP1–NP9: leave-one-donor-out HO bundles, the second mouse test draw, seeds 0 / 1, stress recipes; plan §8.8, §14). M13 registers it (`merxen.annotation.simulate.register_gate_p_hook`, from `scripts/acceptance/new_panel.py`); until then the option is refused before any compute. |

## `merxen annotate`

The annotation MAP step (plan §3.3): MapMyCells per sample and required
bundle, standalone on published clustered H5ADs or on a
`CLUSTERING_SQUIDPY_PREPARE` directory. It never writes (`--out` or
`--work-dir`) into the inputs' results tree: not below an input's directory,
not below the results root of a published clustered H5AD or of any input
under a `<root>/<pair>/<seg>/clustering_squidpy/` layout, and not below a
`--results-root`.

```bash
merxen annotate --species human \
  --from-clustered-h5ad results/P7513/proseg_hybrid/clustering_squidpy/clustering_squidpy_out/merscope/P7513_MERSCOPE_clustered.h5ad \
  --from-clustered-h5ad results/P7513/proseg_hybrid/clustering_squidpy/clustering_squidpy_out/xenium/P7513_XENIUM_clustered.h5ad \
  --store /media/mathieubo/SSD1/MerXen/annotation_references \
  --gene-id-fallback-csv /path/to/WHB/gene.csv \
  --out shadow/P7513/proseg_hybrid
```

| Option | Meaning |
|---|---|
| `--from-clustered-h5ad PATH` | A published `<sid>_clustered.h5ad` (table cells, raw counts in `layers["counts"]`); repeat once per platform. Pair, segmentation and platform come from the results path. |
| `--prepared-dir DIR` | Instead: prepared H5ADs (counts in `X`, every segmented object; objects below `--min-counts` are not mapped). |
| `--store DIR`, `--store-large DIR` | Reference store(s); the bundle of each required (reference, `panel_hash`) is the one complete bundle of the current builder version built with the large-panel prefilter `--annotation-config` asks for (none by default); of several, the one with the current resolvability version. |
| `--bundle KEY=DIR`, `--bundle-ref PATH` | Use this bundle directory (`KEY` = reference id or run id, e.g. `whb_frontal_supc_clus_setc`) or this `bundle_ref.json` instead of the store lookup. |
| `--panel-dir DIR` | `merxen annotation-panel` output; default: the panel is computed from the inputs into `<out>/panel`. |
| `--references IDS` | Comma-separated reference ids to map (default: every primary and secondary bundle the panel requires). |
| `--annotation-config PATH` | `AnnotationConfig` JSON (thresholds, `xplat_sensitivity_segmentations`, `ctm_version`, ...). |
| `--clustering-config PATH` | With `--prepared-dir`: the `clustering_squidpy_config.json` of the run (pair id, sample platforms and `min_counts`, which `--min-counts` may not contradict). |
| `--min-counts N` | Table-cell threshold (the clustering `min_counts`; default: the clustering config's, else 10). |
| `--n-processors N` | MapMyCells processes (default `$MERXEN_ANNOTATION_MAP_N_PROCESSORS` or 6). |
| `--work-dir DIR` | Scratch for the query H5ADs, restricted lookups and extended JSONs (default `<out>/.work`, removed); refused inside a results tree, as `--out`. |
| `--results-root DIR` | A results tree `--out` and `--work-dir` must stay out of (repeatable), on top of the inputs' own. |
| `--keep-extended-json`, `--reuse/--no-reuse`, `--reuse-from DIR` | Keep the gzipped extended JSON; reuse identical runs of a `map_manifest.json` (default: `--out`). |
| `--gene-id-fallback-csv PATH` | Local symbol → Ensembl table (M0e), as for `annotation-panel`. |
| `--declared-ids-file KEY=PATH` | Per sample id or platform, a panel file (e.g. the Xenium `gene_panel.json`) whose native gene IDs complete the declared features a clustered H5AD's `min_cells` filter dropped from `var`, so its declared panel hash is the prepared H5AD's; without it those features are resolved by symbol and listed as `declared_ids_incomplete`. |
| `--platforms`, `--no-provisional` | Map only these platforms; skip the provisional labels. |
| `--require-bundle-refs` | Map only the bundles given with `--bundle-ref` / `--bundle`; a missing one fails instead of being looked up in the store (what the pipeline task passes: it maps exactly the bundles `ANNOTATE_REFERENCE_PREP` resolved). A needed subset bundle is then only recorded as `requested`, never looked up in `--store`. Refs of roles MAP does not map (`wmb_region_share`) are accepted and not opened. |
| `--allow-refused-panel` | For a refused panel (`required_bundles.json` status `refused`), write `map_manifest.json` with `panel_status: refused`, its reasons and no runs, and exit 0 (pipeline runs: RESOLVE then writes statuses only). Without it a refused panel is an error. |

MapMyCells runs as a subprocess of `merxen.analysis.mapmycells_entrypoint`
with the validated configuration (bootstrap factor 0.5, 100 iterations,
seed 0, raw normalisation, `cloud_safe` off, one BLAS / numba thread per
worker, no GPU); the installed `cell_type_mapper` must be the configured
version (1.7.2). Outputs under `--out`: `<platform>/<sid>_mmc_<run_id>.parquet`
(one row per cell × taxonomy level: assignment, name, bootstrap and
aggregate probability, `avg_correlation`, runner-ups 1–5 with probabilities
and correlations, `directly_assigned`), `<platform>/<sid>_ct_provisional.parquet`
and `map_manifest.json` ([Reference-based annotation](stages/annotation.md#mapping-merxen-annotate)).
A run is reused when the manifest in `--reuse-from` has the same query
fingerprint, bundle `build_hash`, engine parameters, ctm version, tidy
schema version and (restricted) lookup, and, with `--keep-extended-json`,
kept its extended JSON (which is copied). A manifest that cannot be read
(the `-stub-run` one, another layout or schema version) disables reuse
with a warning. Each use of a required bundle is a run: a `per_platform`
pair maps each platform on its own panel plus the intersection (`_xpanel`),
and two uses on the same gene set are mapped once and recorded under both
run ids.

---


## `merxen annotate-resolve`

The annotation RESOLVE step (plan §3.4; M4): turns a MAP output
(`map_manifest.json` and its tidy parquets) into the per-cell label tables,
standalone on published MAP outputs or as the `CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE`
pipeline task (which passes `--prepared-dir`, `--clustering-config`,
`--bundle-ref` per staged ref, `--require-bundle-refs` and
`--no-alignment-lookup`). It reads each sample's counts from the
manifest's inputs (or `--prepared-dir`), checks them against the sample
fingerprint MAP recorded, applies resolvability-gated emission reweighted to
the dataset's soft composition, the floors, the dataset gate, the
degraded-mode consensus and the flags, and writes under `--out` (never into
the inputs' results tree or the MAP output):

| File | Content |
|---|---|
| `<platform>/<sid>_celltype_labels.parquet` | The §4.1 label table (every object; validated with `schema.validate_label_table`); the provenance JSON is also in the parquet schema (`merxen_annotation`). |
| `<platform>/<sid>_annotation_manifest.json` | `AnnotationProvenance` (§4.6): the JSON string `uns["merxen_annotation_json"]` holds. |
| `<pair>_resolve_summary.json` | Per sample: trust (and banner), degraded mode, gate level and warning with reasons, confident share per level of table cells and of segmented objects (the `--n-segmented` count when given, the gate warning's denominator), resolvable share per level, emission, COP control, realised flag rates per class × platform (H16: `rate` over the stratum's confident broad calls, `informative`, `informative_h16`) and the compositions; per pair: the JSD of every composition kind, whole section and shared tissue mask, with its 95% block-bootstrap CI. |

```bash
merxen annotate-resolve \
  --map-dir shadow/P7513/proseg_hybrid \
  --current-bundles --store /media/mathieubo/SSD1/MerXen/annotation_references \
  --gene-id-fallback-csv /path/to/WHB/gene.csv \
  --n-segmented P7513_MERSCOPE=211744 --n-segmented P7513_XENIUM=167738 \
  --out resolve/P7513/proseg_hybrid
```

| Option | Meaning |
|---|---|
| `--map-dir DIR` | The MAP output (`map_manifest.json`, `<platform>/<sid>_mmc_<run_id>.parquet`); every parquet must still have the sha256 the manifest recorded. |
| `--panel-dir DIR` | `annotation-panel` output (default `<map-dir>/panel`): the panel files (trust diagnostics, the flags' query genes), `panel_report.json` and `required_bundles.json`. Each panel's family is re-derived from the current `validated_panels.csv`. |
| `--bundle KEY=DIR` | Resolve a run (`KEY` = run id or reference id) with this bundle; it must be of the run's reference and panel and have the marker lookup the run mapped with. |
| `--current-bundles --store DIR [--store-large DIR]` | Resolve every run with the store's current bundle of its reference and panel (current builder, current resolvability tables), under the same checks. |
| `--bundle-ref PATH` | A `bundle_ref.json` of `annotation-reference-prep` (repeatable): each run is resolved with the bundle of the ref of its reference and panel; a run no ref names keeps its own bundle, and a ref of another build is an override under the same checks. |
| `--require-bundle-refs` | Every run must have a `--bundle-ref` with the `build_hash` it mapped with (else the MAP output is stale and the command fails): no store lookup and no override (`--bundle`, `--current-bundles` are refused), as a pipeline task. |
| `--prepared-dir DIR` | Read the counts from these prepared H5ADs instead of the manifest's inputs (a refused panel, whose manifest lists no sample, needs it). |
| `--clustering-config PATH` | With `--prepared-dir`: the `clustering_squidpy_config.json` MAP read. Its sample platforms are used as MAP used them, and its `min_counts` and `pair_id` must be the MAP manifest's. |
| `--gene-id-fallback-csv PATH` | The gene-ID fallback table MAP used (otherwise the counts do not fingerprint the same). |
| `--n-segmented SID=N` | Segmented objects of a sample: the denominator of the segmented-object gate warning (a published clustered H5AD holds table cells only). |
| `--alignment-dir DIR` | The pair's `align_out` (shared tissue mask); default `<results>/<pair>/alignment/align_out` of the inputs' results tree, when present. |
| `--no-alignment-lookup` | Never take that default: the mask comes only from `--alignment-dir` (a pipeline task gets it from ALIGN's channel, never from a published file ALIGN may still be writing). |
| `--annotation-config PATH`, `--species` | `AnnotationConfig` JSON; the species defaults to the manifest's (mouse RESOLVE is M6: a mouse MAP output fails with a clean error). |
| `--platforms`, `--n-bootstrap`, `--tile-um`, `--seed`, `--results-root` | Resolve only these platforms; block-bootstrap replicates (200), tile edge (500 µm) and seed (0); a results tree `--out` must stay out of. |

The human acceptance criteria (plan §14) are re-measured on these outputs
with `scripts/acceptance/resolve_criteria.py` (see
[the annotation stage](stages/annotation.md#shadow-baselines-m3)); it never
changes a pre-registered threshold.

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
