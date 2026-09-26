# MapMyCells

Runs local Allen Institute MapMyCells annotation on the per-platform AnnData
objects produced by the Squidpy clustering stage. It supports the Allen Whole
Human Brain (WHB) taxonomy and the Yao et al. 2023 Whole Mouse Brain (WMB)
taxonomy, including human-to-mouse ortholog mapping. WHB is the human default;
with `--species mouse`, WMB whole-brain mapping becomes the workflow default.

## What it does

For each platform in a pair:

1. Read `<sample_id>_clustered.h5ad` from `clustering_squidpy_out/<platform>/`.
2. Write a MapMyCells query H5AD with raw counts from `layers["counts"]` copied
   into `X`. MapMyCells expects the query cell-by-gene matrix in `X`.
3. Run MapMyCells locally through `python -m merxen.analysis.mapmycells_entrypoint`
   for the configured reference mode: `whole_brain`, `region`, or `both`.
   The whole-brain path uses configured files or downloads Allen's published
   marker lookup and precomputed-stat assets. The region path builds or reuses
   a strict atlas/ROI-specific reference in the durable MapMyCells cache first.
4. Save the extended JSON, CSV, mapper log, stdout/stderr logs, command
   manifest, query H5AD, standalone UMAP/spatial PNG/PDF plots, and a clustered H5AD
   annotated with MapMyCells assignments in `obs` columns prefixed with
   `mapmycells_`. The H5AD also records MapMyCells metadata in
   `uns["merxen_mapmycells"]`, including the paths to the separate PNGs; the
   plot images themselves are not embedded in the H5AD. The extended JSON and
   the mapper, stdout and stderr logs are not embedded either. `uns` records
   their paths and SHA-256 digests (`extended_json_sha256`, `log_sha256`,
   `stdout_log_sha256`, `stderr_log_sha256`) and keeps only the short command
   JSON as `command_json_text`. Annotated H5ADs written before this change
   still embed these files as `*_text` entries. For the P7513 MERSCOPE
   section, the embedded extended JSON was about 258 MB of a 652 MB file.

Set `--mapmycells_plots_only true` to regenerate the annotated H5AD and plots
from an existing published `mapmycells_out/` directory without preparing a new
query H5AD, rebuilding a region reference, or rerunning MapMyCells. This is
useful after changing plot code. Use it with `--only_stage mapmycells` and the
same `--outdir`, `--mapmycells_reference_mode`, and `--mapmycells_region_name`
used for the original run.

The default `mapmycells_bootstrap_factor` is `0.9` because these data are
spatial transcriptomics panels where the newer single-cell-oriented lower
defaults can be less stable.

For human region mode, `mapmycells_region_labels` contains Allen WHB
`region_of_interest_label` values. The human default is
`["Human A44-A45", "Human A46", "Human A32", "Human ACC"]`, but the
implementation accepts a list, a JSON list, or a comma-separated string so this
can be adjusted to different frontal region sets later.

## Whole Mouse Brain compatibility

For a mouse dataset, the pipeline-level switch is sufficient for whole-brain
mapping:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --species mouse \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_region_cache_dir /durable/mapmycells-cache
```

This resolves to `reference_mode=whole_brain`, `reference_atlas=wmb`, and
`query_species=mouse`. Mouse Ensembl IDs (`ENSMUSG…`) are retained from Xenium
or reference metadata. For symbol-only MERSCOPE panels, MerXen uses cached WMB
expression metadata where available; otherwise it downloads the gene-mapper
database and lets MapMyCells resolve the symbols. The large gene database is
therefore not normally needed when usable mouse Ensembl metadata already
exists.

Set `mapmycells_reference_atlas=wmb` to use the Yao et al. 2023 WMB taxonomy.
For human MerXen queries, also keep `mapmycells_query_species=human`. MerXen then
downloads and caches Allen's full-WMB precomputed stats, mouse marker lookup,
and the `mmc_gene_mapper` ortholog database, and passes the database through
`gene_mapping.db_path`. It also drops `CCN20230722_SUPT` by default, matching
Allen's human-to-WMB example. `cell_type_mapper` 1.7.2 or newer is required.

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode whole_brain \
    --mapmycells_reference_atlas wmb \
    --mapmycells_query_species human \
    --mapmycells_region_cache_dir /durable/mapmycells-cache
```

The automatic full-WMB downloads are approximately 1.4 GB for stats, 14 MB for
markers, and 16.2 GB for the cross-species gene database. Explicit
`mapmycells_marker_lookup_path`, `mapmycells_precomputed_stats_path`, and
`mapmycells_gene_mapping_db_path` values override the downloads.

For a mouse-region-specific WMB reference, select mouse
`region_of_interest_acronym` values such as `MOp` or `VIS`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --species mouse \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode region \
    --mapmycells_reference_atlas wmb \
    --mapmycells_query_species mouse \
    --mapmycells_region_name motor \
    --mapmycells_region_labels MOp \
    --mapmycells_region_cache_dir /durable/mapmycells-cache
```

Region generation downloads WMB metadata and only the raw expression-matrix
shards named by the selected cells' `feature_matrix_label` values. Individual
shards can be several GB, so the cache must have substantial free space.

## Region reference cache

Generated region references are immutable once written: MerXen never
modifies or deletes a completed build in the shared cache. Each build lives in
its own directory
under `<mapmycells_region_cache_dir>/references/`, named
`<prefix>_<region_name>-<hash>`, where `<prefix>` is `region` (WHB) or
`wmb_region` (WMB) and `<hash>` is the first 16 hex digits of the SHA-256 of
the reference configuration: region name and labels,
`region_min_cells_per_leaf`, `region_query_markers_n_per_utility`, atlas,
query species, hierarchy, normalization, Allen manifest URL and `drop_level`.
Each directory holds `precompute/precomputed_stats.h5`, `reference_markers/`,
`query_markers/query_markers.n<N>.json`, `region_cell_metadata.csv` and
`region_reference_manifest.json`, which records the full `config_hash`,
`created_at` and the paths of the build.

For each run, MerXen picks the reference in this order:

1. The newest complete content-hashed build for the requested configuration.
2. A legacy in-place reference, `references/<prefix>_<region_name>/`, written
   by MerXen before content-hashed builds existed. It is adopted **read-only**
   when its stats and query-marker files exist and its recorded `config`
   matches the request. Keys that older versions did not record are compared
   as the values those versions always used: `reference_atlas = "whb"`,
   `query_species = "human"`, `drop_level = null`. The manifest is never
   rewritten; the copy recorded in outputs has `cache_layout =
   "legacy_in_place"` and the resolved on-disk paths, because legacy
   manifests can hold stale absolute paths from before the cache was moved.
3. Otherwise, a new build. It is assembled in a private
   `references/.staging-*` directory and renamed into place only when
   complete, so other tasks never see a partial build. The provenance that
   `cell_type_mapper` writes inside its own outputs (for example
   `metadata.config` in the query-marker JSON) therefore names the staging
   path; `region_reference_manifest.json` names the final paths. If the build
   fails, only that staging directory is removed. A task killed outright (for
   example with `SIGKILL`) can leave a `.staging-*` directory behind. Such a
   directory is never used and can be deleted by hand when no build is
   running.

A configuration change therefore selects a different directory. It never
invalidates, overwrites or deletes an existing build. Builds of one
configuration are serialised with a `references/.<prefix>_<region_name>-<hash>.lock`
file lock, so concurrent `MAPMYCELLS` tasks wait for one build and then reuse
it.

`mapmycells_region_force_rebuild=true` always writes a new
`<prefix>_<region_name>-<hash>-rebuild-<UTC timestamp>` directory, which later
runs then prefer, and it leaves every earlier build intact. A task that waited
for another task's rebuild of the same configuration reuses that build. Each
build uses about 2.4 GB for the frontal WHB reference, and old builds are
never pruned automatically. Force a rebuild on a single pair, then turn it off.

## Nextflow process

[`MAPMYCELLS`](../../workflows/modules/mapmycells.nf) — one instance per
`pair_id`.

- **Input:** `tuple(pair_id, clustering_squidpy_out/)`.
- **CLI:** `merxen mapmycells --config mapmycells_config.json`.
- **Output:** `tuple(pair_id, mapmycells_out/)`.
- **publishDir:** `${outdir}/${pair_id}/mapmycells/` (copy mode).

This stage is opt-in. The default `stop_stage` remains `clustering_squidpy` so
existing runs do not require reference files. Run it with:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode both
```

Use `--only_stage mapmycells` to reuse an existing
`${outdir}/<pair_id>/clustering_squidpy/clustering_squidpy_out/` directory.

## Config Schema

`MapMyCellsConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `mapmycells_out/` is populated. |
| `samples` | One or two sample configs: `sample_id`, `platform`, `anndata_path`, optional `query_layer`, optional `gene_id_column`, optional `obs_id_column`. |
| `reference_mode` | `whole_brain`, `region`, or `both`; workflow default is `both` for human and `whole_brain` for mouse. |
| `reference_atlas` | `whb` or `wmb`; the workflow selects WHB for human and WMB for mouse. |
| `query_species` | `human` or `mouse`; controls whether WMB mapping needs cross-species gene mapping. |
| `auto_download_references` | Download missing Allen stats, markers, and the WMB gene mapper into the durable cache. |
| `marker_lookup_path` | Optional explicit whole-brain JSON marker lookup; otherwise downloaded when enabled. |
| `precomputed_stats_path` | Optional explicit whole-brain HDF5 stats file; otherwise downloaded when enabled. |
| `gene_mapping_db_path` | Optional `mmc_gene_mapper` SQLite database; required for human-to-WMB mapping when automatic downloads are disabled. |
| `region_name` / `region_labels` | Short output name and WHB ROI labels or WMB ROI acronyms used to build the strict region reference. |
| `region_cache_dir` | Durable cache for Allen WHB/WMB downloads and generated region stats/marker files. |
| `region_min_cells_per_leaf` | Minimum ROI cells required for a leaf `cluster_alias` to stay in the region taxonomy. |
| `region_force_rebuild` | Build a new region reference directory even if a matching one exists; earlier builds are kept (see [Region reference cache](#region-reference-cache)). |
| `region_query_markers_n_per_utility` | Marker count target for region `QueryMarkerRunner`. |
| `drop_level` | Optional taxonomy level to drop before mapping, such as the Whole Mouse Brain supertype level. |
| `normalization` | Passed to `type_assignment.normalization`; `raw` means MapMyCells converts query counts internally. |
| `bootstrap_factor` | Marker downsampling factor per bootstrap iteration. Defaults to `0.9` for spatial data. |
| `bootstrap_iteration` | Number of bootstrapping iterations. |
| `n_processors` / `chunk_size` / `rng_seed` | MapMyCells parallelism and reproducibility controls. |
| `max_gb` / `tmp_dir` | Optional mapper memory and temporary storage controls. |
| `cloud_safe` / `flatten` / `verbose_csv` | Direct MapMyCells CLI options. |
| `plots_only` | Reuse existing mapper CSV/extended JSON outputs and regenerate only annotated H5AD + plots. |

When explicit reference paths are configured, workflow preflight validates them
before any tasks start. Automatically downloaded files are checked against the
sizes in Allen's manifest and partial downloads can resume.

The Allen ABC manifest names a fixed release, so MerXen downloads it once per
cache and keeps a copy at `<cache>/abc_manifests/releases/<release>/manifest.json`.
Here `<cache>` is `mapmycells_region_cache_dir`,
`clustering_squidpy_broad_reference_cache_dir` or `mecr_reference_cache_dir`;
on Dwight all three are the same SSD1 directory. The MapMyCells, WMB
clustering-reference and MECR download helpers read that copy. When every file
they need is already cached with the size the manifest records, they use no
network. To prepare an offline cache, download the manifest to that path once.
If the copy is unreadable, it is downloaded again.

## Outputs

Written under `mapmycells_out/<platform>/`:

| Kind | File | Contents |
|------|------|----------|
| Query AnnData | `<sample_id>_mapmycells_query.h5ad` | Mapper input with query counts in `X`. |
| CSV | `<sample_id>_mapmycells.csv` | Per-cell taxonomy assignments and probabilities. |
| Extended JSON | `<sample_id>_mapmycells_extended.json` | Full MapMyCells result, config, logs, marker genes, and taxonomy tree. |
| Log | `<sample_id>_mapmycells.log` | Mapper log output. |
| Stdout log | `<sample_id>_mapmycells_stdout.log` | Captured process stdout, including the exact command line. |
| Stderr log | `<sample_id>_mapmycells_stderr.log` | Captured process stderr for startup/import errors and mapper tracebacks. |
| Command manifest | `<sample_id>_mapmycells_command.json` | Exact command used for the local mapper call. |
| UMAP plot | `<sample_id>_mapmycells_umap.png` | Existing Squidpy/Scanpy UMAP coordinates colored by MapMyCells assignment. |
| UMAP cluster-by-supercluster plots | `<sample_id>_mapmycells_umap_cluster_by_supercluster/supercluster_<name>.png` | Per-supercluster UMAPs with cells outside the supercluster in grey and member cells colored by MapMyCells cluster. |
| Spatial plot | `<sample_id>_mapmycells_spatial.png` | Spatial coordinates colored by MapMyCells assignment. |
| Quality scatter | `<sample_id>_mapmycells_quality_scatter.png` | Extended-JSON QC panels for supercluster and cluster assignments: cell complexity vs average correlation/bootstrap probability, correlation vs bootstrap probability, aggregate probability, and runner-up margin. |
| Supercluster QC | `<sample_id>_mapmycells_supercluster_assignment_qc.png` | Supercluster cell counts, confidence summaries, and low-confidence fractions. |
| Cluster QC | `<sample_id>_mapmycells_cluster_assignment_qc.png` | Cluster cell counts, confidence summaries, and low-confidence fractions. |
| Supercluster spatial grid | `<sample_id>_mapmycells_spatial_supercluster_grid.png` | Small-multiple spatial grid with each supercluster highlighted in red against all other cells in grey. |
| Annotated AnnData | `<sample_id>_mapmycells_annotated.h5ad` | Clustered AnnData with MapMyCells assignments added to `obs` and mapper metadata in `uns["merxen_mapmycells"]` (paths and SHA-256 digests of the extended JSON and logs, not their contents). |

Each listed `.png` plot is also written as a same-stem `.pdf`.

Region-specific outputs use the same file names under
`mapmycells_out/region_<mapmycells_region_name>/<platform>/`. Their annotated
H5AD columns use the prefix `mapmycells_region_<region_name>_`, and metadata is
stored in `uns["merxen_mapmycells_region_<region_name>"]`.

Full-WMB outputs are written under `mapmycells_out/wmb/<platform>/` with the
`mapmycells_wmb_` column prefix. WMB region outputs are written under
`mapmycells_out/wmb_region_<region_name>/<platform>/` with the
`mapmycells_wmb_region_<region_name>_` prefix, keeping them distinct from WHB
annotations.

The stage also writes `<pair_id>_mapmycells_manifest.json` at the top of
`mapmycells_out/`, including whole-brain and region reference paths, ROI labels,
filtering counts, and per-sample outputs. It also records the installed
`cell_type_mapper` distribution as `cell_type_mapper_version` and, for a git
install, `cell_type_mapper_commit`. The mapper subprocess uses the same Python
interpreter, so this is the version that mapped the cells. In plots-only mode
it is the version that regenerated the plots. Each newly built region
reference records the same two fields in its `region_reference_manifest.json`.
