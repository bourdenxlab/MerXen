# MapMyCells

Runs local Allen Institute MapMyCells annotation on the per-platform AnnData
objects produced by the Squidpy clustering stage.

## What it does

For each platform in a pair:

1. Read `<sample_id>_clustered.h5ad` from `clustering_squidpy_out/<platform>/`.
2. Write a MapMyCells query H5AD with raw counts from `layers["counts"]` copied
   into `X`. MapMyCells expects the query cell-by-gene matrix in `X`.
3. Run `python -m cell_type_mapper.cli.from_specified_markers` locally with the
   configured marker lookup JSON and precomputed stats HDF5 reference files.
4. Save the extended JSON, CSV, mapper log, command manifest, query H5AD, and a
   clustered H5AD annotated with the CSV assignments in `obs` columns prefixed
   with `mapmycells_`.

The default `mapmycells_bootstrap_factor` is `0.9` because these data are
spatial transcriptomics panels where the newer single-cell-oriented lower
defaults can be less stable.

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
    --mapmycells_marker_lookup_path /path/to/query_markers.json \
    --mapmycells_precomputed_stats_path /path/to/precomputed_stats.h5
```

Use `--only_stage mapmycells` to reuse an existing
`${outdir}/<pair_id>/clustering_squidpy/clustering_squidpy_out/` directory.

## Config Schema

`MapMyCellsConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `mapmycells_out/` is populated. |
| `samples` | MERSCOPE and Xenium sample configs: `sample_id`, `platform`, `anndata_path`, optional `query_layer`, optional `gene_id_column`, optional `obs_id_column`. |
| `marker_lookup_path` | JSON marker lookup file downloaded or generated for the reference taxonomy. |
| `precomputed_stats_path` | HDF5 precomputed stats file for the reference taxonomy. |
| `drop_level` | Optional taxonomy level to drop before mapping, such as the Whole Mouse Brain supertype level. |
| `normalization` | Passed to `type_assignment.normalization`; `raw` means MapMyCells converts query counts internally. |
| `bootstrap_factor` | Marker downsampling factor per bootstrap iteration. Defaults to `0.9` for spatial data. |
| `bootstrap_iteration` | Number of bootstrapping iterations. |
| `n_processors` / `chunk_size` / `rng_seed` | MapMyCells parallelism and reproducibility controls. |
| `max_gb` / `tmp_dir` | Optional mapper memory and temporary storage controls. |
| `cloud_safe` / `flatten` / `verbose_csv` | Direct MapMyCells CLI options. |

## Outputs

Written under `mapmycells_out/<platform>/`:

| Kind | File | Contents |
|------|------|----------|
| Query AnnData | `<sample_id>_mapmycells_query.h5ad` | Mapper input with query counts in `X`. |
| CSV | `<sample_id>_mapmycells.csv` | Per-cell taxonomy assignments and probabilities. |
| Extended JSON | `<sample_id>_mapmycells_extended.json` | Full MapMyCells result, config, logs, marker genes, and taxonomy tree. |
| Log | `<sample_id>_mapmycells.log` | Mapper log output. |
| Command manifest | `<sample_id>_mapmycells_command.json` | Exact command used for the local mapper call. |
| Annotated AnnData | `<sample_id>_mapmycells_annotated.h5ad` | Clustered AnnData with MapMyCells assignments added to `obs`. |

The stage also writes `<pair_id>_mapmycells_manifest.json` at the top of
`mapmycells_out/`.
