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
  enrich             Enrich a segmented zarr with per-shape tables
  qc                 Compute geometry and assignment QC metrics
  align              Align MERSCOPE into paired Xenium coordinates
  alignment-qc       Compute post-alignment QC metrics
  compare            Run cross-platform gene-level comparison
  visualize          Generate visualization artifacts for a pair
  clustering-squidpy Run Scanpy/Squidpy clustering analysis
  mapmycells         Run local MapMyCells cell type assignment
```

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

## `merxen align`

Align a MERSCOPE section into paired Xenium coordinates.

```bash
merxen align --config align_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `AlignmentConfig`. |

This command requires the optional alignment dependencies:

```bash
pip install spateo-release==1.1.1
pip install "anndata>=0.12.10"
```

Details: [Section alignment](stages/alignment.md).

---

## `merxen alignment-qc`

Compute SABench-style grid metrics and a centroid overlay for an aligned pair.

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

Generate visualization artifacts for a paired dataset comparison.

```bash
merxen visualize --config visualize_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against [`VisualizationConfig`](../src/merxen/config.py#L186). |

Details: [Stage 6 — Visualization](stages/visualization.md).

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
