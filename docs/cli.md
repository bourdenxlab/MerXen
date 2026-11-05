# CLI reference

The `merxen` CLI is a Click command group registered by `pyproject.toml`
([line 44](../pyproject.toml#L44)) and defined at
[src/merxen/cli/__init__.py](../src/merxen/cli/__init__.py). Most subcommands
correspond to exactly one Nextflow process, take a single `--config
<path>.json` argument, and validate it against the matching Pydantic model
in [src/merxen/config.py](../src/merxen/config.py) (the manual-registration
models are in
[src/merxen/manual_registration/config.py](../src/merxen/manual_registration/config.py));
`convert-czi` and `manual-registration-fit` are ad-hoc tools without a process.

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
  register-images    Register OME-TIFF images with Xenium Explorer matrices
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
  convert-czi        Convert one scene of a Zeiss CZI to a pyramidal OME-TIFF
  manual-registration-prepare
                     Convert/validate both images, build the prepared store
  manual-registration-register
                     Register the moving image from the saved landmarks
  manual-registration-fit
                     Fit the saved landmarks and print what register would flag
  manual-registration-compare
                     Compare several manual-registration runs of one pair
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

## `merxen register-images`

Register external OME-TIFF images (e.g. post-Xenium IF) onto the reference
image grid using Xenium Explorer matrices, refine the affine on DAPI, and write
the resampled images into the SpatialData zarr.

```bash
merxen register-images --config register_images_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ImageRegistrationConfig`. |
| `--force-rerun` | Re-register images even when an identical registration is recorded. |

Details: [Image registration](stages/register-images.md).

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

## `merxen build-viewer-caches`

Materialize label masks, outlines and image pyramids into a SpatialData zarr for
the viewer.

```bash
merxen build-viewer-caches --config viewer_cache_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ViewerCacheConfig`. |
| `--force-rerun` | Rebuild caches even when an up-to-date marker is present. |

A config with `image_only: true` (and optional `image_keys`) builds only the image
pyramids of those elements and needs neither `platform` nor `original_data_path`;
the [manual-registration prepare workflow](stages/manual-registration.md) uses it
for prepared stores.

---

## `merxen convert-czi`

Convert one scene of a Zeiss CZI to a tiled, pyramidal OME-TIFF. The CZI is read
by the isolated CZI runtime (`MERXEN_CZI_PYTHON`, see
[Development workflow](development.md#dependency-management)). It is an
ad-hoc tool: prepare converts CZI rows itself.

```bash
merxen convert-czi --input slide.czi --output slide.ome.tif [--scene S] [--channel-names "DAPI;AF568"]
```

| Option | Description |
|--------|-------------|
| `--input FILE` / `--output FILE` | The CZI and the `.ome.tif` / `.ome.tiff` to write. |
| `--scene TEXT` | Scene index or name; required when the CZI has several. |
| `--channel-names TEXT` | Override all channel names, `;`-separated, in C order. |
| `--pixel-size-um FLOAT` | Pixel size in µm (default: the CZI's `Scaling` metadata). |
| `--image-name TEXT` | OME image name (default `image`). |
| `--workers INT` | Reader/compression threads (default `$DASK_NUM_WORKERS` or `min(8, CPUs)`). |
| `--tile-size INT` / `--compression {zlib,zstd,none}` | Output tiling (default 1024) and compression (default `zlib`). |
| `--summary-json FILE` | Write a JSON summary of the conversion. |
| `--czi-python FILE` | Python of the CZI runtime (default `$MERXEN_CZI_PYTHON`). |

---

## `merxen manual-registration-prepare`

Convert or validate both images of one pair and build the image-only prepared store
the viewer labels.

```bash
merxen manual-registration-prepare --config prepare_config.json [--force]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ManualRegistrationPrepareConfig`. |
| `--force` | Rebuild even when the image grid changed under existing landmarks; the old store is kept as `<name>.superseded-<UTC>`. The landmarks are carried over only when both images keep their shape and pixel size; otherwise they stay in the superseded store and the new one must be labelled again. |

Details: [Manual landmark registration](stages/manual-registration.md).

---

## `merxen manual-registration-register`

Register the moving image onto the reference from the landmarks saved in the
prepared store (one preset).

```bash
merxen manual-registration-register --config register_config.json [--force-rerun]
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ManualRegistrationRunConfig`. |
| `--force-rerun` | Re-register even when an identical result exists. |

Details: [Manual landmark registration](stages/manual-registration.md).

---

## `merxen manual-registration-fit`

Fit the saved landmarks and print what register would flag (outliers, scale, reflection
check, spread). Reads only; it takes seconds and is the authoritative check of the
landmarks before a long register run.

```bash
merxen manual-registration-fit --store prepared.zarr --preset serial_section [--settings "k=v;preset:k=v"] [--json fit.json]
```

| Option | Description |
|--------|-------------|
| `--store DIR` | Prepared store holding the labelling. |
| `--preset {same_section,serial_section}` | Preset whose landmark settings apply. |
| `--settings TEXT` | Overrides, as the samplesheet's `registration_settings`. |
| `--json FILE` | Also write the fit as JSON. |

Details: [Manual landmark registration](stages/manual-registration.md).

---

## `merxen manual-registration-compare`

Compare several manual-registration runs of one pair (comparison CSV, JSON and
figure).

```bash
merxen manual-registration-compare --config compare_config.json
```

| Option | Description |
|--------|-------------|
| `--config PATH` | JSON validated against `ManualRegistrationCompareConfig`. |

Details: [Manual landmark registration](stages/manual-registration.md#reading-the-qc).

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
