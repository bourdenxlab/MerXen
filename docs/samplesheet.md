# Samplesheet format

The samplesheet is a CSV with one row per biological sample or adjacent-section
pair. By default, rows inherit `--analysis_mode`, `--enable_alignment`,
`--analysis_segmentation`, `--start_stage`, `--stop_stage`, and `--only_stage`
from the Nextflow command or config, but each row can override those settings
with optional columns. In the default `analysis_mode=paired`, a row must contain
one MERSCOPE and one Xenium dataset. In `analysis_mode=merscope` or
`analysis_mode=xenium`, only the selected platform's source/cache columns are
required. A template lives at
[workflows/samplesheet.example.csv](../workflows/samplesheet.example.csv).

## Columns

| Column | Required | Description |
|--------|----------|-------------|
| `pair_id` | **yes** | Unique identifier for this row. Used as the top-level output directory name. |
| `analysis_mode` | no | Row-level mode: `paired`, `merscope`, or `xenium`. Blank inherits `--analysis_mode`. |
| `enable_alignment` | no | Row-level alignment switch: `true` or `false`. Blank inherits `--enable_alignment`; only paired rows can run alignment. |
| `analysis_segmentation` | no | Row-level downstream branch set: `both`, `reseg`, `original_seg`, or comma-separated combinations. Blank inherits `--analysis_segmentation`. |
| `start_stage` | no | Row-level first stage. Blank inherits `--start_stage` unless `only_stage` applies. |
| `stop_stage` | no | Row-level final stage. Blank inherits `--stop_stage` unless `only_stage` applies. |
| `only_stage` | no | Row-level single-stage override. If set, it overrides that row's start/stop stage settings. |
| `cortical_depth_enabled` | no | Row-level cortical-depth switch. Blank inherits `--cortical_depth_enabled`. |
| `merscope_dir` | required for MERSCOPE modes if no cache | Path to the raw MERSCOPE region export folder (contains `transcripts.parquet`, `cell_boundaries/`, `images/`, etc.). |
| `merscope_spatialdata_path` | required for MERSCOPE modes if no raw dir | Path to an existing (or desired) reusable MERSCOPE SpatialData zarr. If it exists, the build step is **skipped** unless `--force_spatialdata_build true` is passed to Nextflow. |
| `merscope_image_prefix` | no | Prefix used to match z-plane image keys when more than one run is present. |
| `merscope_z_range` | no | Inclusive z-layer range as `start-end`. Defaults to `0-6`. |
| `merscope_transform_path` | no | Override path to the `micron_to_mosaic_pixel_transform.csv`. If not set, MerXen looks inside `merscope_dir`. |
| `merscope_channels` | no | Comma-separated channel names for Cellpose. Defaults to `DAPI,PolyT`. |
| `merscope_voxel_layers` | no | ProSeg voxel layer count for MERSCOPE. Defaults to `7` (from `nextflow.config`). |
| `xenium_dir` | required for Xenium modes if no cache | Path to the raw Xenium export folder. |
| `xenium_spatialdata_path` | required for Xenium modes if no raw dir | Path to an existing (or desired) reusable Xenium SpatialData zarr. Cached the same way as the MERSCOPE path. |
| `xenium_channels` | no | Comma-separated channel names for Cellpose. Defaults to `DAPI,18S`. |
| `xenium_min_qv` | no | Minimum transcript quality value to retain. Defaults to `20`. |
| `xenium_voxel_layers` | no | ProSeg voxel layer count for Xenium. Defaults to `2`. |
| `xenium_spec_path` | no | Override path to `experiment.xenium` or `specs.json` used to derive the micron→pixel transform. |

### Cortical-depth annotation columns

When cortical depth is enabled, each active platform must provide either a
combined role-labelled annotation GeoJSON or separate pial boundary GeoJSON
files. Gray/white matter boundaries are optional for pial-only mask/QC pieces.
Platform-specific columns are preferred; generic
columns are accepted when one row uses a single active platform or the same
annotation should be reused.

| Column pattern | Description |
|----------------|-------------|
| `<platform>_cortical_depth_annotation_geojson` | Combined GeoJSON with role-labelled pial, tissue-edge, optional WM, and optional mask features. `<platform>` is `merscope` or `xenium`. Generic alias: `cortical_depth_annotation_geojson`. |
| `<platform>_pial_boundary_geojson` | Pial boundary polyline. Generic alias: `pial_boundary_geojson`. |
| `<platform>_wm_boundary_geojson` | Optional gray/white matter boundary polyline. Aliases include `grey_white_boundary_geojson`, `gray_white_boundary_geojson`, and `gm_wm_boundary_geojson`. |
| `<platform>_side_boundaries_geojson` | Tissue-edge polyline. New piece-aware annotations should contain exactly one edge line. Generic alias: `side_boundaries_geojson`. |
| `<platform>_exclusion_masks_geojson` | Optional exclusion polygons for tears, folds, vessels, or artefacts. Generic alias: `exclusion_masks_geojson`. |
| `<platform>_cortical_ribbon_geojson` | Optional complete ribbon polygon. Generic alias: `cortical_ribbon_geojson`. |

### Aliases

`merscope_zarr_path` is accepted as an alias for `merscope_spatialdata_path`
for backwards compatibility.

## Validation rules

From [workflows/main.nf](../workflows/main.nf):

- Every row must have a non-empty `pair_id`.
- In `analysis_mode=paired`, the row must provide **either**
  `merscope_dir` **or** `merscope_spatialdata_path`, and **either**
  `xenium_dir` **or** `xenium_spatialdata_path`.
- In `analysis_mode=merscope`, only MERSCOPE source/cache columns are
  required.
- In `analysis_mode=xenium`, only Xenium source/cache columns are required.
- Blank row-level analysis, alignment, or stage settings inherit the matching Nextflow
  parameter. Row-level `only_stage` overrides row-level `start_stage` and
  `stop_stage`; when a row sets either start/stop column, the global
  `--only_stage` fallback is ignored for that row.

The Python-side parser lives in
[src/merxen/io/samplesheet.py:52](../src/merxen/io/samplesheet.py#L52). It is
used by unit tests and can be invoked directly by scripts, but the Nextflow
workflow parses the CSV itself.

## Minimal example

Paired mode:

```csv
pair_id,analysis_mode,enable_alignment,merscope_dir,xenium_dir
EXAMPLE01,paired,true,/path/to/merscope/EXAMPLE01/region_R1,/path/to/xenium/EXAMPLE01
```

This is enough to run the whole pipeline end-to-end with default parameters.
All other columns fall back to defaults.

Xenium-only mode:

```csv
pair_id,analysis_mode,enable_alignment,xenium_dir,xenium_spatialdata_path,xenium_channels,xenium_min_qv
EXAMPLE01,xenium,false,/path/to/xenium/EXAMPLE01,/path/to/cache/EXAMPLE01_xenium.zarr,"DAPI,18S",20
```

The row-level `analysis_mode` is enough to run this alongside paired or
MERSCOPE-only rows in the same file. MERSCOPE-only rows are analogous: provide
`merscope_dir` or `merscope_spatialdata_path` and set `analysis_mode=merscope`.

Cortical-depth example for a Xenium-only row:

```csv
pair_id,analysis_mode,cortical_depth_enabled,xenium_spatialdata_path,xenium_pial_boundary_geojson,xenium_wm_boundary_geojson,xenium_exclusion_masks_geojson
EXAMPLE04,xenium,true,/path/to/cache/EXAMPLE04_xenium.zarr,/path/to/EXAMPLE04_pia.geojson,/path/to/EXAMPLE04_wm.geojson,/path/to/EXAMPLE04_exclusions.geojson
```

## Full example

```csv
pair_id,analysis_mode,enable_alignment,analysis_segmentation,start_stage,stop_stage,only_stage,merscope_dir,merscope_spatialdata_path,merscope_image_prefix,merscope_z_range,merscope_transform_path,merscope_channels,xenium_dir,xenium_spatialdata_path,xenium_channels,xenium_min_qv,merscope_voxel_layers,xenium_voxel_layers,xenium_spec_path
EXAMPLE01,paired,true,both,,,,/path/to/merscope/EXAMPLE01,/path/to/cache/EXAMPLE01_merscope.zarr,,0-6,,"DAPI,PolyT",/path/to/xenium/EXAMPLE01,/path/to/cache/EXAMPLE01_xenium.zarr,"DAPI,18S",20,7,2,
EXAMPLE02,merscope,false,reseg,segment,enrich,,/path/to/merscope/EXAMPLE02,,,,,"DAPI,PolyT",,,,,7,,
EXAMPLE03,xenium,false,original_seg,,,visualize,,,,,,,/path/to/xenium/EXAMPLE03,,"DAPI,18S",20,,2,
```

Row 1 uses cached SpatialData zarrs if they already exist. Row 2 runs only
MERSCOPE from segmentation through enrichment. Row 3 runs only the Xenium
visualization stage, reading prior outputs from `--outdir`.

## Tips

- Quote any list-valued field that contains commas (`merscope_channels`,
  `xenium_channels`).
- Leave a field empty (`,,`) to fall back to the Nextflow default.
- Prefer absolute paths. Relative paths are resolved from the shell that ran
  `nextflow`, not from the work directory.
- Keep one samplesheet per batch. Nextflow runs all rows in parallel up to the
  executor limit.
