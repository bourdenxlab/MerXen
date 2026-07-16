# Distance from object

`distance_from_object` is an opt-in terminal analysis stage for registered
polygon objects such as amyloid plaques or tau tangles. It annotates cell
centroids with distance to the nearest polygon edge, creates grey-matter-only
near/far pseudobulks, and runs paired PyDESeq2 across tissue blocks.

No image registration is performed here. Object GeoJSON and cell coordinates
must already share the same registered coordinate system.

## Required inputs

Each active platform needs:

- a GeoJSON containing one Polygon or MultiPolygon feature per object;
- `object_type` (the user-provided class/name) and a unique `object_id` on each
  feature; and
- `cortical_depth_annotation` in each selected cell table, normally written by
  `compute_cortical_depth` with the values `grey_matter`, `white_matter`, and
  `outside_brain`.

The napari comparison viewer can create, validate, export, and reload the
object GeoJSON. Use the samplesheet columns
`merscope_distance_object_annotation_geojson` and
`xenium_distance_object_annotation_geojson`. The generic
`distance_object_annotation_geojson` alias is useful for single-platform rows.

When cortical depth and object distance are enabled in the same run, MerXen
automatically writes tissue annotations to all distance-analysis tables before
calculating distances. For a stage-only rerun, the selected tables must already
contain `cortical_depth_annotation`.

## Segmentation branches

The default is all three requested branches:

| Config name | SpatialData table | Shape layer |
|-------------|-------------------|-------------|
| `reseg` | `table_MOSAIK_proseg` | `MOSAIK_proseg` |
| `original_seg` | `table_original` | vendor cell boundaries |
| `proseg_mask` | `table_MOSAIK_cellpose` | `MOSAIK_cellpose` |

`proseg_mask` is the existing Cellpose mask-derived cell table used as the
initial mask input to ProSeg; it is exposed under this analysis-specific name
without changing the rest of MerXen's segmentation naming.

## Distance and proximity rules

For every cell centroid, MerXen finds the nearest object polygon boundary.
It records both unsigned edge distance and signed distance (negative inside a
polygon), the nearest object ID/type, nearest boundary point, inside/outside
status, and a QC flag.

Default bins preserve the source analysis:

| Label | Rule |
|-------|------|
| `near` | inside an object, or edge distance `< 50 µm` |
| `middle` | `50 µm <= distance < 100 µm` |
| `far` | `100 µm <= distance <= 200 µm` |
| `beyond_max` | distance `> 200 µm` |

All bins remain in per-cell metadata. Only `grey_matter` cells labelled
`near` or `far` enter pseudobulk differential expression; middle,
beyond-range, white-matter, and outside-brain cells are excluded from DE.

## Paired pseudobulk design

Raw gene counts are summed separately for near and far cells within each
`pair_id`. Here `pair_id` is the biological tissue-block identifier and is the
paired blocking factor, not a technical replicate ID.

MERSCOPE and Xenium are analysed separately. Within each platform and
segmentation branch, only blocks having both a near and a far pseudobulk are
retained. PyDESeq2 uses:

```text
design:   ~ pair_id + proximity
contrast: proximity, near, far
```

No reclustering is performed and existing table annotations are preserved.

## Running

Run cortical annotation and object distance together:

```bash
nextflow run workflows/main.nf \
    --samplesheet samples.csv \
    --cortical_depth_enabled true \
    --distance_from_object_enabled true \
    --outdir ./results
```

Or rerun only object distance against previously annotated durable zarrs:

```bash
nextflow run workflows/main.nf \
    --samplesheet samples.csv \
    --distance_from_object_enabled true \
    --only_stage distance_from_object \
    --outdir ./results
```

## Outputs

Per pair and platform, `distance_from_object_out/` contains a normalized copy
of the registered object GeoJSON and, for each segmentation, a per-cell
Parquet sidecar, near/far pseudobulk H5AD and sample CSV, spatial QC plots, and
a JSON summary. When table writing is enabled, the distance columns are also
added to the existing SpatialData tables in place.

After all pair-level tasks finish, each platform gets a cohort directory with
paired pseudobulk counts, sample metadata, PyDESeq2 CSV/Parquet results,
PNG/PDF volcano plots, and a summary recording completed or skipped branches.
Branches with fewer than `distance_from_object_min_pairs` complete blocks are
reported as skipped rather than producing invalid statistics.
