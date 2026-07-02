# Cortical Depth

Computes 2D cortical-depth coordinates for human cortex Xenium/MERSCOPE
sections from user-annotated pial, tissue-edge, and optional gray/white matter
boundaries.

The stage solves a 2D Laplace equation inside a cortical ribbon, traces
gradient streamlines from pia to white matter, and annotates each selected
cell table with both raw Laplace depth and a 2D equal-area approximation to
equivolumetric depth. It is opt-in because it requires per-sample boundary
annotations.

## Inputs

The stage consumes the current per-platform `latest_spatialdata.zarr` and
GeoJSON annotations saved from napari or another tool in the same coordinate
system as the SpatialData table/shape centroids.

Required, either as separate files or as role-labelled features in one combined
GeoJSON:

| Annotation | Meaning |
|------------|---------|
| Pial boundary polyline | Depth 0 boundary for a depth piece, or the surface boundary for a mask/QC-only piece. |
| Tissue-edge polyline | One global tissue edge. It may be U-shaped or closed/box-like and is used to construct piece masks when no explicit ribbon is supplied. |

Optional:

| Annotation | Meaning |
|------------|---------|
| Gray/white matter boundary polyline | Depth 1 boundary for a tissue piece. Pieces without WM are kept as `mask_qc_only` and do not receive Laplace/equivolumetric depth. |
| Exclusion polygons | Tears, folds, blood vessels, or artefacts removed from the ribbon. |
| Cortical ribbon polygon | Complete ribbon mask for a piece. Use this when edge + pia + optional WM is ambiguous, especially for pial-only pieces. |

For combined GeoJSON files, feature properties such as `role`, `type`, `name`,
`label`, `boundary`, or `classification.name` are matched against aliases like
`pial_boundary`, `grey_white_boundary`, `side_boundary`, `exclusion`, and
`cortical_ribbon`. Use `tissue_piece_id` to group pia, optional WM, exclusions,
and ribbon polygons for independent tissue pieces.

## Running

Enable the stage and stop there:

```bash
nextflow run workflows/main.nf \
  --samplesheet samples.csv \
  --outdir results \
  --cortical_depth_enabled true \
  --stop_stage compute_cortical_depth
```

Or run only this stage from already-published upstream outputs:

```bash
nextflow run workflows/main.nf \
  --samplesheet samples.csv \
  --outdir results \
  --cortical_depth_enabled true \
  --only_stage compute_cortical_depth
```

The samplesheet must include platform-specific annotation columns such as
`xenium_pial_boundary_geojson` and `xenium_wm_boundary_geojson`, or generic
columns such as `pial_boundary_geojson` / `wm_boundary_geojson`.

## Method

1. Build clean 2D cortical piece polygons from the tissue edge, pial boundary,
   optional WM boundary, optional ribbon polygon, and exclusion masks.
2. Rasterize the ribbon at `--cortical_depth_raster_resolution_um`.
3. For pieces with WM, solve `del^2 phi = 0` with `phi=0` at pia and `phi=1`
   at gray/white matter. Pial-only pieces are retained as mask/QC-only regions.
   Non-Dirichlet mask edges are handled as zero-normal-gradient boundaries.
4. Trace normalized-gradient streamlines from the pial boundary to the WM
   boundary.
5. Assign cells by bilinear interpolation of the Laplace field and nearest
   streamline lookup.
6. Compute `equivolumetric_depth` as a per-column equal-area percentile:
   ribbon pixels are assigned to nearest streamlines, then each column strip is
   converted from Laplace depth to cumulative area fraction.

## Output Columns

Added to every selected AnnData table:

| Column | Meaning |
|--------|---------|
| `inside_cortical_ribbon` | Cell centroid falls inside the rasterized cortical ribbon. |
| `laplace_depth` | Bilinear interpolation of the Laplace scalar field, pia=0 and WM=1. |
| `equivolumetric_depth` | 2D equal-area depth within the nearest streamline column. Preferred for layer-relevant comparisons. |
| `distance_to_pia_um` | Distance along nearest streamline from pia to the nearest streamline sample. |
| `distance_to_wm_um` | Remaining streamline distance to the WM boundary. |
| `streamline_thickness_um` | Total nearest-streamline length. |
| `tangential_position_um` | Pial arc-length coordinate of the nearest streamline seed. |
| `nearest_streamline_id` / `column_id` | Nearest streamline/column identifier. |
| `cortical_depth_qc_flag` | `assigned`, `outside_ribbon`, `near_side_boundary`, `no_laplace_depth`, or `no_streamline`. |

## Outputs

Published under
`${outdir}/<pair_id>/<platform>/compute_cortical_depth/`:

| File | Contents |
|------|----------|
| `compute_cortical_depth_out/cortical_ribbon_mask.tif` | Raster ribbon mask. |
| `compute_cortical_depth_out/streamlines.geojson` | Streamlines as GeoJSON LineStrings. |
| `compute_cortical_depth_out/streamlines.parquet` | Point-level streamline table. |
| `compute_cortical_depth_out/depth_contours.geojson` | Laplace depth contours. |
| `compute_cortical_depth_out/equivolumetric_depth_contours.geojson` | Equal-area depth contours. |
| `compute_cortical_depth_out/<segmentation>/*_cells_with_cortical_depth.parquet` | Per-cell depth sidecar for each selected segmentation branch. |
| `compute_cortical_depth_out/*_cortical_depth_overlay.png` | Ribbon, boundaries, contours, and streamlines. PDF copy is also written. |
| `compute_cortical_depth_out/<segmentation>/*_cells_laplace_depth.png` | Cells colored by Laplace depth. PDF copy is also written. |
| `compute_cortical_depth_out/<segmentation>/*_cells_equivolumetric_depth.png` | Cells colored by equal-area depth. PDF copy is also written. |
| `compute_cortical_depth_out/cortical_depth_qc_summary.json` | Cell counts, streamline thickness stats, failed/flagged streamlines, warnings. |

The stage updates the source `latest_spatialdata.zarr` in place by default.
Set `--cortical_depth_write_spatialdata_table false` to write sidecars and QC
without replacing SpatialData tables.

## Interpreting Depths

`laplace_depth` is the harmonic coordinate between pia and white matter. It is
excellent for defining smooth streamlines and a stable inside-ribbon coordinate,
but equal Laplace intervals should not be interpreted as literal histological
layer boundaries.

`equivolumetric_depth` applies a 2D equal-area correction inspired by Bok's
principle: neighboring streamline strips preserve cumulative area while local
thickness changes with curvature. Use it preferentially for cross-sample and
layer-relevant analyses.

## Limitations

- This stage does not segment histological cortical layers.
- The equivolumetric output is a 2D equal-area approximation, not true 3D
  volumetric depth.
- Boundary quality dominates result quality. Check QC overlays for flipped,
  incomplete, self-crossing, or poorly aligned annotations.
- Cells near artificial side boundaries are flagged because their streamlines
  may be influenced by the manually closed ribbon.
