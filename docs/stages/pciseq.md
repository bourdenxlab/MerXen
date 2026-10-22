# pciSeq cell typing

The optional `pciseq` stage runs immediately after `clustering_squidpy` for each
active platform and selected analysis segmentation. It uses the public
[`pciSeq.fit` API](https://github.com/acycliq/pciSeq_3d) to estimate cell types
and transcript-to-cell assignments jointly, from RNA spots, segmentation masks,
and a supplied single-cell reference. It does not use Leiden labels as training
labels. Existing clustering and raw expression counts remain in the annotated
output.

## Run it

Enable the stage and supply a reference expression matrix:

```bash
nextflow run workflows/main.nf -profile conda \
    --samplesheet workflows/samplesheet.csv \
    --pciseq_enabled true \
    --pciseq_reference_path references/mean_expression.csv
```

Enabling pciSeq extends the historical default stop from clustering to pciSeq.
Later enabled terminal analyses retain their normal stop precedence. Explicit
single-stage selections and other stop stages remain exact. To run from
published clustering outputs:

```bash
nextflow run workflows/main.nf -profile conda \
    --samplesheet workflows/samplesheet.csv \
    --only_stage pciseq \
    --pciseq_enabled true \
    --pciseq_reference_path references/mean_expression.csv
```

`pci_seq` is an accepted stage alias. Samplesheets may override
`pciseq_enabled` and `pciseq_reference_path` per row; blank cells inherit the
global setting. `analysis_segmentation` determines the branches, with all four
selected by default. A published-input run checks the exact clustered H5AD,
latest SpatialData, and native segmentation layers before starting tasks.

The reference is a CSV or TSV with gene identifiers in the first column and
cell types in the remaining columns:

```text
gene,Excitatory,Inhibitory,Astrocyte
SLC17A7,20,0.1,0
GAD1,0.1,18,0
AQP4,0,0,15
```

Values must be finite, nonnegative **mean expression**, with unique genes and
unique cell-type names. Match reference gene identifiers to the native
transcript gene names, including case. Supply raw mean expression rather than
log-transformed expression; MerXen cannot infer the reference's normalization
from its values. Every class needs positive expression in the retained panel.
`Zero` is reserved by pciSeq for its low-expression class. Atlas marker-lookup
JSONs and MapMyCells precomputed statistics are different formats and cannot be
used directly as this matrix. Choose a biologically appropriate reference for
the sample's species and tissue.

## Geometry and identifiers

The default `project_2d` mode uses MerXen's native cell polygons and measured
transcript XY positions. It transforms both into `pciseq_coordinate_system`,
rasterizes the polygons at `pciseq_pixel_size_um`, and explicitly records a 2D
projection. It does not infer a 3D mask from polygons. Aligned/nonrigid shapes
used for clustering plots are not used as native segmentation seeds.

Input cells and spots are independent of clustering's cell/gene QC subset.
Extracellular spots with supported genes are retained for probabilistic
assignment. Preparation reports exclusions for controls, invalid coordinates,
unsupported genes, and out-of-bounds spots. Native hybrid segmentation permits
overlapping cell polygons. By default, raster pixels inside multiple overlapping
polygons become background seeds so pciSeq can estimate their parent cells
probabilistically. Source polygons and assignments remain intact, and the
manifest records this adaptation and its pixel count. Set
`pciseq_overlap_policy=error` to reject overlapping polygons. Shared boundaries
between polygons that only touch use a deterministic cell-ID tie break.

Preparation rejects unusable geometries, cells that lose every seed pixel,
and grids above the pixel budget. Upstream densifies segmentation planes during
geometry extraction, so sparse input files do not remove the raster memory cost.

Canonical cell IDs are mapped reversibly to dense mask labels. Native transcript
IDs are retained through filtering. Finalization joins by these mappings,
including for nonconsecutive uint64 cell IDs; it does not rely on row ordering.
The derived table covers the clustered cell subset. Its original counts,
clustering annotations, and geometry linkage are retained.

For genuine 3D typing, set `pciseq_geometry_mode=labels_3d` and supply
`pciseq_labels_path` to an integer NPY array with dimensions `(z, y, x)`.
Background is zero and positive labels must match the native segmentation's
canonical cell IDs across planes. Supply `pciseq_origin_um` and
`pciseq_voxel_size_um` in **x, y, z** order, in the declared coordinate system.
Measured transcript Z coordinates are required. For example, in a Nextflow
parameter file:

```yaml
pciseq_enabled: true
pciseq_reference_path: references/mean_expression.csv
pciseq_geometry_mode: labels_3d
pciseq_labels_path: inputs/cell_labels.npy
pciseq_origin_um: [0, 0, 0]
pciseq_voxel_size_um: [0.25, 0.25, 1.5]
```

The pinned upstream release requires equal X/Y pixel sizes for correct
in-mask lookup; Z spacing may differ. Resample volumes with rectangular XY
pixels before supplying them.

The mask setting is global to that invocation. Run sample/segmentation-specific
volumes in separate invocations or use the Python CLI with one configuration
per acquisition.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `pciseq_enabled` | `false` | Enable the optional stage. |
| `pciseq_reference_path` | `null` | Required genes-by-cell-type reference CSV/TSV. |
| `pciseq_geometry_mode` | `project_2d` | Native polygon projection or explicit `labels_3d`. |
| `pciseq_overlap_policy` | `background` | Put polygon-overlap seed pixels in background; `error` rejects overlaps. |
| `pciseq_pixel_size_um` | `1.0` | Raster resolution for polygon projection. |
| `pciseq_max_pixels` | `100000000` | Maximum number of raster voxels before rejecting input. |
| `pciseq_points_key` | `null` | Optional explicit native transcript points element. |
| `pciseq_coordinate_system` | `global` | Shared physical coordinate frame for shapes and points. |
| `pciseq_labels_path` | `null` | Genuine 3D label-volume NPY. |
| `pciseq_origin_um` | `[0,0,0]` | 3D grid physical origin, in x/y/z order. |
| `pciseq_voxel_size_um` | `[1,1,1]` | 3D voxel dimensions, in x/y/z order. |
| `pciseq_max_iter` | `1000` | Model iteration limit. |
| `pciseq_n_neighbors` | `9` | Candidate parent count, capped at the available cells. |
| `pciseq_options` | `{}` | Additional upstream model options, such as `Inefficiency`. |
| `pciseq_write_spatialdata_table` | `true` | Write a separate derived table under MerXen's store lock. |
| `pciseq_conda` | `envs/environment.pciseq.yml` | Dedicated compute environment. |
| `pciseq_container` | `null` | Dedicated compute image for container profiles. |

The integration manages gene filtering, output location, voxel size, candidate
count, iteration limit, flat-cell retention, and headless operation. These
settings cannot be overridden through `pciseq_options`. Use a parameter file
for array/dictionary settings.

## Outputs and interpretation

Outputs are published under
`<outdir>/<pair_id>/<segmentation>/pciseq/pciseq_out/<platform>/`:

- `annotated.h5ad`: clustered observations with `pciseq_cell_type`,
  `pciseq_probability`, `pciseq_status`, and reported class probability summaries.
- `cell_assignments.csv` and `spot_assignments.csv`: assignments linked to
  canonical cell/transcript IDs.
- `cell_data.json` and `gene_data.json`: raw upstream result records.
- Cell/spot ID maps and preparation/computation manifests: coordinate choices,
  exclusions, reference provenance, model options, and upstream version/commit.
- With table writing enabled, a separate `<native_table_key>_pciseq` table in
  the source SpatialData store.

Upstream's returned class probabilities are thresholded and truncated to three
decimals. Missing classes are not measured zeros, and these summaries must not
be treated as complete posterior distributions. `Zero` is a model class, not a
biological cell type. The public `fit` API returns normally when the iteration
limit is reached and does not expose convergence status; the computation
manifest records that status as unavailable. Review upstream iteration logs
and assignment quality before using results for analysis.

## Runtime and direct CLI

`PCISEQ_PREPARE` and `PCISEQ_FINALIZE` run in MerXen's standard environment.
`PCISEQ_COMPUTE` uses the pinned upstream commit
`5e2c0fd1841f8a4022a2fa341ece965687e343dd` in a separate runtime. The stage runs
without a viewer or upstream SpatialData exports. Subsequent clustering
consumers wait until all pciSeq tasks for the pair complete.

Create the optional environment with
`conda env create -f envs/environment.pciseq.yml`. For Apptainer, build
`containers/Dockerfile.pciseq` and set `pciseq_container` in the private site
config or invocation, or set `MERXEN_PCISEQ_CONTAINER`. The container needs the
same access to the repository
and inputs as other stages.

`merxen pciseq --config pciseq_config.json --compute-python <runtime-python>`
runs all three stages with `PciSeqConfig`. The optional interpreter selects the
isolated runtime; the active MerXen interpreter performs spatial I/O. Required
JSON fields are `pair_id`, `sample_id`, `platform`, `segmentation`, `source_h5ad`,
`spatialdata_path`, and `reference_path`; other fields match the parameter table
without the `pciseq_` prefix. Nextflow generates this configuration automatically.

To refresh only this runtime's dependency lock and conda cache identity:

```bash
uv pip compile --group pciseq-runtime --python-version 3.12 \
    --python-platform linux -o requirements/requirements.pciseq.lock
python scripts/update_env_lock_hash.py \
    --lock requirements/requirements.pciseq.lock --env envs/environment.pciseq.yml
```

The implementation plan is in
[plans/pciseq-integration-plan.md](../plans/pciseq-integration-plan.md).
