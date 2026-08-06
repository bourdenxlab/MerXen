# GASTON spatial domains

GASTON is an opt-in terminal stage that learns spatial domains independently
for every selected sample, platform, and segmentation. Enable it with
`--gaston_enabled true`. Its segmentation selector is independent of
`analysis_segmentation` and defaults to `proseg_hybrid`.

The stage follows the settings from the official
[GASTON repository](https://github.com/raphael-group/GASTON) and
[motor-cortex tutorial](https://gaston.readthedocs.io/en/latest/notebooks/tutorials/motor_cortex_tutorial.html),
using the pinned `gaston-spatial` 0.0.2 commit `79577c4`.

## Input contract

For each branch, `GASTON_PREPARE` reads raw counts from
`layers["counts"]` in the clustered H5AD and joins cells by the SpatialData
table's `instance_key`. It always computes centroids from the explicit native
shape:

| Segmentation | Native shape |
|--------------|--------------|
| `reseg` | `MOSAIK_proseg` |
| `original_seg` | `merscope_cell_boundaries` or `xenium_cell_boundaries` |
| `proseg_mask` | `MOSAIK_cellpose` |
| `proseg_hybrid` | `MOSAIK_proseg_hybrid` |

Aligned shapes and `obsm["spatial"]` are deliberately ignored. Preparation
rejects duplicate or unmatched cell IDs, non-finite coordinates, invalid
counts, and empty expression matrices. It uses all cells surviving clustering,
retains eligible non-control genes up to `gaston_max_genes`, and writes a
checksum-bearing portable input bundle. Multiple substantial spatial
components produce a warning but are never cropped automatically.

## Process graph

```text
GASTON_PREPARE
  -> GASTON_GLM_PCA
    -> GASTON_TRAIN (one task per seed)
      -> GASTON_POSTPROCESS
        -> GASTON_IMPORT
```

GLM-PCA and postprocessing are CPU-only. Training requests one GPU per seed;
on Dwight its `maxForks=1` task uses the existing shared GPU flock, while a
scheduler profile can allocate physically isolated GPUs. Successful seed tasks
are independently resumable. A numerically failed restart writes a failed seed
manifest so all configured seeds can still be ranked; the stage fails if none
produces a finite loss and complete model.

`gaston_glmpca_iterations` retains the configured initial budget of 30.
Because whole acquisitions can converge more slowly than tutorial subsets,
GLM-PCA may continue to its library-default ceiling of 1,000 iterations. The
manifest records both budgets, the final relative-deviance change, and the
termination reason. Non-finite output or failure to meet the library's
`1e-4` relative-deviance tolerance by that ceiling still fails explicitly.

Postprocessing selects the finite minimum-loss restart and evaluates domain
likelihoods across the configured K range. A supplied `gaston_num_domains`
wins directly. Otherwise Kneedle uses a convex, decreasing curve. If no knee
exists, the stage fails after writing its diagnostic likelihood table unless
`gaston_auto_k_fallback` is configured. Isodepth is written raw: its direction
is unanchored and arbitrary, with no anatomical reversal or scaling.

## SpatialData safety

`GASTON_IMPORT` first writes a standalone annotated H5AD. When SpatialData
import is enabled, it locks `<latest_spatialdata.zarr>.merxen-write.lock`,
re-reads the clustered table, merges only the four `gaston_*` columns, reparses
the original region and instance keys, and replaces the table. Existing
clustering, MENDER, and other annotations are preserved. The original
segmentation table and native shapes are never modified.

Use `--only_stage gaston` to consume already-published clustered H5ADs and
latest zarrs. Missing files or clustered tables fail during preflight; the
workflow does not silently recluster them.

See [Configuration](../configuration.md) for all controls and
[Outputs](../outputs.md) for the published layout.
