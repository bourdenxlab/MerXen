# MENDER spatial domains

MENDER is an opt-in terminal analysis that identifies spatial domains from a
categorical cell-state annotation and native cell centroids. MerXen runs every
MERSCOPE, Xenium or ATERA acquisition independently with `MENDER_single`;
platforms and segmentation branches are never concatenated as slices.

ATERA rows use the native (vendor) segmentation, whose only analysis layer is
`original_seg` (shape `atera_cell_boundaries`). A default or `all`
`mender_segmentations` value resolves to `original_seg` on those rows; naming
another layer for an ATERA row is an error.

MENDER is listed after existing stages. Its inputs are released by the shared
per-pair token only when that pair's current terminal stage is complete. The
token excludes other independent terminal analyses such as GASTON, so the two
may overlap and neither consumes the other's output.

Enable the default ProSeg-hybrid analysis with:

```bash
nextflow run workflows/main.nf \
  --samplesheet samples.csv \
  --mender_enabled true \
  --outdir results
```

When the historical `stop_stage=clustering_squidpy` default is unchanged,
enabling MENDER extends the effective stop to `mender`. Use a comma-separated
subset or `all` for other branches:

```bash
--mender_segmentations reseg,proseg_hybrid
--mender_segmentations all
```

`all` expands to `reseg`, `original_seg`, `proseg_mask`, and
`proseg_hybrid`. These branches are added only to the clustering prerequisites;
they do not expand the segmentation set used by QC, MECR, comparison,
visualization, or spatial-gene analysis.

## Data contract

`MENDER_PREPARE` reads the clustered H5AD and its corresponding derived
SpatialData table in MerXen's modern environment. Only `obs` and `uns` of the
H5AD are read (through `h5py`); `X`, layers and embeddings, which are large for
a whole-transcriptome table, are never loaded. It requires the configured
categorical state column (default `hierarchical_cluster`) and never falls back
to ordinary Leiden. It then replaces any aligned coordinates with centroids
loaded from the explicit native shape and exports only:

```text
cell_id  native_x  native_y  cell_state
```

`MENDER_COMPUTE` receives this Parquet table, constructs a zero-expression
AnnData, and runs the repository version pinned at commit
`b29dc5ea352a2762cb7bf49d44ee661f0009f694`. The default call is equivalent to:

```python
model = MENDER.MENDER_single(adata, ct_obs="cell_state", random_seed=666)
model.set_MENDER_para(
    nn_mode="radius",
    nn_para=20,
    count_rep="s",
    include_self=False,
    n_scales=5,
)
model.run_representation()
model.run_clustering_normal(-0.8, run_umap=True)
```

These layer-focused defaults use five separate 20 µm shells spanning a
maximum radius of 100 µm. Excluding the central cell reduces sensitivity to
isolated cell-state assignments, while the lower Leiden resolution favours
larger domains. The input state remains `hierarchical_cluster`; MerXen does not
collapse or remap its categories.

Resolution mode passes a negative resolution. Target-K mode requires an
integer of at least two:

```bash
--mender_clustering_mode target_k --mender_target_k 12
```

The wrapper fails if MENDER drops or duplicates a cell, changes native
coordinates, omits a domain, produces only one domain, or fails to create its
context AnnData.

`MENDER_FINALIZE` joins domains by immutable cell ID and publishes the
annotated source H5AD, context H5AD, manifests, tables, and plots. It also
opens the source H5AD backed (`backed="r"`), which keeps `X` on disk; anndata
still loads layers (such as `counts`) and embeddings into memory, so the saving
is the `X` part of the table. The annotated copy is written from it and is
byte-identical to the copy an in-memory read writes.
`MENDER_IMPORT` then takes the shared `<zarr>.merxen-write.lock` (beside the
real store, with the staged symlink resolved, so every writer reaching the
store takes the same lock), rereads the latest store, and updates only `mender_domain` and `uns["merxen_mender"]` in
the derived clustered table. The import is separate so finalized standalone
artifacts remain published if the SpatialData write fails.

## Published-output restart

The stage can consume an earlier clustering run without rerunning clustering:

```bash
nextflow run workflows/main.nf \
  --samplesheet samples.csv \
  --mender_enabled true \
  --only_stage mender \
  --outdir results
```

For every selected platform and segmentation, this mode requires both
`clustering_squidpy_out/<platform>/<sample_id>_clustered.h5ad` and the
corresponding derived clustered table in the published
`latest_spatialdata.zarr`.

## CPU environment and Apptainer

Only `MENDER_COMPUTE` uses `envs/environment.mender.yml`. The environment contains
the old AnnData 0.9, Scanpy 1.9, and Squidpy 1.2-era stack; these packages are
not added to MerXen's base project dependencies. It also pins umap-learn's
nearest-neighbour stack (`pynndescent` 0.5.13, `numba` 0.55.2, `llvmlite`
0.38.1): left open, a fresh environment resolved `pynndescent` 0.7, whose
parallel subtree builder numba 0.55 cannot compile, and `MENDER_COMPUTE`
failed in `sc.pp.neighbors` on the full ATERA section (91,381 cells; under
4,096 cells Scanpy 1.9 computes exact neighbours and never reaches it). Build a portable CPU image and
override its path as needed:

```bash
docker build -f containers/Dockerfile.mender -t merxen-mender .
apptainer build merxen-mender.sif docker-daemon://merxen-mender:latest

nextflow run workflows/main.nf \
  -profile apptainer \
  --samplesheet samples.csv \
  --mender_enabled true \
  --mender_container file:///absolute/path/merxen-mender.sif
```

MENDER is CPU-only: the process clears `CUDA_VISIBLE_DEVICES`, uses the HTC/CPU
queue, and must not be given Apptainer `--nv`.

Dwight reserves 64 GB for prepare, 192 GB for compute, 64 GB for finalize, and
48 GB for import. Compute has `maxForks=2`, so two simultaneous acquisitions
reserve 384 GB within the executor's 640 GB budget. External profiles may
override every process resource.
