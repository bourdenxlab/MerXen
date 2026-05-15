# Section alignment

> **Status: implemented as an optional stage.** Enable with
> `--enable_alignment true` after installing Spateo.

## Intent

Adjacent MERSCOPE and Xenium sections are physically offset and can deform
during sample preparation. The alignment stage maps MERSCOPE xy coordinates
into the Xenium coordinate system so spatial analyses can compare equivalent
tissue regions across platforms.

## Method

`ALIGN` builds paired AnnData objects from enriched SpatialData cell tables and
cell-boundary centroids, then runs Spateo `morpho_align` with Xenium as the
fixed/reference section and MERSCOPE as the moving section. The stage records
both Spateo rigid and non-rigid coordinates.

MERSCOPE outputs are transformed back into SpatialData with:

- an affine matrix fitted from raw MERSCOPE centroids to Spateo rigid
  coordinates;
- a thin-plate/RBF residual displacement field fitted from raw MERSCOPE
  centroids to Spateo non-rigid coordinates.

The default downstream coordinate set is non-rigid. The rigid transform and
alignment coordinate tables are retained for inspection.

## Nextflow

`ALIGN` runs after per-platform `QC` and before `COMPARE` when
`params.enable_alignment = true`. `ALIGN_QC` then computes post-alignment QC
metrics and overlays. When alignment is disabled, `COMPARE` and `VISUALIZE`
continue to receive the enriched zarrs directly.

Key parameters live in `workflows/nextflow.config`:

| Param | Default | Description |
|-------|---------|-------------|
| `enable_alignment` | `false` | Run `ALIGN` and `ALIGN_QC`. |
| `alignment_device` | `auto` | Spateo device; `auto` chooses CUDA when available. |
| `alignment_dtype` | `float32` | Spateo tensor precision; keeps GPU memory lower. |
| `alignment_selected_mode` | `nonrigid` | Coordinate set used for transformed outputs. |
| `alignment_max_iter` | `360` | Spateo optimization iterations. |
| `alignment_nonrigid_start_iter` | `220` | Iteration where non-rigid refinement starts. |
| `alignment_beta` | `0.005` | Spateo non-rigid kernel width. |
| `alignment_lambda_vf` | `3000.0` | Spateo vector-field regularization. |
| `alignment_k` | `15` | Spateo low-rank control points. |
| `alignment_partial_robust_level` | `100` | Robustness level for partial overlap. |
| `alignment_allow_flip` | `true` | Allow a mirrored coarse initialization before rigid/non-rigid refinement. |
| `alignment_svi_mode` | `false` | Use full pairwise matching on the sampled cells instead of SVI mini-batches. |
| `alignment_n_sampling` | `1000` | Stochastic variational batch size for GPU memory control. |
| `alignment_sparse_top_k` | `512` | Sparse matching top-k used by Spateo. |
| `alignment_chunk_capacity` | `1` | Spateo chunk capacity for lower peak memory. |
| `alignment_use_hvg` | `false` | Use the full shared panel instead of HVGs. |
| `alignment_n_top_genes` | `100` | HVG feature count used for alignment. |
| `alignment_use_pca` | `true` | Run joint PCA on shared expression features before Spateo. |
| `alignment_n_pcs` | `50` | Number of joint PCA components used for Spateo matching. |
| `alignment_max_alignment_cells` | `35000` | Deterministic per-platform cell subsample used for Spateo optimization. |
| `alignment_seed` | `21` | Seed for deterministic alignment subsampling. |
| `alignment_max_nonrigid_anchors` | `5000` | Maximum RBF anchors used when applying non-rigid transforms. |
| `alignment_qc_grid_rows` / `alignment_qc_grid_cols` | `10` / `10` | SABench-style QC grid. |

These defaults come from the P7513 tuning notebook. They use the shared panel,
joint PCA, mirrored initialization, no SVI, and a 35k-cell per-platform
subsample so the non-rigid pass fits comfortably on a 24 GB GPU.

## CLI

```bash
merxen align --config align_config.json
merxen alignment-qc --config alignment_qc_config.json
```

`AlignmentConfig` and `AlignmentQCConfig` in `src/merxen/config.py` are the
Python contracts for these JSON files.

## Installation note

Spateo 1.1.1 imports older AnnData/Cellpose symbols through its broader
package import path. MerXen keeps modern SpatialData/AnnData/Cellpose for the
rest of the pipeline and applies narrow runtime compatibility shims before
loading `spateo.align`.

In the current MerXen environment, the tested install sequence is:

```bash
pip install spateo-release==1.1.1
pip install "anndata>=0.12.10"
```

`pip check` may still report `dynamo-release`'s declared `anndata<0.11`
constraint, but the alignment wrapper only uses Spateo's alignment API.

## Outputs

Published under `${outdir}/<pair_id>/alignment/`:

| File | Contents |
|------|----------|
| `align_out/alignment_transform.json` | Affine matrix, serialized RBF metadata, Spateo parameters, displacement summary. |
| `align_out/alignment_coords/*.csv` | Raw, rigid, and non-rigid alignment centroids. |

`ALIGN` updates the existing MERSCOPE latest zarr in place. Raw elements are
left untouched, rigid affine transforms are saved to the `merxen_xenium`
coordinate system, and new `*_aligned_nonrigid` vector elements are added with
materialized non-rigid coordinates. Xenium remains the fixed reference and is
not copied.

Published under `${outdir}/<pair_id>/alignment_qc/`:

| File | Contents |
|------|----------|
| `alignment_qc_out/<pair_id>_alignment_qc.json` | SABench-style grid metrics and point-distance summary. |
| `alignment_qc_out/<pair_id>_alignment_qc_metrics.csv` | Single-row CSV form of the same metrics. |
| `alignment_qc_out/<pair_id>_alignment_overlay.png` | Xenium/MERSCOPE centroid overlay after alignment. |

The alignment overlay PNG is also written as a same-stem PDF.

## Notes

The first implementation uses cell-level gene features and centroids. Image
feature extraction is represented in the config and metadata, but is skipped
unless the SpatialData image elements expose an unambiguous xy mapping.
