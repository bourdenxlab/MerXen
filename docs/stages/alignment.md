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
| `alignment_max_iter` | `500` | Spateo optimization iterations. |
| `alignment_beta` | `1.0` | Spateo non-rigid kernel width. |
| `alignment_lambda_vf` | `1.0` | Spateo vector-field regularization. |
| `alignment_k` | `50` | Spateo low-rank control points. |
| `alignment_partial_robust_level` | `50` | Robustness level for partial overlap. |
| `alignment_n_sampling` | `1000` | Stochastic variational batch size for GPU memory control. |
| `alignment_chunk_capacity` | `1` | Spateo chunk capacity for lower peak memory. |
| `alignment_n_top_genes` | `100` | HVG feature count used for alignment. |
| `alignment_max_nonrigid_anchors` | `5000` | Maximum RBF anchors used when applying non-rigid transforms. |
| `alignment_qc_grid_rows` / `alignment_qc_grid_cols` | `10` / `10` | SABench-style QC grid. |

These defaults are intentionally conservative for large pairs such as P7513 on
a 24 GB GPU. Increase `alignment_n_sampling`, `alignment_n_top_genes`, or
`alignment_k` only after the QC overlay looks stable.

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
| `align_out/merscope_aligned.zarr` | MERSCOPE SpatialData copy with transformed shapes and points. |
| `align_out/xenium_aligned.zarr` | Xenium reference SpatialData copy. |
| `align_out/alignment_transform.json` | Affine matrix, RBF metadata, Spateo parameters, displacement summary. |
| `align_out/alignment_coords/*.csv` | Raw, rigid, and non-rigid alignment centroids. |

Published under `${outdir}/<pair_id>/alignment_qc/`:

| File | Contents |
|------|----------|
| `alignment_qc_out/<pair_id>_alignment_qc.json` | SABench-style grid metrics and point-distance summary. |
| `alignment_qc_out/<pair_id>_alignment_qc_metrics.csv` | Single-row CSV form of the same metrics. |
| `alignment_qc_out/<pair_id>_alignment_overlay.png` | Xenium/MERSCOPE centroid overlay after alignment. |

## Notes

The first implementation uses cell-level gene features and centroids. Image
feature extraction is represented in the config and metadata, but is skipped
unless the SpatialData image elements expose an unambiguous xy mapping.
