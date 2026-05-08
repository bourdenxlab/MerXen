process CLUSTERING_SQUIDPY {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/clustering_squidpy" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        path(merscope_zarr, stageAs: "merscope_latest_input.zarr"),
        val(xenium_zarr)

    output:
    tuple val(pair_id), path("clustering_squidpy_out")

    script:
    """
    set -euo pipefail

    cat > clustering_squidpy_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "clustering_squidpy_out",
  "samples": [
    {
      "sample_id": "${pair_id}_MERSCOPE",
      "platform": "MERSCOPE",
      "zarr_path": "${merscope_zarr}"
    },
    {
      "sample_id": "${pair_id}_XENIUM",
      "platform": "XENIUM",
      "zarr_path": "${xenium_zarr}"
    }
  ],
  "drop_control_features": ${params.clustering_squidpy_drop_control_features},
  "min_counts": ${params.clustering_squidpy_min_counts},
  "min_cells": ${params.clustering_squidpy_min_cells},
  "normalize_target_sum": ${params.clustering_squidpy_normalize_target_sum},
  "normalize_exclude_highly_expressed": ${params.clustering_squidpy_normalize_exclude_highly_expressed},
  "normalize_max_fraction": ${params.clustering_squidpy_normalize_max_fraction},
  "n_pcs": ${params.clustering_squidpy_n_pcs},
  "n_neighbors": ${params.clustering_squidpy_n_neighbors},
  "leiden_resolution": ${params.clustering_squidpy_leiden_resolution},
  "umap_min_dist": ${params.clustering_squidpy_umap_min_dist},
  "umap_spread": ${params.clustering_squidpy_umap_spread},
  "random_seed": ${params.clustering_squidpy_random_seed},
  "spatial_point_size": ${params.clustering_squidpy_spatial_point_size},
  "figure_dpi": ${params.clustering_squidpy_figure_dpi},
  "use_gpu": ${params.clustering_squidpy_use_gpu}
}
JSON

    merxen clustering-squidpy --config clustering_squidpy_config.json
    """
}
