process SPATIAL_GENE_ANALYSIS {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/spatial_gene_analysis" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json)

    output:
    tuple val(pair_id),
        val(segmentation),
        path("spatial_gene_analysis_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export VECLIB_MAXIMUM_THREADS="${task.cpus}"
    export BLIS_NUM_THREADS="${task.cpus}"
    export RAYON_NUM_THREADS="${task.cpus}"
    export POLARS_MAX_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"

    cat > spatial_gene_analysis_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "spatial_gene_analysis_out",
  "samples": ${samples_json},
  "drop_control_features": ${params.spatial_gene_analysis_drop_control_features},
  "min_counts": ${params.spatial_gene_analysis_min_counts},
  "min_cells": ${params.spatial_gene_analysis_min_cells},
  "normalize_target_sum": ${params.spatial_gene_analysis_normalize_target_sum},
  "normalize_exclude_highly_expressed": ${params.spatial_gene_analysis_normalize_exclude_highly_expressed},
  "normalize_max_fraction": ${params.spatial_gene_analysis_normalize_max_fraction},
  "n_neighbors": ${params.spatial_gene_analysis_n_neighbors},
  "top_n": ${params.spatial_gene_analysis_top_n},
  "spatial_point_size": ${params.spatial_gene_analysis_spatial_point_size},
  "figure_dpi": ${params.spatial_gene_analysis_figure_dpi}
}
JSON

    merxen spatial-gene-analysis --config spatial_gene_analysis_config.json
    """
}
