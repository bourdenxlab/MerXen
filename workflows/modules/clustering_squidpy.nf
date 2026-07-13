process CLUSTERING_SQUIDPY_PREPARE {
    tag "${pair_id}:${segmentation}"

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json)

    output:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path("clustering_squidpy_config.json"),
        path("clustering_prepare_out")

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

    cat > clustering_squidpy_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "clustering_squidpy_out",
  "samples": ${samples_json},
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
  "spatial_scatter_point_size": ${params.clustering_squidpy_spatial_scatter_point_size},
  "figure_dpi": ${params.clustering_squidpy_figure_dpi},
  "use_gpu": ${params.clustering_squidpy_use_gpu},
  "write_spatialdata_table": ${params.clustering_squidpy_write_spatialdata_table},
  "hierarchical_enabled": ${params.clustering_squidpy_hierarchical_enabled},
  "broad_round": {
    "leiden_resolution": ${params.clustering_squidpy_broad_leiden_resolution}
  },
  "subcluster_round": {
    "leiden_resolution": ${params.clustering_squidpy_subcluster_leiden_resolution}
  },
  "subcluster_resolution_overrides": ${groovy.json.JsonOutput.toJson(params.clustering_squidpy_subcluster_resolution_overrides ?: [:])},
  "neuron_split_round": {
    "leiden_resolution": ${params.clustering_squidpy_neuron_split_leiden_resolution}
  },
  "neuron_subcluster_round": {
    "leiden_resolution": ${params.clustering_squidpy_neuron_subcluster_leiden_resolution}
  },
  "min_branch_cells": ${params.clustering_squidpy_min_branch_cells},
  "broad_annotation": {
    "marker_lookup_path": ${params.clustering_squidpy_broad_marker_lookup_path ? groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_marker_lookup_path.toString()) : "null"},
    "taxonomy_metadata_path": ${params.clustering_squidpy_broad_taxonomy_metadata_path ? groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_taxonomy_metadata_path.toString()) : "null"},
    "cluster_membership_path": ${params.clustering_squidpy_broad_cluster_membership_path ? groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_cluster_membership_path.toString()) : "null"},
    "reference_cache_dir": ${params.clustering_squidpy_broad_reference_cache_dir ? groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_reference_cache_dir.toString()) : "null"},
    "marker_level": ${groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_marker_level.toString())},
    "min_marker_overlap": ${params.clustering_squidpy_broad_min_marker_overlap},
    "max_markers_per_label": ${params.clustering_squidpy_broad_max_markers_per_label},
    "score_margin_threshold": ${params.clustering_squidpy_broad_score_margin_threshold},
    "unknown_label": ${groovy.json.JsonOutput.toJson(params.clustering_squidpy_broad_unknown_label.toString())}
  }
}
JSON

    python -m merxen.clustering_squidpy_stages prepare \
        --config clustering_squidpy_config.json \
        --output-dir clustering_prepare_out
    """
}

process CLUSTERING_SQUIDPY_COMPUTE {
    tag "${pair_id}:${segmentation}"

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path(clustering_config),
        path(prepared_dir)

    output:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path("clustering_compute_out")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"

    if ${params.clustering_squidpy_gpu_vram_monitor}; then
        mkdir -p clustering_compute_out/gpu_vram
        gpu_vram_stem="${pair_id}_${segmentation}"
        python -m merxen.clustering_squidpy_stages compute \
            --config "${clustering_config}" \
            --input-dir "${prepared_dir}" \
            --output-dir clustering_compute_out &
        clustering_pid=\$!
        python -m merxen.monitoring.gpu_vram \
            --pid "\${clustering_pid}" \
            --interval-seconds ${params.clustering_squidpy_gpu_vram_monitor_interval_seconds} \
            --samples-path "clustering_compute_out/gpu_vram/\${gpu_vram_stem}_samples.tsv" \
            --summary-path "clustering_compute_out/gpu_vram/\${gpu_vram_stem}_summary.json" &
        monitor_pid=\$!

        set +e
        wait "\${clustering_pid}"
        clustering_exit=\$?
        wait "\${monitor_pid}" || true
        set -e
        exit "\${clustering_exit}"
    fi

    python -m merxen.clustering_squidpy_stages compute \
        --config "${clustering_config}" \
        --input-dir "${prepared_dir}" \
        --output-dir clustering_compute_out
    """
}

process CLUSTERING_SQUIDPY_FINALIZE {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/clustering_squidpy" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path(computed_dir)

    output:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path("clustering_squidpy_out")

    script:
    """
    set -euo pipefail
    cat > clustering_finalize_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "clustering_squidpy_out",
  "samples": ${samples_json},
  "write_spatialdata_table": ${params.clustering_squidpy_write_spatialdata_table}
}
JSON

    python -m merxen.clustering_squidpy_stages finalize \
        --config clustering_finalize_config.json \
        --input-dir "${computed_dir}" \
        --output-dir clustering_squidpy_out
    """
}
