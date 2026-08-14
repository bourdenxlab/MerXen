process ALIGN {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/alignment" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(merscope_zarr_path),
        val(xenium_zarr_path),
        path(merscope_tissue_annotation, stageAs: "merscope_tissue_annotation.geojson"),
        path(xenium_tissue_annotation, stageAs: "xenium_tissue_annotation.geojson")

    output:
    tuple val(pair_id),
        val(merscope_zarr_path),
        val(xenium_zarr_path),
        path("align_out")

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

    if [[ "${params.alignment_backend}" == "valis" ]] && ${params.alignment_bootstrap_dependencies}; then
        if ! merxen check-alignment-deps --backend valis >/dev/null 2>&1; then
            python -m pip install --no-input --no-deps \\
                "${params.alignment_valis_requirement}"
        fi
    elif [[ "${params.alignment_backend}" == "legacy_spateo" ]] && ${params.alignment_bootstrap_dependencies}; then
        if ! merxen check-alignment-deps --backend legacy_spateo >/dev/null 2>&1; then
            python -m pip install --no-input \\
                "${params.alignment_dynamo_requirement}" \\
                "${params.alignment_spateo_requirement}"
            python -m pip install --no-input --upgrade \\
                "${params.alignment_anndata_requirement}"
        fi
    fi

    merxen check-alignment-deps --backend "${params.alignment_backend}"

    cat > align_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "merscope_zarr_path": "${merscope_zarr_path}",
  "xenium_zarr_path": "${xenium_zarr_path}",
  "output_dir": "align_out",
  "backend": "${params.alignment_backend}",
  "fixed_platform": "${params.alignment_fixed_platform}",
  "moving_platform": "${params.alignment_moving_platform}",
  "merscope_image": {
    "image_key": "${params.alignment_merscope_image_key}",
    "tissue_annotation_path": ${params.alignment_backend == "valis" ? "\"${merscope_tissue_annotation}\"" : "null"},
    "dapi_channel": "${params.alignment_merscope_dapi_channel}",
    "pixel_size_um": ${params.alignment_merscope_pixel_size_um == null ? "null" : params.alignment_merscope_pixel_size_um}
  },
  "xenium_image": {
    "image_key": "${params.alignment_xenium_image_key}",
    "tissue_annotation_path": ${params.alignment_backend == "valis" ? "\"${xenium_tissue_annotation}\"" : "null"},
    "dapi_channel": "${params.alignment_xenium_dapi_channel}",
    "pixel_size_um": ${params.alignment_xenium_pixel_size_um == null ? "null" : params.alignment_xenium_pixel_size_um}
  },
  "valis": {
    "registration_pixel_size_um": ${params.alignment_registration_pixel_size_um == null ? "null" : params.alignment_registration_pixel_size_um},
    "registration_source_max_dim_px": ${params.alignment_registration_source_max_dim_px},
    "max_processed_image_dim_px": ${params.alignment_valis_max_processed_image_dim_px},
    "max_non_rigid_registration_dim_px": ${params.alignment_valis_max_non_rigid_registration_dim_px},
    "thumbnail_size": ${params.alignment_valis_thumbnail_size},
    "global_transform": "${params.alignment_valis_global_transform}",
    "random_seed": ${params.alignment_seed},
    "coordinate_system_name": "${params.alignment_coordinate_system_name}",
    "transform_transcripts": ${params.alignment_transform_transcripts},
    "transform_centroids": ${params.alignment_transform_centroids},
    "transform_polygons": ${params.alignment_transform_polygons},
    "mark_shared_tissue_domain": ${params.alignment_mark_shared_tissue_domain},
    "resume": ${params.alignment_resume},
    "preprocessing": {
      "background_sigma_um": ${params.alignment_background_sigma_um},
      "background_boundary_mode": "${params.alignment_background_boundary_mode}",
      "lower_percentile": ${params.alignment_intensity_lower_percentile},
      "upper_percentile": ${params.alignment_intensity_upper_percentile},
      "compression": "${params.alignment_intensity_compression}",
      "clahe_clip_limit": ${params.alignment_clahe_clip_limit},
      "smoothing_sigma_um": ${params.alignment_smoothing_sigma_um},
      "edge_taper_um": ${params.alignment_edge_taper_um},
      "edge_exclusion_um": ${params.alignment_edge_exclusion_um},
      "mask_smoothing_sigma_um": ${params.alignment_mask_smoothing_sigma_um},
      "mask_closing_radius_um": ${params.alignment_mask_closing_radius_um},
      "mask_min_area_um2": ${params.alignment_mask_min_area_um2},
      "mask_hole_area_um2": ${params.alignment_mask_hole_area_um2},
      "mask_dilation_um": ${params.alignment_mask_dilation_um}
    },
    "orientation": {
      "max_dimension_px": ${params.alignment_orientation_max_dim_px},
      "coarse_step_degrees": ${params.alignment_orientation_coarse_step_degrees},
      "refine_step_degrees": ${params.alignment_orientation_refine_step_degrees},
      "final_step_degrees": ${params.alignment_orientation_final_step_degrees},
      "allow_reflection": ${params.alignment_allow_reflection},
      "reflection_mode": "${params.alignment_reflection_mode}",
      "reflection_minimum_score_improvement": ${params.alignment_reflection_minimum_score_improvement},
      "translation_candidates_per_angle": ${params.alignment_orientation_translation_candidates_per_angle},
      "coarse_translation_radius_px": ${params.alignment_orientation_coarse_translation_radius_px},
      "refine_translation_radius_px": ${params.alignment_orientation_refine_translation_radius_px},
      "final_translation_radius_px": ${params.alignment_orientation_final_translation_radius_px},
      "minimum_fixed_overlap_fraction": ${params.alignment_orientation_min_fixed_overlap_fraction},
      "minimum_moving_overlap_fraction": ${params.alignment_orientation_min_moving_overlap_fraction},
      "minimum_retained_moving_fraction": ${params.alignment_orientation_min_retained_moving_fraction},
      "minimum_relative_dice": ${params.alignment_orientation_min_relative_dice},
      "initial_angle_degrees": ${params.alignment_orientation_initial_angle_degrees == null ? 'null' : params.alignment_orientation_initial_angle_degrees},
      "initial_translation_x_um": ${params.alignment_orientation_initial_translation_x_um == null ? 'null' : params.alignment_orientation_initial_translation_x_um},
      "initial_translation_y_um": ${params.alignment_orientation_initial_translation_y_um == null ? 'null' : params.alignment_orientation_initial_translation_y_um},
      "local_fine_search_enabled": ${params.alignment_orientation_local_fine_search_enabled},
      "local_fine_angle_radius_degrees": ${params.alignment_orientation_local_fine_angle_radius_degrees},
      "local_fine_translation_radius_um": ${params.alignment_orientation_local_fine_translation_radius_um},
      "local_fine_coarse_angle_step_degrees": ${params.alignment_orientation_local_fine_coarse_angle_step_degrees},
      "local_fine_coarse_translation_step_um": ${params.alignment_orientation_local_fine_coarse_translation_step_um},
      "local_fine_refine_angle_step_degrees": ${params.alignment_orientation_local_fine_refine_angle_step_degrees},
      "local_fine_refine_translation_step_um": ${params.alignment_orientation_local_fine_refine_translation_step_um},
      "local_fine_maxima_to_refine": ${params.alignment_orientation_local_fine_maxima_to_refine},
      "local_fine_competing_score_margin": ${params.alignment_orientation_local_fine_competing_score_margin}
    },
    "partial_overlap": {
      "enabled": ${params.alignment_partial_overlap_enabled},
      "max_dimension_px": ${params.alignment_partial_overlap_max_dim_px},
      "angle_search_radius_degrees": ${params.alignment_partial_overlap_angle_radius_degrees},
      "coarse_angle_step_degrees": ${params.alignment_partial_overlap_angle_step_degrees},
      "maximum_translation_um": ${params.alignment_partial_overlap_max_translation_um},
      "retained_boundary_fraction": ${params.alignment_partial_overlap_retained_boundary_fraction},
      "boundary_distance_scale_um": ${params.alignment_partial_overlap_boundary_distance_scale_um},
      "density_sigma_um": ${params.alignment_partial_overlap_density_sigma_um},
      "minimum_fixed_overlap_fraction": ${params.alignment_partial_overlap_min_fixed_overlap_fraction},
      "minimum_moving_overlap_fraction": ${params.alignment_partial_overlap_min_moving_overlap_fraction},
      "candidates_to_refine": ${params.alignment_partial_overlap_candidates_to_refine}
    },
    "features": {
      "num_features": ${params.alignment_valis_num_features},
      "device": "${params.alignment_device}"
    },
    "non_rigid": {
      "enabled": ${params.alignment_valis_non_rigid_enabled},
      "backend": "${params.alignment_valis_non_rigid_backend}",
      "compose_non_rigid": false,
      "grid_spacing_ratio": ${params.alignment_valis_non_rigid_grid_spacing_ratio},
      "maximum_iterations": ${params.alignment_valis_non_rigid_maximum_iterations},
      "smoothing_sigma_ratio": ${params.alignment_valis_non_rigid_smoothing_sigma_ratio},
      "field_sample_spacing_px": ${params.alignment_valis_field_sample_spacing_px}
    },
    "qc": {
      "minimum_global_dice": ${params.alignment_qc_minimum_global_dice},
      "minimum_global_mutual_information": ${params.alignment_qc_minimum_global_mutual_information},
      "minimum_global_inliers": ${params.alignment_qc_minimum_global_inliers},
      "minimum_inlier_coverage": ${params.alignment_qc_minimum_inlier_coverage},
      "non_rigid_minimum_nmi_improvement": ${params.alignment_qc_non_rigid_minimum_nmi_improvement},
      "non_rigid_maximum_p95_displacement_um": ${params.alignment_qc_non_rigid_maximum_p95_displacement_um},
      "non_rigid_maximum_coherent_rotation_degrees": ${params.alignment_qc_non_rigid_maximum_coherent_rotation_degrees},
      "non_rigid_maximum_coherent_translation_um": ${params.alignment_qc_non_rigid_maximum_coherent_translation_um},
      "non_rigid_maximum_density_correlation_degradation": ${params.alignment_qc_non_rigid_maximum_density_correlation_degradation},
      "non_rigid_maximum_robust_score_degradation": ${params.alignment_qc_non_rigid_maximum_robust_score_degradation},
      "non_rigid_maximum_tissue_dice_degradation": ${params.alignment_qc_non_rigid_maximum_tissue_dice_degradation}
    }
  },
  "legacy_spateo": {
    "mode": "${params.alignment_spateo_mode}",
    "device": "${params.alignment_device}",
    "dtype": "${params.alignment_dtype}",
    "selected_mode": "${params.alignment_selected_mode}",
    "max_iter": ${params.alignment_max_iter},
    "nonrigid_start_iter": ${params.alignment_nonrigid_start_iter},
    "beta": ${params.alignment_beta},
    "lambda_vf": ${params.alignment_lambda_vf},
    "k": ${params.alignment_k},
    "partial_robust_level": ${params.alignment_partial_robust_level},
    "allow_flip": ${params.alignment_allow_flip},
    "SVI_mode": ${params.alignment_svi_mode},
    "n_sampling": ${params.alignment_n_sampling},
    "sparse_top_k": ${params.alignment_sparse_top_k},
    "sparse_calculation_mode": ${params.alignment_sparse_calculation_mode},
    "use_chunk": ${params.alignment_use_chunk},
    "chunk_capacity": ${params.alignment_chunk_capacity},
    "use_hvg": ${params.alignment_use_hvg},
    "n_top_genes": ${params.alignment_n_top_genes},
    "use_pca": ${params.alignment_use_pca},
    "n_pcs": ${params.alignment_n_pcs},
    "max_alignment_cells": ${params.alignment_max_alignment_cells},
    "alignment_seed": ${params.alignment_seed},
    "rbf_neighbors": ${params.alignment_rbf_neighbors},
    "rbf_smoothing": ${params.alignment_rbf_smoothing},
    "max_nonrigid_anchors": ${params.alignment_max_nonrigid_anchors}
  }
}
JSON

    export PYTORCH_CUDA_ALLOC_CONF="${params.alignment_pytorch_cuda_alloc_conf}"
    merxen align --config align_config.json
    """
}

process ALIGN_QC {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/alignment_qc" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(merscope_zarr),
        val(xenium_zarr),
        path(align_out)

    output:
    tuple val(pair_id), path("alignment_qc_out")

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

    cat > alignment_qc_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "merscope_zarr_path": "${merscope_zarr}",
  "xenium_zarr_path": "${xenium_zarr}",
  "transform_json_path": "${align_out}/alignment_transform.json",
  "output_dir": "alignment_qc_out",
  "grid_rows": ${params.alignment_qc_grid_rows},
  "grid_cols": ${params.alignment_qc_grid_cols}
}
JSON

    merxen alignment-qc --config alignment_qc_config.json
    """
}
