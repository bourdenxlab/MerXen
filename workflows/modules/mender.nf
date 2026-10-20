process MENDER_PREPARE {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(clustered_h5ad, stageAs: "source_clustered.h5ad"),
        val(_terminal_token)

    output:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path("mender_config.json"),
        path("mender_prepare_out")

    script:
    def targetKJson = params.mender_target_k == null ? "null" : (params.mender_target_k as int).toString()
    def cellStateKeyJson = groovy.json.JsonOutput.toJson(params.mender_cell_state_key.toString())
    def missingPolicyJson = groovy.json.JsonOutput.toJson(params.mender_missing_state_policy.toString())
    def nnModeJson = groovy.json.JsonOutput.toJson(params.mender_nn_mode.toString())
    def countRepJson = groovy.json.JsonOutput.toJson(params.mender_count_rep.toString())
    def clusteringModeJson = groovy.json.JsonOutput.toJson(params.mender_clustering_mode.toString())
    def sourceTableJson = groovy.json.JsonOutput.toJson(source_spatialdata_table.toString())
    def nativeShapeJson = groovy.json.JsonOutput.toJson(native_shape_key.toString())
    def spatialdataPathJson = groovy.json.JsonOutput.toJson(spatialdata_path.toString())
    def sourceH5adJson = groovy.json.JsonOutput.toJson(clustered_h5ad.toString())
    def sourceH5adOriginJson = groovy.json.JsonOutput.toJson(source_h5ad_path.toString())
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"

    cat > mender_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "sample_id": "${sample_id}",
  "platform": "${platform}",
  "segmentation": "${segmentation}",
  "source_h5ad": ${sourceH5adJson},
  "source_h5ad_origin": ${sourceH5adOriginJson},
  "spatialdata_path": ${spatialdataPathJson},
  "source_spatialdata_table": ${sourceTableJson},
  "native_shape_key": ${nativeShapeJson},
  "output_dir": "mender_out",
  "cell_state_key": ${cellStateKeyJson},
  "missing_state_policy": ${missingPolicyJson},
  "nn_mode": ${nnModeJson},
  "radius_um": ${params.mender_radius_um},
  "n_scales": ${params.mender_n_scales},
  "count_rep": ${countRepJson},
  "include_self": ${params.mender_include_self},
  "clustering_mode": ${clusteringModeJson},
  "leiden_resolution": ${params.mender_leiden_resolution},
  "target_k": ${targetKJson},
  "random_seed": ${params.mender_random_seed},
  "run_umap": ${params.mender_run_umap},
  "write_spatialdata_table": ${params.mender_write_spatialdata_table},
  "figure_dpi": ${params.mender_figure_dpi}
}
JSON

    python -m merxen.mender_stages prepare \
        --config mender_config.json \
        --output-dir mender_prepare_out
    """
}

process MENDER_COMPUTE {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(mender_config),
        path(prepared_dir)

    output:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(mender_config),
        path(prepared_dir),
        path("mender_compute_out")

    script:
    """
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"

    python -m merxen.mender_compute \
        --config "${mender_config}" \
        --input-dir "${prepared_dir}" \
        --output-dir mender_compute_out
    """
}

process MENDER_FINALIZE {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir {
        "${params.outdir}/${pair_id}/${segmentation}/mender"
    }, mode: "copy", overwrite: true, pattern: "mender_out/**"

    input:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(mender_config),
        path(prepared_dir),
        path(computed_dir),
        path(clustered_h5ad, stageAs: "source_clustered.h5ad")

    output:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(mender_config),
        path("mender_out/${platform.toLowerCase()}")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"

    python -m merxen.mender_stages finalize \
        --config "${mender_config}" \
        --source-h5ad "${clustered_h5ad}" \
        --prepared-dir "${prepared_dir}" \
        --computed-dir "${computed_dir}" \
        --output-dir mender_out
    """
}

process MENDER_IMPORT {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir {
        "${params.outdir}/${pair_id}/${segmentation}/mender/mender_out/${platform.toLowerCase()}"
    }, mode: "copy", overwrite: true

    input:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(spatialdata_path),
        val(source_spatialdata_table),
        val(native_shape_key),
        val(source_h5ad_path),
        path(mender_config),
        path(finalized_dir)

    output:
    tuple val(task_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        path("spatialdata_import_manifest.json")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"

    python -m merxen.mender_stages import \
        --config "${mender_config}" \
        --finalized-dir "${finalized_dir}" \
        --output-dir .
    """
}
