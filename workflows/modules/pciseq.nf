process PCISEQ_PREPARE {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(pair_id),
        val(segmentation),
        val(platform),
        val(config_json),
        path(source_h5ad, stageAs: "source_clustered.h5ad"),
        path(reference_file),
        path(label_files)

    output:
    tuple val(pair_id), val(segmentation), val(platform),
        path("pciseq_config.json"),
        path("source_clustered.h5ad"),
        path("pciseq_prepared")

    script:
    def config = new groovy.json.JsonSlurper().parseText(config_json.toString())
    config.reference_path = reference_file.toString()
    config.labels_path = !label_files ? null : (
        label_files instanceof Collection ? label_files[0] : label_files
    ).toString()
    def stagedConfig = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(config))
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    cat > pciseq_config.json <<'JSON'
${stagedConfig}
JSON
    python -m merxen.pciseq_stages prepare \
        --config pciseq_config.json \
        --output-dir pciseq_prepared
    """

    stub:
    """
    mkdir -p pciseq_prepared
    echo '{}' > pciseq_config.json
    """
}

process PCISEQ_COMPUTE {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(pair_id), val(segmentation), val(platform),
        path(config), path(source_h5ad), path(prepared_dir)

    output:
    tuple val(pair_id), val(segmentation), val(platform),
        path(config), path(source_h5ad), path(prepared_dir), path("pciseq_computed")

    script:
    """
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export LOKY_MAX_CPU_COUNT="${task.cpus}"
    python -m merxen.pciseq_compute \
        --prepared-dir "${prepared_dir}" \
        --output-dir pciseq_computed
    """

    stub:
    """
    mkdir -p pciseq_computed
    """
}

process PCISEQ_FINALIZE {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir {
        "${params.outdir}/${pair_id}/${segmentation}/pciseq"
    }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id), val(segmentation), val(platform),
        path(config), path(source_h5ad), path(prepared_dir), path(computed_dir)

    output:
    tuple val(pair_id), val(segmentation), val(platform), path("pciseq_out/${platform.toLowerCase()}")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    python -m merxen.pciseq_stages finalize \
        --config "${config}" \
        --prepared-dir "${prepared_dir}" \
        --computed-dir "${computed_dir}" \
        --output-dir "pciseq_out/${platform.toLowerCase()}"
    """

    stub:
    """
    mkdir -p pciseq_out/${platform.toLowerCase()}
    echo '{}' > pciseq_out/${platform.toLowerCase()}/stub_manifest.json
    """
}
