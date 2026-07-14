process DISTANCE_FROM_OBJECT_ANNOTATE {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/distance_from_object" }, mode: "symlink", overwrite: true

    input:
    tuple val(key),
        val(pair_id),
        val(platform),
        val(distance_config_json),
        path(latest_zarr),
        path(object_annotations)

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        path("latest_input.zarr"),
        path("distance_from_object_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    if [[ ! -e latest_input.zarr ]]; then
        ln -s ${latest_zarr} latest_input.zarr
    fi
    if [[ "${object_annotations}" != "object_annotations.geojson" ]]; then
        ln -s ${object_annotations} object_annotations.geojson
    fi

    cat > distance_from_object_config.json <<'JSON'
${distance_config_json}
JSON

    merxen distance-from-object --config distance_from_object_config.json
    """
}


process DISTANCE_FROM_OBJECT_COHORT {
    tag "${platform}"

    publishDir { "${params.outdir}/distance_from_object/cohort/${platform.toLowerCase()}" }, mode: "symlink", overwrite: true

    input:
    tuple val(platform),
        val(cohort_config_json),
        path(annotation_output_dirs, stageAs: "pair_outputs/dir??/*")

    output:
    tuple val(platform), path("distance_from_object_cohort_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    cat > distance_from_object_cohort_config.json <<'JSON'
${cohort_config_json}
JSON

    merxen distance-from-object-cohort --config distance_from_object_cohort_config.json
    """
}
