process COMPUTE_CORTICAL_DEPTH {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/compute_cortical_depth" }, mode: "symlink", overwrite: true

    input:
    tuple val(key),
        val(pair_id),
        val(platform),
        val(cortical_depth_config_json),
        path(latest_zarr)

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        path("latest_input.zarr"),
        path("compute_cortical_depth_out")

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

    if [[ ! -e latest_input.zarr ]]; then
        ln -s ${latest_zarr} latest_input.zarr
    fi

    cat > cortical_depth_config.json <<'JSON'
${cortical_depth_config_json}
JSON

    merxen compute-cortical-depth --config cortical_depth_config.json
    """
}
