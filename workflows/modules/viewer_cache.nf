process VIEWER_CACHE {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/viewer_cache" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(viewer_cache_config_json), path(latest_zarr)

    output:
    tuple val(key), val(pair_id), val(platform), path("latest_input.zarr"), path("viewer_cache_out")

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

    cat > viewer_cache_config.json <<'JSON'
${viewer_cache_config_json}
JSON

    merxen build-viewer-caches --config viewer_cache_config.json
    """
}
