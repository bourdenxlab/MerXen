process ENRICH {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/enrichment" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(enrich_config_json), path(latest_zarr), path(mask_path), path(nuclei_mask_path)

    output:
    tuple val(key), val(pair_id), val(platform), path("latest_input.zarr"), path("enrich_out")

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

    ln -s ${latest_zarr} latest_input.zarr
    ln -s ${mask_path} enrich_input_mask.npy
    ln -s ${nuclei_mask_path} enrich_input_nuclei_mask.npy

    cat > enrich_config.json <<'JSON'
${enrich_config_json}
JSON

    merxen enrich --config enrich_config.json
    """
}
