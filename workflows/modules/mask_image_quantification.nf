process MASK_IMAGE_QUANTIFICATION {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/mask_image_quantification" }, mode: "symlink", overwrite: true

    input:
    tuple val(key),
        val(pair_id),
        val(platform),
        val(mask_image_quantification_config_json),
        path(latest_zarr),
        path(mask_path)

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        path("latest_input.zarr"),
        path("mask_image_quantification_out")

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
    if [[ ! -e mask_image_quantification_input_mask.npy ]]; then
        ln -s ${mask_path} mask_image_quantification_input_mask.npy
    fi

    cat > mask_image_quantification_config.json <<'JSON'
${mask_image_quantification_config_json}
JSON

    merxen mask-image-quantification --config mask_image_quantification_config.json
    """
}
