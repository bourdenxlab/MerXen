process COMPARE {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/comparison" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(merscope_zarr),
        val(xenium_zarr),
        val(merscope_table_key),
        val(xenium_table_key)

    output:
    tuple val(pair_id), val(segmentation), path("compare_out")

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

    cat > compare_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "merscope_zarr_path": "${merscope_zarr}",
  "xenium_zarr_path": "${xenium_zarr}",
  "output_dir": "compare_out",
  "merscope_table_key": "${merscope_table_key}",
  "xenium_table_key": "${xenium_table_key}"
}
JSON

    merxen compare --config compare_config.json
    """
}
