process VISUALIZE {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/visualization" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json)

    output:
    tuple val(pair_id), val(segmentation), path("visualize_out")

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

    cat > visualize_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "visualize_out",
  "samples": ${samples_json}
}
JSON

    merxen visualize --config visualize_config.json
    """
}
