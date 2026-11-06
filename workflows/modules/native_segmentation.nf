// NATIVE_LATEST: the latest store of a native (vendor-segmentation) row.
//
// The vendor masks, shapes and table of the built source store become the
// latest store at <outdir>/<pair>/<platform>/latest/latest_spatialdata.zarr
// (payloads hardlinked read-only, metadata copied, renamed elements written
// fresh); no Cellpose or ProSeg task runs. The task emits the SEGMENT tuple
// shape, with three distinct empty placeholders in the Cellpose mask,
// transcript CSV and nuclei mask slots (distinct names: ENRICH stages both mask
// slots).
process NATIVE_LATEST {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(native_config_json), path(source_zarr, stageAs: "native_source.zarr")

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        path("native_out/native_latest.zarr"),
        path("native_out/NO_CELLPOSE_MASK"),
        path("native_out/NO_TRANSCRIPTS_CSV"),
        path("native_out/NO_NUCLEI_MASK"),
        emit: segmentation
    path("native_out/native_latest_report.json"), emit: report

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

    mkdir -p native_out

    cat > native_latest_config.json <<'JSON'
${native_config_json}
JSON

    merxen native-latest --config native_latest_config.json

    : > native_out/NO_CELLPOSE_MASK
    : > native_out/NO_TRANSCRIPTS_CSV
    : > native_out/NO_NUCLEI_MASK
    """
}
