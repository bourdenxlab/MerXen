// Bounded, assignment-independent gene-density raster before Atera Cellpose.
process ATERA_TRANSCRIPT_DENSITY {
    tag "${pair_id}:${platform}"
    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/transcript_density" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(config_json), path(source_zarr)

    output:
    tuple val(key), val(pair_id), val(platform), path("density_out/transcript_density.zarr"), path("density_out/transcript_density.json"), emit: density

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    cat > density_config.json <<'JSON'
${config_json}
JSON
    merxen atera-transcript-density --config density_config.json --embed
    """
}
