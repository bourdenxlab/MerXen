// Optional acceptance harness; production segmentation and published masks are inputs only.
nextflow.enable.dsl = 2

params.source_root = null
params.python = null
params.config = null
params.source_zarr = null
params.baseline_mask = null
params.baseline_stats = null
params.protected_root = null
params.gpu_lock_file = null
params.image_key = null
params.tile_index = 44
params.outdir = null

process HITL0_TILE_CHECK {
    executor 'local'
    cpus 4
    memory '32 GB'
    maxForks 1
    cache false
    publishDir params.outdir, mode: 'copy', overwrite: false

    beforeScript """
        set -euo pipefail
        exec 9<"${params.gpu_lock_file}"
        echo "Waiting for MerXen GPU lock" >&2
        flock 9
        echo "Acquired MerXen GPU lock" >&2
    """

    input:
    path checker

    output:
    path 'tile_check_out'

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS=${task.cpus}
    export OPENBLAS_NUM_THREADS=${task.cpus}
    export MKL_NUM_THREADS=${task.cpus}
    export PYTHONPATH="${params.source_root}/src"
    "${params.python}" "${checker}" \
        --config "${params.config}" \
        --source-zarr "${params.source_zarr}" \
        --baseline-mask "${params.baseline_mask}" \
        --baseline-stats "${params.baseline_stats}" \
        --protected-root "${params.protected_root}" \
        --image-key "${params.image_key}" \
        --tile-index ${params.tile_index} \
        --output-dir tile_check_out \
        --report-only
    """
}

workflow {
    ['source_root', 'python', 'config', 'source_zarr', 'baseline_mask',
     'baseline_stats', 'protected_root', 'gpu_lock_file', 'image_key', 'outdir'].each { name ->
        if (!params[name]) {
            error("Missing required --${name}")
        }
    }
    def protectedRoot = new File(params.protected_root.toString()).canonicalFile.toPath()
    def outputRoot = new File(params.outdir.toString()).canonicalFile.toPath()
    if (outputRoot.startsWith(protectedRoot)) {
        error('Acceptance --outdir must be outside --protected_root')
    }
    if (outputRoot.toFile().exists() && !outputRoot.toFile().isDirectory()) {
        error('Acceptance --outdir must be a directory')
    }
    if (outputRoot.toFile().exists() && outputRoot.toFile().list().length > 0) {
        error('Acceptance --outdir must be new or empty')
    }
    HITL0_TILE_CHECK(file("${params.source_root}/scripts/acceptance/hitl0_tile_check.py"))
}
