#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.hitl_preparation_manifest = null
params.hitl_preparation_max_forks = 4

process PREPARE_HITL_VIEWER {
    tag "${dataset_alias}"
    cpus 4
    memory '64 GB'
    maxForks params.hitl_preparation_max_forks

    input:
    tuple val(dataset_alias), val(config_json)

    output:
    tuple val(dataset_alias), path('preparation_complete.json')

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"
    cat > preparation_config.json <<'JSON'
${config_json}
JSON
    merxen cellpose-hitl-prepare-viewer --config preparation_config.json
    python - <<'PY'
import json
import shutil
from pathlib import Path
config = json.loads(Path('preparation_config.json').read_text())
shutil.copy2(Path(config['output_dir']) / 'preparation.json', 'preparation_complete.json')
PY
    """
}

workflow {
    // The MerXen release of this run (docs/versioning.md): logged, and appended
    // to <outdir>/nextflow/merxen_versions.jsonl except under -preview.
    def pipelineVersion = PipelineVersion.launch(
        workflow, nextflow.version.toString(), file(params.outdir).toAbsolutePath(),
    )
    log.info(pipelineVersion.banner)
    if (pipelineVersion.warning) {
        log.warn(pipelineVersion.warning)
    }

    if (!params.hitl_preparation_manifest) {
        error('Provide --hitl_preparation_manifest with existing review/native stores and baseline grids')
    }
    def manifest = new groovy.json.JsonSlurper().parse(
        file(params.hitl_preparation_manifest).toFile())
    if (!(manifest instanceof List) || !manifest) {
        error('The preparation manifest must be a nonempty JSON list')
    }
    def stores = manifest.collect { new File(it.source_zarr.toString()).canonicalPath }
    if (stores.unique(false).size() != stores.size()) {
        error('Only one preparation task may write each review store')
    }
    def aliases = manifest.collect { it.dataset_alias }
    if (aliases.unique(false).size() != aliases.size()) {
        error('Preparation dataset aliases must be unique')
    }
    PREPARE_HITL_VIEWER(Channel.fromList(manifest).map { entry ->
        tuple(entry.dataset_alias,
            groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(entry)))
    })
}
