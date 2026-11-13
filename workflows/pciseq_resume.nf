#!/usr/bin/env nextflow

// Resume a frozen preparation without rebuilding masks or reference profiles.
nextflow.enable.dsl = 2

include { PCISEQ_COMPUTE; PCISEQ_COMPUTE_BATCHED; PCISEQ_FINALIZE } from './modules/pciseq'

params.pciseq_prepared_dir = null
params.pciseq_prepared_config = null

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

    if (!params.pciseq_prepared_dir || !params.pciseq_prepared_config) {
        error "Provide --pciseq_prepared_dir and --pciseq_prepared_config"
    }
    def settingErrors = PciseqSettings.errors(params)
    if (settingErrors) {
        error settingErrors.join("\n")
    }
    def prepared = file(params.pciseq_prepared_dir, checkIfExists: true)
    def configFile = file(params.pciseq_prepared_config, checkIfExists: true)
    def parser = new groovy.json.JsonSlurper()
    def manifest = parser.parse(prepared.resolve('manifest.json').toFile())
    def config = parser.parse(configFile.toFile())
    ['pair_id', 'platform', 'segmentation'].each { field ->
        if (manifest[field] != config[field]) {
            error "Prepared pciSeq manifest and configuration disagree: ${field}"
        }
    }
    def source = java.nio.file.Paths.get(config.source_h5ad.toString())
    if (!source.isAbsolute()) {
        source = configFile.parent.resolve(source)
    }
    source = file(source, checkIfExists: true)
    if (!file(config.spatialdata_path, checkIfExists: true).toFile().isDirectory()) {
        error "Prepared pciSeq SpatialData path must be an existing directory"
    }
    log.info "Reusing prepared pciSeq inputs for ${manifest.pair_id}:" +
        "${manifest.platform}:${manifest.segmentation}; " +
        "${manifest.spot_count} transcripts, ${manifest.cell_count} cells"
    prepared_ch = Channel.of(tuple(
        manifest.pair_id,
        manifest.segmentation,
        manifest.platform,
        configFile,
        source,
        prepared,
    ))
    if (PciseqSettings.batched(params)) {
        computed_ch = PCISEQ_COMPUTE_BATCHED(prepared_ch)
    } else {
        computed_ch = PCISEQ_COMPUTE(prepared_ch)
    }
    PCISEQ_FINALIZE(computed_ch)
}
