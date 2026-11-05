#!/usr/bin/env nextflow
// Manual landmark registration, step 1 of 2: convert/normalise both images of each samplesheet row and
// write the image-only prepared store the viewer labels (docs/stages/manual-registration.md).
nextflow.enable.dsl = 2

include { MANUAL_REGISTRATION_PREPARE } from "./modules/manual_registration"
include { VIEWER_CACHE } from "./modules/viewer_cache"

workflow {
    def siteConfigProblems = SiteConfigCheck.problems(
        workflow.profile, workflow.containerEngine, params, System.getenv(),
    )
    if (siteConfigProblems) {
        error(siteConfigProblems.join("\n"))
    }
    if (!params.samplesheet) {
        error "Missing required parameter: --samplesheet"
    }
    def parsed = ManualRegistrationSheet.parse(
        file(params.samplesheet, checkIfExists: true), params, "prepare",
    )
    parsed.warnings.each { warning -> log.warn(warning) }
    if (parsed.errors) {
        error("Preflight checks failed:\n" + parsed.errors.collect { e -> " - ${e}" }.join("\n"))
    }
    def rowsById = parsed.rows.collectEntries { row -> [(row.id): row] }

    prepare_inputs_ch = channel.fromList(
        parsed.rows.collect { row -> tuple(row.id, ManualRegistrationSheet.prepareConfigJson(row, params)) }
    )
    prepared_ch = MANUAL_REGISTRATION_PREPARE(prepare_inputs_ch)

    viewer_cache_inputs_ch = prepared_ch.map { id, _prepareOut ->
        def row = rowsById[id]
        tuple(
            "${id}|PREPARED",
            id,
            "PREPARED",
            ManualRegistrationSheet.viewerCacheConfigJson(row, params),
            file(ManualRegistrationSheet.preparedStorePath(params, row)),
        )
    }
    VIEWER_CACHE(viewer_cache_inputs_ch).subscribe { _key, id, _platform, _zarr, _out ->
        log.info(
            "[${id}] prepared store ready: ${ManualRegistrationSheet.preparedStorePath(params, rowsById[id])}. " +
            "Next: viewer > Dataset loader > Manual Registration > Load image data; place landmarks, press " +
            "Finish labelling, then run workflows/manual_registration_register.nf with the same samplesheet."
        )
    }
}
