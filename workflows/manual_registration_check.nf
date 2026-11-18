#!/usr/bin/env nextflow
// Manual landmark registration, landmark check: for every samplesheet row, fit the robust landmark field
// to the landmarks saved in the prepared store, check every click against the images, and write the check
// JSON, its figure and the check attribute of the store (docs/stages/manual-registration.md). A draft
// labelling is checked too. Landmarks labelled on another machine are checked with
// `merxen manual-registration-check --labelling <uploaded zarr.json>` (no attribute is written then).
nextflow.enable.dsl = 2

include { MANUAL_REGISTRATION_CHECK } from "./modules/manual_registration"

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
        file(params.samplesheet, checkIfExists: true), params, "check",
    )
    parsed.warnings.each { warning -> log.warn(warning) }
    def errors = parsed.errors + (parsed.errors ? [] : ManualRegistrationSheet.checkPreflight(parsed.rows, params))
    if (errors) {
        error("Preflight checks failed:\n" + errors.collect { e -> " - ${e}" }.join("\n"))
    }
    def checks = ManualRegistrationSheet.plannedChecks(parsed.rows, params)
    def checksById = checks.collectEntries { check -> [(check.check_id): check] }
    checks.each { check -> log.info("[${check.check_id}] landmark check -> ${check.check_dir}") }

    checked_ch = MANUAL_REGISTRATION_CHECK(
        channel.fromList(checks.collect { check -> tuple(check.check_id, check.id, check.config_json) })
    )
    checked_ch.subscribe { checkId, _id, _out ->
        log.info("[${checkId}] check: ${checksById[checkId].check_json}")
    }
}
