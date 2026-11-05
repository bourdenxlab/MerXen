#!/usr/bin/env nextflow
// Manual landmark registration, step 2 of 2: for every samplesheet row and preset, fit the landmarks the
// viewer saved, refine and write the registered result store; compare the runs of one pair
// (docs/stages/manual-registration.md).
nextflow.enable.dsl = 2

include { MANUAL_REGISTRATION_REGISTER; MANUAL_REGISTRATION_COMPARE } from "./modules/manual_registration"

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
        file(params.samplesheet, checkIfExists: true), params, "register",
    )
    parsed.warnings.each { warning -> log.warn(warning) }
    def errors = parsed.errors + (parsed.errors ? [] : ManualRegistrationSheet.registerPreflight(parsed.rows, params))
    if (errors) {
        error("Preflight checks failed:\n" + errors.collect { e -> " - ${e}" }.join("\n"))
    }
    def runs = ManualRegistrationSheet.plannedRuns(parsed.rows, params)
    def runsById = runs.collectEntries { run -> [(run.run_id): run] }
    runs.each { run -> log.info("[${run.run_id}] preset ${run.preset} -> ${run.run_dir}") }

    registered_ch = MANUAL_REGISTRATION_REGISTER(
        channel.fromList(runs.collect { run -> tuple(run.run_id, run.id, run.config_json) })
    )
    registered_ch.subscribe { runId, _id, _out ->
        log.info("[${runId}] result store: ${runsById[runId].result_store}")
    }
    compare_inputs_ch = registered_ch
        .map { runId, id, _out -> tuple(id, runsById[runId].summary_path) }
        .groupTuple()
        .filter { _id, summaryPaths -> summaryPaths.size() >= 2 }
        .map { id, summaryPaths ->
            tuple(id, ManualRegistrationSheet.compareConfigJson(id, summaryPaths.sort(false), params))
        }
    MANUAL_REGISTRATION_COMPARE(compare_inputs_ch).subscribe { id, _out ->
        log.info("[${id}] run comparison: ${ManualRegistrationSheet.comparisonCsvPath(params, id)}")
    }
}
