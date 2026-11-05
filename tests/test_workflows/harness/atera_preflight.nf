// Appended by test_atera_workflow.py to the Groovy functions of
// workflows/main.nf: resolves each samplesheet row with rowSampleSettings and
// runs runPreflightChecks on it, as the workflow block does for every row, and
// prints one line per row:
//
//     CASE <TAB> <case id> <TAB> <compact JSON of the result or error>
//
// The workflow runs the checks inside one map over all rows, so an error in
// any row aborts the whole run (a mixed samplesheet included).

def check(String id, Map row) {
    def value
    try {
        def settings = rowSampleSettings(row, params)
        runPreflightChecks(row, settings, params)
        value = [
            result: "OK",
            active_platforms: settings.active_platforms,
            selected_stages: settings.selected_stages,
        ]
    } catch (Throwable error) {
        value = [error_class: error.getClass().getName(), message: error.getMessage()]
    }
    println "CASE\t${id}\t" + groovy.json.JsonOutput.toJson(value)
}

workflow {
    def atera = [pair_id: "PAIR_A", analysis_mode: "atera", atera_dir: "raw/atera"]
    check("atera/stop_build", atera + [stop_stage: "build_spatialdata"])
    check("atera/no_stop_stage", atera)
    check("atera/only_build", atera + [only_stage: "build_spatialdata"])
    check("atera/only_segment", atera + [only_stage: "segment"])
    check("atera/stop_segment", atera + [stop_stage: "segment"])
    check(
        "mixed/xenium",
        [
            pair_id: "PAIR_B",
            analysis_mode: "xenium",
            xenium_dir: "raw/xenium",
            stop_stage: "build_spatialdata",
        ],
    )
    check("mixed/atera", atera + [pair_id: "PAIR_C"])
    check("mixed/atera_stop_build", atera + [pair_id: "PAIR_D", stop_stage: "build_spatialdata"])
    println "DONE"
}
