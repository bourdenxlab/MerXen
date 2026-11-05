// Appended by test_atera_workflow.py, after an include of the real
// BUILD_SPATIALDATA module and the Groovy functions of workflows/main.nf. It
// resolves one ATERA samplesheet row as the workflow does (settings,
// preflight, build JSON) and runs the BUILD_SPATIALDATA process on it.

workflow {
    def row = [
        pair_id: "PAIR_A",
        analysis_mode: "atera",
        atera_dir: params.atera_dir,
        stop_stage: "build_spatialdata",
    ]
    def settings = rowSampleSettings(row, params)
    runPreflightChecks(row, settings, params)
    def buildConfig = buildConfigForPlatform(row, "PAIR_A", "ATERA")
    def json = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(buildConfig))
    println "CONFIG\t" + groovy.json.JsonOutput.toJson(buildConfig)
    BUILD_SPATIALDATA(channel.of(tuple("PAIR_A|ATERA", "PAIR_A", "ATERA", json)))
        .view { result -> "BUILT\t${result[3]}" }
}
