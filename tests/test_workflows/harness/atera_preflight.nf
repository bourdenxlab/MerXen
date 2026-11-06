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
            segmentation_source: settings.segmentation_source,
            analysis_segmentations: settings.analysis_segmentations,
            mender_segmentations: settings.mender_segmentations,
            stage_order: settings.stage_order,
            selected_stages: settings.selected_stages,
            run_native_segment: settings.run_native_segment,
            run_segment_nuclei: settings.run_segment_nuclei,
            run_mecr: settings.run_mecr,
            run_mapmycells: settings.run_mapmycells,
            transcript_analysis: settings.spatial_gene_analysis_transcript_analysis_enabled,
            setting_notes: settings.setting_notes,
        ]
    } catch (Throwable error) {
        value = [error_class: error.getClass().getName(), message: error.getMessage()]
    }
    println "CASE\t${id}\t" + groovy.json.JsonOutput.toJson(value)
}

def checkStages(String id, Map row, Map settings) {
    def errors = []
    appendAteraPreflightChecks(errors, row, settings, params)
    println "CASE\t${id}\t" + groovy.json.JsonOutput.toJson([errors: errors])
}

workflow {
    def atera = [pair_id: "PAIR_A", analysis_mode: "atera", atera_dir: "raw/atera"]
    // Passing rows: the native route through clustering (and MENDER).
    check("atera/no_stop_stage", atera)
    check("atera/mender", atera + [mender_enabled: "true"])
    // An explicit stop at MENDER passes over the standalone MapMyCells stage.
    check("atera/stop_mender", atera + [mender_enabled: "true", stop_stage: "mender"])
    check("atera/stop_build", atera + [stop_stage: "build_spatialdata"])
    check("atera/only_segment", atera + [only_stage: "segment"])
    check("atera/start_enrich", atera + [start_stage: "enrich", stop_stage: "enrich"])
    check("atera/segmentation_source_native", atera + [segmentation_source: "vendor"])
    check("atera/analysis_original", atera + [analysis_segmentation: "original"])
    check("atera/analysis_all", atera + [analysis_segmentation: "all"])
    check("atera/mecr_off", atera + [mecr_enabled: "false"])
    // Refused rows, one message each.
    check("atera/stop_mapmycells", atera + [stop_stage: "mapmycells"])
    check("atera/only_mapmycells", atera + [only_stage: "mapmycells"])
    check(
        "atera/start_mapmycells",
        atera + [mender_enabled: "true", start_stage: "mapmycells", stop_stage: "mender"],
    )
    check("atera/pciseq", atera + [pciseq_enabled: "true"])
    check("atera/gaston", atera + [gaston_enabled: "true"])
    check("atera/mecr", atera + [mecr_enabled: "true"])
    check(
        "atera/transcript_analysis",
        atera + [spatial_gene_analysis_transcript_analysis_enabled: "true"],
    )
    check("atera/cortical_depth", atera + [cortical_depth_enabled: "true"])
    check("atera/distance_from_object", atera + [distance_from_object_enabled: "true"])
    check("atera/alignment", atera + [enable_alignment: "true"])
    check("atera/cellpose_only", atera + [segmentation_source: "cellpose_only"])
    check("atera/unknown_source", atera + [segmentation_source: "magic"])
    check("atera/analysis_reseg", atera + [analysis_segmentation: "reseg"])
    check("atera/mender_hybrid", atera + [mender_enabled: "true", mender_segmentations: "proseg_hybrid"])
    check("atera/segment_nuclei", atera + [only_stage: "segment_nuclei"])
    // Mixed samplesheet: the XENIUM row keeps its four layers and Cellpose.
    def xenium = [pair_id: "PAIR_B", analysis_mode: "xenium", xenium_dir: "raw/xenium"]
    check("mixed/xenium", xenium + [stop_stage: "build_spatialdata"])
    check("mixed/xenium_enrich", xenium + [stop_stage: "enrich"])
    check("mixed/xenium_native", xenium + [segmentation_source: "native"])
    check("mixed/atera", atera + [pair_id: "PAIR_C"])
    // A row whose stage selection is empty (no stage at all) is refused too.
    checkStages(
        "atera/no_stage",
        atera + [pair_id: "PAIR_E"],
        [pair_id: "PAIR_E", active_platforms: ["ATERA"], selected_stages: [], stage_order: []],
    )
    println "DONE"
}
