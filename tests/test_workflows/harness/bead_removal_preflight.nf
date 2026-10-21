// Appended to the Groovy functions of workflows/main.nf by
// test_spatialdata_build_config.py: runs the bead-removal preflight against a
// published source zarr and prints one line per case.

def mkSettings(Map extra) {
    return [
        pair_id: "p1",
        active_platforms: ["MERSCOPE"],
        run_build: false,
        need_build_results: true,
    ] + extra
}

def check(String label, Map row, Map settings) {
    def errors = []
    appendMerscopeBeadRemovalPreflightChecks(errors, row, settings, params)
    println "CASE ${label}: " + (errors ? "ERROR" : "OK")
}

def writeRecord(File zarr, Map attributes) {
    new File(zarr, "zarr.json").text = groovy.json.JsonOutput.toJson(
        [zarr_format: 3, node_type: "group", attributes: attributes]
    )
}

def beadAttributes(Map beadRemoval) {
    return [merxen_image_preprocessing: [bead_removal: beadRemoval]]
}

workflow {
    def root = new File(params.root)
    def zarr = new File(
        root, "p1/merscope/spatialdata/spatialdata_out/source_spatialdata.zarr"
    )
    zarr.mkdirs()
    def on = [enabled: true, fill: [enabled: false]]
    def onFill = [enabled: true, fill: [enabled: true]]
    def yes = [merscope_bead_removal: "true"]
    def no = [merscope_bead_removal: "false"]
    def withDir = yes + [merscope_dir: root.toString()]

    writeRecord(zarr, [:])
    check("legacy_off", no, mkSettings([:]))
    check("legacy_on", yes, mkSettings([:]))
    writeRecord(zarr, beadAttributes(on))
    check("on_on", yes, mkSettings([:]))
    check("on_off", no, mkSettings([:]))
    check("on_on_not_needed", yes, mkSettings([need_build_results: false]))
    check("on_on_xenium_only", yes, mkSettings([active_platforms: ["XENIUM"]]))
    writeRecord(zarr, beadAttributes(onFill))
    check("fill_on", yes, mkSettings([:]))
    check("blank_z_zarr_only", yes, mkSettings([run_build: true]))
    check("blank_z_with_dir", withDir, mkSettings([run_build: true]))
    check(
        "z_1_6_with_dir",
        withDir + [merscope_z_range: "1-6"],
        mkSettings([run_build: true])
    )
    new File(zarr, "zarr.json").delete()
    new File(zarr, ".zattrs").text = groovy.json.JsonOutput.toJson(beadAttributes(on))
    check("zattrs_on_off", no, mkSettings([:]))
    println "CONFIG " + groovy.json.JsonOutput.toJson(
        buildConfigForPlatform(
            withDir + [pair_id: "p1"], "p1", "MERSCOPE"
        ).merscope.bead_removal
    )
}
