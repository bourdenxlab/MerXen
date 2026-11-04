// Appended by test_platform_registry_groovy.py to the Groovy functions of
// workflows/main.nf and the span wrappers of the golden harness
// (enrichRestartField, viewerTransformField, ...). It prints
//
//     REGISTRY <TAB> <compact JSON of PlatformSpec.REGISTRY>
//     UNKNOWN  <TAB> <case id> <TAB> <compact JSON of the result or error>
//
// The REGISTRY line feeds the parity test with src/merxen/platforms.py. The
// UNKNOWN lines call every site converted to a registry lookup with a platform
// that is not registered (ATERA is not supported before AT-4, and lookups are
// exact, so "merscope" is not either); each must raise instead of falling into
// another platform's branch.
//
// Nextflow strict syntax: only top-level functions may be called by name.

def printRegistry() {
    def registry = PlatformSpec.REGISTRY.collectEntries { name, spec -> [name, spec.toMap()] }
    println "REGISTRY\t" + groovy.json.JsonOutput.toJson(registry)
}

def recordUnknown(String id, Closure body) {
    def value
    try {
        value = [result: body.call()?.toString()]
    } catch (Throwable error) {
        value = [error_class: error.getClass().getName(), message: error.getMessage()]
    }
    println "UNKNOWN\t${id}\t" + groovy.json.JsonOutput.toJson(value)
}

def unknownRow() {
    return [
        pair_id: "PAIR_A",
        merscope_dir: "raw/merscope",
        merscope_spatialdata_path: "raw/merscope_spatialdata.zarr",
        merscope_transform_path: "raw/merscope/micron_to_mosaic_pixel_transform.csv",
        xenium_dir: "raw/xenium",
        xenium_spatialdata_path: "raw/xenium_spatialdata.zarr",
        xenium_spec_path: "raw/xenium/experiment.xenium",
    ]
}

def unknownPlatformCases(String platform) {
    def row = unknownRow()
    [
        "proseg",
        "reseg",
        "original",
        "original_seg",
        "proseg_hybrid",
        "cellpose",
        "proseg_mask",
        "proseg_geometry_assignment",
    ].each { segmentation ->
        recordUnknown("${platform}/analysis_layer_keys/${segmentation}") {
            analysisLayerKeys(platform, segmentation)
        }
    }
    recordUnknown("${platform}/pciseq_spatialdata_path") {
        pciseqSpatialdataPath(row, [pair_id: "PAIR_A"], platform, params)
    }
    recordUnknown("${platform}/enrich_restart_field") { enrichRestartField(row, platform) }
    recordUnknown("${platform}/viewer_transform_field") { viewerTransformField(row, platform) }
    recordUnknown("${platform}/require_platform_input") {
        requirePlatformInput(row, "PAIR_A", platform)
    }
    recordUnknown("${platform}/build_config") { buildConfigForPlatform(row, "PAIR_A", platform) }
    recordUnknown("${platform}/segment_meta") { segmentMetaForPlatform(row, platform, params) }
    recordUnknown("${platform}/channels_for_row") {
        CellposeSettings.channelsForRow(row, platform)
    }
    recordUnknown("${platform}/channels_for_row_override") {
        // The row overrides the channels, so no platform default is needed.
        CellposeSettings.channelsForRow(
            row + [atera_channels: "DAPI,18S", merscope_channels: "DAPI,18S"],
            platform,
        )
    }
}

workflow {
    printRegistry()
    unknownPlatformCases("ATERA")
    unknownPlatformCases("merscope")
    println "DONE"
}
