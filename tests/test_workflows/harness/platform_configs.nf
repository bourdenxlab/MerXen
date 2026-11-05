// Appended by test_platform_config_golden.py to the Groovy functions of
// workflows/main.nf and to wrappers around the inline config builders of its
// workflow block (renderSegment, renderEnrich, renderViewerCache,
// viewerTransformField, enrichRestartField) and the QC script (renderQcScript).
// It renders the platform-dependent configs of MERSCOPE and XENIUM rows and
// prints one line per case:
//
//     CASE <TAB> <golden file> <TAB> <case id> <TAB> <compact JSON>
//
// A task receives prettyPrint(toJson(map)); prettyPrint is deterministic, so a
// byte-identical compact JSON means a byte-identical task input. A value that
// is already a string (prettyPrint output of a helper) is printed as a JSON
// string. A case that raises records the class and message of its error
// (expected only for ids ending in "_error" and the BUILD cases of the rows
// without inputs).
//
// Nextflow strict syntax: only top-level functions may be called by name, so
// no local closure is called directly.

def emit(String group, String id, value) {
    println "CASE\t${group}\t${id}\t" + groovy.json.JsonOutput.toJson(value)
}

def attempt(String group, String id, Closure body) {
    def value
    try {
        value = body.call()
    } catch (Throwable error) {
        value = [error_class: error.getClass().getName(), message: error.getMessage()]
    }
    emit(group, id, value)
}

def pairRow(Map fields) {
    return [pair_id: "PAIR_A"] + fields
}

def xeniumRows() {
    def dir = [xenium_dir: "raw/xenium"]
    return [
        dir_only: pairRow(dir),
        spatialdata_only: pairRow([xenium_spatialdata_path: "raw/xenium_spatialdata.zarr"]),
        dir_spatialdata_spec: pairRow(dir + [
            xenium_spatialdata_path: "raw/xenium_spatialdata.zarr",
            xenium_spec_path: "raw/xenium/experiment.xenium",
        ]),
        channels: pairRow(dir + [xenium_channels: "DAPI, 18S ,ATP1A1,"]),
        min_qv: pairRow(dir + [xenium_min_qv: "30"]),
        voxel_layers: pairRow(dir + [xenium_voxel_layers: "3"]),
        cellpose_overrides: pairRow(dir + [
            xenium_cellpose_cellprob: "-1.5",
            xenium_cellpose_diameter: "28",
        ]),
        mask_filter_overrides: pairRow(dir + [
            xenium_mask_max_eccentricity: "0.95",
            xenium_cellpose_flow_threshold: "0.6",
            xenium_cellpose_final_max_area_um2: "900",
        ]),
        custom_model: pairRow(dir + [xenium_cellpose_model: "model_bundle"]),
        blank_fields: pairRow(dir + [
            xenium_spatialdata_path: "  ",
            xenium_spec_path: "  ",
            xenium_channels: "  ",
            xenium_min_qv: "  ",
            xenium_voxel_layers: "  ",
            xenium_cellpose_cellprob: "  ",
        ]),
        missing_inputs: pairRow([:]),
        blank_inputs: pairRow([xenium_dir: "  ", xenium_spatialdata_path: " "]),
    ]
}

def merscopeRows() {
    def dir = [merscope_dir: "raw/merscope"]
    return [
        dir_only: pairRow(dir),
        zarr_legacy_only: pairRow([merscope_zarr_path: "raw/merscope_legacy.zarr"]),
        spatialdata_and_zarr: pairRow([
            merscope_spatialdata_path: "raw/merscope_spatialdata.zarr",
            merscope_zarr_path: "raw/merscope_legacy.zarr",
        ]),
        z_range: pairRow(dir + [merscope_z_range: "1-5"]),
        z_range_malformed: pairRow(dir + [merscope_z_range: "3"]),
        transform: pairRow(dir + [
            merscope_transform_path: "raw/merscope/micron_to_mosaic_pixel_transform.csv",
        ]),
        prefix_channels_voxels: pairRow(dir + [
            merscope_image_prefix: "mosaic_",
            merscope_channels: "DAPI,PolyT,Cellbound1",
            merscope_voxel_layers: "5",
        ]),
        mask_eccentricity: pairRow(dir + [merscope_mask_max_eccentricity: "0.9"]),
        cellpose_overrides: pairRow(dir + [
            merscope_cellpose_cellprob: "-0.5",
            merscope_cellpose_diameter: "40",
            merscope_cellpose_flow_threshold: "0.3",
            merscope_cellpose_final_max_area_um2: "700",
        ]),
        bead_removal: pairRow(dir + [merscope_bead_removal: "true"]),
        bead_removal_off: pairRow(dir + [merscope_bead_removal: "false"]),
        blank_fields: pairRow(dir + [
            merscope_spatialdata_path: "  ",
            merscope_zarr_path: "  ",
            merscope_transform_path: "  ",
            merscope_z_range: "  ",
            merscope_image_prefix: "  ",
            merscope_channels: "  ",
            merscope_voxel_layers: "  ",
            merscope_bead_removal: "  ",
            merscope_mask_max_eccentricity: "  ",
        ]),
        missing_inputs: pairRow([:]),
        blank_inputs: pairRow([merscope_dir: "  ", merscope_zarr_path: " "]),
    ]
}

// A placeholder custom-model bundle under the launch directory, so the
// <prefix>_cellpose_model branch of CellposeSettings.settings renders its
// model fields (the weights are never read here).
def writeModelBundle() {
    def bundle = new File("model_bundle")
    bundle.mkdirs()
    new File(bundle, "model").text = "placeholder weights"
    new File(bundle, "model.json").text =
        groovy.json.JsonOutput.toJson([weights_sha256: "a" * 64])
}

def rowsFor(String platform) {
    return platform == "MERSCOPE" ? merscopeRows() : xeniumRows()
}

def segmentationNames() {
    return [
        "proseg",
        "reseg",
        "original",
        "original_seg",
        "proseg_hybrid",
        "cellpose",
        "proseg_mask",
        "proseg_geometry_assignment",
    ]
}

def segmentFor(Map row, String platform) {
    def meta = segmentMetaForPlatform(row, platform, params)
    return renderSegment(
        "PAIR_A|${platform}", "PAIR_A", platform, "source_spatialdata.zarr", meta, true, true
    )
}

def buildAndSegmentCases(String launch, String platform) {
    rowsFor(platform).each { name, row ->
        def id = "${launch}/${platform}/${name}"
        attempt("build", id) { buildConfigForPlatform(row, "PAIR_A", platform) }
        attempt("segment", id) { segmentFor(row, platform) }
    }
}

def rowCases(String launch, String platform) {
    rowsFor(platform).each { name, row ->
        def id = "${launch}/${platform}/${name}"
        attempt("build", "${id}/inputs") { requirePlatformInput(row, "PAIR_A", platform) }
        attempt("segment", "${id}/channels_for_row") {
            CellposeSettings.channelsForRow(row, platform)
        }
        attempt("enrich", "${id}/restart_latest_zarr") { enrichRestartField(row, platform) }
        attempt("viewer_cache", "${id}/transform_field") {
            viewerTransformField(row, platform)
        }
        attempt("viewer_cache", id) {
            renderViewerCache(
                "PAIR_A|${platform}",
                "PAIR_A",
                platform,
                "enriched_latest.zarr",
                viewerTransformField(row, platform),
            )
        }
        attempt("pciseq", "${id}/spatialdata_path") {
            // A Path (an Iterable of Paths) would recurse in toJson.
            pciseqSpatialdataPath(row, [pair_id: "PAIR_A"], platform, params).toString()
        }
    }
}

def qcHeredoc(String script) {
    def start = script.indexOf("<<JSON\n")
    def end = script.indexOf("\nJSON", start)
    if (start < 0 || end < 0) {
        throw new IllegalStateException("QC script has no JSON heredoc")
    }
    return script.substring(start + "<<JSON\n".length(), end)
}

def platformCases(String launch, String platform) {
    def id = "${launch}/${platform}"
    attempt("enrich", id) {
        renderEnrich(
            "PAIR_A|${platform}",
            "PAIR_A",
            platform,
            "latest.zarr",
            "mask.npy",
            "nuclei_mask.npy",
            "source_spatialdata.zarr",
        )
    }
    segmentationNames().each { segmentation ->
        def caseId = "${id}/${segmentation}"
        attempt("analysis_layers", caseId) { analysisLayerKeys(platform, segmentation) }
        attempt("qc", caseId) {
            def keys = analysisLayerKeys(platform, segmentation)
            qcHeredoc(renderQcScript(
                "PAIR_A",
                platform,
                segmentation,
                "latest_input.zarr",
                keys.table_key,
                keys.shape_key,
                [cpus: 4],
            ).toString())
        }
        attempt("pciseq", caseId) {
            pciseqConfigJson(
                "PAIR_A",
                segmentation,
                platform,
                "PAIR_A_${platform}",
                "results/PAIR_A/${platform.toLowerCase()}/latest/latest_spatialdata.zarr",
                [pair_id: "PAIR_A"],
                params,
            )
        }
    }
    attempt("analysis_layers", "${id}/unknown_segmentation_error") {
        analysisLayerKeys(platform, "not_a_segmentation")
    }
}

def samplesCases(String launch) {
    def paths = [
        MERSCOPE: "results/PAIR_A/merscope/latest/latest_spatialdata.zarr",
        XENIUM: "results/PAIR_A/xenium/latest/latest_spatialdata.zarr",
    ]
    def platformSets = [
        paired: ["MERSCOPE", "XENIUM"],
        merscope: ["MERSCOPE"],
        xenium: ["XENIUM"],
    ]
    platformSets.each { label, platforms ->
        segmentationNames().each { segmentation ->
            def id = "${launch}/samples/${label}/${segmentation}"
            attempt("analysis_layers", id) {
                samplesJsonForSegmentation("PAIR_A", platforms, paths, segmentation)
            }
            attempt("analysis_layers", "${id}/no_paths") {
                samplesJsonForSegmentation("PAIR_A", platforms, [:], segmentation)
            }
        }
    }
}

def modeCases(String launch) {
    def aliases = [
        "paired", "pair", "both", "merscope", "merfish", "m", "xenium", "x",
        " Paired ", "MERSCOPE", "Xenium", "both-", "", "multi_error",
    ]
    aliases.each { alias ->
        def isError = alias.endsWith("_error")
        def raw = isError ? alias.substring(0, alias.length() - 6) : alias
        attempt("modes", "${launch}/alias/[${raw}]" + (isError ? "_error" : "")) {
            def mode = normalizeAnalysisMode(raw)
            [mode: mode, active_platforms: activePlatformsForMode(mode)]
        }
    }
    attempt("modes", "${launch}/alias/null") {
        def mode = normalizeAnalysisMode(null)
        [mode: mode, active_platforms: activePlatformsForMode(mode)]
    }
    attempt("modes", "${launch}/active_platforms/unknown_error") {
        activePlatformsForMode("unknown")
    }
    def row = pairRow([
        merscope_dir: "raw/merscope",
        merscope_transform_path: "raw/merscope/micron_to_mosaic_pixel_transform.csv",
        xenium_dir: "raw/xenium",
        xenium_spec_path: "raw/xenium/experiment.xenium",
    ])
    ["paired", "merscope", "xenium"].each { mode ->
        activePlatformsForMode(normalizeAnalysisMode(mode)).each { platform ->
            def id = "${launch}/paired_row/${mode}/${platform}"
            attempt("modes", "${id}/build") { buildConfigForPlatform(row, "PAIR_A", platform) }
            attempt("modes", "${id}/segment") { segmentFor(row, platform) }
        }
    }
}

workflow {
    def launch = params.golden_launch?.toString()
    if (!(launch in ["L0", "L1", "L2", "L3"])) {
        throw new IllegalArgumentException("--golden_launch must be L0, L1, L2 or L3")
    }
    writeModelBundle()
    if (launch == "L0") {
        ["MERSCOPE", "XENIUM"].each { platform ->
            buildAndSegmentCases(launch, platform)
            rowCases(launch, platform)
            platformCases(launch, platform)
        }
        samplesCases(launch)
        modeCases(launch)
    } else {
        buildAndSegmentCases(launch, "MERSCOPE")
    }
    println "DONE ${launch}"
}
