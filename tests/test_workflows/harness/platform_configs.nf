// Appended by test_platform_config_golden.py to the Groovy functions of
// workflows/main.nf and to wrappers around the inline config builders of its
// workflow block (renderSegment, renderEnrich, renderViewerCache,
// viewerTransformField, enrichRestartField, renderMaskImageQuantification) and
// the QC script (renderQcScript).
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

// ATERA rows: the BUILD cases (plan AT-4) and, below, the native-route
// cases of AT-6 (ateraNativeCases); no Cellpose SEGMENT case.
def ateraRows() {
    def dir = [atera_dir: "raw/atera"]
    return [
        dir_only: pairRow(dir),
        spatialdata_only: pairRow([atera_spatialdata_path: "raw/atera_spatialdata.zarr"]),
        dir_spatialdata_spec: pairRow(dir + [
            atera_spatialdata_path: "raw/atera_spatialdata.zarr",
            atera_spec_path: "raw/atera/experiment.spatial",
        ]),
        blank_fields: pairRow(dir + [atera_spatialdata_path: "  ", atera_spec_path: "  "]),
        other_platform_fields: pairRow(dir + [
            xenium_dir: "raw/xenium",
            xenium_spec_path: "raw/xenium/experiment.xenium",
            merscope_dir: "raw/merscope",
        ]),
        missing_inputs: pairRow([:]),
        blank_inputs: pairRow([atera_dir: "  ", atera_spatialdata_path: " "]),
    ]
}

def ateraBuildCases(String launch) {
    ateraRows().each { name, row ->
        def id = "${launch}/ATERA/${name}"
        attempt("build", id) { buildConfigForPlatform(row, "PAIR_A", "ATERA") }
        if (launch == "L0") {
            attempt("build", "${id}/inputs") { requirePlatformInput(row, "PAIR_A", "ATERA") }
        }
    }
}

// AT-6 native route of ATERA rows: NATIVE_LATEST, ENRICH (no masks), the
// viewer cache (base image), the analysis layers and sample JSONs. XENIUM and
// MERSCOPE cases are untouched; these only add case ids.
def ateraNativeCases(String launch) {
    def id = "${launch}/ATERA"
    attempt("native", "${id}/native_latest") {
        nativeLatestConfigForPlatform("PAIR_A", "ATERA", params)
    }
    ateraRows().each { name, row ->
        def rowId = "${id}/${name}"
        attempt("native", "${rowId}/segmentation_source") {
            segmentationSourceForRow(row, ["ATERA"], params)
        }
        attempt("enrich", "${rowId}/restart_latest_zarr") { enrichRestartField(row, "ATERA") }
        attempt("viewer_cache", "${rowId}/transform_field") {
            viewerTransformField(row, "ATERA")
        }
        attempt("viewer_cache", rowId) {
            renderViewerCache(
                "PAIR_A|ATERA",
                "PAIR_A",
                "ATERA",
                "enriched_latest.zarr",
                viewerTransformField(row, "ATERA"),
            )
        }
    }
    [
        "native", "vendor", "instrument", "cellpose", "cellpose_only", "proseg",
        "cellpose_proseg", "Cellpose-ProSeg",
    ].each { raw ->
        attempt("native", "${launch}/segmentation_source/[${raw}]") {
            normalizeSegmentationSource(raw)
        }
    }
    attempt("native", "${launch}/segmentation_source/unknown_error") {
        normalizeSegmentationSource("magic")
    }
    attempt("native", "${launch}/segmentation_source/atera_cellpose_only_error") {
        segmentationSourceForRow(
            [pair_id: "PAIR_A", segmentation_source: "cellpose_only"], ["ATERA"], params
        )
    }
    attempt("native", "${launch}/segmentation_source/xenium_native_error") {
        segmentationSourceForRow(
            [pair_id: "PAIR_A", segmentation_source: "native"], ["XENIUM"], params
        )
    }
    ["native", "cellpose_only", "cellpose_proseg"].each { source ->
        attempt("native", "${launch}/layers/${source}") { availableSegmentationLayers(source) }
        attempt("native", "${launch}/stage_order/${source}") {
            activeStageOrder(
                false, false, true, true, false, false, true, false, true, false,
                false, false, source,
            )
        }
    }
    attempt("enrich", id) {
        renderEnrich(
            "PAIR_A|ATERA",
            "PAIR_A",
            "ATERA",
            "native_latest.zarr",
            "NO_CELLPOSE_MASK",
            "NO_NUCLEI_MASK",
            "source_spatialdata.zarr",
        )
    }
    attempt("mask_image_quantification", id) {
        renderMaskImageQuantification(
            "PAIR_A|ATERA", "PAIR_A", "ATERA", "latest.zarr", "NO_CELLPOSE_MASK"
        )
    }
    ["original", "original_seg"].each { segmentation ->
        attempt("analysis_layers", "${id}/${segmentation}") {
            analysisLayerKeys("ATERA", segmentation)
        }
    }
    def paths = [ATERA: "results/PAIR_A/atera/latest/latest_spatialdata.zarr"]
    attempt("analysis_layers", "${launch}/samples/atera/original_seg") {
        samplesJsonForSegmentation("PAIR_A", ["ATERA"], paths, "original_seg")
    }
    attempt("analysis_layers", "${launch}/samples/atera/original_seg/spatial_genes") {
        spatialGeneSamplesJson(
            samplesJsonForSegmentation("PAIR_A", ["ATERA"], paths, "original_seg"),
            pairRow([atera_dir: "raw/atera"]),
        )
    }
}

// AT-9 ATERA ProSeg route (segmentation_source=native_proseg): plan, tile and
// stitch configs, the ENRICH and MIQ configs of a native_proseg row, the
// source and layer helpers. XENIUM and MERSCOPE cases are untouched.
def ateraProsegPlanText() {
    return groovy.json.JsonOutput.toJson([
        format_version: 1,
        source_build_fingerprint: "f" * 64,
        pixel_size_um: 0.21249222,
        mask_shape_px: [42548, 61369],
        tile_subset: null,
        tiles: [
            [tile_id: "c0_r0", grid: [0, 0], core_um: [0.0, 0.0, 2000.0, 2000.0],
             window_px: [0, 9883, 0, 9883], owned_labels: [1, 2, 5], run: true],
            [tile_id: "c1_r0", grid: [1, 0], core_um: [2000.0, 0.0, 4000.0, 2000.0],
             window_px: [8941, 19295, 0, 9883], owned_labels: [], run: false],
            [tile_id: "c0_r1", grid: [0, 1], core_um: [0.0, 2000.0, 2000.0, 4000.0],
             window_px: [0, 9883, 8941, 19295], owned_labels: [3, 4], run: true],
        ],
    ])
}

def ateraProsegCases(String launch) {
    def id = "${launch}/ATERA/native_proseg"
    attempt("atera_proseg", "${id}/plan") {
        ateraProsegPlanConfig("PAIR_A", "ATERA", params)
    }
    attempt("atera_proseg", "${id}/model") { ateraProsegModelParams(params) }
    attempt("atera_proseg", "${id}/tile_specs") {
        ateraProsegTileSpecs(
            ateraProsegPlanText(), "PAIR_A", "ATERA", "data/atera_source.zarr", params,
        )
    }
    attempt("atera_proseg", "${id}/stitch") {
        ateraProsegStitchConfig("PAIR_A", "ATERA", ["c0_r1", "c0_r0"], params)
    }
    attempt("atera_proseg", "${id}/enrich") {
        renderEnrich(
            "PAIR_A|ATERA",
            "PAIR_A",
            "ATERA",
            "proseg_latest.zarr",
            "NO_CELLPOSE_MASK",
            "NO_NUCLEI_MASK",
            "source_spatialdata.zarr",
            "native_proseg",
        )
    }
    attempt("atera_proseg", "${id}/enrich_native_row") {
        renderEnrich(
            "PAIR_A|ATERA",
            "PAIR_A",
            "ATERA",
            "native_latest.zarr",
            "NO_CELLPOSE_MASK",
            "NO_NUCLEI_MASK",
            "source_spatialdata.zarr",
            "native",
        )
    }
    attempt("atera_proseg", "${id}/mask_image_quantification") {
        renderMaskImageQuantification(
            "PAIR_A|ATERA", "PAIR_A", "ATERA", "latest.zarr", "NO_CELLPOSE_MASK",
            "native_proseg",
        )
    }
    attempt("atera_proseg", "${id}/layers") { availableSegmentationLayers("native_proseg") }
    attempt("atera_proseg", "${id}/stage_order") {
        activeStageOrder(
            false, false, true, true, false, false, true, false, true, false,
            false, false, "native_proseg",
        )
    }
    ["native_proseg", "vendor_proseg", "instrument_proseg", "Vendor-ProSeg", "proseg"].each {
        raw ->
            attempt("atera_proseg", "${launch}/segmentation_source/[${raw}]") {
                normalizeSegmentationSource(raw)
            }
    }
    attempt("atera_proseg", "${id}/row_source") {
        segmentationSourceForRow(
            [pair_id: "PAIR_A", segmentation_source: "native_proseg"], ["ATERA"], params
        )
    }
    attempt("atera_proseg", "${launch}/segmentation_source/xenium_native_proseg_error") {
        segmentationSourceForRow(
            [pair_id: "PAIR_A", segmentation_source: "native_proseg"], ["XENIUM"], params
        )
    }
    ["proseg_vendor", "vendor_proseg"].each { segmentation ->
        attempt("atera_proseg", "${id}/layer_keys/${segmentation}") {
            analysisLayerKeys("ATERA", segmentation)
        }
    }
    def notes = []
    attempt("atera_proseg", "${id}/mender_default") {
        filterSegmentationsForSource(
            ["proseg_hybrid"], "native_proseg", false, "mender_segmentations", "PAIR_A",
            notes,
        ) + notes
    }
    attempt("atera_proseg", "${id}/published_restart_error") {
        publishedAteraProsegOutput(params, "PAIR_A", "ATERA", "proseg_latest.zarr")
    }
}

// AT-7 ATERA Cellpose route (segmentation_source=cellpose_proseg): the Cellpose
// segment config (DAPI + 18S on morphology_focus, no CSV), the Cellpose plan,
// tile and stitch configs, ENRICH and MIQ of the row, layers and filters.
def ateraCellposePlanText() {
    return groovy.json.JsonOutput.toJson([
        format_version: 1,
        source_build_fingerprint: "f" * 64,
        pixel_size_um: 0.21249222,
        mask_shape_px: [42548, 61369],
        tile_subset: null,
        prior: [
            kind: "cellpose",
            mask_path: "data/seg/cellpose_masks_tiled.npy",
            mask_sha256: "a" * 64,
            mask_stat: [10443000000L, 1762000000000000000L],
            cellprob_path: "data/seg/cellpose_cellprobs_tiled.npy",
            cellprob_sha256: "b" * 64,
            cellprob_stat: [10443000000L, 1762000000000000001L],
            fingerprint: "c" * 64,
        ],
        prior_weight: [rule: "whole_section_auto", effective: 93.5, n_gene_rows: 1400000000,
            n_labels: 149733],
        tiles: [
            [tile_id: "c0_r0", grid: [0, 0], core_um: [0.0, 0.0, 2000.0, 2000.0],
             window_px: [0, 9883, 0, 9883], owned_labels: [1, 2, 5], run: true],
            [tile_id: "c0_r1", grid: [0, 1], core_um: [0.0, 2000.0, 2000.0, 4000.0],
             window_px: [0, 9883, 8941, 19295], owned_labels: [3, 4], run: true],
        ],
    ])
}

def ateraCellposeCases(String launch) {
    def id = "${launch}/ATERA/cellpose_proseg"
    def group = "atera_cellpose_proseg"
    def row = pairRow([atera_dir: "raw/atera"])
    attempt(group, "${id}/segment_meta") { segmentMetaForPlatform(row, "ATERA", params) }
    attempt(group, "${id}/segment") {
        segmentationConfigMap(
            "PAIR_A", "ATERA", "source_spatialdata.zarr",
            segmentMetaForPlatform(row, "ATERA", params), params,
        )
    }
    attempt(group, "${id}/segment_channels") {
        segmentMetaForPlatform(row + [atera_channels: "DAPI, 18S"], "ATERA", params)
    }
    attempt(group, "${id}/plan") {
        ateraProsegPlanConfig("PAIR_A", "ATERA", params, "cellpose")
    }
    attempt(group, "${id}/model") { ateraProsegModelParams(params, "cellpose") }
    attempt(group, "${id}/tile_specs") {
        ateraProsegTileSpecs(
            ateraCellposePlanText(), "PAIR_A", "ATERA", "data/atera_source.zarr", params,
        )
    }
    attempt(group, "${id}/stitch") {
        ateraProsegStitchConfig("PAIR_A", "ATERA", ["c0_r1", "c0_r0"], params, "cellpose")
    }
    attempt(group, "${id}/enrich") {
        renderEnrich(
            "PAIR_A|ATERA",
            "PAIR_A",
            "ATERA",
            "proseg_latest.zarr",
            "cellpose_masks_tiled.npy",
            "NO_NUCLEI_MASK",
            "source_spatialdata.zarr",
            "cellpose_proseg",
        )
    }
    attempt(group, "${id}/mask_image_quantification") {
        renderMaskImageQuantification(
            "PAIR_A|ATERA", "PAIR_A", "ATERA", "latest.zarr", "cellpose_masks_tiled.npy",
            "cellpose_proseg",
        )
    }
    attempt(group, "${id}/layers") {
        availableSegmentationLayers("cellpose_proseg", ["ATERA"])
    }
    attempt(group, "${id}/stage_order") {
        activeStageOrder(
            false, false, true, true, false, false, true, false, true, false,
            false, false, "cellpose_proseg", ["ATERA"],
        )
    }
    attempt(group, "${id}/routes") {
        [
            native_latest: usesNativeLatest("cellpose_proseg", ["ATERA"]),
            atera_cellpose: runsAteraCellpose("cellpose_proseg", ["ATERA"]),
            atera_proseg: runsAteraProseg("cellpose_proseg", ["ATERA"]),
            xenium_native_latest: usesNativeLatest("cellpose_proseg", ["XENIUM"]),
            xenium_atera_proseg: runsAteraProseg("cellpose_proseg", ["XENIUM"]),
            prior_kind: AteraProsegLayers.priorKindForSource("cellpose_proseg"),
        ]
    }
    attempt(group, "${id}/row_source") {
        segmentationSourceForRow(
            [pair_id: "PAIR_A", segmentation_source: "cellpose_proseg"], ["ATERA"], params
        )
    }
    ["reseg", "proseg", "proseg_mask", "cellpose", "original_seg"].each { segmentation ->
        attempt(group, "${id}/layer_keys/${segmentation}") {
            analysisLayerKeys("ATERA", segmentation)
        }
    }
    def notes = []
    attempt(group, "${id}/all_default") {
        filterSegmentationsForSource(
            normalizeAnalysisSegmentation("all"), "cellpose_proseg", false,
            "analysis_segmentation", "PAIR_A", notes, ["ATERA"],
        ) + notes
    }
    def menderNotes = []
    attempt(group, "${id}/mender_default") {
        filterSegmentationsForSource(
            ["proseg_hybrid"], "cellpose_proseg", false, "mender_segmentations",
            "PAIR_A", menderNotes, ["ATERA"],
        ) + menderNotes
    }
    attempt(group, "${id}/hybrid_strict_error") {
        filterSegmentationsForSource(
            ["proseg_hybrid"], "cellpose_proseg", true, "analysis_segmentation",
            "PAIR_A", [], ["ATERA"],
        )
    }
    def xeniumNotes = []
    attempt(group, "${launch}/XENIUM/all_default") {
        filterSegmentationsForSource(
            normalizeAnalysisSegmentation("all"), "cellpose_proseg", false,
            "analysis_segmentation", "PAIR_A", xeniumNotes, ["XENIUM"],
        ) + xeniumNotes
    }
    attempt(group, "${launch}/XENIUM/vendor_layer_strict_error") {
        filterSegmentationsForSource(
            ["proseg_vendor"], "cellpose_proseg", true, "analysis_segmentation",
            "PAIR_A", [], ["XENIUM"],
        )
    }
    attempt(group, "${id}/published_mask_error") {
        publishedAteraCellposeMask(params, "PAIR_A", "ATERA")
    }
    mixedSegmentationCases(group, launch)
}

// A group word mixed with layer names in a samplesheet value (review of AT-7):
// the group's members a source lacks fall back silently, the named ones are
// strict. One case per platform, source, setting and value.
def mixedSegmentationCases(String group, String launch) {
    def routes = [
        XENIUM: ["cellpose_proseg"],
        MERSCOPE: ["cellpose_proseg"],
        ATERA: ["native", "native_proseg", "cellpose_proseg"],
    ]
    def settings = [
        analysis_segmentation: [
            values: [
                "all,original_seg", "both,proseg_mask", "original_seg,all",
                "all,proseg_hybrid", "both,proseg_vendor",
            ],
            normalizer: { value -> normalizeAnalysisSegmentation(value) },
        ],
        mender_segmentations: [
            values: ["all,reseg", "all,original_seg", "all,proseg_vendor"],
            normalizer: { value -> normalizeMenderSegmentations(value) },
        ],
    ]
    routes.each { platform, sources ->
        sources.each { source ->
            settings.each { label, setting ->
                setting.values.each { value ->
                    def notes = []
                    def named = explicitlyNamedSegmentations(value, setting.normalizer)
                    def lacking = named - availableSegmentationLayers(source, [platform])
                    def key = value.replace(",", "+") + (lacking ? "_error" : "")
                    attempt(group, "${launch}/mixed/${platform}/${source}/${label}/${key}") {
                        resolveRowSegmentations(
                            value, value, setting.normalizer, source, label, "PAIR_A",
                            notes, [platform],
                        ) + notes
                    }
                }
            }
        }
    }
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
    attempt("mask_image_quantification", id) {
        renderMaskImageQuantification(
            "PAIR_A|${platform}", "PAIR_A", platform, "latest.zarr", "mask.npy"
        )
    }
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

// AT-6 QC of native ATERA rows: the registration reference is the registry's
// original cell shape (atera_cell_boundaries), so the original layers render
// a reference equal to the tested shape, which the QC check then skips.
def ateraQcCases(String launch) {
    ["original", "original_seg"].each { segmentation ->
        attempt("qc", "${launch}/ATERA/${segmentation}") {
            def keys = analysisLayerKeys("ATERA", segmentation)
            qcHeredoc(renderQcScript(
                "PAIR_A",
                "ATERA",
                segmentation,
                "latest_input.zarr",
                keys.table_key,
                keys.shape_key,
                [cpus: 4],
            ).toString())
        }
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
        "atera", "a", " ATERA ",
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
    // An atera row builds ATERA only, whatever other platform fields it holds.
    def ateraRow = row + [atera_dir: "raw/atera"]
    activePlatformsForMode(normalizeAnalysisMode("atera")).each { platform ->
        attempt("modes", "${launch}/atera_row/atera/${platform}/build") {
            buildConfigForPlatform(ateraRow, "PAIR_A", platform)
        }
    }
}

workflow {
    def launch = params.golden_launch?.toString()
    if (!(launch in ["L0", "L1", "L2", "L3", "L4"])) {
        throw new IllegalArgumentException("--golden_launch must be L0 to L4")
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
        ateraBuildCases(launch)
        ateraNativeCases(launch)
        ateraQcCases(launch)
        ateraProsegCases(launch)
        ateraCellposeCases(launch)
    } else if (launch == "L4") {
        ateraBuildCases(launch)
    } else {
        buildAndSegmentCases(launch, "MERSCOPE")
    }
    println "DONE ${launch}"
}
