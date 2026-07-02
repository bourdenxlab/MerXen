nextflow.enable.dsl = 2

include { BUILD_SPATIALDATA } from "./modules/spatialdata_build"
include { ENSURE_PROSEG } from "./modules/proseg_bootstrap"
include { SEGMENT } from "./modules/segmentation"
include { ENRICH } from "./modules/enrichment"
include { MASK_IMAGE_QUANTIFICATION } from "./modules/mask_image_quantification"
include { COMPUTE_CORTICAL_DEPTH } from "./modules/compute_cortical_depth"
include { QC } from "./modules/qc"
include { ALIGN; ALIGN_QC } from "./modules/alignment"
include { COMPARE } from "./modules/comparison"
include { VISUALIZE } from "./modules/visualization"
include { CLUSTERING_SQUIDPY } from "./modules/clustering_squidpy"
include { MAPMYCELLS } from "./modules/mapmycells"

def parseChannels(rawValue, defaults) {
    if (rawValue == null) {
        return defaults
    }
    def values = rawValue
        .toString()
        .split(",")
        .collect { value -> value.trim() }
        .findAll { value -> value.length() > 0 }
    return values ? values : defaults
}

def parseRange(rawValue, fallbackStart = 0, fallbackEnd = 6) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return [fallbackStart as int, fallbackEnd as int]
    }
    def parts = rawValue.toString().split("-").collect { value -> value.trim() }
    if (parts.size() != 2) {
        return [fallbackStart as int, fallbackEnd as int]
    }
    return [parts[0] as int, parts[1] as int]
}

def intOrDefault(rawValue, defaultValue) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return defaultValue as int
    }
    return rawValue as int
}

def floatOrDefault(rawValue, defaultValue) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return defaultValue as float
    }
    return rawValue as float
}

def chooseField(row, names) {
    def result = names.find { name ->
        def value = row[name]
        value != null && value.toString().trim().length() > 0
    }
    return result ? row[result].toString().trim() : null
}

def normalizeAnalysisMode(rawValue) {
    def raw = rawValue == null ? "paired" : rawValue.toString().trim()
    if (!raw) {
        raw = "paired"
    }
    def key = raw
        .toLowerCase()
        .replaceAll(/[^a-z0-9]+/, "_")
        .replaceAll(/^_+|_+$/, "")
    def aliases = [
        "paired": "paired",
        "pair": "paired",
        "both": "paired",
        "merscope": "merscope",
        "merfish": "merscope",
        "m": "merscope",
        "xenium": "xenium",
        "x": "xenium",
    ]
    if (!aliases.containsKey(key)) {
        throw new IllegalArgumentException(
            "Unknown analysis_mode '${raw}'. Valid values: paired, merscope, xenium"
        )
    }
    return aliases[key]
}

def activePlatformsForMode(analysisMode) {
    if (analysisMode == "paired") {
        return ["MERSCOPE", "XENIUM"]
    }
    if (analysisMode == "merscope") {
        return ["MERSCOPE"]
    }
    if (analysisMode == "xenium") {
        return ["XENIUM"]
    }
    throw new IllegalArgumentException("Unknown analysis mode: ${analysisMode}")
}

def normalizeAnalysisSegmentation(rawValue) {
    def raw = rawValue == null ? "both" : rawValue.toString().trim()
    if (!raw) {
        raw = "both"
    }
    def aliases = [
        "both": ["reseg", "original_seg"],
        "all": ["reseg", "original_seg"],
        "reseg": ["reseg"],
        "resegmented": ["reseg"],
        "proseg": ["reseg"],
        "mosaik": ["reseg"],
        "original": ["original_seg"],
        "original_seg": ["original_seg"],
        "original_segmentation": ["original_seg"],
        "instrument": ["original_seg"],
        "instrument_seg": ["original_seg"],
        "instrument_segmentation": ["original_seg"],
    ]
    def selected = []
    raw
        .split(",")
        .collect { value -> value.trim() }
        .findAll { value -> value.length() > 0 }
        .each { value ->
            def key = value
                .toLowerCase()
                .replaceAll(/[^a-z0-9]+/, "_")
                .replaceAll(/^_+|_+$/, "")
            if (!aliases.containsKey(key)) {
                throw new IllegalArgumentException(
                    "Unknown analysis_segmentation '${value}'. Valid values: " +
                    "both, reseg, original_seg"
                )
            }
            aliases[key].each { segmentation ->
                if (!selected.contains(segmentation)) {
                    selected << segmentation
                }
            }
        }
    return selected ? selected : ["reseg", "original_seg"]
}

def requirePlatformInput(row, pairId, platform) {
    if (platform == "MERSCOPE") {
        def merscopeDir = chooseField(row, ["merscope_dir"])
        def merscopeSpatialdataPath = chooseField(
            row,
            ["merscope_spatialdata_path", "merscope_zarr_path"]
        )
        if (!merscopeDir && !merscopeSpatialdataPath) {
            error(
                "Samplesheet row for ${pairId} must provide " +
                "merscope_dir or merscope_spatialdata_path"
            )
        }
        return [inputDir: merscopeDir, spatialdataPath: merscopeSpatialdataPath]
    }

    if (platform == "XENIUM") {
        def xeniumDir = chooseField(row, ["xenium_dir"])
        def xeniumSpatialdataPath = chooseField(row, ["xenium_spatialdata_path"])
        if (!xeniumDir && !xeniumSpatialdataPath) {
            error(
                "Samplesheet row for ${pairId} must provide " +
                "xenium_dir or xenium_spatialdata_path"
            )
        }
        return [inputDir: xeniumDir, spatialdataPath: xeniumSpatialdataPath]
    }

    throw new IllegalArgumentException("Unknown platform: ${platform}")
}

def buildConfigForPlatform(row, pairId, platform) {
    def input = requirePlatformInput(row, pairId, platform)
    if (platform == "MERSCOPE") {
        def zRange = parseRange(row.merscope_z_range, 0, 6)
        return [
            dataset_name: "${pairId}_MERSCOPE",
            platform: "MERSCOPE",
            input_path: input.inputDir ?: input.spatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: input.spatialdataPath ?: null,
            merscope_transform_path: chooseField(row, ["merscope_transform_path"]) ?: null,
            merscope: [
                z_layers: (zRange[0]..zRange[1]).collect { layer -> layer as int },
            ],
            xenium: [:],
        ]
    }

    if (platform == "XENIUM") {
        return [
            dataset_name: "${pairId}_XENIUM",
            platform: "XENIUM",
            input_path: input.inputDir ?: input.spatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: input.spatialdataPath ?: null,
            xenium_spec_path: chooseField(row, ["xenium_spec_path"]) ?: null,
            merscope: [:],
            xenium: [:],
        ]
    }

    throw new IllegalArgumentException("Unknown platform: ${platform}")
}

def segmentMetaForPlatform(row, platform, params) {
    if (platform == "MERSCOPE") {
        return [
            channels: parseChannels(row.merscope_channels, ["DAPI", "PolyT"]),
            image_prefix: chooseField(row, ["merscope_image_prefix"]) ?: null,
            z_range: parseRange(row.merscope_z_range, 0, 6),
            transform_path: chooseField(row, ["merscope_transform_path"]) ?: null,
            xenium_spec_path: null,
            min_qv: null,
            voxel_layers: intOrDefault(
                row.merscope_voxel_layers,
                params.default_merscope_voxel_layers
            ),
        ]
    }

    if (platform == "XENIUM") {
        return [
            channels: parseChannels(row.xenium_channels, ["DAPI", "18S"]),
            image_prefix: null,
            z_range: null,
            transform_path: null,
            xenium_spec_path: chooseField(row, ["xenium_spec_path"]) ?: null,
            min_qv: floatOrDefault(row.xenium_min_qv, params.xenium_min_qv),
            voxel_layers: intOrDefault(
                row.xenium_voxel_layers,
                params.default_xenium_voxel_layers
            ),
        ]
    }

    throw new IllegalArgumentException("Unknown platform: ${platform}")
}

def samplesJsonForPlatforms(pairId, platforms, platformPaths = [:]) {
    def samples = platforms.collect { platform ->
        def sample = [
            sample_id: "${pairId}_${platform}",
            platform: platform,
        ]
        if (platformPaths.containsKey(platform)) {
            sample.zarr_path = platformPaths[platform].toString()
        }
        return sample
    }
    return groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(samples))
}

def analysisLayerKeys(platform, segmentation) {
    if (segmentation == "reseg") {
        return [
            table_key: "table_MOSAIK_proseg",
            shape_key: "MOSAIK_proseg",
        ]
    }
    if (segmentation == "original_seg") {
        return [
            table_key: "table_original",
            shape_key: platform == "MERSCOPE"
                ? "merscope_cell_boundaries"
                : "xenium_cell_boundaries",
        ]
    }
    throw new IllegalArgumentException("Unknown analysis segmentation: ${segmentation}")
}

def samplesJsonForSegmentation(pairId, platforms, platformPaths, segmentation) {
    def samples = platforms.collect { platform ->
        def layerKeys = analysisLayerKeys(platform, segmentation)
        def sample = [
            sample_id: "${pairId}_${platform}",
            platform: platform,
            segmentation: segmentation,
            table_key: layerKeys.table_key,
            shape_key: layerKeys.shape_key,
        ]
        if (platformPaths.containsKey(platform)) {
            sample.zarr_path = platformPaths[platform].toString()
        }
        return sample
    }
    return groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(samples))
}

def corticalDepthConfigForPlatform(
    row,
    pairId,
    platform,
    analysisSegmentations,
    params
) {
    def tables = analysisSegmentations.collect { segmentation ->
        def layerKeys = analysisLayerKeys(platform, segmentation)
        [
            segmentation: segmentation,
            table_key: layerKeys.table_key,
            shape_key: layerKeys.shape_key,
        ]
    }
    return [
        dataset_name: "${pairId}_${platform}",
        platform: platform,
        latest_zarr_path: "latest_input.zarr",
        output_dir: "compute_cortical_depth_out",
        tables: tables,
        pial_boundary_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "pial")
        ),
        wm_boundary_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "wm")
        ),
        side_boundary_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "side")
        ),
        exclusion_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "exclusion")
        ),
        ribbon_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "ribbon")
        ),
        annotation_path: optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "annotation")
        ),
        coordinate_unit_um: params.cortical_depth_coordinate_unit_um,
        raster_resolution_um: params.cortical_depth_raster_resolution_um,
        raster_padding_um: params.cortical_depth_raster_padding_um,
        boundary_band_um: params.cortical_depth_boundary_band_um,
        boundary_smoothing_window: params.cortical_depth_boundary_smoothing_window,
        streamline_spacing_um: params.cortical_depth_streamline_spacing_um,
        streamline_step_um: params.cortical_depth_streamline_step_um,
        streamline_max_steps: params.cortical_depth_streamline_max_steps,
        streamline_resample_points: params.cortical_depth_streamline_resample_points,
        side_boundary_distance_um: params.cortical_depth_side_boundary_distance_um,
        contour_levels: params.cortical_depth_contour_levels,
        write_spatialdata_table: params.cortical_depth_write_spatialdata_table,
    ]
}

def samplesJsonFromGroupedZarrs(pairId, activePlatforms, platformNames, zarrPaths) {
    def names = platformNames.collect { platformName -> platformName.toString() }
    def pathsByPlatform = [:]
    activePlatforms.each { platform ->
        def idx = names.indexOf(platform)
        if (idx < 0) {
            error "Missing ${platform} latest zarr for ${pairId}"
        }
        pathsByPlatform[platform] = zarrPaths[idx].toString()
    }
    return samplesJsonForPlatforms(pairId, activePlatforms, pathsByPlatform)
}

def normalizeStage(rawValue, paramName) {
    def raw = rawValue == null ? "" : rawValue.toString().trim()
    if (!raw) {
        throw new IllegalArgumentException("Missing required stage value: ${paramName}")
    }
    def key = raw
        .toLowerCase()
        .replaceAll(/[^a-z0-9]+/, "_")
        .replaceAll(/^_+|_+$/, "")
    def aliases = [
        "build": "build_spatialdata",
        "build_spatialdata": "build_spatialdata",
        "spatialdata": "build_spatialdata",
        "spatialdata_build": "build_spatialdata",
        "segment": "segment",
        "segmentation": "segment",
        "enrich": "enrich",
        "enrichment": "enrich",
        "mask_image_quantification": "mask_image_quantification",
        "image_quantification": "mask_image_quantification",
        "quantify_images": "mask_image_quantification",
        "mask_quantification": "mask_image_quantification",
        "compute_cortical_depth": "compute_cortical_depth",
        "cortical_depth": "compute_cortical_depth",
        "laplace_depth": "compute_cortical_depth",
        "qc": "qc",
        "align": "align",
        "alignment": "align",
        "align_qc": "align_qc",
        "alignment_qc": "align_qc",
        "compare": "compare",
        "comparison": "compare",
        "visualize": "visualize",
        "visualise": "visualize",
        "visualization": "visualize",
        "visualisation": "visualize",
        "cluster": "clustering_squidpy",
        "clustering": "clustering_squidpy",
        "clustering_squidpy": "clustering_squidpy",
        "squidpy": "clustering_squidpy",
        "cell_type_mapping": "mapmycells",
        "celltype_mapping": "mapmycells",
        "map_my_cells": "mapmycells",
        "mapmycells": "mapmycells",
    ]
    if (!aliases.containsKey(key)) {
        throw new IllegalArgumentException(
            "Unknown ${paramName} '${raw}'. Valid stages: " +
            "build_spatialdata, segment, enrich, mask_image_quantification, " +
            "compute_cortical_depth, qc, align, align_qc, " +
            "compare, visualize, clustering_squidpy, mapmycells"
        )
    }
    return aliases[key]
}

def activeStageOrder(
    alignmentEnabled,
    pairedMode,
    maskImageQuantificationEnabled,
    corticalDepthEnabled
) {
    def stages = ["build_spatialdata", "segment", "enrich"]
    if (maskImageQuantificationEnabled) {
        stages += ["mask_image_quantification"]
    }
    if (corticalDepthEnabled) {
        stages += ["compute_cortical_depth"]
    }
    stages += ["qc"]
    if (pairedMode && alignmentEnabled) {
        stages += ["align", "align_qc"]
    }
    if (pairedMode) {
        stages += ["compare"]
    }
    stages += ["visualize", "clustering_squidpy", "mapmycells"]
    return stages
}

def validateStage(stage, stages, paramName, alignmentEnabled) {
    if (!stages.contains(stage)) {
        def hint = ""
        if (stage in ["align", "align_qc"] && !alignmentEnabled) {
            hint = " Pass --enable_alignment true to use alignment stages."
        }
        if (stage == "compute_cortical_depth") {
            hint = " Pass --cortical_depth_enabled true to use cortical depth."
        }
        throw new IllegalArgumentException(
            "${paramName} '${stage}' is not active for this run.${hint} " +
            "Active stages: ${stages.join(', ')}"
        )
    }
}

def stageInRange(stage, startStage, stopStage, stages) {
    def stageIdx = stages.indexOf(stage)
    if (stageIdx < 0) {
        return false
    }
    return stageIdx >= stages.indexOf(startStage) && stageIdx <= stages.indexOf(stopStage)
}

def requireExistingPath(rawPath, label) {
    def p = java.nio.file.Paths.get(rawPath.toString()).toAbsolutePath().normalize()
    if (!p.toFile().exists()) {
        throw new IllegalArgumentException(
            "Missing expected ${label}: ${p}\n" +
            "This path is required because an upstream stage was skipped. " +
            "Either run from an earlier stage or restore the published output."
        )
    }
    return p
}

def isBlankPath(rawPath) {
    if (rawPath == null) {
        return true
    }
    def value = rawPath.toString().trim()
    return !value || value.toLowerCase() == "null"
}

def normalizedPath(rawPath) {
    return java.nio.file.Paths.get(rawPath.toString()).toAbsolutePath().normalize()
}

def appendPreflightFileCheck(errors, rawPath, label) {
    if (isBlankPath(rawPath)) {
        errors << "Missing required path parameter for ${label}"
        return
    }
    def path = normalizedPath(rawPath)
    if (!path.toFile().exists()) {
        errors << "Missing ${label}: ${path}"
    }
}

def findCachedTaxonomyMetadataPath(rawCacheDir) {
    if (isBlankPath(rawCacheDir)) {
        return null
    }
    def root = normalizedPath(rawCacheDir)
    def taxonomyRoot = root.resolve("abc_whb/metadata/WHB-taxonomy").toFile()
    if (!taxonomyRoot.isDirectory()) {
        return null
    }
    def matches = []
    taxonomyRoot.eachDir { versionDir ->
        def candidate = versionDir.toPath().resolve("cluster_annotation_term.csv")
        if (candidate.toFile().exists()) {
            matches << candidate
        }
    }
    return matches ? matches.sort { path -> path.toString() }[-1] : null
}

def appendClusteringSquidpyPreflightChecks(errors, settings, params) {
    def hierarchicalEnabled = boolOrDefault(
        params.clustering_squidpy_hierarchical_enabled,
        true,
        "clustering_squidpy_hierarchical_enabled",
    )
    if (!(settings.run_clustering_squidpy && hierarchicalEnabled)) {
        return
    }
    appendPreflightFileCheck(
        errors,
        params.clustering_squidpy_broad_marker_lookup_path,
        "CLUSTERING_SQUIDPY broad marker lookup",
    )
    if (isBlankPath(params.clustering_squidpy_broad_taxonomy_metadata_path)) {
        def cachedTaxonomy = findCachedTaxonomyMetadataPath(
            params.clustering_squidpy_broad_reference_cache_dir,
        )
        if (cachedTaxonomy == null) {
            errors << (
                "Missing CLUSTERING_SQUIDPY broad taxonomy metadata: " +
                "set --clustering_squidpy_broad_taxonomy_metadata_path " +
                "or provide a cache containing " +
                "abc_whb/metadata/WHB-taxonomy/*/cluster_annotation_term.csv"
            )
        }
    } else {
        appendPreflightFileCheck(
            errors,
            params.clustering_squidpy_broad_taxonomy_metadata_path,
            "CLUSTERING_SQUIDPY broad taxonomy metadata",
        )
    }
    if (!isBlankPath(params.clustering_squidpy_broad_cluster_membership_path)) {
        appendPreflightFileCheck(
            errors,
            params.clustering_squidpy_broad_cluster_membership_path,
            "CLUSTERING_SQUIDPY broad cluster membership metadata",
        )
    }
}

def appendMapMyCellsPreflightChecks(errors, settings, params) {
    if (!settings.run_mapmycells) {
        return
    }
    validateMapMyCellsParams(params)
    def plotsOnly = params.mapmycells_plots_only == null
        ? false
        : params.mapmycells_plots_only.toString().trim().toLowerCase() == "true"
    if (plotsOnly) {
        return
    }
    def referenceMode = params.mapmycells_reference_mode == null
        ? "both"
        : params.mapmycells_reference_mode.toString().trim().toLowerCase()
    if (referenceMode in ["whole_brain", "both"]) {
        appendPreflightFileCheck(
            errors,
            params.mapmycells_marker_lookup_path,
            "MAPMYCELLS whole-brain marker lookup",
        )
        appendPreflightFileCheck(
            errors,
            params.mapmycells_precomputed_stats_path,
            "MAPMYCELLS whole-brain precomputed stats",
        )
    }
}

def optionalNormalizedPathString(rawPath) {
    return isBlankPath(rawPath) ? null : normalizedPath(rawPath).toString()
}

def corticalDepthAnnotationPath(row, platform, role) {
    def prefix = platform.toString().toLowerCase()
    def candidatesByRole = [
        annotation: [
            "${prefix}_cortical_depth_annotation_geojson",
            "${prefix}_cortical_depth_annotations_geojson",
            "${prefix}_cortical_depth_annotation_path",
            "cortical_depth_annotation_geojson",
            "cortical_depth_annotations_geojson",
            "cortical_depth_annotation_path",
        ],
        pial: [
            "${prefix}_pial_boundary_geojson",
            "${prefix}_pia_boundary_geojson",
            "${prefix}_pial_boundary_path",
            "pial_boundary_geojson",
            "pia_boundary_geojson",
            "pial_boundary_path",
        ],
        wm: [
            "${prefix}_wm_boundary_geojson",
            "${prefix}_grey_white_boundary_geojson",
            "${prefix}_gray_white_boundary_geojson",
            "${prefix}_gm_wm_boundary_geojson",
            "${prefix}_wm_boundary_path",
            "wm_boundary_geojson",
            "grey_white_boundary_geojson",
            "gray_white_boundary_geojson",
            "gm_wm_boundary_geojson",
            "wm_boundary_path",
        ],
        side: [
            "${prefix}_side_boundary_geojson",
            "${prefix}_side_boundaries_geojson",
            "${prefix}_tissue_edge_geojson",
            "side_boundary_geojson",
            "side_boundaries_geojson",
            "tissue_edge_geojson",
        ],
        exclusion: [
            "${prefix}_exclusion_mask_geojson",
            "${prefix}_exclusion_masks_geojson",
            "${prefix}_cortical_depth_exclusion_geojson",
            "exclusion_mask_geojson",
            "exclusion_masks_geojson",
            "cortical_depth_exclusion_geojson",
        ],
        ribbon: [
            "${prefix}_cortical_ribbon_geojson",
            "${prefix}_ribbon_geojson",
            "${prefix}_cortical_ribbon_path",
            "cortical_ribbon_geojson",
            "ribbon_geojson",
            "cortical_ribbon_path",
        ],
    ]
    return chooseField(row, candidatesByRole[role] ?: [])
}

def appendCorticalDepthPreflightChecks(errors, row, settings, _params) {
    if (!settings.run_compute_cortical_depth) {
        return
    }
    settings.active_platforms.each { platform ->
        def labelPrefix = "COMPUTE_CORTICAL_DEPTH ${settings.pair_id}:${platform}"
        def annotationPath = corticalDepthAnnotationPath(row, platform, "annotation")
        if (!isBlankPath(annotationPath)) {
            appendPreflightFileCheck(
                errors,
                annotationPath,
                "${labelPrefix} combined annotation GeoJSON",
            )
        } else {
            appendPreflightFileCheck(
                errors,
                corticalDepthAnnotationPath(row, platform, "pial"),
                "${labelPrefix} pial boundary GeoJSON",
            )
        }
        ["side", "exclusion", "ribbon"].each { role ->
            def optionalPath = corticalDepthAnnotationPath(row, platform, role)
            if (!isBlankPath(optionalPath)) {
                appendPreflightFileCheck(
                    errors,
                    optionalPath,
                    "${labelPrefix} ${role} annotation GeoJSON",
                )
            }
        }
    }
}

def runPreflightChecks(row, settings, params) {
    def errors = []
    appendClusteringSquidpyPreflightChecks(errors, settings, params)
    appendMapMyCellsPreflightChecks(errors, settings, params)
    appendCorticalDepthPreflightChecks(errors, row, settings, params)
    if (errors) {
        throw new IllegalArgumentException(
            "Preflight checks failed for sample ${settings.pair_id} " +
            "(selected stages: ${settings.selected_stages.join(' -> ')}):\n" +
            errors.collect { errorMessage -> " - ${errorMessage}" }.join("\n")
        )
    }
}

def publishedDatasetPath(outdir, pairId, platform, suffix) {
    return "${outdir}/${pairId}/${platform.toLowerCase()}/${suffix}"
}

def publishedPairPath(outdir, pairId, suffix) {
    return "${outdir}/${pairId}/${suffix}"
}

def rowFieldOrDefault(row, fieldName, fallback) {
    def value = chooseField(row, [fieldName])
    return value == null ? fallback : value
}

def boolOrDefault(rawValue, defaultValue, label) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return defaultValue as boolean
    }
    def key = rawValue
        .toString()
        .trim()
        .toLowerCase()
    if (key in ["true", "t", "yes", "y", "1"]) {
        return true
    }
    if (key in ["false", "f", "no", "n", "0"]) {
        return false
    }
    throw new IllegalArgumentException(
        "Unknown boolean value for ${label}: '${rawValue}'. " +
        "Use true or false."
    )
}

def rowSampleSettings(row, params) {
    def pairId = row.pair_id?.toString()?.trim()
    if (!pairId) {
        error "Found samplesheet row with missing pair_id: ${row}"
    }

    def analysisMode = normalizeAnalysisMode(
        rowFieldOrDefault(row, "analysis_mode", params.analysis_mode)
    )
    def pairedMode = analysisMode == "paired"
    def activePlatforms = activePlatformsForMode(analysisMode)
    def analysisSegmentations = normalizeAnalysisSegmentation(
        rowFieldOrDefault(row, "analysis_segmentation", params.analysis_segmentation)
    )
    def requestedAlignmentEnabled = boolOrDefault(
        rowFieldOrDefault(row, "enable_alignment", params.enable_alignment),
        false,
        "enable_alignment for ${pairId}",
    )
    def alignmentEnabled = pairedMode && requestedAlignmentEnabled
    def maskImageQuantificationEnabled = boolOrDefault(
        rowFieldOrDefault(
            row,
            "mask_image_quantification_enabled",
            params.mask_image_quantification_enabled,
        ),
        true,
        "mask_image_quantification_enabled for ${pairId}",
    )
    def corticalDepthEnabled = boolOrDefault(
        rowFieldOrDefault(
            row,
            "cortical_depth_enabled",
            params.cortical_depth_enabled,
        ),
        false,
        "cortical_depth_enabled for ${pairId}",
    )

    def rowOnlyStageRaw = chooseField(row, ["only_stage"])
    def rowStartStageRaw = chooseField(row, ["start_stage"])
    def rowStopStageRaw = chooseField(row, ["stop_stage"])
    def hasRowStageRange = rowStartStageRaw != null || rowStopStageRaw != null
    def globalOnlyStageRaw = params.only_stage == null
        ? null
        : params.only_stage.toString().trim()
    if (!globalOnlyStageRaw) {
        globalOnlyStageRaw = null
    }

    def onlyStageRaw = rowOnlyStageRaw ?: (hasRowStageRange ? null : globalOnlyStageRaw)
    def startStageRaw = onlyStageRaw ?: (rowStartStageRaw ?: params.start_stage)
    def stopStageRaw = onlyStageRaw ?: (rowStopStageRaw ?: params.stop_stage)
    def startParamName = onlyStageRaw ? "only_stage" : "start_stage"
    def stopParamName = onlyStageRaw ? "only_stage" : "stop_stage"
    if (rowOnlyStageRaw) {
        startParamName = "samplesheet only_stage for ${pairId}"
        stopParamName = "samplesheet only_stage for ${pairId}"
    } else if (hasRowStageRange) {
        startParamName = "samplesheet start_stage for ${pairId}"
        stopParamName = "samplesheet stop_stage for ${pairId}"
    }

    def stageOrder = activeStageOrder(
        alignmentEnabled,
        pairedMode,
        maskImageQuantificationEnabled,
        corticalDepthEnabled,
    )
    def startStage = normalizeStage(startStageRaw, startParamName)
    def stopStage = normalizeStage(stopStageRaw, stopParamName)
    validateStage(startStage, stageOrder, startParamName, alignmentEnabled)
    validateStage(stopStage, stageOrder, stopParamName, alignmentEnabled)
    if (stageOrder.indexOf(startStage) > stageOrder.indexOf(stopStage)) {
        error(
            "Samplesheet row ${pairId} has start_stage '${startStage}' " +
            "after stop_stage '${stopStage}'."
        )
    }

    def runBuild = stageInRange("build_spatialdata", startStage, stopStage, stageOrder)
    def runSegment = stageInRange("segment", startStage, stopStage, stageOrder)
    def runEnrich = stageInRange("enrich", startStage, stopStage, stageOrder)
    def runMaskImageQuantification = stageInRange(
        "mask_image_quantification",
        startStage,
        stopStage,
        stageOrder,
    )
    def runComputeCorticalDepth = stageInRange(
        "compute_cortical_depth",
        startStage,
        stopStage,
        stageOrder,
    )
    def runQc = stageInRange("qc", startStage, stopStage, stageOrder)
    def runAlign = stageInRange("align", startStage, stopStage, stageOrder)
    def runAlignQc = stageInRange("align_qc", startStage, stopStage, stageOrder)
    def runCompare = stageInRange("compare", startStage, stopStage, stageOrder)
    def runVisualize = stageInRange("visualize", startStage, stopStage, stageOrder)
    def runClusteringSquidpy = stageInRange(
        "clustering_squidpy",
        startStage,
        stopStage,
        stageOrder,
    )
    def runMapMyCells = stageInRange("mapmycells", startStage, stopStage, stageOrder)
    def needAnalysisZarrs =
        runAlign || runAlignQc || runCompare || runVisualize || runClusteringSquidpy
    def needAlignmentResults =
        pairedMode && alignmentEnabled && needAnalysisZarrs
    def needAlignmentDownstream =
        pairedMode && alignmentEnabled && (runCompare || runVisualize || runClusteringSquidpy)

    return [
        pair_id: pairId,
        analysis_mode: analysisMode,
        enable_alignment: alignmentEnabled,
        active_platforms: activePlatforms,
        paired_mode: pairedMode,
        analysis_segmentations: analysisSegmentations,
        stage_order: stageOrder,
        start_stage: startStage,
        stop_stage: stopStage,
        selected_stages: stageOrder.findAll { stage ->
            stageInRange(stage, startStage, stopStage, stageOrder)
        },
        run_build: runBuild,
        run_segment: runSegment,
        run_enrich: runEnrich,
        run_mask_image_quantification: runMaskImageQuantification,
        run_compute_cortical_depth: runComputeCorticalDepth,
        run_qc: runQc,
        run_align: runAlign,
        run_align_qc: runAlignQc,
        run_compare: runCompare,
        run_visualize: runVisualize,
        run_clustering_squidpy: runClusteringSquidpy,
        run_mapmycells: runMapMyCells,
        need_build_results: runSegment || runEnrich,
        need_enriched_zarrs: (
            runMaskImageQuantification ||
            runComputeCorticalDepth ||
            runQc ||
            needAnalysisZarrs
        ),
        need_quantified_zarrs: runMaskImageQuantification,
        need_analysis_zarrs: needAnalysisZarrs,
        need_alignment_results: needAlignmentResults,
        need_alignment_downstream: needAlignmentDownstream,
    ]
}

def validateMapMyCellsParams(params) {
    def mapMyCellsReferenceMode = params.mapmycells_reference_mode == null
        ? "both"
        : params.mapmycells_reference_mode.toString().trim().toLowerCase()
    def mapMyCellsPlotsOnly = params.mapmycells_plots_only == null
        ? false
        : params.mapmycells_plots_only.toString().trim().toLowerCase() == "true"
    if (!(mapMyCellsReferenceMode in ["whole_brain", "region", "both"])) {
        throw new IllegalArgumentException(
            "Invalid MAPMYCELLS --mapmycells_reference_mode " +
            "'${params.mapmycells_reference_mode}'. Valid values: " +
            "whole_brain, region, both"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["whole_brain", "both"] &&
        !params.mapmycells_marker_lookup_path) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS: --mapmycells_marker_lookup_path"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["whole_brain", "both"] &&
        !params.mapmycells_precomputed_stats_path) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS: --mapmycells_precomputed_stats_path"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["region", "both"] &&
        !params.mapmycells_region_labels) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS region mode: " +
            "--mapmycells_region_labels"
        )
    }
}

workflow {
    if (!params.samplesheet) {
        error "Missing required parameter: --samplesheet"
    }

    samplesheet_ch = channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true, sep: ",", quote: '"', strip: true)

    sample_rows_raw_ch = samplesheet_ch.map { row ->
        def settings = rowSampleSettings(row, params)
        log.info(
            "Sample ${settings.pair_id}: analysis_mode=${settings.analysis_mode}; " +
            "enable_alignment=${settings.enable_alignment}; " +
            "cortical_depth=${settings.run_compute_cortical_depth}; " +
            "active platforms=${settings.active_platforms.join(', ')}; " +
            "analysis segmentations=${settings.analysis_segmentations.join(', ')}; " +
            "selected stages=${settings.selected_stages.join(' -> ')}"
        )
        tuple(settings.pair_id, row, settings)
    }

    preflight_done_ch = sample_rows_raw_ch
        .map { _pairId, row, settings ->
            runPreflightChecks(row, settings, params)
            true
        }
        .collect()
        .map { true }

    sample_rows_ch = sample_rows_raw_ch
        .combine(preflight_done_ch)
        .map { pairId, row, settings, _doneFlag ->
            tuple(pairId, row, settings)
        }

    build_inputs_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!settings.run_build) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                def buildConfig = buildConfigForPlatform(row, pairId, platform)
                tuple(
                    key,
                    pairId,
                    platform,
                    groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(buildConfig)),
                )
            }
        }
    }

    build_task_results_ch = BUILD_SPATIALDATA(build_inputs_ch)

    build_published_results_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_build_results && !settings.run_build)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                def sourceSpatialdata = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "spatialdata/spatialdata_out/source_spatialdata.zarr",
                    ),
                    "BUILD_SPATIALDATA output for ${pairId}:${platform}",
                )
                tuple(key, pairId, platform, sourceSpatialdata)
            }
        }
    }

    build_results_ch = build_task_results_ch.mix(build_published_results_ch)

    segment_meta_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!settings.run_segment) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple(
                    "${pairId}|${platform}",
                    segmentMetaForPlatform(row, platform, params),
                )
            }
        }
    }

    proseg_trigger_ch = segment_meta_ch.map { true }.take(1)
    proseg_path_ch = ENSURE_PROSEG(proseg_trigger_ch)

    segment_inputs_ch = build_results_ch
        .join(segment_meta_ch)
        .combine(proseg_path_ch)
        .map { key, pairId, platform, sourceSpatialdata, meta, prosegPathFile ->
            def prosegBinaryPath = prosegPathFile.text.trim()
            def persistentLatestZarrPath = file(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    platform,
                    "latest/latest_spatialdata.zarr",
                )
            ).toAbsolutePath().toString()
            def persistentMaskPath = file(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    platform,
                    "segmentation/cellpose_masks_tiled.npy",
                )
            ).toAbsolutePath().toString()
            def persistentTranscriptsPath = file(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    platform,
                    "segmentation/transcripts_for_proseg.csv",
                )
            ).toAbsolutePath().toString()
            def persistentStitchingStatsPath = file(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    platform,
                    "segmentation/cellpose_stitching_stats.json",
                )
            ).toAbsolutePath().toString()
            def baseConfig = [
                cellpose: [
                    model_type: params.cellpose_model_type,
                    gpu: params.cellpose_gpu,
                    diameter: params.cellpose_diameter,
                    flow_threshold: params.cellpose_flow_threshold,
                    cellprob_threshold: params.cellpose_cellprob,
                    tile_overlap: params.cellpose_tile_overlap,
                    bsize: params.cellpose_bsize,
                    factor_rescale: 1.0,
                ],
                tiling: [
                    tile_size_candidates: params.cellpose_tile_size_candidates,
                    stitch_overlap_px: params.cellpose_stitch_overlap_px,
                    min_tile_size: params.cellpose_min_tile_size,
                    status_every_tiles: params.cellpose_stitch_status_every_tiles,
                    filter_per_tile: params.cellpose_filter_per_tile,
                    duplicate_iou_threshold: params.cellpose_duplicate_iou_threshold,
                    duplicate_overlap_fraction: params.cellpose_duplicate_overlap_fraction,
                    min_remaining_fraction: params.cellpose_min_remaining_fraction,
                    edge_touch_policy: params.cellpose_edge_touch_policy,
                    write_stitching_stats: params.cellpose_write_stitching_stats,
                ],
                mask_filter: [
                    final_min_area_um2: params.cellpose_final_min_area_um2,
                    final_max_area_um2: params.cellpose_final_max_area_um2,
                    final_filter_chunk_mb: params.cellpose_final_filter_chunk_mb,
                ],
                proseg: [
                    binary_path: prosegBinaryPath,
                    samples: params.proseg_samples,
                    voxel_size: params.proseg_voxel_size,
                    burnin_voxel_size: params.proseg_burnin_voxel_size,
                    nuclear_reassignment_prob: params.proseg_nuclear_reassignment_prob,
                    diffusion_probability: params.proseg_diffusion_probability,
                    cell_compactness: params.proseg_cell_compactness,
                    num_threads: params.proseg_num_threads,
                    voxel_layers: 2,
                ],
                memory: [
                    max_system_ram_gb: params.max_ram_gb,
                    memory_warn_gb: params.warn_ram_gb,
                    transcript_chunk_rows: params.transcript_chunk_rows,
                ],
            ]

            def segmentConfig = baseConfig + [
                dataset: [
                    name: "${pairId}_${platform}",
                    platform: platform,
                    data_path: sourceSpatialdata.toString(),
                    channels: meta.channels,
                    output_dir: "segment_out",
                    persistent_latest_zarr_path: persistentLatestZarrPath,
                    persistent_mask_path: persistentMaskPath,
                    persistent_transcripts_path: persistentTranscriptsPath,
                    persistent_cellpose_stitching_stats_path: persistentStitchingStatsPath,
                    image_prefix: meta.image_prefix,
                    z_range: meta.z_range,
                    transform_path: meta.transform_path,
                    xenium_spec_path: meta.xenium_spec_path,
                    min_qv: meta.min_qv,
                    proseg_overrides: [voxel_layers: meta.voxel_layers],
                ],
            ]

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(segmentConfig)),
            )
        }

    segment_task_results_ch = SEGMENT(segment_inputs_ch)

    segment_published_results_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!((settings.run_enrich || settings.run_mask_image_quantification) &&
              !settings.run_segment)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                def latestZarr = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/segment_out/proseg_base_latest.zarr",
                    ),
                    "SEGMENT latest zarr for ${pairId}:${platform}",
                )
                def maskPath = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/cellpose_masks_tiled.npy",
                    ),
                    "SEGMENT Cellpose mask for ${pairId}:${platform}",
                )
                def transcriptsCsv = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/transcripts_for_proseg.csv",
                    ),
                    "SEGMENT transcript CSV for ${pairId}:${platform}",
                )
                tuple(key, pairId, platform, latestZarr, maskPath, transcriptsCsv)
            }
        }
    }

    segment_results_ch = segment_task_results_ch.mix(segment_published_results_ch)

    metadata_ch = build_results_ch.map { key, pairId, platform, sourceSpatialdata ->
        tuple(key, pairId, platform, sourceSpatialdata.toString())
    }

    enrich_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_enrich) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", true)
            }
        }
    }

    enrich_inputs_ch = segment_results_ch
        .join(metadata_ch)
        .join(enrich_gate_ch)
        .map {
            key, pairId, platform, latestZarr, maskPath, _transcriptsCsv,
            pairMeta, platformMeta, originalDataPath, _runFlag ->
            if (pairId != pairMeta || platform != platformMeta) {
                error(
                    "Internal channel mismatch for key=${key}: " +
                    "${pairId}/${platform} vs ${pairMeta}/${platformMeta}"
                )
            }

            def persistentLatestZarrPath = file(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    platform,
                    "latest/latest_spatialdata.zarr",
                )
            ).toAbsolutePath().toString()

            def enrichConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                mask_path: "enrich_input_mask.npy",
                original_data_path: originalDataPath,
                output_dir: "enrich_out",
                persistent_output_path: persistentLatestZarrPath,
                transform_path: null,
            ]

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(enrichConfig)),
                latestZarr,
                maskPath,
            )
        }

    enrich_task_results_ch = ENRICH(enrich_inputs_ch)

    enrich_published_results_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_enriched_zarrs && !settings.run_enrich)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                def latestZarr = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "latest/latest_spatialdata.zarr",
                    ),
                    "ENRICH latest zarr for ${pairId}:${platform}",
                )
                def enrichOut = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "enrichment/enrich_out",
                    ),
                    "ENRICH output directory for ${pairId}:${platform}",
                )
                tuple(key, pairId, platform, latestZarr, enrichOut)
            }
        }
    }

    enrich_results_ch = enrich_task_results_ch.mix(enrich_published_results_ch)

    enriched_zarrs_ch = enrich_results_ch.map {
        key, pairId, platform, enrichedLatestZarr, _enrichOutDir ->
            tuple(
                key,
                pairId,
                platform,
                java.nio.file.Paths.get(enrichedLatestZarr.toString()).toRealPath().toString(),
            )
    }

    mask_image_quantification_gate_ch = sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!settings.run_mask_image_quantification) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    tuple("${pairId}|${platform}", true)
                }
            }
    }

    mask_image_quantification_masks_ch = segment_results_ch.map {
        key, _pairId, _platform, _latestZarr, maskPath, _transcriptsCsv ->
            tuple(key, maskPath)
    }

    mask_image_quantification_inputs_ch = enriched_zarrs_ch
        .join(mask_image_quantification_masks_ch)
        .join(mask_image_quantification_gate_ch)
        .map { key, pairId, platform, enrichedLatestZarr, maskPath, _runFlag ->
            def quantConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                mask_path: "mask_image_quantification_input_mask.npy",
                output_dir: "mask_image_quantification_out",
            ]

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(quantConfig)),
                enrichedLatestZarr,
                maskPath,
            )
        }

    mask_image_quantification_results_ch = MASK_IMAGE_QUANTIFICATION(
        mask_image_quantification_inputs_ch
    )

    quantified_zarrs_ch = mask_image_quantification_results_ch.map {
        key, pairId, platform, quantifiedLatestZarr, _quantOutDir ->
            tuple(
                key,
                pairId,
                platform,
                java.nio.file.Paths.get(quantifiedLatestZarr.toString()).toRealPath().toString(),
            )
    }

    enriched_downstream_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_enriched_zarrs && !settings.need_quantified_zarrs)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", true)
            }
        }
    }

    enriched_downstream_zarrs_ch = enriched_zarrs_ch
        .join(enriched_downstream_gate_ch)
        .map { key, pairId, platform, enrichedLatestZarr, _runFlag ->
            tuple(key, pairId, platform, enrichedLatestZarr)
        }

    downstream_zarrs_ch = enriched_downstream_zarrs_ch.mix(quantified_zarrs_ch)

    compute_cortical_depth_gate_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!settings.run_compute_cortical_depth) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple(
                    "${pairId}|${platform}",
                    row,
                    settings.analysis_segmentations,
                )
            }
        }
    }

    compute_cortical_depth_inputs_ch = downstream_zarrs_ch
        .join(compute_cortical_depth_gate_ch)
        .map { key, pairId, platform, latestZarr, row, analysisSegmentations ->
            def depthConfig = corticalDepthConfigForPlatform(
                row,
                pairId,
                platform,
                analysisSegmentations,
                params,
            )
            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(depthConfig)),
                latestZarr,
            )
        }

    compute_cortical_depth_results_ch = COMPUTE_CORTICAL_DEPTH(
        compute_cortical_depth_inputs_ch
    )

    cortical_depth_zarrs_ch = compute_cortical_depth_results_ch.map {
        key, pairId, platform, latestZarr, _corticalDepthOutDir ->
            tuple(
                key,
                pairId,
                platform,
                java.nio.file.Paths.get(latestZarr.toString()).toRealPath().toString(),
            )
    }

    no_cortical_depth_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_enriched_zarrs && !settings.run_compute_cortical_depth)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", true)
            }
        }
    }

    no_cortical_depth_zarrs_ch = downstream_zarrs_ch
        .join(no_cortical_depth_gate_ch)
        .map { key, pairId, platform, latestZarr, _runFlag ->
            tuple(key, pairId, platform, latestZarr)
        }

    analysis_ready_zarrs_ch =
        no_cortical_depth_zarrs_ch.mix(cortical_depth_zarrs_ch)

    qc_branch_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_qc) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", settings.analysis_segmentations)
            }
        }
    }

    qc_inputs_ch = analysis_ready_zarrs_ch
        .join(qc_branch_gate_ch)
        .flatMap { key, pairId, platform, enrichedLatestZarr, analysisSegmentations ->
            analysisSegmentations.collect { segmentation ->
                def layerKeys = analysisLayerKeys(platform, segmentation)
                tuple(
                    "${key}|${segmentation}",
                    pairId,
                    platform,
                    segmentation,
                    enrichedLatestZarr,
                    layerKeys.table_key,
                    layerKeys.shape_key,
                )
            }
        }

    qc_results_ch = QC(qc_inputs_ch)

    analysis_branch_settings_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.need_analysis_zarrs) {
            []
        } else {
            settings.active_platforms.collectMany { platform ->
                settings.analysis_segmentations.collect { segmentation ->
                    tuple("${pairId}|${platform}|${segmentation}", settings)
                }
            }
        }
    }

    analysis_from_qc_ch = qc_results_ch
        .map {
            _key, pairId, platform, segmentation, enrichedLatestZarr,
            _qcOutDir, tableKey, shapeKey ->
                tuple(
                    "${pairId}|${platform}|${segmentation}",
                    pairId,
                    segmentation,
                    platform,
                    java.nio.file.Paths.get(enrichedLatestZarr.toString()).toRealPath().toString(),
                    tableKey,
                    shapeKey,
                )
        }
        .join(analysis_branch_settings_ch)
        .map {
            _branchKey, pairId, segmentation, platform, zarrPath,
            tableKey, shapeKey, settings ->
                tuple(
                    pairId,
                    segmentation,
                    platform,
                    zarrPath,
                    tableKey,
                    shapeKey,
                    settings,
                )
        }

    analysis_no_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_analysis_zarrs && !settings.run_qc)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", settings.analysis_segmentations, settings)
            }
        }
    }

    analysis_without_qc_ch = analysis_ready_zarrs_ch
        .join(analysis_no_qc_gate_ch)
        .flatMap {
            _key, pairId, platform, enrichedLatestZarr, analysisSegmentations, settings ->
                analysisSegmentations.collect { segmentation ->
                    def layerKeys = analysisLayerKeys(platform, segmentation)
                    tuple(
                        pairId,
                        segmentation,
                        platform,
                        enrichedLatestZarr,
                        layerKeys.table_key,
                        layerKeys.shape_key,
                        settings,
                    )
                }
        }

    analysis_dataset_zarrs_ch = analysis_from_qc_ch.mix(analysis_without_qc_ch)

    merscope_zarr_ch = analysis_ready_zarrs_ch
        .filter { _key, _pairId, platform, _zarrPath -> platform == "MERSCOPE" }
        .map { _key, pairId, _platform, zarrPath -> tuple(pairId, zarrPath) }

    xenium_zarr_ch = analysis_ready_zarrs_ch
        .filter { _key, _pairId, platform, _zarrPath -> platform == "XENIUM" }
        .map { _key, pairId, _platform, zarrPath -> tuple(pairId, zarrPath) }

    paired_need_zarrs_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.paired_mode && settings.need_analysis_zarrs)) {
            []
        } else {
            [tuple(pairId, settings)]
        }
    }

    paired_zarrs_ch = merscope_zarr_ch
        .join(xenium_zarr_ch)
        .join(paired_need_zarrs_ch)
        .map { pairId, merscopePath, xeniumPath, settings ->
            tuple(pairId, merscopePath, xeniumPath, settings)
        }

    merscope_analysis_ch = analysis_dataset_zarrs_ch
        .filter {
            _pairId, _segmentation, platform, _zarrPath, _tableKey, _shapeKey, settings ->
                settings.paired_mode && platform == "MERSCOPE"
        }
        .map {
            pairId, segmentation, _platform, zarrPath, tableKey, shapeKey, settings ->
                tuple(
                    "${pairId}|${segmentation}",
                    pairId,
                    segmentation,
                    zarrPath,
                    tableKey,
                    shapeKey,
                    settings,
                )
        }

    xenium_analysis_ch = analysis_dataset_zarrs_ch
        .filter {
            _pairId, _segmentation, platform, _zarrPath, _tableKey, _shapeKey, settings ->
                settings.paired_mode && platform == "XENIUM"
        }
        .map {
            pairId, segmentation, _platform, zarrPath, tableKey, shapeKey, _settings ->
                tuple(
                    "${pairId}|${segmentation}",
                    zarrPath,
                    tableKey,
                    shapeKey,
                )
        }

    paired_analysis_zarrs_ch = merscope_analysis_ch
        .join(xenium_analysis_ch)
        .map {
            _branchKey, pairId, segmentation, merscopePath, merscopeTableKey,
            merscopeShapeKey, settings, xeniumPath, xeniumTableKey, xeniumShapeKey ->
                tuple(
                    pairId,
                    segmentation,
                    merscopePath,
                    xeniumPath,
                    merscopeTableKey,
                    merscopeShapeKey,
                    xeniumTableKey,
                    xeniumShapeKey,
                    settings,
                )
        }

    align_inputs_ch = paired_zarrs_ch
        .filter { _pairId, _merscopePath, _xeniumPath, settings -> settings.run_align }
        .map { pairId, merscopePath, xeniumPath, _settings ->
            tuple(pairId, merscopePath, xeniumPath)
        }

    alignment_task_results_ch = ALIGN(align_inputs_ch)

    alignment_published_results_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_alignment_results && !settings.run_align)) {
            []
        } else {
            def merscopeLatest = requireExistingPath(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    "MERSCOPE",
                    "latest/latest_spatialdata.zarr",
                ),
                "QC/enriched MERSCOPE latest zarr for ${pairId}",
            )
            def xeniumLatest = requireExistingPath(
                publishedDatasetPath(
                    params.outdir,
                    pairId,
                    "XENIUM",
                    "latest/latest_spatialdata.zarr",
                ),
                "QC/enriched XENIUM latest zarr for ${pairId}",
            )
            def transformJson = requireExistingPath(
                publishedPairPath(
                    params.outdir,
                    pairId,
                    "alignment/align_out/alignment_transform.json",
                ),
                "ALIGN transform JSON for ${pairId}",
            )
            def coordsDir = requireExistingPath(
                publishedPairPath(
                    params.outdir,
                    pairId,
                    "alignment/align_out/alignment_coords",
                ),
                "ALIGN coordinate directory for ${pairId}",
            )
            [
                tuple(
                    pairId,
                    merscopeLatest.toRealPath().toString(),
                    xeniumLatest.toRealPath().toString(),
                    transformJson,
                    coordsDir,
                )
            ]
        }
    }

    alignment_results_ch = alignment_task_results_ch.mix(alignment_published_results_ch)

    align_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_align_qc) {
            []
        } else {
            [tuple(pairId, true)]
        }
    }

    align_qc_inputs_ch = alignment_results_ch
        .join(align_qc_gate_ch)
        .map { pairId, merscopeLatest, xeniumLatest, transformJson, coordsDir, _runFlag ->
            tuple(pairId, merscopeLatest, xeniumLatest, transformJson, coordsDir)
        }

    alignment_qc_results_ch = ALIGN_QC(align_qc_inputs_ch)

    alignment_done_no_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_alignment_downstream && !settings.run_align_qc)) {
            []
        } else {
            [tuple(pairId, settings.analysis_segmentations)]
        }
    }

    alignment_done_no_qc_ch = alignment_results_ch
        .join(alignment_done_no_qc_gate_ch)
        .flatMap {
            pairId, _merscopeLatest, _xeniumLatest, _transformJson, _coordsDir, analysisSegmentations ->
                analysisSegmentations.collect { segmentation ->
                    tuple("${pairId}|${segmentation}", true)
                }
        }

    alignment_done_after_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_alignment_downstream && settings.run_align_qc)) {
            []
        } else {
            [tuple(pairId, settings.analysis_segmentations)]
        }
    }

    alignment_done_after_qc_ch = alignment_qc_results_ch
        .join(alignment_done_after_qc_gate_ch)
        .flatMap { pairId, _alignQcOut, analysisSegmentations ->
            analysisSegmentations.collect { segmentation ->
                tuple("${pairId}|${segmentation}", true)
            }
        }

    alignment_done_branch_ch = alignment_done_no_qc_ch.mix(alignment_done_after_qc_ch)

    paired_downstream_no_align_ch = paired_analysis_zarrs_ch
        .filter {
            _pairId, _segmentation, _merscopePath, _xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                !settings.enable_alignment &&
                    (settings.run_compare ||
                     settings.run_visualize ||
                     settings.run_clustering_squidpy)
        }

    paired_downstream_align_ch = paired_analysis_zarrs_ch
        .filter {
            _pairId, _segmentation, _merscopePath, _xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                settings.enable_alignment &&
                    (settings.run_compare ||
                     settings.run_visualize ||
                     settings.run_clustering_squidpy)
        }
        .map {
            pairId, segmentation, merscopePath, xeniumPath, merscopeTableKey,
            merscopeShapeKey, xeniumTableKey, xeniumShapeKey, settings ->
                tuple(
                    "${pairId}|${segmentation}",
                    pairId,
                    segmentation,
                    merscopePath,
                    xeniumPath,
                    merscopeTableKey,
                    merscopeShapeKey,
                    xeniumTableKey,
                    xeniumShapeKey,
                    settings,
                )
        }
        .join(alignment_done_branch_ch)
        .map {
            _branchKey, pairId, segmentation, merscopePath, xeniumPath,
            merscopeTableKey, merscopeShapeKey, xeniumTableKey,
            xeniumShapeKey, settings, _doneFlag ->
                tuple(
                    pairId,
                    segmentation,
                    merscopePath,
                    xeniumPath,
                    merscopeTableKey,
                    merscopeShapeKey,
                    xeniumTableKey,
                    xeniumShapeKey,
                    settings,
                )
        }

    paired_downstream_zarrs_ch = paired_downstream_no_align_ch.mix(paired_downstream_align_ch)

    compare_inputs_ch = paired_downstream_zarrs_ch
        .filter {
            _pairId, _segmentation, _merscopePath, _xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                settings.run_compare
        }
        .map {
            pairId, segmentation, merscopePath, xeniumPath, merscopeTableKey,
            _merscopeShapeKey, xeniumTableKey, _xeniumShapeKey, _settings ->
                tuple(
                    pairId,
                    segmentation,
                    merscopePath,
                    xeniumPath,
                    merscopeTableKey,
                    xeniumTableKey,
                )
        }

    compare_results_ch = COMPARE(compare_inputs_ch)

    compare_done_ch = compare_results_ch.map { pairId, segmentation, _compareOutDir ->
        tuple("${pairId}|${segmentation}", true)
    }

    analysis_samples_from_aligned_pairs_ch = paired_downstream_zarrs_ch
        .filter {
            _pairId, _segmentation, _merscopePath, _xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                settings.enable_alignment &&
                    (settings.run_visualize || settings.run_clustering_squidpy)
        }
        .map {
            pairId, segmentation, merscopePath, xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                tuple(
                    pairId,
                    segmentation,
                    samplesJsonForSegmentation(
                        pairId,
                        ["MERSCOPE", "XENIUM"],
                        ["MERSCOPE": merscopePath, "XENIUM": xeniumPath],
                        segmentation,
                    ),
                    settings,
                )
        }

    analysis_samples_from_dataset_ch = analysis_dataset_zarrs_ch
        .filter {
            _pairId, _segmentation, _platform, _zarrPath, _tableKey, _shapeKey, settings ->
                (!settings.paired_mode || !settings.enable_alignment) &&
                    (settings.run_visualize || settings.run_clustering_squidpy)
        }
        .map {
            pairId, segmentation, platform, zarrPath, _tableKey, _shapeKey, settings ->
                tuple(
                    "${pairId}|${segmentation}",
                    pairId,
                    segmentation,
                    settings,
                    platform,
                    zarrPath,
                )
        }
        .groupTuple()
        .map { _branchKey, pairIds, segmentations, settingsList, platformNames, zarrPaths ->
            def pairId = pairIds[0].toString()
            def segmentation = segmentations[0].toString()
            def settings = settingsList[0]
            def pathsByPlatform = [:]
            platformNames.eachWithIndex { platform, idx ->
                pathsByPlatform[platform.toString()] = zarrPaths[idx].toString()
            }
            tuple(
                pairId,
                segmentation,
                samplesJsonForSegmentation(
                    pairId,
                    settings.active_platforms,
                    pathsByPlatform,
                    segmentation,
                ),
                settings,
            )
        }

    analysis_samples_ch =
        analysis_samples_from_aligned_pairs_ch.mix(analysis_samples_from_dataset_ch)

    visualize_without_compare_ch = analysis_samples_ch
        .filter { _pairId, _segmentation, _samplesJson, settings ->
            settings.run_visualize && !settings.run_compare
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple(pairId, segmentation, samplesJson)
        }

    visualize_after_compare_ch = analysis_samples_ch
        .filter { _pairId, _segmentation, _samplesJson, settings ->
            settings.run_visualize && settings.run_compare
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson)
        }
        .join(compare_done_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, _doneFlag ->
            tuple(pairId, segmentation, samplesJson)
        }

    visualize_inputs_ch = visualize_without_compare_ch.mix(visualize_after_compare_ch)

    visualize_results_ch = VISUALIZE(visualize_inputs_ch)

    visualize_done_ch = visualize_results_ch.map { pairId, segmentation, _visualizeOutDir ->
        tuple("${pairId}|${segmentation}", true)
    }

    clustering_without_visualize_ch = analysis_samples_ch
        .filter { _pairId, _segmentation, _samplesJson, settings ->
            settings.run_clustering_squidpy && !settings.run_visualize
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple(pairId, segmentation, samplesJson)
        }

    clustering_after_visualize_ch = analysis_samples_ch
        .filter { _pairId, _segmentation, _samplesJson, settings ->
            settings.run_clustering_squidpy && settings.run_visualize
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson)
        }
        .join(visualize_done_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, _doneFlag ->
            tuple(pairId, segmentation, samplesJson)
        }

    clustering_inputs_ch =
        clustering_without_visualize_ch.mix(clustering_after_visualize_ch)

    clustering_results_ch = CLUSTERING_SQUIDPY(clustering_inputs_ch)

    mapmycells_after_clustering_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.run_mapmycells && settings.run_clustering_squidpy)) {
            []
        } else {
            validateMapMyCellsParams(params)
            settings.analysis_segmentations.collect { segmentation ->
                tuple("${pairId}|${segmentation}", true)
            }
        }
    }

    mapmycells_from_clustering_ch = clustering_results_ch
        .map { pairId, segmentation, samplesJson, clusteringOutDir ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson, clusteringOutDir)
        }
        .join(mapmycells_after_clustering_gate_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringOutDir, _runFlag ->
            tuple(pairId, segmentation, samplesJson, clusteringOutDir)
        }

    mapmycells_published_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.run_mapmycells && !settings.run_clustering_squidpy)) {
            []
        } else {
            validateMapMyCellsParams(params)
            settings.analysis_segmentations.collect { segmentation ->
                def clusteringOut = requireExistingPath(
                    publishedPairPath(
                        params.outdir,
                        pairId,
                        "${segmentation}/clustering_squidpy/clustering_squidpy_out",
                    ),
                    "CLUSTERING_SQUIDPY ${segmentation} output directory for ${pairId}",
                )
                tuple(
                    pairId,
                    segmentation,
                    samplesJsonForSegmentation(
                        pairId,
                        settings.active_platforms,
                        [:],
                        segmentation,
                    ),
                    clusteringOut,
                )
            }
        }
    }

    mapmycells_inputs_ch = mapmycells_from_clustering_ch.mix(mapmycells_published_ch)

    MAPMYCELLS(mapmycells_inputs_ch)
}
